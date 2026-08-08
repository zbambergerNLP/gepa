"""FrontierCS evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the Frontier-CS setup (https://github.com/FrontierCS/Frontier-CS):
open-ended CS research problems benchmarked via an auto-research framework.
The program is either 1-stage (single proposal) or 2-stage (literature review
-> proposal), mirroring IFBench's 2-stage architecture. The metric is a
rubric-based LLM-judge score (mean pass rate over rubric criteria) with
per-criterion feedback.

Dataset: HF ``FrontierCS/Frontier-CS`` (~100 open-ended CS research problems
per the paper) or a local ``data/frontiercs.jsonl`` fallback. Splits are
30/30/30 (train/val/test) via seed-0 shuffle, noting the paper's ~100-problem
pool; use --train-limit / --val-limit / --test-limit and --data-path to
override. Budget default is 4000 metric calls (within the 3000-5000 stretch
range). Like IFBench, the 2-stage program has two prompts optimized jointly.

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python examples/frontiercs/main.py [--condition vanilla|random|action|all]
        [--program 2stage|1stage] [--max-metric-calls N]
        [--train-limit N] [--val-limit N] [--test-limit N]
        [--data-path path/to/frontiercs.jsonl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.frontiercs.utils import frontiercs_metric, load_frontiercs_dataset, run_single_stage, run_two_stage
from gepa.core.action_tracking import ActionDiversityCallback
from gepa.lm import LM
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)
from gepa.strategies.action_space import (
    DEFAULT_ACTIONS,
    RandomActionSelector,
    VerbalizedActionSelector,
    build_structured_actions,
)

# Seed instructions: 2-stage = literature survey + proposal drafting
SEED_CANDIDATE = {
    "literature_review": (
        "Survey the most relevant prior work, baselines, and key papers for the given CS research problem. "
        "Summarize the state of the art, open gaps, and evaluation practices."
    ),
    "draft_proposal": (
        "Draft a complete CS research proposal for the given problem, building on the literature review. "
        "Include the core idea, novelty over prior work, detailed method, evaluation plan with baselines "
        "and metrics, and limitations."
    ),
}

SEED_CANDIDATE_1STAGE = {
    "research_proposal": (
        "Draft a complete CS research proposal for the given problem. "
        "Include the core idea, novelty over prior work, detailed method, evaluation plan with baselines "
        "and metrics, and limitations."
    ),
}

_CONDITION_DIR_NAMES = {
    "vanilla": "frontiercs_vanilla",
    "random": "frontiercs_random_action",
    "action": "frontiercs_verbalized_action",
}


def _structured_seed(task_sentence: str) -> str:
    """Wrap a seed sentence in a best-practice markdown skeleton.

    The paper's seed sentence is preserved verbatim in the Task section; other
    sections start as explicit placeholders for section-scoped actions to fill.
    """
    return (
        "## Role\nYou are an expert CS researcher.\n\n"
        f"## Task\n{task_sentence}\n\n"
        "## Rules\n(none yet)\n\n"
        "## Output Format\n(none yet)\n\n"
        "## Examples\n(none yet)"
    )


def condition_run_dir(condition: str, program: str, tag: str = "") -> str:
    suffix = "_1stage" if program == "1stage" else ""
    tag_suffix = f"_{tag}" if tag else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{suffix}{tag_suffix}"


def seed_candidate(program: str, seed_style: str = "plain") -> dict:
    seed = dict(SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {component: _structured_seed(text) for component, text in seed.items()}
    return seed


def run_program(candidate: dict, problem: str, program: str, model: str, api_base: str | None) -> tuple[str | None, str]:
    """Run the candidate program on a problem, returning (stage1, final)."""
    if program == "1stage":
        prompt = candidate.get("research_proposal") or next(iter(candidate.values()))
        return None, run_single_stage(prompt, problem, model=model, api_base=api_base)
    lit = candidate.get("literature_review", "")
    prop = candidate.get("draft_proposal", "")
    # Fallback: if candidate has generic keys (e.g., from 1stage seed used in 2stage), map them
    if not lit and not prop:
        vals = list(candidate.values())
        if len(vals) >= 2:
            lit, prop = vals[0], vals[1]
        elif len(vals) == 1:
            lit, prop = vals[0], vals[0]
    return run_two_stage(lit, prop, problem, model=model, api_base=api_base)


def make_evaluator(
    solver_model: str,
    api_base: str | None = None,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
    program: str = "2stage",
):
    """Create an evaluator function closed over the solver/judge models."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        problem = example.get("prompt") or example.get("problem") or ""
        _, final = run_program(candidate, problem, program, solver_model, api_base)
        j_model = judge_model or solver_model
        j_base = judge_api_base if judge_model is not None else api_base
        score, feedback = frontiercs_metric(final, example, judge_model=j_model, judge_api_base=j_base)
        side_info: SideInfo = {
            "score": score,
            "problem": problem,
            "output": final,
            "execution_feedback": feedback,
            "area": example.get("area", ""),
            "difficulty": example.get("difficulty", ""),
        }
        # Include stage1 for debugging / reflection signal
        # run_program already returns it but we re-run to avoid double-capture; instead store final only.
        # The GEPA proposer sees execution_feedback which already lists rubric pass/fail.
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate: dict,
    dataset: list[dict],
    solver_model: str,
    api_base: str | None = None,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
    max_workers: int = 12,
    program: str = "2stage",
) -> float:
    """Evaluate a candidate on a dataset, returning mean rubric score."""

    def score_one(example: dict) -> float:
        problem = example.get("prompt") or example.get("problem") or ""
        _, final = run_program(candidate, problem, program, solver_model, api_base)
        j_model = judge_model or solver_model
        j_base = judge_api_base if judge_model is not None else api_base
        score, _ = frontiercs_metric(final, example, judge_model=j_model, judge_api_base=j_base)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
    """Textual diversity of explored candidates, per component."""
    if not candidates:
        return {}
    diversity: dict[str, dict[str, float]] = {}
    for component in candidates[0]:
        texts = [c[component] for c in candidates]
        token_sets = [set(t.lower().split()) for t in texts]
        distances = []
        for a, b in itertools.combinations(token_sets, 2):
            union = a | b
            distances.append(1.0 - (len(a & b) / len(union)) if union else 0.0)
        diversity[component] = {
            "mean_pairwise_jaccard_distance": sum(distances) / len(distances) if distances else 0.0,
            "num_unique_texts": float(len(set(texts))),
        }
    return diversity


def dump_candidates(result, run_dir: str) -> str:
    """Write all explored candidates (with lineage and scores) to candidates.json."""
    payload = {
        "best_idx": result.best_idx,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "candidates": result.candidates,
        "parents": result.parents,
        "val_aggregate_scores": result.val_aggregate_scores,
        "discovery_eval_counts": result.discovery_eval_counts,
    }
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "candidates.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def dump_action_summary(tracker: ActionDiversityCallback, run_dir: str, selector=None) -> str:
    """Persist the action tracker's aggregate summary plus raw per-action data."""
    payload = {
        "summary": tracker.summary(),
        "action_score_deltas": dict(tracker.action_score_deltas),
        "action_texts": dict(tracker.action_texts),
    }
    if selector is not None and getattr(selector, "history", None):
        payload["verbalized_history"] = selector.history
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "action_summary.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def build_config(condition: str, args, reflection_lm_kwargs: dict):
    """Build the GEPAConfig for one condition. Returns (config, action_selector)."""
    action_space = build_structured_actions() if args.actions == "structured" else DEFAULT_ACTIONS
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(args.reflection_model, **reflection_lm_kwargs),
        )

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=condition_run_dir(condition, args.program, args.tag),
            max_metric_calls=args.max_metric_calls,
            parallel=True,
            max_workers=12,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs or None,
            action_selector=action_selector,
        ),
    )
    return config, action_selector


def run_condition(
    name: str,
    seed: dict,
    trainset: list[dict],
    valset: list[dict],
    config: GEPAConfig,
    evaluator,
    callbacks: list | None = None,
):
    """Run one optimization condition and return the result."""
    print(f"\n{'=' * 60}")
    print(f"  Running: {name}")
    print(f"{'=' * 60}\n")

    if callbacks:
        config.callbacks = callbacks

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        config=config,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="FrontierCS evaluation for action-conditioned reflection")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to FrontierCS JSONL (one task per line). If omitted, tries HF FrontierCS/Frontier-CS then synthetic fallback.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=4000,
        help="Budget per condition (stretch 3000-5000; default 4000, scaled Wave B: 15000)",
    )
    parser.add_argument(
        "--solver-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Solver LM model (litellm format)"
    )
    parser.add_argument(
        "--reflection-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Reflection LM model (litellm format)"
    )
    parser.add_argument("--judge-model", type=str, default=None, help="Judge LM for rubric (default: solver_model)")
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 30)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 30)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size for final evaluation (paper: 30)")
    parser.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed")
    parser.add_argument(
        "--program",
        type=str,
        default="2stage",
        choices=["2stage", "1stage"],
        help="Program structure: 2stage (literature-then-proposal) or 1stage (single proposal)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["vanilla", "random", "action", "all"],
        help="Which condition(s) to run",
    )
    parser.add_argument(
        "--seed-style",
        type=str,
        default="plain",
        choices=["plain", "structured"],
        help="Seed prompts: plain sentences or a markdown skeleton (Role/Task/Rules/Output Format/Examples)",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="default",
        choices=["default", "structured"],
        help="Action space: DEFAULT_ACTIONS or section-scoped structured actions (implies --seed-style structured)",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev1)")
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    trainset, valset, testset = load_frontiercs_dataset(
        data_path=args.data_path, seed=args.seed, train_limit=args.train_limit, val_limit=args.val_limit, test_limit=args.test_limit
    )
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples (FrontierCS {args.program}, {args.seed_style})")
    if args.data_path:
        print(f"  (from {args.data_path})")
    else:
        print("  (from FrontierCS/Frontier-CS via datasets; paper ~100 problems, sliced 30/30/30)")

    evaluator = make_evaluator(
        args.solver_model, api_base=args.api_base, judge_model=args.judge_model, judge_api_base=args.api_base, program=args.program
    )

    reflection_lm_kwargs = {}
    if args.api_base is not None:
        reflection_lm_kwargs["api_base"] = args.api_base

    conditions = ["vanilla", "random", "action"] if args.condition == "all" else [args.condition]

    results = {}
    trackers: dict[str, ActionDiversityCallback] = {}
    for condition in conditions:
        config, selector = build_config(condition, args, reflection_lm_kwargs)
        callbacks = None
        if condition in ("random", "action"):
            trackers[condition] = ActionDiversityCallback()
            callbacks = [trackers[condition]]
        results[condition] = run_condition(
            f"{condition} GEPA (FrontierCS {args.program}, {args.seed_style} seeds)",
            seed_candidate(args.program, args.seed_style),
            trainset,
            valset,
            config,
            evaluator,
            callbacks=callbacks,
        )
        run_dir = condition_run_dir(condition, args.program, args.tag)
        path = dump_candidates(results[condition], run_dir)
        print(f"[{condition}] wrote {path}")
        if condition in trackers:
            path = dump_action_summary(trackers[condition], run_dir, selector=selector)
            print(f"[{condition}] wrote {path}")

    # Report: best prompts (full text)
    print(f"\n{'=' * 60}")
    print("  Best prompts")
    print(f"{'=' * 60}")
    for name, result in results.items():
        print(f"\n----- [{name}] best candidate (val score {result.val_aggregate_scores[result.best_idx]:.4f}) -----")
        for component, text in result.best_candidate.items():
            print(f"\n[{name}] {component}:\n{text}")

    # Report: test rubric score + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    baseline_score = evaluate_on_set(
        seed_candidate(args.program, args.seed_style),
        testset,
        args.solver_model,
        api_base=args.api_base,
        judge_model=args.judge_model,
        judge_api_base=args.api_base,
        program=args.program,
    )
    print(f"Baseline (seed prompts) test rubric score: {baseline_score:.3f} on {len(testset)} examples\n")

    for name, result in results.items():
        test_score = evaluate_on_set(
            result.best_candidate, testset, args.solver_model, api_base=args.api_base, judge_model=args.judge_model, judge_api_base=args.api_base, program=args.program
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test rubric score:        {test_score:.3f}")
        for component, stats in diversity.items():
            print(
                f"  diversity[{component}]: jaccard_dist={stats['mean_pairwise_jaccard_distance']:.3f} "
                f"unique={int(stats['num_unique_texts'])}/{len(result.candidates)}"
            )
        print()

    # Action diversity metrics (random / action conditions)
    for name, tracker in trackers.items():
        print(f"{'=' * 60}")
        print(f"  Action Diversity Metrics [{name}]")
        print(f"{'=' * 60}\n")
        summary = tracker.summary()
        print(f"Total proposals: {summary['total_proposals']}")
        print(f"Total accepted:  {summary['total_accepted']}")
        print(f"\nPer-action proposal counts: {summary['action_proposal_counts']}")
        print(f"Per-action acceptance rates: {summary['action_acceptance_rates']}")
        print(f"Textual diversity per iteration: {summary['textual_diversity_per_iteration']}\n")


if __name__ == "__main__":
    main()
