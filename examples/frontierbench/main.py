"""FrontierBench evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the Frontier-Bench setup (https://github.com/laude-institute/frontier-bench):
harder agentic research tasks from the Terminal-Bench authors. Each task is an
end-to-end research assignment scored by an LLM judge / test harness pass.

The program is either 1-stage (single execution) or 2-stage (research plan ->
execution), mirroring the auto-research / Terminal-Bench agentic pattern. The
metric is task success (mean pass rate over success criteria, 0-1) with
per-criterion feedback via LLM judge, with a heuristic fallback.

Dataset: HF ``laude/frontier-bench`` (harder agentic research tasks, Terminal-Bench
lineage) or a local ``data/frontierbench.jsonl`` fallback. Splits are 30/30/30
(train/val/test) via seed-0 shuffle; use --train-limit / --val-limit /
--test-limit and --data-path to override. Budget default is 4000 metric calls
(within the 3000-5000 stretch range). Like IFBench/PUPA, the 2-stage program
has two prompts optimized jointly.

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python examples/frontierbench/main.py [--condition vanilla|random|action|all]
        [--program 2stage|1stage] [--max-metric-calls N]
        [--train-limit N] [--val-limit N] [--test-limit N]
        [--data-path path/to/frontierbench.jsonl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.frontierbench.utils import frontierbench_metric, load_frontierbench_dataset, run_single_stage, run_two_stage
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

# Seed instructions: 2-stage = plan then execution
SEED_CANDIDATE = {
    "research_plan": (
        "Create a detailed research plan for the given task. Outline the methodology, "
        "required tools and data, step-by-step execution, validation checks, and how "
        "you will handle failure modes."
    ),
    "execute_task": (
        "Execute the research task following the research plan. Provide a complete "
        "output including methodology, concrete results or code, analysis, and "
        "discussion of limitations. Validate your claims."
    ),
}

SEED_CANDIDATE_1STAGE = {
    "execute_task": (
        "Execute the research task end-to-end. Provide a complete output including "
        "methodology, concrete results or code, analysis, and discussion of "
        "limitations. Validate your claims."
    ),
}

_CONDITION_DIR_NAMES = {
    "vanilla": "frontierbench_vanilla",
    "random": "frontierbench_random_action",
    "action": "frontierbench_verbalized_action",
}


def _structured_seed(task_sentence: str) -> str:
    """Wrap a seed sentence in a best-practice markdown skeleton."""
    return (
        "## Role\nYou are an expert agentic researcher.\n\n"
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


def run_program(candidate: dict, task: str, program: str, model: str, api_base: str | None) -> tuple[str | None, str]:
    """Run the candidate program on a task, returning (stage1, final)."""
    if program == "1stage":
        prompt = candidate.get("execute_task") or next(iter(candidate.values()))
        return None, run_single_stage(prompt, task, model=model, api_base=api_base)
    plan_p = candidate.get("research_plan", "")
    exec_p = candidate.get("execute_task", "")
    if not plan_p and not exec_p:
        vals = list(candidate.values())
        if len(vals) >= 2:
            plan_p, exec_p = vals[0], vals[1]
        elif len(vals) == 1:
            plan_p, exec_p = vals[0], vals[0]
    return run_two_stage(plan_p, exec_p, task, model=model, api_base=api_base)


def make_evaluator(
    solver_model: str,
    api_base: str | None = None,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
    program: str = "2stage",
):
    """Create an evaluator function closed over the solver/judge models."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        task = example.get("prompt") or example.get("task") or example.get("instruction") or ""
        _, final = run_program(candidate, task, program, solver_model, api_base)
        j_model = judge_model or solver_model
        j_base = judge_api_base if judge_model is not None else api_base
        score, feedback = frontierbench_metric(final, example, judge_model=j_model, judge_api_base=j_base)
        side_info: SideInfo = {
            "score": score,
            "task": task,
            "output": final,
            "execution_feedback": feedback,
            "category": example.get("category", ""),
            "difficulty": example.get("difficulty", ""),
        }
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
    """Evaluate a candidate on a dataset, returning mean task-success."""

    def score_one(example: dict) -> float:
        task = example.get("prompt") or example.get("task") or example.get("instruction") or ""
        _, final = run_program(candidate, task, program, solver_model, api_base)
        j_model = judge_model or solver_model
        j_base = judge_api_base if judge_model is not None else api_base
        score, _ = frontierbench_metric(final, example, judge_model=j_model, judge_api_base=j_base)
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
    parser = argparse.ArgumentParser(description="FrontierBench evaluation for action-conditioned reflection")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to FrontierBench JSONL (one task per line). If omitted, tries HF laude/frontier-bench then synthetic fallback.",
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
    parser.add_argument("--judge-model", type=str, default=None, help="Judge LM for task success (default: solver_model)")
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
        help="Program structure: 2stage (plan-then-execute) or 1stage (single execution)",
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

    trainset, valset, testset = load_frontierbench_dataset(
        data_path=args.data_path, seed=args.seed, train_limit=args.train_limit, val_limit=args.val_limit, test_limit=args.test_limit
    )
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples (FrontierBench {args.program}, {args.seed_style})")
    if args.data_path:
        print(f"  (from {args.data_path})")
    else:
        print("  (from laude/frontier-bench via datasets; Terminal-Bench lineage, sliced 30/30/30)")

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
            f"{condition} GEPA (FrontierBench {args.program}, {args.seed_style} seeds)",
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

    # Report: test success + diversity
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
    print(f"Baseline (seed prompts) test task success: {baseline_score:.3f} on {len(testset)} examples\n")

    for name, result in results.items():
        test_score = evaluate_on_set(
            result.best_candidate, testset, args.solver_model, api_base=args.api_base, judge_model=args.judge_model, judge_api_base=args.api_base, program=args.program
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test task success:        {test_score:.3f}")
        for component, stats in diversity.items():
            print(
                f"  diversity[{component}]: jaccard_dist={stats['mean_pairwise_jaccard_distance']:.3f} "
                f"unique={int(stats['num_unique_texts'])}/{len(result.candidates)}"
            )
        print()

    # Action diversity metrics
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
