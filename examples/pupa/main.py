"""PUPA evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the GEPA paper's PUPA setup (Columbia-NLP/PUPA, pupa_tnb):
a 1-stage program (single system_prompt) whose prompt is optimized on
111 train examples, with 111 val examples for Pareto selection and a
held-out test set for final scoring. The metric is aggregate
(quality + leakage) / 2. Default budget matches the paper's PUPA
budget (2426); scale to 15k for Wave B diversity study like IFBench.

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python examples/pupa/main.py [--condition vanilla|random|action|all]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.pupa.utils import load_pupa_dataset, pupa_metric, run_pupa_single_stage
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

# Seed: single system prompt for privacy-conscious redaction
SEED_CANDIDATE = {
    "system_prompt": "You are a helpful assistant. You are a privacy-conscious assistant that redacts PII while preserving the user's intent."
}

# Structured seed variant (markdown skeleton for section-scoped actions)
def _structured_seed(text: str) -> str:
    return (
        "## Role\nYou are a privacy-conscious assistant.\n\n"
        f"## Task\n{text}\n\n"
        "## Rules\n- Redact all PII (names, locations, organizations) with [REDACTED]\n- Preserve the user's intent and meaning\n\n"
        "## Output Format\nReturn only the redacted query, no explanation.\n\n"
        "## Examples\n(none yet)"
    )


_CONDITION_DIR_NAMES = {
    "vanilla": "pupa_vanilla",
    "random": "pupa_random_action",
    "action": "pupa_verbalized_action",
}


def condition_run_dir(condition: str, tag: str = "") -> str:
    tag_suffix = f"_{tag}" if tag else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{tag_suffix}"


def seed_candidate(seed_style: str = "plain") -> dict:
    seed = dict(SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {k: _structured_seed(v) for k, v in seed.items()}
    return seed


def make_evaluator(solver_model: str, api_base: str | None = None, judge_model: str | None = None, judge_api_base: str | None = None):
    """Create evaluator closed over solver/judge models."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        response = run_pupa_single_stage(
            candidate["system_prompt"], example["prompt"], model=solver_model, api_base=api_base
        )
        # Use solver_model as judge if no separate judge provided (like tests use reflection_lm)
        j_model = judge_model or solver_model
        j_base = judge_api_base if judge_model else api_base
        score, feedback = pupa_metric(response, example, judge_model=j_model, judge_api_base=j_base)
        side_info: SideInfo = {
            "score": score,
            "user_query": example["prompt"],
            "output": response,
            "execution_feedback": feedback,
            "gold": example.get("answer", ""),
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
) -> float:
    """Evaluate candidate on dataset, returning mean aggregate score."""

    def score_one(example: dict) -> float:
        response = run_pupa_single_stage(
            candidate["system_prompt"], example["prompt"], model=solver_model, api_base=api_base
        )
        j_model = judge_model or solver_model
        j_base = judge_api_base if judge_model else api_base
        score, _ = pupa_metric(response, example, judge_model=j_model, judge_api_base=j_base)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
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
            run_dir=condition_run_dir(condition, args.tag),
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
    parser = argparse.ArgumentParser(description="PUPA evaluation for action-conditioned reflection")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=2426,
        help="Budget per condition (paper PUPA: 2426, scaled Wave B: 15000)",
    )
    parser.add_argument(
        "--solver-model", type=str, default="hosted_vllm/Qwen3-8B", help="Solver LM model (litellm format)"
    )
    parser.add_argument(
        "--reflection-model", type=str, default="hosted_vllm/Qwen3-8B", help="Reflection LM model (litellm format)"
    )
    parser.add_argument("--judge-model", type=str, default=None, help="Judge LM for quality (default: solver_model)")
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 111)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 111)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size for final evaluation")
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
        help="Seed prompts: plain or markdown skeleton (Role/Task/Rules/Output Format/Examples)",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="default",
        choices=["default", "structured"],
        help="Action space: DEFAULT_ACTIONS or section-scoped structured actions (implies --seed-style structured)",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. pupa_rev1)")
    parser.add_argument("--config", type=str, default="pupa_tnb", choices=["pupa_tnb", "pupa_new"], help="PUPA config")
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    trainset, valset, testset = load_pupa_dataset(config=args.config)
    if args.train_limit is not None:
        trainset = trainset[: args.train_limit]
    if args.val_limit is not None:
        valset = valset[: args.val_limit]
    if args.test_limit is not None:
        testset = testset[: args.test_limit]
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples (PUPA {args.config})")

    evaluator = make_evaluator(
        args.solver_model, api_base=args.api_base, judge_model=args.judge_model, judge_api_base=args.api_base
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
            f"{condition} GEPA (PUPA {args.seed_style} seeds)",
            seed_candidate(args.seed_style),
            trainset,
            valset,
            config,
            evaluator,
            callbacks=callbacks,
        )
        run_dir = condition_run_dir(condition, args.tag)
        path = dump_candidates(results[condition], run_dir)
        print(f"[{condition}] wrote {path}")
        if condition in trackers:
            path = dump_action_summary(trackers[condition], run_dir, selector=selector)
            print(f"[{condition}] wrote {path}")

    # Report: best prompts
    print(f"\n{'=' * 60}")
    print("  Best prompts")
    print(f"{'=' * 60}")
    for name, result in results.items():
        print(f"\n----- [{name}] best candidate (val score {result.val_aggregate_scores[result.best_idx]:.4f}) -----")
        for component, text in result.best_candidate.items():
            print(f"\n[{name}] {component}:\n{text}")

    # Report: test aggregate + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    baseline_score = evaluate_on_set(
        seed_candidate(args.seed_style),
        testset,
        args.solver_model,
        api_base=args.api_base,
        judge_model=args.judge_model,
        judge_api_base=args.api_base,
    )
    print(f"Baseline (seed prompts) test aggregate (quality+leakage)/2: {baseline_score:.3f} on {len(testset)} examples\n")

    for name, result in results.items():
        test_score = evaluate_on_set(
            result.best_candidate, testset, args.solver_model, api_base=args.api_base, judge_model=args.judge_model, judge_api_base=args.api_base
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test aggregate:           {test_score:.3f}")
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
