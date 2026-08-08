"""LiveBench-Math evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the GEPA paper's LiveBench-Math setup (White et al. 2025,
n=368 math, contamination-limited, AMC/AIME/symbolic algebra/olympiad)
with a single-step CoT program (one optimized instruction, one LM call).
The metric is exact-match accuracy after answer normalization. Splits
are 122 train / 123 val / 123 test shuffled seed 0 (Terrarium split
100/100/168 is available via --splits terrarium). Default budget
1839 metric calls matches the paper's LiveBench-Math budget (like the
GEPA release note), scaled Wave B uses 5000.

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python examples/livebench_math/main.py [--condition vanilla|random|action|all]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.livebench_math.utils import (
    livebench_metric,
    load_livebench_math_dataset,
    run_livebench_single_stage,
)
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

SEED_CANDIDATE = {
    "instruction": "Solve the math problem carefully. Show your reasoning step by step and give the final answer clearly after 'Final Answer:'."
}

_CONDITION_DIR_NAMES = {
    "vanilla": "livebench_math_vanilla",
    "random": "livebench_math_random_action",
    "action": "livebench_math_verbalized_action",
}


def _structured_seed(text: str) -> str:
    """Wrap seed sentence in a best-practice markdown skeleton."""
    return (
        "## Role\nYou are an expert competition mathematician.\n\n"
        f"## Task\n{text}\n\n"
        "## Rules\n- Be precise and rigorous\n- Show key steps before the final answer\n\n"
        "## Output Format\nReasoning, then a line 'Final Answer:' followed by the answer.\n\n"
        "## Examples\n(none yet)"
    )


def condition_run_dir(condition: str, tag: str = "") -> str:
    tag_suffix = f"_{tag}" if tag else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{tag_suffix}"


def seed_candidate(seed_style: str = "plain") -> dict:
    seed = dict(SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {k: _structured_seed(v) for k, v in seed.items()}
    return seed


def make_evaluator(solver_model: str, api_base: str | None = None):
    """Create evaluator closed over solver model name."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        prompt = candidate.get("instruction") or candidate.get("system_prompt") or next(iter(candidate.values()))
        problem = example.get("prompt") or example.get("problem") or example.get("input", "")
        raw_output = run_livebench_single_stage(prompt, problem, model=solver_model, api_base=api_base)
        score, feedback = livebench_metric(raw_output, example)
        side_info: SideInfo = {
            "score": score,
            "problem": problem,
            "output": raw_output,
            "execution_feedback": feedback,
            "answer": str(example.get("answer", "")),
        }
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate: dict,
    dataset: list[dict],
    solver_model: str,
    api_base: str | None = None,
    max_workers: int = 16,
) -> float:
    """Evaluate candidate on dataset, returning mean accuracy."""
    prompt = candidate.get("instruction") or candidate.get("system_prompt") or next(iter(candidate.values()))

    def score_one(example: dict) -> float:
        problem = example.get("prompt") or example.get("problem") or example.get("input", "")
        out = run_livebench_single_stage(prompt, problem, model=solver_model, api_base=api_base)
        score, _ = livebench_metric(out, example)
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
            run_dir=condition_run_dir(condition, args.tag),
            max_metric_calls=args.max_metric_calls,
            parallel=True,
            max_workers=24,
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
    parser = argparse.ArgumentParser(description="LiveBench-Math evaluation for action-conditioned reflection")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=1839,
        help="Budget per condition (paper LiveBench-Math: 1839, scaled Terrarium 5000, Wave B 15000)",
    )
    parser.add_argument(
        "--solver-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Solver LM model (litellm format)"
    )
    parser.add_argument(
        "--reflection-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Reflection LM model (litellm format)"
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)"
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 122)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 123)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size for final evaluation (paper: 123)")
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
        help="Seed prompts: plain paper sentence or markdown skeleton (Role/Task/Rules/Output Format/Examples)",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="default",
        choices=["default", "structured"],
        help="Action space: DEFAULT_ACTIONS or section-scoped structured actions (implies --seed-style structured)",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. livebench_rev1)")
    parser.add_argument(
        "--splits",
        type=str,
        default="paper",
        choices=["paper", "terrarium"],
        help="Splits: paper 122/123/123 or terrarium 100/100/168 (from GEPA parallel-proposals release)",
    )
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    splits = (100, 100, 168) if args.splits == "terrarium" else None
    trainset, valset, testset = load_livebench_math_dataset(splits=splits)

    if args.train_limit is not None:
        trainset = trainset[: args.train_limit]
    if args.val_limit is not None:
        valset = valset[: args.val_limit]
    if args.test_limit is not None:
        testset = testset[: args.test_limit]
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples (LiveBench-Math {args.splits})")

    evaluator = make_evaluator(args.solver_model, api_base=args.api_base)

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
            f"{condition} GEPA (LiveBench-Math {args.seed_style} seeds)",
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

    # Report: test accuracy + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    baseline_score = evaluate_on_set(
        seed_candidate(args.seed_style),
        testset,
        args.solver_model,
        api_base=args.api_base,
    )
    print(f"Baseline (seed prompts) test accuracy: {baseline_score:.2%} on {len(testset)} examples\n")

    for name, result in results.items():
        test_score = evaluate_on_set(
            result.best_candidate, testset, args.solver_model, api_base=args.api_base
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test accuracy:            {test_score:.2%}")
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
