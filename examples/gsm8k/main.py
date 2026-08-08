"""GSM8K evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates a GSM8K (Cobbe et al. 2021, HF ``gsm8k`` / ``openai/gsm8k`` main
config, 7.5K train / 1K test typical; here 150 train / 300 val / 300 test
deterministic shuffle seed 0 with headroom for 200/300/300) setup mirroring
``examples/aime_math`` and ``examples/ifbench``. The program is single-step
CoT (one optimized ``instruction``, one LM call with ``Final Answer:``
marker), metric is exact match after numeric/boxed normalization with
solution-aware feedback, and the default budget is 5000 metric calls
(paper heavy ~5K, legacy vs scaled).

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python -m examples.gsm8k.main [--condition vanilla|random|action|all]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
        [--data-path PATH] [--solver-model MODEL] [--api-base URL]
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.gsm8k.utils import (
    DEFECTIVE_SEED_CANDIDATE,
    get_defective_seed,
    gsm8k_metric,
    load_gsm8k_dataset,
    run_gsm8k_single_stage,
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
    "instruction": "Solve the math problem carefully. Break down the steps and provide the final answer as a single number.",
}


_CONDITION_DIR_NAMES = {
    "vanilla": "gsm8k_vanilla",
    "random": "gsm8k_random_action",
    "action": "gsm8k_verbalized_action",
}


def _structured_seed(text: str) -> str:
    """Wrap seed sentence in a best-practice markdown skeleton.

    The seed sentence is preserved verbatim in the Task section; other
    sections start as explicit placeholders for section-scoped actions to fill.
    """
    return (
        "## Role\nYou are an expert grade-school math tutor.\n\n"
        f"## Task\n{text}\n\n"
        "## Rules\n(none yet)\n\n"
        "## Output Format\n(none yet)\n\n"
        "## Examples\n(none yet)"
    )


def condition_run_dir(condition: str, tag: str = "") -> str:
    tag_suffix = f"_{tag}" if tag else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{tag_suffix}"


def seed_candidate(seed_style: str = "plain", defective: bool = False, defective_variant: str = "default") -> dict:
    """Return the seed candidate dict.

    Args:
        seed_style: "plain" or "structured".
        defective: if True, return the VISTA-style defective seed instead of
            the standard seed (for recovery tests).
        defective_variant: "default" or "alt" when defective=True.
    """
    if defective:
        seed = get_defective_seed(variant=defective_variant)
        if seed_style == "structured":
            seed = {k: _structured_seed(v) for k, v in seed.items()}
        return seed
    seed = dict(SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {k: _structured_seed(v) for k, v in seed.items()}
    return seed


def make_evaluator(solver_model: str, api_base: str | None = None):
    """Create evaluator closed over solver model name."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        prompt = candidate.get("instruction") or candidate.get("system_prompt") or next(iter(candidate.values()))
        problem = example.get("prompt") or example.get("problem") or example.get("input", "") or example.get("question", "")
        raw_output = run_gsm8k_single_stage(prompt, problem, model=solver_model, api_base=api_base)
        score, feedback = gsm8k_metric(example, raw_output)
        side_info: SideInfo = {
            "score": score,
            "problem": problem,
            "output": raw_output,
            "execution_feedback": feedback,
            "answer": str(example.get("answer_number") or example.get("answer", "")),
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
        problem = example.get("prompt") or example.get("problem") or example.get("input", "") or example.get("question", "")
        out = run_gsm8k_single_stage(prompt, problem, model=solver_model, api_base=api_base)
        score, _ = gsm8k_metric(example, out)
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
    parser = argparse.ArgumentParser(description="GSM8K evaluation for action-conditioned reflection")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=5000,
        help="Budget per condition (paper heavy ~5k, legacy vs scaled; use 5000 default)",
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
    parser.add_argument("--data-path", type=str, default=None, help="Local GSM8K JSONL path or directory (overrides HF)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (default: 150)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (default: 300)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size for final evaluation (default: 300)")
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
        help="Seed prompts: plain sentence or markdown skeleton (Role/Task/Rules/Output Format/Examples)",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="default",
        choices=["default", "structured"],
        help="Action space: DEFAULT_ACTIONS or section-scoped structured actions (implies --seed-style structured)",
    )
    parser.add_argument(
        "--defective-seed",
        action="store_true",
        default=False,
        help="Use defective seed variant (for VISTA recovery test)",
    )
    parser.add_argument(
        "--defective-variant",
        type=str,
        default="default",
        choices=["default", "alt"],
        help="Defective seed variant when --defective-seed is set",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. gsm8k_rev1)")
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    trainset, valset, testset = load_gsm8k_dataset(data_path=args.data_path)
    if args.train_limit is not None:
        trainset = trainset[: args.train_limit]
    if args.val_limit is not None:
        valset = valset[: args.val_limit]
    if args.test_limit is not None:
        testset = testset[: args.test_limit]
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples (GSM8K seed 0, 150/300/300)")

    evaluator = make_evaluator(args.solver_model, api_base=args.api_base)

    reflection_lm_kwargs = {}
    if args.api_base is not None:
        reflection_lm_kwargs["api_base"] = args.api_base

    conditions = ["vanilla", "random", "action"] if args.condition == "all" else [args.condition]

    seed_kwargs = dict(seed_style=args.seed_style, defective=args.defective_seed, defective_variant=args.defective_variant)

    results = {}
    trackers: dict[str, ActionDiversityCallback] = {}
    for condition in conditions:
        config, selector = build_config(condition, args, reflection_lm_kwargs)
        callbacks = None
        if condition in ("random", "action"):
            trackers[condition] = ActionDiversityCallback()
            callbacks = [trackers[condition]]
        results[condition] = run_condition(
            f"{condition} GEPA (GSM8K {args.seed_style} seeds{' defective' if args.defective_seed else ''})",
            seed_candidate(**seed_kwargs),
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

    # Report: best prompts (full text)
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
        seed_candidate(**seed_kwargs),
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
