"""AppWorld evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the GEPA paper's AppWorld setup (Trivedi et al. 2024,
https://appworld.dev — 9 everyday apps, 168 tool APIs, 750 tasks):
a skill-based agent program whose prompt(s) are optimized on 60 train
examples, with 75 val examples for Pareto selection and the remaining test
tasks held out for final scoring. The metric is task success rate (all
subtasks/evaluations pass, TGC). The default budget of 3000-5000 metric calls
mirrors the paper's MIPRO-style budget (3000 for MIPRO-light, ~5000 for heavy;
see GEPA paper Sec 4 / MIPROv2 ablations and AppWorld evaluation budget
discussion). Use 4000 as the default like other mid-scale benchmarks.

Programs:
    1stage  - skill-based agent: one optimized system prompt
    2stage  - plan-then-execute: plan prompt + execution prompt

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python -m examples.appworld.main [--condition vanilla|random|action|all]
        [--program 1stage|2stage] [--max-metric-calls N]
        [--train-limit N] [--val-limit N] [--test-limit N]
        [--data-path PATH]
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.appworld.utils import (
    appworld_metric,
    load_appworld_dataset,
    run_single_stage,
    run_two_stage,
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

# Seed prompts: single system prompt (1stage) and plan/execute pair (2stage).
# Tuned to describe the AppWorld tool-use task without assuming a specific
# harness — the real AppWorld supervisor runs evaluation code; offline the
# metric uses subtask string checks.
SEED_CANDIDATE_1STAGE = {
    "system_prompt": (
        "You are an AI assistant that helps users complete everyday tasks "
        "using a suite of apps (email, calendar, contacts, banking, shopping, "
        "and more). You have access to 168 tool APIs. Read the task, think "
        "step by step about which tools to call and in what order, then "
        "produce the final answer or tool-call sequence that completes the task."
    ),
}

SEED_CANDIDATE_2STAGE = {
    "plan": (
        "You are a planning assistant for the AppWorld environment (9 apps, "
        "168 tool APIs). Given a task, produce a concise high-level plan: "
        "list the subtasks, the apps/tools each subtask needs, and the order "
        "to execute them. Do not yet produce the final answer — just the plan."
    ),
    "execute": (
        "You are an execution assistant for the AppWorld environment. Given a "
        "task and a plan, carry out the plan step by step using the available "
        "tool APIs. Produce the final answer or action sequence that completes "
        "the task. Ensure every subtask in the plan is addressed."
    ),
}

_CONDITION_DIR_NAMES = {
    "vanilla": "appworld_vanilla",
    "random": "appworld_random_action",
    "action": "appworld_verbalized_action",
}


def _structured_seed(text: str) -> str:
    """Wrap a seed sentence in a best-practice markdown skeleton.

    The paper's seed sentence is preserved verbatim in the Task section; other
    sections start as explicit placeholders for section-scoped actions to fill.
    """
    return (
        "## Role\n(none yet)\n\n"
        f"## Task\n{text}\n\n"
        "## Rules\n(none yet)\n\n"
        "## Output Format\n(none yet)\n\n"
        "## Examples\n(none yet)"
    )


def condition_run_dir(condition: str, program: str, tag: str = "") -> str:
    suffix = "_1stage" if program == "1stage" else ""
    tag_suffix = f"_{tag}" if tag else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{suffix}{tag_suffix}"


def seed_candidate(program: str, seed_style: str = "plain") -> dict:
    base = SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE_2STAGE
    seed = dict(base)
    if seed_style == "structured":
        seed = {component: _structured_seed(text) for component, text in seed.items()}
    return seed


def run_program(candidate: dict, task: str, program: str, model: str, api_base: str | None) -> tuple[str | None, str]:
    """Run the candidate program on a task, returning (plan_or_None, final_response)."""
    if program == "1stage":
        return None, run_single_stage(candidate["system_prompt"], task, model=model, api_base=api_base)
    plan, final = run_two_stage(
        candidate["plan"],
        candidate["execute"],
        task,
        model=model,
        api_base=api_base,
    )
    return plan, final


def make_evaluator(solver_model: str, api_base: str | None = None, program: str = "1stage"):
    """Create an evaluator function closed over the solver model name."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        response, final_response = run_program(candidate, example["prompt"], program, solver_model, api_base)
        score, feedback = appworld_metric(final_response, example)
        side_info: SideInfo = {
            "score": score,
            "query": example["prompt"],
            "output": final_response,
            "execution_feedback": feedback,
        }
        if response is not None:
            side_info["stage1_response"] = response
        # Preserve task id for analysis
        if example.get("task_id"):
            side_info["task_id"] = example["task_id"]
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate: dict,
    dataset: list[dict],
    solver_model: str,
    api_base: str | None = None,
    max_workers: int = 24,
    program: str = "1stage",
) -> float:
    """Evaluate a candidate on a dataset, returning mean task success rate."""

    def score_one(example: dict) -> float:
        _, final_response = run_program(candidate, example["prompt"], program, solver_model, api_base)
        score, _ = appworld_metric(final_response, example)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
    """Textual diversity of explored candidates, per component.

    For each component, computes the mean pairwise Jaccard distance
    (1 - |A intersect B| / |A union B| over lowercased token sets) across all
    explored candidate texts, plus the number of unique texts.
    """
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
    parser = argparse.ArgumentParser(description="AppWorld evaluation for action-conditioned reflection")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=4000,
        help="Budget per condition (paper MIPRO-style: 3000 light / 5000 heavy; AppWorld default 4000)",
    )
    parser.add_argument(
        "--solver-model", type=str, default="hosted_vllm/Qwen3-8B", help="Solver LM model (litellm format)"
    )
    parser.add_argument(
        "--reflection-model", type=str, default="hosted_vllm/Qwen3-8B", help="Reflection LM model (litellm format)"
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to AppWorld data file or directory (JSONL/JSON); else tries HF appworld/appworld then data/*.jsonl",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for splits")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (default: 60)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (default: 75)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size for final evaluation")
    parser.add_argument(
        "--program",
        type=str,
        default="1stage",
        choices=["1stage", "2stage"],
        help="Program structure: 1stage (skill-based agent) or 2stage (plan then execute)",
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
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev1, 4k)")
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    trainset, valset, testset = load_appworld_dataset(data_path=args.data_path, split_seed=args.seed)
    if args.train_limit is not None:
        trainset = trainset[: args.train_limit]
    if args.val_limit is not None:
        valset = valset[: args.val_limit]
    if args.test_limit is not None:
        testset = testset[: args.test_limit]
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples ({args.program})")

    evaluator = make_evaluator(args.solver_model, api_base=args.api_base, program=args.program)

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
            f"{condition} GEPA ({args.program}, {args.seed_style} seeds)",
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

    # Report: test success rate + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    baseline_score = evaluate_on_set(
        seed_candidate(args.program, args.seed_style),
        testset,
        args.solver_model,
        api_base=args.api_base,
        program=args.program,
    )
    print(f"Baseline (seed prompts) test task success rate: {baseline_score:.2%} on {len(testset)} examples\n")

    for name, result in results.items():
        test_score = evaluate_on_set(
            result.best_candidate, testset, args.solver_model, api_base=args.api_base, program=args.program
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test task success rate:   {test_score:.2%}")
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
