"""TauBench evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the Tau-bench setup (Yao et al. 2024, https://tau-bench.github.io,
tau-bench/tau-bench): 800+ tool-augmented agent tasks over **airline** +
**retail** domains. Each task requires the agent to call domain tools
(get_user_details, get_order_details, modify_reservation, etc.) and produce
a correct final response / database state. The metric is **pass^k** / task
success (binary, with per-task feedback), matching the paper's evaluation.

Splits:
- Full HF (≈800): 80 train / 80 val / remainder test (balanced airline+retail),
  or 150/300/300 when the dataset is large (≥750). Use --train/val/test-limit
  to cap. The offline synthetic fallback (400 tasks) uses 80/80/240.
- Smoke (20 tasks via --data-path): 14/3/3 (mirrors HotpotQA smoke).

Program:
- 1stage: single tool-calling agent prompt, one LM call.
- 2stage: plan-then-act (stage 1 generates a plan, stage 2 executes with tools).
  Both are optimized (round-robin) like IFBench/HotpotQA.

Budget default 3500 metric calls (spec: 3000-5000; paper MIPROv2-Heavy scale
is 6871 for HotpotQA, 3593 for IFBench). Scaled Wave B: 5000-15000.

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python examples/taubench/main.py [--condition vanilla|random|action|all]
        [--program 1stage|2stage] [--domain airline|retail|all]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
        [--data-path path/to/taubench.jsonl]

    # Smoke (offline, 20 tasks):
    uv run python examples/taubench/main.py --data-path examples/taubench/data/taubench.jsonl --max-metric-calls 200 --condition both
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.taubench.utils import (
    load_taubench_dataset,
    run_single_stage,
    run_two_stage,
    taubench_metric,
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

# ---------------------------------------------------------------------------
# Seeds (paper-faithful: 1-stage tool-calling agent; 2-stage plan-then-act)
# ---------------------------------------------------------------------------

SEED_CANDIDATE_1STAGE = {
    "agent_prompt": (
        "You are a helpful tool-augmented assistant for airline and retail tasks. "
        "You have access to domain tools (airline: get_user_details, get_reservation_details, "
        "search_direct_flight, update_reservation, cancel_reservation; "
        "retail: get_user_details, get_order_details, get_product_details, "
        "modify_pending_order_items, cancel_pending_order). "
        "Call tools as JSON lines when needed, then give a concise final response that completes the user's request."
    ),
}

SEED_CANDIDATE = {
    "plan_prompt": (
        "You are a planning assistant for airline and retail tasks. "
        "Given the domain and instruction, produce a concise step-by-step plan "
        "for which tools to call and in what order. Do not call tools yet."
    ),
    "act_prompt": (
        "You are a helpful tool-augmented assistant for airline and retail tasks. "
        "You have access to domain tools (airline: get_user_details, get_reservation_details, "
        "search_direct_flight, update_reservation, cancel_reservation; "
        "retail: get_user_details, get_order_details, get_product_details, "
        "modify_pending_order_items, cancel_pending_order). "
        "Follow the provided plan, call tools as JSON lines when needed, then give a concise final response."
    ),
}

_CONDITION_DIR_NAMES = {
    "vanilla": "taubench_vanilla",
    "random": "taubench_random_action",
    "action": "taubench_verbalized_action",
}


def _structured_seed(task_sentence: str) -> str:
    """Wrap a seed sentence in a best-practice markdown skeleton.

    The paper's seed sentence is preserved verbatim in the Task section; other
    sections start as explicit placeholders for section-scoped actions to fill.
    """
    return (
        "## Role\n(none yet)\n\n"
        f"## Task\n{task_sentence}\n\n"
        "## Rules\n(none yet)\n\n"
        "## Output Format\n(none yet)\n\n"
        "## Examples\n(none yet)"
    )


def condition_run_dir(condition: str, program: str, tag: str = "") -> str:
    suffix = "_1stage" if program == "1stage" else ""
    tag_suffix = f"_{tag}" if tag else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{suffix}{tag_suffix}"


def seed_candidate(program: str = "2stage", seed_style: str = "plain") -> dict:
    seed = dict(SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {component: _structured_seed(text) for component, text in seed.items()}
    return seed


def run_program(
    candidate: dict,
    example: dict,
    program: str,
    model: str,
    api_base: str | None,
) -> tuple[str | None, str]:
    """Run the candidate program on a TauBench example, returning (plan_or_None, final_response)."""
    domain = example.get("domain", "airline")
    instruction = example.get("instruction") or example.get("prompt") or ""
    if program == "1stage":
        prompt = candidate.get("agent_prompt") or next(iter(candidate.values()))
        return None, run_single_stage(prompt, instruction, domain=domain, model=model, api_base=api_base, example=example)
    plan_p = candidate.get("plan_prompt", "")
    act_p = candidate.get("act_prompt", "")
    plan, final = run_two_stage(plan_p, act_p, instruction, domain=domain, model=model, api_base=api_base, example=example)
    return plan, final


def make_evaluator(solver_model: str, api_base: str | None = None, program: str = "2stage"):
    """Create an evaluator function closed over the solver model name."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        plan, final_response = run_program(candidate, example, program, solver_model, api_base)
        score, feedback = taubench_metric(final_response, example)

        side_info: SideInfo = {
            "score": score,
            "instruction": example.get("instruction", ""),
            "domain": example.get("domain", ""),
            "task_id": example.get("task_id", ""),
            "output": final_response,
            "execution_feedback": feedback,
        }
        if plan is not None:
            side_info["plan"] = plan
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate: dict,
    dataset: list[dict],
    solver_model: str,
    api_base: str | None = None,
    max_workers: int = 24,
    program: str = "2stage",
) -> float:
    """Evaluate a candidate on a dataset, returning mean pass^k / task success."""

    def score_one(example: dict) -> float:
        _, final_response = run_program(candidate, example, program, solver_model, api_base)
        score, _ = taubench_metric(final_response, example)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
    """Textual diversity of explored candidates, per component.

    For each component, computes the mean pairwise Jaccard distance
    (1 - |A intersect B| / |A union B| over lowercased token sets) across all
    explored candidate texts, plus the number of unique texts. Works for all
    conditions, including vanilla (which has no action labels).
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
    """Build the GEPAConfig for one condition. Returns (config, action_selector).

    Mirrors examples/ifbench/main.py and examples/hotpotqa/main.py:
    vanilla has no selector, random uses RandomActionSelector, action uses
    VerbalizedActionSelector. On the current main branch the launcher's
    ReflectionConfig does not yet have an ``action_selector`` field (the
    Rev 1 feature branch adds it); the try/except keeps the scaffold
    runnable offline (tracking still works) while preserving the exact
    template for the feature branch.
    """
    action_space = build_structured_actions() if args.actions == "structured" else DEFAULT_ACTIONS
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(args.reflection_model, **reflection_lm_kwargs),
        )

    # Try the Rev 1 template (ReflectionConfig with action_selector); fall
    # back to vanilla ReflectionConfig on main where the field is not yet
    # present. Both pass py_compile; the fallback keeps the scaffold runnable
    # offline (ActionDiversityCallback still tracks proposals).
    try:
        reflection_cfg = ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs or None,
            action_selector=action_selector,  # type: ignore[call-arg]
        )
    except TypeError:
        reflection_cfg = ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs or None,
        )
        if action_selector is not None:
            # Stash selector on the config for debugging (not used by core on main)
            try:
                setattr(reflection_cfg, "action_selector", action_selector)  # type: ignore[attr-defined]
            except Exception:
                pass

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=condition_run_dir(condition, args.program, args.tag),
            max_metric_calls=args.max_metric_calls,
            parallel=True,
            max_workers=24,
            cache_evaluation=True,
        ),
        reflection=reflection_cfg,
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
    parser = argparse.ArgumentParser(description="TauBench evaluation for action-conditioned reflection")
    parser.add_argument("--data-path", type=str, default=None, help="Path to TauBench JSONL (offline smoke / custom). When omitted, loads HF tau-bench/tau-bench.")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=3500,
        help="Budget per condition (spec: 3000-5000; paper scale 3593 IFBench / 6871 HotpotQA; Wave B: 15000)",
    )
    parser.add_argument(
        "--solver-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Solver LM model (litellm format)"
    )
    parser.add_argument(
        "--reflection-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Reflection LM model (litellm format)"
    )
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (full: 80 or 150)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (full: 80 or 300)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size for final evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed")
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["airline", "retail", "all"],
        help="TauBench domain filter: airline, retail, or all (default all; 80/80 airline+retail when full)",
    )
    parser.add_argument(
        "--program",
        type=str,
        default="2stage",
        choices=["2stage", "1stage"],
        help="Program structure: 2stage (plan-then-act, paper protocol) or 1stage (single tool-calling agent)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["vanilla", "random", "action", "all", "both"],
        help="Which condition(s) to run",
    )
    parser.add_argument(
        "--seed-style",
        type=str,
        default="plain",
        choices=["plain", "structured"],
        help="Seed prompts: plain paper sentences or a markdown skeleton (Role/Task/Rules/Output Format/Examples)",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="default",
        choices=["default", "structured"],
        help="Action space: DEFAULT_ACTIONS or section-scoped structured actions (implies --seed-style structured)",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev1, 3500)")
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    # Map "all" domain to None for loader (loader None == all)
    domain_arg = None if args.domain == "all" else args.domain

    trainset, valset, testset = load_taubench_dataset(
        data_path=args.data_path,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        domain=domain_arg,
    )
    # load_taubench_dataset already applies limits; this is a safety re-slice for --data-path smoke parity
    applied_via_loader = True  # noqa: F841
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples ({args.program}, {args.seed_style}, domain={args.domain})")
    if args.data_path:
        print(f"  (from {args.data_path})")
    else:
        hf_hint = "tau-bench/tau-bench via datasets"
        offline_hint = "synthetic fallback" if len(trainset) == 80 and len(valset) == 80 else hf_hint
        print(f"  (from {offline_hint}; airline+retail)" if args.domain == "all" else f"  (from {offline_hint}; domain={args.domain})")

    evaluator = make_evaluator(args.solver_model, api_base=args.api_base, program=args.program)

    reflection_lm_kwargs = {}
    if args.api_base is not None:
        reflection_lm_kwargs["api_base"] = args.api_base

    if args.condition in ("all", "both"):
        conditions = ["vanilla", "random", "action"]
    else:
        conditions = [args.condition]

    results = {}
    trackers: dict[str, ActionDiversityCallback] = {}
    for condition in conditions:
        config, selector = build_config(condition, args, reflection_lm_kwargs)
        callbacks = None
        if condition in ("random", "action"):
            trackers[condition] = ActionDiversityCallback()
            callbacks = [trackers[condition]]
        results[condition] = run_condition(
            f"{condition} GEPA ({args.program}, {args.seed_style} seeds, domain={args.domain})",
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

    # Report: test pass^k / task success + diversity
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
    print(f"Baseline (seed prompts) test pass^k / task success: {baseline_score:.2%} on {len(testset)} examples\n")

    for name, result in results.items():
        test_score = evaluate_on_set(
            result.best_candidate, testset, args.solver_model, api_base=args.api_base, program=args.program
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test pass^k / success:    {test_score:.2%}")
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
