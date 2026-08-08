"""HotpotQA evaluation: vanilla GEPA vs random vs verbalized action selection.

Replicates the GEPA paper's HotpotQA setup (hotpot_qa distractor, 113K) with
exact splits (150 train / 300 val / 300 test like paper Table 1), a 2-stage
query-generation program, and the official token-F1/EM metrics. The default
budget of 6871 metric calls matches the paper (MIPROv2-Heavy's invocation count
for HotpotQA). Scaled and smoke budgets are also supported.

Conditions:
    vanilla  - stock GEPA reflective mutation
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python examples/hotpotqa/main.py [--condition vanilla|random|action|all]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
    # Smoke (20 ex, 14/3/3):
    uv run python examples/hotpotqa/main.py --data-path examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl --max-metric-calls 200 --condition both
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

from examples.hotpotqa.utils import (
    em_score,
    f1_score,
    hotpotqa_metric,
    load_hotpotqa_dataset,
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

# 2-stage query-generation seeds (paper-faithful)
# Stage 1: query generation for the second hop; Stage 2: multi-hop answering.
SEED_CANDIDATE = {
    "generate_query": "Generate a concise search query that captures the missing information needed to answer the multi-hop question.",
    "generate_answer": (
        "You are a multi-hop question answering system. Read the provided context passages "
        "carefully and answer the question. Some passages may be irrelevant distractors. "
        "Chain your reasoning across multiple passages to find the answer. "
        "Give a concise, direct answer."
    ),
}

# 1-stage ablation (single prompt)
SEED_CANDIDATE_1STAGE = {
    "answer_question": (
        "You are a multi-hop question answering system. Read the provided context passages "
        "carefully and answer the question. Some passages may be irrelevant distractors. "
        "Chain your reasoning across multiple passages to find the answer. "
        "Give a concise, direct answer."
    ),
}

_INITIAL_PROMPT = SEED_CANDIDATE_1STAGE["answer_question"]

_CONDITION_DIR_NAMES = {
    "vanilla": "hotpotqa_vanilla",
    "random": "hotpotqa_random_action",
    "action": "hotpotqa_verbalized_action",
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


def run_program(candidate: dict, context: str, question: str, program: str, model: str, api_base: str | None) -> tuple[str | None, str]:
    """Run the candidate program, returning (stage1_query_or_None, final_answer)."""
    if isinstance(candidate, str):
        return None, run_single_stage(candidate, context, question, model=model, api_base=api_base)
    if program == "1stage":
        prompt = candidate.get("answer_question") or next(iter(candidate.values()))
        return None, run_single_stage(prompt, context, question, model=model, api_base=api_base)
    q_prompt = candidate.get("generate_query", "")
    a_prompt = candidate.get("generate_answer", "")
    query, answer = run_two_stage(q_prompt, a_prompt, context, question, model=model, api_base=api_base)
    return query, answer


def make_evaluator(solver_model: str, api_base: str | None = None, program: str = "2stage"):
    """Create an evaluator function closed over the solver model name."""

    def evaluate(candidate, example: dict) -> tuple[float, SideInfo]:
        # candidate may be dict (optimized) or str (legacy)
        if isinstance(candidate, str):
            # 1-stage string seed
            prediction = run_single_stage(candidate, example["context"], example["question"], model=solver_model, api_base=api_base)
            stage1 = None
        elif program == "1stage":
            prompt = candidate.get("answer_question") or next(iter(candidate.values()))
            prediction = run_single_stage(prompt, example["context"], example["question"], model=solver_model, api_base=api_base)
            stage1 = None
        else:
            q_prompt = candidate.get("generate_query", "")
            a_prompt = candidate.get("generate_answer", "")
            stage1, prediction = run_two_stage(q_prompt, a_prompt, example["context"], example["question"], model=solver_model, api_base=api_base)

        score, feedback = hotpotqa_metric(prediction, example["answer"])

        side_info: SideInfo = {
            "score": score,
            "question": example["question"],
            "output": prediction,
            "answer": example["answer"],
            "execution_feedback": feedback,
        }
        if stage1 is not None:
            side_info["generated_query"] = stage1
        # Also include EM for logging (not primary score)
        side_info["em"] = em_score(prediction, example["answer"])
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate,
    dataset: list[dict],
    solver_model: str,
    api_base: str | None = None,
    max_workers: int = 24,
    program: str = "2stage",
) -> tuple[float, float]:
    """Evaluate a candidate on a dataset, returning (mean F1, mean EM)."""

    def score_one(example: dict) -> tuple[float, float]:
        if isinstance(candidate, str):
            pred = run_single_stage(candidate, example["context"], example["question"], model=solver_model, api_base=api_base)
        elif program == "1stage":
            prompt = candidate.get("answer_question") or next(iter(candidate.values()))
            pred = run_single_stage(prompt, example["context"], example["question"], model=solver_model, api_base=api_base)
        else:
            q_prompt = candidate.get("generate_query", "")
            a_prompt = candidate.get("generate_answer", "")
            _, pred = run_two_stage(q_prompt, a_prompt, example["context"], example["question"], model=solver_model, api_base=api_base)
        return f1_score(pred, example["answer"]), em_score(pred, example["answer"])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    if not scores:
        return 0.0, 0.0
    mean_f1 = sum(s[0] for s in scores) / len(scores)
    mean_em = sum(s[1] for s in scores) / len(scores)
    return mean_f1, mean_em


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
    """Textual diversity of explored candidates, per component."""
    if not candidates:
        return {}
    # Normalize string candidates to dict for diversity
    norm = []
    for c in candidates:
        if isinstance(c, str):
            norm.append({"prompt": c})
        else:
            norm.append(c)
    if not norm:
        return {}
    diversity: dict[str, dict[str, float]] = {}
    for component in norm[0]:
        texts = [c[component] for c in norm]
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
    import inspect

    action_space = build_structured_actions() if args.actions == "structured" else DEFAULT_ACTIONS
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(args.reflection_model, **(reflection_lm_kwargs or {})),
        )

    # Support both old GEPAConfig (with ReflectionConfig.action_selector) and new
    # engine-pluggable path where action_selector lives on the reflection strategy.
    engine_cfg = EngineConfig(
        run_dir=condition_run_dir(condition, args.program, args.tag),
        max_metric_calls=args.max_metric_calls,
        parallel=True,
        max_workers=24,
        cache_evaluation=True,
    )

    # Prefer the legacy ReflectionConfig.action_selector if the installed GEPA still has it
    try:
        sig = inspect.signature(ReflectionConfig)
        if "action_selector" in sig.parameters:
            config = GEPAConfig(
                engine=engine_cfg,
                reflection=ReflectionConfig(
                    reflection_lm=args.reflection_model,
                    reflection_lm_kwargs=reflection_lm_kwargs or None,
                    action_selector=action_selector,
                ),
            )
            return config, action_selector
    except Exception:
        pass

    # New path: wrap the selector in a StatelessReflectionLM and pass as reflection_strategy
    if action_selector is not None:
        try:
            sig2 = inspect.signature(ReflectionConfig)
            if "reflection_strategy" in sig2.parameters:
                from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM

                lm = LM(args.reflection_model, **(reflection_lm_kwargs or {}))
                strategy = StatelessReflectionLM(lm=lm, action_selector=action_selector)
                config = GEPAConfig(
                    engine=engine_cfg,
                    reflection=ReflectionConfig(
                        reflection_lm=args.reflection_model,
                        reflection_lm_kwargs=reflection_lm_kwargs or None,
                        reflection_strategy=strategy,
                    ),
                )
                return config, action_selector
        except Exception as e:
            print(f"WARNING: action_selector via reflection_strategy failed ({e}); falling back to vanilla reflection.")

    config = GEPAConfig(
        engine=engine_cfg,
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs or None,
        ),
    )
    return config, action_selector


def run_condition(
    name: str,
    seed: dict | str,
    trainset: list[dict],
    valset: list[dict],
    config: GEPAConfig,
    evaluator,
    callbacks: list | None = None,
):
    """Run one optimization condition and return the result."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}\n")

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
    parser = argparse.ArgumentParser(description="HotpotQA evaluation for action-conditioned reflection")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HotpotQA JSONL sample (smoke, 14/3/3)")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=6871,
        help="Budget per condition (paper: 6871, smoke: 200, scaled Wave B: 15000)",
    )
    parser.add_argument("--solver-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Solver LM model (litellm format)")
    parser.add_argument("--reflection-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Reflection LM model (litellm format)")
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 150)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 300)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size (paper: 300)")
    parser.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed")
    parser.add_argument(
        "--program",
        type=str,
        default="2stage",
        choices=["2stage", "1stage"],
        help="Program structure: 2stage (query-generation paper protocol) or 1stage (single-turn ablation)",
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
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev2, 6871)")
    args = parser.parse_args()

    if args.actions == "structured" and args.seed_style != "structured":
        print("--actions structured implies --seed-style structured; overriding seed style.")
        args.seed_style = "structured"

    # Load dataset: HF distractor 150/300/300 or smoke fallback
    trainset, valset, testset = load_hotpotqa_dataset(
        data_path=args.data_path, train_limit=args.train_limit, val_limit=args.val_limit, test_limit=args.test_limit, seed=args.seed
    )
    # Apply limits again for explicit --train/--val/--test limits (already handled, but keep for --data-path smoke)
    # Already applied in loader.

    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples ({args.program}, {args.seed_style})")
    if args.data_path:
        print(f"  (from {args.data_path})")
    else:
        print("  (from hotpot_qa distractor via datasets; paper Table 1: 150/300/300)")

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
        # Seed: 1stage uses single-prompt dict; 2stage uses 2-prompt dict; legacy string still works
        seed = seed_candidate(args.program, args.seed_style)
        # Backward compat: if program 1stage and someone expects string, we still pass dict
        results[condition] = run_condition(
            f"{condition} GEPA ({args.program}, {args.seed_style} seeds)",
            seed,
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
    print(f"\n{'='*60}")
    print("  Best prompts")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"\n----- [{name}] best candidate (val score {result.val_aggregate_scores[result.best_idx]:.4f}) -----")
        cand = result.best_candidate
        if isinstance(cand, str):
            print(cand)
        else:
            for component, text in cand.items():
                print(f"\n[{name}] {component}:\n{text}")

    # Report: test F1/EM + diversity
    print(f"\n{'='*60}")
    print("  Comparison")
    print(f"{'='*60}\n")

    baseline = seed_candidate(args.program, args.seed_style)
    baseline_f1, baseline_em = evaluate_on_set(baseline, testset, args.solver_model, api_base=args.api_base, program=args.program)
    print(f"Baseline (seed prompts) test F1: {baseline_f1:.2%} EM: {baseline_em:.2%} on {len(testset)} examples\n")

    for name, result in results.items():
        test_f1, test_em = evaluate_on_set(result.best_candidate, testset, args.solver_model, api_base=args.api_base, program=args.program)
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score (F1):      {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test F1:                  {test_f1:.2%}")
        print(f"  test EM:                  {test_em:.2%}")
        for component, stats in diversity.items():
            print(
                f"  diversity[{component}]: jaccard_dist={stats['mean_pairwise_jaccard_distance']:.3f} "
                f"unique={int(stats['num_unique_texts'])}/{len(result.candidates)}"
            )
        print()

    # Action diversity metrics (random / action conditions)
    for name, tracker in trackers.items():
        print(f"{'='*60}")
        print(f"  Action Diversity Metrics [{name}]")
        print(f"{'='*60}\n")
        summary = tracker.summary()
        print(f"Total proposals: {summary['total_proposals']}")
        print(f"Total accepted:  {summary['total_accepted']}")
        print(f"\nPer-action proposal counts: {summary['action_proposal_counts']}")
        print(f"Per-action acceptance rates: {summary['action_acceptance_rates']}")
        print(f"Textual diversity per iteration: {summary['textual_diversity_per_iteration']}\n")


if __name__ == "__main__":
    main()
