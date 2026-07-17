"""HotpotQA evaluation: vanilla GEPA vs action-conditioned reflection.

Runs two optimization conditions on a 20-record HotpotQA distractor sample
(14 train / 6 val) and compares final F1, convergence, and proposal diversity.

Usage:
    export OPENAI_API_KEY=...
    uv run python examples/hotpotqa/main.py [--data-path PATH] [--max-metric-calls N]
"""

import argparse

from examples.hotpotqa.utils import (
    f1_score,
    hotpotqa_metric,
    load_hotpotqa_dataset,
    run_llm,
)
from gepa.core.action_tracking import ActionDiversityCallback
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)
from gepa.strategies.action_space import DEFAULT_ACTIONS, RoundRobinActionSelector


INITIAL_PROMPT = (
    "You are a multi-hop question answering system. Read the provided context passages "
    "carefully and answer the question. Some passages may be irrelevant distractors. "
    "Chain your reasoning across multiple passages to find the answer. "
    "Give a concise, direct answer."
)


def make_evaluator(solver_model: str, api_base: str | None = None):
    """Create an evaluator function closed over the solver model name."""

    def evaluate(candidate: str, example: dict) -> tuple[float, SideInfo]:
        prediction = run_llm(candidate, example["context"], example["question"], model=solver_model, api_base=api_base)
        score, feedback = hotpotqa_metric(prediction, example["answer"])

        side_info: SideInfo = {
            "score": score,
            "question": example["question"],
            "output": prediction,
            "answer": example["answer"],
            "execution_feedback": feedback,
        }
        return score, side_info

    return evaluate


def evaluate_on_valset(prompt: str, valset: list[dict], solver_model: str, api_base: str | None = None) -> float:
    """Evaluate a prompt on the valset, returning mean F1."""
    scores = []
    for ex in valset:
        prediction = run_llm(prompt, ex["context"], ex["question"], model=solver_model, api_base=api_base)
        scores.append(f1_score(prediction, ex["answer"]))
    return sum(scores) / len(scores) if scores else 0.0


def run_condition(
    name: str,
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
        seed_candidate=INITIAL_PROMPT,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        config=config,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="HotpotQA evaluation for action-conditioned reflection")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HotpotQA JSONL sample")
    parser.add_argument("--max-metric-calls", type=int, default=200, help="Budget per condition")
    parser.add_argument("--solver-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Solver LM model (litellm format)")
    parser.add_argument("--reflection-model", type=str, default="hosted_vllm/Qwen3.5-9B", help="Reflection LM model (litellm format)")
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument(
        "--condition",
        type=str,
        default="both",
        choices=["vanilla", "action", "both"],
        help="Which condition(s) to run",
    )
    args = parser.parse_args()

    trainset, valset = load_hotpotqa_dataset(args.data_path)
    print(f"Loaded {len(trainset)} train / {len(valset)} val examples")

    evaluator = make_evaluator(args.solver_model, api_base=args.api_base)
    results = {}
    tracker = None

    reflection_lm_kwargs = {}
    if args.api_base is not None:
        reflection_lm_kwargs["api_base"] = args.api_base

    # Condition A: Vanilla GEPA
    if args.condition in ("vanilla", "both"):
        vanilla_config = GEPAConfig(
            engine=EngineConfig(
                run_dir="outputs/hotpotqa_vanilla",
                max_metric_calls=args.max_metric_calls,
                parallel=True,
                max_workers=8,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=args.reflection_model,
                reflection_lm_kwargs=reflection_lm_kwargs or None,
            ),
        )
        results["vanilla"] = run_condition("Vanilla GEPA", trainset, valset, vanilla_config, evaluator)

    # Condition B: Action-Conditioned Reflection
    if args.condition in ("action", "both"):
        tracker = ActionDiversityCallback()
        action_config = GEPAConfig(
            engine=EngineConfig(
                run_dir="outputs/hotpotqa_action_conditioned",
                max_metric_calls=args.max_metric_calls,
                parallel=True,
                max_workers=8,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=args.reflection_model,
                reflection_lm_kwargs=reflection_lm_kwargs or None,
                action_selector=RoundRobinActionSelector(DEFAULT_ACTIONS),
            ),
        )
        results["action"] = run_condition(
            "Action-Conditioned GEPA", trainset, valset, action_config, evaluator, callbacks=[tracker]
        )

    # Report results
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}\n")

    for name, result in results.items():
        best_prompt = result.best_candidate
        print(f"[{name}] Best prompt:\n  {best_prompt[:200]}{'...' if len(best_prompt) > 200 else ''}\n")

    # Evaluate best prompts on valset
    print("Evaluating best prompts on valset...\n")
    for name, result in results.items():
        val_score = evaluate_on_valset(result.best_candidate, valset, args.solver_model, api_base=args.api_base)
        print(f"[{name}] Val F1: {val_score:.2%}")

    # Baseline
    baseline_score = evaluate_on_valset(INITIAL_PROMPT, valset, args.solver_model, api_base=args.api_base)
    print(f"\nBaseline (initial prompt) Val F1: {baseline_score:.2%}")

    # Action diversity metrics
    if tracker is not None:
        print(f"\n{'='*60}")
        print("  Action Diversity Metrics")
        print(f"{'='*60}\n")
        summary = tracker.summary()
        print(f"Total proposals: {summary['total_proposals']}")
        print(f"Total accepted:  {summary['total_accepted']}")
        print(f"\nPer-action proposal counts: {summary['action_proposal_counts']}")
        print(f"Per-action acceptance rates: {summary['action_acceptance_rates']}")
        print(f"Textual diversity per iteration: {summary['textual_diversity_per_iteration']}")


if __name__ == "__main__":
    main()
