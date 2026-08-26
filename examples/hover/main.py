"""HoVer evaluation: vanilla GEPA vs Controller/Manifestor/ReAct V2/RLM.

Replicates the GEPA artifact's HoVer experiment over official v1.1 claims with
exactly three supporting documents. The program performs three live Wikipedia
retrieval hops and optimizes two summarizers and two query generators on 150 train
examples, with 300 val examples for Pareto selection and 300 test examples
held out for final scoring. The primary metric is complete gold-document
retrieval, with supporting-document recall reported separately. The default budget
of 7051 metric calls matches the paper.

Conditions:
    vanilla  - stock GEPA reflective mutation
    react_v2 - section/action Controller, Manifestor configured for the model provider, ReAct V2 proposer
    rlm      - RLM proposer executing trusted model code in process
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python -m examples.hover.main [--condition vanilla|react_v2|rlm|both]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from examples.common.react_v2 import (
    benchmark_data_identity,
    build_react_v2_strategy,
    ensure_wikipedia_run_contract,
    experiment_run_key,
    file_sha256,
    resolve_template_family,
    structured_prompt,
)
from examples.common.wikipedia import DEFAULT_WIKIPEDIA_ENDPOINT, WikipediaClient, WikipediaPassage, WikipediaRetriever
from examples.hover.utils import (
    DATA_DIR,
    HOVER_TRAIN_FILE,
    hover_metric,
    hover_recall,
    load_hover_dataset,
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
from gepa.proposer.reflective_mutation.rlm_environment import RLMBudget
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.action_space import (
    RandomActionSelector,
    VerbalizedActionSelector,
    stateless_selector_policy_contract,
)
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import (
    build_stateless_action_menu,
    controller_policy_contract,
    semantic_action_catalog,
    stateless_action_menu_contract,
)

# GEPA artifact components: summarize1 -> create_query_hop2 -> summarize2 -> create_query_hop3.
SEED_CANDIDATE = {
    "summarize1": "Summarize the first-hop Wikipedia passages and preserve clues needed for further retrieval.",
    "create_query_hop2": "Generate a focused second-hop Wikipedia query from the claim and first-hop summary.",
    "summarize2": "Synthesize the first-hop summary with second-hop passages to expose missing evidence links.",
    "create_query_hop3": "Generate a final Wikipedia query from the claim and both retrieval-hop summaries.",
}

# Ablation seed: generate one query, then score the pages that query retrieves.
SEED_CANDIDATE_1STAGE = {
    "retrieve": "Write one focused Wikipedia search query for the evidence needed to verify the claim."
}

_CONDITION_DIR_NAMES = {
    "vanilla": "hover_vanilla",
    "react_v2": "hover_react_v2",
    "rlm": "hover_rlm",
    "random": "hover_random_action",
    "action": "hover_verbalized_action",
}


def _component_kinds(program: str) -> dict[str, str]:
    """Identify every optimized HOVER instruction as a system message."""
    seed = SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE
    return dict.fromkeys(seed, "system_prompt")


def _rlm_budget() -> RLMBudget:
    """Return the RLM budget matched to ReAct V2's eight proposer turns."""
    return RLMBudget(
        max_root_iterations=4,
        max_child_iterations=2,
        max_repl_calls=6,
        max_llm_queries=2,
        max_rlm_queries=1,
        max_recursion_depth=1,
        max_exec_seconds=5,
        max_output_chars=4000,
    )


def _rlm_max_model_calls(budget: RLMBudget) -> int:
    """Derive the root, child, and leaf model-call cap."""
    return budget.max_model_calls


def condition_run_dir(condition: str, program: str, tag: str = "", run_key: str = "") -> str:
    suffix = "_1stage" if program == "1stage" else ""
    tag_suffix = f"_{tag}" if tag else ""
    key_suffix = f"_{run_key}" if run_key else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{suffix}{key_suffix}{tag_suffix}"


def build_run_contract(condition: str, args) -> dict:
    """Build the complete persisted configuration for one condition."""
    family = resolve_template_family(args.template_family, args.solver_model)
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    operated = condition in ("react_v2", "rlm")
    reflection_level = args.reflection_level if operated else 0
    edit_tool_set = args.edit_tool_set if operated else None
    stateless_semantic = condition in ("random", "action")
    template = TEMPLATE_FAMILIES[family]["system_prompt"]
    rlm_budget = _rlm_budget() if condition == "rlm" else None
    return {
        "schema_version": 3,
        "benchmark": "hover-wikipedia",
        "condition": condition,
        "models": {
            "solver": args.solver_model,
            "solver_api_base": solver_api_base,
            "reflection": args.reflection_model,
            "reflection_api_base": reflection_api_base,
        },
        "optimizer": {
            "max_metric_calls": args.max_metric_calls,
            "seed": args.seed,
            "seed_style": args.seed_style,
            "template_family": family,
            "component_kinds": _component_kinds(args.program),
            "reflection_level": reflection_level,
            "edit_tool_set": edit_tool_set,
            "proposer_backend": condition if operated else "stateless",
            "rlm_budget": asdict(rlm_budget) if rlm_budget is not None else None,
            "max_proposer_model_calls": (
                _rlm_max_model_calls(rlm_budget) if rlm_budget is not None else 8 if condition == "react_v2" else None
            ),
            "semantic_action_space": (
                semantic_action_catalog("prompt") if reflection_level == 2 or stateless_semantic else None
            ),
            "semantic_controller_policy": controller_policy_contract() if reflection_level == 2 else None,
            "stateless_action_menu": stateless_action_menu_contract(template) if stateless_semantic else None,
            "stateless_selector_policy": (
                stateless_selector_policy_contract("random" if condition == "random" else "verbalized")
                if stateless_semantic
                else None
            ),
        },
        "program": {
            "name": args.program,
            "retrieval_k": args.retrieval_k,
            "final_retrieval_k": args.final_retrieval_k,
            "parallel_workers": 24,
            "cache_evaluation": True,
        },
        "retrieval": {
            "endpoint": args.wikipedia_endpoint,
            "cache_path": args.wikipedia_cache,
            "timeout_sec": args.wikipedia_timeout,
        },
        "data": args.data_identity,
        "tag": args.tag,
    }


def _run_key(condition: str, args) -> str:
    family = resolve_template_family(args.template_family, args.solver_model)
    contract = build_run_contract(condition, args)
    return experiment_run_key(
        condition=condition,
        template_family=family,
        reflection_level=contract["optimizer"]["reflection_level"],
        edit_tool_set=contract["optimizer"]["edit_tool_set"] or "none",
        settings=contract,
    )


def seed_candidate(program: str, seed_style: str = "plain", template_family: str = "generic") -> dict:
    seed = dict(SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {
            component: structured_prompt(text, template_family, "system_prompt") for component, text in seed.items()
        }
    return seed


def run_program(
    candidate: dict,
    claim: str,
    program: str,
    model: str,
    api_base: str | None,
    retriever: WikipediaRetriever,
    retrieval_k: int,
    final_retrieval_k: int,
) -> tuple[str | None, list[WikipediaPassage]]:
    """Run a candidate and return generated queries plus retrieved pages."""
    if program == "1stage":
        return None, run_single_stage(
            candidate["retrieve"],
            claim,
            retriever,
            model=model,
            api_base=api_base,
            retrieval_k=final_retrieval_k,
        )
    return run_two_stage(
        candidate["summarize1"],
        candidate["create_query_hop2"],
        candidate["summarize2"],
        candidate["create_query_hop3"],
        claim,
        retriever,
        model=model,
        api_base=api_base,
        retrieval_k=retrieval_k,
        final_retrieval_k=final_retrieval_k,
    )


def make_evaluator(
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    program: str = "2stage",
    retrieval_k: int = 7,
    final_retrieval_k: int = 10,
):
    """Create an evaluator function closed over the solver model name."""

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        response, retrieved_docs = run_program(
            candidate,
            example["prompt"],
            program,
            solver_model,
            api_base,
            retriever,
            retrieval_k,
            final_retrieval_k,
        )
        score, feedback = hover_metric(retrieved_docs, example)
        recall = hover_recall(retrieved_docs, example)

        side_info: SideInfo = {
            "score": score,
            "claim": example["prompt"],
            "output": [passage.title for passage in retrieved_docs],
            "execution_feedback": feedback,
            "recall": recall,
        }
        if response is not None:
            side_info["stage1_queries"] = response
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate: dict,
    dataset: list[dict],
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    max_workers: int = 24,
    program: str = "2stage",
    retrieval_k: int = 7,
    final_retrieval_k: int = 10,
) -> dict[str, float]:
    """Evaluate a candidate, returning complete-retrieval rate and recall."""

    def score_one(example: dict) -> tuple[float, float]:
        _, retrieved_docs = run_program(
            candidate,
            example["prompt"],
            program,
            solver_model,
            api_base,
            retriever,
            retrieval_k,
            final_retrieval_k,
        )
        score, _ = hover_metric(retrieved_docs, example)
        rec = hover_recall(retrieved_docs, example)
        return score, rec

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    if not scores:
        return {"complete": 0.0, "recall": 0.0}
    f1s = [s for s, _ in scores]
    recs = [r for _, r in scores]
    return {"complete": sum(f1s) / len(f1s), "recall": sum(recs) / len(recs)}


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


def dump_candidates(result, run_dir: str, run_contract: dict) -> str:
    """Write all explored candidates (with lineage and scores) to candidates.json."""
    payload = {
        "run_contract": run_contract,
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


def build_config(condition: str, args, reflection_lm_kwargs: dict, run_dir: str | None = None):
    """Build the GEPAConfig for one condition. Returns (config, action_selector)."""
    resolved_family = resolve_template_family(args.template_family, args.solver_model)
    template = TEMPLATE_FAMILIES[resolved_family]["system_prompt"]
    action_space = build_stateless_action_menu(template)
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(args.reflection_model, **reflection_lm_kwargs),
        )

    reflection_strategy = None
    if condition == "react_v2":
        reflection_strategy, _ = build_react_v2_strategy(
            reflection_model=args.reflection_model,
            task_model=args.solver_model,
            lm_kwargs=reflection_lm_kwargs,
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            template_family=args.template_family,
            component_kinds=_component_kinds(args.program),
        )
    elif condition == "rlm":
        manifestor_kwargs = {**reflection_lm_kwargs, "temperature": 0}
        reflection_strategy = ThreeRoleReflectionLM(
            base_lm=LM(args.reflection_model, **reflection_lm_kwargs),
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            component_kinds=_component_kinds(args.program),
            template_family=resolved_family,
            manifestor_lm=LM(args.reflection_model, **manifestor_kwargs),
            proposer_model=args.reflection_model,
            proposer_backend="rlm",
            rlm_budget=_rlm_budget(),
        )

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir or condition_run_dir(condition, args.program, args.tag, _run_key(condition, args)),
            max_metric_calls=args.max_metric_calls,
            parallel=True,
            max_workers=24,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs or None,
            reflection_strategy=reflection_strategy,
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
    parser = argparse.ArgumentParser(description="HoVer evaluation for action-conditioned reflection")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=7051,
        help="Budget per condition (paper: 7051)",
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
    parser.add_argument("--solver-api-base", type=str, default=None, help="Base URL used only by the student/solver LM")
    parser.add_argument(
        "--reflection-api-base", type=str, default=None, help="Base URL used only by the reflection/proposer LM"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing official HoVer v1.1 JSON")
    parser.add_argument("--smoke", action="store_true", help="Use the explicit three-record smoke dataset")
    parser.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed")
    parser.add_argument(
        "--wikipedia-endpoint",
        type=str,
        default=DEFAULT_WIKIPEDIA_ENDPOINT,
        help="MediaWiki API endpoint used for live retrieval",
    )
    parser.add_argument("--wikipedia-cache", type=str, default=None, help="SQLite retrieval cache path")
    parser.add_argument("--wikipedia-timeout", type=float, default=20.0, help="MediaWiki request timeout in seconds")
    parser.add_argument("--retrieval-k", type=int, default=7, help="Pages retrieved in hops one and two (artifact: 7)")
    parser.add_argument("--final-retrieval-k", type=int, default=10, help="Pages retrieved in hop three (artifact: 10)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 150)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 300)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size (paper: 300)")
    parser.add_argument(
        "--program",
        type=str,
        default="2stage",
        choices=["2stage", "1stage"],
        help="Program structure: 2stage (two generated queries across three retrieval hops) or 1stage ablation",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="both",
        choices=["vanilla", "react_v2", "rlm", "random", "action", "all", "both"],
        help="Condition to run; rlm executes trusted model code in process and provides no security isolation",
    )
    parser.add_argument(
        "--seed-style",
        type=str,
        default="structured",
        choices=["plain", "structured"],
        help="Seed prompts: plain paper sentences or the template selected for the solver provider",
    )
    parser.add_argument(
        "--reflection-level",
        type=int,
        default=2,
        choices=[1, 2],
        help="Reflection level: 1 selects a section; 2 also selects and applies a semantic action",
    )
    parser.add_argument(
        "--edit-tool-set",
        choices=["minimal", "broad"],
        default="broad",
        help="Edit tools: insert/delete only, or insert/delete/replace/move; RLM requires broad",
    )
    parser.add_argument(
        "--template-family",
        choices=["auto", "generic", "openai", "anthropic", "google", "alibaba"],
        default="auto",
        help="Prompt template family; auto selects one from the student/solver model",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev2, 48h)")
    args = parser.parse_args()

    dataset_kwargs = {"seed": args.seed, "smoke": args.smoke}
    if args.data_dir is not None:
        dataset_kwargs["data_dir"] = args.data_dir
    trainset, valset, testset = load_hover_dataset(**dataset_kwargs)
    if args.train_limit is not None:
        trainset = trainset[: args.train_limit]
    if args.val_limit is not None:
        valset = valset[: args.val_limit]
    if args.test_limit is not None:
        testset = testset[: args.test_limit]
    if args.smoke:
        data_source = {"type": "built-in-smoke", "name": "hover-three-record-v1"}
    else:
        data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir is not None else DATA_DIR.resolve()
        data_path = data_dir / HOVER_TRAIN_FILE
        data_source = {"type": "json", "path": str(data_path), "sha256": file_sha256(data_path)}
    args.data_identity = benchmark_data_identity(
        source=data_source,
        trainset=trainset,
        valset=valset,
        testset=testset,
    )
    print(f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples ({args.program})")
    print(
        "  (official HoVer v1.1, exactly three unique gold documents)" if not args.smoke else "  (explicit smoke data)"
    )

    retriever = WikipediaClient(
        endpoint=args.wikipedia_endpoint,
        cache_path=args.wikipedia_cache,
        timeout=args.wikipedia_timeout,
    )
    args.wikipedia_cache = (
        str(retriever.cache_path.expanduser().resolve()) if retriever.cache_path is not None else None
    )
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    evaluator = make_evaluator(
        args.solver_model,
        retriever,
        api_base=solver_api_base,
        program=args.program,
        retrieval_k=args.retrieval_k,
        final_retrieval_k=args.final_retrieval_k,
    )

    reflection_lm_kwargs = {}
    if reflection_api_base is not None:
        reflection_lm_kwargs["api_base"] = reflection_api_base

    if args.condition == "all":
        conditions = ["vanilla", "react_v2", "random", "action"]
    elif args.condition == "both":
        conditions = ["vanilla", "react_v2"]
    else:
        conditions = [args.condition]

    resolved_family = resolve_template_family(args.template_family, args.solver_model)
    semantic_conditions = {"react_v2", "rlm", "random", "action"}.intersection(conditions)
    if semantic_conditions and args.seed_style != "structured":
        parser.error(f"--condition {', '.join(sorted(semantic_conditions))} requires --seed-style structured")
    if "rlm" in conditions and args.edit_tool_set != "broad":
        parser.error("--condition rlm requires --edit-tool-set broad")
    if "rlm" in conditions and args.reflection_level != 2:
        parser.error("--condition rlm requires --reflection-level 2")

    results = {}
    trackers: dict[str, ActionDiversityCallback] = {}
    for condition in conditions:
        run_contract = build_run_contract(condition, args)
        run_dir = condition_run_dir(condition, args.program, args.tag, _run_key(condition, args))
        ensure_wikipedia_run_contract(run_dir, run_contract)
        config, selector = build_config(condition, args, reflection_lm_kwargs, run_dir=run_dir)
        callbacks = None
        if condition in ("react_v2", "rlm", "random", "action"):
            trackers[condition] = ActionDiversityCallback()
            callbacks = [trackers[condition]]
        results[condition] = run_condition(
            f"{condition} GEPA ({args.program}, {args.seed_style} seeds)",
            seed_candidate(args.program, args.seed_style, resolved_family),
            trainset,
            valset,
            config,
            evaluator,
            callbacks=callbacks,
        )
        path = dump_candidates(results[condition], run_dir, run_contract)
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

    # Report: complete-retrieval rate + recall + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    baseline_scores = evaluate_on_set(
        seed_candidate(args.program, args.seed_style, resolved_family),
        testset,
        args.solver_model,
        retriever,
        api_base=solver_api_base,
        program=args.program,
        retrieval_k=args.retrieval_k,
        final_retrieval_k=args.final_retrieval_k,
    )
    print(
        f"Baseline (seed prompts) test: complete={baseline_scores['complete']:.3f} "
        f"recall={baseline_scores['recall']:.3f} on {len(testset)} examples\n"
    )

    for name, result in results.items():
        test_scores = evaluate_on_set(
            result.best_candidate,
            testset,
            args.solver_model,
            retriever,
            api_base=solver_api_base,
            program=args.program,
            retrieval_k=args.retrieval_k,
            final_retrieval_k=args.final_retrieval_k,
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score:           {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  complete retrieval:       {test_scores['complete']:.3f}")
        print(f"  test recall:              {test_scores['recall']:.3f}")
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
