"""HoVer evaluation: vanilla GEPA vs Controller/Manifestor/ReAct V2.

Replicates the GEPA artifact's HoVer experiment over official v1.1 claims with
exactly three supporting documents. The program performs three frozen Wiki-2017
retrieval hops and optimizes two summarizers and two query generators on 150 train
examples, with 300 val examples for Pareto selection and 300 test examples
held out for final scoring. The primary metric is complete gold-document
retrieval, with supporting-document recall reported separately. The default budget
of 7051 metric calls matches the paper.

Conditions:
    vanilla  - stock GEPA reflective mutation
    react_v2 - section/action Controller, Manifestor configured for the model provider, ReAct V2 proposer
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python -m examples.hover.main [--condition vanilla|react_v2|random|action|all|both]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
"""

import argparse
import itertools
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_request_overrides,
    resolve_experiment_model,
    validate_experiment_model_pair,
)
from examples.common.react_v2 import (
    benchmark_data_identity,
    build_react_v2_strategy,
    ensure_wikipedia_run_contract,
    experiment_run_key,
    resolve_template_family,
    structured_prompt,
)
from examples.common.wiki17_bm25 import DEFAULT_WIKI17_ROOT, GEPA_ARTIFACT_COMMIT, Wiki17BM25Retriever
from examples.common.wikipedia import WikipediaPassage, WikipediaRetriever
from examples.hover.utils import (
    DATA_DIR,
    HOVER_ELIGIBLE_COUNT,
    HOVER_HF_REVISION,
    HOVER_SOURCE_REVISION,
    HOVER_TRAIN_SHA256,
    HOVER_TRAIN_SIZE,
    artifact_component_records,
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
from gepa.strategies.action_space import (
    RandomActionSelector,
    VerbalizedActionSelector,
    stateless_selector_policy_contract,
)
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    SEMANTIC_ACTIONS,
    STATELESS_ACTION_MENU_VERSION,
    StatelessActionConstraint,
)

# GEPA artifact components: summarize1 -> create_query_hop2 -> summarize2 -> create_query_hop3.
SEED_CANDIDATE = {
    "summarize1": "Given the fields `claim`, `passages`, produce the fields `summary`.",
    "create_query_hop2": "Given the fields `claim`, `summary_1`, produce the fields `query`.",
    "summarize2": "Given the fields `claim`, `context`, `passages`, produce the fields `summary`.",
    "create_query_hop3": "Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.",
}

# Ablation seed: generate one query, then score the pages that query retrieves.
SEED_CANDIDATE_1STAGE = {
    "retrieve": "Write one focused Wikipedia search query for the evidence needed to verify the claim."
}

_CONDITION_DIR_NAMES = {
    "vanilla": "hover_vanilla",
    "react_v2": "hover_react_v2",
    "random": "hover_random_action",
    "action": "hover_verbalized_action",
}


def _component_kinds(program: str) -> dict[str, str]:
    """Identify every optimized HOVER instruction as a system message.

    Args:
        program: ``"1stage"`` or the three-hop program variant.

    Returns:
        Mapping from each seed component to ``system_prompt``.
    """
    seed = SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE
    return dict.fromkeys(seed, "system_prompt")


def condition_run_dir(condition: str, program: str, tag: str = "", run_key: str = "") -> str:
    """Build the output directory for one HOVER condition.

    Args:
        condition: Experiment condition naming the directory family.
        program: Program variant; ``"1stage"`` adds its identifying suffix.
        tag: Optional human-readable run suffix.
        run_key: Optional compatibility key for resumable state.

    Returns:
        Relative output directory for the condition.
    """
    suffix = "_1stage" if program == "1stage" else ""
    tag_suffix = f"_{tag}" if tag else ""
    key_suffix = f"_{run_key}" if run_key else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{suffix}{key_suffix}{tag_suffix}"


def build_run_contract(condition: str, args) -> dict:
    """Build the complete persisted configuration for one condition.

    Args:
        condition: Optimization condition being recorded.
        args: Parsed experiment arguments and computed data identity.

    Returns:
        JSON-serializable model, optimizer, retrieval, and data contract.
    """
    validate_experiment_model_pair(args.solver_model, args.reflection_model)
    family = resolve_template_family(args.template_family, args.solver_model)
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    solver_runtime_model = resolve_experiment_model(args.solver_model, args.api_profile)
    reflection_runtime_model = resolve_experiment_model(args.reflection_model, args.api_profile)
    operated = condition == "react_v2"
    reflection_level = args.reflection_level if operated else 0
    edit_tool_set = args.edit_tool_set if operated else None
    stateless_semantic = condition in ("random", "action")
    template = TEMPLATE_FAMILIES[family]["system_prompt"]
    rendered_seed = seed_candidate(args.program, args.seed_style, family)
    stateless_action_menu = None
    if stateless_semantic:
        stateless_action_menu = {
            "version": STATELESS_ACTION_MENU_VERSION,
            "semantic_action_catalog_version": SEMANTIC_ACTION_CATALOGS[template.kind]["version"],
            "kind": template.kind,
            "sections": list(template.sections),
            "choices": [
                {
                    "id": f"{spec.name}@{section}/{spec.edit_tool.value}",
                    "semantic_action": spec.name,
                    "operator": spec.edit_tool.value,
                    "target_section": section,
                }
                for section in template.sections
                for spec in SEMANTIC_ACTIONS
            ],
        }
    return {
        "schema_version": 5,
        "benchmark": "hover-train-wiki17",
        "reference_artifact_commit": GEPA_ARTIFACT_COMMIT,
        "condition": condition,
        "models": {
            "api_profile": args.api_profile,
            "solver": args.solver_model,
            "solver_runtime": solver_runtime_model,
            "solver_api_base": solver_api_base,
            "solver_decoding": experiment_decoding(args.solver_model),
            "solver_request_overrides": experiment_request_overrides(solver_runtime_model),
            "solver_num_retries": EXPERIMENT_NUM_RETRIES,
            "reflection": args.reflection_model,
            "reflection_runtime": reflection_runtime_model,
            "reflection_api_base": reflection_api_base,
            "reflection_decoding": experiment_decoding(args.reflection_model),
            "reflection_request_overrides": experiment_request_overrides(reflection_runtime_model),
            "reflection_num_retries": EXPERIMENT_NUM_RETRIES,
        },
        "optimizer": {
            "max_metric_calls": args.max_metric_calls,
            "seed": args.seed,
            "candidate_selection_strategy": "pareto",
            "frontier_type": "instance",
            "validation_evaluation": "full_eval",
            "acceptance_criterion": "strict_improvement",
            "batch_sampler": "epoch_shuffled",
            "reflection_minibatch_size": 3,
            "component_selector": "round_robin",
            "skip_perfect_score": True,
            "perfect_score": 1.0,
            "merge": None,
            "vanilla_reflection_prompt": "canonical_gepa" if condition == "vanilla" else None,
            "seed_style": args.seed_style,
            "artifact_seed_instructions": dict(SEED_CANDIDATE_1STAGE if args.program == "1stage" else SEED_CANDIDATE),
            "rendered_seed": rendered_seed,
            "template_family": family,
            "component_kinds": _component_kinds(args.program),
            "reflection_level": reflection_level,
            "edit_tool_set": edit_tool_set,
            "semantic_action_space": (
                deepcopy(SEMANTIC_ACTION_CATALOGS["prompt"]) if reflection_level == 2 or stateless_semantic else None
            ),
            "semantic_controller_policy": deepcopy(CONTROLLER_POLICY_CONTRACT) if reflection_level == 2 else None,
            "stateless_action_menu": stateless_action_menu,
            "stateless_selector_policy": (
                stateless_selector_policy_contract("random" if condition == "random" else "verbalized")
                if stateless_semantic
                else None
            ),
        },
        "program": {
            "name": args.program,
            "predictor_type": "dspy_chain_of_thought" if args.program == "2stage" else "direct",
            "predictor_adapter": "dspy_chat_adapter" if args.program == "2stage" else None,
            "retrieval_k": args.retrieval_k,
            "final_retrieval_k": args.final_retrieval_k,
            "parallel_workers": args.max_workers,
            "cache_evaluation": True,
            "primary_metric": "complete_gold_document_retrieval",
            "reported_supplemental_metric": "gold_document_recall",
            "task_inputs": ["claim"],
            "components": list(rendered_seed),
            "component_output_fields": (
                {
                    "summarize1": ["reasoning", "summary"],
                    "create_query_hop2": ["reasoning", "query"],
                    "summarize2": ["reasoning", "summary"],
                    "create_query_hop3": ["reasoning", "query"],
                }
                if args.program == "2stage"
                else {"retrieve": ["query"]}
            ),
        },
        "retrieval": Wiki17BM25Retriever(args.wiki17_dir).provenance(),
        "data": args.data_identity,
        "tag": args.tag,
    }


def _run_key(condition: str, args) -> str:
    """Fingerprint the material settings of one HOVER run.

    Args:
        condition: Optimization condition represented by the key.
        args: Parsed experiment arguments used to build the run contract.

    Returns:
        Stable compatibility key for the output directory.
    """
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
    """Build the initial prompt components for a program variant.

    Args:
        program: Single-stage or three-hop HOVER program.
        seed_style: ``"plain"`` or provider-structured seed rendering.
        template_family: Provider family used for structured seeds.

    Returns:
        Independent seed text for every optimized component.
    """
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
) -> tuple[str | None, list[WikipediaPassage], dict[str, object]]:
    """Run a candidate and retain its generated queries and component trace.

    Args:
        candidate: Current prompt component mapping.
        claim: HOVER claim whose evidence must be retrieved.
        program: Single-stage or three-hop execution path.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retriever: Wikipedia passage retriever.
        retrieval_k: Passages requested for intermediate hops.
        final_retrieval_k: Passages requested by the scoring retrieval.

    Returns:
        Intermediate query text when present, every passage retrieved in hop
        order, and the execution trace used for component-specific feedback.
    """
    if program == "1stage":
        retrieved = run_single_stage(
            candidate["retrieve"],
            claim,
            retriever,
            model=model,
            api_base=api_base,
            retrieval_k=final_retrieval_k,
        )
        return None, retrieved, {"retrieved_documents": retrieved}
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
    """Create a HOVER evaluator closed over solver and retrieval settings.

    Args:
        solver_model: Model used to execute candidate prompts.
        retriever: Wikipedia passage retriever.
        api_base: Optional solver API endpoint.
        program: Single-stage or three-hop execution path.
        retrieval_k: Passages requested for intermediate hops.
        final_retrieval_k: Passages requested by the scoring retrieval.

    Returns:
        Evaluator accepted by ``optimize_anything``.
    """

    def evaluate(candidate: dict, example: dict) -> tuple[float, SideInfo]:
        """Score one candidate on one claim and retain reflection evidence.

        Args:
            candidate: Prompt components being evaluated.
            example: Claim and supporting-document record.

        Returns:
            Complete-retrieval score and execution details used for reflection.
        """
        _response, retrieved_docs, trace = run_program(
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

        if program == "1stage":
            side_info: SideInfo = {
                "retrieve_specific_info": {
                    "Inputs": {"claim": example["claim"]},
                    "Generated Outputs": {"retrieved_docs": [passage.title for passage in retrieved_docs]},
                    "Feedback": feedback,
                }
            }
        else:
            records = artifact_component_records(example, trace, score)
            side_info = {f"{component}_specific_info": record for component, record in records.items()}
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate: dict,
    dataset: list[dict],
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    max_workers: int = 32,
    program: str = "2stage",
    retrieval_k: int = 7,
    final_retrieval_k: int = 10,
) -> dict[str, float]:
    """Evaluate a candidate, returning complete-retrieval rate and recall.

    Args:
        candidate: Prompt components being evaluated.
        dataset: Claims and supporting-document records.
        solver_model: Model used to execute candidate prompts.
        retriever: Wikipedia passage retriever.
        api_base: Optional solver API endpoint.
        max_workers: Maximum concurrent examples.
        program: Single-stage or three-hop execution path.
        retrieval_k: Passages requested for intermediate hops.
        final_retrieval_k: Passages requested by the scoring retrieval.

    Returns:
        Mean complete-retrieval and document-recall scores.
    """

    def score_one(example: dict) -> tuple[float, float]:
        """Run and score one HOVER example.

        Args:
            example: Claim and supporting-document record.

        Returns:
            Complete-retrieval and document-recall scores.
        """
        _, retrieved_docs, _trace = run_program(
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

    Args:
        candidates: Explored prompt-component mappings.

    Returns:
        Mean pairwise token-set distance and unique-text count per component.
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
    """Write explored candidates, lineage, scores, and contract to JSON.

    Args:
        result: Completed GEPA result containing candidate history.
        run_dir: Directory that receives ``candidates.json``.
        run_contract: Material configuration attached to the artifact.

    Returns:
        Path to the written candidate artifact.
    """
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
    """Persist aggregate and per-action diversity evidence.

    Args:
        tracker: Callback containing action outcomes and generated text.
        run_dir: Directory that receives ``action_summary.json``.
        selector: Optional verbalized selector whose history is included.

    Returns:
        Path to the written action summary.
    """
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
    """Build optimizer configuration and any stateless action selector.

    Args:
        condition: Optimization condition to configure.
        args: Parsed experiment arguments.
        reflection_lm_kwargs: Reflection-model client settings.
        run_dir: Optional explicit output directory.

    Returns:
        GEPA configuration and the condition's optional action selector.
    """
    resolved_family = resolve_template_family(args.template_family, args.solver_model)
    reflection_runtime_model = resolve_experiment_model(args.reflection_model, args.api_profile)
    template = TEMPLATE_FAMILIES[resolved_family]["system_prompt"]
    action_space = [
        StatelessActionConstraint(spec, section, template) for section in template.sections for spec in SEMANTIC_ACTIONS
    ]
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(reflection_runtime_model, **reflection_lm_kwargs),
        )

    reflection_strategy = None
    if condition == "react_v2":
        reflection_strategy, _ = build_react_v2_strategy(
            reflection_model=reflection_runtime_model,
            task_model=args.solver_model,
            proposer_model=args.reflection_model,
            lm_kwargs=reflection_lm_kwargs,
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            template_family=args.template_family,
            component_kinds=_component_kinds(args.program),
            rng=random.Random(args.seed),
        )

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir or condition_run_dir(condition, args.program, args.tag, _run_key(condition, args)),
            seed=args.seed,
            max_metric_calls=args.max_metric_calls,
            val_evaluation_policy="full_eval",
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            acceptance_criterion="strict_improvement",
            parallel=True,
            max_workers=args.max_workers,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            skip_perfect_score=True,
            perfect_score=1.0,
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=3,
            module_selector="round_robin",
            reflection_lm=reflection_runtime_model,
            reflection_lm_kwargs=reflection_lm_kwargs or None,
            reflection_strategy=reflection_strategy,
            reflection_prompt_template=InstructionProposalSignature.default_prompt_template,
            action_selector=action_selector,  # type: ignore[arg-type]
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
    """Run one optimization condition and return its GEPA result.

    Args:
        name: Human-readable condition label printed before execution.
        seed: Initial prompt-component mapping.
        trainset: Records used for candidate discovery.
        valset: Records used for validation.
        config: GEPA engine and reflection configuration.
        evaluator: Candidate evaluation callable.
        callbacks: Optional callbacks attached before optimization.

    Returns:
        Completed result from ``optimize_anything``.
    """
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
        config=config,  # type: ignore[arg-type]
    )

    return result


def main():
    """Parse CLI arguments and run the requested HOVER conditions."""
    parser = argparse.ArgumentParser(description="HoVer evaluation for action-conditioned reflection")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=7051,
        help="Budget per condition (paper: 7051)",
    )
    parser.add_argument(
        "--solver-model",
        type=str,
        default=QWEN3_8_27B_MODEL,
        help="Student model; use the same supported model as --reflection-model",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=QWEN3_8_27B_MODEL,
        help="Proposer model; use the same supported model as --solver-model",
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)"
    )
    parser.add_argument("--solver-api-base", type=str, default=None, help="Base URL used only by the student/solver LM")
    parser.add_argument(
        "--reflection-api-base", type=str, default=None, help="Base URL used only by the reflection/proposer LM"
    )
    parser.add_argument(
        "--api-profile",
        choices=["direct", "openrouter"],
        default="direct",
        help="API route for both model roles; OpenRouter uses fixed provider endpoints",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing official HoVer v1.1 JSON")
    parser.add_argument("--smoke", action="store_true", help="Use the explicit three-record smoke dataset")
    parser.add_argument("--seed", type=int, default=0, help="Experiment seed (artifact: 0)")
    parser.add_argument(
        "--wiki17-dir",
        type=Path,
        default=DEFAULT_WIKI17_ROOT,
        help="Prepared frozen Wiki-2017 corpus and BM25S index directory",
    )
    parser.add_argument("--retrieval-k", type=int, default=7, help="Pages retrieved in hops one and two (artifact: 7)")
    parser.add_argument("--final-retrieval-k", type=int, default=10, help="Pages retrieved in hop three (artifact: 10)")
    parser.add_argument("--max-workers", type=int, default=32, help="Parallel evaluator workers (artifact: 32)")
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
        choices=["vanilla", "react_v2", "random", "action", "all", "both"],
        help="Optimization condition to run",
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
        help="Edit tools: insert/delete only, or insert/delete/replace/move",
    )
    parser.add_argument(
        "--template-family",
        choices=["auto", "generic", "openai", "anthropic", "google", "alibaba"],
        default="auto",
        help="Prompt template family; auto selects one from the student/solver model",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev2, 48h)")
    args = parser.parse_args()
    try:
        validate_experiment_model_pair(args.solver_model, args.reflection_model)
    except ValueError as exc:
        parser.error(str(exc))

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
        data_source = {
            "type": "official-hover-v1.1",
            "path": str(data_dir),
            "reference_huggingface_script_revision": HOVER_HF_REVISION,
            "source_revision": HOVER_SOURCE_REVISION,
            "train_sha256": HOVER_TRAIN_SHA256,
            "train_size": HOVER_TRAIN_SIZE,
            "eligible_records": HOVER_ELIGIBLE_COUNT,
            "source_split": "train",
            "split_policy": "seed0-shuffle-then-ordered-40-40-20-and-independent-seed1-sampling",
            "experiment_seed": args.seed,
        }
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

    retriever = Wiki17BM25Retriever(args.wiki17_dir)
    print(
        f"  (retrieval: frozen Wiki-2017 BM25S {retriever.provenance()['bm25s_version']}, "
        f"k1={retriever.provenance()['k1']}, b={retriever.provenance()['b']})"
    )
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    solver_runtime_model = resolve_experiment_model(args.solver_model, args.api_profile)
    reflection_runtime_model = resolve_experiment_model(args.reflection_model, args.api_profile)
    evaluator = make_evaluator(
        solver_runtime_model,
        retriever,
        api_base=solver_api_base,
        program=args.program,
        retrieval_k=args.retrieval_k,
        final_retrieval_k=args.final_retrieval_k,
    )

    reflection_lm_kwargs = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(reflection_runtime_model),
        **experiment_request_overrides(reflection_runtime_model),
    }
    if reflection_api_base is not None:
        reflection_lm_kwargs["api_base"] = reflection_api_base

    if args.condition == "all":
        conditions = ["vanilla", "react_v2", "random", "action"]
    elif args.condition == "both":
        conditions = ["vanilla", "react_v2"]
    else:
        conditions = [args.condition]

    resolved_family = resolve_template_family(args.template_family, args.solver_model)
    semantic_conditions = {"react_v2", "random", "action"}.intersection(conditions)
    if semantic_conditions and args.seed_style != "structured":
        parser.error(f"--condition {', '.join(sorted(semantic_conditions))} requires --seed-style structured")
    results = {}
    trackers: dict[str, ActionDiversityCallback] = {}
    for condition in conditions:
        run_contract = build_run_contract(condition, args)
        run_dir = condition_run_dir(condition, args.program, args.tag, _run_key(condition, args))
        ensure_wikipedia_run_contract(run_dir, run_contract)
        config, selector = build_config(condition, args, reflection_lm_kwargs, run_dir=run_dir)
        callbacks = None
        if condition in ("react_v2", "random", "action"):
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

    for name, result in results.items():
        test_scores = evaluate_on_set(
            result.best_candidate,
            testset,
            solver_runtime_model,
            retriever,
            api_base=solver_api_base,
            max_workers=args.max_workers,
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
