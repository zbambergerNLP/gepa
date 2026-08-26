"""HotpotQA evaluation: vanilla GEPA vs Controller/Manifestor/ReAct V2.

Replicates the GEPA artifact's HotpotQA setup with ``hotpot_qa/fullwiki``,
two live Wikipedia retrieval hops, four optimized program components, and
official exact-match/token-F1 metrics. The default
budget of 6871 metric calls matches the paper (MIPROv2-Heavy's invocation count
for HotpotQA). Scaled and smoke budgets are also supported.

Conditions:
    vanilla  - stock GEPA reflective mutation
    react_v2 - section/action Controller, Manifestor configured for the model provider, ReAct V2 proposer
    random   - action-conditioned reflection, actions picked uniformly at random
    action   - action-conditioned reflection with verbalized sampling

Usage:
    uv run python -m examples.hotpotqa.main [--condition vanilla|react_v2|both]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
    # Smoke (20 ex, 14/3/3):
    uv run python -m examples.hotpotqa.main --data-path examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl --max-metric-calls 200 --condition both
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
from examples.common.wikipedia import DEFAULT_WIKIPEDIA_ENDPOINT, WikipediaClient, WikipediaRetriever
from examples.hotpotqa.utils import (
    f1_score,
    hotpotqa_metric,
    load_hotpotqa_dataset,
    normalize_answer,
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
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    SEMANTIC_ACTIONS,
    STATELESS_ACTION_MENU_VERSION,
    StatelessActionConstraint,
)

# GEPA artifact components: summarize1 -> create_query_hop2 -> summarize2 -> final_answer.
SEED_CANDIDATE = {
    "summarize1": "Summarize the first-hop passages, preserving facts and connections needed for follow-up retrieval.",
    "create_query_hop2": "Generate a concise second-hop Wikipedia query from the question and first-hop summary.",
    "summarize2": "Synthesize the first-hop summary and second-hop passages into the evidence needed to answer the question.",
    "final_answer": "Answer the question exactly and concisely using the two retrieval-hop summaries.",
}

# 1-stage ablation (single prompt)
SEED_CANDIDATE_1STAGE = {
    "answer_question": (
        "Answer the question from the retrieved passages. Ignore irrelevant passages and "
        "combine evidence across pages. Return a concise answer."
    ),
}

_INITIAL_PROMPT = SEED_CANDIDATE_1STAGE["answer_question"]

_CONDITION_DIR_NAMES = {
    "vanilla": "hotpotqa_vanilla",
    "react_v2": "hotpotqa_react_v2",
    "random": "hotpotqa_random_action",
    "action": "hotpotqa_verbalized_action",
}


def _component_kinds(program: str) -> dict[str, str]:
    """Identify every optimized HotPotQA instruction as a system message.

    Args:
        program: ``"1stage"`` or the two-stage program variant.

    Returns:
        Mapping from each seed component to ``system_prompt``.
    """
    seed = SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE
    return dict.fromkeys(seed, "system_prompt")


def condition_run_dir(condition: str, program: str, tag: str = "", run_key: str = "") -> str:
    """Build the output directory for one HotPotQA condition.

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
    family = resolve_template_family(args.template_family, args.solver_model)
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    reflection_level = args.reflection_level if condition == "react_v2" else 0
    edit_tool_set = args.edit_tool_set if condition == "react_v2" else None
    stateless_semantic = condition in ("random", "action")
    template = TEMPLATE_FAMILIES[family]["system_prompt"]
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
        "schema_version": 3,
        "benchmark": "hotpotqa-wikipedia",
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
            "semantic_action_space": (
                deepcopy(SEMANTIC_ACTION_CATALOGS["prompt"])
                if reflection_level == 2 or stateless_semantic
                else None
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
            "retrieval_k": args.retrieval_k,
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
    """Fingerprint the material settings of one HotPotQA run.

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


def seed_candidate(
    program: str = "2stage",
    seed_style: str = "plain",
    template_family: str = "generic",
) -> dict:
    """Build the initial prompt components for a program variant.

    Args:
        program: Single-stage or two-stage HotPotQA program.
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
    question: str,
    program: str,
    model: str,
    api_base: str | None,
    retriever: WikipediaRetriever,
    retrieval_k: int,
) -> tuple[str | None, str]:
    """Run a candidate and return its second-hop query and final answer.

    Args:
        candidate: Current prompt component mapping, or a legacy prompt string.
        question: HotPotQA question to answer.
        program: Single-stage or two-stage execution path.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retriever: Wikipedia passage retriever.
        retrieval_k: Passages requested for each retrieval hop.

    Returns:
        Generated second-hop query when present and the final answer text.
    """
    if isinstance(candidate, str):
        return None, run_single_stage(
            candidate, question, retriever, model=model, api_base=api_base, retrieval_k=retrieval_k
        )
    if program == "1stage":
        prompt = candidate.get("answer_question") or next(iter(candidate.values()))
        return None, run_single_stage(
            prompt, question, retriever, model=model, api_base=api_base, retrieval_k=retrieval_k
        )
    query, answer = run_two_stage(
        candidate.get("summarize1", ""),
        candidate.get("create_query_hop2", ""),
        candidate.get("summarize2", ""),
        candidate.get("final_answer", ""),
        question,
        retriever,
        model=model,
        api_base=api_base,
        retrieval_k=retrieval_k,
    )
    return query, answer


def make_evaluator(
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    program: str = "2stage",
    retrieval_k: int = 7,
):
    """Create a HotPotQA evaluator closed over solver and retrieval settings.

    Args:
        solver_model: Model used to execute candidate prompts.
        retriever: Wikipedia passage retriever.
        api_base: Optional solver API endpoint.
        program: Single-stage or two-stage execution path.
        retrieval_k: Passages requested for each retrieval hop.

    Returns:
        Evaluator accepted by ``optimize_anything``.
    """

    def evaluate(candidate, example: dict) -> tuple[float, SideInfo]:
        """Score one candidate on one question and retain reflection evidence.

        Args:
            candidate: Prompt components being evaluated.
            example: Question and reference answer record.

        Returns:
            Primary score and execution details used for reflection.
        """
        stage1, prediction = run_program(
            candidate,
            example["question"],
            program,
            solver_model,
            api_base,
            retriever,
            retrieval_k,
        )

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
        side_info["em"] = float(normalize_answer(prediction) == normalize_answer(example["answer"]))
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate,
    dataset: list[dict],
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    max_workers: int = 24,
    program: str = "2stage",
    retrieval_k: int = 7,
) -> tuple[float, float]:
    """Evaluate a candidate on a dataset, returning mean exact match and F1.

    Args:
        candidate: Prompt components being evaluated.
        dataset: Question and answer records.
        solver_model: Model used to execute candidate prompts.
        retriever: Wikipedia passage retriever.
        api_base: Optional solver API endpoint.
        max_workers: Maximum concurrent examples.
        program: Single-stage or two-stage execution path.
        retrieval_k: Passages requested for each retrieval hop.

    Returns:
        Mean exact-match and token-F1 scores, or zeros for an empty dataset.
    """

    def score_one(example: dict) -> tuple[float, float]:
        """Run and score one HotPotQA example.

        Args:
            example: Question and reference answer record.

        Returns:
            Exact-match and token-F1 scores.
        """
        _, pred = run_program(
            candidate,
            example["question"],
            program,
            solver_model,
            api_base,
            retriever,
            retrieval_k,
        )
        exact_match = float(normalize_answer(pred) == normalize_answer(example["answer"]))
        return exact_match, f1_score(pred, example["answer"])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    if not scores:
        return 0.0, 0.0
    mean_em = sum(s[0] for s in scores) / len(scores)
    mean_f1 = sum(s[1] for s in scores) / len(scores)
    return mean_em, mean_f1


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
    """Measure textual diversity of explored candidates per component.

    Args:
        candidates: Explored prompt mappings. Legacy string values are
            normalized when encountered.

    Returns:
        Mean pairwise token-set distance and unique-text count per component.
    """
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
    template = TEMPLATE_FAMILIES[resolved_family]["system_prompt"]
    action_space = [
        StatelessActionConstraint(spec, section, template)
        for section in template.sections
        for spec in SEMANTIC_ACTIONS
    ]
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(args.reflection_model, **(reflection_lm_kwargs or {})),
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
    seed: dict | str,
    trainset: list[dict],
    valset: list[dict],
    config: GEPAConfig,
    evaluator,
    callbacks: list | None = None,
):
    """Run one optimization condition and return its GEPA result.

    Args:
        name: Human-readable condition label printed before execution.
        seed: Initial prompt or prompt-component mapping.
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
        config=config,
    )

    return result


def main():
    """Parse CLI arguments and run the requested HotPotQA conditions."""
    parser = argparse.ArgumentParser(description="HotpotQA evaluation for action-conditioned reflection")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HotpotQA JSONL sample (smoke, 14/3/3)")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=6871,
        help="Budget per condition (paper: 6871, smoke: 200, scaled Wave B: 15000)",
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
    parser.add_argument(
        "--wikipedia-endpoint",
        type=str,
        default=DEFAULT_WIKIPEDIA_ENDPOINT,
        help="MediaWiki API endpoint used for live retrieval",
    )
    parser.add_argument("--wikipedia-cache", type=str, default=None, help="SQLite retrieval cache path")
    parser.add_argument("--wikipedia-timeout", type=float, default=20.0, help="MediaWiki request timeout in seconds")
    parser.add_argument("--retrieval-k", type=int, default=7, help="Wikipedia pages retrieved per hop (artifact: 7)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 150)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 300)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size (paper: 300)")
    parser.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed")
    parser.add_argument(
        "--program",
        type=str,
        default="2stage",
        choices=["2stage", "1stage"],
        help="Program structure: 2stage (two retrieval hops, four artifact components) or 1stage ablation",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="both",
        choices=["vanilla", "react_v2", "random", "action", "all", "both"],
        help="Which condition(s) to run",
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
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev2, 6871)")
    args = parser.parse_args()

    trainset, valset, testset = load_hotpotqa_dataset(
        data_path=args.data_path,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    if args.data_path is None:
        data_source = {"type": "huggingface", "dataset": "hotpot_qa", "config": "fullwiki"}
    else:
        data_path = Path(args.data_path).expanduser().resolve()
        data_source = {"type": "jsonl", "path": str(data_path), "sha256": file_sha256(data_path)}
    args.data_identity = benchmark_data_identity(
        source=data_source,
        trainset=trainset,
        valset=valset,
        testset=testset,
    )

    print(
        f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples ({args.program}, {args.seed_style})"
    )
    if args.data_path:
        print(f"  (explicit smoke data from {args.data_path}; bundled passages ignored)")
    else:
        print("  (from hotpot_qa/fullwiki; bundled contexts ignored; paper Table 1: 150/300/300)")

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
        seed = seed_candidate(args.program, args.seed_style, resolved_family)
        results[condition] = run_condition(
            f"{condition} GEPA ({args.program}, {args.seed_style} seeds)",
            seed,
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
        cand = result.best_candidate
        if isinstance(cand, str):
            print(cand)
        else:
            for component, text in cand.items():
                print(f"\n[{name}] {component}:\n{text}")

    # Report: test EM/F1 + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    baseline = seed_candidate(args.program, args.seed_style, resolved_family)
    baseline_em, baseline_f1 = evaluate_on_set(
        baseline,
        testset,
        args.solver_model,
        retriever,
        api_base=solver_api_base,
        program=args.program,
        retrieval_k=args.retrieval_k,
    )
    print(f"Baseline (seed prompts) test EM: {baseline_em:.2%} F1: {baseline_f1:.2%} on {len(testset)} examples\n")

    for name, result in results.items():
        test_em, test_f1 = evaluate_on_set(
            result.best_candidate,
            testset,
            args.solver_model,
            retriever,
            api_base=solver_api_base,
            program=args.program,
            retrieval_k=args.retrieval_k,
        )
        diversity = prompt_diversity(result.candidates)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score (EM):      {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test EM:                  {test_em:.2%}")
        print(f"  test F1:                  {test_f1:.2%}")
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
