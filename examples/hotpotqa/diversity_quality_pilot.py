"""Compare persistent sibling policies on fixed training-only editing opportunities."""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from examples.common.pilot_checks import atomic_json, digest, require_contract
from examples.common.react_v2 import benchmark_data_identity, resolve_template_family
from examples.common.wiki17_bm25 import Wiki17BM25Retriever
from examples.hotpotqa.generalization_pilot import evaluate_records, paired_outcomes, source_identity
from examples.hotpotqa.main import (
    _is_task_output_parse_error,
    _validate_scientific_data_identity,
    _verify_scientific_retriever_integrity,
    build_config,
    build_hotpotqa_task_lm,
    build_parser,
    build_run_contract,
    run_program,
    seed_candidate,
)
from examples.hotpotqa.pilot import observed_kwargs
from examples.hotpotqa.utils import (
    HOTPOTQA_HF_REVISION,
    artifact_component_records,
    load_hotpotqa_dataset,
    normalize_answer,
)
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM

COMPONENTS = ("summarize1", "create_query_hop2", "summarize2", "final_answer")
ARMS = {
    "sibling_diverse": ("sibling_diverse", "generative"),
    "diversity_quality_generative": ("diversity_quality", "generative"),
    "diversity_quality_jev": ("diversity_quality", "jev"),
}
PROTOCOL = {
    "identity": "diversity-quality-paired-training-pilot-v1",
    "components": list(COMPONENTS) * 3,
    "proposal_batches": [list(range(3 * i, 3 * i + 3)) for i in range(12)],
    "transfer_train_indices": list(range(36, 48)),
    "arms": ARMS,
    "controller_selection": "verbalized_in_all_arms",
    "parent": "original node 0 for every opportunity; independent persistent population per arm",
    "arm_order": "rotate by opportunity index",
    "seed": 0,
    "recovery": "finite native planner recovery within the selected component only",
    "admission": "changed candidate with strictly higher mean EM on its actual three training examples",
    "accepted_pairs": "consume only on actual admitted child; retain across opportunities",
    "skip": "skip all arms when parent answers all three training examples correctly",
    "transfer_order": "evaluate original and candidate transfer only after every generation and admission decision",
    "primary_measure": "paired transfer exact-match gain; retain exhaustion, ties and losses",
    "secondary_measures": ["training gain", "accepted siblings", "redundancy", "generation errors", "latency", "usage"],
    "unique_training_examples": 48,
    "maximum_logical_evaluations": 588,
    "metric": "normalized_exact_match_only",
    "supplementary_metrics": [],
    "missing_component_trace": "retain actual question, whole-program score and raw feedback with an explicit availability flag",
    "validation_or_test_evaluation": False,
    "interpretation": "small paired diagnostic; training-admitted pilot children are not full-validation candidates",
}


@dataclass
class PilotPopulation:
    """Store real training-admitted prompt nodes without inventing validation scores."""

    nodes: list[dict[str, Any]]

    @property
    def program_candidates(self) -> list[dict[str, str]]:
        """Expose only the parent/child texts needed by the native acceptance invariant."""
        return [node["candidate"] for node in self.nodes]


def make_evaluator(solver_model, retriever, api_base, *, solver_lm_kwargs, reflection_diagnostics):
    """Use the unchanged two-stage task pipeline with only normalized exact match."""
    task_lm = build_hotpotqa_task_lm(solver_model, api_base, solver_lm_kwargs)

    def evaluate(candidate, example):
        try:
            _query, prediction, trace = run_program(
                candidate,
                example["question"],
                "2stage",
                solver_model,
                api_base,
                retriever,
                7,
                task_lm,
                solver_lm_kwargs,
            )
        except ValueError as exc:
            if not _is_task_output_parse_error(exc):
                raise
            return 0.0, {
                "evaluation_error": {
                    "type": "task_output_parse_error",
                    "message": "Task-model output omitted DSPy's required structured fields; this example scored 0.",
                }
            }
        score = float(normalize_answer(prediction) == normalize_answer(example["answer"]))
        records = artifact_component_records(example, trace, score, include_diagnostics=reflection_diagnostics)
        return score, {f"{component}_specific_info": record for component, record in records.items()}

    return evaluate


def _save(path: Path, value: dict[str, Any]) -> None:
    """Seal a pilot step and its strategy state together before advancing."""
    atomic_json(path, {"record": value, "sha256": digest(value)})


def _load(path: Path) -> dict[str, Any]:
    """Reject modified checkpoint or proposal evidence before reuse."""
    saved = json.loads(path.read_text())
    if saved["sha256"] != digest(saved["record"]):
        raise ValueError(f"Pilot step hash mismatch: {path}")
    return saved["record"]


def _snapshot(strategy: ThreeRoleReflectionLM, population: PilotPopulation) -> dict[str, Any]:
    """Capture policy memory, RNG, role cursors and the actual node ledger."""
    return {"strategy_state": strategy.get_batch_retry_state(), "nodes": deepcopy(population.nodes)}


def _restore(strategy: ThreeRoleReflectionLM, population: PilotPopulation, snapshot: dict[str, Any]) -> None:
    """Restore JSON-encoded RNG tuples while preserving the other state schemas."""
    state = deepcopy(snapshot["strategy_state"])

    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value

    state["rng_state"] = tuples(state["rng_state"])
    strategy.set_batch_retry_state(state)
    population.nodes = deepcopy(snapshot["nodes"])


def build_strategy(settings: argparse.Namespace, directory: Path, arm: str) -> ThreeRoleReflectionLM:
    """Keep the production role clients/decoding while changing only proposal policy."""
    config, _ = build_config(
        "react_v2",
        settings,
        observed_kwargs(settings.reflection_model, settings.reflection_api_base, directory, "optimizer"),
        str(directory),
    )
    base = config.reflection.reflection_strategy
    if not isinstance(base, ThreeRoleReflectionLM):
        raise TypeError("Pilot requires the actual three-role reflection strategy")
    policy, backend = ARMS[arm]
    return ThreeRoleReflectionLM(
        base_lm=base.base_lm,
        controller_lm=base.controller_lm,
        manifestor_lm=base.manifestor_lm,
        level=base.level,
        edit_tool_set=base.edit_tool_set,
        templates=base.templates,
        component_kinds=base.component_kinds,
        template_family=base.template_family,
        controller_selection="verbalized",
        editor_mode="single_call",
        proposer_model=base.proposer_model,
        rng=base.rng,
        k=base.k,
        tau=base.tau,
        text_limits=base.text_limits,
        base_lm_run_identity=base.base_lm_run_identity,
        controller_lm_run_identity=base.controller_lm_run_identity,
        manifestor_lm_run_identity=base.manifestor_lm_run_identity,
        proposal_policy=policy,
        novelty_backend=backend,
    )


def _proposal_step(
    directory: Path,
    strategy: ThreeRoleReflectionLM,
    population: PilotPopulation,
    parent: dict[str, str],
    component: str,
    opportunity: int,
    examples: list[dict],
    original: list[dict],
    evaluate,
    workers: int,
) -> dict[str, Any]:
    """Generate, train-score and admit one proposal with recoverable policy state."""
    ids = [example["id"] for example in examples]
    evidence = []
    for example, record in zip(examples, original, strict=True):
        row = deepcopy(record["feedback"].get(f"{component}_specific_info"))
        if not row:
            row = {
                "Inputs": {"question": example.get("question")},
                "Feedback": deepcopy(record["feedback"]),
                "whole_program_score": record["score"],
                "component_trace_available": False,
            }
        evidence.append(row)
    request = {
        "parent": parent,
        "component": component,
        "opportunity": opportunity,
        "training_ids": ids,
        "reflection_records": {component: evidence},
    }
    require_contract(directory, request)
    pre = _snapshot(strategy, population)
    pre_path = directory / "pre.json"
    if pre_path.exists():
        if digest(_load(pre_path)) != digest(pre):
            raise ValueError("Pilot pre-step state does not match the preceding completed opportunity")
    else:
        _save(pre_path, pre)
    decision_path = directory / "decision.json"
    if decision_path.exists():
        decision = _load(decision_path)
        _restore(strategy, population, decision["after"])
        return decision["comparison"]

    generated_path = directory / "generated.json"
    if generated_path.exists():
        generated = _load(generated_path)
        _restore(strategy, population, generated["after"])
        proposal = generated["proposal"]
    else:
        started = time.monotonic()
        result, _ = strategy.reflect(
            parent,
            {component: evidence},
            [component],
            metadata={
                "candidate_idx": 0,
                "optimizer_iteration": opportunity + 1,
                "minibatch_ids": ids,
                "proposal_slot": 0,
            },
        )
        candidate = {**parent, **result.new_texts}
        if any(candidate[name] != parent[name] for name in parent if name != component):
            raise ValueError("Pilot policy modified an unselected component")
        proposal = {
            "candidate": candidate,
            "changed": candidate != parent,
            "seconds": time.monotonic() - started,
            "metadata": result.metadata,
            "prompts": result.prompts,
            "raw_lm_outputs": result.raw_lm_outputs,
        }
        _save(generated_path, {"proposal": proposal, "after": _snapshot(strategy, population)})
    revised = (
        evaluate_records(directory / "training", proposal["candidate"], examples, evaluate, workers, compute_f1=False)
        if proposal["changed"]
        else original
    )
    training = paired_outcomes(original, revised)
    accepted = proposal["changed"] and training["delta"] > 0
    planner = strategy.sibling_planner
    if planner is None:
        raise TypeError("Pilot arm lacks its native sibling planner")
    if proposal["changed"]:
        child_proposal = CandidateProposal(
            candidate=proposal["candidate"],
            parent_program_ids=[0],
            subsample_indices=ids,
            subsample_scores_before=[row["score"] for row in original],
            subsample_scores_after=[row["score"] for row in revised],
            metadata=deepcopy(proposal["metadata"]),
        )
        planner.observe_evaluation(child_proposal, parent)
        proposal["metadata"] = child_proposal.metadata
        if accepted:
            # Native acceptance only needs real node text; this pilot has no validation population.
            state = cast(GEPAState, population)
            planner.validate_children([child_proposal], state)
            child_id = len(population.nodes)
            population.nodes.append(
                {
                    "id": child_id,
                    "parent": 0,
                    "candidate": proposal["candidate"],
                    "component": component,
                    "opportunity": opportunity,
                    "training_ids": ids,
                    "training": training,
                    "metadata": child_proposal.metadata,
                }
            )
            planner.accept_child(child_proposal, child_id, state)
    row = {
        "opportunity": opportunity,
        "component": component,
        "changed": proposal["changed"],
        "accepted": accepted,
        "training": training,
        "proposal_seconds": proposal["seconds"],
        "generation_outcome": proposal["metadata"].get("generation_outcome"),
        "candidate": proposal["candidate"],
        "proposal_metadata": proposal["metadata"],
        "missing_component_trace_count": sum(row.get("component_trace_available") is False for row in evidence),
    }
    _save(decision_path, {"comparison": row, "after": _snapshot(strategy, population)})
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run matched training proposals before revealing any transfer outcomes."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY must be available before the pilot starts")
    settings = build_parser().parse_args(
        [
            "--solver-model",
            args.model,
            "--reflection-model",
            args.reflection_model,
            "--solver-api-base",
            args.api_base,
            "--reflection-api-base",
            args.reflection_api_base,
            "--wiki17-dir",
            str(args.wiki17_dir),
            "--max-workers",
            str(args.workers),
            "--condition",
            "react_v2",
            "--text-limits",
            "null",
            "--editor-mode",
            "single_call",
        ]
    )
    train, val, test = load_hotpotqa_dataset(seed=0)
    settings.data_identity = benchmark_data_identity(
        source={
            "type": "huggingface",
            "dataset": "hotpot_qa",
            "config": "fullwiki",
            "revision": HOTPOTQA_HF_REVISION,
            "source_split": "train",
            "split_policy": "ordered-40-40-20-then-independent-seed1-sampling",
            "experiment_seed": 0,
        },
        trainset=train,
        valset=val,
        testset=test,
    )
    settings.enforce_scientific_contract = True
    _validate_scientific_data_identity(settings)
    settings.enforce_scientific_contract = False
    del val, test
    if len(train) < 48 or len({row["id"] for row in train[:48]}) != 48:
        raise ValueError("Pilot requires 48 distinct ordered training examples")
    retriever = Wiki17BM25Retriever(args.wiki17_dir)
    _verify_scientific_retriever_integrity(retriever)
    settings.retrieval_provenance = retriever.provenance()
    parent = seed_candidate("2stage", "structured", resolve_template_family("auto", args.model))
    runtime = build_run_contract("react_v2", settings)
    runtime.setdefault("program", {})["reported_supplemental_metric"] = None
    if "optimizer" in runtime:
        runtime["configured_search_not_executed"] = runtime.pop("optimizer")
    runtime["pilot_evaluation"] = {"metric": "normalized_exact_match", "compute_f1": False}
    require_contract(
        args.output_dir,
        {
            "protocol": PROTOCOL,
            "source": source_identity(Path(__file__).resolve().parents[2]),
            "runtime": runtime,
            "parent": parent,
            "training_examples": train[:48],
        },
    )
    strategies = {}
    populations = {}
    for arm in ARMS:
        arm_dir = args.output_dir / arm
        strategy = build_strategy(settings, arm_dir / "runtime", arm)
        require_contract(arm_dir / "identity", {"protocol": PROTOCOL, "strategy": strategy.run_contract(parent)})
        if strategy.sibling_planner is None:
            raise TypeError("Pilot arm lacks a sibling planner")
        strategy.sibling_planner.bind_run_dir(str(arm_dir / "runtime"))
        strategies[arm] = strategy
        populations[arm] = PilotPopulation([{"id": 0, "kind": "original", "candidate": deepcopy(parent)}])
    evaluate = make_evaluator(
        args.model,
        retriever,
        args.api_base,
        solver_lm_kwargs=observed_kwargs(args.model, args.api_base, args.output_dir, "solver"),
        reflection_diagnostics=True,
    )
    original = evaluate_records(
        args.output_dir / "parent-training", parent, train[:36], evaluate, args.workers, compute_f1=False
    )
    comparisons = []
    for index, (component, indices) in enumerate(
        zip(PROTOCOL["components"], PROTOCOL["proposal_batches"], strict=True)
    ):
        before = [original[i] for i in indices]
        if all(row["score"] == 1 for row in before):
            for arm in ARMS:
                comparisons.append(
                    {"opportunity": index, "component": component, "arm": arm, "perfect_batch_skip": True}
                )
            continue
        arms = list(ARMS)
        arms = arms[index % len(arms) :] + arms[: index % len(arms)]
        for arm in arms:
            row = _proposal_step(
                args.output_dir / arm / f"opportunity-{index}",
                strategies[arm],
                populations[arm],
                parent,
                component,
                index,
                [train[i] for i in indices],
                before,
                evaluate,
                args.workers,
            )
            comparisons.append({"arm": arm, **row})
    _save(
        args.output_dir / "generation-complete.json",
        {
            "comparisons": comparisons,
            "final_states": {arm: _snapshot(strategies[arm], populations[arm]) for arm in ARMS},
        },
    )
    transfer_examples = [train[i] for i in PROTOCOL["transfer_train_indices"]]
    original_transfer = evaluate_records(
        args.output_dir / "parent-transfer",
        parent,
        transfer_examples,
        evaluate,
        args.workers,
        compute_f1=False,
    )
    for row in comparisons:
        if row.get("perfect_batch_skip"):
            continue
        directory = args.output_dir / row["arm"] / f"opportunity-{row['opportunity']}"
        revised_transfer = (
            evaluate_records(
                directory / "transfer", row["candidate"], transfer_examples, evaluate, args.workers, compute_f1=False
            )
            if row["changed"]
            else original_transfer
        )
        row["transfer"] = paired_outcomes(original_transfer, revised_transfer)
        _save(directory / "comparison.json", row)
    summary = {
        "protocol": PROTOCOL,
        "comparisons": comparisons,
        "populations": {arm: populations[arm].nodes for arm in ARMS},
        "logical_evaluations": 48 + 15 * sum(bool(row.get("changed")) for row in comparisons),
        "opportunities_with_missing_component_trace": sorted(
            {row["opportunity"] for row in comparisons if row.get("missing_component_trace_count", 0)}
        ),
        "execution_completed": True,
        "completed_ablation": False,
        "heldout_complete": False,
        "review_required": True,
    }
    _save(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Run the authorized diagnostic against separately allocated model endpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "api-base", "reflection-model", "reflection-api-base"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wiki17-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
