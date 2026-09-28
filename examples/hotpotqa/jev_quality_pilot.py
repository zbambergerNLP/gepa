"""Compare Jev and verbalized Controllers on fixed training-only proposal batches."""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

from examples.common.pilot_checks import atomic_json, require_contract
from examples.common.react_v2 import benchmark_data_identity, resolve_template_family
from examples.common.wiki17_bm25 import Wiki17BM25Retriever
from examples.hotpotqa.generalization_pilot import evaluate_records, paired_outcomes, source_identity
from examples.hotpotqa.main import (
    _validate_scientific_data_identity,
    _verify_scientific_retriever_integrity,
    build_config,
    build_parser,
    build_run_contract,
    make_evaluator,
    seed_candidate,
)
from examples.hotpotqa.pilot import observed_kwargs
from examples.hotpotqa.utils import HOTPOTQA_HF_REVISION, load_hotpotqa_dataset

COMPONENTS = ("summarize1", "create_query_hop2", "summarize2", "final_answer")
PROTOCOL = {
    "identity": "jev-paired-proposal-quality-v1",
    "controller_policy": "jev_joint_action_section_v3",
    "proposal_batches": [list(range(3 * i, 3 * i + 3)) for i in range(8)],
    "components": list(COMPONENTS) * 2,
    "transfer_train_indices": list(range(24, 36)),
    "arms": ["verbalized", "jev"],
    "parent": "original prompts for every opportunity",
    "selection": "one proposal per arm; no retries for semantic failure, no-op, tie or loss",
    "skip": "skip both arms if all three parent training answers are correct",
    "primary_measure": "paired transfer exact-match change from parent, including no-ops",
    "secondary_measures": ["training gain", "edit validity", "latency", "tokens", "provider failures"],
    "interpretation": "small paired training diagnostic; not held-out performance or an independent search",
    "validation_or_test_evaluation": False,
}


def run(args: argparse.Namespace) -> dict:
    """Generate matched edits and persist every evaluation and proposal outcome."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY must be available before starting the pilot")
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
    retriever = Wiki17BM25Retriever(args.wiki17_dir)
    _verify_scientific_retriever_integrity(retriever)
    settings.retrieval_provenance = retriever.provenance()
    parent = seed_candidate("2stage", "structured", resolve_template_family("auto", args.model))
    contract = {
        "protocol": PROTOCOL,
        "source": source_identity(Path(__file__).resolve().parents[2]),
        "runtime": build_run_contract("react_v2", settings),
        "parent": parent,
        "training_examples": train[:36],
    }
    require_contract(args.output_dir, contract)
    evaluate = make_evaluator(
        args.model,
        retriever,
        args.api_base,
        solver_lm_kwargs=observed_kwargs(args.model, args.api_base, args.output_dir, "solver"),
        reflection_diagnostics=True,
    )
    original = evaluate_records(
        args.output_dir / "parent", parent, train[:36], evaluate, args.workers, compute_f1=False
    )
    transfer = PROTOCOL["transfer_train_indices"]
    comparisons = []
    for index, (component, indices) in enumerate(
        zip(PROTOCOL["components"], PROTOCOL["proposal_batches"], strict=True)
    ):
        if all(original[i]["score"] == 1 for i in indices):
            comparisons.append({"opportunity": index, "component": component, "perfect_batch_skip": True})
            continue
        evidence = [deepcopy(original[i]["feedback"].get(f"{component}_specific_info", {})) for i in indices]
        if any(not row for row in evidence):
            raise RuntimeError("Parent format failure prevents matched reflection evidence; preserve and review")
        for arm in PROTOCOL["arms"] if index % 2 == 0 else list(reversed(PROTOCOL["arms"])):
            directory = args.output_dir / f"opportunity-{index}" / arm
            request = {
                "component": component,
                "parent": parent,
                "reflection_records": {component: evidence},
                "arm": arm,
                "seed": index,
                "protocol": PROTOCOL["identity"],
            }
            require_contract(directory, request)
            settings.controller_selection = arm
            settings.seed = index
            config, _ = build_config(
                "react_v2",
                settings,
                observed_kwargs(args.reflection_model, args.reflection_api_base, directory, "optimizer"),
                str(directory),
            )
            strategy = config.reflection.reflection_strategy
            if strategy is None:
                raise RuntimeError("Missing three-role strategy")
            output = directory / "proposal.json"
            if output.exists():
                proposal = json.loads(output.read_text())
            else:
                started = time.monotonic()
                result, _ = strategy.reflect(parent, {component: evidence}, [component])
                candidate = {**parent, **result.new_texts}
                proposal = {
                    "candidate": candidate,
                    "changed": candidate != parent,
                    "seconds": time.monotonic() - started,
                    "metadata": result.metadata,
                    "prompts": result.prompts,
                    "raw_lm_outputs": result.raw_lm_outputs,
                }
                atomic_json(output, proposal)
            chosen = indices + transfer
            before = [original[i] for i in chosen]
            after = (
                evaluate_records(
                    directory / "evaluation",
                    proposal["candidate"],
                    [train[i] for i in chosen],
                    evaluate,
                    args.workers,
                    compute_f1=False,
                )
                if proposal["changed"]
                else before
            )
            row = {
                "opportunity": index,
                "component": component,
                "arm": arm,
                "changed": proposal["changed"],
                "proposal_seconds": proposal["seconds"],
                "training": paired_outcomes(before[:3], after[:3]),
                "transfer": paired_outcomes(before[3:], after[3:]),
            }
            atomic_json(directory / "comparison.json", row)
            comparisons.append(row)
    summary = {
        "protocol": PROTOCOL,
        "comparisons": comparisons,
        "execution_completed": True,
        "completed_ablation": False,
        "heldout_complete": False,
        "review_required": True,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Run a separately authorized training-only pilot against allocated endpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "api-base", "reflection-model", "reflection-api-base"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wiki17-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
