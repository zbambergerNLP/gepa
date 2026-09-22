"""Compare two pinned FOREST revisions on fixed training and synthetic diagnostics.

Every proposal starts from the original prompts. Transfer examples never enter
reflection or selection. This is a qualification experiment, not a campaign
ablation, held-out test, or statistical demonstration of generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from examples.common.pilot_checks import atomic_json, digest, require_contract
from examples.common.react_v2 import benchmark_data_identity, resolve_template_family
from examples.common.wiki17_bm25 import Wiki17BM25Retriever
from examples.common.wikipedia import WikipediaPassage
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
from examples.hotpotqa.tracking import HotpotqaWandb, provider_usage
from examples.hotpotqa.utils import HOTPOTQA_HF_REVISION, f1_score, load_hotpotqa_dataset

COMPONENTS = ("summarize1", "create_query_hop2", "summarize2", "final_answer")
PROTOCOL = {
    "version": 1,
    "proposal_train_indices": {name: list(range(i * 3, i * 3 + 3)) for i, name in enumerate(COMPONENTS)},
    "transfer_train_indices": list(range(12, 36)),
    "proposal_seed": 0,
    "proposals_per_component_per_revision": 1,
    "parent": "original_prompts_for_every_proposal",
    "comparison_order": "control_revised_alternating_by_component",
    "selection": "none; score every proposal including ordinary rejections and no-ops",
    "transfer_used_for_reflection": False,
    "validation_or_test_evaluation": False,
    "usefulness_signal": "revised transfer mean exceeds control and original, with no additional synthetic losses",
    "interpretation": "small paired development diagnostic; not proof across all examples",
}


def partition_training(train: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Freeze disjoint proposal batches and transfer cases before any inference."""
    if len(train) < 36 or len({row["id"] for row in train[:36]}) != 36:
        raise ValueError("Qualification requires 36 distinct ordered training examples")
    return (
        {name: [train[i] for i in indices] for name, indices in PROTOCOL["proposal_train_indices"].items()},
        [train[i] for i in PROTOCOL["transfer_train_indices"]],
    )


def source_identity(root: Path) -> dict:
    """Verify immutable staged source bytes before loading a comparison revision."""
    commit = (root / ".gepa-source-commit").read_text().strip()
    manifest = (root / ".gepa-source-manifest.sha256").read_text().strip()
    if hashlib.sha256((root / ".gepa-source-manifest.sha256sums").read_bytes()).hexdigest() != manifest:
        raise ValueError(f"Source manifest changed: {root}")
    subprocess.run(["sha256sum", "--check", "--status", ".gepa-source-manifest.sha256sums"], cwd=root, check=True)
    return {"commit": commit, "manifest_sha256": manifest, "directory": str(root)}


def evaluate_records(directory: Path, candidate: dict, examples: list[dict], evaluate, workers: int) -> list[dict]:
    """Persist scored outcomes, including task-format zeros, with exact recovery identities."""
    require_contract(directory, {"candidate": candidate, "examples": examples})

    def one(item: tuple[int, dict]) -> dict:
        """Reuse only an intact record for this exact candidate and example."""
        index, example = item
        path = directory / "records" / f"{index:04d}.json"
        identity = digest({"candidate": candidate, "example": example})
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["sha256"] != digest(saved["record"]) or saved["record"]["identity"] != identity:
                raise ValueError(f"Evaluation record changed: {path}")
            return saved["record"]
        score, feedback = evaluate(candidate, example)
        prediction = feedback.get("final_answer_specific_info", {}).get("Generated Outputs", {}).get("answer", "")
        record = {
            "id": example["id"],
            "identity": identity,
            "score": score,
            "f1": f1_score(prediction, example["answer"]),
            "feedback": feedback,
            "allocation": os.environ.get("SLURM_JOB_ID"),
        }
        atomic_json(path, {"record": record, "sha256": digest(record)})
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, enumerate(examples)))


def paired_outcomes(original: list[dict], revised: list[dict]) -> dict:
    """Report net benefit and losses without selecting only successful examples."""
    if not original or [r["id"] for r in original] != [r["id"] for r in revised]:
        raise ValueError("Paired scoring requires identical nonempty ordered example IDs")
    pairs = list(zip(original, revised, strict=True))
    return {
        "count": len(pairs),
        "original_mean": sum(r["score"] for r in original) / len(pairs),
        "candidate_mean": sum(r["score"] for r in revised) / len(pairs),
        "delta": sum(b["score"] - a["score"] for a, b in pairs) / len(pairs),
        "wins": sum(b["score"] > a["score"] for a, b in pairs),
        "losses": sum(b["score"] < a["score"] for a, b in pairs),
        "unchanged": sum(b["score"] == a["score"] for a, b in pairs),
        "already_correct_regressions": [a["id"] for a, b in pairs if a["score"] == 1 and b["score"] < 1],
    }


class DiagnosticRetriever:
    """Supply fixed synthetic passages, keeping this diagnostic separate from BM25 transfer."""

    def __init__(self, passages: list[list[str]]):
        """Retain source passages for a single controlled diagnostic question."""
        self.passages = [WikipediaPassage(title, text) for title, text in passages]

    def search(self, query: str, limit: int = 7) -> list[WikipediaPassage]:
        """Return the same evidence at both hops regardless of generated query."""
        return self.passages[:limit]


def proposal_worker(request_file: Path) -> None:
    """Generate exactly one proposal using imports from the selected pinned revision."""
    request = json.loads(request_file.read_text())
    directory = request_file.parent
    settings = argparse.Namespace(**request["settings"])
    kwargs = observed_kwargs(settings.reflection_model, settings.reflection_api_base, directory, "optimizer")
    config, _ = build_config("react_v2", settings, kwargs, str(directory))
    strategy = config.reflection.reflection_strategy
    if strategy is None:
        raise RuntimeError("FOREST strategy was not constructed")
    proposal, _ = strategy.reflect(request["candidate"], request["reflection_records"], [request["component"]])
    candidate = {**request["candidate"], **proposal.new_texts}
    atomic_json(
        directory / "proposal.json",
        {
            "request_sha256": digest(request),
            "candidate": candidate,
            "changed": candidate != request["candidate"],
            "metadata": proposal.metadata,
            "prompts": proposal.prompts,
            "raw_lm_outputs": proposal.raw_lm_outputs,
            "strategy_contract": strategy.run_contract(request["candidate"]),
            "imported_source": str(Path(sys.modules["gepa"].__file__).resolve()),
        },
    )


def run_comparison(args: argparse.Namespace) -> dict:
    """Compare all fixed proposal pairs without tuning or migrating the current campaign."""
    root = Path(__file__).resolve().parents[2]
    sources = {"control": source_identity(args.control_source), "revised": source_identity(root)}
    if (
        sources["control"]["commit"] != args.control_commit
        or sources["control"]["commit"] == sources["revised"]["commit"]
    ):
        raise ValueError("Comparison requires the exact distinct control revision")
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
            "--enforce-scientific-contract",
            "--text-limits",
            "null",
            "--editor-mode",
            "single_call",
        ]
    )
    train, validation, test = load_hotpotqa_dataset(seed=0)
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
        valset=validation,
        testset=test,
    )
    _validate_scientific_data_identity(settings)
    del validation, test
    batches, transfer = partition_training(train)
    diagnostics = json.loads((Path(__file__).with_name("generalization_cases.json")).read_text())
    retriever = Wiki17BM25Retriever(args.wiki17_dir)
    _verify_scientific_retriever_integrity(retriever)
    settings.retrieval_provenance = retriever.provenance()
    runtime = build_run_contract("react_v2", settings)
    candidate = seed_candidate("2stage", "structured", resolve_template_family("auto", args.model))
    contract = {
        "protocol": PROTOCOL,
        "sources": sources,
        "runtime": runtime,
        "candidate": candidate,
        "proposal_batches": batches,
        "transfer": transfer,
        "synthetic_diagnostics": diagnostics,
    }
    require_contract(args.output_dir, contract)
    tracking_contract = {
        **runtime,
        "qualification_comparison": {"protocol": PROTOCOL, "sources": sources, "contract_sha256": digest(contract)},
    }
    tracker = (
        HotpotqaWandb(
            args.output_dir, tracking_contract, args.wandb_project, args.wandb_entity, kind="generalization-qualification"
        )
        if args.wandb_project
        else None
    )

    def evaluate(candidate: dict, example: dict) -> tuple[float, dict]:
        """Use the production solver settings and keep synthetic evidence explicitly separate."""
        if "passages" in example:
            return diagnostic_evaluators[example["id"]](candidate, example)
        return natural_evaluator(candidate, example)

    solver_kwargs = observed_kwargs(args.model, args.api_base, args.output_dir, "solver")
    natural_evaluator = make_evaluator(
        args.model, retriever, args.api_base, solver_lm_kwargs=solver_kwargs, reflection_diagnostics=True
    )
    # DSPy configuration belongs to this thread; worker threads only execute evaluators.
    diagnostic_evaluators = {
        example["id"]: make_evaluator(
            args.model,
            DiagnosticRetriever(example["passages"]),
            args.api_base,
            solver_lm_kwargs=solver_kwargs,
            reflection_diagnostics=True,
        )
        for example in diagnostics
    }
    comparisons = []
    try:
        baseline = evaluate_records(
            args.output_dir / "original", candidate, train[:36] + diagnostics, evaluate, args.workers
        )
        original_by_id = {row["id"]: row for row in baseline}
        for index, component in enumerate(COMPONENTS):
            for variant in ("control", "revised") if index % 2 == 0 else ("revised", "control"):
                directory = args.output_dir / variant / component
                records = [
                    deepcopy(original_by_id[e["id"]]["feedback"].get(f"{component}_specific_info", {}))
                    for e in batches[component]
                ]
                if any(not record for record in records):
                    raise RuntimeError(
                        "Original task-format failure prevents a matched reflection batch; preserve and review"
                    )
                if variant == "control":
                    records = [{k: row[k] for k in ("Inputs", "Generated Outputs", "Feedback")} for row in records]
                request = {
                    "settings": vars(settings),
                    "component": component,
                    "candidate": candidate,
                    "reflection_records": {component: records},
                    "source": sources[variant],
                }
                require_contract(directory, request)
                request_file = directory / "request.json"
                atomic_json(request_file, request)
                proposal_path = directory / "proposal.json"
                if not proposal_path.exists():
                    source_root = sources[variant]["directory"]
                    env = {**os.environ, "PYTHONPATH": f"{source_root}/src:{source_root}"}
                    command = [
                        os.environ.get("GEPA_UV_BIN", "uv"),
                        "run",
                        "--no-project",
                        "--python",
                        sys.executable,
                        "python",
                        str(Path(__file__).resolve()),
                        "--proposal-request",
                        str(request_file),
                    ]
                    with (directory / "proposal-worker.log").open("a") as log:
                        subprocess.run(
                            command, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                        )
                proposal = json.loads(proposal_path.read_text())
                if proposal["request_sha256"] != digest(request):
                    raise ValueError("Proposal does not belong to the fixed request")
                if not Path(proposal["imported_source"]).is_relative_to(Path(sources[variant]["directory"])):
                    raise ValueError("Proposal worker imported the wrong source")
                cases = batches[component] + transfer + diagnostics
                original = [original_by_id[e["id"]] for e in cases]
                scored = (
                    evaluate_records(directory / "evaluation", proposal["candidate"], cases, evaluate, args.workers)
                    if proposal["changed"]
                    else original
                )
                row = {
                    "variant": variant,
                    "component": component,
                    "changed": proposal["changed"],
                    "proposal_sha256": digest(proposal),
                    "proposal_batch": paired_outcomes(original[:3], scored[:3]),
                    "transfer": paired_outcomes(original[3:27], scored[3:27]),
                    "synthetic": paired_outcomes(original[27:], scored[27:]),
                    "synthetic_categories": {
                        category: paired_outcomes(
                            [a for a, e in zip(original[27:], diagnostics, strict=True) if e["category"] == category],
                            [a for a, e in zip(scored[27:], diagnostics, strict=True) if e["category"] == category],
                        )
                        for category in sorted({e["category"] for e in diagnostics})
                    },
                    "actual_metric_evaluations": len(scored) if proposal["changed"] else 0,
                    "retained_task_format_errors": sum(bool(r["feedback"].get("evaluation_error")) for r in scored),
                }
                row["would_accept_on_proposal_batch"] = row["proposal_batch"]["delta"] > 0
                comparisons.append(row)
                atomic_json(directory / "comparison.json", row)
        grouped = {v: [r for r in comparisons if r["variant"] == v] for v in sources}
        means = {v: sum(r["transfer"]["candidate_mean"] for r in rows) / len(rows) for v, rows in grouped.items()}
        baseline_mean = paired_outcomes(baseline[12:36], baseline[12:36])["original_mean"]
        losses = {v: sum(r["synthetic"]["losses"] for r in rows) for v, rows in grouped.items()}
        summary = {
            "protocol": PROTOCOL,
            "contract_sha256": digest(contract),
            "comparisons": comparisons,
            "transfer_means": means,
            "original_transfer_mean": baseline_mean,
            "synthetic_losses": losses,
            "usefulness_signal": means["revised"] > max(means["control"], baseline_mean)
            and losses["revised"] <= losses["control"],
            "actual_metric_evaluations": len(baseline) + sum(r["actual_metric_evaluations"] for r in comparisons),
            "original_task_format_errors": sum(bool(r["feedback"].get("evaluation_error")) for r in baseline),
            "physical_provider_usage": {
                str(p.relative_to(args.output_dir)): provider_usage(p)
                for p in args.output_dir.rglob("provider-attempts.jsonl")
            },
            "heldout_complete": False,
            "completed_ablation": False,
            "execution_completed": True,
            "qualification_review_required": True,
        }
        atomic_json(args.output_dir / "generalization-summary.json", summary)
        if tracker and tracker.run:
            tracker._log({f"qualification/transfer_{v}": mean for v, mean in means.items()})
            try:
                tracker.run.summary.update(
                    {
                        "usefulness_signal": summary["usefulness_signal"],
                        "completed_ablation": False,
                        "heldout_complete": False,
                        "actual_metric_evaluations": summary["actual_metric_evaluations"],
                        "physical_provider_usage": summary["physical_provider_usage"],
                        "comparisons": comparisons,
                    }
                )
            except Exception as exc:
                tracker._error(exc)
        return summary
    finally:
        if tracker:
            tracker.finish()


def main() -> None:
    """Run the comparison or its isolated, source-pinned proposal subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-request", type=Path)
    parser.add_argument("--control-source", type=Path)
    parser.add_argument("--control-commit")
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--reflection-model")
    parser.add_argument("--reflection-api-base")
    parser.add_argument("--wiki17-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if args.proposal_request:
        proposal_worker(args.proposal_request)
    else:
        for field in (
            "control_source",
            "control_commit",
            "model",
            "api_base",
            "reflection_model",
            "reflection_api_base",
            "wiki17_dir",
            "output_dir",
        ):
            if getattr(args, field) is None:
                parser.error(f"--{field.replace('_', '-')} is required for comparison")
        if args.workers < 1:
            parser.error("--workers must be positive")
        run_comparison(args)


if __name__ == "__main__":
    main()
