"""Freeze each completed Terminal-Bench ablation and repeat held-out Pass@1 tests."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import statistics
from pathlib import Path
from typing import Any

from examples.common.recovery import run_guarded, seal_progress
from examples.terminalbench.main import (
    EVALUATION_PROTOCOL,
    EXPERIMENT_MANIFESTS,
    FOREST_CONDITIONS,
    REPO_ROOT,
    RUN_CONTRACT_FILENAME,
    SCOPE_CAMPAIGN_CELLS,
    TEST_REPETITIONS,
    build_run_contract,
)
from examples.terminalbench.pilot import validate_review
from examples.terminalbench.runtime import load_runtime_record, validate_identity
from examples.terminalbench.tracking import add_tracking_arguments, record_tracking_error, report_completed
from gepa.adapters.terminal_bench_adapter import (
    HarborCLI,
    TerminalBenchManifest,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import seed_documents
from gepa.adapters.terminal_bench_adapter.text_scope import DEFAULT_OPTIMIZATION_SCOPE, TerminalBenchTextScope
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState
from gepa.strategies.text_limits import resolve_text_limits

FROZEN_COMPARISON_FILENAME = "frozen-comparison.json"
METHOD_SPECIFIC_FIELDS = {
    "condition",
    "budget",
    "optimization_budget",
    "controller_selection",
    "proposer_backend",
    "reflection_level",
    "reflection_role_decoding",
    "max_proposer_model_calls",
    "react_execution",
    "semantic_action_space",
    "semantic_controller_policy",
    "stateless_selector_policy",
    "manifest",
    "optimization_scope",
    "text_scope",
    "component_kinds",
    "seed_document_digest",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace an artifact so interruption cannot leave partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_completed_run(
    run_dir: Path, condition: str, budget: str, optimization_scope: str = DEFAULT_OPTIMIZATION_SCOPE
) -> tuple[TerminalBenchManifest, dict[str, Any]]:
    """Select the validation winner from one completed, trusted local checkpoint.

    Args:
        run_dir: Directory produced by the Terminal-Bench optimization CLI.
        condition: Required method from the six-configuration campaign.
        budget: Required standard or double training budget for this cell.
        optimization_scope: Required text scope for the labeled campaign cell.

    Returns:
        Pinned manifest and the initial/selected harnesses with run provenance.

    Raises:
        ValueError: The run is incomplete, uses a partial split, or has drifted
            from its recorded protocol, model settings, or seed candidate.
    """
    contract = json.loads((run_dir / RUN_CONTRACT_FILENAME).read_text())
    if contract.get("optimization_scope") != optimization_scope:
        raise ValueError(f"{run_dir}: expected the {optimization_scope} optimization scope")
    if contract.get("budget") != budget:
        raise ValueError(f"{run_dir}: expected the {budget} budget for {condition}")
    if contract.get("reflection_level") != (2 if condition in FOREST_CONDITIONS else 0):
        raise ValueError(f"{run_dir}: reflection level does not match the campaign method {condition}")
    if contract.get("experiment") not in EXPERIMENT_MANIFESTS:
        raise ValueError(f"{run_dir}: only Terminal-Bench 2.1 runs may enter final comparison")
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[contract["experiment"]])
    expected = build_run_contract(
        argparse.Namespace(**contract),
        manifest,
        manifest.tasks("train"),
        manifest.tasks("val"),
        condition,
        contract["template_family"],
    )
    if {**contract, "manifest": str(manifest.path)} != expected:
        raise ValueError(
            f"{run_dir}: expected a matching {condition} run on the complete training and validation splits"
        )
    validate_review(contract.get("pilot_review"), contract, manifest)
    for role in ("student", "proposer"):
        validate_identity((contract.get("execution_runtime") or {}).get(role), contract[f"{role}_model"])
    state = GEPAState.load(str(run_dir))
    completed_iterations = state.i + 1
    if completed_iterations != contract["optimization_budget"]["max_iterations"]:
        epochs = contract["optimization_budget"]["training_epochs"]
        raise ValueError(f"{run_dir}: optimization has not completed its {epochs}-epoch budget")
    result = GEPAResult.from_state(state)
    scope = TerminalBenchTextScope(optimization_scope, contract["template_family"])
    if result.candidates[0] != scope.seed_candidate():
        raise ValueError(f"{run_dir}: checkpoint seed differs from the run contract")
    expected_val_ids = set(range(len(manifest.splits["val"])))
    if any(set(scores) != expected_val_ids for scores in result.val_subscores):
        raise ValueError(f"{run_dir}: candidate validation coverage is incomplete")
    if not all(math.isfinite(score) for score in result.val_aggregate_scores):
        raise ValueError(f"{run_dir}: validation scores must be finite")
    selected = scope.materialize(result.candidates[result.best_idx])
    return manifest, {
        "run_dir": str(run_dir.resolve()),
        "contract": contract,
        "completed_iterations": completed_iterations,
        "optimization_metric_calls": result.total_metric_calls,
        "selected_candidate_index": result.best_idx,
        "validation_scores": result.val_aggregate_scores,
        "initial": seed_documents(contract["template_family"]),
        "selected": selected,
    }


def freeze_comparison(run_dirs: dict[str, Path]) -> tuple[TerminalBenchManifest, dict[str, Any]]:
    """Freeze the supplied validation winners without waiting for later ablations.

    Args:
        run_dirs: One or more completed scope/method/budget cells for one model.

    Returns:
        Manifest and immutable harnesses, including the common initial one.

    Raises:
        ValueError: A cell is unknown, incomplete, or shared experimental settings differ.
    """
    if not run_dirs or not set(run_dirs).issubset(SCOPE_CAMPAIGN_CELLS):
        raise ValueError(f"Supply one or more supported campaign cells: {', '.join(SCOPE_CAMPAIGN_CELLS)}")
    runs = {}
    for label, (scope_name, condition, budget) in SCOPE_CAMPAIGN_CELLS.items():
        if label not in run_dirs:
            continue
        manifest, runs[label] = load_completed_run(run_dirs[label], condition, budget, scope_name)
    reference = next(iter(runs.values()))
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[reference["contract"]["experiment"]])
    shared = {key: value for key, value in reference["contract"].items() if key not in METHOD_SPECIFIC_FIELDS}
    for label, run in runs.items():
        other = {key: value for key, value in run["contract"].items() if key not in METHOD_SPECIFIC_FIELDS}
        if shared != other or reference["initial"] != run["initial"]:
            raise ValueError(f"{label}: all ablations must share benchmark, model settings, seed, and splits")
    candidates = {"initial": reference["initial"], **{label: run["selected"] for label, run in runs.items()}}
    return manifest, {
        "schema_version": 4,
        "protocol": dict(EVALUATION_PROTOCOL),
        "shared_configuration": shared,
        "source_runs": {
            label: {key: value for key, value in run.items() if key not in {"initial", "selected"}}
            for label, run in runs.items()
        },
        "harnesses": {
            label: {
                "documents": candidate,
                "candidate_digest": manifest.candidate_digest(candidate),
                "optimization_scope": "reference" if label == "initial" else SCOPE_CAMPAIGN_CELLS[label][0],
            }
            for label, candidate in candidates.items()
        },
    }


def _extend_comparison(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Add matched ablations without replacing any already-frozen winner or source."""
    collections = {"source_runs", "harnesses"}
    if {key: value for key, value in existing.items() if key not in collections} != {
        key: value for key, value in incoming.items() if key not in collections
    }:
        raise ValueError("This test directory belongs to a different frozen comparison: shared configuration changed")
    merged = dict(existing)
    for field in collections:
        previous, additions = existing[field], incoming[field]
        if any(previous[label] != additions[label] for label in previous.keys() & additions.keys()):
            raise ValueError("This test directory belongs to a different frozen comparison: frozen cell changed")
        combined = {**previous, **additions}
        merged[field] = {label: combined[label] for label in ("initial", *SCOPE_CAMPAIGN_CELLS) if label in combined}
    return merged


def _validate_manifest(manifest: TerminalBenchManifest, comparison: dict[str, Any]) -> None:
    """Require the exact recorded data and ordered split membership for every test call."""
    expected = {
        "experiment": manifest.experiment,
        "dataset": manifest.dataset,
        "task_refs": manifest.task_refs,
        "split_policy": manifest.split_policy,
        **{f"{split}_task_ids": manifest.splits[split] for split in ("train", "val", "test")},
    }
    if any(comparison["shared_configuration"].get(key) != value for key, value in expected.items()):
        raise ValueError("Test manifest must match the frozen data and identical train/validation/test splits")


def _validate_repetition(record: dict[str, Any], identity: dict[str, Any], task_ids: list[str]) -> None:
    """Reject reused or incomplete results instead of silently altering Pass@1."""
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Test repetition does not match its frozen harness and repetition index")
    scores = record.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(task_ids):
        raise ValueError("Test repetition must contain every held-out task exactly once")
    if any(score not in (0.0, 1.0) for score in scores.values()):
        raise ValueError("Pass@1 requires binary official verifier rewards")
    if not isinstance(record.get("evaluation_id"), str) or not record["evaluation_id"]:
        raise ValueError("Test repetition has no Harbor evaluation identity")


def evaluate_comparison(
    manifest: TerminalBenchManifest, comparison: dict[str, Any], output_dir: Path, harbor: HarborCLI
) -> dict[str, Any]:
    """Resume three fresh test repetitions per frozen harness and summarize Pass@1.

    Args:
        manifest: Pinned benchmark shared by every optimization run.
        comparison: Common initial harness and the newly completed validation winners.
        output_dir: Dedicated comparison directory; use one writer at a time.
        harbor: Runner with the recorded student model and runtime settings.

    Returns:
        Mean and sample standard deviation over three complete test repetitions.

    Raises:
        ValueError: Frozen identity changed or saved test results are invalid.
    """
    _validate_manifest(manifest, comparison)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / FROZEN_COMPARISON_FILENAME
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text())
        comparison = _extend_comparison(existing, comparison)
        if comparison != existing:
            # A previous summary only covered earlier cells; do not present it as complete.
            (output_dir / "summary.json").unlink(missing_ok=True)
            _write_json(frozen_path, comparison)
    else:
        if any(output_dir.glob("*-repetition-*.json")) or (output_dir / "summary.json").exists():
            raise ValueError("Existing test results have no frozen comparison")
        _write_json(frozen_path, comparison)

    # Winners are already materialized into complete runtime bundles in both scopes.
    adapter = TerminusAdapter(manifest, harbor, text_scope=TerminalBenchTextScope("all_text"))
    task_ids = manifest.splits["test"]
    records: dict[tuple[str, int], dict[str, Any]] = {}
    identities = {
        (label, repetition): {
            "harness": label,
            "repetition": repetition,
            "candidate_digest": harness["candidate_digest"],
        }
        for repetition in range(1, TEST_REPETITIONS + 1)
        for label, harness in comparison["harnesses"].items()
    }
    seen_evaluations: set[str] = set()
    for (label, repetition), identity in identities.items():
        path = output_dir / f"{label}-repetition-{repetition}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        _validate_repetition(record, identity, task_ids)
        if record["evaluation_id"] in seen_evaluations:
            raise ValueError("Each test repetition must use a distinct Harbor evaluation")
        seen_evaluations.add(record["evaluation_id"])
        records[label, repetition] = record

    for (label, repetition), identity in identities.items():
        if (label, repetition) in records:
            continue
        print(f"Testing {label}, repetition {repetition}/{TEST_REPETITIONS} ({len(task_ids)} tasks)", flush=True)
        batch = adapter.evaluate(manifest.tasks("test"), comparison["harnesses"][label]["documents"])
        job = batch.outputs[0]
        record = {
            **identity,
            "candidate_digest": job["candidate_digest"],
            "evaluation_id": job["evaluation_id"],
            "job_dir": job["job_dir"],
            "config_path": job["config_path"],
            "scores": {output["task_id"]: score for output, score in zip(batch.outputs, batch.scores, strict=True)},
        }
        _validate_repetition(record, identity, task_ids)
        if record["evaluation_id"] in seen_evaluations:
            raise ValueError("Each test repetition must use a distinct Harbor evaluation")
        seen_evaluations.add(record["evaluation_id"])
        _write_json(output_dir / f"{label}-repetition-{repetition}.json", record)
        repetitions = sorted(output_dir.glob("*-repetition-*.json"))
        seal_progress(output_dir, len(repetitions), repetitions)
        records[label, repetition] = record

    summary: dict[str, Any] = {
        "complete": True,
        "campaign_complete": set(comparison["source_runs"]) == set(SCOPE_CAMPAIGN_CELLS),
        "completed_cells": list(comparison["source_runs"]),
        "pending_cells": [label for label in SCOPE_CAMPAIGN_CELLS if label not in comparison["source_runs"]],
        "experiment": manifest.experiment,
        "student_model": comparison["shared_configuration"]["student_model"],
        "protocol": dict(EVALUATION_PROTOCOL),
        "test_task_count": len(task_ids),
        "score_units": "fraction",
        "harnesses": {},
    }
    for label, harness in comparison["harnesses"].items():
        scores = [
            statistics.mean(records[label, repetition]["scores"].values())
            for repetition in range(1, TEST_REPETITIONS + 1)
        ]
        summary["harnesses"][label] = {
            "optimization_scope": harness["optimization_scope"],
            "candidate_digest": harness["candidate_digest"],
            "repetition_pass_at_1": scores,
            "mean_pass_at_1": statistics.mean(scores),
            "std_pass_at_1": statistics.stdev(scores),
            "task_attempts": len(task_ids) * TEST_REPETITIONS,
        }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Evaluate completed local cells and extend their model's matched comparison."""
    parser = argparse.ArgumentParser(description="Three frozen Terminal-Bench Pass@1 test repetitions")
    add_tracking_arguments(parser)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        metavar="CELL=PATH",
        help=f"Supply any completed cells, one per flag: {', '.join(SCOPE_CAMPAIGN_CELLS)}",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument(
        "--runtime-record", type=Path, help="Current local task-server record from the runtime launcher"
    )
    args = parser.parse_args(argv)
    run_dirs = {}
    for specification in args.run_dir:
        label, separator, path = specification.partition("=")
        if not separator or label not in SCOPE_CAMPAIGN_CELLS or not path or label in run_dirs:
            parser.error("Each --run-dir must specify a distinct supported CELL=PATH")
        run_dirs[label] = Path(path)
    manifest, comparison = freeze_comparison(run_dirs)
    contract = comparison["shared_configuration"]
    try:
        runtime = load_runtime_record(args.runtime_record, contract["student_model"], contract["student_api_base"])
        if runtime != contract["execution_runtime"]["student"]:
            raise ValueError("Final evaluation runtime differs from the pilot and optimization; collect a new pilot")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    harbor = HarborCLI(
        manifest=manifest,
        student_model=contract["student_model"],
        student_api_base=contract["student_api_base"],
        work_dir=args.output_dir / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=contract["n_concurrent"],
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        process_timeout_sec=contract["harbor_process_timeout_sec"],
        text_limits=resolve_text_limits(contract["text_limits"]),
        student_agent_kwargs={
            "token_limits": contract["token_limits"],
            "model_info": contract["student_model_info"],
            "llm_kwargs": {
                "num_retries": contract["student_num_retries"],
                **contract["student_decoding"],
                **contract["student_request_overrides"],
            },
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation is already writing to this output directory")
        summary = evaluate_comparison(manifest, comparison, args.output_dir, harbor)
        if args.wandb_project:
            for label, directory in run_dirs.items():
                try:
                    report_completed(
                        directory,
                        args.output_dir,
                        label,
                        args.wandb_project,
                        args.wandb_entity,
                        args.wandb_group,
                    )
                except Exception as exc:
                    record_tracking_error(directory, exc)
    for label, scores in summary["harnesses"].items():
        print(f"{label}: Pass@1 {scores['mean_pass_at_1']:.2%} +/- {scores['std_pass_at_1']:.2%}")
    print(f"Saved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    run_guarded(main)
