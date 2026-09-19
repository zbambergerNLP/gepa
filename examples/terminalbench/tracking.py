"""Observe Terminal-Bench offline and report verified held-out results without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import uuid
from functools import wraps
from pathlib import Path

from examples.common.recovery import file_digest
from examples.terminalbench.token_usage import summarize_usage


def _digest(value: object) -> str:
    """Identify the exact contract or reporting inputs deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same optional offline destination on optimization and evaluation CLIs."""
    for name in ("project", "entity", "group"):
        parser.add_argument(
            f"--wandb-{name}",
            default=os.environ.get(f"TERMINALBENCH_WANDB_{name.upper()}"),
            help=f"Offline W&B {name}; logging is enabled only when a project is supplied",
        )


def record_tracking_error(directory: Path, exc: Exception) -> None:
    """Keep telemetry failures separate from scientific run outcomes."""
    try:
        root = directory / "tracking"
        root.mkdir(parents=True, exist_ok=True)
        with (root / "errors.jsonl").open("a") as stream:
            stream.write(json.dumps({"error_type": type(exc).__name__, "message": str(exc)}) + "\n")
    except OSError:
        pass
    print(f"W&B reporting error: {type(exc).__name__}; inspect {directory / 'tracking/errors.jsonl'}")


def _observational(method):
    """Prevent disabled or failed telemetry from changing optimizer behavior."""

    @wraps(method)
    def observe(self, *args, **kwargs):
        if self.run is not None:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                record_tracking_error(self.directory, exc)

    return observe


class TerminalbenchWandb:
    """Keep each physical allocation separate while grouping one logical campaign."""

    def __init__(
        self,
        directory: Path,
        contract: dict,
        project: str | None,
        entity: str | None = None,
        group: str | None = None,
        *,
        kind: str = "optimization",
        usage_roots: tuple[Path, ...] = (),
    ):
        self.directory = directory
        self.run = None
        self.last_work = -1
        self.usage_roots = usage_roots or (directory,)
        if not project:
            return
        try:
            import wandb

            self.wandb = wandb
            root = directory / "tracking"
            root.mkdir(parents=True, exist_ok=True)
            cell = _digest(contract)
            allocation = os.environ.get("SLURM_JOB_ID", "local")
            self.run = wandb.init(
                project=project,
                entity=entity,
                group=group or cell[:12],
                mode="offline",
                dir=str(root),
                id=f"{cell[:10]}-{kind}-{uuid.uuid4().hex[:8]}",
                job_type=kind,
                name=f"{contract['optimization_scope']}-{contract['condition']}-{contract['budget']}-{kind}-{allocation}",
                config={"run_contract": contract, "allocation": allocation, "cell_sha256": cell},
                settings=wandb.Settings(console="off", disable_git=True, disable_code=True),
            )
            self.run.define_metric("optimization/metric_calls")
            for pattern in ("validation/*", "prompt/*"):
                self.run.define_metric(pattern, step_metric="optimization/metric_calls")
            self.run.summary.update(
                {"optimization_complete": False, "heldout_complete": False, "completed_ablation": False}
            )
            (root / f"{self.run.id}.json").write_text(
                json.dumps(
                    {
                        "id": self.run.id,
                        "project": project,
                        "entity": entity,
                        "group": group or cell[:12],
                        "allocation": allocation,
                        "cell_sha256": cell,
                        "kind": kind,
                        "offline_directory": str(Path(self.run.dir).parent),
                    },
                    indent=2,
                )
                + "\n"
            )
        except Exception as exc:
            record_tracking_error(directory, exc)

    def _state(self, state) -> None:
        """Log the completed validation frontier without modifying optimizer state."""
        assert self.run is not None
        if state.total_num_evals == self.last_work:
            return
        scores = [statistics.mean(values.values()) for values in state.prog_candidate_val_subscores]
        best = max(range(len(scores)), key=scores.__getitem__)
        self.run.log(
            {
                "optimization/metric_calls": state.total_num_evals,
                "optimization/full_candidates": len(scores),
                "validation/best_pass_at_1": scores[best],
                "validation/original_pass_at_1": scores[0],
                "prompt/best_characters": sum(map(len, state.program_candidates[best].values())),
            }
        )
        self.last_work = state.total_num_evals

    @_observational
    def on_iteration_start(self, event: dict) -> None:
        """Include the seed or restored checkpoint at the start of an allocation."""
        self._state(event["state"])

    @_observational
    def on_iteration_end(self, event: dict) -> None:
        """Record accepted and rejected proposals against actual logical work."""
        assert self.run is not None
        self._state(event["state"])
        self.run.log(
            {"optimization/iteration": event["iteration"], "proposal/accepted": int(event["proposal_accepted"])}
        )

    @_observational
    def on_proposal_end(self, event: dict) -> None:
        """Preserve proposed text size and action metadata, including no-ops."""
        assert self.run is not None
        self.run.log(
            {
                "optimization/iteration": event["iteration"],
                "proposal/characters": sum(map(len, event["new_instructions"].values())),
                "proposal/metadata": event.get("metadata", {}),
            }
        )

    @_observational
    def on_optimization_end(self, event: dict) -> None:
        """Record the final validation candidates separately from held-out results."""
        assert self.run is not None
        state = event["final_state"]
        self._state(state)
        self.run.log(
            {
                "candidates": self.wandb.Table(
                    columns=["candidate", "validation_pass_at_1", "prompts"],
                    data=[
                        [i, statistics.mean(scores.values()), json.dumps(candidate, ensure_ascii=False)]
                        for i, (candidate, scores) in enumerate(
                            zip(state.program_candidates, state.prog_candidate_val_subscores, strict=True)
                        )
                    ],
                )
            }
        )
        self.run.summary["optimization_complete"] = True

    def finish(self) -> None:
        """Flush even after an interrupted optimization; never claim held-out completion here."""
        if self.run is not None:
            try:
                self.run.summary["physical_usage_cumulative"] = summarize_usage(list(self.usage_roots))
                self.run.summary["usage_scope"] = (
                    "all recorded calls in the optimization and Harbor roots, including prior allocations"
                )
            except Exception as exc:
                record_tracking_error(self.directory, exc)
            try:
                self.run.finish()
            except Exception as exc:
                record_tracking_error(self.directory, exc)
            finally:
                self.run = None


def report_completed(
    directory: Path,
    heldout: Path,
    cell: str,
    project: str,
    entity: str | None = None,
    group: str | None = None,
) -> dict:
    """Create one idempotent offline report after verifying the winner and all test repetitions.

    This reads trusted local checkpoints. It never launches Harbor or model calls.
    The marker covers the input hashes and destination; copying the complete tracking
    directory preserves its offline log and evidence for later ``wandb sync``.
    """
    from examples.terminalbench.evaluate import (
        FROZEN_COMPARISON_FILENAME,
        TEST_REPETITIONS,
        _validate_repetition,
        freeze_comparison,
    )
    from examples.terminalbench.main import RUN_CONTRACT_FILENAME

    manifest, expected = freeze_comparison({cell: directory})
    frozen_path, summary_path = heldout / FROZEN_COMPARISON_FILENAME, heldout / "summary.json"
    captured = {"frozen-comparison.json": frozen_path.read_bytes(), "summary.json": summary_path.read_bytes()}
    frozen, summary = json.loads(captured["frozen-comparison.json"]), json.loads(captured["summary.json"])
    if any(frozen[key] != expected[key] for key in ("shared_configuration", "protocol", "schema_version")):
        raise ValueError("Held-out configuration differs from the optimization run")
    source = frozen["source_runs"][cell]
    if {k: v for k, v in source.items() if k != "run_dir"} != {
        k: v for k, v in expected["source_runs"][cell].items() if k != "run_dir"
    }:
        raise ValueError("Held-out source differs from the completed optimization checkpoint")
    if summary.get("complete") is not True or cell not in summary["completed_cells"]:
        raise ValueError("Held-out comparison is incomplete")
    expected_metadata = {
        "protocol": expected["protocol"],
        "test_task_count": len(manifest.splits["test"]),
        "score_units": "fraction",
        "experiment": manifest.experiment,
        "student_model": expected["shared_configuration"]["student_model"],
    }
    if any(summary.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Held-out summary metadata differs from the frozen configuration")
    evidence = {
        "run-contract.json": directory / RUN_CONTRACT_FILENAME,
        "frozen-comparison.json": frozen_path,
        "summary.json": summary_path,
    }
    captured["run-contract.json"] = evidence["run-contract.json"].read_bytes()
    if json.loads(captured["run-contract.json"]) != source["contract"]:
        raise ValueError("Run contract changed during reporting")
    scores = {}
    seen = set()
    for label in ("initial", cell):
        harness = expected["harnesses"][label]
        if frozen["harnesses"][label] != harness:
            raise ValueError("Frozen winner or baseline differs from the optimization run")
        repetitions = []
        for repetition in range(1, TEST_REPETITIONS + 1):
            name = f"{label}-repetition-{repetition}.json"
            path = evidence[name] = heldout / name
            captured[name] = path.read_bytes()
            record = json.loads(captured[name])
            _validate_repetition(
                record,
                {"harness": label, "repetition": repetition, "candidate_digest": harness["candidate_digest"]},
                manifest.splits["test"],
            )
            if record["evaluation_id"] in seen:
                raise ValueError("Held-out repetitions reuse a Harbor evaluation")
            seen.add(record["evaluation_id"])
            repetitions.append(statistics.mean(record["scores"].values()))
        scores[label] = {
            "optimization_scope": harness["optimization_scope"],
            "candidate_digest": harness["candidate_digest"],
            "repetition_pass_at_1": repetitions,
            "mean_pass_at_1": statistics.mean(repetitions),
            "std_pass_at_1": statistics.stdev(repetitions),
            "task_attempts": len(manifest.splits["test"]) * TEST_REPETITIONS,
        }
        if scores[label] != summary["harnesses"][label]:
            raise ValueError("Held-out summary differs from the recorded task scores")
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()}
    identity = {"input_sha256": hashes, "cell": cell, "project": project, "entity": entity, "group": group}
    # Other cells extend the shared comparison; they must not duplicate this cell's report.
    stable_identity = {
        **identity,
        "source": source,
        "input_sha256": {
            name: digest for name, digest in hashes.items() if name not in {"frozen-comparison.json", "summary.json"}
        },
    }
    stable_identity["source"] = {key: value for key, value in source.items() if key != "run_dir"}
    marker = directory / "tracking" / f"completed-{_digest(stable_identity)[:16]}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    tracker = TerminalbenchWandb(directory, source["contract"], project, entity, group, kind="result")
    if tracker.run is None:
        raise RuntimeError("W&B result initialization failed; source artifacts remain intact")
    try:
        target = Path(tracker.run.dir) / "evidence"
        target.mkdir(parents=True, exist_ok=True)
        for name, path in evidence.items():
            if file_digest(path) != hashes[name]:
                raise ValueError("Result evidence changed during reporting")
            (target / name).write_bytes(captured[name])
        tracker.run.save(str(target / "*.json"), base_path=str(tracker.run.dir), policy="now")
        tracker.run.config.update(identity)
        tracker.run.summary.update(
            {
                "optimization_complete": True,
                "heldout_complete": True,
                "completed_ablation": True,
                "optimization/metric_calls": source["optimization_metric_calls"],
                "heldout/pass_at_1": scores[cell]["mean_pass_at_1"],
                "heldout/std_pass_at_1": scores[cell]["std_pass_at_1"],
                "baseline/pass_at_1": scores["initial"]["mean_pass_at_1"],
                "heldout/pass_at_1_gain": scores[cell]["mean_pass_at_1"] - scores["initial"]["mean_pass_at_1"],
                "test_repetitions": TEST_REPETITIONS,
                "test_task_count": len(manifest.splits["test"]),
            }
        )
        proof = {**identity, "id": tracker.run.id, "offline_directory": str(Path(tracker.run.dir).parent)}
        tracker.run.finish()
        tracker.run = None
        marker.write_text(json.dumps(proof, indent=2) + "\n")
        return proof
    finally:
        tracker.finish()


def main() -> None:
    """Backfill a verified portable archive without performing new evaluations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--heldout-dir", required=True, type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--group")
    args = parser.parse_args()
    print(
        json.dumps(
            report_completed(
                args.run_dir,
                args.heldout_dir,
                args.cell,
                args.project,
                args.entity,
                args.group,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
