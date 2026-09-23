"""Record offline W&B progress and backfill completed cells from saved evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

from examples.common.recovery import file_digest
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG


def _digest(value: object) -> str:
    """Identify a logical cell independently of its physical allocation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def provider_usage(path: Path) -> dict:
    """Keep physical calls, reported tokens, and failures grouped by allocation and role."""
    totals = {}
    if not path.exists():
        return totals
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = f"{row.get('allocation_job_id')}/{row['role']}/{row.get('requested_model')}"
            group = totals.setdefault(key, {"calls": 0, "transport_errors": 0, "response_errors": 0, "empty_completions": 0})
            group["calls"] += 1
            group["transport_errors"] += row.get("transport_outcome", row.get("outcome")) != "success"
            group["response_errors"] += bool(row.get("response_error"))
            group["empty_completions"] += row.get("empty_completion") is True
            for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                value = row.get(field)
                group.setdefault(field, 0)
                group.setdefault(f"{field}_unreported_calls", 0)
                if value is None:
                    group[f"{field}_unreported_calls"] += 1
                else:
                    group[field] += value
    return totals


class HotpotqaWandb:
    """Observe optimization without changing selection, recovery, or evaluation."""

    def __init__(
        self, directory: Path, contract: dict, project: str, entity: str | None = None, *, kind: str = "optimization"
    ):
        """Open an offline segment; cloud synchronization is a separate operation."""
        self.directory = directory
        self.contract = contract
        self.kind = kind
        self.cell = _digest(contract)
        self.last_work = -1
        self.run = None
        root = directory / "tracking"
        root.mkdir(parents=True, exist_ok=True)
        allocation = os.environ.get("SLURM_JOB_ID", "local")
        self.identity = {"cell_sha256": self.cell, "allocation": allocation, "kind": kind}
        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                mode="offline",
                dir=str(root),
                id=f"{self.cell[:10]}-{kind}-{uuid.uuid4().hex[:8]}",
                group=contract.get("execution_runtime", {}).get("campaign_id") or self.cell[:12],
                name=f"{contract['condition']}-{contract['optimizer']['max_metric_calls']}-{kind}-{allocation}",
                job_type=kind,
                config={"run_contract": contract, **self.identity},
                settings=wandb.Settings(console="off", disable_git=True, disable_code=True),
            )
            self.run.define_metric("optimization/metric_calls")
            self.run.define_metric("validation/*", step_metric="optimization/metric_calls")
            self.run.define_metric("prompt/*", step_metric="optimization/metric_calls")
            self.run.summary["heldout_complete"] = False
            (root / f"{self.run.id}.json").write_text(
                json.dumps(
                    {
                        **self.identity,
                        "id": self.run.id,
                        "project": project,
                        "entity": entity,
                        "offline_directory": str(Path(self.run.dir).parent),
                    },
                    indent=2,
                )
                + "\n"
            )
        except Exception as exc:
            self._error(exc)

    def _error(self, exc: Exception) -> None:
        """Preserve a reporting failure without interrupting the experiment."""
        with (self.directory / "tracking" / "errors.jsonl").open("a") as stream:
            stream.write(json.dumps({**self.identity, "error_type": type(exc).__name__, "message": str(exc)}) + "\n")
        print(f"W&B reporting error recorded: {type(exc).__name__}")

    def _log(self, metrics: dict) -> None:
        """Write one observation without exposing a telemetry failure to GEPA."""
        if self.run is not None:
            try:
                self.run.log(metrics)
            except Exception as exc:
                self._error(exc)

    def _state(self, state) -> None:
        """Log validation and prompt size against actual logical evaluations."""
        if state.total_num_evals == self.last_work:
            return
        self.last_work = state.total_num_evals
        scores = [sum(values.values()) / len(values) for values in state.prog_candidate_val_subscores]
        best = max(range(len(scores)), key=scores.__getitem__)
        self._log(
            {
                "optimization/metric_calls": self.last_work,
                "optimization/full_candidates": len(scores),
                "validation/best_exact_match": scores[best],
                "validation/original_exact_match": scores[0],
                "validation/gain": scores[best] - scores[0],
                "prompt/best_characters": sum(map(len, state.program_candidates[best].values())),
            }
        )

    def on_iteration_start(self, event: dict) -> None:
        """Include the seed and restored checkpoint in this allocation segment."""
        self._state(event["state"])

    def on_iteration_end(self, event: dict) -> None:
        """Record the complete iteration, including accepted or rejected proposals."""
        self._state(event["state"])
        self._log({"optimization/iteration": event["iteration"], "proposal/accepted": int(event["proposal_accepted"])})

    def on_proposal_end(self, event: dict) -> None:
        """Record the proposed text size without changing proposal acceptance."""
        self._log(
            {
                "optimization/iteration": event["iteration"],
                "proposal/characters": sum(map(len, event["new_instructions"].values())),
                "proposal/metadata": event.get("metadata", {}),
            }
        )

    def on_optimization_end(self, event: dict) -> None:
        """Close the optimizer segment without claiming held-out completion."""
        self._state(event["final_state"])
        self.finish()

    def finish(self) -> None:
        """Flush the portable offline log."""
        if self.run is not None:
            try:
                self.run.summary["provider_usage_by_allocation"] = provider_usage(
                    self.directory / PROVIDER_ATTEMPT_LOG
                )
            except Exception as exc:
                self._error(exc)
            try:
                self.run.finish()
            except Exception as exc:
                self._error(exc)
            finally:
                self.run = None


def report_completed(directory: Path, project: str, entity: str | None = None) -> dict:
    """Backfill a completed cell without inference, state migration, or resumption.

    Args:
        directory: Verified portable cell archive, or a locally completed cell.
        project: W&B destination project.
        entity: Optional W&B account or team.

    Returns:
        Input hashes and the offline-report identity, suitable for later sync.

    Raises:
        ValueError: Winner, source contract, or validation metrics disagree.
    """
    candidates_path = directory / "candidates.json"
    metrics_path = directory / "final_metrics.json"
    payload = json.loads(candidates_path.read_text())
    final = json.loads(metrics_path.read_text())
    contract = json.loads((directory / "wikipedia-run-contract.json").read_text())
    if contract != payload["run_contract"]:
        raise ValueError("Candidate and run contracts differ")
    best = payload["best_idx"]
    if (
        _digest(payload["candidates"][best]) != final["candidate_sha256"]
        or payload["val_aggregate_scores"][best] != final["best_validation_exact_match"]
    ):
        raise ValueError("Frozen winner and final metrics differ")
    hashes = {
        p.name: file_digest(p) for p in (candidates_path, metrics_path, directory / "wikipedia-run-contract.json")
    }
    marker = directory / "tracking" / f"completed-{_digest(hashes)[:16]}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    tracker = HotpotqaWandb(directory, contract, project, entity, kind="result")
    if tracker.run is None:
        raise RuntimeError("W&B result initialization failed; original artifacts remain intact")
    try:
        tracker.run.config.update({"input_sha256": hashes})
        best_so_far = 0.0
        rows = []
        for index, (candidate, score, discovered) in enumerate(
            zip(
                payload["candidates"],
                payload["val_aggregate_scores"],
                payload["discovery_eval_counts"],
                strict=True,
            )
        ):
            best_so_far = max(best_so_far, score)
            tracker.run.log(
                {
                    "optimization/metric_calls": discovered,
                    "validation/candidate_exact_match": score,
                    "validation/best_exact_match": best_so_far,
                    "prompt/candidate_characters": sum(map(len, candidate.values())),
                }
            )
            rows.append([index, discovered, score, json.dumps(candidate, ensure_ascii=False)])
        tracker.run.log(
            {
                "candidates": tracker.wandb.Table(
                    columns=["candidate", "discovery_metric_calls", "validation_em", "prompts"], data=rows
                )
            }
        )
        tracker.run.summary.update(
            {
                "heldout_complete": True,
                "final_metrics": final,
                "optimization/metric_calls": payload["total_metric_calls"],
                "heldout/exact_match": final["test_exact_match"],
                "heldout/f1": final["test_f1"],
                "heldout/exact_match_gain": final["test_exact_match_gain"],
                "baseline/exact_match": final["baseline"]["test_exact_match"],
                "provider_usage_by_allocation": provider_usage(directory / PROVIDER_ATTEMPT_LOG),
            }
        )
        action_path = directory / "action_summary.json"
        if action_path.exists():
            tracker.run.summary["action_summary"] = json.loads(action_path.read_text())
        proof = {
            "id": tracker.run.id,
            "input_sha256": hashes,
            "offline_directory": str(Path(tracker.run.dir).parent),
            "project": project,
            "entity": entity,
        }
        tracker.run.finish()
        tracker.run = None
        marker.write_text(json.dumps(proof, indent=2) + "\n")
        return proof
    finally:
        tracker.finish()


def main() -> None:
    """Create a portable offline report from completed saved artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    args = parser.parse_args()
    print(json.dumps(report_completed(args.directory, args.project, args.entity), indent=2))


if __name__ == "__main__":
    main()
