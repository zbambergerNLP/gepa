"""Drain cloud allocations at durable task boundaries without changing run budgets."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from examples.common.artifacts import atomic_json, digest
from gepa.core.adapter import EvaluationBatch

POLICY = "ordered-task-windows-v1"
PAUSED_EXIT_CODE = 75


class AllocationPause(BaseException):
    """Leave an unfinished stage intact rather than score a deadline as a task error."""


def pause_due() -> bool:
    """Read the allocation deadline independently of the scientific stopping budget."""
    deadline = os.environ.get("GEPA_PAUSE_AT")
    return deadline is not None and time.time() >= float(deadline)


def check_pause() -> None:
    """Drain only between complete task groups or optimizer operations."""
    if pause_due():
        raise AllocationPause()


class TaskWindows:
    """Replay completed positions of one batch, never an unresolved started group.

    The optimizer checkpoints the scope and ordinal along with adapter state.
    Its existing evaluation journal restores that cursor when it replays a whole
    parent/child batch. Repeated training IDs therefore remain distinct attempts.
    """

    def __init__(self, root: Path, width: int):
        self.root, self.width = root / ".task-windows", width
        self.scope, self.ordinal = "seed", 0

    def on_iteration_start(self, event) -> None:
        """Use the checkpoint-stable iteration number, not random directory IDs."""
        self.scope, self.ordinal = f"iteration-{event['iteration']}", 0
        check_pause()

    def get_state(self) -> dict:
        """Return the next logical batch position for checkpoint or journal replay."""
        return {"policy": POLICY, "scope": self.scope, "ordinal": self.ordinal}

    def set_state(self, state: dict) -> None:
        """Refuse incompatible cursors instead of repeating or skipping tasks."""
        if state.get("policy") != POLICY or type(state.get("ordinal")) is not int or state["ordinal"] < 0:
            raise ValueError("Invalid continuation cursor")
        if not isinstance(state.get("scope"), str):
            raise ValueError("Invalid continuation scope")
        self.scope, self.ordinal = state["scope"], state["ordinal"]

    def evaluate(self, adapter, batch, candidate, capture_traces, context, validate):
        """Keep native concurrency inside each group and retain exact ordered outputs."""
        directory = self.root / self.scope / f"batch-{self.ordinal:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        request = digest([batch, candidate, capture_traces, context, self.width])
        contract = directory / "request.json"
        if contract.exists():
            if json.loads(contract.read_text()) != {"sha256": request}:
                raise ValueError("Continuation batch order, candidate, or settings changed")
        else:
            atomic_json(contract, {"sha256": request})
        results, elapsed = [], 0.0
        for start in range(0, len(batch), self.width):
            path = directory / f"window-{start:06d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("state") != "sealed":
                    raise RuntimeError(f"Unresolved task group; do not rerun: {path}")
                payload = record["payload"]
                if record.get("sha256") != digest(payload) or payload["request_sha256"] != request:
                    raise ValueError(f"Continuation evidence checksum changed: {path}")
                result = EvaluationBatch(**payload["evaluation"])
                if payload["adapter_state"] is not None:
                    adapter.set_adapter_state(payload["adapter_state"])
            else:
                check_pause()
                # A crash after this durable claim is uncertain even if no model
                # request is visible. Only a clean pause is eligible for rollover.
                with path.open("x") as stream:
                    json.dump({"state": "started", "request_sha256": request}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                started = time.perf_counter()
                result = adapter.evaluate(batch[start : start + self.width], candidate, capture_traces=capture_traces)
                validate(result, len(batch[start : start + self.width]), capture_traces)
                getter = getattr(adapter, "get_adapter_state", None)
                payload = {
                    "request_sha256": request,
                    "evaluation": asdict(result),
                    "adapter_state": getter() if getter else None,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                atomic_json(path, {"state": "sealed", "payload": payload, "sha256": digest(payload)})
            validate(result, len(batch[start : start + self.width]), capture_traces)
            results.append(result)
            elapsed += payload["elapsed_seconds"]
        self.ordinal += 1
        return EvaluationBatch(
            outputs=[row for result in results for row in result.outputs],
            scores=[score for result in results for score in result.scores],
            trajectories=[row for result in results for row in result.trajectories] if capture_traces else None,
            objective_scores=(
                [row for result in results for row in result.objective_scores]
                if results and all(result.objective_scores is not None for result in results)
                else None
            ),
            num_metric_calls=sum(
                result.num_metric_calls if result.num_metric_calls is not None else len(result.scores)
                for result in results
            ),
        ), elapsed
