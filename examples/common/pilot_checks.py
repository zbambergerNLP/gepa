"""Persist training-pilot evidence and verify a real optimizer cycle."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from examples.common.recovery import seal_progress

METHODS = ("vanilla", "react_v2", "react_v2_random", "action")
OPTIMIZER_PILOT_PROTOCOL = {
    "version": 2,
    "split": "train",
    "minibatch_size": 3,
    "cycles": 1,
    "stop_after": "completed_candidate_reevaluation_and_decision",
    "metric_improvement_required": False,
    "budget_reuse": ["standard", "double"],
}


def digest(value: Any) -> str:
    """Hash a JSON value independently of formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON artifact only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require_contract(directory: Path, contract: dict) -> Path:
    """Reject pilot resume with changed data, prompts, source, or runtime."""
    path = directory / "pilot-contract.json"
    normalized = json.loads(json.dumps(contract, default=str))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"Pilot configuration changed: {path}; use a fresh pilot directory")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Nonempty pilot directory has no contract: {directory}")
    else:
        atomic_json(path, normalized)
    return path


class CycleEvidence:
    """Observe production callbacks without changing selection or acceptance."""

    def __init__(self, directory: Path):
        """Attach evidence to one isolated, resumable optimizer check."""
        self.directory = directory
        self.events: dict[str, Any] = {}

    def _save(self) -> None:
        """Retain original callback evidence for diagnosis."""
        atomic_json(self.directory / "optimizer-cycle.json", self.events)

    def on_iteration_start(self, event: dict) -> None:
        """Discard an interrupted iteration's observations before replay."""
        if self.events and self.events.get("iteration") != event["iteration"]:
            atomic_json(self.directory / "incomplete-iterations" / f"{self.events['iteration']}.json", self.events)
        self.events = {"iteration": event["iteration"]}
        self._save()

    def on_reflective_dataset_built(self, event: dict) -> None:
        """Retain the actual task feedback supplied to reflection."""
        self.events["reflection"] = event
        self._save()

    def on_proposal_end(self, event: dict) -> None:
        """Retain proposal text and role/tool diagnostics."""
        self.events["proposal"] = event
        self._save()

    def on_candidate_accepted(self, event: dict) -> None:
        """Record normal acceptance without requiring it for qualification."""
        self.events["decision"] = {**event, "accepted": True}
        self._save()

    def on_evaluation_end(self, event: dict) -> None:
        """Retain the candidate's real three-example reevaluation."""
        if event.get("candidate_idx") is None and not event.get("is_seed_candidate"):
            self.events["reevaluation"] = event
            self._save()

    def on_candidate_rejected(self, event: dict) -> None:
        """Record normal rejection as a valid completed cycle."""
        self.events["decision"] = {**event, "accepted": False}
        self._save()

    def on_optimization_end(self, event: dict) -> None:
        """Seal the final optimizer checkpoint and its stage evidence."""
        self.events["finished"] = True
        self.events["metric_calls"] = event["total_metric_calls"]
        self._save()
        checkpoint = self.directory / "gepa_state.bin"
        if checkpoint.exists():
            seal_progress(
                self.directory, event["total_metric_calls"], [checkpoint, self.directory / "optimizer-cycle.json"]
            )

    def get_state(self) -> dict:
        """Persist callback observations with GEPA's exact-resume state."""
        return self.events

    def set_state(self, state: dict) -> None:
        """Restore observations when the completed checkpoint is resumed."""
        self.events = state

    def completed_cycle(self, state: Any) -> bool:
        """Stop only after a changed proposal receives a full reevaluation and decision."""
        return (
            "reflection" in self.events
            and bool(self.events.get("proposal", {}).get("new_instructions"))
            and len(self.events.get("reevaluation", {}).get("scores", [])) == 3
            and "decision" in self.events
        )

    def verify(self) -> dict:
        """Require every stage, accepting a tied or worse evaluated candidate."""
        if not self.events.get("finished") or any(
            key not in self.events for key in ("reflection", "proposal", "reevaluation", "decision")
        ):
            raise RuntimeError(
                "Optimizer pilot did not exercise a complete cycle; inspect stage evidence (including perfect-batch skips)"
            )
        if not self.events["proposal"].get("new_instructions"):
            raise RuntimeError("Optimizer pilot produced no candidate to reevaluate")
        if len(self.events["reevaluation"].get("scores", [])) != 3:
            raise RuntimeError("Optimizer pilot did not reevaluate all three training examples")
        summary = {
            "protocol": OPTIMIZER_PILOT_PROTOCOL,
            "completed_cycles": 1,
            "decision": self.events["decision"],
            "evidence_sha256": digest(self.events),
        }
        for name in ("pilot-contract.json", "terminalbench-run-contract.json"):
            if (self.directory / name).exists():
                summary["contract_file"] = name
                summary["contract_sha256"] = digest(json.loads((self.directory / name).read_text()))
        atomic_json(self.directory / "optimizer-pilot-complete.json", summary)
        return summary


def load_cycle(directory: Path) -> dict:
    """Verify completed optimizer evidence without rerunning any model calls."""
    summary = json.loads((directory / "optimizer-pilot-complete.json").read_text())
    evidence = json.loads((directory / "optimizer-cycle.json").read_text())
    if (
        summary.get("protocol") != OPTIMIZER_PILOT_PROTOCOL
        or summary.get("completed_cycles") != 1
        or not evidence.get("finished")
        or not evidence.get("reflection")
        or not evidence.get("proposal", {}).get("new_instructions")
        or len(evidence.get("reevaluation", {}).get("scores", [])) != 3
        or "accepted" not in evidence.get("decision", {})
        or summary.get("decision") != evidence.get("decision")
        or summary.get("evidence_sha256") != digest(evidence)
        or summary.get("contract_file") not in ("pilot-contract.json", "terminalbench-run-contract.json")
        or summary.get("contract_sha256") != digest(json.loads((directory / summary["contract_file"]).read_text()))
    ):
        raise ValueError(f"Incomplete or changed optimizer pilot evidence: {directory}")
    return summary
