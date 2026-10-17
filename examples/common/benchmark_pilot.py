"""Require auditable proposal and evaluation evidence from optimizer pilots."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from examples.common.artifacts import digest
from gepa.core.callbacks import (
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    CandidateSelectedEvent,
    EvaluationEndEvent,
    IterationStartEvent,
    ProposalEndEvent,
)


class OptimizerPilotEvidence:
    """Checkpoint actual proposal stages without changing optimizer decisions."""

    def __init__(self, contract_sha256: str):
        self.contract_sha256 = contract_sha256
        self.proposals: dict[str, dict[str, Any]] = {}
        self.parents: dict[int, dict[str, str]] = {}
        self.task_number = 0

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        """Align task ordinals with the core's per-iteration proposal identifiers."""
        self.parents = {}
        self.task_number = 0

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        """Retain sampled opportunities so skipped reflection remains visible."""
        self.parents[event["candidate_idx"]] = deepcopy(event["candidate"])
        proposal_id = f"{event['iteration']}-{self.task_number}"
        self.task_number += 1
        self.proposals[proposal_id] = {
            "proposal_id": proposal_id,
            "iteration": event["iteration"],
            "parent_candidate_sha256": digest(event["candidate"]),
            "stage": "sampled",
        }

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        """Identify generated candidates independently of whether they improve."""
        metadata = event.get("metadata", {})
        record = self.proposals[metadata["proposal_id"]]
        if not event["new_instructions"]:
            record["stage"] = "no_candidate"
            return
        parent_idx = int(next(iter(metadata["parent_branch_history_lengths"])))
        record.update(
            candidate_sha256=digest({**self.parents[parent_idx], **event["new_instructions"]}),
            stage="proposed",
        )

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        """Associate completed child evaluations with proposals in the core's batch order."""
        if event["candidate_idx"] is not None or event["is_seed_candidate"] or len(event["parent_ids"]) != 1:
            return
        record = next(
            item
            for item in self.proposals.values()
            if item["iteration"] == event["iteration"] and item["stage"] == "proposed"
        )
        record.update(scores=list(event["scores"]), outputs_sha256=digest(event["outputs"]), stage="evaluated")

    def _outcome(self, event: CandidateAcceptedEvent | CandidateRejectedEvent, outcome: str) -> None:
        """Count only generated candidates whose child evaluation reached admission."""
        record = self.proposals.get(event["metadata"].get("proposal_id", ""))
        if record is not None and record["stage"] == "evaluated" and record["scores"]:
            record.update(stage=outcome, old_score=event["old_score"], new_score=event["new_score"])

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        """Record a completed accepted reflective cycle, excluding merges."""
        self._outcome(event, "accepted")

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        """Record an evaluated rejection as a successfully exercised optimizer cycle."""
        self._outcome(event, "rejected")

    def get_state(self) -> dict[str, Any]:
        """Use the core's durable callback snapshots to preserve completed evidence."""
        return {"contract_sha256": self.contract_sha256, "proposals": deepcopy(self.proposals)}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore evidence only for the exact pilot contract being resumed."""
        if state.get("contract_sha256") != self.contract_sha256:
            raise ValueError("Optimizer pilot checkpoint evidence has a different run contract")
        self.proposals = deepcopy(state["proposals"])

    def completion_evidence(self) -> dict[str, Any]:
        """Refuse success when no proposed candidate completed training evaluation and admission."""
        completed = sum(item["stage"] in {"accepted", "rejected"} for item in self.proposals.values())
        if not completed:
            raise RuntimeError(
                "Optimizer pilot completed no proposal/evaluation cycle; inspect the optimizer log. "
                "No pilot winner or success summary was written."
            )
        return {
            "schema_version": 1,
            "contract_sha256": self.contract_sha256,
            "completed_cycles": completed,
            "proposals": deepcopy(list(self.proposals.values())),
        }
