# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Callback for tracking per-action statistics and proposal diversity.

When action-conditioned reflection is active, each proposal is tagged with the
action that constrained it (e.g. ``"add_constraint"``, ``"restructure"``).
This callback collects per-action counts, acceptance rates, score deltas,
and textual diversity metrics for analysis.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from typing import Any

from gepa.core.callbacks import (
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    ProposalEndEvent,
)


class ActionDiversityCallback:
    """Observational callback tracking per-action statistics and proposal diversity.

    Each event carries the proposal's metadata (``event["metadata"]``, populated
    by the proposer and threaded through ``CandidateProposal.metadata``), so
    accept/reject outcomes are attributed to the action recorded on the event
    itself, so event order does not affect attribution. The engine fires all
    rejections before acceptances within an iteration.

    Usage::

        tracker = ActionDiversityCallback()
        result = optimize(..., callbacks=[tracker], action_selector=...)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self.action_proposal_counts: dict[str, int] = defaultdict(int)
        self.action_acceptance_counts: dict[str, int] = defaultdict(int)
        self.action_rejection_counts: dict[str, int] = defaultdict(int)
        self.action_score_deltas: dict[str, list[float]] = defaultdict(list)

        self.action_texts: dict[str, list[str]] = defaultdict(list)

        self._iteration_texts: dict[int, list[str]] = defaultdict(list)
        self._current_iteration: int = -1

    @staticmethod
    def _action_from_event(event: Any) -> str | None:
        metadata = event.get("metadata") or {}
        return metadata.get("action")

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        """Record the proposed instruction and its action (if present).

        A length-capped attempt reaches here with empty ``new_instructions`` (#7):
        it still counts toward the action's proposal total, but contributes no
        text to the diversity metrics (an empty string would read as maximally
        dissimilar and inflate them).

        Args:
            event: The proposal-end event; its ``metadata["action"]`` names the
                action credited with the proposal.
        """
        self._current_iteration = event["iteration"]
        new_instructions = event["new_instructions"]

        action_name = self._action_from_event(event)
        if action_name:
            self.action_proposal_counts[action_name] += 1
            if new_instructions:
                self.action_texts[action_name].append(" ".join(new_instructions.values()))

        if new_instructions:
            self._iteration_texts[self._current_iteration].append(" ".join(new_instructions.values()))

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        """Attribute the acceptance and its score delta to the event's action.

        Accepted and rejected proposals both feed ``action_score_deltas``, so the
        field captures each action's full outcome rather than only its rejections.
        ``old_score`` is tolerated as optional for legacy/synthetic events.

        Args:
            event: The acceptance event; ``metadata["action"]`` names the action
                and ``new_score - old_score`` is the recorded delta.
        """
        action_name = self._action_from_event(event)
        if not action_name:
            return
        self.action_acceptance_counts[action_name] += 1
        old_score = event.get("old_score")
        if old_score is not None:
            self.action_score_deltas[action_name].append(event["new_score"] - old_score)

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        """Attribute the rejection to the action recorded on the event."""
        action_name = self._action_from_event(event)
        if action_name:
            self.action_rejection_counts[action_name] += 1
            self.action_score_deltas[action_name].append(event["new_score"] - event["old_score"])

    def textual_diversity(self) -> dict[str, float]:
        """Compute mean pairwise textual dissimilarity per iteration.

        Returns a dict mapping iteration number (as string) to the mean
        pairwise dissimilarity (1 - SequenceMatcher.ratio()) of sibling
        proposals within that iteration.  Higher values indicate more diverse
        proposals.
        """
        result: dict[str, float] = {}
        for iteration, texts in self._iteration_texts.items():
            if len(texts) < 2:
                continue
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
                    similarities.append(1.0 - ratio)
            result[str(iteration)] = sum(similarities) / len(similarities) if similarities else 0.0
        return result

    def summary(self) -> dict[str, Any]:
        """Return all accumulated metrics as a flat dict suitable for logging."""
        acceptance_rates: dict[str, float] = {}
        for action in self.action_proposal_counts:
            total = self.action_proposal_counts[action]
            accepted = self.action_acceptance_counts.get(action, 0)
            acceptance_rates[action] = accepted / total if total > 0 else 0.0

        return {
            "action_proposal_counts": dict(self.action_proposal_counts),
            "action_acceptance_counts": dict(self.action_acceptance_counts),
            "action_rejection_counts": dict(self.action_rejection_counts),
            "action_acceptance_rates": acceptance_rates,
            "action_score_deltas": {k: list(v) for k, v in self.action_score_deltas.items()},
            "textual_diversity_per_iteration": self.textual_diversity(),
            "total_proposals": sum(self.action_proposal_counts.values()),
            "total_accepted": sum(self.action_acceptance_counts.values()),
        }
