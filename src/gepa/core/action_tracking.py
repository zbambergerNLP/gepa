# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Callback for tracking per-action statistics and proposal diversity.

When action-conditioned reflection is active, each proposal is tagged with the
semantic action that constrained it (e.g. ``"contextualize"``, ``"resequence"``).
This callback collects per-action counts, acceptance rates, score deltas,
and textual diversity metrics for analysis.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
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

    def __init__(self, selector: Any | None = None) -> None:
        """Initialize per-action, per-iteration, and outcome metrics.

        Args:
            selector: Optional action selector whose verbalized distribution
                history must survive optimizer resume with these metrics.
        """
        self.action_proposal_counts: dict[str, int] = defaultdict(int)
        self.action_acceptance_counts: dict[str, int] = defaultdict(int)
        self.action_rejection_counts: dict[str, int] = defaultdict(int)
        self.action_score_deltas: dict[str, list[float]] = defaultdict(list)

        self.action_texts: dict[str, list[str]] = defaultdict(list)

        self._iteration_texts: dict[int, list[str]] = defaultdict(list)
        self._current_iteration: int = -1
        self.selector = selector

    def get_state(self) -> dict[str, Any]:
        """Return a durable snapshot of all accumulated mechanism evidence.

        Returns:
            Counts, score deltas, proposal texts, per-iteration texts, current
            iteration, and optional verbalized-selector history.
        """
        selector_history = None
        if self.selector is not None and hasattr(self.selector, "history"):
            selector_history = deepcopy(self.selector.history)
        return {
            "action_proposal_counts": dict(self.action_proposal_counts),
            "action_acceptance_counts": dict(self.action_acceptance_counts),
            "action_rejection_counts": dict(self.action_rejection_counts),
            "action_score_deltas": {key: list(values) for key, values in self.action_score_deltas.items()},
            "action_texts": {key: list(values) for key, values in self.action_texts.items()},
            "iteration_texts": {key: list(values) for key, values in self._iteration_texts.items()},
            "current_iteration": self._current_iteration,
            "selector_history": selector_history,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore mechanism evidence from a durable optimizer checkpoint.

        Args:
            state: Snapshot previously returned by :meth:`get_state`.

        Raises:
            TypeError: A persisted collection or selector history has an
                incompatible type.
        """
        mapping_fields = {
            "action_proposal_counts",
            "action_acceptance_counts",
            "action_rejection_counts",
            "action_score_deltas",
            "action_texts",
            "iteration_texts",
        }
        for field_name in mapping_fields:
            if not isinstance(state.get(field_name, {}), Mapping):
                raise TypeError(f"Persisted {field_name} must be a mapping")

        self.action_proposal_counts = defaultdict(
            int,
            {str(key): int(value) for key, value in state.get("action_proposal_counts", {}).items()},
        )
        self.action_acceptance_counts = defaultdict(
            int,
            {str(key): int(value) for key, value in state.get("action_acceptance_counts", {}).items()},
        )
        self.action_rejection_counts = defaultdict(
            int,
            {str(key): int(value) for key, value in state.get("action_rejection_counts", {}).items()},
        )
        self.action_score_deltas = defaultdict(
            list,
            {
                str(key): [float(value) for value in values]
                for key, values in state.get("action_score_deltas", {}).items()
            },
        )
        self.action_texts = defaultdict(
            list,
            {str(key): [str(value) for value in values] for key, values in state.get("action_texts", {}).items()},
        )
        self._iteration_texts = defaultdict(
            list,
            {
                int(key): [str(value) for value in values]
                for key, values in state.get("iteration_texts", {}).items()
            },
        )
        self._current_iteration = int(state.get("current_iteration", -1))

        selector_history = state.get("selector_history")
        if selector_history is not None:
            if not isinstance(selector_history, list):
                raise TypeError("Persisted selector_history must be a list or None")
            if self.selector is not None and hasattr(self.selector, "history"):
                self.selector.history = deepcopy(selector_history)

    @staticmethod
    def _action_from_event(event: Any) -> str | None:
        """Read an optional semantic-action label from callback metadata.

        Args:
            event: Mapping-like callback event carrying optional metadata.

        Returns:
            Recorded action name, or ``None`` when the event is unconditioned.
        """
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
        """Attribute a rejection and score delta to its recorded action.

        Args:
            event: Rejection event whose metadata identifies the action and
                whose old and new scores define the recorded delta.
        """
        action_name = self._action_from_event(event)
        if action_name:
            self.action_rejection_counts[action_name] += 1
            self.action_score_deltas[action_name].append(event["new_score"] - event["old_score"])

    def textual_diversity(self) -> dict[str, float]:
        """Compute mean pairwise textual dissimilarity per iteration.

        Returns:
            Iteration strings mapped to mean pairwise dissimilarity
            (``1 - SequenceMatcher.ratio()``) among sibling proposals. An
            iteration with fewer than two non-empty proposals is omitted.
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
        """Return all accumulated action and diversity metrics.

        Returns:
            Flat logging payload with counts, acceptance rates, score deltas,
            per-iteration textual diversity, and aggregate proposal totals.
        """
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
