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

    The callback correlates ``on_proposal_end`` events (which carry the proposed
    instruction text) with ``on_candidate_accepted`` / ``on_candidate_rejected``
    events (which carry scores) using a per-iteration queue.  This works because
    the engine fires proposal_end, then accepted/rejected for each proposal in
    order within an iteration.

    Usage::

        tracker = ActionDiversityCallback()
        result = optimize(..., callbacks=[tracker], action_selector=...)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        # Per-action counters.
        self.action_proposal_counts: dict[str, int] = defaultdict(int)
        self.action_acceptance_counts: dict[str, int] = defaultdict(int)
        self.action_rejection_counts: dict[str, int] = defaultdict(int)
        self.action_score_deltas: dict[str, list[float]] = defaultdict(list)

        # Proposed texts per action (for textual diversity).
        self.action_texts: dict[str, list[str]] = defaultdict(list)

        # Queue of (action_name, new_instructions) for correlating with accept/reject.
        self._pending_proposals: list[tuple[str, dict[str, str]]] = []

        # Per-iteration sibling texts for diversity measurement.
        self._iteration_texts: dict[int, list[str]] = defaultdict(list)
        self._current_iteration: int = -1

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        """Record the proposed instruction and its action (if present)."""
        self._current_iteration = event["iteration"]
        new_instructions = event["new_instructions"]

        # Extract action name from the prompt suffix (the action constraint
        # block is appended after the template).  The action name is stored in
        # ReflectionProposal.metadata["action"] and threaded into
        # CandidateProposal.metadata, but ProposalEndEvent only carries prompts
        # and raw_lm_outputs.  We detect the action by scanning for the
        # "--- ACTION CONSTRAINT ---" marker in the rendered prompt.
        action_name = self._extract_action_from_prompts(event.get("prompts", {}))

        if action_name:
            self.action_proposal_counts[action_name] += 1
            # Store the concatenated instruction text for diversity analysis.
            text = " ".join(new_instructions.values())
            self.action_texts[action_name].append(text)
            self._pending_proposals.append((action_name, new_instructions))
        else:
            self._pending_proposals.append(("_unconditioned", new_instructions))

        # Collect all sibling texts within this iteration for diversity.
        text = " ".join(new_instructions.values())
        self._iteration_texts[self._current_iteration].append(text)

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        """Correlate acceptance with the pending proposal's action."""
        if not self._pending_proposals:
            return
        action_name, _ = self._pending_proposals.pop(0)
        if action_name != "_unconditioned":
            self.action_acceptance_counts[action_name] += 1

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        """Correlate rejection with the pending proposal's action."""
        if not self._pending_proposals:
            return
        action_name, _ = self._pending_proposals.pop(0)
        if action_name != "_unconditioned":
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
        # Per-action acceptance rates.
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

    @staticmethod
    def _extract_action_from_prompts(prompts: dict[str, Any]) -> str | None:
        """Extract action name from rendered prompts by looking for the constraint marker."""
        for _comp, prompt in prompts.items():
            text = ""
            if isinstance(prompt, str):
                text = prompt
            elif isinstance(prompt, list):
                # Multimodal messages list.
                for msg in prompt:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text += content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text += part.get("text", "")

            marker = "--- ACTION CONSTRAINT ---"
            idx = text.find(marker)
            if idx != -1:
                # Parse "You MUST make exactly one type of edit: <action_name>"
                after_marker = text[idx + len(marker) :]
                for line in after_marker.split("\n"):
                    line = line.strip()
                    if line.startswith("You MUST make exactly one type of edit:"):
                        return line.split(":")[-1].strip()
        return None
