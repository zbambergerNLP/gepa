"""Keep evaluated edit outcomes local to the parent node that produced them."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from difflib import unified_diff
from typing import Any

PROPOSAL_MEMORY_VERSION = 1
PROPOSAL_HISTORY_WINDOW = 8
PROPOSAL_FAILURE_FREE_ATTEMPTS = 1
PROPOSAL_FAILURE_PENALTY = 0.25
PROPOSAL_MIN_WEIGHT_MULTIPLIER = 0.5
PROPOSAL_HISTORY_DIFF_CHARS = 1200
MEMORY_CONTRACT = {
    "version": PROPOSAL_MEMORY_VERSION,
    "history_window": PROPOSAL_HISTORY_WINDOW,
    "history_diff_chars": PROPOSAL_HISTORY_DIFF_CHARS,
    "scope": "parent_node_id/component; no inherited or cross-node history",
    "observations": "changed proposals evaluated against the parent on the same training minibatch only",
    "penalty_scope": "exact before text and action/section among recent node/component outcomes",
    "free_unsuccessful_attempts": PROPOSAL_FAILURE_FREE_ATTEMPTS,
    "penalty_strength": PROPOSAL_FAILURE_PENALTY,
    "minimum_weight_multiplier": PROPOSAL_MIN_WEIGHT_MULTIPLIER,
    "unsuccessful": "training ties and regressions only; no generation or provider failures",
    "success": "strict training mean improvement resets the matching failure count",
    "sampling": "multiply model weights before normalization and positive-support exploration",
}


class ProposalMemory:
    """Preserve evaluated edits without inheriting sibling restrictions or rewards."""

    def __init__(self) -> None:
        """Start with no evaluated proposals."""
        self._records: dict[str, dict[str, Any]] = {}

    def record_evaluation(
        self,
        parent_id: int,
        component: str,
        before: str,
        after: str,
        minibatch_ids: Sequence[str | int],
        action_pair: str,
        action_name: str,
        section: str,
        scores_before: Sequence[float],
        scores_after: Sequence[float],
        attempt_id: str,
        training_evidence: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record a changed edit scored against its parent on the same training batch.

        Repeated delivery of an identical attempt is idempotent. Generation errors,
        no-ops and duplicate blocks cannot enter this evaluated-outcome memory.

        Raises:
            ValueError: Identity, batch, scores or a repeated attempt conflict.
        """
        if type(parent_id) is not int or parent_id < 0:
            raise ValueError("Proposal memory requires a nonnegative parent node ID")
        if any(
            not isinstance(value, str) or not value
            for value in (component, action_pair, action_name, section, attempt_id)
        ):
            raise ValueError("Proposal memory requires nonempty string identities")
        if not isinstance(before, str) or not isinstance(after, str) or before == after:
            raise ValueError("Only changed, evaluated component text enters proposal memory")
        if (
            isinstance(minibatch_ids, str | bytes)
            or not minibatch_ids
            or any(type(value) not in (str, int) for value in minibatch_ids)
            or len(minibatch_ids) != len(scores_before)
            or len(minibatch_ids) != len(scores_after)
        ):
            raise ValueError("Proposal memory requires matched nonempty minibatch IDs and scores")
        if any(
            type(value) not in (float, int) or not math.isfinite(value) for value in (*scores_before, *scores_after)
        ):
            raise ValueError("Proposal memory requires finite numeric training scores")
        if training_evidence is not None and (
            not isinstance(training_evidence, Sequence)
            or isinstance(training_evidence, str | bytes)
            or any(not isinstance(entry, Mapping) for entry in training_evidence)
        ):
            raise ValueError("Proposal memory training evidence must be a sequence of records")
        before_scores, after_scores = list(map(float, scores_before)), list(map(float, scores_after))
        mean_before = math.fsum(before_scores) / len(before_scores)
        mean_after = math.fsum(after_scores) / len(after_scores)
        gain = mean_after - mean_before
        record = {
            "evaluated": True,
            "attempt_id": attempt_id,
            "parent_id": parent_id,
            "component": component,
            "before": before,
            "after": after,
            "minibatch_ids": list(minibatch_ids),
            "training_evidence": deepcopy([dict(entry) for entry in training_evidence or []]),
            "action_pair": action_pair,
            "action_name": action_name,
            "section": section,
            "scores_before": before_scores,
            "scores_after": after_scores,
            "training_mean_before": mean_before,
            "training_mean_after": mean_after,
            "training_gain": gain,
            "outcome": "improvement" if gain > 0 else "regression" if gain < 0 else "tie",
        }
        if attempt_id in self._records and self._records[attempt_id] != record:
            raise ValueError("Conflicting evaluated proposal for one attempt ID")
        self._records[attempt_id] = record
        return deepcopy(record)

    def matches(self, parent_id: int, component: str, before: str) -> list[dict[str, Any]]:
        """Return every evaluated edit from this node, component and exact parent text."""
        return deepcopy(
            [
                record
                for record in self._records.values()
                if record["parent_id"] == parent_id and record["component"] == component and record["before"] == before
            ]
        )

    def context(self, parent_id: int, component: str, before: str) -> dict[str, Any]:
        """Expose recent node-local outcomes, marking exact component-text matches."""
        recent = []
        for record in self._recent(parent_id, component):
            diff = "\n".join(
                unified_diff(
                    record["before"].splitlines(),
                    record["after"].splitlines(),
                    fromfile="parent",
                    tofile="proposal",
                    lineterm="",
                )
            )
            recent.append(
                {
                    **deepcopy(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in {"before", "after", "training_evidence"}
                        }
                    ),
                    "before_sha256": hashlib.sha256(record["before"].encode("utf-8")).hexdigest(),
                    "after_sha256": hashlib.sha256(record["after"].encode("utf-8")).hexdigest(),
                    "same_prompt": record["before"] == before,
                    "diff_excerpt": diff[:PROPOSAL_HISTORY_DIFF_CHARS],
                    "diff_truncated": len(diff) > PROPOSAL_HISTORY_DIFF_CHARS,
                }
            )
        return {"parent_id": parent_id, "component": component, "recent_attempts": recent}

    def _recent(self, parent_id: int, component: str) -> list[dict[str, Any]]:
        """Select recent node-local records without rendering potentially long edit diffs."""
        return [
            record
            for record in self._records.values()
            if record["parent_id"] == parent_id and record["component"] == component
        ][-PROPOSAL_HISTORY_WINDOW:]

    def multipliers(self, parent_id: int, component: str, before: str, pairs: Sequence[str]) -> dict[str, float]:
        """Gently discount repeated ties or regressions without removing any pair."""
        failures: Counter[str] = Counter()
        for record in self._recent(parent_id, component):
            if record["before"] != before or record["action_pair"] not in pairs:
                continue
            pair = record["action_pair"]
            if record["outcome"] == "improvement":
                failures[pair] = 0
            else:
                failures[pair] += 1
        return {
            pair: max(
                PROPOSAL_MIN_WEIGHT_MULTIPLIER,
                1 / (1 + PROPOSAL_FAILURE_PENALTY * (count - PROPOSAL_FAILURE_FREE_ATTEMPTS)),
            )
            for pair, count in failures.items()
            if count > PROPOSAL_FAILURE_FREE_ATTEMPTS
        }

    def get_state(self) -> dict[str, Any]:
        """Snapshot complete evaluation history in its deterministic observation order."""
        return {"version": PROPOSAL_MEMORY_VERSION, "records": deepcopy(list(self._records.values()))}

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore validated records atomically, rejecting changed outcomes or identities.

        Raises:
            ValueError: The version, shape, duplicate identities or derived outcomes differ.
        """
        if (
            set(state) != {"version", "records"}
            or type(state["version"]) is not int
            or state["version"] != PROPOSAL_MEMORY_VERSION
        ):
            raise ValueError("Proposal memory checkpoint version or shape mismatch")
        if not isinstance(state["records"], list):
            raise ValueError("Proposal memory checkpoint records must be a list")
        restored = ProposalMemory()
        fields = (
            "parent_id",
            "component",
            "before",
            "after",
            "minibatch_ids",
            "action_pair",
            "action_name",
            "section",
            "scores_before",
            "scores_after",
            "attempt_id",
            "training_evidence",
        )
        for record in state["records"]:
            try:
                if (
                    not isinstance(record, dict)
                    or record.get("evaluated") is not True
                    or record["attempt_id"] in restored._records
                    or any(
                        type(record[key]) not in (int, float) or not math.isfinite(record[key])
                        for key in ("training_mean_before", "training_mean_after", "training_gain")
                    )
                ):
                    raise ValueError("Invalid or duplicate proposal memory record")
                result = restored.record_evaluation(**{key: record[key] for key in fields})
                if result != record:
                    raise ValueError("Proposal memory checkpoint outcome or shape mismatch")
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("Malformed proposal memory checkpoint record") from exc
        self._records = restored._records
