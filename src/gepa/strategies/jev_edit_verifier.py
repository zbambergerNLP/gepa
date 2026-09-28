"""Classify sibling edit redundancy with a journaled, deterministic Jev choice."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from gepa.response_journal import ACTIVE_RESPONSE_JOURNAL_SCOPE, ResponseJournalError
from gepa.strategies.jev_controller import JEV_CONTROLLER_POLICY_CONTRACT, JevController

JEV_DUPLICATE_THRESHOLD = 0.8
JEV_EDIT_VERIFIER_CONTRACT = {
    "identity": "jev-sibling-edit-verifier-v1",
    "model": JEV_CONTROLLER_POLICY_CONTRACT["model"],
    "api_base": JEV_CONTROLLER_POLICY_CONTRACT["api_base"],
    "sdk_version": JEV_CONTROLLER_POLICY_CONTRACT["sdk_version"],
    "primitive": "choice",
    "context": "full proposed edit, canonical parent and supplied same-parent/component history",
    "choices": "distinct, uncertain, and one duplicate alternative per eligible prior attempt",
    "selection": "deterministic provider argmax; no sampling or exploration",
    "duplicate_threshold": JEV_DUPLICATE_THRESHOLD,
    "threshold_interpretation": "conservative engineering heuristic, not calibrated confidence",
    "below_duplicate_threshold": "uncertain; allow evaluation",
    "probability_normalization": deepcopy(JEV_CONTROLLER_POLICY_CONTRACT["probability_normalization"]),
    "retry": deepcopy(JEV_CONTROLLER_POLICY_CONTRACT["retry"]),
    "invalid_distribution": "fail closed without generative fallback or semantic retry",
    "reason": "mechanical attribution of typed classification; no generated model rationale",
    "journal": "durable response and physical attempt journal required before returning a verdict",
}


class JevEditVerifier(JevController):
    """Use the shared Jev transport without sharing the Controller's selection policy."""

    JOURNAL_NAMESPACE = "jev-edit-verifier"
    ROLE = "edit_verifier"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        response_journal_path: str | Path | None = None,
        attempt_log_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            response_journal_path=response_journal_path,
            attempt_log_path=attempt_log_path,
        )
        self.last_evidence: dict[str, Any] | None = None

    def run_contract(self) -> dict[str, Any]:
        """Return the verifier identity independently of Controller configuration."""
        return deepcopy(JEV_EDIT_VERIFIER_CONTRACT)

    def classify_edit(self, evidence: dict[str, Any], instructions: str) -> str:
        """Classify redundancy against identifiable, eligible prior attempts.

        Args:
            evidence: Complete comparison payload with a ``previous_records`` list.
                Each record has a unique ``attempt_id`` and an explicit
                ``duplicate_judgment_allowed`` flag supplied by the novelty gate.
            instructions: The same canonical semantic rules as the generative verifier.

        Returns:
            A JSON object with verdict, matched_attempt_id and an attributed reason.

        Raises:
            ValueError: Prior attempt identities are missing, ambiguous or excessive.
            ResponseJournalError: The caller has not bound durable paths and a scope.
        """
        self.last_evidence = None
        records = evidence.get("previous_records")
        if not isinstance(records, list) or not records or len(records) > 253:
            raise ValueError("Jev edit verification requires 1..253 identifiable prior records.")
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError("Every prior edit must be an identifiable record.")
        ids = [record.get("attempt_id") for record in records]
        if any(not isinstance(attempt_id, str) or not attempt_id for attempt_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Prior edit attempt identities must be nonempty and unique.")
        criteria: dict[str, Any] = {
            "distinct": "The proposed operational change is meaningfully different from all supplied prior edits.",
            "uncertain": "Available edit semantics or training evidence do not establish either redundancy or novelty.",
        }
        matched_ids: dict[str, str] = {}
        for index, record in enumerate(records):
            if record.get("duplicate_judgment_allowed") is not True:
                continue
            choice = f"duplicate_{index}"
            matched_ids[choice] = record["attempt_id"]
            criteria[choice] = {
                "description": "The proposed edit repeats this prior operational change without a meaningful new experiment.",
                "attempt_id": record["attempt_id"],
            }
        if not matched_ids:
            result = {
                "verdict": "uncertain",
                "matched_attempt_id": None,
                "reason": "No supplied prior attempt has sufficient training evidence for a duplicate judgment.",
            }
            self.last_evidence = {"model_called": False, "decision": result}
            return json.dumps(result)
        if self._journal is None or self._attempt_log is None or ACTIVE_RESPONSE_JOURNAL_SCOPE.get() is None:
            raise ResponseJournalError("Bind Jev verifier response/attempt journals and a logical scope before use.")
        payload = self.request_choice(
            state=evidence,
            instructions=instructions
            + "\nUse the typed alternatives instead of emitting a JSON response. A duplicate alternative refers only "
            "to its named prior attempt. Choose uncertain when the evidence is ambiguous. Do not infer a duplicate "
            "from lexical overlap or the action label alone.",
            criteria=criteria,
        )
        choice = payload["response"]["answers"]["edit"]["choice"]
        probability = payload["probs"][choice]
        if choice in matched_ids and probability >= JEV_DUPLICATE_THRESHOLD:
            result = {
                "verdict": "duplicate",
                "matched_attempt_id": matched_ids[choice],
                "reason": f"Jev classified the edit as duplicate of supplied attempt {matched_ids[choice]}.",
            }
        elif choice == "distinct":
            result = {
                "verdict": "distinct",
                "matched_attempt_id": None,
                "reason": "Jev classified the edit as distinct from the supplied prior attempts.",
            }
        else:
            result = {
                "verdict": "uncertain",
                "matched_attempt_id": None,
                "reason": "Jev selected a duplicate alternative below the configured rejection threshold."
                if choice in matched_ids
                else "Jev classified the edit as uncertain.",
            }
        self.last_evidence = {
            **payload,
            "model_called": True,
            "selected_choice": choice,
            "selected_probability": probability,
            "duplicate_threshold": JEV_DUPLICATE_THRESHOLD,
            "calibrated_confidence": False,
            "decision": deepcopy(result),
        }
        return json.dumps(result, ensure_ascii=False)
