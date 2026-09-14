"""Replay completed evaluation batches within an interrupted optimizer iteration."""

from __future__ import annotations

import base64
import hashlib
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gepa.response_journal import RESPONSE_JOURNAL_SCHEMA_VERSION, ResponseJournalError, ResumeResponseJournal

try:
    import cloudpickle
except ImportError:
    cloudpickle = pickle


class EvaluationJournal:
    """Preserve feedback used by journaled reflection without caching later iterations."""

    def __init__(self, run_dir: str, *, use_cloudpickle: bool = False):
        """Configure private, checksummed records using the checkpoint serializer."""
        self.path = Path(run_dir) / ".evaluation-journal" / "responses.sqlite3"
        self.serializer = cloudpickle if use_cloudpickle else pickle

    def evaluate(
        self,
        iteration: int,
        phase: str,
        items: list,
        adapter: Any,
        execute: Callable[[], list],
    ) -> list:
        """Restore a completed batch and adapter state, or execute and persist it.

        Args:
            iteration: Checkpoint-stable iteration number.
            phase: Parent reflection or child reevaluation stage.
            items: Exact ordered candidates and examples to evaluate.
            adapter: Evaluator whose optional state must advance on replay too.
            execute: Real batch evaluation, invoked only for an unrecorded slot.

        Returns:
            The original batch outputs, scores, and trajectories.

        Raises:
            ResponseJournalError: Persisted inputs or serialized output are invalid.
        """
        journal = ResumeResponseJournal(self.path, phase)
        scope = f"optimizer-iteration-{iteration}"
        request_hash = hashlib.sha256(self.serializer.dumps(items)).hexdigest()
        payload = journal.load(scope, 0, request_hash)
        if payload is not None:
            if payload.get("kind") != "evaluation_batch":
                raise ResponseJournalError("Evaluation journal contains an unsupported result.")
            try:
                results, adapter_state = self.serializer.loads(base64.b64decode(payload["data"], validate=True))
            except Exception as exc:
                raise ResponseJournalError("Evaluation journal contains invalid serialized data.") from exc
            if adapter_state is not None:
                adapter.set_adapter_state(adapter_state)
            return results
        results = execute()
        get_state = getattr(adapter, "get_adapter_state", None)
        adapter_state = get_state() if callable(get_state) else None
        data = base64.b64encode(self.serializer.dumps((results, adapter_state))).decode("ascii")
        journal.store(
            scope,
            0,
            request_hash,
            {"schema_version": RESPONSE_JOURNAL_SCHEMA_VERSION, "kind": "evaluation_batch", "data": data},
        )
        return results
