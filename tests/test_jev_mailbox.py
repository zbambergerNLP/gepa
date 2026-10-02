"""Keep a resident GPU job bound to one coordinator, job and frozen source."""

import time

import pytest

from examples.hotpotqa import jev_mailbox
from gepa.response_journal import ResponseJournalError, canonical_request_digest
from gepa.strategies.jev_handoff import load, save


@pytest.mark.parametrize(
    "change", [{"job_id": "other"}, {"source_commit": "other"}, {"status": "failed"}, {"time_unix": 0}]
)
def test_preflight_rejects_wrong_or_stale_coordinator(tmp_path, change):
    record = {"job_id": "123", "source_commit": "source", "status": "ready", "time_unix": time.time()}
    save(tmp_path / "coordinator.json", record)
    jev_mailbox.check_ready(tmp_path, "123", "source")
    save(tmp_path / "coordinator.json", {**record, **change})
    with pytest.raises(ResponseJournalError):
        jev_mailbox.check_ready(tmp_path, "123", "source")


def pending(directory):
    request = {"source_commit": "source", "request": {"original": True}}
    key = canonical_request_digest(request)
    path = directory / key / "request.json"
    save(path, request)
    save(path.with_name("waiting.json"), {"allocation": "123", "request_sha256": key})
    return path


def test_coordinator_resolves_once_without_scheduler_mutations(tmp_path, monkeypatch):
    path = pending(tmp_path)
    calls = []

    def resolve(request_path, key):
        calls.append((request_path, key))
        save(request_path.with_name("response.json"), {"request_sha256": path.parent.name, "error_type": None})

    monkeypatch.setattr(jev_mailbox, "resolve_saved_request", resolve)
    assert jev_mailbox.resolve_pending(tmp_path, "123", "source", "private") == 1
    assert jev_mailbox.resolve_pending(tmp_path, "123", "source", "private") == 0
    assert calls == [(path, "private")]


@pytest.mark.parametrize("field,value", [("allocation", "other"), ("request_sha256", "wrong")])
def test_coordinator_never_resolves_wrong_waiting_identity(tmp_path, monkeypatch, field, value):
    path = pending(tmp_path)
    marker = path.with_name("waiting.json")
    save(marker, {**load(marker), field: value})
    monkeypatch.setattr(jev_mailbox, "resolve_saved_request", lambda *_: pytest.fail("API must not be called"))
    with pytest.raises(ResponseJournalError):
        jev_mailbox.resolve_pending(tmp_path, "123", "source", "private")


def test_coordinator_preserves_failure_without_another_call(tmp_path, monkeypatch):
    path = pending(tmp_path)
    save(path.with_name("response.json"), {"request_sha256": path.parent.name, "error_type": "JevControllerError"})
    monkeypatch.setattr(jev_mailbox, "resolve_saved_request", lambda *_: pytest.fail("API must not be called"))
    with pytest.raises(ResponseJournalError, match="failure retained"):
        jev_mailbox.resolve_pending(tmp_path, "123", "source", "private")
