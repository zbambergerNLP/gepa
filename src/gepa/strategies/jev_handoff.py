"""Stage typed requests as files when an allocation has no external network."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.response_journal import ACTIVE_RESPONSE_JOURNAL_SCOPE, ResponseJournalError, canonical_request_digest
from gepa.strategies.jev_constants import (
    HANDOFF_ATTEMPT_LOG,
    HANDOFF_ENV,
    HANDOFF_EXIT_CODE,
    HANDOFF_POLL_SECONDS,
    HANDOFF_REQUEST_FILE,
    HANDOFF_RESPONSE_FILE,
    HANDOFF_SCHEMA_VERSION,
    HANDOFF_STARTED_FILE,
    HANDOFF_WAIT_SECONDS,
    HANDOFF_WAITING_FILE,
    JEV_PRIVATE_FILE_MODE,
    JEV_QUESTION_NAME,
)

if TYPE_CHECKING:
    from gepa.strategies.jev_controller import JevController


def save(path: Path, record: dict[str, Any]) -> None:
    """Seal an artifact before exposing it to the other execution stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, JEV_PRIVATE_FILE_MODE)
        json.dump({"record": record, "sha256": canonical_request_digest(record)}, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    """Reject an incomplete or modified handoff artifact."""
    saved = json.loads(path.read_text())
    if canonical_request_digest(saved["record"]) != saved["sha256"]:
        raise ResponseJournalError(f"Jev handoff checksum mismatch: {path.name}")
    return saved["record"]


def exchange(controller: JevController, request: dict[str, Any]) -> dict[str, Any]:
    """Wait for an external result while retaining the allocated model servers."""
    scope = ACTIVE_RESPONSE_JOURNAL_SCOPE.get()
    if not scope or controller._journal is None or controller._attempt_log is None:
        raise ResponseJournalError("Offline Jev requires durable response/attempt journals and a logical scope.")
    record = {
        "schema": HANDOFF_SCHEMA_VERSION,
        "policy": controller.run_contract(),
        "namespace": controller.JOURNAL_NAMESPACE,
        "role": controller.ROLE,
        "scope": scope,
        "ordinal": controller._ordinals.get(scope, 0),
        "request": request,
        "source_commit": os.environ.get("HOTPOTQA_SOURCE_COMMIT"),
    }
    key = canonical_request_digest(record)
    directory = Path(os.environ[HANDOFF_ENV]) / key
    request_path = directory / HANDOFF_REQUEST_FILE
    if request_path.exists():
        if load(request_path) != record:
            raise ResponseJournalError("Jev handoff identity changed.")
    else:
        save(request_path, record)
    response_path = directory / HANDOFF_RESPONSE_FILE
    if not response_path.exists():
        save(directory / HANDOFF_WAITING_FILE, {"request_sha256": key, "allocation": os.environ.get("SLURM_JOB_ID")})
        print(f"JEV_HANDOFF_PENDING={request_path}", flush=True)
        deadline = time.monotonic() + HANDOFF_WAIT_SECONDS
        while not response_path.exists():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"JEV_HANDOFF_TIMEOUT={request_path}", flush=True)
                raise SystemExit(HANDOFF_EXIT_CODE)
            time.sleep(min(HANDOFF_POLL_SECONDS, remaining))
    response = load(response_path)
    if response["request_sha256"] != key:
        raise ResponseJournalError("Jev handoff response belongs to another request.")
    payload = response.get("payload")
    if payload is not None:
        controller._validate(payload["response"], set(request["questions"][JEV_QUESTION_NAME]["criteria"]))
    existing = {}
    if controller._attempt_log.exists():
        for line in controller._attempt_log.read_text().splitlines():
            row = json.loads(line)
            existing[(row["request_id"], row["attempt"], row["event"])] = row
    for row in response["attempts"]:
        identity = (row["request_id"], row["attempt"], row["event"])
        if identity in existing:
            if existing[identity] != row:
                raise ResponseJournalError("Jev external attempt disagrees with its imported record.")
            continue
        controller._log(row)
        if row["event"] == "finished" and row.get("usage") is not None:
            controller._charge(row["usage"])
        existing[identity] = row
    if payload is None:
        raise ResponseJournalError("External Jev request failed; inspect its preserved response and attempts.")
    return payload


def resolve(request_path: Path, controller: JevController) -> Path:
    """Execute a saved request off-cluster, retaining the native retry allowance."""
    if os.environ.get("SLURM_JOB_ID") or os.environ.get(HANDOFF_ENV):
        raise ValueError("Resolve Jev requests outside a Slurm allocation and outside offline transport mode.")
    request = load(request_path)
    key = canonical_request_digest(request)
    if request_path.parent.name != key or request["policy"] != controller.run_contract():
        raise ResponseJournalError("External resolver policy or request identity mismatch.")
    if request["namespace"] != controller.JOURNAL_NAMESPACE or request["role"] != controller.ROLE:
        raise ResponseJournalError("External resolver uses the wrong Jev role.")
    response_path = request_path.with_name(HANDOFF_RESPONSE_FILE)
    if response_path.exists():
        if load(response_path)["request_sha256"] != key:
            raise ResponseJournalError("Existing external response identity mismatch.")
        return response_path
    # A crash after a remote request can leave an unknown outcome. Never reroll it silently.
    with request_path.with_name(HANDOFF_STARTED_FILE).open("x") as stream:
        json.dump({"request_sha256": key, "time_unix": time.time()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    controller._attempt_log = request_path.with_name(HANDOFF_ATTEMPT_LOG)
    if controller._attempt_log.exists():
        raise ResponseJournalError("External attempt log already exists without a result; review before recovery.")
    token = ACTIVE_RESPONSE_JOURNAL_SCOPE.set(request["scope"])
    payload = None
    error = None
    try:
        payload = controller._live(
            request["request"], set(request["request"]["questions"][JEV_QUESTION_NAME]["criteria"])
        )
    except Exception as exc:
        error = type(exc).__name__
    finally:
        ACTIVE_RESPONSE_JOURNAL_SCOPE.reset(token)
        controller.close()
    attempts = (
        [json.loads(line) for line in controller._attempt_log.read_text().splitlines()]
        if controller._attempt_log.exists()
        else []
    )
    save(response_path, {"request_sha256": key, "payload": payload, "attempts": attempts, "error_type": error})
    return response_path
