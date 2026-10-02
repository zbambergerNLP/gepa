"""Coordinate one GPU job from an internet-enabled host using shared files."""

import argparse
import fcntl
import os
import socket
import subprocess
import time
from pathlib import Path

from examples.hotpotqa.resolve_jev_handoff import resolve_saved_request
from gepa.response_journal import ResponseJournalError
from gepa.strategies.jev_constants import HANDOFF_REQUEST_FILE, HANDOFF_RESPONSE_FILE, HANDOFF_WAITING_FILE
from gepa.strategies.jev_handoff import HANDOFF_ENV, load, save

ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
COORDINATOR_STATUS_FILE = "coordinator.json"
COORDINATOR_LOCK_FILE = "coordinator.lock"
COORDINATOR_HEARTBEAT_MAX_AGE_SECONDS = 150
SLURM_QUERY_TIMEOUT_SECONDS = 20
SLURM_POLL_INTERVAL_SECONDS = 60
ACTIVE_MAILBOX_POLL_SECONDS = 5
IDLE_MAILBOX_POLL_SECONDS = 60
KEY_FILE_FORBIDDEN_PERMISSIONS = 0o077


def check_ready(directory: Path, job: str, source: str) -> None:
    """Require the matching live coordinator before loading GPU models.

    Args:
        directory: Shared mailbox containing the sealed coordinator status.
        job: Expected Slurm job ID.
        source: Expected frozen source commit.

    Raises:
        ResponseJournalError: The status checksum, identity, readiness or heartbeat
            age fails validation.
    """
    record = load(directory / COORDINATOR_STATUS_FILE)
    if record["job_id"] != job or record["source_commit"] != source or record["status"] != "ready":
        raise ResponseJournalError("Jev coordinator identity or status mismatch.")
    if not 0 <= time.time() - record["time_unix"] <= COORDINATOR_HEARTBEAT_MAX_AGE_SECONDS:
        raise ResponseJournalError("Jev coordinator heartbeat is stale.")


def resolve_pending(directory: Path, job: str, source: str, key: str) -> int:
    """Resolve only requests owned by this exact job and frozen source.

    Args:
        directory: Shared mailbox containing request subdirectories.
        job: Slurm job ID that must own each waiting marker.
        source: Frozen source commit that must match each request.
        key: TypeSafe credential used only by the external resolver.

    Returns:
        Number of newly resolved requests, excluding completed responses reused.

    Raises:
        ResponseJournalError: A sealed artifact or request identity is invalid,
            or a saved response records an external failure.
        FileExistsError: An unresolved started marker requires review.
        SystemExit: A new external request fails and its evidence is retained.
    """
    resolved = 0
    for waiting_path in sorted(directory.glob(f"*/{HANDOFF_WAITING_FILE}")):
        request_path = waiting_path.with_name(HANDOFF_REQUEST_FILE)
        waiting = load(waiting_path)
        request = load(request_path)
        if waiting["allocation"] != job or request["source_commit"] != source:
            raise ResponseJournalError("Jev mailbox request belongs to another job or source.")
        if waiting["request_sha256"] != request_path.parent.name:
            raise ResponseJournalError("Jev mailbox waiting marker disagrees with request identity.")
        if request_path.with_name(HANDOFF_RESPONSE_FILE).exists():
            response = load(request_path.with_name(HANDOFF_RESPONSE_FILE))
            if response["request_sha256"] != request_path.parent.name:
                raise ResponseJournalError("Jev mailbox response identity mismatch.")
            if response["error_type"] is not None:
                raise ResponseJournalError("External Jev failure retained; review before recovery.")
            continue
        resolve_saved_request(request_path, key)
        resolved += 1
    return resolved


def job_state(job: str) -> str:
    """Read the allocation state without submitting or changing a job.

    Args:
        job: Slurm job ID to query through ``sacct``.

    Returns:
        Scheduler state from the single row matching the requested job.

    Raises:
        RuntimeError: Accounting does not return exactly one matching job row.
        subprocess.CalledProcessError: The accounting command fails.
        subprocess.TimeoutExpired: The scheduler query exceeds its timeout.
    """
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", job, "-o", "JobIDRaw,State", "-P"],
        check=True,
        capture_output=True,
        text=True,
        timeout=SLURM_QUERY_TIMEOUT_SECONDS,
    )
    rows = [row.split("|") for row in result.stdout.splitlines() if row.split("|")[0] == job]
    if len(rows) != 1:
        raise RuntimeError("The coordinator cannot establish the job's Slurm state.")
    return rows[0][1].split()[0]


def serve(directory: Path, job: str, source: str, key_file: Path) -> None:
    """Serve one bounded job without running models or forwarding network traffic.

    Publish coordinator heartbeats and resolve waiting requests while the job is
    running. Stop on a terminal scheduler state; record and propagate failures.

    Args:
        directory: Shared mailbox to lock for this coordinator.
        job: Slurm job ID whose lifecycle bounds the service.
        source: Source commit that must match the local frozen-source marker.
        key_file: Nonempty TypeSafe credential file accessible only to its owner.

    Raises:
        ValueError: Execution context, frozen source or credential permissions
            violate the coordinator contract.
        BlockingIOError: Another coordinator already holds the mailbox lock.
        ResponseJournalError: Request or response evidence fails validation.
        RuntimeError: The scheduler cannot establish a unique job state.
    """
    if os.environ.get("SLURM_JOB_ID") or os.environ.get(HANDOFF_ENV):
        raise ValueError("The coordinator must run outside a compute allocation.")
    if source != (Path(__file__).resolve().parents[2] / ".gepa-source-commit").read_text().strip():
        raise ValueError("Coordinator source differs from the frozen job source.")
    key = key_file.read_text().strip()
    if not key or key_file.stat().st_mode & KEY_FILE_FORBIDDEN_PERMISSIONS:
        raise ValueError("The API key must be nonempty and readable only by its owner.")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / COORDINATOR_LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {"job_id": job, "source_commit": source, "pid": os.getpid(), "host": socket.gethostname()}
        try:
            state = "PENDING"
            next_status = 0.0
            while True:
                if time.monotonic() >= next_status:
                    state = job_state(job)
                    next_status = time.monotonic() + SLURM_POLL_INTERVAL_SECONDS
                    if state not in ACTIVE_STATES:
                        save(
                            directory / COORDINATOR_STATUS_FILE,
                            {**metadata, "status": "stopped", "slurm_state": state, "time_unix": time.time()},
                        )
                        return
                save(
                    directory / COORDINATOR_STATUS_FILE,
                    {**metadata, "status": "ready", "slurm_state": state, "time_unix": time.time()},
                )
                if state == "RUNNING":
                    resolve_pending(directory, job, source, key)
                time.sleep(ACTIVE_MAILBOX_POLL_SECONDS if state == "RUNNING" else IDLE_MAILBOX_POLL_SECONDS)
        except BaseException as exc:
            save(
                directory / COORDINATOR_STATUS_FILE,
                {**metadata, "status": "failed", "error_type": type(exc).__name__, "time_unix": time.time()},
            )
            raise


def main() -> None:
    """Check the coordinator or run it on an authorized internet-enabled host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "check-ready"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.mode == "check-ready":
        check_ready(args.directory, args.job, args.source)
    else:
        if args.key_file is None:
            parser.error("serve requires --key-file")
        serve(args.directory, args.job, args.source, args.key_file)


if __name__ == "__main__":
    main()
