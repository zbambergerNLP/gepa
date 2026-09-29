"""Coordinate one GPU pilot from an internet-enabled host using shared files."""

import argparse
import fcntl
import os
import socket
import subprocess
import time
from pathlib import Path

from examples.hotpotqa.resolve_jev_handoff import resolve_saved_request
from gepa.response_journal import ResponseJournalError
from gepa.strategies.jev_handoff import HANDOFF_ENV, load, save

ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


def check_ready(directory: Path, job: str, source: str) -> None:
    """Require the matching live coordinator before loading GPU models."""
    record = load(directory / "coordinator.json")
    if record["job_id"] != job or record["source_commit"] != source or record["status"] != "ready":
        raise ResponseJournalError("Jev coordinator identity or status mismatch.")
    if not 0 <= time.time() - record["time_unix"] <= 150:
        raise ResponseJournalError("Jev coordinator heartbeat is stale.")


def resolve_pending(directory: Path, job: str, source: str, key: str) -> int:
    """Resolve only requests owned by this exact job and frozen source."""
    resolved = 0
    for waiting_path in sorted(directory.glob("*/waiting.json")):
        request_path = waiting_path.with_name("request.json")
        waiting = load(waiting_path)
        request = load(request_path)
        if waiting["allocation"] != job or request["source_commit"] != source:
            raise ResponseJournalError("Jev mailbox request belongs to another job or source.")
        if waiting["request_sha256"] != request_path.parent.name:
            raise ResponseJournalError("Jev mailbox waiting marker disagrees with request identity.")
        if request_path.with_name("response.json").exists():
            response = load(request_path.with_name("response.json"))
            if response["request_sha256"] != request_path.parent.name:
                raise ResponseJournalError("Jev mailbox response identity mismatch.")
            if response["error_type"] is not None:
                raise ResponseJournalError("External Jev failure retained; review before recovery.")
            continue
        resolve_saved_request(request_path, key)
        resolved += 1
    return resolved


def job_state(job: str) -> str:
    """Read the allocation state without submitting or changing a job."""
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", job, "-o", "JobIDRaw,State", "-P"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = [row.split("|") for row in result.stdout.splitlines() if row.split("|")[0] == job]
    if len(rows) != 1:
        raise RuntimeError("The coordinator cannot establish the pilot's Slurm state.")
    return rows[0][1].split()[0]


def serve(directory: Path, job: str, source: str, key_file: Path) -> None:
    """Serve one bounded pilot without running models or forwarding network traffic."""
    if os.environ.get("SLURM_JOB_ID") or os.environ.get(HANDOFF_ENV):
        raise ValueError("The coordinator must run outside a compute allocation.")
    if source != (Path(__file__).resolve().parents[2] / ".gepa-source-commit").read_text().strip():
        raise ValueError("Coordinator source differs from the frozen pilot source.")
    key = key_file.read_text().strip()
    if not key or key_file.stat().st_mode & 0o077:
        raise ValueError("The API key must be nonempty and readable only by its owner.")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {"job_id": job, "source_commit": source, "pid": os.getpid(), "host": socket.gethostname()}
        try:
            state = "PENDING"
            next_status = 0.0
            while True:
                if time.monotonic() >= next_status:
                    state = job_state(job)
                    next_status = time.monotonic() + 60
                    if state not in ACTIVE_STATES:
                        save(
                            directory / "coordinator.json",
                            {**metadata, "status": "stopped", "slurm_state": state, "time_unix": time.time()},
                        )
                        return
                save(
                    directory / "coordinator.json",
                    {**metadata, "status": "ready", "slurm_state": state, "time_unix": time.time()},
                )
                if state == "RUNNING":
                    resolve_pending(directory, job, source, key)
                time.sleep(5 if state == "RUNNING" else 60)
        except BaseException as exc:
            save(
                directory / "coordinator.json",
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
