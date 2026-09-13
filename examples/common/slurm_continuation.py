"""Advance a serial Slurm campaign or resume a verified allocation timeout."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from examples.common.recovery import _write, file_digest, snapshot


def run(command: list[str]) -> str:
    """Execute a scheduler command without inherited Slurm resource overrides."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SBATCH_", "SLURM_"))}
    return subprocess.run(command, check=True, capture_output=True, text=True, env=env).stdout.strip()


def job_id(output: str) -> str:
    """Reject ambiguous submission replies instead of submitting again."""
    result = output.split(";")[0]
    if not re.fullmatch(r"[0-9]+", result):
        raise RuntimeError(f"Ambiguous Slurm submission reply: {output!r}; inspect queued jobs before retrying")
    return result


def accounting(identifier: str) -> tuple[str, str]:
    """Read the allocation's final state, excluding individual job steps."""
    rows = run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            identifier,
            "--format=JobIDRaw,State,ExitCode",
        ]
    ).splitlines()
    matching = [row.split("|") for row in rows if row.split("|")[0] == identifier]
    if len(matching) != 1:
        raise RuntimeError("Allocation accounting is unavailable or ambiguous; no continuation was submitted")
    return matching[0][1].split()[0].rstrip("+"), matching[0][2]


def can_continue(before: dict[str, int], after: dict[str, int]) -> bool:
    """Require new durable work without losing any previously recorded work."""
    return (
        bool(after)
        and all(after.get(key, -1) >= value for key, value in before.items())
        and any(value > before.get(key, 0) for key, value in after.items())
    )


def prepare_export(path: Path, registry: Path, error_file: Path, source: str) -> None:
    """Bind generic worker environments to the controller's recovery evidence."""
    entries = path.read_bytes().split(b"\0")
    values = {}
    for entry in entries:
        if not entry:
            continue
        key, separator, value = entry.partition(b"=")
        if not separator or key in values:
            raise ValueError("Malformed or duplicate environment entry")
        values[key] = value
    required = {
        b"GEPA_SOURCE_COMMIT": source.encode(),
        b"GEPA_RECOVERY_REGISTRY": str(registry).encode(),
        b"GEPA_RECOVERY_ERROR_FILE": str(error_file).encode(),
    }
    if b"HOTPOTQA_SOURCE_COMMIT" in values and values[b"HOTPOTQA_SOURCE_COMMIT"] != source.encode():
        raise ValueError("Worker source differs from the continuation plan")
    for key, value in required.items():
        if key in values and values[key] != value:
            raise ValueError("Worker recovery settings differ from the continuation plan")
        values[key] = value
    with path.open("wb") as stream:
        stream.write(b"".join(key + b"=" + value + b"\0" for key, value in values.items()))
        stream.flush()
        os.fsync(stream.fileno())


def dispatch(path: Path, plan: dict) -> None:
    """Hold the worker until its dependent continuation controller is saved."""
    cell = plan["cells"][plan["index"]]
    if file_digest(Path(cell["export_file"])) != cell["export_sha256"]:
        raise ValueError("Submission environment changed; refusing to alter the resumed runtime")
    before = snapshot(Path(cell["registry"]), plan["source_commit"])
    plan["status"] = "submitting"
    plan["before"] = before
    _write(path, plan)
    command = cell["command"]
    worker = job_id(run([command[0], "--hold", *command[1:]]))
    plan["worker"] = worker
    _write(path, plan)
    source = Path(plan["source_dir"])
    wrapped = shlex.join(
        [
            "env",
            f"PYTHONPATH={source / 'src'}:{source}",
            sys.executable,
            "-m",
            "examples.common.slurm_continuation",
            "advance",
            "--plan",
            str(path),
            "--job-id",
            worker,
        ]
    )
    controller = job_id(
        run(
            [
                command[0],
                "--parsable",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--mem=2G",
                "--time=00:10:00",
                "--export=NONE",
                f"--dependency=afterany:{worker}",
                f"--chdir={source}",
                "--job-name=gepa-continuation",
                f"--output={path.parent}/controller-%j.log",
                "--wrap",
                wrapped,
            ]
        )
    )
    plan.update(status="active", controller=controller)
    _write(path, plan)
    run(["scontrol", "release", worker])
    print(f"Submitted {cell['name']}: worker {worker}, continuation controller {controller}")


def advance(path: Path, plan: dict, identifier: str) -> None:
    """Advance only the current worker and stop on every non-timeout failure."""
    if plan["status"] != "active" or plan.get("worker") != identifier:
        print("Stale or duplicate continuation controller; no jobs submitted")
        return
    state, exit_code = accounting(identifier)
    cell = plan["cells"][plan["index"]]
    plan.setdefault("history", []).append(
        {"job_id": identifier, "cell": cell["name"], "state": state, "exit_code": exit_code}
    )
    if Path(cell["error_file"]).exists():
        raise RuntimeError("The logical run reported an unresolved execution error")
    if state == "COMPLETED" and exit_code == "0:0":
        plan["index"] += 1
        if plan["index"] == len(plan["cells"]):
            plan["status"] = "complete"
            _write(path, plan)
            print("All campaign cells completed successfully")
            return
    elif state == "TIMEOUT":
        after = snapshot(Path(cell["registry"]), plan["source_commit"])
        if not can_continue(plan["before"], after):
            raise RuntimeError("Allocation expired without new verified saved work; continuation stopped")
        plan["history"][-1]["saved_progress"] = after
    else:
        raise RuntimeError(f"Allocation ended as {state} ({exit_code}); continuation stopped")
    dispatch(path, plan)


def main(argv: list[str] | None = None) -> None:
    """Build a pinned plan, then submit or advance it under an exclusive lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("add", "start", "advance"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--name")
    parser.add_argument("--export-file", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--error-file", type=Path)
    parser.add_argument("--job-id")
    args, command = parser.parse_known_args(argv)
    path = args.plan.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = json.loads(path.read_text()) if path.exists() else None
        if args.action == "add":
            if (
                not args.name
                or not args.export_file
                or not args.registry
                or not args.error_file
                or not args.source_commit
            ):
                parser.error("add requires name, source, export file, registry, and error file")
            if not command or command[0] != "--" or len(command) < 3:
                parser.error("Pass the complete sbatch command after --")
            if plan is None:
                plan = {
                    "status": "draft",
                    "source_commit": args.source_commit,
                    "source_dir": str(Path.cwd()),
                    "cells": [],
                    "index": 0,
                }
            if (
                plan["status"] != "draft"
                or plan["source_commit"] != args.source_commit
                or any(cell["name"] == args.name for cell in plan["cells"])
            ):
                raise ValueError("Plan already exists or source changed; do not duplicate a campaign")
            prepare_export(args.export_file, args.registry, args.error_file, args.source_commit)
            plan["cells"].append(
                {
                    "name": args.name,
                    "command": command[1:],
                    "export_file": str(args.export_file),
                    "export_sha256": file_digest(args.export_file),
                    "registry": str(args.registry),
                    "error_file": str(args.error_file),
                }
            )
            _write(path, plan)
            return
        if plan is None:
            raise ValueError("No campaign plan exists")
        marker = Path(plan["source_dir"]) / ".gepa-source-commit"
        if not marker.exists() or marker.read_text().strip() != plan["source_commit"]:
            raise ValueError("Pinned source marker changed; refusing to submit")
        if args.action == "start" and (plan["status"] != "draft" or not plan["cells"]):
            raise ValueError("Campaign has already been submitted or has no cells")
        try:
            if args.action == "start":
                dispatch(path, plan)
            else:
                if not args.job_id:
                    parser.error("advance requires --job-id")
                advance(path, plan, args.job_id)
        except Exception as exc:
            # A held worker may exist after an ambiguous submission; keep its
            # ID and never retry automatically or release it without a watcher.
            plan.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            _write(path, plan)
            raise


if __name__ == "__main__":
    main()
