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


def run(command: list[str], *, cwd: str | None = None) -> str:
    """Execute a scheduler command without inherited Slurm resource overrides."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SBATCH_", "SLURM_"))}
    return subprocess.run(command, check=True, capture_output=True, text=True, env=env, cwd=cwd).stdout.strip()


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
    if "source_dir" in cell:
        verify_source(cell["source_dir"], cell["source_commit"])
    before = snapshot(Path(cell["registry"]), cell.get("source_commit", plan["source_commit"]))
    plan["status"] = "submitting"
    plan["before"] = before
    _write(path, plan)
    command = cell["command"]
    worker = job_id(run([command[0], "--hold", *command[1:]], cwd=cell.get("source_dir", plan["source_dir"])))
    plan["worker"] = worker
    _write(path, plan)
    controller = submit_controller(path, plan, worker)
    plan.update(status="active", controller=controller)
    _write(path, plan)
    run(["scontrol", "release", worker])
    print(f"Submitted {cell['name']}: worker {worker}, continuation controller {controller}")


def verify_source(directory: str, commit: str) -> None:
    """Reject missing or changed immutable source identities."""
    marker = Path(directory) / ".gepa-source-commit"
    if not marker.exists() or marker.read_text().strip() != commit:
        raise ValueError("Pinned source marker changed; refusing to submit")


def submit_controller(path: Path, plan: dict, worker: str, *, held: bool = False) -> str:
    """Submit one watcher from the explicitly pinned controller revision."""
    source = Path(plan.get("controller_source_dir", plan["source_dir"]))
    if "controller_source_dir" in plan:
        verify_source(str(source), plan["controller_source_commit"])
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
    return job_id(
        run(
            [
                plan["cells"][plan["index"]]["command"][0],
                *(["--hold"] if held else []),
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


def extend(path: Path, plan: dict, extension: dict, after: str, worker: str, controller: str) -> None:
    """Insert a reviewed future cell while replacing only the pending watcher."""
    if plan.get("status") != "active" or plan.get("worker") != worker or plan.get("controller") != controller:
        raise ValueError("The active campaign changed; inspect it before extending")
    if extension.get("status") != "draft" or len(extension.get("cells", [])) != 1:
        raise ValueError("An extension must be an unsubmitted single-cell plan")
    cell = dict(extension["cells"][0])
    names = [existing["name"] for existing in plan["cells"]]
    if cell["name"] in names or after not in names or names.index(after) <= plan["index"]:
        raise ValueError("An extension may only insert a new cell after an unstarted future cell")
    verify_source(extension["source_dir"], extension["source_commit"])
    if Path(extension["source_dir"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Run the extension controller from the new immutable source")
    if file_digest(Path(cell["export_file"])) != cell["export_sha256"]:
        raise ValueError("Extension environment changed")
    if snapshot(Path(cell["registry"]), extension["source_commit"]) or Path(cell["error_file"]).exists():
        raise ValueError("The added cell already has recovery evidence")
    if accounting(worker)[0] not in {"PENDING", "RUNNING"} or accounting(controller)[0] != "PENDING":
        raise ValueError("Extend only while the worker is live and its watcher is pending")
    backup = path.with_name(f"plan-before-{cell['name']}.json")
    if backup.exists():
        raise ValueError("An extension backup already exists; inspect the previous attempt")
    _write(backup, plan)
    # Holding a pending watcher closes the completion race; never signal the
    # active worker. Persist partial transitions so uncertain submits cannot repeat.
    run(["scontrol", "hold", controller])
    cell.update(source_commit=extension["source_commit"], source_dir=extension["source_dir"])
    plan["cells"].insert(names.index(after) + 1, cell)
    plan.update(
        status="extending",
        controller_source_dir=extension["source_dir"],
        controller_source_commit=extension["source_commit"],
    )
    _write(path, plan)
    replacement = submit_controller(path, plan, worker, held=True)
    plan["replacement_controller"] = replacement
    _write(path, plan)
    run(["scancel", controller])
    plan.setdefault("controller_replacements", []).append(
        {"previous": controller, "replacement": replacement, "worker_unchanged": worker, "added_cell": cell["name"]}
    )
    plan.update(status="active", controller=replacement)
    _write(path, plan)
    run(["scontrol", "release", replacement])
    print(f"Added {cell['name']}; worker {worker} unchanged, continuation controller {replacement}")


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
        after = snapshot(Path(cell["registry"]), cell.get("source_commit", plan["source_commit"]))
        if not can_continue(plan["before"], after):
            raise RuntimeError("Allocation expired without new verified saved work; continuation stopped")
        plan["history"][-1]["saved_progress"] = after
    else:
        raise RuntimeError(f"Allocation ended as {state} ({exit_code}); continuation stopped")
    dispatch(path, plan)


def replace_future(path: Path, plan: dict, replacement_path: Path, proof_path: Path, worker: str, controller: str) -> None:
    """Replace only untouched future cells after a separately verified qualification.

    Preserve the current cell's source, exports and checkpoint registry, including
    when its next allocation must resume after a timeout. Never signal its worker.
    """
    replacement = json.loads(replacement_path.read_text())
    proof = json.loads(proof_path.read_text())
    if plan.get("status") != "active" or plan.get("worker") != worker or plan.get("controller") != controller:
        raise ValueError("The active campaign changed; inspect before replacing future cells")
    if replacement.get("status") != "draft" or replacement["source_commit"] == plan["source_commit"]:
        raise ValueError("Future cells require a fresh, unsubmitted source plan")
    names = [cell["name"] for cell in plan["cells"]]
    if names != [cell["name"] for cell in replacement["cells"]] or plan["index"] >= len(names) - 1:
        raise ValueError("The replacement must preserve the complete ordered cell menu")
    verify_source(replacement["source_dir"], replacement["source_commit"])
    if Path(replacement["source_dir"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Execute the handoff from the new immutable source")
    if proof.get("status") != "qualified" or proof.get("source_commit") != replacement["source_commit"]:
        raise ValueError("A matching completed qualification review is required")
    if accounting(str(proof["job_id"])) != ("COMPLETED", "0:0"):
        raise ValueError("Qualification did not complete successfully")
    future = replacement["cells"][plan["index"] + 1:]
    for cell in future:
        if file_digest(Path(cell["export_file"])) != cell["export_sha256"]:
            raise ValueError("Replacement environment changed")
        if snapshot(Path(cell["registry"]), replacement["source_commit"]) or Path(cell["error_file"]).exists():
            raise ValueError("A replacement cell already has execution evidence")
    for cell in plan["cells"][plan["index"] + 1:]:
        if snapshot(Path(cell["registry"]), cell.get("source_commit", plan["source_commit"])) or Path(cell["error_file"]).exists():
            raise ValueError("An old future cell has started; its evidence cannot be replaced")
    if accounting(worker)[0] not in {"PENDING", "RUNNING", "TIMEOUT", "COMPLETED"} or accounting(controller)[0] != "PENDING":
        raise ValueError("Worker or controller needs investigation before handoff")
    backup = path.with_name(f"plan-before-source-{replacement['source_commit'][:12]}.json")
    if backup.exists():
        raise ValueError("A handoff backup exists; inspect the previous transition")
    run(["scontrol", "hold", controller])
    _write(backup, plan)
    plan["cells"] = [*plan["cells"][:plan["index"] + 1], *[
        {**cell, "source_dir": replacement["source_dir"], "source_commit": replacement["source_commit"]}
        for cell in future
    ]]
    plan.update(status="replacing_future", controller_source_dir=replacement["source_dir"],
                controller_source_commit=replacement["source_commit"],
                handoff_qualification={"path": str(proof_path), "sha256": file_digest(proof_path)})
    _write(path, plan)
    watcher = submit_controller(path, plan, worker, held=True)
    plan["replacement_controller"] = watcher
    _write(path, plan)
    replacement.update(status="attached", attached_to=str(path), replacement_cells=[cell["name"] for cell in future])
    _write(replacement_path, replacement)
    run(["scancel", controller])
    plan.setdefault("controller_replacements", []).append({
        "previous": controller, "replacement": watcher, "worker_unchanged": worker,
        "future_source": replacement["source_commit"],
    })
    plan.update(status="active", controller=watcher)
    _write(path, plan)
    run(["scontrol", "release", watcher])
    print(f"Replaced {len(future)} unstarted cells; worker {worker} unchanged; controller {watcher}")


def main(argv: list[str] | None = None) -> None:
    """Build a pinned plan, then submit or advance it under an exclusive lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("add", "start", "advance", "extend", "replace-future"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--name")
    parser.add_argument("--export-file", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--error-file", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--controller-id")
    parser.add_argument("--extension-plan", type=Path)
    parser.add_argument("--after")
    parser.add_argument("--qualification-proof", type=Path)
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
        verify_source(plan["source_dir"], plan["source_commit"])
        if args.action == "replace-future":
            if not all((args.extension_plan, args.qualification_proof, args.job_id, args.controller_id)):
                parser.error("replace-future requires extension-plan, qualification-proof, job-id, and controller-id")
            replace_future(path, plan, args.extension_plan, args.qualification_proof, args.job_id, args.controller_id)
            return
        if args.action == "extend":
            if not all((args.extension_plan, args.after, args.job_id, args.controller_id)):
                parser.error("extend requires extension-plan, after, job-id, and controller-id")
            extend(path, plan, json.loads(args.extension_plan.read_text()), args.after, args.job_id, args.controller_id)
            return
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
