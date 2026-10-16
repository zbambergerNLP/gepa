"""Prepare pinned inputs or supervise one benchmark inside a Modal Sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from examples.common.continuation import PAUSED_EXIT_CODE, POLICY
from examples.common.remote_runtime import request_json
from examples.modal.config import MODELS, benchmark_arguments, digest, embedding_arguments, roles, validate_plan
from examples.modal.continuation import allocation_directory, runtime_identity, snapshot_state, verify_state
from examples.modal.gateway import atomic_json

PYTHON = ["uv", "run", "--no-sync", "python"]
RESULTS = Path("/results")


def data_key(plan: dict) -> str:
    """Keep preparation independent of optimization budgets and model calls."""
    return digest({key: plan[key] for key in ("benchmark", "source_sha256", "embedding", "terminalbench_scope")})


def preparation_commands(plan: dict) -> list[list[str]]:
    """Reuse official corpus loaders and existing hash verification code."""
    commands = {
        "hotpotqa": [
            [*PYTHON, "-m", "examples.common.wiki17_bm25", "prepare", "--root", "/data/wiki17"],
            [*PYTHON, "-m", "examples.common.wiki17_bm25", "verify", "--deep", "--root", "/data/wiki17"],
            [*PYTHON, "-c", "from examples.hotpotqa.main import load_benchmark_data; load_benchmark_data(None)"],
        ],
        "decisionbench": [
            [
                *PYTHON,
                "-c",
                "from pathlib import Path; from examples.decisionbench.utils import load_decisionbench; load_decisionbench(cache_dir=Path('/data/decisionbench'))",
            ]
        ],
        "obliqbench": [
            [
                *PYTHON,
                "-m",
                "examples.obliqbench.prepare",
                "--build-index",
                "--data-dir",
                "/data/obliq/data",
                "--index-dir",
                "/data/obliq/index",
                *embedding_arguments(plan),
            ]
        ],
        "appworld": [
            [
                *PYTHON,
                "-c",
                "from pathlib import Path; from examples.appworld.utils import load_dataset; from examples.appworld.runtime import inspect_runtime; load_dataset(Path('/opt/appworld-world')); inspect_runtime(Path('/opt/appworld-world'), Path('/opt/appworld/bin/python'))",
            ]
        ],
        "taubench": [
            [
                *PYTHON,
                "-c",
                "from pathlib import Path; from examples.taubench.utils import load_data; load_data(Path('/opt/tau'))",
            ]
        ],
        "terminalbench": [
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "/opt/harbor/bin/python",
                "python",
                "-m",
                "examples.modal.terminalbench_probe",
                "--scope",
                plan["terminalbench_scope"],
                "--output",
                "/results/terminalbench-environments",
            ]
        ],
    }
    # Both pinned checkpoints are prepared once, including for solver-only pilots,
    # so that the same sealed bundle can support the subsequent optimizer pilot.
    for role in MODELS.values():
        commands[plan["benchmark"]].append(
            [
                *PYTHON,
                "-m",
                "examples.common.model_snapshot",
                "prepare",
                "--model-profile",
                role["profile"],
                "--root",
                "/models/" + role["profile"],
            ]
        )
    return commands[plan["benchmark"]]


def docker_start(log):
    """Start an actual Docker daemon in the VM; fail before model requests."""
    daemon = subprocess.Popen(["dockerd", "--host=unix:///var/run/docker.sock"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return daemon
        if daemon.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError("Docker failed to start in the VM sandbox; inspect console.log")


def seal_files(root: Path, output: Path, *, exclude: tuple[str, ...] = ()) -> None:
    """Hash closed artifacts for independent verification after download."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path == output or path.is_symlink() or not path.is_file() or path.relative_to(root).parts[0] in exclude:
            continue
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        files[str(path.relative_to(root))] = checksum.hexdigest()
    atomic_json(output, {"schema_version": 1, "files": files})


def prepare_writable_caches(plan: dict, data: Path, results: Path, cache: Path) -> None:
    """Keep immutable datasets shared while giving each run its own cache writes."""
    if (data / "huggingface").exists():
        shutil.copytree(data / "huggingface", cache, symlinks=True)
    if plan["benchmark"] == "hotpotqa":
        root = results / "wiki17"
        root.mkdir(exist_ok=True)
        for source in (data / "wiki17").iterdir():
            if not source.name.startswith("retriever_cache_"):
                target = root / source.name
                if target.is_symlink() and target.readlink() == source:
                    continue
                target.symlink_to(source, target_is_directory=source.is_dir())


def main() -> None:
    """Drain one allocation, preserving a resumable run separately from completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    args = parser.parse_args()
    plan = json.loads(os.environ["GEPA_PLAN"])
    validate_plan(plan)
    RESULTS.mkdir(exist_ok=True)
    epoch = int(os.environ.get("GEPA_ALLOCATION_EPOCH", "0"))
    directory = RESULTS / allocation_directory(epoch) if args.phase == "run" else RESULTS
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / (args.phase + "-started.json")
    with claim.open("x") as stream:
        json.dump({"plan_sha256": digest(plan), "started_unix": time.time()}, stream)
    subprocess.run(["sync", "/results"], check=True)
    endpoints = json.loads(os.environ.get("GEPA_ENDPOINTS", "{}"))
    status = {"phase": args.phase, "state": "starting", "plan_sha256": digest(plan), "epoch": epoch}
    status_path = directory / (args.phase + "-status.json")
    log = (directory / (args.phase + "-console.log")).open("ab", buffering=0)
    process = daemon = None

    def interrupted(signum, _frame):
        raise InterruptedError(f"Allocation received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.phase == "run" and epoch:
            verify_state(RESULTS, RESULTS / allocation_directory(epoch - 1))
        if plan["benchmark"] == "terminalbench":
            daemon = docker_start(log)
        if args.phase == "prepare":
            for command in preparation_commands(plan):
                status.update(state="preparing", command=command, updated_unix=time.time())
                atomic_json(status_path, status)
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(Path("/data/ready.json"), {"key": data_key(plan), "source_sha256": plan["source_sha256"]})
            atomic_json(Path("/models/ready.json"), {"models": MODELS})
            subprocess.run(["sync", "/data", "/models"], check=True)
        else:
            ready = json.loads(Path("/data/ready.json").read_text())
            if ready["key"] != data_key(plan):
                raise ValueError("Prepared data do not match this benchmark/source/retrieval configuration")
            prepare_writable_caches(plan, Path("/data"), RESULTS, Path("/tmp/huggingface"))
            if plan["benchmark"] == "appworld" and not (RESULTS / "appworld").exists():
                shutil.copytree("/opt/appworld-world", RESULTS / "appworld")
            records = {}
            for role in roles(plan):
                endpoint = endpoints[role]["api_base"]
                records[role] = {**request_json(endpoint, "/runtime"), "api_base": endpoint}
                atomic_json(RESULTS / f"{role}-runtime.json", records[role])
                atomic_json(directory / f"{role}-runtime.json", records[role])
            execution = {
                "schema_version": 1,
                "provider": "modal",
                "source_sha256": plan["source_sha256"],
                "driver": json.loads(os.environ["GEPA_DRIVER_IDENTITY"]),
                "roles": records,
            }
            if epoch:
                previous = json.loads((RESULTS / allocation_directory(epoch - 1) / "execution.json").read_text())
                if runtime_identity(execution) != runtime_identity(previous):
                    raise ValueError("Continuation source, image, resources or actual model runtime changed")
            atomic_json(RESULTS / "execution.json", execution)
            atomic_json(directory / "execution.json", execution)
            command = [*PYTHON, "-m", f"examples.{plan['benchmark']}.main", *benchmark_arguments(plan, endpoints)]
            atomic_json(RESULTS / "command.json", command)
            environment = {
                **os.environ,
                "HF_HOME": "/tmp/huggingface",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "GEPA_PAUSE_RECEIPT": str(directory / "pause.json"),
            }
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                status.update(state="running", updated_unix=time.time())
                atomic_json(status_path, status)
                atomic_json(RESULTS / "run-status.json", status)
                subprocess.run(["sync", "/results"], check=True)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode == PAUSED_EXIT_CODE:
                proof = json.loads((directory / "pause.json").read_text())
                if proof != {"state": "paused", "policy": POLICY, "reason": "allocation_deadline"}:
                    raise ValueError("Missing or incompatible graceful pause evidence")
                status.update(state="paused", returncode=PAUSED_EXIT_CODE)
            elif process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        if status["state"] != "paused":
            status.update(state="completed", returncode=0)
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, returncode=getattr(error, "returncode", None))
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        cleanup = {}
        for role, endpoint in endpoints.items():
            try:
                request_json(endpoint["api_base"], "/shutdown", method="POST", timeout=15)
                cleanup[role] = "shutdown_requested"
            except Exception as error:
                cleanup[role] = {"unknown": type(error).__name__}
        if daemon is not None and daemon.poll() is None:
            daemon.terminate()
        log.close()
        status.update(updated_unix=time.time(), worker_cleanup=cleanup)
        atomic_json(status_path, status)
        if args.phase == "run":
            atomic_json(RESULTS / "run-status.json", status)
            snapshot_state(RESULTS, directory)
        seal_files(directory, directory / (args.phase + "-files.json"), exclude=tuple(MODELS))
        subprocess.run(["sync", "/results"], check=True)


if __name__ == "__main__":
    main()
