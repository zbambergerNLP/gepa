"""Own a five-H200 allocation with disjoint Qwen and DeepSeek serving processes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from examples.common.pilot_checks import atomic_json
from examples.hotpotqa.main import TEACHER_RUNTIME_KEYS

PROFILE = "deepseek-teacher-qwen-student"
READY_NAMES = {
    "SOLVER_MODEL",
    "SOLVER_API_BASE",
    "SOLVER_SERVED_NAME",
    "GEN_PID",
    "GEN_PORT",
    "LOG_DIR",
    "WIKI17_DIR",
    "WIKI17_INTEGRITY_SHA256",
    "WIKI17_REVISION",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
}


def split_devices(devices: list[str]) -> dict[str, list[str]]:
    """Assign exactly one Qwen GPU and four distinct DeepSeek GPUs."""
    if len(devices) != 5 or len(set(devices)) != 5 or not all(x.startswith("GPU-") for x in devices):
        raise ValueError("The paired profile requires exactly five distinct allocated GPU UUIDs")
    return {"qwen3.8-27b": devices[4:], "deepseek-v4.1-flash": devices[:4]}


def write_ready() -> None:
    """Publish only the verified child runtime and its allocation-specific endpoint."""
    environment = {key: value for key, value in os.environ.items() if key in READY_NAMES or key.startswith("HOTPOTQA_")}
    atomic_json(
        Path(os.environ["HOTPOTQA_SERVER_READY_FILE"]),
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "profile": os.environ["MODEL_PROFILE"],
            "gpu_uuids": os.environ["CUDA_VISIBLE_DEVICES"].split(","),
            "environment": environment,
        },
    )


def combine_runtimes(qwen: dict, deepseek: dict, allocation: dict[str, list[str]], job_id: str) -> dict[str, str]:
    """Require both endpoints to belong to this allocation and source before routing roles."""
    for ready, profile in ((qwen, "qwen3.8-27b"), (deepseek, "deepseek-v4.1-flash")):
        if ready["job_id"] != job_id or ready["profile"] != profile or ready["gpu_uuids"] != allocation[profile]:
            raise ValueError("Serving readiness belongs to a different allocation or GPU assignment")
    qe, de = qwen["environment"], deepseek["environment"]
    for key in (
        "HOTPOTQA_SOURCE_COMMIT",
        "HOTPOTQA_SOURCE_MANIFEST_SHA256",
        "HOTPOTQA_GEPA_ENV_SHA256",
        "HOTPOTQA_CAMPAIGN_ID",
        "WIKI17_INTEGRITY_SHA256",
    ):
        if qe[key] != de[key]:
            raise ValueError(f"Teacher and student disagree on {key}")
    if (
        qe["SOLVER_MODEL"] != "hosted_vllm/Qwen/Qwen3.8-27B"
        or de["SOLVER_MODEL"] != "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
    ):
        raise ValueError("Incorrect teacher/student model direction")
    if qe["SOLVER_API_BASE"] == de["SOLVER_API_BASE"]:
        raise ValueError("Teacher and student must have distinct local endpoints")
    teacher = {"HOTPOTQA_" + key: de["HOTPOTQA_" + key] for key in TEACHER_RUNTIME_KEYS}
    return {
        **qe,
        "MODEL_PROFILE": PROFILE,
        "REFLECTION_MODEL": de["SOLVER_MODEL"],
        "REFLECTION_API_BASE": de["SOLVER_API_BASE"],
        "HOTPOTQA_TEACHER_RUNTIME": json.dumps(teacher, sort_keys=True, separators=(",", ":")),
        "GEN_PID": de["GEN_PID"],
    }


def main() -> None:
    """Start both canonical servers and release the allocation when its workload ends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-ready", action="store_true")
    args = parser.parse_args()
    if args.write_ready:
        write_ready()
        return
    environment = dict(os.environ)
    if environment.get("MODEL_PROFILE") != PROFILE or environment.get("HOTPOTQA_PRODUCTION_LAUNCH") != "1":
        raise ValueError("Use the canonical prepared paired-model launch environment")
    root = Path(environment["SCRATCH_BASE"])
    log_dir = (
        root
        / "logs/hotpotqa"
        / environment["HOTPOTQA_CAMPAIGN_ID"]
        / environment["HOTPOTQA_SOURCE_COMMIT"]
        / PROFILE
        / environment["SLURM_JOB_ID"]
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    # Attempt-specific paths prevent a requeued job from trusting a former server's endpoint.
    exchange = log_dir / str(time.time_ns())
    exchange.mkdir()
    uv = environment["GEPA_UV_BIN"]
    device_code = "import json,torch; print(json.dumps(['GPU-'+str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(torch.cuda.device_count())]))"
    devices = json.loads(
        subprocess.check_output(
            [
                uv,
                "run",
                "--no-project",
                "--python",
                str(root / ".serving-venv/bin/python"),
                "python",
                "-c",
                device_code,
            ],
            text=True,
        )
    )
    allocation = split_devices(devices)
    # This runner executes on Linux; macOS type stubs omit the affinity API.
    cpus = sorted(os.sched_getaffinity(0))  # pyright: ignore[reportAttributeAccessIssue]
    if len(cpus) < 40:
        raise ValueError("The paired profile requires at least forty allocated CPU cores")
    atomic_json(
        exchange / "allocation.json", {"gpu_uuids": allocation, "qwen_cpus": cpus[:8], "deepseek_cpus": cpus[8:40]}
    )
    children: list[subprocess.Popen] = []
    streams = []

    def interrupted(signum, frame):
        """Ensure Slurm termination also releases both serving process groups."""
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for profile, gpu_uuids in allocation.items():
            teacher = profile == "deepseek-v4.1-flash"
            venv = root / (".serving-venv-deepseek-v4.1-flash" if teacher else ".serving-venv")
            ready_file = exchange / f"{profile}.json"
            child_env = {
                **environment,
                "MODEL_PROFILE": profile,
                "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids),
                "SERVING_VENV_DIR": str(venv),
                "HOTPOTQA_SERVING_ENV_SHA256": environment[
                    "HOTPOTQA_TEACHER_SERVING_ENV_SHA256" if teacher else "HOTPOTQA_SERVING_ENV_SHA256"
                ],
                "VLLM_TENSOR_PARALLEL_SIZE": "4" if teacher else "1",
                "VLLM_DATA_PARALLEL_SIZE": "1",
                "VLLM_API_SERVER_COUNT": "1",
                "VLLM_MAX_NUM_SEQS": environment.get("TEACHER_MAX_NUM_SEQS", "2")
                if teacher
                else environment["VLLM_MAX_NUM_SEQS"],
                "HOTPOTQA_SERVER_READY_FILE": str(ready_file),
            }
            child_env.pop("HOTPOTQA_TEACHER_RUNTIME", None)
            child_env.pop("GEN_PORT", None)
            stream = (exchange / f"{profile}.log").open("w")
            streams.append(stream)
            role_cpus = cpus[8:40] if teacher else cpus[:8]
            children.append(
                subprocess.Popen(
                    [
                        "taskset",
                        "--cpu-list",
                        ",".join(map(str, role_cpus)),
                        "bash",
                        "examples/hotpotqa/run_hotpotqa.sbatch",
                    ],
                    env=child_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + int(environment.get("HEALTH_TIMEOUT", "3600")) + 600
        while not all((exchange / f"{profile}.json").exists() for profile in allocation):
            if any(child.poll() is not None for child in children):
                raise RuntimeError(f"A serving process failed during startup; inspect {exchange}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Paired serving startup exceeded its deadline; inspect {exchange}")
            time.sleep(2)
        ready = [json.loads((exchange / f"{profile}.json").read_text()) for profile in allocation]
        runtime = combine_runtimes(ready[0], ready[1], allocation, environment["SLURM_JOB_ID"])
        # Preserve parent scientific intent; server child readiness must not select a pilot or cell.
        workload_env = {**environment, **runtime}
        for key in ("HOTPOTQA_PILOT_ONLY", "HOTPOTQA_PILOT_ROOT", "HOTPOTQA_PILOT_STAGE", "MAX_WORKERS"):
            if key in environment:
                workload_env[key] = environment[key]
        workload_env.update(LOG_DIR=str(log_dir), OPENAI_API_KEY="EMPTY", PYTHONUNBUFFERED="1")
        workload = subprocess.Popen(
            ["bash", "scripts/della/remote/hotpotqa_workload.sh"], env=workload_env, start_new_session=True
        )
        children.append(workload)
        while workload.poll() is None:
            if any(child.poll() is not None for child in children[:-1]):
                raise RuntimeError("A serving process exited while the paired workload was active")
            time.sleep(2)
        if workload.returncode:
            raise SystemExit(workload.returncode)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
