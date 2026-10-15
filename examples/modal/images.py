"""Build immutable benchmark and serving images from an exported Git revision."""

from __future__ import annotations

from pathlib import Path

from examples.modal.config import MODELS, UV_VERSION

# Pin the upstream amd64 CUDA toolkit; the two Python serving locks stay separate.
CUDA_IMAGE = "nvidia/cuda@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c"


def base_image(modal, source: Path, *, cuda: bool = False):
    """Copy only the exported source, never the local environment or credentials."""
    image = modal.Image.from_registry(CUDA_IMAGE if cuda else "ubuntu:24.04", add_python="3.12")
    return (
        image.apt_install("git", "build-essential", "curl", "ca-certificates", "libnuma1", "libgomp1")
        .pip_install(f"uv=={UV_VERSION}")
        .add_local_dir(source, "/opt/gepa", copy=True)
        .workdir("/opt/gepa")
        .env({"PYTHONPATH": "/opt/gepa/src:/opt/gepa", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    )


def benchmark_image(modal, source: Path, benchmark: str):
    """Keep AppWorld, Tau, and Harbor dependencies out of the optimizer environment."""
    image = base_image(modal, source)
    extras = {
        "hotpotqa": "--extra wiki17 --group hotpotqa-task-program",
        "decisionbench": "--extra decisionbench",
        "obliqbench": "--extra obliqbench",
    }
    image = image.run_commands(
        "uv sync --frozen --python 3.12.11 --extra full --extra jev " + extras.get(benchmark, "")
    )
    if benchmark == "appworld":
        image = image.run_commands(
            "uv run --no-sync python -m examples.appworld.prepare --root /opt/appworld-world --venv /opt/appworld"
        )
    if benchmark == "taubench":
        from examples.taubench.benchmark_settings import (
            PYTHON_VERSION,
            RUNTIME_SUPPLEMENTS,
            UPSTREAM_REPOSITORY,
            UPSTREAM_REVISION,
        )

        image = image.run_commands(
            f"git clone {UPSTREAM_REPOSITORY} /opt/tau",
            f"git -C /opt/tau checkout --detach {UPSTREAM_REVISION}",
            f"uv sync --project /opt/tau --frozen --extra knowledge --python {PYTHON_VERSION}",
            "uv pip install --python /opt/tau/.venv/bin/python " + " ".join(RUNTIME_SUPPLEMENTS),
        )
    if benchmark == "terminalbench":
        image = image.apt_install("docker.io", "docker-compose-v2", "iptables").run_commands(
            "uv venv --python 3.12.11 /opt/harbor",
            "uv pip install --python /opt/harbor/bin/python harbor==0.22.0",
        )
    return image


def serving_image(modal, source: Path, role: str):
    """Install the exact existing hash-locked serving environment for one role."""
    lock = MODELS[role]["lock"]
    command = f"uv pip sync --python /opt/serving/bin/python --require-hashes {lock}"
    return (
        base_image(modal, source, cuda=True)
        .env({"NCCL_SOCKET_IFNAME": "=lo", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
        .run_commands(
            "uv venv --python 3.12.7 /opt/serving",
            command,
            command + " --reinstall-package nvidia-cutlass-dsl-libs-cu13",
            command + " --reinstall-package nvidia-cutlass-dsl-libs-base",
            "uv pip check --python /opt/serving/bin/python",
        )
    )
