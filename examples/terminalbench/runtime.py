"""Bind benchmark evidence to a running local vLLM server and its material settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from copy import deepcopy
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    QWEN3_8_27B_MODEL,
    experiment_model_version,
    validate_experiment_vllm_version,
)

RUNTIME_POLICY = {
    "version": 1,
    "source": "local_vllm_launcher",
    "checks": ["pilot", "optimization", "resume", "final_evaluation"],
    "identity_excludes": ["hostname", "boot_id", "pid", "process_start", "model_path", "record_path", "port"],
}


def _digest(value: Any) -> str:
    """Hash material JSON independently of formatting and key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def process_identity(pid: int) -> dict[str, Any]:
    """Read Linux process identity, rejecting dead servers and reused process IDs."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in {"Z", "X"}:
        raise ValueError("The recorded model server has exited")
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid": pid,
        "process_start": fields[19],
    }


def validate_identity(identity: Any, model: str) -> dict[str, Any]:
    """Require a complete, internally consistent material serving identity."""
    if not isinstance(identity, dict) or identity.get("policy") != RUNTIME_POLICY or identity.get("model") != model:
        raise ValueError("Missing or incompatible Terminal-Bench execution runtime")
    required = {
        "policy",
        "model",
        "model_revision",
        "model_integrity_sha256",
        "software",
        "gpus",
        "launch_arguments",
        "environment",
        "precision",
        "parallelism",
        "sha256",
    }
    if set(identity) != required or identity.get("model_revision") != experiment_model_version(model):
        raise ValueError("Runtime must identify the pinned checkpoint and complete serving configuration")
    if identity["sha256"] != _digest({key: value for key, value in identity.items() if key != "sha256"}):
        raise ValueError("Execution runtime fingerprint changed")
    checksum = identity["model_integrity_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("Runtime must identify verified model bytes")
    software = identity["software"]
    if not isinstance(software, dict) or not all(
        isinstance(software.get(key), str) and software[key]
        for key in ("python", "vllm", "torch", "cuda", "transformers", "packages_sha256")
    ):
        raise ValueError("Runtime must identify the serving software")
    validate_experiment_vllm_version(model, software["vllm"])
    gpus = identity["gpus"]
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(
            not isinstance(gpu, dict)
            or not all(gpu.get(key) for key in ("name", "compute_capability", "memory_bytes", "driver_version"))
            for gpu in gpus
        )
    ):
        raise ValueError("Runtime must identify the allocated GPUs")
    if (
        not isinstance(identity["launch_arguments"], list)
        or not identity["launch_arguments"]
        or not all(isinstance(arg, str) for arg in identity["launch_arguments"])
        or not isinstance(identity["environment"], dict)
    ):
        raise ValueError("Runtime must record server launch arguments and environment")
    if (
        not isinstance(identity["precision"], dict)
        or not isinstance(identity["parallelism"], dict)
        or not all(identity["precision"].get(key) for key in ("dtype", "kv_cache_dtype"))
        or not all(
            type(identity["parallelism"].get(key)) is int and identity["parallelism"][key] > 0
            for key in ("tensor", "data")
        )
    ):
        raise ValueError("Runtime must record numerical precision and parallelism")
    return identity


def server_listens(pid: int, port: int) -> bool:
    """Require the local API socket to belong to the launcher process or its workers."""
    rows = Path(f"/proc/{pid}/net/tcp").read_text().splitlines()[1:]
    inodes = {
        row.split()[9] for row in rows if row.split()[3] == "0A" and int(row.split()[1].split(":")[1], 16) == port
    }
    if not inodes:
        return False
    pending, visited = [pid], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        process = Path(f"/proc/{current}")
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    if str(descriptor.readlink()) in {f"socket:[{inode}]" for inode in inodes}:
                        return True
                except FileNotFoundError:
                    continue
            pending.extend(int(child) for child in (process / "task" / str(current) / "children").read_text().split())
        except FileNotFoundError:
            continue
    return False


def load_runtime_record(path: Path | None, model: str, api_base: str | None) -> dict[str, Any]:
    """Read fresh launcher evidence before any evaluation or checkpoint resume.

    The record is local evidence, not remote attestation. It must be regenerated
    by the launcher on each serving node after every server restart.
    """
    if path is None:
        raise ValueError("Supply --runtime-record from examples.terminalbench.runtime on this serving node")
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Unsupported model-server runtime record")
    identity = validate_identity(record.get("identity"), model)
    server = record.get("server", {})
    if not isinstance(server, dict):
        raise ValueError("Missing model-server process identity")
    pid = server.get("pid")
    if type(pid) is not int or pid <= 0 or server != process_identity(pid):
        raise ValueError("Runtime record does not belong to the current live server; relaunch on this node")
    endpoint = urlsplit(api_base or "")
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
        or endpoint.port != record.get("port")
        or endpoint.path.rstrip("/") != "/v1"
        or endpoint.query
        or endpoint.fragment
        or endpoint.username
        or endpoint.password
    ):
        raise ValueError("The model endpoint must match the recorded local server and port")
    if not server_listens(pid, record["port"]):
        raise ValueError("The recorded server does not own the API port; wait for readiness or relaunch it")
    return deepcopy(identity)


def load_role_runtimes(args: argparse.Namespace) -> dict[str, Any]:
    """Validate task and optimizer servers, sharing one record when appropriate."""
    return {
        "student": load_runtime_record(args.runtime_record, args.student_model, args.student_api_base),
        "proposer": load_runtime_record(
            args.proposer_runtime_record or args.runtime_record, args.proposer_model, args.proposer_api_base
        ),
    }


def collect_identity(model: str, model_path: Path, arguments: list[str], options: argparse.Namespace) -> dict[str, Any]:
    """Inspect checkpoint bytes, installed software, and visible GPUs in the serving interpreter."""
    import torch  # type: ignore[import-not-found]

    from examples.common.model_snapshot import verify_model_snapshot

    profile = "qwen3.8-27b" if model == QWEN3_8_27B_MODEL else "deepseek-v4.1-flash"
    manifest = verify_model_snapshot(model_path, profile)
    driver_versions = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if not driver_versions or len(set(driver_versions)) != 1:
        raise ValueError("Expected one NVIDIA driver version on the serving node")
    gpus = [
        {
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            "driver_version": driver_versions[0].strip(),
        }
        for index in range(torch.cuda.device_count())
    ]
    packages = sorted((dist.metadata["Name"], dist.version) for dist in distributions())
    environment = {
        key: value
        for key, value in os.environ.items()
        if (
            key.startswith(("VLLM_", "NCCL_"))
            or key
            in {
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            }
        )
        and not any(secret in key.upper() for secret in ("KEY", "SECRET", "PASSWORD"))
        and not key.upper().endswith("_TOKEN")
        and key not in {"VLLM_HOST_IP", "VLLM_PORT", "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT"}
    }
    identity = {
        "policy": deepcopy(RUNTIME_POLICY),
        "model": model,
        "model_revision": experiment_model_version(model),
        "model_integrity_sha256": _digest(manifest),
        "software": {
            "python": platform.python_version(),
            "vllm": version("vllm"),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "transformers": version("transformers"),
            "packages_sha256": _digest(packages),
        },
        "gpus": gpus,
        "launch_arguments": arguments,
        "environment": environment,
        "precision": {"dtype": options.dtype, "kv_cache_dtype": options.kv_cache_dtype},
        "parallelism": {"tensor": options.tensor_parallel_size, "data": options.data_parallel_size},
    }
    identity["sha256"] = _digest(identity)
    validate_identity(identity, model)
    return identity


def main(argv: list[str] | None = None) -> None:
    """Record the actual runtime, then replace this Linux process with vLLM serve."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=EXPERIMENT_MODELS, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-record", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("serve_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.serve_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    managed = {
        "--model",
        "--host",
        "--port",
        "--served-model-name",
        "--api-key",
        "--hf-token",
        "--config",
        "--tokenizer",
        "--revision",
        "--tokenizer-revision",
        "--chat-template",
        "--hf-config-path",
        "--lora-modules",
        "--prompt-adapters",
    }
    if any(
        arg.startswith("--") and any(flag.startswith(arg.split("=", 1)[0].replace("_", "-")) for flag in managed)
        for arg in arguments
    ):
        parser.error(
            "Model, endpoint, and checkpoint identity are managed by the launcher; put API keys in the environment"
        )
    serving = argparse.ArgumentParser(allow_abbrev=False)
    serving.add_argument("--dtype", required=True)
    serving.add_argument("--kv-cache-dtype", required=True)
    serving.add_argument("--tensor-parallel-size", "-tp", type=int, required=True)
    serving.add_argument("--data-parallel-size", "-dp", type=int, required=True)
    options, _ = serving.parse_known_args(arguments)
    executable = Path(sys.executable).parent / "vllm"
    try:
        if not executable.is_file() or not 0 < args.port < 65536:
            raise ValueError("Run this launcher with the vLLM environment's Python and a valid port")
        server = process_identity(os.getpid())
        identity = collect_identity(args.model, args.model_path.resolve(), arguments, options)
        record = {"schema_version": 1, "identity": identity, "server": server, "port": args.port}
        args.runtime_record.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.runtime_record.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.runtime_record)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(executable),
            "serve",
            str(args.model_path.resolve()),
            "--served-model-name",
            args.model.removeprefix("hosted_vllm/"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            *arguments,
        ],
    )


if __name__ == "__main__":
    main()
