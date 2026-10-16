"""Describe cloud resources separately from benchmark scientific arguments."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from examples.common.experiment_models import DEFAULT_PROPOSER_MODEL, DEFAULT_SOLVER_MODEL

SDK_VERSION = "1.6.0"
UV_VERSION = "0.9.13"
BENCHMARKS = ("hotpotqa", "decisionbench", "obliqbench", "appworld", "terminalbench", "taubench")
ROOT = Path(__file__).resolve().parents[2]
COMMON_SERVE_ARGS = [
    "--data-parallel-size",
    "1",
    "--api-server-count",
    "1",
    "--gpu-memory-utilization",
    "0.92",
    "--max-model-len",
    "262144",
    "--max-num-seqs",
    "1",
    "--max-num-batched-tokens",
    "16384",
    "--dtype",
    "bfloat16",
    "--seed",
    "0",
    "--no-enable-prefix-caching",
    "--language-model-only",
]
MODELS = {
    "solver": {
        "model": DEFAULT_SOLVER_MODEL,
        "profile": "qwen3.8-27b",
        "gpu": "H200:2",
        "cpu": 8,
        "memory": 131072,
        "lock": "examples/hotpotqa/serving/requirements-x86_64-linux-py312.txt",
        "arguments": [
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--tensor-parallel-size",
            "2",
            "--kv-cache-dtype",
            "auto",
            *COMMON_SERVE_ARGS,
        ],
    },
    "proposer": {
        "model": DEFAULT_PROPOSER_MODEL,
        "profile": "deepseek-v4.1-flash",
        "gpu": "H200:4",
        "cpu": 16,
        "memory": 786432,
        "lock": "examples/hotpotqa/serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt",
        "arguments": [
            "--enable-auto-tool-choice",
            "--tokenizer-mode",
            "deepseek_v41",
            "--reasoning-parser",
            "deepseek_v41",
            "--tool-call-parser",
            "deepseek_v41",
            "--tensor-parallel-size",
            "4",
            "--enable-expert-parallel",
            "--kv-cache-dtype",
            "fp8",
            "--engram-config",
            '{"cpu_offload":true}',
            *COMMON_SERVE_ARGS,
        ],
    },
}


def digest(value: object) -> str:
    """Hash public configuration deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan: dict) -> None:
    """Reject overrides of infrastructure-owned paths, models, and credentials."""
    if plan.get("schema_version") != 1 or plan.get("benchmark") not in BENCHMARKS:
        raise ValueError("Unsupported Modal benchmark plan")
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,47}", plan.get("run_id", "")):
        raise ValueError("run_id must be 3-48 lowercase letters, digits, or hyphens, beginning with a letter")
    if plan.get("mode") not in {"pilot", "optimizer-pilot", "optimize", "baseline"}:
        raise ValueError("Select an explicit benchmark mode")
    if type(plan.get("timeout")) is not int or not 600 <= plan["timeout"] <= 86400:
        raise ValueError("timeout must be 600-86400 seconds; it is an allocation limit, not a new metric budget")
    if type(drain_seconds(plan)) is not int or not 60 <= drain_seconds(plan) <= plan["timeout"] - 300:
        raise ValueError("drain_seconds must leave at least 300 seconds for startup and useful work")
    if not isinstance(plan.get("arguments"), list) or not all(isinstance(x, str) for x in plan["arguments"]):
        raise ValueError("arguments must be a JSON list of CLI strings")
    reserved = {
        "--mode",
        "--run-dir",
        "--model",
        "--solver-model",
        "--reflection-model",
        "--proposer-model",
        "--api-base",
        "--solver-api-base",
        "--reflection-api-base",
        "--execution-runtime",
        "--runtime-record",
        "--proposer-runtime-record",
        "--container-runtime",
        "--task-scope",
        "--appworld-root",
        "--appworld-python",
        "--tau-source",
        "--wiki17-dir",
        "--data-path",
        "--data-file",
        "--dataset-cache",
        "--data-dir",
        "--index-dir",
        "--harbor-work-dir",
        "--harbor-executable",
        "--docker-executable",
        "--offline-task-bundle",
        "--singularity-image-cache",
        "--embedding-device",
        "--embedding-attention",
        "--embedding-dtype",
        "--embedding-batch-size",
    }
    for arg in plan["arguments"]:
        option = arg.split("=", 1)[0]
        if option.startswith("--") and any(name.startswith(option) for name in reserved):
            raise ValueError(f"Infrastructure owns {arg.split('=', 1)[0]}; use the corresponding plan field")
    if plan.get("terminalbench_scope") not in {"della88", "all89"}:
        raise ValueError("TerminalBench scope must be explicitly della88 or all89")
    embedding = plan.get("embedding", {})
    if (
        embedding.get("device") not in {"cpu", "cuda"}
        or embedding.get("attention") not in {"eager", "sdpa"}
        or embedding.get("dtype") not in {"float32", "bfloat16"}
        or type(embedding.get("batch_size")) is not int
        or embedding["batch_size"] < 1
    ):
        raise ValueError("Incomplete OBLIQ embedding profile")
    if not isinstance(plan.get("secrets"), list) or not all(isinstance(x, str) and x for x in plan["secrets"]):
        raise ValueError("secrets must contain Modal Secret names, never credential values")


def roles(plan: dict) -> list[str]:
    """Allocate only the model roles used by the selected mode."""
    return ["solver", "proposer"] if plan["mode"] in {"optimizer-pilot", "optimize"} else ["solver"]


def embedding_arguments(plan: dict) -> list[str]:
    """Use exactly the same encoder configuration during preparation and evaluation."""
    return [
        item
        for key, value in plan["embedding"].items()
        for item in ("--embedding-" + key.replace("_", "-"), str(value))
    ]


def drain_seconds(plan: dict) -> int:
    """Reserve time to finish an in-flight task group, proposal and evidence sealing."""
    return plan.get("drain_seconds", min(7200, plan["timeout"] // 4))


def benchmark_arguments(plan: dict, endpoints: dict[str, dict]) -> list[str]:
    """Pass scientific arguments through unchanged and supply cloud-local paths."""
    name = plan["benchmark"]
    args = ["--mode", plan["mode"], "--run-dir", "/results/benchmark", "--execution-runtime", "/results/execution.json"]
    args += ["--solver-api-base", endpoints["solver"]["api_base"]]
    if "proposer" in endpoints:
        args += ["--reflection-api-base", endpoints["proposer"]["api_base"]]
    paths = {
        "hotpotqa": ["--wiki17-dir", "/results/wiki17"],
        "decisionbench": ["--dataset-cache", "/data/decisionbench"],
        "obliqbench": [
            "--data-dir",
            "/data/obliq/data",
            "--index-dir",
            "/data/obliq/index",
            *embedding_arguments(plan),
        ],
        "appworld": ["--appworld-root", "/results/appworld", "--appworld-python", "/opt/appworld/bin/python"],
        "taubench": ["--tau-source", "/opt/tau"],
        "terminalbench": [
            "--container-runtime",
            "docker",
            "--task-scope",
            plan["terminalbench_scope"],
            "--runtime-record",
            "/results/solver-runtime.json",
            "--harbor-executable",
            "/opt/harbor/bin/harbor",
        ],
    }
    if name == "terminalbench" and "proposer" in endpoints:
        paths[name] += ["--proposer-runtime-record", "/results/proposer-runtime.json"]
    return [*args, *paths[name], *plan["arguments"]]
