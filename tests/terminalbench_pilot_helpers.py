"""Create offline pilot artifacts for campaign and checkpoint integration tests."""

import json
from pathlib import Path

import pytest

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, experiment_model_version
from examples.terminalbench import canary, evaluate, runtime
from examples.terminalbench.pilot import (
    PILOT_PROTOCOL,
    PILOT_SCHEMA_VERSION,
    complete_pilot,
    load_completed_pilot,
    run_runtime,
)
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope


def runtime_fixture(model: str):
    """Describe an offline-only server without depending on the test machine's GPUs."""
    identity = {
        "policy": runtime.RUNTIME_POLICY,
        "model": model,
        "model_revision": experiment_model_version(model),
        "model_integrity_sha256": "a" * 64,
        "software": {
            "python": "3.12.8",
            "vllm": "0.1.1.dev5+ge77daef89" if model == DEEPSEEK_V4_1_FLASH_MODEL else "0.25.1",
            "torch": "2.10.0",
            "cuda": "13.0",
            "transformers": "5.0.0",
            "packages_sha256": "b" * 64,
        },
        "gpus": [
            {
                "name": "NVIDIA H200",
                "compute_capability": [9, 0],
                "memory_bytes": 141_000_000_000,
                "driver_version": "580",
            }
        ]
        * 8,
        "launch_arguments": [
            "--dtype",
            "auto",
            "--kv-cache-dtype",
            "fp8",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "1",
        ],
        "environment": {},
        "precision": {"dtype": "auto", "kv_cache_dtype": "fp8"},
        "parallelism": {"tensor": 8, "data": 1},
    }
    identity["sha256"] = runtime._digest(identity)
    return identity


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch):
    """Replace only live server discovery for pre-existing offline benchmark tests."""

    def load(path, model, api_base):
        return runtime_fixture(model)

    monkeypatch.setattr(runtime, "load_runtime_record", load)
    monkeypatch.setattr(canary, "load_runtime_record", load)
    monkeypatch.setattr(evaluate, "load_runtime_record", load)


def write_pilot_fixture(root: Path, contract, manifest) -> Path:
    """Write both completed stages using a campaign's exact task runtime."""
    if contract.get("execution_runtime") is None:
        contract["execution_runtime"] = {
            role: runtime_fixture(contract[f"{role}_model"]) for role in ("student", "proposer")
        }
    smoke = None
    scope = TerminalBenchTextScope("system_prompt", contract["template_family"])
    for stage, count in (("smoke", 3), ("full", 30)):
        directory = root / stage
        directory.mkdir(parents=True, exist_ok=True)
        ids = manifest.splits["train"][:count]
        config = {
            **run_runtime(contract),
            "schema_version": PILOT_SCHEMA_VERSION,
            "pilot_protocol": PILOT_PROTOCOL,
            "stage": stage,
            "split": "train",
            "optimization_scope": scope.name,
            "text_scope": scope.contract(),
            "candidate_digest": manifest.candidate_digest(scope.materialize(scope.seed_candidate())),
            "task_ids": ids,
            "task_refs": {task_id: manifest.task_refs[task_id] for task_id in ids},
            "smoke_evidence": smoke,
        }
        (directory / "canary-config.json").write_text(json.dumps(config))
        outputs = [{"task_id": task_id, "reward": 0.0, "errors": []} for task_id in ids]
        (directory / "task-results.json").write_text(json.dumps(outputs))
        (directory / "token-usage-summary.json").write_text(
            json.dumps({"schema_version": 1, "files": [], "models": {}})
        )
        complete_pilot(directory, 60.0)
        smoke = load_completed_pilot(directory, manifest, stage)
    return root / "full"
