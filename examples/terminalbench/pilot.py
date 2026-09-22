"""Validate training-only pilot evidence before freezing campaign settings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from examples.terminalbench.runtime import validate_identity
from gepa.adapters.terminal_bench_adapter import TerminalBenchManifest
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope

PILOT_SCHEMA_VERSION = 9
PILOT_PROTOCOL = {
    "version": 1,
    "smoke_tasks": 3,
    "full_tasks": 30,
    "split": "train",
    "harness": "initial",
    "review_metrics": ["token_usage", "cutoffs", "timeouts", "throughput"],
    "scope_reuse": "identical_initial_runtime_text",
}
RUNTIME_FIELDS = (
    "execution_runtime",
    "adapter",
    "provider_retry_policy",
    "experiment",
    "dataset",
    "task_context_settings",
    "model",
    "model_version",
    "api_base",
    "template_family",
    "n_concurrent",
    "reference_seed_digest",
    "student_agent_kwargs",
    "token_usage_policy",
    "text_limits",
    "harbor_process_timeout_sec",
)
ARTIFACTS = ("canary-config.json", "task-results.json", "token-usage-summary.json", "pilot-summary.json")


def digest(value: Any) -> str:
    """Hash JSON content independently of formatting or filesystem location."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    """Extract the task runtime actually recorded by an optimization run."""
    return {
        **{key: contract[key] for key in RUNTIME_FIELDS if key in contract},
        "execution_runtime": (contract.get("execution_runtime") or {}).get("student"),
        "model": contract["student_model"],
        "model_version": contract["student_model_version"],
        "api_base": contract["student_api_base"],
        "student_agent_kwargs": {
            "token_limits": contract["token_limits"],
            "model_info": contract["student_model_info"],
            "llm_kwargs": {
                "num_retries": contract["student_num_retries"],
                **contract["student_decoding"],
                **contract["student_request_overrides"],
            },
        },
    }


def validate_runtime(
    config: dict[str, Any], runtime: dict[str, Any], *, allow_concurrency_change: bool = False
) -> None:
    """Reject settings drift while allowing smoke-to-full concurrency calibration."""
    for key in RUNTIME_FIELDS:
        if key == "n_concurrent" and allow_concurrency_change:
            continue
        if key not in config or key not in runtime or config[key] != runtime[key]:
            raise ValueError(f"Pilot runtime does not match the campaign: {key}")


def validate_snapshot(evidence: dict[str, Any], manifest: TerminalBenchManifest, stage: str) -> None:
    """Require completed, correctly sized training coverage and an intact stage chain."""
    config = evidence.get("config")
    complete = evidence.get("completion")
    if not isinstance(config, dict) or not isinstance(complete, dict):
        raise ValueError("Pilot configuration or completion evidence is missing")
    expected_ids = manifest.splits["train"][:3] if stage == "smoke" else manifest.splits["train"]
    if (
        config.get("schema_version") != PILOT_SCHEMA_VERSION
        or config.get("pilot_protocol") != PILOT_PROTOCOL
        or config.get("stage") != stage
        or config.get("split") != "train"
        or config.get("dataset") != manifest.dataset
        or config.get("task_ids") != expected_ids
        or config.get("task_refs") != {task_id: manifest.task_refs[task_id] for task_id in expected_ids}
        or complete.get("schema_version") != 1
        or complete.get("task_count") != len(expected_ids)
        or complete.get("artifacts", {}).get("canary-config.json") != digest(config)
        or set(complete.get("artifacts", {})) != set(ARTIFACTS)
    ):
        raise ValueError(f"A completed {stage} pilot on exactly {len(expected_ids)} training tasks is required")
    scope = TerminalBenchTextScope(config["optimization_scope"], config["template_family"])
    validate_identity(config.get("execution_runtime"), config["model"])
    if config.get("text_scope") != scope.contract() or config.get("candidate_digest") != manifest.candidate_digest(
        scope.materialize(scope.seed_candidate())
    ):
        raise ValueError("Pilot must evaluate the initial harness with the recorded text scope")
    if stage == "full":
        smoke = config.get("smoke_evidence")
        if not isinstance(smoke, dict):
            raise ValueError("The full pilot requires completed smoke evidence")
        validate_snapshot(smoke, manifest, "smoke")
        validate_runtime(smoke["config"], config, allow_concurrency_change=True)


def complete_pilot(directory: Path, elapsed_seconds: float) -> None:
    """Write a completion marker only after all scored tasks and usage are saved."""
    config = json.loads((directory / "canary-config.json").read_text())
    outputs = json.loads((directory / "task-results.json").read_text())
    if [output["task_id"] for output in outputs] != config["task_ids"]:
        raise ValueError("Pilot results do not cover the exact requested training tasks")
    if any(not math.isfinite(float(output["reward"])) for output in outputs):
        raise ValueError("Pilot rewards must be finite")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Pilot elapsed time must be positive and finite")
    summary = {
        "schema_version": 1,
        "stage": config["stage"],
        "task_count": len(outputs),
        "elapsed_seconds": elapsed_seconds,
        "tasks_per_hour": len(outputs) * 3600 / elapsed_seconds,
        "timed_out_task_ids": [output["task_id"] for output in outputs if output["errors"]],
        "mean_reward": sum(float(output["reward"]) for output in outputs) / len(outputs),
        "review_metrics": PILOT_PROTOCOL["review_metrics"],
        "token_usage_summary": "token-usage-summary.json",
        "task_evidence": "task-results.json",
    }
    (directory / "pilot-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    complete = {
        "schema_version": 1,
        "task_count": len(outputs),
        "artifacts": {name: digest(json.loads((directory / name).read_text())) for name in ARTIFACTS},
    }
    temporary = directory / "pilot-complete.json.tmp"
    temporary.write_text(json.dumps(complete, indent=2) + "\n")
    temporary.replace(directory / "pilot-complete.json")


def load_completed_pilot(directory: Path, manifest: TerminalBenchManifest, stage: str) -> dict[str, Any]:
    """Read and verify saved pilot artifacts before accepting them as evidence."""
    complete = json.loads((directory / "pilot-complete.json").read_text())
    artifacts = {name: json.loads((directory / name).read_text()) for name in ARTIFACTS}
    if complete.get("artifacts") != {name: digest(value) for name, value in artifacts.items()}:
        raise ValueError("Pilot artifacts changed after completion; rerun the pilot")
    evidence = {"config": artifacts["canary-config.json"], "completion": complete}
    validate_snapshot(evidence, manifest, stage)
    return evidence


def validate_review(review: Any, contract: dict[str, Any], manifest: TerminalBenchManifest) -> None:
    """Require the agreed human review and a matching full training pilot."""
    if not isinstance(review, dict) or review.get("reviewed_metrics") != PILOT_PROTOCOL["review_metrics"]:
        raise ValueError("Review both pilot stages, then supply --reviewed-pilot with the full pilot directory")
    evidence = review.get("full_pilot")
    if not isinstance(evidence, dict):
        raise ValueError("Reviewed full-pilot evidence is missing")
    validate_snapshot(evidence, manifest, "full")
    validate_runtime(evidence["config"], run_runtime(contract))


def review_pilot(directory: Path, contract: dict[str, Any], manifest: TerminalBenchManifest) -> dict[str, Any]:
    """Record the review explicitly attested by the campaign's --reviewed-pilot flag."""
    review = {
        "reviewed_metrics": list(PILOT_PROTOCOL["review_metrics"]),
        "full_pilot": load_completed_pilot(directory, manifest, "full"),
    }
    validate_review(review, contract, manifest)
    return review
