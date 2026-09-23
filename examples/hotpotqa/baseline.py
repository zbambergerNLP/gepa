"""Identify the shared starting-prompt baseline and verify its held-out evidence."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from examples.hotpotqa.source_compatibility import comparison_runtime
from gepa.strategies.forest_constants import SOLVER_ROLE

BASELINE_PROTOCOL = {
    "version": 1,
    "candidate": "initial_prompts",
    "test_repetitions": 1,
    "sharing": "same_campaign_model_data_and_task_runtime",
    "timing": "with_first_completed_ablation_test",
    "optimizer_feedback": False,
}
BASELINE_CONTRACT_FILENAME = "baseline-contract.json"


def baseline_digest(value: dict) -> str:
    """Hash a baseline identity or candidate using canonical JSON."""
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def build_baseline_contract(run_contract: dict) -> dict:
    """Keep all task-evaluation settings while excluding optimizer method and budget."""
    if run_contract.get("baseline_protocol") != BASELINE_PROTOCOL:
        raise ValueError("HotPotQA run does not record the shared starting-baseline protocol.")
    return deepcopy(
        {
            "schema_version": 1,
            "protocol": BASELINE_PROTOCOL,
            "candidate": run_contract["optimizer"]["rendered_seed"],
            "models": {key: value for key, value in run_contract["models"].items() if key.startswith(SOLVER_ROLE)},
            **{
                key: run_contract[key]
                for key in (
                    "benchmark",
                    "reference_artifact_commit",
                    "scientific_contract_enforced",
                    "provider_retry_policy",
                    "program",
                    "retrieval",
                    "data",
                )
            },
            "execution_runtime": comparison_runtime(run_contract),
        }
    )


def baseline_directory(run_dir: Path, baseline_contract: dict) -> Path:
    """Share one directory across matched ablations, independent of method and budget tags."""
    return run_dir.parent / "hotpotqa-baselines" / baseline_digest(baseline_contract)


def load_baseline_record(run_dir: Path, run_contract: dict) -> dict:
    """Load the completed baseline only when its frozen identity and summary agree."""
    expected = build_baseline_contract(run_contract)
    directory = baseline_directory(run_dir, expected)
    recorded = json.loads((directory / BASELINE_CONTRACT_FILENAME).read_text())
    if recorded != expected:
        raise ValueError("Shared HotPotQA baseline configuration changed.")
    candidate_digest = baseline_digest(expected["candidate"])
    summary = json.loads((directory / "heldout" / candidate_digest / "summary.json").read_text())
    if (
        summary.get("schema_version") != 1
        or summary.get("candidate_sha256") != candidate_digest
        or summary.get("example_count") != expected["data"]["splits"]["test"]["count"]
    ):
        raise ValueError("Shared HotPotQA baseline does not match the initial prompts and test split.")
    for metric in ("exact_match", "f1"):
        value = summary.get(metric)
        if not isinstance(value, float | int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Shared HotPotQA baseline has invalid {metric}.")
    return {
        "contract_sha256": baseline_digest(expected),
        "candidate_sha256": candidate_digest,
        "test_example_count": summary["example_count"],
        "test_exact_match": summary["exact_match"],
        "test_f1": summary["f1"],
    }
