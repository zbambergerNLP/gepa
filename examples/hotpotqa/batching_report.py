"""Compare training throughput with all non-batching experiment settings fixed."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from examples.common.pilot_checks import atomic_json, digest
from examples.hotpotqa.pilot import PILOT_PROTOCOL, validate_calibration
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG


def compare_profiles(directories: list[Path]) -> dict:
    """Rank completed training measurements and reject incomparable profiles.

    Args:
        directories: Pilot roots from the same model and ordered training sample.

    Returns:
        Measured throughput, the fastest candidate, and required operational review.

    Raises:
        ValueError: Evidence is incomplete, incomparable, or contains duplicate profiles.
    """
    rows = []
    shared = None
    seen = set()
    for directory in directories:
        stage = directory / "throughput"
        summary = validate_calibration(stage, PILOT_PROTOCOL["throughput"])
        if summary["unfinished_allocation_windows"]:
            raise ValueError(f"Incomplete timing window in {stage}; repeat the throughput measurement")
        contract = json.loads((stage / "pilot-contract.json").read_text())
        comparison = deepcopy(contract)
        execution = comparison["runtime"]["execution_runtime"]
        arguments = dict(item.split("=", 1) for item in execution["serve_arguments"].split(";"))
        sequences = int(arguments.pop("max_num_seqs"))
        single = "true" if sequences == 1 else "false"
        if sequences not in (1, 2, 4) or sequences in seen:
            raise ValueError("Profiles must use distinct active-request limits from 1, 2, and 4")
        if (
            arguments.pop("single_sequence_replicas") != single
            or execution.pop("vllm_single_sequence_replicas") != single
        ):
            raise ValueError("Serving metadata disagrees about the active-request limit")
        seen.add(sequences)
        execution.pop("campaign_id")
        execution["serve_arguments"] = arguments
        identity = digest(comparison)
        if shared is not None and identity != shared:
            raise ValueError("Batching measurements changed source, data, prompts, workers, or another runtime setting")
        shared = identity
        usage = stage / PROVIDER_ATTEMPT_LOG
        attempts = [json.loads(line) for line in usage.read_text().splitlines()]
        if not attempts or any(row.get("outcome") != "success" for row in attempts):
            raise ValueError(f"Missing or failed physical model requests in {usage}")
        rows.append(
            {
                "directory": str(directory),
                "active_requests": sequences,
                "questions_per_hour": summary["questions_per_hour"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "evidence_sha256": digest(summary),
            }
        )
    if not rows:
        raise ValueError("At least one completed profile is required")
    best = max(rows, key=lambda row: (row["questions_per_hour"], -row["active_requests"]))
    return {
        "profiles": rows,
        "fastest_measured_profile": best,
        "operational_review_required": ["GPU preemptions", "GPU and host memory", "queueing and request latency"],
        "production_qualified": False,
        "scope": "Twelve fixed training questions; full calibration and optimizer checks are separate",
    }


def main() -> None:
    """Write a reproducible comparison without promoting a profile to production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_profiles(args.directories)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
