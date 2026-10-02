"""Report pilot readiness and measured cross-model request overlap."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from examples.common.pilot_checks import METHODS, atomic_json, digest, load_cycle
from examples.hotpotqa.pilot import validate_calibration
from examples.terminalbench.token_usage import summarize_usage
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG


def request_intervals(path: Path) -> list[tuple[float, float]]:
    """Merge recorded physical request intervals, excluding idle gaps."""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    intervals = sorted(
        (
            datetime.fromisoformat(row["timestamp"]).timestamp() - row["elapsed_seconds"],
            datetime.fromisoformat(row["timestamp"]).timestamp(),
        )
        for row in rows
    )
    if any(not math.isfinite(start) or not math.isfinite(end) or start > end for start, end in intervals):
        raise ValueError("Invalid provider request timing")
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_seconds(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> float:
    """Count intersection time between two unions of actual request intervals."""
    i = j = 0
    overlap = 0.0
    while i < len(first) and j < len(second):
        overlap += max(0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def report(root: Path) -> dict:
    """Retain incomplete evidence while withholding runtime qualification."""
    models = {}
    intervals = []
    shared_identity = None
    for profile in ("qwen3.8-27b", "deepseek-v4.1-flash"):
        directory = root / profile
        evidence: dict = {"complete": False, "issues": []}
        try:
            runtime = json.loads((directory / "full" / "pilot-contract.json").read_text())["runtime"]
            evidence["runtime"] = runtime
            identity = digest(
                {
                    "data": runtime["data"],
                    "retrieval": runtime["retrieval"],
                    "source": runtime["execution_runtime"]["source_commit"],
                    "campaign": runtime["execution_runtime"]["campaign_id"],
                }
            )
            if shared_identity is not None and identity != shared_identity:
                raise ValueError("Model pilots use different source, data, retrieval, or campaign identities")
            shared_identity = identity
        except (OSError, ValueError, KeyError) as exc:
            evidence["issues"].append(f"runtime: {exc}")
        for stage, count in (("smoke", 3), ("full", 150)):
            try:
                evidence[stage] = validate_calibration(directory / stage, count)
            except (OSError, ValueError, KeyError) as exc:
                evidence["issues"].append(f"{stage}: {exc}")
        evidence["optimizer_checks"] = {}
        for method in METHODS:
            try:
                evidence["optimizer_checks"][method] = load_cycle(directory / "optimizer" / method)
            except (OSError, ValueError, KeyError) as exc:
                evidence["issues"].append(f"{method}: {exc}")
        try:
            logs = list(directory.rglob(PROVIDER_ATTEMPT_LOG))
            evidence["usage"] = summarize_usage(logs)
            for check in (
                directory / "smoke",
                directory / "full",
                *(directory / "optimizer" / method for method in METHODS),
            ):
                usage_file = check / PROVIDER_ATTEMPT_LOG
                if not usage_file.exists() or not usage_file.stat().st_size:
                    evidence["issues"].append(f"Missing provider usage: {usage_file}")
            evidence["allocation_job_ids"] = sorted(
                {
                    row["allocation_job_id"]
                    for path in logs
                    for line in path.read_text().splitlines()
                    if (row := json.loads(line)).get("allocation_job_id") is not None
                }
            )
            intervals.append(request_intervals(directory / "full" / PROVIDER_ATTEMPT_LOG))
        except (OSError, ValueError, KeyError) as exc:
            evidence["issues"].append(f"usage: {exc}")
            intervals.append([])
        evidence["complete"] = not evidence["issues"]
        models[profile] = evidence
    overlap = overlap_seconds(*intervals)
    return {
        "schema_version": 1,
        "models": models,
        "complete": all(model["complete"] for model in models.values()),
        "full_stage_request_overlap_seconds": overlap,
        "full_stage_requests_overlapped": overlap > 0,
        "overlap_basis": "recorded request intervals, including server queueing; not GPU kernel timing",
        "schedule_decision": "review_required",
        "metric_improvement_required": False,
    }


def main(argv: list[str] | None = None) -> None:
    """Write a readiness report without treating partial results as a failed fetch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(args.root)
    atomic_json(args.output, result)
    print(
        f"Pilot completion: {result['complete']}; full-stage request overlap: {result['full_stage_request_overlap_seconds']:.1f}s"
    )
    print(f"Review: {args.output}")


if __name__ == "__main__":
    main()
