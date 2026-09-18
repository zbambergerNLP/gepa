"""Reject cross-allocation readiness, overlapping devices, and swapped model roles."""

from copy import deepcopy

import pytest

from examples.hotpotqa.main import TEACHER_RUNTIME_KEYS
from examples.hotpotqa.paired_runtime import combine_runtimes, split_devices


@pytest.mark.parametrize("devices", [[], ["GPU-a"] * 5, [f"GPU-{i}" for i in range(4)], [str(i) for i in range(5)]])
def test_exact_five_distinct_allocated_devices(devices):
    with pytest.raises(ValueError, match="five distinct"):
        split_devices(devices)


def readiness():
    """Construct the two explicit verified role records from one allocation."""
    allocation = split_devices([f"GPU-{i}" for i in range(5)])
    shared = dict.fromkeys(
        (
            "HOTPOTQA_SOURCE_COMMIT",
            "HOTPOTQA_SOURCE_MANIFEST_SHA256",
            "HOTPOTQA_GEPA_ENV_SHA256",
            "HOTPOTQA_CAMPAIGN_ID",
            "WIKI17_INTEGRITY_SHA256",
        ),
        "same",
    )
    records = []
    for profile, model, port in (
        ("qwen3.8-27b", "Qwen/Qwen3.8-27B", 8000),
        ("deepseek-v4.1-flash", "deepseek-ai/DeepSeek-V4.1-Flash", 8001),
    ):
        records.append(
            {
                "job_id": "123",
                "profile": profile,
                "gpu_uuids": allocation[profile],
                "environment": {
                    **shared,
                    **{"HOTPOTQA_" + key: profile for key in TEACHER_RUNTIME_KEYS},
                    "SOLVER_MODEL": "hosted_vllm/" + model,
                    "SOLVER_API_BASE": f"http://127.0.0.1:{port}/v1",
                    "GEN_PID": "456",
                },
            }
        )
    return records, allocation


def test_routes_teacher_and_preserves_both_runtime_identities():
    records, allocation = readiness()
    before = deepcopy(records)
    env = combine_runtimes(*records, allocation, "123")
    assert env["SOLVER_MODEL"] == "hosted_vllm/Qwen/Qwen3.8-27B"
    assert env["REFLECTION_MODEL"] == "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
    assert env["SOLVER_API_BASE"] != env["REFLECTION_API_BASE"]
    assert "deepseek-v4.1-flash" in env["HOTPOTQA_TEACHER_RUNTIME"]
    assert records == before
    assert set(allocation["qwen3.8-27b"]).isdisjoint(allocation["deepseek-v4.1-flash"])


@pytest.mark.parametrize("damage", ["job", "gpu", "source", "model", "endpoint"])
def test_stale_or_miswired_teacher_cannot_run(damage):
    records, allocation = readiness()
    teacher = records[1]
    if damage == "job":
        teacher["job_id"] = "122"
    elif damage == "gpu":
        teacher["gpu_uuids"] = allocation["qwen3.8-27b"]
    elif damage == "source":
        teacher["environment"]["HOTPOTQA_SOURCE_COMMIT"] = "other"
    elif damage == "model":
        teacher["environment"]["SOLVER_MODEL"] = records[0]["environment"]["SOLVER_MODEL"]
    else:
        teacher["environment"]["SOLVER_API_BASE"] = records[0]["environment"]["SOLVER_API_BASE"]
    with pytest.raises(ValueError):
        combine_runtimes(*records, allocation, "123")
