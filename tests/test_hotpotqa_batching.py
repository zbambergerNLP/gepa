"""Verify batching comparisons preserve the experiment and reject failed requests."""

import json

import pytest

from examples.hotpotqa import batching_report


def profiles(tmp_path, monkeypatch):
    """Build isolated comparison fixtures; calibration integrity has separate tests."""
    roots = []
    rates = {1: 40.0, 2: 75.0, 4: 65.0}
    monkeypatch.setattr(
        batching_report,
        "validate_calibration",
        lambda path, count: {
            "questions_per_hour": rates[int(path.parent.name)],
            "elapsed_seconds": 30,
            "unfinished_allocation_windows": 0,
        },
    )
    for sequences in rates:
        root = tmp_path / str(sequences)
        stage = root / "throughput"
        stage.mkdir(parents=True)
        single = "true" if sequences == 1 else "false"
        contract = {
            "runtime": {
                "data": {"train": "unchanged"},
                "models": {"max_tokens": 16384},
                "execution_runtime": {
                    "source_commit": "a" * 40,
                    "campaign_id": f"pilot-{sequences}",
                    "serve_arguments": f"tp=1;max_num_seqs={sequences};single_sequence_replicas={single}",
                    "vllm_single_sequence_replicas": single,
                },
            },
            "workers": 12,
            "examples": list(range(12)),
        }
        (stage / "pilot-contract.json").write_text(json.dumps(contract))
        (stage / "provider-attempts.jsonl").write_text('{"outcome":"success"}\n')
        roots.append(root)
    return roots


def test_rank_uses_completed_questions_and_does_not_qualify_production(tmp_path, monkeypatch):
    """More concurrency need not be faster and a timing result is not qualification."""
    result = batching_report.compare_profiles(profiles(tmp_path, monkeypatch))
    assert result["fastest_measured_profile"]["active_requests"] == 2
    assert result["production_qualified"] is False
    assert result["operational_review_required"]


@pytest.mark.parametrize("damage", ["data", "models", "workers", "failed_request", "duplicate"])
def test_incomparable_or_failed_profiles_cannot_win(tmp_path, monkeypatch, damage):
    """Reject faster measurements obtained by changing the workload or hiding failures."""
    roots = profiles(tmp_path, monkeypatch)
    path = roots[1] / "throughput" / "pilot-contract.json"
    contract = json.loads(path.read_text())
    if damage in ("data", "models"):
        contract["runtime"][damage] = {"changed": True}
    elif damage == "workers":
        contract["workers"] = 99
    elif damage == "failed_request":
        (path.parent / "provider-attempts.jsonl").write_text('{"outcome":"error"}\n')
    else:
        roots.append(roots[0])
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        batching_report.compare_profiles(roots)
