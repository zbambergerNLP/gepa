"""Verify offline reporting stays observational and preserves scientific labels."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.hotpotqa.tracking import HotpotqaWandb, _digest, provider_usage, report_completed


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    run = Mock()
    run.id = "offline-test"
    run.dir = str(tmp_path / "wandb" / "run" / "files")
    run.summary = {}
    module = Mock()
    module.init.return_value = run
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module, run


def test_progress_never_claims_heldout_completion_and_cannot_change_state(tmp_path, sdk):
    module, run = sdk
    state = SimpleNamespace(
        total_num_evals=300, prog_candidate_val_subscores=[{0: 0, 1: 1}], program_candidates=[{"sys": "original"}]
    )
    observer = HotpotqaWandb(tmp_path, {"condition": "react_v2", "optimizer": {"max_metric_calls": 6871}}, "project")
    observer.on_iteration_start({"state": state})
    observer.on_iteration_start({"state": state})
    assert run.log.call_count == 1
    assert module.init.call_args.kwargs["mode"] == "offline"
    assert run.summary["heldout_complete"] is False
    assert state.program_candidates == [{"sys": "original"}] and state.total_num_evals == 300
    run.log.side_effect = RuntimeError("Telemetry unavailable")
    state.total_num_evals = 306
    observer.on_iteration_end({"state": state, "iteration": 1, "proposal_accepted": False})
    assert (tmp_path / "tracking/errors.jsonl").exists()


def test_completed_backfill_checks_winner_and_is_idempotent(tmp_path, sdk):
    module, run = sdk
    contract = {
        "condition": "vanilla",
        "optimizer": {"max_metric_calls": 6871},
        "execution_runtime": {"campaign_id": "study"},
    }
    candidates = [{"sys": "original"}, {"sys": "winner"}]
    payload = {
        "run_contract": contract,
        "candidates": candidates,
        "best_idx": 1,
        "val_aggregate_scores": [0.5, 0.75],
        "discovery_eval_counts": [0, 306],
        "total_metric_calls": 6930,
    }
    final = {
        "candidate_sha256": _digest(candidates[1]),
        "best_validation_exact_match": 0.75,
        "test_exact_match": 0.7,
        "test_f1": 0.8,
        "test_exact_match_gain": 0.2,
        "baseline": {"test_exact_match": 0.5},
    }
    for name, obj in [
        ("wikipedia-run-contract.json", contract),
        ("candidates.json", payload),
        ("final_metrics.json", final),
    ]:
        (tmp_path / name).write_text(json.dumps(obj))
    proof = report_completed(tmp_path, "project", "entity")
    assert report_completed(tmp_path, "project", "entity") == proof
    assert module.init.call_count == 1
    assert run.summary["heldout/exact_match"] == 0.7
    assert run.summary["optimization/metric_calls"] == 6930
    assert run.summary["heldout_complete"] is True
    final["candidate_sha256"] = _digest(candidates[0])
    (tmp_path / "final_metrics.json").write_text(json.dumps(final))
    with pytest.raises(ValueError, match="winner"):
        report_completed(tmp_path, "project", "entity")


def test_usage_keeps_allocations_and_missing_token_counts_separate(tmp_path):
    path = tmp_path / "provider-attempts.jsonl"
    rows = [
        {
            "allocation_job_id": "old",
            "role": "solver",
            "requested_model": "Qwen",
            "outcome": "error",
            "prompt_tokens": None,
        },
        {
            "allocation_job_id": "new",
            "role": "solver",
            "requested_model": "Qwen",
            "outcome": "success",
            "prompt_tokens": 40,
            "completion_tokens": 10,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    totals = provider_usage(path)
    assert totals["old/solver/Qwen"]["transport_errors"] == 1
    assert totals["new/solver/Qwen"]["transport_errors"] == 0
    assert totals["old/solver/Qwen"]["prompt_tokens_unreported_calls"] == 1
    assert totals["new/solver/Qwen"]["completion_tokens"] == 10
