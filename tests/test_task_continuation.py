"""Exercise deadline rollover without GPUs, network providers, or scientific rerolls."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_benchmark_runner import definition

from examples.common import benchmark_runner as runner
from examples.common.continuation import POLICY, AllocationPause, TaskWindows
from examples.common.react_v2 import structured_prompt
from examples.local_api.profile import BENCHMARKS
from gepa.core.adapter import EvaluationBatch


@pytest.fixture
def clock(monkeypatch):
    now = [100]
    monkeypatch.setattr("examples.common.continuation.time.time", lambda: now[0])
    monkeypatch.setenv("GEPA_PAUSE_AT", "200")
    return now


def test_task_windows_resume_positions_preserve_duplicate_ids_and_scores(tmp_path, clock):
    calls = []

    def evaluate(rows, candidate, capture_traces):
        calls.append(deepcopy(rows))
        clock[0] = 201
        return EvaluationBatch(outputs=[{"elapsed_seconds": 1}], scores=[0], trajectories=[{"error": "scored_zero"}])

    adapter = SimpleNamespace(evaluate=evaluate)
    batch = [{"id": "same"}, {"id": "same"}]
    first = TaskWindows(tmp_path, 1)
    with pytest.raises(AllocationPause):
        first.evaluate(adapter, batch, {}, True, ["train", 0], runner.validate_evaluation)
    assert len(calls) == 1
    clock[0] = 100
    result, _ = TaskWindows(tmp_path, 1).evaluate(adapter, batch, {}, True, ["train", 0], runner.validate_evaluation)
    assert len(calls) == 2
    assert result.scores == [0, 0] and len(result.trajectories) == 2
    assert result.num_metric_calls == 2


def test_unknown_task_window_cannot_be_rerun(tmp_path, clock):
    adapter = SimpleNamespace(evaluate=Mock(side_effect=TimeoutError("unknown external outcome")))
    args = (adapter, [{"id": "one"}], {}, True, [], runner.validate_evaluation)
    with pytest.raises(TimeoutError):
        TaskWindows(tmp_path, 1).evaluate(*args)
    with pytest.raises(RuntimeError, match="Unresolved"):
        TaskWindows(tmp_path, 1).evaluate(*args)
    assert adapter.evaluate.call_count == 1


def test_window_contract_and_checksum_drift_are_rejected(tmp_path, clock):
    adapter = SimpleNamespace(evaluate=Mock(return_value=EvaluationBatch(outputs=[{"elapsed_seconds": 1}], scores=[1])))
    args = (adapter, [{"id": "one"}], {}, False, [], runner.validate_evaluation)
    TaskWindows(tmp_path, 1).evaluate(*args)
    with pytest.raises(ValueError, match="order"):
        TaskWindows(tmp_path, 1).evaluate(adapter, [{"id": "changed"}], {}, False, [], runner.validate_evaluation)
    path = next(tmp_path.rglob("window-*.json"))
    record = json.loads(path.read_text())
    record["payload"]["evaluation"]["scores"] = [0]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        TaskWindows(tmp_path, 1).evaluate(*args)
    assert adapter.evaluate.call_count == 1


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_each_benchmark_pauses_and_resumes_a_partial_heldout_repetition(tmp_path, monkeypatch, clock, benchmark):
    original = definition(tmp_path)
    original = replace(
        original,
        testset=[{"id": "test1"}, {"id": "test2"}],
        runtime={"task_continuation": {"policy": POLICY, "width": 1}},
    )
    call = original.adapter.evaluate

    def timed(*args, **kwargs):
        result = call(*args, **kwargs)
        clock[0] = 201
        return result

    monkeypatch.setattr(original.adapter, "evaluate", timed)
    monkeypatch.setenv("GEPA_PAUSE_RECEIPT", str(tmp_path / "pause.json"))
    kwargs = {
        "benchmark_name": benchmark,
        "build_benchmark": lambda *_: original,
        "add_arguments": lambda _: None,
        "argv": ["--mode", "baseline", "--run-dir", str(tmp_path / "run")],
    }
    assert runner.run_cli(**kwargs) == 75
    assert not list(tmp_path.rglob("summary.json"))
    clock[0] = 100
    assert runner.run_cli(**kwargs) == 0
    assert len(original.adapter.calls) == 2
    clock[0] = 100
    assert runner.run_cli(**kwargs) == 0
    assert len(original.adapter.calls) == 2  # One baseline, even across allocations.


@pytest.mark.parametrize("pause_after", [1, 2, 3, 4, 5, 6, 7])
def test_real_optimizer_recovery_never_repeats_completed_work_or_freezes_early(
    tmp_path, monkeypatch, clock, pause_after
):
    original = definition(tmp_path / "run")
    original = replace(
        original, valset=[{"id": "val1"}, {"id": "val2"}], runtime={"task_continuation": {"policy": POLICY, "width": 1}}
    )
    calls = []
    completed = []
    original_evaluate = original.adapter.evaluate

    def evaluate(rows, candidate, **kwargs):
        result = original_evaluate(rows, candidate, **kwargs)
        completed.append((deepcopy(rows), deepcopy(candidate)))
        if len(completed) == pause_after:
            clock[0] = 201
        return result

    monkeypatch.setattr(original.adapter, "evaluate", evaluate)
    improved = structured_prompt("Use improved instructions.", template_family="alibaba")

    def propose(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="deepseek-runtime",
            system_fingerprint="fp-test",
            usage=None,
            _hidden_params={},
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="```\n" + improved + "\n```", reasoning_content="", tool_calls=[]),
                )
            ],
        )

    monkeypatch.setattr("litellm.completion", propose)
    monkeypatch.setenv("GEPA_PAUSE_RECEIPT", str(tmp_path / "pause.json"))

    args = {
        "benchmark_name": "local",
        "build_benchmark": lambda *_: original,
        "add_arguments": lambda _: None,
        "argv": [
            "--run-dir",
            str(tmp_path / "run"),
            "--condition",
            "vanilla",
            "--max-metric-calls",
            "6",
            "--reflection-minibatch-size",
            "1",
            "--solver-api-base",
            "https://solver-old.example/v1",
            "--reflection-api-base",
            "https://proposer-old.example/v1",
        ],
    }
    first = runner.run_cli(**args)
    assert first == 75
    if pause_after <= 5:
        assert not (tmp_path / "run/vanilla/frozen-winner.json").exists()
        assert not list(tmp_path.rglob("heldout/repetition-*.json"))
    clock[0] = 100
    assert runner.run_cli(**args) == 0
    summary = json.loads((tmp_path / "run/vanilla/summary.json").read_text())
    assert summary["winner"]["total_metric_calls"] == 6
    assert summary["winner"]["validation_score"] == 1
    assert summary["test"]["mean_score"] == 1 and summary["baseline"]["mean_score"] == 0
    assert len(calls) == 1  # Real LM response journal, even when pausing after proposal generation.
    assert len(completed) == 8  # 6 optimizer + 1 held-out + 1 baseline, counted once each.
