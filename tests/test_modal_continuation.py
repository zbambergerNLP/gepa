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
from examples.modal import cli
from examples.modal.config import BENCHMARKS, validate_plan
from examples.modal.continuation import allocation_directory, runtime_identity, snapshot_state, verify_state
from examples.modal.worker import seal_files
from gepa.core.adapter import EvaluationBatch
from gepa.response_journal import canonical_request_digest, register_verified_remote_endpoint


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

    def runtime(_path, models, **kwargs):
        cloud = {"provider": "modal", "source_sha256": "source", "driver": {"image_id": "driver"}}
        for role in ("solver", "proposer"):
            cloud[role] = {"identity": {"sha256": role + "-runtime"}, "image_id": role, "source_sha256": "source"}
            register_verified_remote_endpoint(getattr(models, role + "_api_base"), cloud[role])
        return cloud

    monkeypatch.setattr("examples.common.remote_runtime.load_execution_runtime", runtime)
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
            "--execution-runtime",
            str(tmp_path / "runtime.json"),
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
    args["argv"] = [arg.replace("-old.example", "-new.example") for arg in args["argv"]]
    assert runner.run_cli(**args) == 0
    summary = json.loads((tmp_path / "run/vanilla/summary.json").read_text())
    assert summary["winner"]["total_metric_calls"] == 6
    assert summary["winner"]["validation_score"] == 1
    assert summary["test"]["mean_score"] == 1 and summary["baseline"]["mean_score"] == 0
    assert len(calls) == 1  # Real LM response journal, even when pausing after proposal generation.
    assert len(completed) == 8  # 6 optimizer + 1 held-out + 1 baseline, counted once each.


def test_only_authenticated_runtime_registration_allows_endpoint_rotation():
    first, second = "https://first.example/v1", "https://second.example/v1"
    request = {"api_base": first, "messages": ["same"], "extra_headers": {"X-Gepa-Runtime-Sha256": "a" * 64}}
    assert canonical_request_digest(request) != canonical_request_digest({**request, "api_base": second})
    identity = {"identity": {"sha256": "a" * 64}, "image_id": "same", "source_sha256": "same"}
    register_verified_remote_endpoint(first, identity)
    register_verified_remote_endpoint(second, identity)
    assert canonical_request_digest(request) == canonical_request_digest({**request, "api_base": second})
    register_verified_remote_endpoint(second, {**identity, "image_id": "different"})
    assert canonical_request_digest(request) != canonical_request_digest({**request, "api_base": second})


def test_allocation_snapshot_remains_immutable_and_rejects_changed_checkpoint(tmp_path):
    directory = tmp_path / allocation_directory(0)
    directory.mkdir(parents=True)
    checkpoint = tmp_path / "gepa_state.bin"
    checkpoint.write_bytes(b"first allocation checkpoint")
    proof = snapshot_state(tmp_path, directory)
    verify_state(tmp_path, directory)
    assert proof["files"]["gepa_state.bin"]
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_state(tmp_path, directory)


def test_runtime_continuation_preserves_material_fields():
    old = {
        "source_sha256": "source",
        "driver": {"image_id": "driver"},
        "roles": {"solver": {"identity": "model", "image_id": "image", "api_base": "first", "allocation": "old"}},
    }
    new = deepcopy(old)
    new["roles"]["solver"].update(api_base="second", allocation="new")
    assert runtime_identity(old) == runtime_identity(new)
    new["roles"]["solver"]["identity"] = "changed"
    assert runtime_identity(old) != runtime_identity(new)


@pytest.mark.parametrize("state,exit_code", [("failed", 0), ("completed", 0), ("paused", None), ("paused", 1)])
def test_resume_refuses_failure_completion_running_or_unknown_before_allocation(
    tmp_path, monkeypatch, state, exit_code
):
    plan = {
        "schema_version": 1,
        "benchmark": "decisionbench",
        "run_id": "test-run",
        "mode": "pilot",
        "timeout": 86400,
        "arguments": [],
        "secrets": [],
        "terminalbench_scope": "della88",
        "embedding": {"device": "cpu", "attention": "sdpa", "dtype": "float32", "batch_size": 1},
    }
    validate_plan(plan)
    receipt = {
        "schema_version": 2,
        "phase": "run",
        "epoch": 0,
        "plan": plan,
        "allocations": {"solver": {"id": "solver"}, "driver": {"id": "driver"}},
        "results_volume": "test",
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    fake = SimpleNamespace(
        Sandbox=SimpleNamespace(from_id=lambda _: SimpleNamespace(poll=lambda: exit_code)),
        Volume=SimpleNamespace(from_name=Mock()),
    )
    monkeypatch.setattr(cli, "sdk", lambda: fake)
    monkeypatch.setattr(cli, "read_volume_json", lambda *args: {"state": state})
    launch = Mock()
    monkeypatch.setattr(cli, "start_run_workers", launch)
    with pytest.raises(ValueError):
        cli.resume(path, tmp_path / "new.json")
    launch.assert_not_called()
    assert not (tmp_path / "new.json").exists()


def test_follow_is_bounded_and_does_not_restart_completion(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"epoch": 3}))
    monkeypatch.setattr(
        cli,
        "inspect_receipt",
        lambda _: {"allocations": {"driver": {"state": "terminal", "exit_code": 0}}, "worker": {"state": "paused"}},
    )
    launch = Mock()
    monkeypatch.setattr(cli, "resume", launch)
    assert cli.follow(path, 1) == path
    launch.assert_not_called()
    monkeypatch.setattr(cli, "inspect_receipt", lambda _: {"allocations": {}, "worker": {"state": "completed"}})
    assert cli.follow(path, 10) == path
    launch.assert_not_called()


def test_resume_reuses_verified_source_budget_and_volumes_with_exclusive_cloud_claim(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"frozen source")
    plan = {
        "schema_version": 1,
        "benchmark": "decisionbench",
        "run_id": "test-run",
        "mode": "pilot",
        "timeout": 86400,
        "arguments": ["--max-metric-calls", "6871"],
        "secrets": [],
        "terminalbench_scope": "della88",
        "source_sha256": cli.file_digest(source),
        "embedding": {"device": "cpu", "attention": "sdpa", "dtype": "float32", "batch_size": 1},
    }
    results = tmp_path / "results"
    directory = results / allocation_directory(0)
    (directory / "solver").mkdir(parents=True)
    (directory / "solver/runtime.json").write_text("{}")
    seal_files(directory / "solver", directory / "solver/serving-files.json")
    status = {"state": "paused", "epoch": 0, "returncode": 75, "plan_sha256": cli.digest(plan)}
    for path in (results / "run-status.json", directory / "run-status.json"):
        path.write_text(json.dumps(status))
    (directory / "execution.json").write_text(json.dumps({"roles": {"solver": {"identity": "pinned-runtime"}}}))
    seal_files(directory, directory / "run-files.json", exclude=("solver",))
    previous = {
        "schema_version": 2,
        "phase": "run",
        "plan": plan,
        "epoch": 0,
        "source_archive": str(source),
        "source": {"sha256": cli.file_digest(source)},
        "results_volume": "results",
        "data_volume": "data",
        "models_volume": "models",
        "image_id": "driver-image",
        "serving_images": {"solver": "solver-image"},
        "preparation_receipt": "original-preparation",
        "allocations": {"solver": {"id": "old-solver"}, "driver": {"id": "old-driver"}},
    }
    original = tmp_path / "original.json"
    original.write_text(json.dumps(previous))
    volume = SimpleNamespace(read_file=lambda path: iter([(results / path.lstrip("/")).read_bytes()]))
    claim = Mock(side_effect=[True, False])
    fake = SimpleNamespace(
        Sandbox=SimpleNamespace(from_id=lambda _: SimpleNamespace(poll=lambda: 0)),
        Volume=SimpleNamespace(from_name=lambda _: volume),
        Dict=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(put=claim)),
        App=SimpleNamespace(lookup=lambda *a, **k: "app"),
    )
    monkeypatch.setattr(cli, "sdk", lambda: fake)
    launch = Mock()
    monkeypatch.setattr(cli, "start_run_workers", launch)
    output = tmp_path / "next.json"
    cli.resume(original, output)
    after = json.loads(output.read_text())
    for key in ("source", "plan", "results_volume", "data_volume", "models_volume", "image_id", "serving_images"):
        assert after[key] == previous[key]
    assert after["epoch"] == 1 and after["expected_runtimes"] == {"solver": "pinned-runtime"}
    assert after["allocation_history"] == [{"epoch": 0, "allocations": previous["allocations"]}]
    with pytest.raises(RuntimeError, match="already claimed"):
        cli.resume(original, tmp_path / "duplicate.json")
    assert launch.call_count == 1
    with pytest.raises(FileExistsError):
        cli.resume(original, output)
    assert launch.call_count == 1
