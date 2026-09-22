"""Verify offline reporting with real checkpoints and deterministic held-out evidence."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from test_terminalbench_evaluation import _fake_runner, _write_run

from examples.common.recovery import file_digest
from examples.terminalbench import evaluate
from examples.terminalbench import main as campaign
from examples.terminalbench.tracking import TerminalbenchWandb, report_completed


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    runs = []

    def init(**kwargs):
        run = Mock()
        run.id = kwargs["id"]
        run.dir = str(tmp_path / "wandb" / run.id / "files")
        run.summary = {}
        runs.append(run)
        return run

    module = Mock()
    module.init.side_effect = init
    module.login.side_effect = AssertionError("Compute-node logging must not authenticate")
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module, runs


def test_offline_progress_is_observational_and_keeps_allocations_separate(tmp_path, monkeypatch, sdk):
    module, runs = sdk
    contract = {"optimization_scope": "system_prompt", "condition": "react_v2", "budget": "double"}
    state = SimpleNamespace(
        total_num_evals=300,
        prog_candidate_val_subscores=[{0: 0.0, 1: 1.0}],
        program_candidates=[{"instruction_prompt": "original"}],
    )
    before = deepcopy(state)
    monkeypatch.setenv("WANDB_MODE", "online")
    for job in ("first", "replacement"):
        monkeypatch.setenv("SLURM_JOB_ID", job)
        tracker = TerminalbenchWandb(tmp_path, contract, "forest-terminalbench", "gilad-mo12", "campaign")
        tracker.on_iteration_start({"state": state})
        tracker.on_iteration_start({"state": state})
        assert runs[-1].log.call_count == 1
        assert runs[-1].log.call_args.args[0]["validation/best_pass_at_1"] == 0.5
        tracker.on_optimization_end({"final_state": state})
        tracker.finish()
        assert runs[-1].summary["optimization_complete"] is True
        assert runs[-1].summary["heldout_complete"] is False
        assert runs[-1].summary["completed_ablation"] is False
        runs[-1].finish.assert_called_once()
    assert state == before
    assert runs[0].id != runs[1].id
    assert all(call.kwargs["mode"] == "offline" for call in module.init.call_args_list)
    assert [call.kwargs["config"]["allocation"] for call in module.init.call_args_list] == ["first", "replacement"]
    assert {call.kwargs["group"] for call in module.init.call_args_list} == {"campaign"}
    module.login.assert_not_called()


def test_reporting_failures_and_disabled_logging_do_not_break_experiments(tmp_path, sdk):
    module, runs = sdk
    contract = {"optimization_scope": "system_prompt", "condition": "vanilla", "budget": "standard"}
    disabled = TerminalbenchWandb(tmp_path, contract, None)
    disabled.on_iteration_start({})
    disabled.finish()
    module.init.assert_not_called()
    assert not (tmp_path / "tracking").exists()
    tracker = TerminalbenchWandb(tmp_path, contract, "project")
    runs[-1].log.side_effect = RuntimeError("Local reporting failed")
    tracker.on_iteration_start(
        {
            "state": SimpleNamespace(
                total_num_evals=10,
                prog_candidate_val_subscores=[{0: 1}],
                program_candidates=[{"prompt": "seed"}],
            )
        }
    )
    tracker.finish()
    assert runs[-1].summary["optimization_complete"] is False
    assert runs[-1].summary["heldout_complete"] is False
    assert json.loads((tmp_path / "tracking/errors.jsonl").read_text())["error_type"] == "RuntimeError"
    runs[-1].finish.assert_called_once()


def test_optimization_cli_flushes_offline_tracking_when_the_optimizer_fails(tmp_path, monkeypatch, sdk):
    _, runs = sdk
    monkeypatch.setattr(campaign.HarborCLI, "check_requirements", Mock())

    def interrupted(**kwargs):
        assert any(isinstance(callback, TerminalbenchWandb) for callback in kwargs["callbacks"])
        raise RuntimeError("Interrupted optimization")

    monkeypatch.setattr(campaign, "optimize", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted optimization"):
        campaign.main(
            [
                "--condition",
                "vanilla",
                "--run-dir",
                str(tmp_path / "run"),
                "--harbor-work-dir",
                str(tmp_path / "harbor"),
                "--train-limit",
                "1",
                "--val-limit",
                "1",
                "--wandb-project",
                "project",
                "--wandb-entity",
                "entity",
                "--wandb-group",
                "campaign",
            ]
        )
    assert len(runs) == 1
    runs[0].finish.assert_called_once()
    assert runs[0].summary["heldout_complete"] is False


def test_real_sdk_writes_offline_files_without_login_or_api_key(tmp_path, monkeypatch, completed):
    wandb = pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent-netrc"))
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(wandb, "login", Mock(side_effect=AssertionError("No login on compute nodes")))
    contract = {"optimization_scope": "system_prompt", "condition": "vanilla", "budget": "standard"}
    tracker = TerminalbenchWandb(tmp_path, contract, "offline-regression", "local-tests")
    assert tracker.run is not None
    assert tracker.run.settings.mode == "offline"
    offline = Path(tracker.run.dir).parent
    tracker.on_optimization_end(
        {
            "final_state": SimpleNamespace(
                total_num_evals=300,
                prog_candidate_val_subscores=[{0: 0.0, 1: 1.0}],
                program_candidates=[{"instruction_prompt": "seed"}],
            )
        }
    )
    tracker.finish()
    assert list(offline.glob("run-*.wandb"))
    assert list((offline / "files/media/table").glob("*.json"))
    assert not (tmp_path / "tracking/errors.jsonl").exists()
    directory, heldout, cell = completed
    proof = report_completed(directory, heldout, cell, "offline-regression", "local-tests")
    result_offline = Path(proof["offline_directory"])
    assert list(result_offline.glob("run-*.wandb"))
    for name, digest in proof["input_sha256"].items():
        assert file_digest(result_offline / "files/evidence" / name) == digest
    assert not (directory / "tracking/errors.jsonl").exists()


@pytest.fixture
def completed(tmp_path):
    directory = _write_run(tmp_path / "runs", "tb2.1", "react_v2", optimization_scope="system_prompt")
    cell = "system_prompt__react_v2"
    manifest, comparison = evaluate.freeze_comparison({cell: directory})
    heldout = tmp_path / "heldout"
    harbor = _fake_runner(manifest, comparison, heldout)
    evaluate.evaluate_comparison(manifest, comparison, heldout, harbor)
    return directory, heldout, cell


def test_completed_report_checks_and_copies_evidence_without_inference(completed, sdk):
    directory, heldout, cell = completed
    module, runs = sdk
    proof = report_completed(directory, heldout, cell, "project", "entity", "campaign")
    assert report_completed(directory, heldout, cell, "project", "entity", "campaign") == proof
    assert len(runs) == 1
    assert runs[0].summary["completed_ablation"] is True
    assert runs[0].summary["heldout_complete"] is True
    assert runs[0].summary["heldout/pass_at_1_gain"] == 0.0
    assert runs[0].summary["test_repetitions"] == 3
    for name, digest in proof["input_sha256"].items():
        assert file_digest(Path(runs[0].dir) / "evidence" / name) == digest
    assert len(proof["input_sha256"]) == 9  # Contract, frozen comparison, summary, six repetitions.
    # Extension of the aggregate summary must not republish an unchanged cell.
    path = heldout / "summary.json"
    summary = json.loads(path.read_text())
    summary["completed_cells"].append("system_prompt__vanilla")
    path.write_text(json.dumps(summary))
    assert report_completed(directory, heldout, cell, "project", "entity", "campaign") == proof
    assert len(runs) == 1
    report_completed(directory, heldout, cell, "another-project", "entity", "campaign")
    assert len(runs) == 2
    module.login.assert_not_called()


@pytest.mark.parametrize("damage", ["incomplete", "wrong_winner", "missing_task", "reused_evaluation", "wrong_summary"])
def test_result_reporting_rejects_incomplete_or_changed_results(completed, sdk, damage):
    directory, heldout, cell = completed
    module, _ = sdk
    if damage == "incomplete":
        path = heldout / "summary.json"
        row = json.loads(path.read_text())
        row["complete"] = False
    elif damage == "wrong_winner":
        path = heldout / "frozen-comparison.json"
        row = json.loads(path.read_text())
        row["harnesses"][cell]["candidate_digest"] = "changed"
    elif damage == "wrong_summary":
        path = heldout / "summary.json"
        row = json.loads(path.read_text())
        row["harnesses"][cell]["mean_pass_at_1"] = 0.123
    else:
        path = heldout / f"{cell}-repetition-1.json"
        row = json.loads(path.read_text())
        if damage == "missing_task":
            row["scores"].pop(next(iter(row["scores"])))
        else:
            row["evaluation_id"] = json.loads((heldout / "initial-repetition-1.json").read_text())["evaluation_id"]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        report_completed(directory, heldout, cell, "project")
    module.init.assert_not_called()
    assert not (directory / "tracking").exists()
