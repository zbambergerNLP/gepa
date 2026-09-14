"""Exercise allocation continuation without submitting real cluster jobs."""

import json
from pathlib import Path

import pytest

from examples.common import slurm_continuation as controller
from examples.common.recovery import file_digest, seal_progress, snapshot


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Create immutable runtime inputs and a private checkpoint registry."""
    source = "a" * 40
    export = tmp_path / "worker.env"
    export.write_bytes(b"BUDGET=6871\0")
    registry = tmp_path / "registry.json"
    monkeypatch.setenv("GEPA_RECOVERY_REGISTRY", str(registry))
    monkeypatch.setenv("HOTPOTQA_SOURCE_COMMIT", source)
    checkpoint = tmp_path / "run" / "gepa_state.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"saved state")
    seal_progress(checkpoint.parent, 3, [checkpoint])
    plan = {
        "source_commit": source,
        "source_dir": str(tmp_path),
        "index": 0,
        "status": "draft",
        "cells": [
            {
                "name": "first",
                "command": ["sbatch", "--parsable", "--export-file", str(export), "worker.sh"],
                "export_file": str(export),
                "export_sha256": file_digest(export),
                "registry": str(registry),
                "error_file": str(tmp_path / "error.json"),
            }
        ],
    }
    return tmp_path / "plan.json", plan, checkpoint


def test_timeout_resumes_identical_command_only_after_saved_progress(campaign, monkeypatch):
    """A new allocation preserves the environment and receives a watcher before release."""
    path, plan, checkpoint = campaign
    commands = []
    ids = iter(["10", "11", "12", "13"])

    def run(command):
        commands.append(command)
        return next(ids) if command[0] == "sbatch" else ""

    monkeypatch.setattr(controller, "run", run)
    monkeypatch.setattr(controller, "accounting", lambda _: ("TIMEOUT", "0:15"))
    controller.dispatch(path, plan)
    seal_progress(checkpoint.parent, 6, [checkpoint])
    controller.advance(path, plan, "10")
    assert commands[0] == commands[3]
    assert "--dependency=afterany:10" in commands[1]
    assert commands[2] == ["scontrol", "release", "10"]
    assert plan["index"] == 0 and plan["worker"] == "12"
    monkeypatch.setattr(controller, "accounting", lambda _: ("COMPLETED", "0:0"))
    controller.advance(path, plan, "12")
    assert plan["status"] == "complete"
    previous = json.dumps(plan)
    controller.advance(path, plan, "10")
    assert json.dumps(plan) == previous


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"])
def test_execution_failures_do_not_resubmit(campaign, monkeypatch, state):
    """Only allocation time expiry is eligible for automatic continuation."""
    path, plan, _ = campaign
    plan.update(status="active", worker="10", before={})
    monkeypatch.setattr(controller, "accounting", lambda _: (state, "1:0"))
    monkeypatch.setattr(controller, "run", lambda _: pytest.fail("Unexpected submission"))
    with pytest.raises(RuntimeError, match="stopped"):
        controller.advance(path, plan, "10")


def test_timeout_with_no_progress_or_corruption_stops(campaign, monkeypatch):
    """Logs and scores cannot replace new, intact checkpointed work."""
    path, plan, checkpoint = campaign
    registry = Path(plan["cells"][0]["registry"])
    plan.update(status="active", worker="10", before=snapshot(registry, plan["source_commit"]))
    monkeypatch.setattr(controller, "accounting", lambda _: ("TIMEOUT", "0:15"))
    with pytest.raises(RuntimeError, match="without new"):
        controller.advance(path, plan, "10")
    checkpoint.write_bytes(b"partial replacement")
    with pytest.raises(ValueError, match="incomplete"):
        controller.advance(path, plan, "10")


def test_reported_error_blocks_timeout_recovery(campaign, monkeypatch):
    """A timeout must not hide a recorded runtime or parser failure."""
    path, plan, _ = campaign
    plan.update(status="active", worker="10", before={})
    Path(plan["cells"][0]["error_file"]).write_text("{}")
    monkeypatch.setattr(controller, "accounting", lambda _: ("TIMEOUT", "0:15"))
    with pytest.raises(RuntimeError, match="unresolved"):
        controller.advance(path, plan, "10")


def test_ambiguous_submission_is_not_retried(campaign, monkeypatch):
    """Persist an uncertain submission so a caller cannot blindly duplicate it."""
    path, plan, _ = campaign
    calls = []
    monkeypatch.setattr(controller, "run", lambda command: calls.append(command) or "unparseable")
    with pytest.raises(RuntimeError, match="Ambiguous"):
        controller.dispatch(path, plan)
    assert len(calls) == 1
    assert json.loads(path.read_text())["status"] == "submitting"


def test_repeated_start_cannot_duplicate_or_stop_an_active_campaign(campaign, monkeypatch):
    """Reject a second start while leaving its current watcher authoritative."""
    path, plan, _ = campaign
    plan.update(status="active", worker="10")
    path.write_text(json.dumps(plan))
    (path.parent / ".gepa-source-commit").write_text(plan["source_commit"])
    monkeypatch.setattr(controller, "run", lambda _: pytest.fail("Duplicate submission"))
    with pytest.raises(ValueError, match="already been submitted"):
        controller.main(["start", "--plan", str(path)])
    assert json.loads(path.read_text()) == plan


def test_next_ablation_waits_for_successful_optimization_and_test(campaign, monkeypatch):
    """A timeout retries the same cell, while a successful exit releases the next."""
    path, plan, _ = campaign
    plan["cells"].append({**plan["cells"][0], "name": "second"})
    plan.update(status="active", worker="10")
    monkeypatch.setattr(controller, "accounting", lambda _: ("COMPLETED", "0:0"))
    dispatched = []
    monkeypatch.setattr(controller, "dispatch", lambda _, p: dispatched.append(p["cells"][p["index"]]["name"]))
    controller.advance(path, plan, "10")
    assert dispatched == ["second"] and plan["index"] == 1


def test_generic_worker_environment_keeps_original_budget_and_source(tmp_path):
    """Bind either benchmark's worker to recovery without granting more budget."""
    export = tmp_path / "worker.env"
    export.write_bytes(b"BUDGET=6871\0")
    controller.prepare_export(export, tmp_path / "registry", tmp_path / "error", "a" * 40)
    values = dict(entry.split(b"=", 1) for entry in export.read_bytes().split(b"\0") if entry)
    assert values[b"BUDGET"] == b"6871"
    assert values[b"GEPA_SOURCE_COMMIT"] == b"a" * 40
    with pytest.raises(ValueError, match="differ"):
        controller.prepare_export(export, tmp_path / "registry", tmp_path / "error", "b" * 40)
