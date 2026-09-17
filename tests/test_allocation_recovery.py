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

    def run(command, **kwargs):
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
    monkeypatch.setattr(controller, "run", lambda command, **_: calls.append(command) or "unparseable")
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


@pytest.fixture
def extension(campaign, monkeypatch):
    path, plan, _ = campaign
    source = path.parent / "new-source"
    source.mkdir()
    (source / ".gepa-source-commit").write_text("b" * 40)
    monkeypatch.setattr(controller, "__file__", str(source / "examples/common/slurm_continuation.py"))
    export = source / "random.env"
    export.write_bytes(b"BUDGET=6871\0")
    registry = source / "registry.json"
    error = source / "error.json"
    controller.prepare_export(export, registry, error, "b" * 40)
    plan["cells"].append({**plan["cells"][0], "name": "action"})
    plan.update(status="active", worker="10", controller="11", before={})
    added = {
        "status": "draft",
        "source_commit": "b" * 40,
        "source_dir": str(source),
        "cells": [
            {
                "name": "random",
                "command": ["sbatch", "--parsable", "worker.sh"],
                "export_file": str(export),
                "export_sha256": file_digest(export),
                "registry": str(registry),
                "error_file": str(error),
            }
        ],
    }
    return path, plan, added


def test_extension_preserves_workers_exports_and_checkpoints(extension, monkeypatch):
    path, plan, added = extension
    original = json.loads(json.dumps(plan))
    commands = []
    monkeypatch.setattr(controller, "accounting", lambda job: ("RUNNING" if job == "10" else "PENDING", "0:0"))
    monkeypatch.setattr(controller, "run", lambda cmd: commands.append(cmd) or ("12" if cmd[0] == "sbatch" else ""))
    controller.extend(path, plan, added, "action", "10", "11")
    assert plan["cells"][:2] == original["cells"]
    assert plan["source_commit"] == original["source_commit"]
    assert plan["index"] == 0 and plan["worker"] == "10" and plan["before"] == {}
    assert plan["controller"] == "12"
    assert commands[0] == ["scontrol", "hold", "11"]
    assert "--hold" in commands[1] and "--dependency=afterany:10" in commands[1]
    assert commands[2:] == [["scancel", "11"], ["scontrol", "release", "12"]]
    assert json.loads(path.with_name("plan-before-random.json").read_text()) == original
    with pytest.raises(ValueError, match="active campaign changed"):
        controller.extend(path, plan, added, "action", "10", "11")


@pytest.mark.parametrize("problem", ["current_cell", "worker_changed", "watcher_running", "export_changed"])
def test_extension_rejects_drift_before_scheduler_mutation(extension, monkeypatch, problem):
    path, plan, added = extension
    after = "action"
    if problem == "current_cell":
        after = "first"
    elif problem == "worker_changed":
        plan["worker"] = "99"
    elif problem == "export_changed":
        Path(added["cells"][0]["export_file"]).write_bytes(b"changed")
    monkeypatch.setattr(
        controller,
        "accounting",
        lambda job: ("RUNNING" if problem == "watcher_running" or job == "10" else "PENDING", "0:0"),
    )
    monkeypatch.setattr(controller, "run", lambda _: pytest.fail("Scheduler mutation"))
    with pytest.raises(ValueError):
        controller.extend(path, plan, added, after, "10", "11")


def test_extended_cell_timeout_verifies_its_own_source(extension, monkeypatch):
    path, plan, added = extension
    cell = added["cells"][0]
    cell.update(source_commit=added["source_commit"], source_dir=added["source_dir"])
    plan["cells"].append(cell)
    plan["index"] = 2
    checkpoint = Path(added["source_dir"]) / "result"
    checkpoint.write_bytes(b"new random-action state")
    monkeypatch.setenv("HOTPOTQA_SOURCE_COMMIT", added["source_commit"])
    monkeypatch.setenv("GEPA_RECOVERY_REGISTRY", cell["registry"])
    seal_progress(checkpoint.parent, 300, [checkpoint])
    monkeypatch.setattr(controller, "accounting", lambda _: ("TIMEOUT", "0:15"))
    dispatched = []
    monkeypatch.setattr(controller, "dispatch", lambda _, p: dispatched.append(p["index"]))
    controller.advance(path, plan, "10")
    assert dispatched == [2]
    cell["source_commit"] = plan["source_commit"]
    with pytest.raises(ValueError, match="source"):
        controller.advance(path, plan, "10")


def test_new_controller_submits_old_and_new_workers_from_their_pinned_directories(extension, monkeypatch):
    """A newer CPU watcher must not move an old worker to its own checkout."""
    path, plan, added = extension
    plan.update(controller_source_dir=added["source_dir"], controller_source_commit=added["source_commit"])
    calls = []
    identifiers = iter(["20", "21", "22", "23"])

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return next(identifiers) if command[0] == "sbatch" else ""

    monkeypatch.setattr(controller, "run", run)
    controller.dispatch(path, plan)
    assert calls[0][1] == {"cwd": plan["source_dir"]}
    assert f"--chdir={added['source_dir']}" in calls[1][0]
    cell = {**added["cells"][0], "source_dir": added["source_dir"], "source_commit": added["source_commit"]}
    plan["cells"].append(cell)
    plan["index"] = 2
    controller.dispatch(path, plan)
    assert calls[3][1] == {"cwd": added["source_dir"]}
    assert f"--chdir={added['source_dir']}" in calls[4][0]
