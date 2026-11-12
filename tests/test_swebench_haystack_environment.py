"""Keep offline Haystack preparation separate from frozen tests and historical scores."""

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from examples.common.artifacts import digest
from examples.common.benchmark_runner import ensure_contract
from examples.swebench_live import adapter
from examples.swebench_live.proposals.haystack_9742_v1 import deploy, install_dependencies
from examples.swebench_live.proposals.haystack_9742_v1 import environment as native_environment
from examples.swebench_live.proposals.haystack_9742_v1.install_dependencies import verified_wheels
from examples.swebench_live.proposals.haystack_9742_v1.prepare import prepare, proposed_record, validate_proposed_record


@pytest.fixture
def record_path():
    terminal = os.environ.get("SWE_HAYSTACK_TERMINAL_ROOT")
    if terminal is None:
        pytest.skip("Set SWE_HAYSTACK_TERMINAL_ROOT to the verified attempt15 archive")
    return Path(terminal) / "files/swebench_live/native-trials/329ae5d511874fa19542ed3b2052d82e/evaluator-record.json"


def test_preparation_preserves_task_reference_tests_and_score_membership(record_path, tmp_path):
    original_bytes = record_path.read_bytes()
    original = json.loads(original_bytes)
    output = tmp_path / "proposal"
    manifest = prepare(record_path, output)
    proposed = json.loads((output / "proposed-record.json").read_text())
    assert {key for key in original if original[key] != proposed[key]} == {"test_cmds"}
    assert len(proposed["FAIL_TO_PASS"]) == 2
    assert len(proposed["PASS_TO_PASS"]) == 3326
    assert (output / "proposed-test.patch").read_text() == original["test_patch"]
    assert (output / "proposed-reference.patch").read_text() == original["patch"]
    assert record_path.read_bytes() == original_bytes
    assert manifest["status"] == "PROPOSED_NOT_ADOPTED"
    assert not manifest["adoption_authorized"]
    assert manifest["native_validation"] == "NOT_RUN_BY_PREPARATION"
    assert manifest["evaluation_contract"]["evaluator_sha256"] == digest(proposed)
    assert (
        proposed["evaluation_environment_sha256"]
        == hashlib.sha256((output / "environment.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "field", ["base_commit", "patch", "test_patch", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "test_cmds"]
)
def test_changed_input_fails_before_writing(record_path, tmp_path, field):
    record = json.loads(record_path.read_text())
    record[field] = [] if isinstance(record[field], list) else "changed"
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity changed"):
        prepare(altered, tmp_path / "proposal")
    assert not (tmp_path / "proposal").exists()


def test_preparation_does_not_replace_existing_evidence(record_path, tmp_path):
    output = tmp_path / "proposal"
    prepare(record_path, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(record_path, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_old_runtime_contract_cannot_resume_with_offline_record(record_path, tmp_path):
    record = json.loads(record_path.read_text())
    manifest = prepare(record_path, tmp_path / "proposal")
    directory = tmp_path / "resume"
    ensure_contract(directory, "contract.json", {"evaluator_sha256": digest(record)})
    with pytest.raises(ValueError, match="use a new run directory"):
        ensure_contract(directory, "contract.json", manifest["evaluation_contract"])


def test_generated_command_sets_offline_flags_before_starting_unchanged_hatch_arguments(record_path, tmp_path):
    output = tmp_path / "proposal"
    prepare(record_path, output)
    command = json.loads((output / "proposed-record.json").read_text())["test_cmds"][0]
    launcher = tmp_path / "hatch"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "print(json.dumps({'args':sys.argv[1:],'hub':os.environ.get('HF_HUB_OFFLINE'),'transformers':os.environ.get('TRANSFORMERS_OFFLINE')}))\n"
    )
    launcher.chmod(0o755)
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    environment.pop("HF_HUB_OFFLINE", None)
    environment.pop("TRANSFORMERS_OFFLINE", None)
    completed = subprocess.run(shlex.split(command), env=environment, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout) == {"args": ["run", "test:unit", "-rA"], "hub": "1", "transformers": "1"}


@pytest.mark.parametrize("change", ["missing", "changed", "symlink"])
def test_wheel_integrity_rejects_missing_changed_and_linked_files(tmp_path, change):
    wheel = tmp_path / "package.whl"
    data = b"exact package bytes"
    item = {"filename": wheel.name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    if change == "changed":
        wheel.write_bytes(data + b"changed")
    elif change == "symlink":
        target = tmp_path / "target"
        target.write_bytes(data)
        wheel.symlink_to(target)
    with pytest.raises(ValueError):
        verified_wheels({"wheels": [item]}, tmp_path)


def test_installer_rejects_inventory_drift_before_pip(tmp_path, monkeypatch):
    environment = {
        "native_python": sys.executable,
        "site_packages": str(tmp_path),
        "existing_packages": {"kept": "1.0"},
    }
    (tmp_path / "environment.json").write_text(json.dumps(environment))
    monkeypatch.setattr(install_dependencies, "__file__", str(tmp_path / "install_dependencies.py"))
    monkeypatch.setattr(install_dependencies, "package_versions", lambda path: {"kept": "2.0"})
    with pytest.raises(ValueError, match="inventory differs"):
        install_dependencies.install(tmp_path)


def test_installer_detects_existing_package_replacement_after_install(tmp_path, monkeypatch):
    environment = {
        "native_python": sys.executable,
        "site_packages": str(tmp_path),
        "existing_packages": {"kept": "1.0"},
        "added_packages": {"added": "1.0"},
        "wheels": [],
    }
    (tmp_path / "environment.json").write_text(json.dumps(environment))
    monkeypatch.setattr(install_dependencies, "__file__", str(tmp_path / "install_dependencies.py"))
    inventories = iter([{"kept": "1.0"}, {"kept": "2.0", "added": "1.0"}])
    monkeypatch.setattr(install_dependencies, "package_versions", lambda path: next(inventories))
    calls = []
    monkeypatch.setattr(install_dependencies.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    with pytest.raises(ValueError, match="Installed inventory differs"):
        install_dependencies.install(tmp_path)
    assert len(calls) == 1
    assert "--no-index" in calls[0] and "--no-deps" in calls[0]


@pytest.mark.parametrize("field", ["test_cmds", "evaluation_environment_sha256", "test_patch", "FAIL_TO_PASS"])
def test_proposed_identity_rejects_changed_execution_or_tests(record_path, field):
    record = proposed_record(json.loads(record_path.read_text()))
    record[field] = [] if isinstance(record[field], list) else "changed"
    with pytest.raises(ValueError, match="identity changed"):
        validate_proposed_record(record)


@pytest.mark.parametrize("opt_in", [False, True])
def test_actual_adapter_dispatches_environment_only_for_opt_in_record(record_path, tmp_path, monkeypatch, opt_in):
    record = json.loads(record_path.read_text())
    if opt_in:
        record = proposed_record(record)
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "evaluator-record.json").write_text(json.dumps(record))
    (trial / "prediction.json").write_text(json.dumps({"model_patch": "sealed candidate"}))
    image = {"backend": "apptainer", "image_id": "pinned-image", "immutable": "pinned-reference"}
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[-3:] == ["-m", "examples.swebench_live.apptainer_grader", str(trial / "grader-request.json")]
        (trial / "official-result.json").write_text(json.dumps({record["instance_id"]: {"resolved": False}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    adapter.grade(tmp_path, trial, {"id": record["instance_id"]}, image, 16, 1800)
    request = json.loads((trial / "grader-request.json").read_text())
    assert ("evaluation_environment" in request) is opt_in
    if opt_in:
        assert request["evaluation_environment"] == str(
            tmp_path / "evaluation-environments" / record["evaluation_environment_sha256"]
        )
    assert len(calls) == 1


def test_changed_capsule_fails_before_native_install(record_path, tmp_path):
    record = proposed_record(json.loads(record_path.read_text()))
    (tmp_path / "environment.json").write_text("{}")
    with pytest.raises(ValueError, match="capsule changed"):
        native_environment.install_in_grader(None, record, tmp_path)


def test_deployment_copies_only_verified_wheels_and_preserves_existing_capsule(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    packet.mkdir()
    wheelhouse = tmp_path / "downloads"
    wheelhouse.mkdir()
    data = b"pinned wheel"
    (wheelhouse / "package.whl").write_bytes(data)
    (wheelhouse / "reference.patch").write_text("must not enter the capsule")
    environment = {
        "wheels": [{"filename": "package.whl", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]
    }
    (packet / "environment.json").write_text(json.dumps(environment))
    monkeypatch.setattr(deploy, "PACKET", packet)
    output = deploy.stage_environment(wheelhouse, tmp_path / "isolated-assets")
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == {
        "environment.json",
        "wheels/package.whl",
    }
    assert (output / "wheels/package.whl").read_bytes() == data
    with pytest.raises(FileExistsError):
        deploy.stage_environment(wheelhouse, tmp_path / "isolated-assets")
    assert (output / "wheels/package.whl").read_bytes() == data


@pytest.mark.parametrize("opt_in", [False, True])
def test_actual_grader_installs_before_patch_and_preserves_default_constructor(
    record_path, tmp_path, monkeypatch, opt_in
):
    record = json.loads(record_path.read_text())
    if opt_in:
        record = proposed_record(record)
    for name, values in {
        "swebench.harness.constants": {
            "APPLY_PATCH_FAIL": "apply failed",
            "LOG_TEST_OUTPUT": "test_output.txt",
            "TESTS_TIMEOUT": "timeout",
        },
        "swebench.harness.grading": {"get_eval_report": lambda **kwargs: {record["instance_id"]: {"resolved": False}}},
        "swebench.harness.run_evaluation": {"GIT_APPLY_CMDS": ["git apply"]},
        "swebench.harness.test_spec.test_spec": {
            "make_test_spec": lambda *args, **kwargs: SimpleNamespace(eval_script="unchanged official script")
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(adapter.__file__).with_name("apptainer_grader.py")
    spec = importlib.util.spec_from_file_location("haystack_grader_boundary_test", path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    events = []

    class Task:
        def __init__(self, image, memory_gib, **kwargs):
            assert kwargs == {}

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def write_text(self, path, text):
            events.append(path)

        def execute(self, command, **kwargs):
            return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(grader, "ApptainerTask", Task)
    monkeypatch.setattr(native_environment, "install_in_grader", lambda *args: events.append("install"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "record.json").write_text(json.dumps(record))
    (tmp_path / "prediction.json").write_text(
        json.dumps({"instance_id": record["instance_id"], "model_patch": "sealed patch"})
    )
    grader.grade(
        {
            "evaluator_record": "record.json",
            "prediction": "prediction.json",
            "image": {},
            "memory_gib": 16,
            "timeout_seconds": 1800,
            "evaluation_environment": str(tmp_path),
        }
    )
    expected = (
        ["enter", "install", "/tmp/patch.diff", "/eval.sh", "exit"]
        if opt_in
        else ["enter", "/tmp/patch.diff", "/eval.sh", "exit"]
    )
    assert events == expected
