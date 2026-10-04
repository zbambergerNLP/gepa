"""Check live-server binding and runtime drift with offline process/GPU boundaries."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from terminalbench_runtime_helpers import runtime_fixture

from examples.common import model_snapshot
from examples.common.experiment_models import EXPERIMENT_MODELS
from examples.terminalbench import runtime


def write_record(path, identity, server, port=8000):
    """Write a test-only launcher record without bypassing production validation."""
    path.write_text(json.dumps({"schema_version": 1, "identity": identity, "server": server, "port": port}))
    return path


@pytest.fixture
def server(monkeypatch):
    """Replace Linux process discovery while retaining real record validation."""
    binding = {"hostname": "node-a", "boot_id": "boot-a", "pid": 1234, "process_start": "10"}
    monkeypatch.setattr(runtime, "process_identity", lambda pid: deepcopy(binding))
    monkeypatch.setattr(runtime, "server_listens", lambda pid, port: True)
    return binding


def test_linux_process_identity_handles_names_and_rejects_zombies(monkeypatch):
    """Use start time rather than PID alone and exclude dead processes."""
    fields = ["S", *(["0"] * 18), "123456"]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: "boot-id" if str(path).endswith("boot_id") else f"123 (vllm (server)) {' '.join(fields)}",
    )
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "node")
    assert runtime.process_identity(123) == {
        "hostname": "node",
        "boot_id": "boot-id",
        "pid": 123,
        "process_start": "123456",
    }
    fields[0] = "Z"
    with pytest.raises(ValueError, match="exited"):
        runtime.process_identity(123)


@pytest.mark.parametrize("owner", ["123", "456", "other"])
def test_api_socket_must_belong_to_server_process_tree(monkeypatch, owner):
    """Recognize an API worker's inherited socket and reject another server on the port."""

    def read(path):
        if str(path).endswith("/net/tcp"):
            return "header\n0: 0100007F:1F40 00000000:0000 0A 0 0 0 0 0 999\n"
        return "456" if str(path) == "/proc/123/task/123/children" else ""

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "iterdir", lambda path: iter([path / "7"]))
    monkeypatch.setattr(
        Path, "readlink", lambda path: Path("socket:[999]" if str(path) == f"/proc/{owner}/fd/7" else "pipe:[1]")
    )
    assert runtime.server_listens(123, 8000) is (owner != "other")
    assert runtime.server_listens(123, 9000) is False


def test_live_process_without_its_api_socket_cannot_qualify(tmp_path, monkeypatch, server):
    model = EXPERIMENT_MODELS[0]
    path = write_record(tmp_path / "runtime.json", runtime_fixture(model), server)
    monkeypatch.setattr(runtime, "server_listens", lambda pid, port: False)
    with pytest.raises(ValueError, match="does not own"):
        runtime.load_runtime_record(path, model, "http://localhost:8000/v1")


def test_separate_optimizer_server_requires_its_own_current_record(tmp_path, server):
    """Validate both live role endpoints instead of inferring optimizer hardware from the task server."""
    model = EXPERIMENT_MODELS[0]
    task = write_record(tmp_path / "task.json", runtime_fixture(model), server)
    proposer_identity = runtime_fixture(model)
    proposer_identity["parallelism"]["tensor"] = 4
    proposer_identity["sha256"] = runtime._digest(
        {key: value for key, value in proposer_identity.items() if key != "sha256"}
    )
    proposer = write_record(tmp_path / "proposer.json", proposer_identity, server, 9000)
    args = argparse.Namespace(
        runtime_record=task,
        proposer_runtime_record=proposer,
        student_model=model,
        proposer_model=model,
        student_api_base="http://localhost:8000/v1",
        proposer_api_base="http://localhost:9000/v1",
    )
    loaded = runtime.load_role_runtimes(args)
    assert loaded["student"] != loaded["proposer"]
    args.proposer_runtime_record = None
    with pytest.raises(ValueError, match="endpoint"):
        runtime.load_role_runtimes(args)


def test_seed_only_evaluation_does_not_require_an_unused_proposer(tmp_path, server):
    """Require only the live solver for pilots and unoptimized baseline evaluation."""
    model = EXPERIMENT_MODELS[0]
    identity = runtime_fixture(model)
    task = write_record(tmp_path / "task.json", identity, server)
    args = argparse.Namespace(
        runtime_record=task,
        student_model=model,
        student_api_base="http://localhost:8000/v1",
        proposer_runtime_record=tmp_path / "not-running.json",
        proposer_model=EXPERIMENT_MODELS[1],
        proposer_api_base=None,
    )
    assert runtime.load_role_runtimes(args, include_proposer=False) == {"student": identity}
    with pytest.raises(FileNotFoundError):
        runtime.load_role_runtimes(args)


@pytest.mark.parametrize(
    "flag", ["--api-key=secret", "--api_key=secret", "--api-k=secret", "--model=other", "--chat-template=untracked"]
)
def test_launcher_rejects_untracked_overrides_and_secret_flags(tmp_path, monkeypatch, flag):
    execute = Mock()
    monkeypatch.setattr(runtime.os, "execv", execute)
    path = tmp_path / "runtime.json"
    with pytest.raises(SystemExit):
        runtime.main(
            ["--model", EXPERIMENT_MODELS[0], "--model-path", str(tmp_path), "--runtime-record", str(path), "--", flag]
        )
    assert not path.exists()
    execute.assert_not_called()


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
def test_fresh_servers_on_equivalent_nodes_share_identity(tmp_path, server, model):
    """Allow fresh process/node/port/path metadata with identical material configuration."""
    identity = runtime_fixture(model)
    first = write_record(tmp_path / "first.json", identity, server)
    loaded = runtime.load_runtime_record(first, model, "http://localhost:8000/v1")
    server.update(hostname="node-b", boot_id="boot-b", pid=4321, process_start="20")
    second = write_record(tmp_path / "second.json", identity, server, 9000)
    assert runtime.load_runtime_record(second, model, "http://127.0.0.1:9000/v1") == loaded
    with pytest.raises(ValueError, match="current live server"):
        runtime.load_runtime_record(first, model, "http://localhost:8000/v1")


@pytest.mark.parametrize(
    "field,value", [("pid", 4321), ("process_start", "20"), ("boot_id", "new"), ("hostname", "other")]
)
def test_stale_server_record_is_rejected(tmp_path, server, field, value):
    """Reject stale files even when the expected checkpoint/settings have not changed."""
    model = EXPERIMENT_MODELS[0]
    path = write_record(tmp_path / "runtime.json", runtime_fixture(model), server)
    server[field] = value
    with pytest.raises(ValueError, match="current live server"):
        runtime.load_runtime_record(path, model, "http://localhost:8000/v1")


@pytest.mark.parametrize(
    "endpoint", [None, "http://localhost:9000/v1", "http://remote:8000/v1", "http://localhost:8000/other"]
)
def test_runtime_cannot_be_assigned_to_another_endpoint(tmp_path, server, endpoint):
    model = EXPERIMENT_MODELS[0]
    path = write_record(tmp_path / "runtime.json", runtime_fixture(model), server)
    with pytest.raises(ValueError, match="endpoint"):
        runtime.load_runtime_record(path, model, endpoint)


@pytest.mark.parametrize("damage", ["missing", "revision", "fingerprint", "precision", "software", "gpus"])
def test_incomplete_or_changed_runtime_is_rejected(tmp_path, server, damage):
    model = EXPERIMENT_MODELS[0]
    identity = runtime_fixture(model)
    if damage == "missing":
        identity.pop("parallelism")
    elif damage == "revision":
        identity["model_revision"] = "0" * 40
    elif damage == "fingerprint":
        identity["gpus"][0]["name"] = "Other GPU"
    else:
        identity[damage] = None
        identity["sha256"] = runtime._digest({key: value for key, value in identity.items() if key != "sha256"})
    path = write_record(tmp_path / "runtime.json", identity, server)
    with pytest.raises(ValueError):
        runtime.load_runtime_record(path, model, "http://localhost:8000/v1")


def test_collector_uses_verified_checkpoint_and_serving_environment(tmp_path, monkeypatch):
    """Inspect actual launch inputs and exclude physical GPU IDs and credentials."""
    model = EXPERIMENT_MODELS[0]
    verify = Mock(return_value={"verified": "checkpoint bytes"})
    monkeypatch.setattr(model_snapshot, "verify_model_snapshot", verify)
    cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_name=lambda index: "NVIDIA H200",
        get_device_capability=lambda index: (9, 0),
        get_device_properties=lambda index: SimpleNamespace(total_memory=141_000_000_000),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="13.0"), __version__="2.10.0")
    )
    monkeypatch.setattr(runtime, "version", lambda name: {"vllm": "0.25.0", "transformers": "5.0.0"}[name])
    monkeypatch.setattr(
        runtime, "distributions", lambda: [SimpleNamespace(metadata={"Name": "vllm"}, version="0.25.0")]
    )
    monkeypatch.setattr(runtime.subprocess, "run", Mock(return_value=SimpleNamespace(stdout="580\n580\n")))
    monkeypatch.setenv("VLLM_API_KEY", "test-secret")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,2")
    options = argparse.Namespace(dtype="bfloat16", kv_cache_dtype="auto", tensor_parallel_size=1, data_parallel_size=2)
    identity = runtime.collect_identity(model, tmp_path, ["--dtype", "bfloat16"], options)
    verify.assert_called_once_with(tmp_path, "qwen3.8-27b")
    assert identity["model_integrity_sha256"] == runtime._digest(verify.return_value)
    assert identity["parallelism"] == {"tensor": 1, "data": 2}
    assert len(identity["gpus"]) == 2
    assert identity["environment"]["VLLM_BATCH_INVARIANT"] == "0"
    assert "test-secret" not in json.dumps(identity)
    assert "CUDA_VISIBLE_DEVICES" not in identity["environment"]
    runtime.validate_identity(identity, model)


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
def test_launcher_records_and_executes_the_same_arguments(tmp_path, monkeypatch, server, model):
    """Keep the captured process alive through exec and forward all material serving flags."""
    executable = tmp_path / "vllm"
    executable.touch()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    collect = Mock(return_value=runtime_fixture(model))
    execute = Mock()
    monkeypatch.setattr(runtime, "collect_identity", collect)
    monkeypatch.setattr(runtime.os, "execv", execute)
    path = tmp_path / "runtime.json"
    arguments = [
        "--dtype",
        "auto",
        "--kv-cache-dtype",
        "fp8",
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "1",
        "--no-enable-prefix-caching",
    ]
    runtime.main(
        ["--model", model, "--model-path", str(tmp_path / "weights"), "--runtime-record", str(path), "--", *arguments]
    )
    record = json.loads(path.read_text())
    assert record["server"] == server and record["identity"] == collect.return_value
    assert collect.call_args.args[2] == arguments
    command = execute.call_args.args[1]
    assert command[command.index("--port") + 2 :] == arguments
    assert command[:4] == [sys.executable, str(executable), "serve", str(tmp_path / "weights")]
    assert command[command.index("--served-model-name") + 1] == model.removeprefix("hosted_vllm/")
