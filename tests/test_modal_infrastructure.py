"""Check cloud isolation, SDK boundaries, and interrupted-delivery recovery offline."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.common.remote_runtime import REQUEST_HEADER, RUNTIME_HEADER, load_remote_runtime
from examples.modal import cli
from examples.modal.config import BENCHMARKS, MODELS, benchmark_arguments, roles, validate_plan
from examples.modal.gateway import DeliveryJournal, UncertainRequestError
from examples.modal.worker import data_key, preparation_commands, prepare_writable_caches


@pytest.fixture
def plan():
    return {
        "schema_version": 1,
        "benchmark": "hotpotqa",
        "run_id": "test-run-01",
        "mode": "optimize",
        "timeout": 43200,
        "region": None,
        "secrets": ["gepa-research"],
        "terminalbench_scope": "della88",
        "embedding": {"device": "cuda", "attention": "sdpa", "dtype": "bfloat16", "batch_size": 1},
        "arguments": ["--condition", "both", "--seed", "0", "--max-metric-calls", "6871"],
        "source_sha256": "a" * 64,
    }


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_all_benchmarks_keep_science_and_prepare_pinned_models(plan, benchmark):
    plan["benchmark"] = benchmark
    validate_plan(plan)
    endpoints = {role: {"api_base": f"https://{role}.example/v1"} for role in roles(plan)}
    argv = benchmark_arguments(plan, endpoints)
    assert argv[-len(plan["arguments"]) :] == plan["arguments"]
    assert argv[argv.index("--run-dir") + 1] == "/results/benchmark"
    assert "--execution-runtime" in argv
    commands = preparation_commands(plan)
    assert all(role["profile"] in str(commands) for role in MODELS.values())
    if benchmark == "terminalbench":
        assert cli.resources(plan)["runtime"] == "vm"
        assert "gpu" not in cli.resources(plan)
        assert argv[argv.index("--task-scope") + 1] == "della88"
    if benchmark == "obliqbench":
        for option in ("--embedding-device", "--embedding-attention", "--embedding-dtype", "--embedding-batch-size"):
            command = commands[0]
            assert argv[argv.index(option) + 1] == command[command.index(option) + 1]
        assert cli.resources(plan)["gpu"] == "H200"


@pytest.mark.parametrize(
    "option", ["--run-dir=/scratch/old", "--solver-api-base", "--runtime-rec", "--data-file", "--embedding-bat"]
)
def test_managed_options_cannot_be_overridden_even_by_argparse_abbreviations(plan, option):
    plan["arguments"] = [option]
    with pytest.raises(ValueError, match="Infrastructure owns"):
        validate_plan(plan)


def test_preparation_key_preserves_source_and_encoder_but_not_budget(plan):
    original = data_key(plan)
    plan["arguments"][-1] = "13742"
    plan["mode"] = "pilot"
    assert data_key(plan) == original
    assert roles(plan) == ["solver"]
    plan["embedding"]["dtype"] = "float32"
    assert data_key(plan) != original


def test_sealed_response_recovers_delivery_without_duplicate_inference(tmp_path):
    sync, send = Mock(), Mock(return_value=(200, b'{"choices":[{"message":{"content":"ok"}}]}'))
    journal = DeliveryJournal(tmp_path, sync)
    request_id = "12345678-1234-1234-1234-123456789abc"
    first = journal.complete(request_id, b'{"seed":0}', send)
    recovered = DeliveryJournal(tmp_path, sync).complete(request_id, b'{"seed":0}', send)
    assert first == (200, send.return_value[1], False)
    assert recovered == (200, send.return_value[1], True)
    assert send.call_count == 1
    # The fixed output-error retry policy deliberately advances the seed.
    assert journal.complete(request_id, b'{"seed":1}', send)[2] is False
    assert send.call_count == 2
    assert len(list(tmp_path.rglob("attempt-*.json"))) == 2


def test_unknown_start_is_never_rerolled(tmp_path):
    journal = DeliveryJournal(tmp_path, lambda: None)
    send = Mock(side_effect=TimeoutError("provider outcome unknown"))
    request_id = "12345678-1234-1234-1234-123456789abc"
    with pytest.raises(TimeoutError):
        journal.complete(request_id, b"{}", send)
    with pytest.raises(UncertainRequestError, match="unresolved"):
        DeliveryJournal(tmp_path, lambda: None).complete(request_id, b"{}", send)
    assert send.call_count == 1


def test_definitive_http_error_allows_existing_provider_retry(tmp_path):
    send = Mock(side_effect=[(500, b"{}"), (200, b'{"ok":true}')])
    journal = DeliveryJournal(tmp_path, lambda: None)
    request_id = "12345678-1234-1234-1234-123456789abc"
    assert journal.complete(request_id, b"{}", send)[0] == 500
    assert journal.complete(request_id, b"{}", send)[0] == 200
    assert send.call_count == 2


def test_corrupt_sealed_response_blocks_replay(tmp_path):
    journal = DeliveryJournal(tmp_path, lambda: None)
    request_id = "12345678-1234-1234-1234-123456789abc"
    journal.complete(request_id, b"{}", lambda _: (200, b"original"))
    next(tmp_path.rglob("response-*.bin")).write_bytes(b"changed")
    with pytest.raises(UncertainRequestError, match="checksum"):
        journal.complete(request_id, b"{}", Mock())


def test_creation_intent_survives_unknown_submission_and_cannot_repeat(tmp_path):
    modal = SimpleNamespace(Sandbox=SimpleNamespace(create=Mock(side_effect=TimeoutError)))
    receipt = {"allocations": {}}
    path = tmp_path / "receipt.json"
    with pytest.raises(TimeoutError):
        cli.create_once(modal, receipt, path, "solver", name="run-solver", args=["uv", "run"])
    saved = json.loads(path.read_text())
    assert saved["allocations"]["solver"]["state"] == "creation_started"
    assert "id" not in saved["allocations"]["solver"]
    with pytest.raises(ValueError, match="intent"):
        cli.create_once(modal, saved, path, "solver", name="run-solver", args=["uv", "run"])
    modal.Sandbox.create.assert_called_once_with("uv", "run", name="run-solver")


def test_remote_runtime_requires_fresh_endpoint_bound_evidence(monkeypatch):
    identity = {"model": "test", "sha256": "a" * 64}
    record = {
        "schema_version": 2,
        "provider": "modal",
        "api_base": "https://example.test/v1",
        "identity": identity,
        "allocation": "worker-1",
    }
    monkeypatch.setattr("examples.terminalbench.runtime.validate_identity", lambda value, model: value)
    request = Mock(return_value={k: v for k, v in record.items() if k != "api_base"})
    monkeypatch.setattr("examples.common.remote_runtime.request_json", request)
    assert load_remote_runtime(record, "test", record["api_base"]) == identity
    request.return_value["allocation"] = "worker-2"
    with pytest.raises(ValueError, match="changed"):
        load_remote_runtime(record, "test", record["api_base"])
    with pytest.raises(ValueError, match="endpoint"):
        load_remote_runtime(record, "test", "https://other.test/v1")


def test_model_transport_ids_are_opt_in_and_stable_across_retries(monkeypatch):
    from examples.common.provider_retries import complete_with_retries

    monkeypatch.setattr("examples.common.provider_retries.time.sleep", lambda _: None)
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    seen = []

    def send(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            raise ConnectionError("delivery failed")
        return response

    complete_with_retries(send, {"extra_headers": {RUNTIME_HEADER: "a" * 64}, "seed": 0}, {"role": "solver"})
    assert seen[0]["extra_headers"][REQUEST_HEADER] == seen[1]["extra_headers"][REQUEST_HEADER]
    assert seen[0]["seed"] == seen[1]["seed"] == 0
    plain = Mock(return_value=response)
    complete_with_retries(plain, {"seed": 0}, {"role": "solver"})
    assert "extra_headers" not in plain.call_args.kwargs


def test_download_checks_cloud_seals_and_frozen_source(tmp_path, monkeypatch):
    source = b"frozen source"
    payload = b"benchmark result"
    files = {"source.tar": source, "benchmark/result.json": payload}
    files["run-files.json"] = json.dumps(
        {"files": {key: hashlib.sha256(value).hexdigest() for key, value in files.items()}}
    ).encode()
    volume = SimpleNamespace(
        iterdir=lambda *a, **k: [SimpleNamespace(type=1, path=name) for name in files],
        read_file=lambda path: iter([files[path]]),
    )
    fake = SimpleNamespace(
        Volume=SimpleNamespace(from_name=lambda _: volume),
        volume=SimpleNamespace(FileEntryType=SimpleNamespace(FILE=1)),
    )
    monkeypatch.setattr(cli, "sdk", lambda: fake)
    monkeypatch.setattr(cli, "inspect_receipt", lambda _: {"allocations": {"driver": {"state": "terminal"}}})
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "results_volume": "run",
                "source": {"sha256": hashlib.sha256(source).hexdigest()},
                "phase": "run",
                "allocations": {"driver": {"id": "driver"}, "solver": {"id": "solver"}},
            }
        )
    )
    report = cli.download(receipt, tmp_path / "archive")
    assert report["verified_sealed_entries"] == 2
    assert report["missing_seals"] == ["solver/serving-files.json"]
    assert report["all_expected_seals_present"] is False
    files["benchmark/result.json"] = b"corrupted"
    with pytest.raises(ValueError, match="checksum"):
        cli.download(receipt, tmp_path / "bad-archive")


def test_cache_writes_do_not_change_shared_preparation(tmp_path, plan):
    data, results, cache = (tmp_path / name for name in ("data", "results", "cache"))
    (data / "wiki17").mkdir(parents=True)
    results.mkdir()
    (data / "wiki17" / "corpus.jsonl").write_text("immutable corpus")
    (data / "wiki17" / "retriever_cache_old").mkdir()
    (data / "huggingface").mkdir()
    (data / "huggingface" / "dataset").write_text("immutable dataset")
    prepare_writable_caches(plan, data, results, cache)
    assert (results / "wiki17" / "corpus.jsonl").is_symlink()
    assert not (results / "wiki17" / "retriever_cache_old").exists()
    (results / "wiki17" / "retriever_cache_new").mkdir()
    assert not (data / "wiki17" / "retriever_cache_new").exists()
    (cache / "dataset").write_text("local write")
    assert (data / "huggingface" / "dataset").read_text() == "immutable dataset"


def test_failed_launch_cleans_only_known_workers_without_a_driver(tmp_path, monkeypatch):
    sandbox = Mock()
    sandbox.poll.return_value = 137
    sdk = SimpleNamespace(Sandbox=SimpleNamespace(from_id=Mock(return_value=sandbox)))
    monkeypatch.setattr(cli, "sdk", lambda: sdk)
    receipt = {"allocations": {"solver": {"id": "known"}, "proposer": {"name": "unknown"}}}
    cli.cleanup_failed_launch(receipt, tmp_path / "receipt.json")
    sdk.Sandbox.from_id.assert_called_once_with("known")
    sandbox.terminate.assert_called_once()
    assert "cleanup" not in receipt["allocations"]["proposer"]
    sdk.Sandbox.from_id.reset_mock()
    receipt["allocations"]["driver"] = {"name": "driver-outcome-unknown"}
    cli.cleanup_failed_launch(receipt, tmp_path / "receipt.json")
    sdk.Sandbox.from_id.assert_not_called()


def test_runtime_startup_waits_only_for_matching_live_evidence(monkeypatch):
    sandbox = Mock()
    sandbox.poll.return_value = None
    proof = {"source_sha256": "source", "image_id": "image"}
    request = Mock(side_effect=[ConnectionError(), proof])
    monkeypatch.setattr(cli, "request_json", request)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    assert cli.wait_for_runtime(sandbox, "https://test/v1", "key", {"sha256": "source"}, "image") == proof
    request.side_effect = None
    request.return_value = {**proof, "image_id": "changed"}
    with pytest.raises(ValueError, match="frozen source and image"):
        cli.wait_for_runtime(sandbox, "https://test/v1", "key", {"sha256": "source"}, "image")
    sandbox.poll.return_value = 1
    with pytest.raises(RuntimeError, match="exited"):
        cli.wait_for_runtime(sandbox, "https://test/v1", "key", {"sha256": "source"}, "image")


def test_harbor_report_separates_cloud_deliveries_from_physical_inference(tmp_path):
    from examples.terminalbench.token_usage import summarize_usage

    record = {
        "schema_version": 2,
        "requested_model": "solver",
        "role": "task_agent",
        "error_type": None,
        "attempt_kind": "modal_delivery",
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "reasoning_tokens": None,
        "length_finish": False,
        "output_cap_reached": False,
        "context_cap_reached": False,
    }
    path = tmp_path / "token-usage.jsonl"
    path.write_text(json.dumps(record) + "\n")
    report = summarize_usage([path])
    assert report["models"] == {}
    assert report["modal_deliveries"]["solver"]["task_agent"]["calls"] == 1
    assert "gateway" in report["modal_physical_attempt_authority"]
