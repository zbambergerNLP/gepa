"""Verify exact Tau evidence references with synthetic data and no provider calls."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_benchmark_runner import definition as local_definition
from test_jev_controller import select
from test_jev_input_size import mock_response
from test_jev_input_size import setup_controller as _setup_controller
from test_three_role import PROMPT, strategy

from examples.common import benchmark_runner as runner
from examples.common.benchmark_runner import ensure_contract
from examples.common.benchmark_types import BenchmarkModels
from examples.common.react_v2 import build_react_v2_strategy
from examples.taubench import main as tau_main
from gepa.proposer.reflective_mutation.three_role import ensure_reflection_run_contract
from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.jev_controller import JevController, JevControllerError
from gepa.strategies.jev_evidence import (
    TAU_KB_JEV_POLICY,
    TAU_KB_REFERENCE_FORMAT,
    TAU_KB_REFERENCE_GUIDANCE,
    EvidenceReferenceError,
    decode_tau_record,
    encode_tau_record,
    factor_tau_evidence,
)


@pytest.fixture
def setup_controller(tmp_path):
    yield from _setup_controller.__wrapped__(tmp_path)


def kb_return(documents):
    blocks = [
        f"{rank}. Title {doc_id}\n   ID: {doc_id}\n   Score: 0.5000\n   Content: {content}\n"
        for rank, (doc_id, content) in enumerate(documents, 1)
    ]
    return "\n".join(blocks) + "\n\n[Timing: retrieval=7ms, total=9ms]"


def record(documents=None, repeats=2, task_id="task_fixture"):
    documents = documents or [("doc_a", "policy text " * 100)]
    messages = [{"role": "user", "content": "Keep my exact request."}]
    for index in range(repeats):
        messages.extend(
            [
                {"role": "assistant", "tool_calls": [{"name": "KB_search", "arguments": {"query": f"q{index}"}}]},
                {"role": "tool", "content": kb_return(documents), "tool_call_id": f"call{index}"},
            ]
        )
    return {
        "Inputs": {"domain": "banking_knowledge", "task_id": task_id},
        "Generated Outputs": messages,
        "Feedback": {"official_reward": 0, "termination_reason": "user_stop"},
    }


def referenced_content(encoded):
    return encoded["record"]["Generated Outputs"][2]["content"]


def test_exact_deterministic_reconstruction_keeps_ids_versions_order_and_escaping():
    body = 'עברית 🌳 \\ "quotes"\nline\n' * 90
    documents = [("doc_a", body), ("doc_a", body + "changed"), ("doc_b", body)]
    original = record(documents, repeats=3)
    preserved = deepcopy(original)
    encoded = encode_tau_record(original)
    assert original == preserved
    assert encoded == encode_tau_record(original)
    assert encoded["documents"] == [{"document_id": key, "content": value} for key, value in documents]
    assert decode_tau_record(json.loads(json.dumps(encoded))) == original
    assert encoded["record"]["Inputs"] == original["Inputs"]
    assert encoded["record"]["Feedback"] == original["Feedback"]
    for before, after in zip(original["Generated Outputs"], encoded["record"]["Generated Outputs"], strict=True):
        assert {k: v for k, v in before.items() if k != "content"} == {k: v for k, v in after.items() if k != "content"}
    assert len(json.dumps(encoded).encode()) < len(json.dumps(original).encode())


def test_only_repeated_id_and_body_pairs_are_factored():
    original = record(repeats=1)
    original["Generated Outputs"].append({"role": "tool", "content": kb_return([("doc_other", "policy text " * 100)])})
    original["Generated Outputs"].append({"role": "tool", "content": kb_return([("doc_a", "changed " * 100)])})
    assert encode_tau_record(original) == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(format="unknown"),
        lambda value: value["documents"].append(deepcopy(value["documents"][0])),
        lambda value: value["documents"].append({"document_id": "doc_unused", "content": "unused"}),
        lambda value: value["documents"][0].update(document_id="doc_wrong"),
        lambda value: referenced_content(value)["parts"][1].update(document=-1),
        lambda value: referenced_content(value)["parts"][1].update(document=100),
        lambda value: referenced_content(value)["parts"][1].update(document=True),
        lambda value: referenced_content(value)["parts"][1].update(extra="unexpected"),
        lambda value: referenced_content(value)["parts"].append({"invalid": 0}),
        lambda value: referenced_content(value)["parts"].__setitem__(1, "removed reference"),
        lambda value: value["record"]["Generated Outputs"][2].update(role="assistant"),
    ],
)
def test_decoder_rejects_malformed_references(mutate):
    encoded = encode_tau_record(record())
    mutate(encoded)
    with pytest.raises(EvidenceReferenceError):
        decode_tau_record(encoded)


@pytest.mark.parametrize(
    "alter",
    [
        lambda text: text.replace("1. Title", "2. Title"),
        lambda text: text.replace("[Timing:", "[Unknown:"),
        lambda text: "preface\n" + text,
    ],
)
def test_unknown_tool_formats_remain_verbatim(alter):
    original = record()
    for message in original["Generated Outputs"]:
        if message.get("role") == "tool":
            message["content"] = alter(message["content"])
    traces = json.dumps([original], indent=2)
    assert factor_tau_evidence(traces) == (traces, False)


@pytest.mark.parametrize(
    "traces",
    [
        '{"id": 1}',
        "[]",
        '[{"Inputs": {}, "Inputs": {"domain": "banking_knowledge"}, "Generated Outputs": []}]',
        '[{"Inputs": {"domain": "swebench"}, "Generated Outputs": []}]',
    ],
)
def test_explicit_tau_input_rejects_invalid_or_other_benchmark_records(traces):
    with pytest.raises(EvidenceReferenceError):
        factor_tau_evidence(traces)


def test_default_controller_keeps_v6_and_original_tau_bytes(setup_controller):
    controller, requests, _ = setup_controller
    traces = json.dumps([record()], indent=2)
    _, metadata = select(controller, traces=traces)
    assert requests[0]["state"]["training_evidence"] == traces
    assert "training_evidence_representation" not in requests[0]["state"]
    assert "evidence_representation" not in controller.run_contract()
    assert metadata["policy"] == "jev_joint_action_section_v6_size_bounded_evidence_pool"


def opt_in(controller):
    configured = JevController(provider="openrouter", evidence_representation=TAU_KB_REFERENCE_FORMAT)
    configured._client = controller._client
    return configured


def test_complete_request_counts_guidance_and_preserves_component_criteria(setup_controller, monkeypatch):
    original_controller, requests, _ = setup_controller
    traces = json.dumps([record()])
    select(original_controller, traces=traces)
    original_request = deepcopy(requests[0])
    controller = opt_in(original_controller)
    _, metadata = select(controller, traces=traces)
    request = requests[-1]
    assert request["questions"] == original_request["questions"]
    assert {
        k: v for k, v in request["state"].items() if k not in {"training_evidence", "training_evidence_representation"}
    } == {k: v for k, v in original_request["state"].items() if k != "training_evidence"}
    assert request["state"]["training_evidence_representation"]["guidance"] == TAU_KB_REFERENCE_GUIDANCE
    assert decode_tau_record(json.loads(request["state"]["training_evidence"])[0]) == json.loads(traces)[0]
    assert metadata["policy"] == TAU_KB_JEV_POLICY
    exact_size = controller._request_utf8_size(request)
    monkeypatch.setattr(controller, "_request_size_limit", lambda: exact_size)
    select(controller, traces=traces)
    monkeypatch.setattr(controller, "_request_size_limit", lambda: exact_size - 1)
    before = len(requests)
    with pytest.raises(JevControllerError, match="client budget"):
        select(controller, traces=traces)
    assert len(requests) == before


def test_original_records_keep_group_weights_and_one_draw(setup_controller):
    original, requests, replies = setup_controller
    controller = opt_in(original)
    records = [record([("doc_a", "x" * 180_000)], task_id=f"task_{index}") for index in range(3)]
    keys = []

    def weighted(request):
        result = mock_response(request)
        body = json.loads(result.content)
        answer = body["answers"]["edit"]
        keys[:] = list(answer["probabilities"])[:2]
        count = len(json.loads(request["state"]["training_evidence"]))
        p = 0.9 if count == 2 else 0.3
        answer["probabilities"] = {
            key: p if key == keys[0] else 1 - p if key == keys[1] else 0.0 for key in answer["probabilities"]
        }
        answer["choice"] = keys[0] if p > 0.5 else keys[1]
        return type(result)(200, json=body)

    replies.extend([weighted, weighted])
    rng = random.Random(17)
    expected = random.Random(17)
    expected.random()
    _, metadata = select(controller, rng=rng, traces=json.dumps(records))
    assert rng.getstate() == expected.getstate()
    assert [group["count"] for group in metadata["evidence_groups"]] == [2, 1]
    assert metadata["probs"][keys[0]] == pytest.approx(0.7)
    decoded = [
        decode_tau_record(row) for request in requests for row in json.loads(request["state"]["training_evidence"])
    ]
    assert decoded == records
    assert all(controller._request_utf8_size(request) <= 524288 for request in requests)


def test_unique_oversized_content_still_fails_before_calls_or_rng_draw(setup_controller):
    original, requests, _ = setup_controller
    controller = opt_in(original)
    rng = random.Random(2)
    before = rng.getstate()
    traces = json.dumps([record(), record([("doc_big", "x" * 600_000)], repeats=1)])
    with pytest.raises(JevControllerError, match="evidence record 1"):
        select(controller, rng=rng, traces=traces)
    assert requests == [] and rng.getstate() == before


def test_v6_checkpoint_and_journal_reject_v7_even_without_repeated_docs(setup_controller, tmp_path):
    original, requests, _ = setup_controller
    current = JevController(
        provider="openrouter",
        evidence_representation=TAU_KB_REFERENCE_FORMAT,
        response_journal_path=tmp_path / "responses.sqlite3",
    )
    current._client = original._client
    traces = json.dumps([record(repeats=1)])
    with response_journal_scope("prior-v6"):
        select(original, traces=traces)
    with response_journal_scope("prior-v6"), pytest.raises(ResponseJournalError, match="mismatch"):
        select(current, traces=traces)
    assert len(requests) == 1
    old_strategy, _ = strategy(2, controller_selection="jev", jev_controller=original)
    new_strategy, _ = strategy(2, controller_selection="jev", jev_controller=current)
    ensure_reflection_run_contract(tmp_path / "run", old_strategy.run_contract({"sys": PROMPT}))
    with pytest.raises(ValueError):
        ensure_reflection_run_contract(tmp_path / "run", new_strategy.run_contract({"sys": PROMPT}))


def test_encoded_groups_resume_without_reissuing_completed_work(setup_controller, tmp_path, monkeypatch):
    original, requests, _ = setup_controller
    records = [record([("doc_a", "x" * 180_000)], task_id=f"task_{index}") for index in range(3)]
    traces = json.dumps(records)

    def configured():
        controller = JevController(
            provider="openrouter",
            evidence_representation=TAU_KB_REFERENCE_FORMAT,
            response_journal_path=tmp_path / "encoded.sqlite3",
        )
        controller._client = original._client
        return controller

    controller = configured()
    store = controller._journal.store

    def interrupt(*args):
        store(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller._journal, "store", interrupt)
    with response_journal_scope("encoded-resume"), pytest.raises(KeyboardInterrupt):
        select(controller, traces=traces)
    assert len(requests) == 1
    with response_journal_scope("encoded-resume"):
        _, metadata = select(configured(), traces=traces)
    assert len(requests) == 2 and metadata["physical_attempts"] == 1
    with response_journal_scope("encoded-resume"):
        _, replay = select(configured(), traces=traces)
    assert len(requests) == 2 and replay["physical_attempts"] == 0
    assert replay["probs"] == metadata["probs"]


def test_tau_runtime_opt_in_is_in_resume_identity_and_shared_builder(monkeypatch, tmp_path):
    manifest = {"dataset_version": "fixture-v4", "exclusion_manifest_sha256": "fixture", "records": []}
    monkeypatch.setattr(tau_main, "load_data", lambda source: ({"train": [], "val": [], "test": []}, manifest))
    monkeypatch.setattr(tau_main, "TauRuntime", lambda *args, **kwargs: None)
    monkeypatch.setattr(tau_main, "upstream_system_prompt", lambda source: "Fixed upstream instructions")
    args = SimpleNamespace(tau_source=tmp_path, run_dir=tmp_path / "run", template_family="alibaba", max_workers=1)
    models = BenchmarkModels("unused/solver", "unused/proposer", None, None, {}, {})
    definition = tau_main.build_benchmark(args, models)
    runtime = definition.runtime
    assert runtime["jev_evidence_representation"] == TAU_KB_REFERENCE_FORMAT
    old_runtime = {key: value for key, value in runtime.items() if key != "jev_evidence_representation"}
    ensure_contract(args.run_dir, "benchmark-run-contract.json", old_runtime)
    with pytest.raises(ValueError, match="use a new run directory"):
        ensure_contract(args.run_dir, "benchmark-run-contract.json", runtime)
    strategy, _ = build_react_v2_strategy(
        reflection_model="unused/proposer",
        task_model="unused/solver",
        lm_kwargs={},
        level=2,
        edit_tool_set="broad",
        template_family="alibaba",
        controller_selection="jev",
        jev_provider="openrouter",
        jev_evidence_representation=runtime["jev_evidence_representation"],
        component_kinds=definition.component_kinds,
    )
    assert strategy.run_contract(definition.seed_candidate)["controller"]["policy"] == TAU_KB_JEV_POLICY


def test_tau_opt_in_rejects_other_inputs_and_joint_components_before_dispatch(setup_controller):
    original, requests, _ = setup_controller
    controller = opt_in(original)
    with pytest.raises(JevControllerError, match="banking_knowledge"):
        select(controller, traces='[{"Inputs": {"domain": "swebench"}, "Generated Outputs": []}]')
    with pytest.raises(JevControllerError, match="single-component"):
        controller.select_components([], components={}, rng=random.Random(0))
    with pytest.raises(ValueError, match="Unsupported Jev evidence"):
        JevController(evidence_representation="unknown")
    assert requests == []


def test_shared_runner_forwards_only_the_explicit_runtime_protocol(monkeypatch, tmp_path):
    benchmark = local_definition(tmp_path)
    benchmark.runtime["jev_evidence_representation"] = TAU_KB_REFERENCE_FORMAT
    received = []

    class StopBeforeInferenceError(Exception):
        pass

    def stop(**kwargs):
        received.append(kwargs["jev_evidence_representation"])
        raise StopBeforeInferenceError

    monkeypatch.setattr(runner, "build_react_v2_strategy", stop)
    with pytest.raises(StopBeforeInferenceError):
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=lambda *args: benchmark,
            add_arguments=lambda parser: None,
            argv=[
                "--run-dir",
                str(tmp_path / "run"),
                "--mode",
                "optimizer-pilot",
                "--condition",
                "react_v2",
                "--pilot-size",
                "2",
            ],
        )
    assert received == [TAU_KB_REFERENCE_FORMAT]
    assert benchmark.adapter.calls == []
