"""Verify pre-dispatch sizing through the pinned SDK without provider calls."""

import json
import random
from copy import deepcopy

import httpx2
import pytest
from test_jev_controller import response, select
from test_three_role import PROMPT, strategy
from typesafe_sdk import RetryPolicy, TypeSafeClient

import gepa.strategies.jev_controller as jev_module
from gepa.proposer.reflective_mutation.three_role import ensure_reflection_run_contract
from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.jev_controller import JevController, JevControllerError


@pytest.fixture
def setup_controller(tmp_path):
    """Use the real SDK with an OpenRouter-shaped, in-process mock transport."""
    requests, replies = [], []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if replies:
            reply = replies.pop(0)
            return reply(payload) if callable(reply) else httpx2.Response(reply, json={"error": "fixture"})
        return mock_response(payload)

    controller = JevController(
        provider="openrouter",
        response_journal_path=tmp_path / "responses.sqlite3",
        attempt_log_path=tmp_path / "attempts.jsonl",
    )
    client = TypeSafeClient(
        api_key="test-secret-key",
        base_url="https://openrouter.ai/api",
        retry=RetryPolicy(max_retries=0),
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    controller._client = client
    yield controller, requests, replies
    client.close()


def mock_response(request):
    """Return synthetic probabilities with the pinned OpenRouter model identity."""
    result = response(request)
    result["model"] = "typesafe/jev-1.13-20260917"
    return httpx2.Response(200, json=result)


def test_three_large_records_split_before_dispatch_and_replay(setup_controller, tmp_path):
    """Split a three-record input by size, preserve all evidence, and replay without calls."""
    controller, requests, replies = setup_controller
    records = [{"id": i, "history": "x" * 300_000, "feedback": {"score": 0}} for i in range(3)]
    traces = json.dumps(records)

    def bounded_reply(request):
        assert len(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()) <= 524_288
        return mock_response(request)

    replies.extend([bounded_reply] * 3)
    rng = random.Random(19)
    before = rng.getstate()
    with response_journal_scope("large-three"):
        choice, metadata = select(controller, rng, traces)
    assert len(requests) == 3
    assert [json.loads(request["state"]["training_evidence"])[0] for request in requests] == records
    assert [group["start"] for group in metadata["evidence_groups"]] == [0, 1, 2]
    assert [group["count"] for group in metadata["evidence_groups"]] == [1, 1, 1]
    assert metadata["physical_attempts"] == 3
    for request in requests:
        assert request["questions"] == requests[0]["questions"]
        assert {k: v for k, v in request["state"].items() if k != "training_evidence"} == {
            k: v for k, v in requests[0]["state"].items() if k != "training_evidence"
        }
    replay = JevController(provider="openrouter", response_journal_path=tmp_path / "responses.sqlite3")
    rng.setstate(before)
    with response_journal_scope("large-three"):
        replay_choice, replay_metadata = select(replay, rng, traces)
    assert replay_choice == choice
    assert replay_metadata["probs"] == metadata["probs"]
    assert replay_metadata["physical_attempts"] == 0
    assert len(requests) == 3


def test_late_oversized_record_fails_before_any_group_or_rng_draw(setup_controller):
    """Reject the whole plan before spending on valid early groups."""
    controller, requests, _ = setup_controller
    traces = json.dumps([{"id": i} for i in range(3)] + [{"history": "x" * 600_000}])
    rng = random.Random(7)
    before = rng.getstate()
    with response_journal_scope("late-oversized"), pytest.raises(JevControllerError, match="evidence record 3"):
        select(controller, rng, traces)
    assert requests == []
    assert rng.getstate() == before
    assert controller.response_journal_cursor_state() == {}
    assert not controller._attempt_log.exists()


@pytest.mark.parametrize("traces", ["x" * 600_000, json.dumps({"history": "x" * 600_000})])
def test_indivisible_evidence_fails_before_dispatch(setup_controller, traces):
    """Apply the same admission check to free text and non-array evidence."""
    controller, requests, _ = setup_controller
    with pytest.raises(JevControllerError, match="client budget"):
        select(controller, traces=traces)
    assert requests == []


def test_questions_and_utf8_bytes_count_toward_budget(setup_controller, monkeypatch):
    """Count multibyte instructions even when the training evidence is tiny."""
    controller, requests, _ = setup_controller
    monkeypatch.setattr(jev_module, "JEV_SELECTION_GUIDANCE", "🌳" * 140_000)
    with pytest.raises(JevControllerError, match="evidence record 0"):
        select(controller, traces='[{"id": 0}]')
    assert requests == []


def test_size_boundary_and_joint_component_guard(setup_controller):
    """Accept exactly the byte budget and reject a larger joint-component request."""
    controller, requests, _ = setup_controller
    request = {"model": "jev-1.13.0", "state": {"components": {}}, "questions": {}}
    overhead = len(json.dumps(request, sort_keys=True, ensure_ascii=False).encode())
    request["state"]["components"] = "x" * (524_288 - overhead)
    controller._check_request_size(request)
    request["state"]["components"] += "é"
    with pytest.raises(JevControllerError, match="524290 UTF-8 JSON bytes"):
        controller._evidence_groups(request)
    assert requests == []


def test_oversized_correction_stops_before_retry(setup_controller, monkeypatch):
    """Retain the failed physical response without sending an oversized correction."""
    controller, requests, replies = setup_controller

    def invalid(request):
        result = response(request)
        result["model"] = "typesafe/jev-1.13-20260917"
        result["answers"]["edit"]["choice"] = "invalid" + "x" * 600_000
        return httpx2.Response(200, json=result)

    replies.append(invalid)
    monkeypatch.setattr(jev_module.time, "sleep", lambda seconds: None)
    with pytest.raises(JevControllerError, match="client budget"):
        select(controller)
    assert len(requests) == 1
    finished = json.loads(controller._attempt_log.read_text().splitlines()[-1])
    assert finished["event"] == "finished" and finished["will_retry"] is False
    assert finished["retry_blocked"] == "input_size_limit"
    assert controller.total_tokens_in == 1000


def test_size_group_failure_is_never_repartitioned_or_reissued(setup_controller, tmp_path):
    """Keep the grouping plan on HTTP 400 and refuse a failed group's replay."""
    controller, requests, replies = setup_controller
    replies.extend([mock_response, 400])
    traces = json.dumps([{"id": i, "history": "x" * 300_000} for i in range(3)])
    rng = random.Random(9)
    before = rng.getstate()
    with response_journal_scope("size-failure"), pytest.raises(JevControllerError):
        select(controller, rng, traces)
    assert len(requests) == 2 and rng.getstate() == before
    replay = JevController(provider="openrouter", response_journal_path=tmp_path / "responses.sqlite3")
    with response_journal_scope("size-failure"), pytest.raises(ResponseJournalError, match="started"):
        select(replay, rng, traces)
    assert len(requests) == 2 and rng.getstate() == before


def test_unequal_size_groups_use_record_weights_and_preserve_unicode(setup_controller):
    """Give every complete record equal weight, regardless of its group's size."""
    controller, requests, replies = setup_controller
    records = [{"history": "x" * 220_000, "feedback": 'שגיאה 🌳\n"error"'} for _ in range(3)]
    keys = []

    def weighted(request):
        result = response(request)
        result["model"] = "typesafe/jev-1.13-20260917"
        answer = result["answers"]["edit"]
        keys[:] = list(answer["probabilities"])[:2]
        p = 0.9 if len(json.loads(request["state"]["training_evidence"])) == 2 else 0.3
        answer["probabilities"] = {
            key: p if key == keys[0] else 1 - p if key == keys[1] else 0.0 for key in answer["probabilities"]
        }
        answer["choice"] = keys[0] if p > 0.5 else keys[1]
        return httpx2.Response(200, json=result)

    replies.extend([weighted, weighted])
    _, metadata = select(controller, traces=json.dumps(records, ensure_ascii=False))
    assert [group["count"] for group in metadata["evidence_groups"]] == [2, 1]
    assert metadata["probs"][keys[0]] == pytest.approx(0.7)
    assert metadata["probs"][keys[1]] == pytest.approx(0.3)
    assert [row for request in requests for row in json.loads(request["state"]["training_evidence"])] == records


@pytest.mark.parametrize("drift", [False, True])
def test_size_groups_resume_or_reject_budget_drift(setup_controller, tmp_path, monkeypatch, drift):
    """Reuse a committed group exactly and reject a changed sizing policy."""
    controller, requests, _ = setup_controller
    traces = json.dumps([{"id": i, "history": "x" * 300_000} for i in range(3)])
    original_store = controller._journal.store

    def interrupt(*args):
        original_store(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller._journal, "store", interrupt)
    with response_journal_scope("size-drift"), pytest.raises(KeyboardInterrupt):
        select(controller, traces=traces)
    replay = JevController(provider="openrouter", response_journal_path=tmp_path / "responses.sqlite3")
    replay._client = controller._client
    if drift:
        monkeypatch.setattr(jev_module, "JEV_OPENROUTER_REQUEST_UTF8_LIMIT", 524_289)
        with response_journal_scope("size-drift"), pytest.raises(ResponseJournalError, match="mismatch"):
            select(replay, traces=traces)
        assert len(requests) == 1
    else:
        with response_journal_scope("size-drift"):
            _, metadata = select(replay, traces=traces)
        assert len(requests) == 3
        assert metadata["physical_attempts"] == 2
        assert replay.total_tokens_in == 3000


def test_v5_reflection_checkpoint_is_incompatible(setup_controller, tmp_path):
    """Reject the old scientific contract before any optimizer or provider work."""
    controller, requests, _ = setup_controller
    reflection, _ = strategy(2, controller_selection="jev", jev_controller=controller)
    current = reflection.run_contract({"sys": PROMPT})
    prior = deepcopy(current)
    prior["controller"]["policy"] = "jev_joint_action_section_v5_evidence_pool"
    prior["controller"].pop("request_sizing")
    ensure_reflection_run_contract(tmp_path / "run", prior)
    with pytest.raises(ValueError):
        ensure_reflection_run_contract(tmp_path / "run", current)
    assert requests == []


def test_manual_response_recovery_cannot_bypass_size_check(setup_controller):
    """Block an oversized correction even through the explicit recovery entry point."""
    controller, requests, _ = setup_controller
    select(controller)
    request = requests[0]
    invalid = response(request)
    invalid["model"] = "typesafe/jev-1.13-20260917"
    invalid["answers"]["edit"]["choice"] = "invalid" + "x" * 600_000
    prior = {
        "event": "finished",
        "attempt": 1,
        "request_id": "fixture-failed-request",
        "outcome": "error",
        "error_type": "JevResponseValidationError",
        "scope": "manual-size",
        "request": request,
        "response": invalid,
    }
    with response_journal_scope("manual-size"), pytest.raises(JevControllerError, match="client budget"):
        controller.retry_failed_response(request, [prior])
    assert len(requests) == 1
