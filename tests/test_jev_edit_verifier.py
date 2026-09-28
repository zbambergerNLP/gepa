"""Exercise the semantic verifier through the pinned SDK with no external calls."""

import json
from copy import deepcopy

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.jev_controller import JEV_MODEL, JevControllerError
from gepa.strategies.jev_edit_verifier import JevEditVerifier

EVIDENCE = {
    "canonical_parent": "Answer using evidence.",
    "proposed_after": "Answer using evidence. Return a bare name.",
    "current_minibatch_ids": [1, 2, 3],
    "current_training_evidence": [{"input": "full question", "feedback": "long answer was incorrect"}],
    "previous_records": [
        {
            "attempt_id": "attempt-a",
            "after": "Answer using evidence. Include no explanation.",
            "minibatch_ids": [1, 2, 3],
            "training_evidence": [{"feedback": "observed training feedback"}],
            "evaluated": True,
            "duplicate_judgment_allowed": True,
        },
        {
            "attempt_id": "attempt-b",
            "after": "Answer using evidence. Give the name alone.",
            "minibatch_ids": [4, 5, 6],
            "training_evidence": [{"feedback": "different observed feedback"}],
            "evaluated": True,
            "duplicate_judgment_allowed": True,
        },
    ],
}


def response(request, probabilities=None):
    """Return a complete typed response with independently inspectable probabilities."""
    choices = request["questions"]["edit"]["criteria"]
    probs = {choice: float(choice == "duplicate_1") for choice in choices}
    if probabilities is not None:
        probs.update(probabilities)
    choice = max(probs, key=probs.get)
    return {
        "model": JEV_MODEL,
        "usage": {"input_tokens": 1000, "output_tokens": 10},
        "answers": {"edit": {"type": "choice", "choice": choice, "confidence": 0.99, "probabilities": probs}},
    }


@pytest.fixture
def setup_verifier(tmp_path):
    """Return an SDK-backed verifier and an inspectable mock network boundary."""
    requests, replies = [], []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if replies:
            reply = replies.pop(0)
            if callable(reply):
                return reply(payload)
            return httpx2.Response(reply, json={"error": "test failure"})
        return httpx2.Response(200, json=response(payload))

    client = TypeSafeClient(
        api_key="test-secret-key",
        base_url="https://api.typesafe.ai",
        retry=RetryPolicy(max_retries=0),
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    verifier = JevEditVerifier(
        api_key="test-secret-key",
        response_journal_path=tmp_path / "responses.sqlite3",
        attempt_log_path=tmp_path / "attempts.jsonl",
    )
    verifier._client = client
    yield verifier, requests, replies
    client.close()


def classify(verifier, evidence=None):
    """Bind one stable logical opportunity to the model comparison."""
    with response_journal_scope("parent-2/component-answer/iteration-8"):
        return json.loads(verifier.classify_edit(evidence or EVIDENCE, "Canonical semantic constraints."))


def test_duplicate_maps_the_exact_prior_attempt_and_preserves_full_evidence(setup_verifier, tmp_path):
    verifier, requests, _ = setup_verifier
    result = classify(verifier)
    assert result == {
        "verdict": "duplicate",
        "matched_attempt_id": "attempt-b",
        "reason": "Jev classified the edit as duplicate of supplied attempt attempt-b.",
    }
    assert requests[0]["state"] == EVIDENCE
    assert "Canonical semantic constraints." in requests[0]["questions"]["edit"]["instructions"]
    assert requests[0]["questions"]["edit"]["criteria"]["duplicate_1"]["attempt_id"] == "attempt-b"
    assert verifier.last_evidence["physical_attempts"] == 1
    assert verifier.last_evidence["calibrated_confidence"] is False
    assert verifier.total_tokens_in == 1000 and verifier.total_tokens_out == 10
    assert verifier.total_cost == pytest.approx(0.000042)
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "finished"]
    assert all(row["role"] == "edit_verifier" for row in rows)
    assert rows[-1]["response"] == verifier.last_evidence["response"]
    assert "test-secret-key" not in (tmp_path / "attempts.jsonl").read_text()
    assert verifier.run_contract()["selection"] == "deterministic provider argmax; no sampling or exploration"
    assert "not calibrated" in verifier.run_contract()["threshold_interpretation"]


@pytest.mark.parametrize(
    ("probabilities", "verdict"),
    [
        ({"duplicate_1": 0.79, "distinct": 0.2, "uncertain": 0.01}, "uncertain"),
        ({"duplicate_1": 0.1, "distinct": 0.9}, "distinct"),
        ({"duplicate_1": 0.1, "uncertain": 0.9}, "uncertain"),
        ({"duplicate_1": 0.8, "distinct": 0.2}, "duplicate"),
    ],
)
def test_ambiguous_duplicate_support_does_not_block(setup_verifier, probabilities, verdict):
    verifier, requests, replies = setup_verifier
    replies.append(lambda request: httpx2.Response(200, json=response(request, probabilities)))
    result = classify(verifier)
    assert result["verdict"] == verdict
    assert result["matched_attempt_id"] == ("attempt-b" if verdict == "duplicate" else None)
    assert len(requests) == 1


def test_bounded_normalization_preserves_raw_probabilities(setup_verifier):
    verifier, _, replies = setup_verifier
    replies.append(lambda request: httpx2.Response(200, json=response(request, {"duplicate_1": 0.8, "distinct": 0.19})))
    assert classify(verifier)["verdict"] == "duplicate"
    record = verifier.last_evidence
    assert record["response"]["answers"]["edit"]["probabilities"]["duplicate_1"] == 0.8
    assert record["probs"]["duplicate_1"] == pytest.approx(0.8 / 0.99)
    assert record["probability_normalization"]["applied"] is True


def test_journal_replay_after_cursor_rewind_never_repeats_network_or_cost(setup_verifier, tmp_path):
    verifier, requests, _ = setup_verifier
    cursor = verifier.response_journal_cursor_state()
    first = classify(verifier)
    verifier.restore_response_journal_cursor_state(cursor)
    assert classify(verifier) == first
    assert verifier.last_evidence["replayed"] is True
    assert verifier.last_evidence["physical_attempts"] == 0
    assert len(requests) == 1 and verifier.total_tokens_in == 1000
    resumed = JevEditVerifier(
        response_journal_path=tmp_path / "responses.sqlite3", attempt_log_path=tmp_path / "attempts.jsonl"
    )
    assert classify(resumed) == first
    assert resumed.total_tokens_in == 1000
    assert resumed._client is None


def test_raw_response_is_durable_before_verdict_exposed(setup_verifier, monkeypatch, tmp_path):
    verifier, requests, _ = setup_verifier

    def fail_store(*args):
        raise ResponseJournalError("test disk failure")

    monkeypatch.setattr(verifier._journal, "store", fail_store)
    with pytest.raises(ResponseJournalError, match="test disk failure"):
        classify(verifier)
    assert len(requests) == 1
    assert verifier.last_evidence is None
    assert verifier.response_journal_cursor_state() == {}
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "finished" and rows[-1]["response"]["model"] == JEV_MODEL


@pytest.mark.parametrize("invalid", ["sum", "missing", "argmax", "model"])
def test_invalid_distribution_is_not_semantically_retried(setup_verifier, invalid):
    verifier, requests, replies = setup_verifier

    def damaged(request):
        result = response(request)
        answer = result["answers"]["edit"]
        if invalid == "sum":
            answer["probabilities"]["duplicate_1"] = 0.5
        elif invalid == "missing":
            answer["probabilities"].pop("uncertain")
        elif invalid == "argmax":
            answer["choice"] = "distinct"
        else:
            result["model"] = "different-model"
        return httpx2.Response(200, json=result)

    replies.append(damaged)
    with pytest.raises(JevControllerError):
        classify(verifier)
    assert len(requests) == 1 and verifier.last_evidence is None


@pytest.mark.parametrize("status, attempts", [(401, 1), (422, 1), (503, 4)])
def test_transport_budget_does_not_multiply(setup_verifier, monkeypatch, status, attempts, tmp_path):
    verifier, requests, replies = setup_verifier
    replies.extend([status] * 5)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    with pytest.raises(JevControllerError):
        classify(verifier)
    assert len(requests) == attempts
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    finished = [row for row in rows if row["event"] == "finished"]
    assert [row["attempt"] for row in finished] == list(range(1, attempts + 1))
    assert len({row["request_id"] for row in finished}) == 1
    assert finished[-1]["will_retry"] is False


def test_recovered_transport_has_one_request_identity(setup_verifier, monkeypatch):
    verifier, requests, replies = setup_verifier
    replies.extend([429, 503, 502])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    assert classify(verifier)["verdict"] == "duplicate"
    assert len(requests) == 4 and requests.count(requests[0]) == 4
    assert verifier.last_evidence["physical_attempts"] == 4


def test_original_deadline_prevents_retry_after_long_delay(setup_verifier):
    verifier, requests, replies = setup_verifier
    replies.append(lambda _: httpx2.Response(429, headers={"retry-after": "300"}, json={"error": "rate limited"}))
    with pytest.raises(JevControllerError):
        classify(verifier)
    assert len(requests) == 1


def test_ineligible_other_evidence_record_cannot_be_named_as_duplicate(setup_verifier):
    verifier, requests, replies = setup_verifier
    evidence = deepcopy(EVIDENCE)
    evidence["previous_records"][0]["duplicate_judgment_allowed"] = False
    evidence["previous_records"][0]["training_evidence"] = None
    assert classify(verifier, evidence)["verdict"] == "duplicate"
    assert "duplicate_0" not in requests[0]["questions"]["edit"]["criteria"]
    evidence["previous_records"][1]["duplicate_judgment_allowed"] = False
    assert classify(verifier, evidence)["verdict"] == "uncertain"
    assert len(requests) == 1 and not verifier.last_evidence["model_called"]


def test_missing_journal_or_scope_fails_before_network(setup_verifier):
    verifier, requests, _ = setup_verifier
    with pytest.raises(ResponseJournalError):
        verifier.classify_edit(EVIDENCE, "constraints")
    with pytest.raises(ResponseJournalError):
        classify(JevEditVerifier())
    assert requests == []


def test_error_evidence_redacts_echoed_credentials(setup_verifier, tmp_path):
    verifier, _, replies = setup_verifier
    replies.append(
        lambda _: httpx2.Response(400, json={"authorization": "test-secret-key", "error": "key test-secret-key"})
    )
    with pytest.raises(JevControllerError):
        classify(verifier)
    log = (tmp_path / "attempts.jsonl").read_text()
    assert "test-secret-key" not in log and "[REDACTED]" in log
