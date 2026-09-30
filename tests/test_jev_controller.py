"""Exercise the typed Controller through the real SDK with a mocked HTTP transport."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace

import httpx2
import pytest
from test_three_role import PROMPT, SYS_REFLECTIVE_DATASET, make_reflective_proposer, strategy
from test_wikipedia_react_v2_config import _hotpot_args
from typesafe_sdk import RetryPolicy, TypeSafeClient

from examples.hotpotqa.main import _run_key, build_config, build_parser, build_run_contract
from gepa.lm import LMRequestExhaustedError
from gepa.proposer.reflective_mutation.three_role import ensure_reflection_run_contract
from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.document_template import TEMPLATES
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.intervention import SEMANTIC_ACTIONS, build_controller_menu
from gepa.strategies.jev_controller import JEV_MODEL, JevController, JevControllerError


@pytest.fixture
def setup_controller(tmp_path):
    """Return a journaled Controller and an inspectable HTTP transport."""
    requests = []
    replies = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if replies:
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
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
    controller = JevController(
        response_journal_path=tmp_path / "responses.sqlite3", attempt_log_path=tmp_path / "attempts.jsonl"
    )
    controller._client = client
    yield controller, requests, replies
    client.close()


def response(request):
    """Choose a real nonempty Rules edit and return the complete distribution."""
    choices = request["questions"]["edit"]["criteria"]
    chosen = next(key for key in choices if key.startswith("reexpress@Rules/"))
    return {
        "model": JEV_MODEL,
        "usage": {"input_tokens": 1000, "output_tokens": 100},
        "answers": {
            "edit": {
                "type": "choice",
                "choice": chosen,
                "confidence": 0.9,
                "probabilities": {key: float(key == chosen) for key in choices},
            }
        },
    }


def select(controller, rng=None, traces="full training evidence"):
    template = TEMPLATES["system_prompt"]
    menu = build_controller_menu(template, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0))
    return controller.select(
        menu,
        sections=template.parse(PROMPT),
        section_descriptions=template.sections,
        traces=traces,
        rng=rng or random.Random(0),
    )


def test_live_sdk_request_preserves_constraints_evidence_and_cost(setup_controller, tmp_path):
    controller, requests, _ = setup_controller
    action, record = select(controller, traces="long evidence " * 10000)
    assert action.menu_id.startswith("reexpress@Rules/")
    assert record["sampled_reasonings"] == [None]
    assert record["physical_attempts"] == 1 and not record["replayed"]
    request = requests[0]
    assert request["model"] == JEV_MODEL
    assert request["state"]["training_evidence"] == "long evidence " * 10000
    assert request["state"]["component"] == "sys"
    choices = request["questions"]["edit"]["criteria"]
    specs = {spec.name: spec for spec in SEMANTIC_ACTIONS}
    for key, criterion in choices.items():
        spec = specs[key.split("@", 1)[0]]
        assert criterion["constraints"] == spec.instruction
        assert criterion["operator"] == spec.edit_tool.value
    assert "NOT a new instruction" in choices["contextualize@Rules/INSERT_TEXT"]["description"]
    assert "end-to-end failure alone does not prove" in request["questions"]["edit"]["instructions"]
    assert all(not key.startswith("prune_context@Task/") for key in choices)
    assert any(key.startswith("contextualize@Task/") for key in choices)
    assert record["excluded_choices"]
    assert all(value == 0 for key, value in record["sampling_probs"].items() if key != action.menu_id)
    assert controller.total_tokens_in == 1000 and controller.total_tokens_out == 100
    assert controller.total_cost == pytest.approx(0.000042)
    log = (tmp_path / "attempts.jsonl").read_text()
    assert "test-secret-key" not in log
    assert [json.loads(line)["event"] for line in log.splitlines()] == ["started", "finished"]
    assert "test-secret-key" not in json.dumps(controller.run_contract())


@pytest.mark.parametrize("damage", ["missing", "extra", "negative", "nan", "sum", "model", "usage", "argmax", "type"])
def test_invalid_results_exhaust_shared_budget_without_sampling_or_fallback(
    setup_controller, damage, tmp_path, monkeypatch
):
    controller, requests, replies = setup_controller

    def damaged(request):
        result = response(request)
        answer = result["answers"]["edit"]
        probs = answer["probabilities"]
        chosen = answer["choice"]
        if damage == "missing":
            probs.pop(next(iter(probs)))
        elif damage == "extra":
            probs["unknown"] = 0.0
        elif damage == "negative":
            probs[chosen] = -1
        elif damage == "nan":
            # The SDK rejects nonnumeric values before our distribution validator.
            probs[chosen] = "NaN"
        elif damage == "sum":
            probs[chosen] = 0.2
        elif damage == "model":
            result["model"] = "jev-future"
        elif damage == "usage":
            result["usage"]["input_tokens"] = None
        elif damage == "argmax":
            answer["choice"] = next(key for key in probs if key != chosen)
        elif damage == "type":
            answer["type"] = "score"
        return httpx2.Response(200, json=result)

    replies.extend([damaged] * 4)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    rng = random.Random(23)
    before = rng.getstate()
    with pytest.raises(LMRequestExhaustedError):
        select(controller, rng)
    assert rng.getstate() == before
    assert len(requests) == (1 if damage in {"model", "type", "nan"} else 4)
    finished = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[-1])
    assert finished["outcome"] == "error" and not finished["will_retry"]


def test_transport_recovery_has_one_shared_budget_and_does_not_perturb_rng(setup_controller, monkeypatch, tmp_path):
    controller, requests, replies = setup_controller
    replies.extend([429, 503, 502])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    _, record = select(controller)
    assert len(requests) == 4 and record["physical_attempts"] == 4
    assert requests.count(requests[0]) == 4
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    finished = [row for row in rows if row["event"] == "finished"]
    assert [row["attempt"] for row in finished] == [1, 2, 3, 4]
    assert [row["will_retry"] for row in finished] == [True, True, True, False]
    assert len({row["request_id"] for row in finished}) == 1


@pytest.mark.parametrize("status, attempts", [(401, 1), (403, 1), (400, 1), (422, 1), (500, 4)])
def test_terminal_failures_never_fall_back_to_generative_controller(setup_controller, monkeypatch, status, attempts):
    controller, requests, replies = setup_controller
    replies.extend([status] * 5)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    reflection, lm = strategy(2, controller_selection="jev", jev_controller=controller)
    with pytest.raises(JevControllerError):
        reflection.reflect({"sys": PROMPT}, SYS_REFLECTIVE_DATASET, ["sys"])
    assert len(requests) == attempts
    assert lm.roles == []


def test_exhaustion_is_fatal_to_upper_batch_fallback(setup_controller, monkeypatch):
    controller, requests, replies = setup_controller
    replies.extend([500] * 8)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    reflection, _ = strategy(2, controller_selection="jev", jev_controller=controller)
    proposer = make_reflective_proposer(reflection)
    jobs = [({"sys": PROMPT}, SYS_REFLECTIVE_DATASET, ["sys"])] * 2
    with pytest.raises(JevControllerError):
        proposer._propose_texts_batch_safe(jobs)
    assert len(requests) == 4


def test_retry_after_beyond_original_deadline_stops(setup_controller):
    controller, requests, replies = setup_controller
    replies.append(lambda _: httpx2.Response(429, headers={"retry-after": "300"}, json={"error": "rate limited"}))
    with pytest.raises(JevControllerError):
        select(controller)
    assert len(requests) == 1


def test_audit_write_failure_prevents_a_provider_request(setup_controller, monkeypatch):
    controller, requests, _ = setup_controller

    def broken_fsync(_):
        raise OSError("Disk full")

    monkeypatch.setattr("gepa.strategies.jev_controller.os.fsync", broken_fsync)
    with pytest.raises(ResponseJournalError):
        select(controller)
    assert not requests


def test_pinned_sdk_is_enforced_before_network(setup_controller, monkeypatch):
    controller, requests, _ = setup_controller
    monkeypatch.setattr("gepa.strategies.jev_controller.typesafe_sdk.__version__", "future")
    with pytest.raises(JevControllerError, match="SDK version"):
        select(controller)
    assert not requests


def test_resume_replays_without_network_or_duplicate_cost_and_rejects_drift(setup_controller, tmp_path):
    controller, requests, _ = setup_controller
    rng = random.Random(42)
    rng_before = rng.getstate()
    with response_journal_scope("iteration:7"):
        chosen, first = select(controller, rng)
    resumed = JevController(
        response_journal_path=tmp_path / "responses.sqlite3", attempt_log_path=tmp_path / "attempts.jsonl"
    )
    resumed_rng = random.Random()
    resumed_rng.setstate(rng_before)
    with response_journal_scope("iteration:7"):
        replayed, record = select(resumed, resumed_rng)
    assert replayed == chosen
    assert record["replayed"] and record["physical_attempts"] == 0
    assert first["request_id"] == record["request_id"]
    assert rng.getstate() == resumed_rng.getstate()
    assert len(requests) == 1
    assert resumed.total_cost == controller.total_cost
    resumed.restore_response_journal_cursor_state({})
    with response_journal_scope("iteration:7"), pytest.raises(ResponseJournalError):
        select(resumed, traces="changed evidence")


def test_seeded_sampling_uses_probabilities_rather_than_api_argmax(setup_controller):
    controller, _, replies = setup_controller

    def weighted(request):
        result = response(request)
        answer = result["answers"]["edit"]
        keys = list(answer["probabilities"])
        answer["probabilities"] = {key: 0.7 if key == keys[0] else 0.3 if key == keys[1] else 0.0 for key in keys}
        answer["choice"] = keys[0]
        return httpx2.Response(200, json=result)

    replies.append(weighted)
    action, record = select(controller, rng=random.Random(0))
    assert action.menu_id != record["jev_argmax"]
    assert record["sampling_probs"][record["jev_argmax"]] == pytest.approx(0.68)
    assert record["sampling_probs"][action.menu_id] == pytest.approx(0.32)


@pytest.mark.parametrize(
    "old_version", ["jev_joint_action_section_v1", "jev_joint_action_section_v2", "jev_joint_action_section_v3"]
)
def test_revised_policy_rejects_old_journal_identity_before_network(
    setup_controller, monkeypatch, tmp_path, old_version
):
    controller, requests, _ = setup_controller
    current_contract = controller.run_contract()
    assert current_contract["policy"] == "jev_joint_action_section_v4"
    old_contract = deepcopy(current_contract)
    old_contract["policy"] = old_version
    old_contract.pop("probability_normalization")
    if old_version == "jev_joint_action_section_v1":
        old_contract.pop("selection_guidance")
        old_contract.pop("canonical_constraints")
    monkeypatch.setattr(controller, "run_contract", lambda: old_contract)
    with response_journal_scope("iteration:policy-check"):
        select(controller)
    resumed = JevController(response_journal_path=tmp_path / "responses.sqlite3")
    with response_journal_scope("iteration:policy-check"), pytest.raises(ResponseJournalError):
        select(resumed)
    assert len(requests) == 1


@pytest.mark.parametrize("total", [0.99, 0.995, 1.0, 1.005, 1.01])
def test_near_unit_probability_maps_normalize_without_retry_and_replay_exactly(setup_controller, tmp_path, total):
    controller, requests, replies = setup_controller
    raw = {}

    def rounded(request):
        result = response(request)
        answer = result["answers"]["edit"]
        second = next(key for key in answer["probabilities"] if key != answer["choice"])
        raw.update(
            {
                key: 0.6 if key == answer["choice"] else total - 0.6 if key == second else 0.0
                for key in answer["probabilities"]
            }
        )
        answer["probabilities"] = raw.copy()
        return httpx2.Response(200, json=result)

    replies.append(rounded)
    rng = random.Random(19)
    initial_rng = rng.getstate()
    with response_journal_scope("iteration:rounded"):
        choice, first = select(controller, rng)
    assert len(requests) == 1 and first["physical_attempts"] == 1
    assert first["raw_probs"] == raw
    assert sum(first["probs"].values()) == pytest.approx(1.0)
    assert sum(first["sampling_probs"].values()) == pytest.approx(1.0)
    assert {key: p for key, p in first["probs"].items() if p == 0} == {key: p for key, p in raw.items() if p == 0}
    positive = [key for key, p in raw.items() if p > 0]
    assert first["probs"][positive[0]] / first["probs"][positive[1]] == pytest.approx(
        raw[positive[0]] / raw[positive[1]]
    )
    audit = first["probability_normalization"]
    assert audit["raw_total"] == pytest.approx(total)
    assert audit["scale"] == pytest.approx(1 / total)
    assert audit["applied"] is (total != 1.0)
    finished = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[-1])
    assert finished["response"]["answers"]["edit"]["probabilities"] == raw
    assert finished["probability_normalization"] == audit
    assert finished["outcome"] == "success" and not finished["will_retry"]
    resumed = JevController(
        response_journal_path=tmp_path / "responses.sqlite3", attempt_log_path=tmp_path / "attempts.jsonl"
    )
    resumed_rng = random.Random()
    resumed_rng.setstate(initial_rng)
    with response_journal_scope("iteration:rounded"):
        replayed, record = select(resumed, resumed_rng)
    assert replayed == choice and record["replayed"] and record["physical_attempts"] == 0
    assert record["probs"] == first["probs"] and record["raw_probs"] == raw
    assert record["probability_normalization"] == audit
    assert resumed_rng.getstate() == rng.getstate()
    assert resumed.total_cost == controller.total_cost
    assert len(requests) == 1


@pytest.mark.parametrize("total", [0.0, 0.2, 0.98, 0.989999, 1.010001, 1.02, 1.5])
def test_normalization_refuses_large_mass_errors(setup_controller, total, monkeypatch):
    controller, requests, replies = setup_controller

    def malformed(request):
        result = response(request)
        answer = result["answers"]["edit"]
        second = next(key for key in answer["probabilities"] if key != answer["choice"])
        answer["probabilities"] = {
            key: total / 2 if key in {answer["choice"], second} else 0.0 for key in answer["probabilities"]
        }
        return httpx2.Response(200, json=result)

    replies.extend([malformed] * 4)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    rng = random.Random(19)
    before = rng.getstate()
    with pytest.raises(JevControllerError):
        select(controller, rng)
    assert len(requests) == 4 and rng.getstate() == before


def test_deadline_prevents_more_physical_requests(setup_controller, monkeypatch):
    controller, requests, replies = setup_controller
    clock = {"now": 0.0}

    def late_reply(_):
        clock["now"] = 31.0
        return httpx2.Response(500, json={"error": "late failure"})

    replies.append(late_reply)
    monkeypatch.setattr(
        "gepa.strategies.jev_controller.time",
        SimpleNamespace(
            monotonic=lambda: clock["now"],
            time=lambda: clock["now"],
            sleep=lambda _: pytest.fail("No time left to retry"),
        ),
    )
    with pytest.raises(JevControllerError):
        select(controller)
    assert len(requests) == 1


def test_failure_bodies_and_unknown_usage_are_retained_without_credentials(setup_controller, tmp_path):
    controller, _, replies = setup_controller
    controller._api_key = "test-secret-key"
    replies.append(lambda _: httpx2.Response(400, json={"message": "test-secret-key", "api_key": "secret"}))
    with pytest.raises(JevControllerError):
        select(controller)
    log = (tmp_path / "attempts.jsonl").read_text()
    assert "test-secret-key" not in log and '"secret"' not in log
    final = json.loads(log.splitlines()[-1])
    assert final["response"]["message"] == "[REDACTED]"
    assert final["usage"] is None


def test_hotpot_wiring_has_distinct_identity_and_separate_attempt_ledger(tmp_path):
    args = _hotpot_args(condition="react_v2", controller_selection="jev", retrieval_provenance={"test": True})
    original_args = _hotpot_args(condition="react_v2", retrieval_provenance={"test": True})
    assert _run_key("react_v2", args) != _run_key("react_v2", original_args)
    contract = build_run_contract("react_v2", args)
    assert contract["optimizer"]["semantic_controller_policy"]["model"] == JEV_MODEL
    assert contract["models"]["reflection_role_decoding"]["controller"]["provider"] == "typesafe"
    config, _ = build_config("react_v2", args, {}, run_dir=str(tmp_path))
    reflection = config.reflection.reflection_strategy
    assert reflection.controller_selection == "jev"
    assert reflection.jev_controller._attempt_log == tmp_path / "jev-provider-attempts.jsonl"
    assert reflection.jev_controller._journal.namespace == "jev-controller"
    assert reflection.base_lm._response_journal.namespace == "proposer"
    assert build_parser().parse_args(["--controller-selection", "jev"]).controller_selection == "jev"
    args.enforce_scientific_contract = True
    with pytest.raises(ValueError, match="Jev"):
        build_run_contract("react_v2", args)


@pytest.mark.parametrize("condition,level", [("vanilla", 2), ("react_v2_random", 2), ("react_v2", 1)])
def test_jev_cannot_silently_change_other_conditions(condition, level):
    args = _hotpot_args(controller_selection="jev", reflection_level=level)
    with pytest.raises(ValueError, match="Jev requires"):
        build_run_contract(condition, args)


def test_strategy_uses_jev_then_manifestor_and_editor_and_restores_batch_state(setup_controller, tmp_path):
    controller, requests, _ = setup_controller
    reflection, lm = strategy(2, controller_selection="jev", jev_controller=controller)
    before = deepcopy(reflection.get_batch_retry_state())
    with response_journal_scope("iteration:1"):
        proposal, _ = reflection.reflect({"sys": PROMPT}, SYS_REFLECTIVE_DATASET, ["sys"])
    assert lm.roles == ["manifestor", "react_v2", "react_v2"]
    assert "be kind" in proposal.new_texts["sys"]
    assert len(requests) == 1
    assert "vague answer" in requests[0]["state"]["training_evidence"]
    assert "test-secret-key" not in json.dumps(proposal.metadata)
    assert reflection.total_cost == controller.total_cost
    reflection.set_batch_retry_state(before)
    assert reflection.get_batch_retry_state() == before
    contract = reflection.run_contract({"sys": PROMPT})
    assert contract["controller"]["model"] == JEV_MODEL
    assert contract["controller_lm"] is None
    assert "Manifestor" in contract["generalization"]["controller_direction"]
    ensure_reflection_run_contract(tmp_path / "run", contract)
    original, _ = strategy(2)
    with pytest.raises(ValueError):
        ensure_reflection_run_contract(tmp_path / "run", original.run_contract({"sys": PROMPT}))


def inconsistent_choice(request):
    result = response(request)
    result["answers"]["edit"]["choice"] = next(
        key for key, value in result["answers"]["edit"]["probabilities"].items() if value == 0
    )
    return httpx2.Response(200, json=result)


def test_response_correction_preserves_task_and_charges_both_attempts(setup_controller, monkeypatch, tmp_path):
    controller, requests, replies = setup_controller
    replies.append(inconsistent_choice)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    _, metadata = select(controller)
    assert len(requests) == metadata["physical_attempts"] == 2
    assert requests[0]["state"] == requests[1]["state"]
    assert requests[0]["questions"]["edit"]["criteria"] == requests[1]["questions"]["edit"]["criteria"]
    instruction = requests[1]["questions"]["edit"]["instructions"]
    assert instruction.startswith(requests[0]["questions"]["edit"]["instructions"])
    assert "argmax choice disagrees" in instruction and "Previous response:" in instruction
    assert "choice must have maximum probability" in instruction
    assert controller.total_tokens_in == 2000
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    finished = [row for row in rows if row["event"] == "finished"]
    assert [row["outcome"] for row in finished] == ["error", "success"]
    assert [row["will_retry"] for row in finished] == [True, False]
    assert len({row["request_id"] for row in finished}) == 1
    assert len({row["logical_request_sha256"] for row in finished}) == 1


def test_response_and_transport_retries_share_four_attempts(setup_controller, monkeypatch, tmp_path):
    controller, requests, replies = setup_controller
    replies.extend([429, inconsistent_choice, 503])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    _, metadata = select(controller)
    assert len(requests) == metadata["physical_attempts"] == 4
    assert requests[0] == requests[1]
    assert requests[2] == requests[3] != requests[0]
    assert controller.total_tokens_in == 2000


def test_response_correction_cannot_extend_original_deadline(setup_controller, monkeypatch, tmp_path):
    controller, requests, replies = setup_controller
    clock = [0.0]
    monkeypatch.setattr("gepa.strategies.jev_controller.time.monotonic", lambda: clock[0])

    def expire(request):
        clock[0] = 31.0
        return inconsistent_choice(request)

    replies.append(expire)
    with pytest.raises(JevControllerError):
        select(controller)
    assert len(requests) == 1
    row = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[-1])
    assert not row["will_retry"]


def test_explicit_recovery_consumes_existing_attempt_and_preserves_failure(setup_controller, monkeypatch, tmp_path):
    controller, requests, replies = setup_controller
    clock = [0.0]
    monkeypatch.setattr("gepa.strategies.jev_controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)

    def expire(request):
        clock[0] = 31.0
        return inconsistent_choice(request)

    replies.append(expire)
    with response_journal_scope("optimizer-iteration-53"), pytest.raises(JevControllerError):
        select(controller)
    path = tmp_path / "attempts.jsonl"
    previous_bytes = path.read_bytes()
    previous = json.loads(previous_bytes.splitlines()[-1])
    clock[0] = 40.0
    with response_journal_scope("optimizer-iteration-53"):
        result = controller.retry_failed_response(previous["request"], [previous])
    assert result["physical_attempts"] == 2 and len(requests) == 2
    assert path.read_bytes().startswith(previous_bytes)
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["attempt"] == 2 and row["request_id"] == previous["request_id"]
    assert row["manual_recovery"]["prior_attempt"] == 1
    assert "argmax choice disagrees" in requests[1]["questions"]["edit"]["instructions"]
    assert controller.total_tokens_in == 2000


def test_explicit_recovery_does_not_grant_four_more_attempts(setup_controller, monkeypatch, tmp_path):
    controller, requests, replies = setup_controller
    clock = [0.0]
    monkeypatch.setattr("gepa.strategies.jev_controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)

    def expire(request):
        clock[0] = 31.0
        return inconsistent_choice(request)

    replies.append(expire)
    with response_journal_scope("optimizer-iteration-53"), pytest.raises(JevControllerError):
        select(controller)
    previous = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[-1])
    clock[0] = 40.0
    replies.extend([inconsistent_choice] * 4)
    with response_journal_scope("optimizer-iteration-53"), pytest.raises(JevControllerError):
        controller.retry_failed_response(previous["request"], [previous])
    assert len(requests) == 4
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["attempt"] for row in rows if row["event"] == "finished"] == [1, 2, 3, 4]


def test_manual_recovery_rejects_changed_request_before_network(setup_controller):
    controller, requests, _ = setup_controller
    with response_journal_scope("different"), pytest.raises(ValueError):
        controller.retry_failed_response(
            {"questions": {"edit": {"criteria": {}}}},
            [
                {
                    "event": "finished",
                    "attempt": 1,
                    "request_id": "prior",
                    "scope": "old",
                    "outcome": "error",
                    "request": {},
                }
            ],
        )
    assert not requests
