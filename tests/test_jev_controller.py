"""Exercise the typed Controller through the real SDK with a mocked HTTP transport."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace

import httpx2
import pytest
from test_generation_recovery import Roles
from test_three_role import PROMPT, SYS_REFLECTIVE_DATASET, make_reflective_proposer, strategy
from test_wikipedia_react_v2_config import _hotpot_args
from typesafe_sdk import RetryPolicy, TypeSafeClient

from examples.hotpotqa.main import _run_key, build_config, build_parser, build_run_contract
from gepa.lm import LMRequestExhaustedError
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM, ensure_reflection_run_contract
from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.document_template import TEMPLATES
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.intervention import SEMANTIC_ACTIONS, build_controller_menu
from gepa.strategies.jev_controller import JEV_MODEL, JevController, JevControllerError
from gepa.strategies.reflection_context import REAL_EDIT_GUIDANCE


@pytest.fixture
def setup_controller(tmp_path):
    """Yield a journaled Controller and an inspectable mocked HTTP transport.

    Args:
        tmp_path: Isolated directory for response and attempt journals.

    Yields:
        Controller, captured request list and mutable queue of mocked replies.
    """
    requests = []
    replies = []

    def handler(request):
        """Record a request and consume the next configured transport reply.

        Args:
            request: Outgoing HTTP request produced by the real SDK.

        Returns:
            Scripted HTTP response, or the default valid typed answer.

        Raises:
            Exception: The next queued reply is an exception to simulate.
        """
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
    """Choose a real nonempty Rules edit and return the complete distribution.

    Args:
        request: Typed payload whose criteria include a Rules reexpression.

    Returns:
        Valid Jev response with token usage and all probability mass on that edit.
    """
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
    """Select from the standard semantic menu using the supplied test Controller.

    Args:
        controller: Controller configured for mocked HTTP or journal replay.
        rng: Optional seeded selection RNG; otherwise use seed zero.
        traces: Training evidence to include unchanged in the request.

    Returns:
        Selected action/section choice and its Controller audit metadata.
    """
    template = TEMPLATES["system_prompt"]
    menu = build_controller_menu(template, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0))
    return controller.select(
        menu,
        sections=template.parse(PROMPT),
        section_descriptions=template.sections,
        traces=traces,
        rng=rng or random.Random(0),
    )


def recovery_strategy(controller, roles):
    return ThreeRoleReflectionLM(
        roles,
        level=2,
        controller_selection="jev",
        jev_controller=controller,
        base_lm_run_identity={"test": "roles"},
        rng=random.Random(0),
    )


def recover(reflection):
    return reflection.reflect(
        {"sys": PROMPT},
        SYS_REFLECTIVE_DATASET,
        ["sys"],
        metadata={"candidate_idx": 0, "iteration_id": "test", "minibatch_ids": [0]},
    )[0]


@pytest.mark.parametrize("failure", ["finish", "empty", "manifestor"])
def test_jev_no_edit_recovery_uses_one_distribution_and_preserves_evidence(setup_controller, failure):
    controller, requests, _ = setup_controller

    class OnceIncompatible(Roles):
        def __call__(self, prompt):
            self.incompatible = failure == "manifestor" and not self.manifestor_prompts
            return super().__call__(prompt)

    roles = OnceIncompatible([failure] if failure != "manifestor" else [])
    reflection = recovery_strategy(controller, roles)
    proposal = recover(reflection)
    assert proposal.new_texts and not roles.controller_prompts
    assert len(requests) == 1
    assert requests[0]["state"]["training_evidence"] == json.dumps(
        SYS_REFLECTIVE_DATASET["sys"], sort_keys=True, ensure_ascii=False
    )
    assert REAL_EDIT_GUIDANCE in requests[0]["questions"]["edit"]["instructions"]
    plan = proposal.metadata["controller_plans"]["sys"]["controller_sampling"]
    records = proposal.metadata["attempt_records"]
    assert records[0]["action_choice"] == plan["sampled"][0] == "reexpress@Rules/REPLACE_TEXT"
    assert plan["physical_attempts"] == 1
    assert {r["controller_sampling"]["source_request_id"] for r in records} == {plan["request_id"]}
    assert len({r["action_choice"] for r in records}) == len(records)
    assert all("physical_attempts" not in r["controller_sampling"] for r in records)
    if failure == "empty":
        assert len(records) == 1 and len(roles.editor_tasks) == 2
    else:
        assert len(records) >= 2 and records[0]["attempt_status"] == "generation_error"
    contract = reflection.run_contract({"sys": PROMPT})
    assert contract["react_execution"]["max_iterations"] == 2
    assert contract["controller"]["model"] == JEV_MODEL


def test_jev_recovery_resumes_without_repeating_the_controller_request(setup_controller, tmp_path):
    controller, requests, _ = setup_controller
    interrupted = recovery_strategy(controller, Roles(["finish"], interrupt_at=1))
    interrupted.recovery_planner.bind_run_dir(str(tmp_path))
    before = interrupted.get_state()
    with pytest.raises(KeyboardInterrupt):
        recover(interrupted)
    resumed_roles = Roles()
    resumed = recovery_strategy(controller, resumed_roles)
    resumed.recovery_planner.bind_run_dir(str(tmp_path))
    resumed.set_state(before)
    proposal = recover(resumed)
    assert proposal.new_texts and len(requests) == 1
    assert proposal.metadata["generation_error_count"] >= 1
    assert not resumed_roles.controller_prompts
    assert len(resumed_roles.editor_tasks) == 1


def test_jev_provider_exhaustion_does_not_trigger_pair_recovery(setup_controller, monkeypatch):
    controller, requests, replies = setup_controller
    replies.extend([500] * 5)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    roles = Roles()
    with pytest.raises(JevControllerError):
        recover(recovery_strategy(controller, roles))
    assert len(requests) == 4
    assert not roles.manifestor_prompts and not roles.editor_tasks


def test_jev_recovery_preserves_first_sample_then_tries_remaining_positive_support(setup_controller):
    controller, requests, replies = setup_controller

    def weighted(request):
        result = response(request)
        answer = result["answers"]["edit"]
        answer["probabilities"] = {
            key: 0.7
            if key == "reexpress@Rules/REPLACE_TEXT"
            else 0.3
            if key == "contextualize@Role/INSERT_TEXT"
            else 0.0
            for key in answer["probabilities"]
        }
        return httpx2.Response(200, json=result)

    replies.append(weighted)
    proposal = recover(recovery_strategy(controller, Roles(["finish"])))
    records = proposal.metadata["attempt_records"]
    plan = proposal.metadata["controller_plans"]["sys"]["controller_sampling"]
    assert len(requests) == 1 and len(records) == 2
    assert records[0]["action_choice"] == plan["sampled"][0]
    assert records[0]["controller_sampling"]["sampled_probabilities"] == pytest.approx(plan["sampled_probabilities"])
    assert {r["action_choice"] for r in records} == {"reexpress@Rules/REPLACE_TEXT", "contextualize@Role/INSERT_TEXT"}
    assert records[1]["controller_sampling"]["phase"] == "positive"


def test_live_sdk_request_preserves_constraints_evidence_and_cost(setup_controller, tmp_path):
    """Preserve complete evidence, canonical constraints and recorded provider usage."""
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
    """Exhaust correction retries without sampling or invoking a fallback."""
    controller, requests, replies = setup_controller

    def damaged(request):
        """Corrupt one configured response field while retaining the other evidence.

        Args:
            request: Typed payload used to construct the initially valid answer.

        Returns:
            HTTP success response containing the selected contract violation.
        """
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
    """Retry the same request within one allowance and retain every physical attempt."""
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


@pytest.mark.parametrize(
    "status, attempts", [(401, 1), (403, 1), (400, 1), (422, 1), (408, 4), (429, 4), (500, 4), (501, 4), (599, 4)]
)
def test_terminal_failures_never_fall_back_to_generative_controller(setup_controller, monkeypatch, status, attempts):
    """Apply status-specific retry limits without calling the generative Controller."""
    controller, requests, replies = setup_controller
    replies.extend([status] * 5)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    reflection, lm = strategy(2, controller_selection="jev", jev_controller=controller)
    with pytest.raises(JevControllerError):
        reflection.reflect({"sys": PROMPT}, SYS_REFLECTIVE_DATASET, ["sys"])
    assert len(requests) == attempts
    assert lm.roles == []


def test_exhaustion_is_fatal_to_upper_batch_fallback(setup_controller, monkeypatch):
    """Prevent batch fallback from restarting an exhausted Jev request."""
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
    """Stop when Retry-After would exceed the original request deadline."""
    controller, requests, replies = setup_controller
    replies.append(lambda _: httpx2.Response(429, headers={"retry-after": "300"}, json={"error": "rate limited"}))
    with pytest.raises(JevControllerError):
        select(controller)
    assert len(requests) == 1


def test_audit_write_failure_prevents_a_provider_request(setup_controller, monkeypatch):
    """Refuse a provider call when its started event cannot be durably recorded."""
    controller, requests, _ = setup_controller

    def broken_fsync(_):
        """Simulate a disk failure while persisting the started event.

        Args:
            _: Ignored file descriptor supplied by the attempt logger.

        Raises:
            OSError: The simulated disk has no remaining space.
        """
        raise OSError("Disk full")

    monkeypatch.setattr("gepa.strategies.jev_controller.os.fsync", broken_fsync)
    with pytest.raises(ResponseJournalError):
        select(controller)
    assert not requests


def test_pinned_sdk_is_enforced_before_network(setup_controller, monkeypatch):
    """Reject SDK version drift before making any provider request."""
    controller, requests, _ = setup_controller
    monkeypatch.setattr("gepa.strategies.jev_controller.typesafe_sdk.__version__", "future")
    with pytest.raises(JevControllerError, match="SDK version"):
        select(controller)
    assert not requests


def test_resume_replays_without_network_or_duplicate_cost_and_rejects_drift(setup_controller, tmp_path):
    """Replay the same decision and cost once while rejecting changed request evidence."""
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


@pytest.mark.parametrize("count", [3, 7, 16, 32])
def test_large_evidence_preserves_every_record_and_samples_once(setup_controller, count):
    """Keep duplicates, Unicode, error records, full sections and every choice."""
    controller, requests, _ = setup_controller
    records = [{"input": "עברית " + "repeated\n" * 500, "id": i} for i in range(count)]
    records[-1] = {"evaluation_error": "parse failure"}
    traces = json.dumps(records, sort_keys=True, ensure_ascii=False)
    rng = random.Random(42)
    reference = random.Random(42)
    reference.random()
    with response_journal_scope("iteration:large"):
        _, metadata = select(controller, rng=rng, traces=traces)
    assert len(requests) == (count + 2) // 3
    assert [row for request in requests for row in json.loads(request["state"]["training_evidence"])] == records
    assert all(request["state"]["sections"] == TEMPLATES["system_prompt"].parse(PROMPT) for request in requests)
    assert all(request["questions"] == requests[0]["questions"] for request in requests)
    assert rng.getstate() == reference.getstate()
    assert controller.total_tokens_in == len(requests) * 1000
    assert metadata["physical_attempts"] == len(requests)
    if count <= 3:
        assert requests[0]["state"]["training_evidence"] == traces
        assert "evidence_groups" not in metadata
    else:
        assert metadata["distribution_source"] == "record_weighted_evidence_pool"
        assert metadata["confidence"] is metadata["raw_probs"] is metadata["jev_argmax"] is None
        assert sum(group["count"] for group in metadata["evidence_groups"]) == count


def test_group_probabilities_weight_records_then_apply_exploration(setup_controller):
    """Normalize each raw map and give a one-record final group its correct weight."""
    controller, _, replies = setup_controller

    def weighted(request):
        result = response(request)
        answer = result["answers"]["edit"]
        keys = list(answer["probabilities"])
        start = json.loads(request["state"]["training_evidence"])[0]["id"]
        chosen = keys[1] if start == 6 else keys[0]
        answer["probabilities"] = {key: 0.995 if key == chosen else 0.0 for key in keys}
        answer["choice"] = chosen
        return httpx2.Response(200, json=result)

    replies.extend([weighted] * 3)
    with response_journal_scope("iteration:weights"):
        _, metadata = select(controller, traces=json.dumps([{"id": i} for i in range(7)]))
    keys = list(metadata["evidence_groups"][0]["payload"]["response"]["answers"]["edit"]["probabilities"])
    assert metadata["probs"][keys[0]] == pytest.approx(6 / 7)
    assert metadata["probs"][keys[1]] == pytest.approx(1 / 7)
    assert metadata["sampling_probs"][keys[0]] == pytest.approx(0.9 * 6 / 7 + 0.05)
    assert metadata["sampling_probs"][keys[1]] == pytest.approx(0.9 / 7 + 0.05)
    assert all(metadata["sampling_probs"][key] == 0 for key in keys[2:])
    assert all(group["payload"]["probability_normalization"]["applied"] for group in metadata["evidence_groups"])


@pytest.mark.parametrize("with_attempt_log", [True, False])
def test_group_replay_restores_usage_once(setup_controller, tmp_path, with_attempt_log):
    """Restore physical usage from either journal source without counting the pool twice."""
    controller, requests, _ = setup_controller
    if not with_attempt_log:
        controller._attempt_log = None
    traces = json.dumps([{"id": i} for i in range(7)])
    rng = random.Random(42)
    with response_journal_scope("iteration:pool-replay"):
        choice, first = select(controller, rng=rng, traces=traces)
    replay = JevController(
        response_journal_path=tmp_path / "responses.sqlite3",
        attempt_log_path=tmp_path / "attempts.jsonl" if with_attempt_log else None,
    )
    replay_rng = random.Random(42)
    with response_journal_scope("iteration:pool-replay"):
        chosen, metadata = select(replay, rng=replay_rng, traces=traces)
    assert chosen == choice and replay_rng.getstate() == rng.getstate()
    assert metadata["replayed"] and metadata["physical_attempts"] == 0
    assert metadata["request_id"] == first["request_id"]
    assert replay.total_tokens_in == controller.total_tokens_in == metadata["usage"]["tokens_in"] == 3000
    assert replay.total_cost == pytest.approx(controller.total_cost)
    assert len(requests) == 3


@pytest.mark.parametrize("drift", [False, True])
def test_partial_group_resume_reuses_completed_work_and_binds_whole_batch(
    setup_controller, tmp_path, monkeypatch, drift
):
    """Replay completed groups and reject changes even to a later unrequested group."""
    controller, requests, _ = setup_controller
    records = [{"id": i} for i in range(7)]
    original_store = controller._journal.store

    def interrupt_after_commit(*args):
        original_store(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller._journal, "store", interrupt_after_commit)
    rng = random.Random(42)
    before = rng.getstate()
    with response_journal_scope("iteration:partial"), pytest.raises(KeyboardInterrupt):
        select(controller, rng=rng, traces=json.dumps(records))
    assert rng.getstate() == before and len(requests) == 1
    replay = JevController(
        response_journal_path=tmp_path / "responses.sqlite3", attempt_log_path=tmp_path / "attempts.jsonl"
    )
    replay._client = controller._client
    if drift:
        records[-1]["id"] = "changed after committed group"
        with response_journal_scope("iteration:partial"), pytest.raises(ResponseJournalError, match="mismatch"):
            select(replay, rng=rng, traces=json.dumps(records))
        assert len(requests) == 1 and rng.getstate() == before
    else:
        with response_journal_scope("iteration:partial"):
            _, metadata = select(replay, rng=rng, traces=json.dumps(records))
        assert len(requests) == 3 and replay.total_tokens_in == 3000
        assert metadata["physical_attempts"] == 2 and metadata["original_physical_attempts"] == 3


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), 400])
def test_started_or_failed_group_cannot_be_blindly_reissued(setup_controller, tmp_path, failure):
    """Keep unknown attempts and terminal errors; never drop a failing group or draw an action."""
    controller, requests, replies = setup_controller

    def interrupt(request):
        raise KeyboardInterrupt

    replies.extend([lambda request: httpx2.Response(200, json=response(request)), interrupt if failure != 400 else 400])
    traces = json.dumps([{"id": i} for i in range(7)])
    rng = random.Random(42)
    before = rng.getstate()
    error = KeyboardInterrupt if failure != 400 else JevControllerError
    with response_journal_scope("iteration:failed-group"), pytest.raises(error):
        select(controller, rng=rng, traces=traces)
    assert len(requests) == 2 and rng.getstate() == before
    replay = JevController(
        response_journal_path=tmp_path / "responses.sqlite3", attempt_log_path=tmp_path / "attempts.jsonl"
    )
    replay._client = controller._client
    with response_journal_scope("iteration:failed-group"), pytest.raises(ResponseJournalError, match="started"):
        select(replay, rng=rng, traces=traces)
    assert len(requests) == 2 and rng.getstate() == before


def test_large_v4_decision_cannot_replay_as_grouped_policy(setup_controller, tmp_path, monkeypatch):
    """Reject an old full-batch journal before issuing any grouped request."""
    controller, requests, _ = setup_controller
    old_policy = controller.run_contract()
    old_policy["policy"] = "jev_joint_action_section_v4"
    monkeypatch.setattr(controller, "run_contract", lambda: old_policy)
    monkeypatch.setattr(controller, "_evidence_groups", lambda request: [])
    traces = json.dumps([{"id": i} for i in range(7)])
    with response_journal_scope("iteration:v4"):
        select(controller, traces=traces)
    replay = JevController(response_journal_path=tmp_path / "responses.sqlite3")
    with response_journal_scope("iteration:v4"), pytest.raises(ResponseJournalError, match="mismatch"):
        select(replay, traces=traces)
    assert len(requests) == 1


def test_seeded_sampling_uses_probabilities_rather_than_api_argmax(setup_controller):
    """Sample with the seeded probability mixture instead of returning the API argmax."""
    controller, _, replies = setup_controller

    def weighted(request):
        """Build a valid answer with two supported choices and a known argmax.

        Args:
            request: Typed payload supplying the complete choice set.

        Returns:
            HTTP response with unequal nonzero weights on its first two choices.
        """
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
    "old_version",
    [
        "jev_joint_action_section_v1",
        "jev_joint_action_section_v2",
        "jev_joint_action_section_v3",
        "jev_joint_action_section_v4",
    ],
)
def test_revised_policy_rejects_old_journal_identity_before_network(
    setup_controller, monkeypatch, tmp_path, old_version
):
    """Reject replay under an older policy identity without another provider call."""
    controller, requests, _ = setup_controller
    current_contract = controller.run_contract()
    assert current_contract["policy"] == "jev_joint_action_section_v5_evidence_pool"
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
    """Normalize bounded mass errors while preserving raw evidence, support and replay."""
    controller, requests, replies = setup_controller
    raw = {}

    def rounded(request):
        """Return the requested near-unit total and retain its raw probability map.

        Args:
            request: Typed payload used to construct the supported choices.

        Returns:
            HTTP response with the test's requested probability total.
        """
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
    """Reject mass outside the normalization tolerance after bounded corrections."""
    controller, requests, replies = setup_controller

    def malformed(request):
        """Construct a complete probability map with an unacceptable total.

        Args:
            request: Typed payload supplying the expected choices.

        Returns:
            HTTP response whose probability mass violates the tolerance.
        """
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
    """Stop after a delayed failure consumes the original request deadline."""
    controller, requests, replies = setup_controller
    clock = {"now": 0.0}

    def late_reply(_):
        """Advance the mocked clock beyond the deadline before returning a failure.

        Args:
            _: Ignored outgoing request payload.

        Returns:
            Retryable server-error response received too late for another attempt.
        """
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
    """Retain failed response evidence and unknown usage while redacting credentials."""
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
    """Give Jev distinct run and journal identities while retaining the campaign guard."""
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
    """Reject Jev outside the supported level-2 FOREST condition."""
    args = _hotpot_args(controller_selection="jev", reflection_level=level)
    with pytest.raises(ValueError, match="Jev requires"):
        build_run_contract(condition, args)


def test_strategy_uses_jev_then_manifestor_and_editor_and_restores_batch_state(setup_controller, tmp_path):
    """Preserve Jev role order, accounting and journal cursors across batch restoration."""
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
    """Return a response whose chosen option disagrees with its probability map.

    Args:
        request: Typed payload supplying the executable choices.

    Returns:
        Successful HTTP response containing an invalid typed answer.
    """
    result = response(request)
    result["answers"]["edit"]["choice"] = next(
        key for key, value in result["answers"]["edit"]["probabilities"].items() if value == 0
    )
    return httpx2.Response(200, json=result)


def test_response_correction_preserves_task_and_charges_both_attempts(setup_controller, monkeypatch, tmp_path):
    """Preserve task evidence and account for the failed and corrected responses."""
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
    """Share one physical-attempt allowance across transport and response failures."""
    controller, requests, replies = setup_controller
    replies.extend([429, inconsistent_choice, 503])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    _, metadata = select(controller)
    assert len(requests) == metadata["physical_attempts"] == 4
    assert requests[0] == requests[1]
    assert requests[2] == requests[3] != requests[0]
    assert controller.total_tokens_in == 2000


def test_response_correction_cannot_extend_original_deadline(setup_controller, monkeypatch, tmp_path):
    """Stop correction attempts once the original deadline expires."""
    controller, requests, replies = setup_controller
    clock = [0.0]
    monkeypatch.setattr("gepa.strategies.jev_controller.time.monotonic", lambda: clock[0])

    def expire(request):
        """Expire the request deadline before returning an invalid answer.

        Args:
            request: Typed payload supplying the executable choices.

        Returns:
            Response whose chosen option disagrees with its probability map.
        """
        clock[0] = 31.0
        return inconsistent_choice(request)

    replies.append(expire)
    with pytest.raises(JevControllerError):
        select(controller)
    assert len(requests) == 1
    row = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[-1])
    assert not row["will_retry"]


def test_explicit_recovery_consumes_existing_attempt_and_preserves_failure(setup_controller, monkeypatch, tmp_path):
    """Resume explicit recovery without rewriting or recounting the prior failure."""
    controller, requests, replies = setup_controller
    clock = [0.0]
    monkeypatch.setattr("gepa.strategies.jev_controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)

    def expire(request):
        """Expire the request deadline before returning an invalid answer.

        Args:
            request: Typed payload supplying the executable choices.

        Returns:
            Response whose chosen option disagrees with its probability map.
        """
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
    """Count archived failures against the original physical-attempt allowance."""
    controller, requests, replies = setup_controller
    clock = [0.0]
    monkeypatch.setattr("gepa.strategies.jev_controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)

    def expire(request):
        """Expire the request deadline before returning an invalid answer.

        Args:
            request: Typed payload supplying the executable choices.

        Returns:
            Response whose chosen option disagrees with its probability map.
        """
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
    """Reject mismatched recovery evidence before making a provider request."""
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
