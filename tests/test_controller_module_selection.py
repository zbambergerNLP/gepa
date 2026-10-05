"""Exercise one joint module/section/action decision through both Controller backends."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest
from test_generation_recovery import Roles
from test_jev_controller import setup_controller as _setup_controller
from test_three_role import PROMPT, ThreeRoleLM, strategy, tool_call

from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.proposer.reflective_mutation.generation_recovery import GenerationRecoveryError
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM, ensure_reflection_run_contract
from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.component_selector import AllReflectionComponentSelector
from gepa.strategies.document_template import TEMPLATE_FAMILIES, TEMPLATES
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool
from gepa.strategies.intervention import SEMANTIC_ACTIONS, build_controller_menu
from gepa.strategies.jev_controller import JEV_MODEL, JevControllerError

CANDIDATE = {"query": PROMPT, "answer": PROMPT}
EVIDENCE = {
    name: [{"Inputs": name + " input", "Generated Outputs": name + " output", "Feedback": name + " mismatch"}]
    for name in CANDIDATE
}
EDIT = tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")


class JointRecoveryRoles(Roles):
    """Score the joint menu once and exercise the real Manifestor/Editor protocol."""

    def __call__(self, prompt):
        if prompt.startswith("Validate and manifest"):
            return super().__call__(prompt)
        self.controller_prompts.append(prompt)
        options = [line[2:].split(": ", 1)[0] for line in prompt.splitlines() if line.startswith("- ") and ": " in line]
        chosen = f"component_{b'answer'.hex()}::reexpress@Rules/REPLACE_TEXT"
        assert chosen in options
        return (
            "<response>"
            + "".join(
                f"<candidate><action>{key}</action><reasoning>Repair answer wording.</reasoning>"
                f"<probability>{int(key == chosen)}</probability></candidate>"
                for key in options
            )
            + "</response>"
        )


def joint_recovery(roles, backend, controller):
    """Configure the default recovery policy with explicit joint selection."""
    reflection = ThreeRoleReflectionLM(
        roles,
        level=2,
        rng=random.Random(0),
        controller_selection=backend,
        jev_controller=controller if backend == "jev" else None,
        base_lm_run_identity={"test": "joint-recovery"},
        editor_mode="single_call",
    )
    reflection.bind_module_selector("controller")
    return reflection


def recover_joint(reflection):
    """Use stable parent and training identities for generation and replay."""
    return reflection.reflect(
        CANDIDATE,
        EVIDENCE,
        list(CANDIDATE),
        metadata={"candidate_idx": 0, "iteration_id": "joint-recovery", "minibatch_ids": [0]},
    )[0]


@pytest.fixture
def setup_controller(tmp_path):
    """Reuse the journaled Jev client with its mocked HTTP transport."""
    yield from _setup_controller.__wrapped__(tmp_path)


class ModuleChoosingLM(ThreeRoleLM):
    """Choose the second module from a complete verbalized distribution."""

    def __call__(self, prompt):
        if not isinstance(prompt, str) or "Choose edit actions that address" not in prompt:
            return super().__call__(prompt)
        self.roles.append("controller")
        self.string_calls.append(prompt)
        options = [line[2:].split(": ", 1)[0] for line in prompt.splitlines() if line.startswith("- ") and ": " in line]
        chosen = f"component_{b'answer'.hex()}::reexpress@Rules/REPLACE_TEXT"
        assert chosen in options
        return (
            "<response>"
            + "".join(
                f"<candidate><action>{key}</action><reasoning>Repair answer wording.</reasoning>"
                f"<probability>{int(key == chosen)}</probability></candidate>"
                for key in options
            )
            + "</response>"
        )


def jev_response(request):
    """Return a complete typed distribution choosing the second module."""
    choices = request["questions"]["edit"]["criteria"]
    chosen = next(
        key for key, choice in choices.items() if choice["component"] == "answer" and "reexpress@Rules/" in key
    )
    return httpx2.Response(
        200,
        json={
            "model": JEV_MODEL,
            "usage": {"input_tokens": 1000, "output_tokens": 100},
            "answers": {
                "edit": {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == chosen) for key in choices},
                }
            },
        },
    )


@pytest.mark.parametrize("backend", ["verbalized", "jev"])
@pytest.mark.parametrize("failure", ["finish", "empty", "unchanged", "manifestor"])
def test_joint_recovery_reuses_one_distribution_after_generation_failure(backend, failure, setup_controller):
    controller, requests, replies = setup_controller
    replies.append(jev_response)

    class OnceIncompatible(JointRecoveryRoles):
        def __call__(self, prompt):
            self.incompatible = failure == "manifestor" and not self.manifestor_prompts
            return super().__call__(prompt)

    roles = OnceIncompatible([failure] if failure != "manifestor" else [])
    before = deepcopy(CANDIDATE)
    proposal = recover_joint(joint_recovery(roles, backend, controller))
    assert CANDIDATE == before
    assert len(proposal.new_texts) == 1
    name, text = next(iter(proposal.new_texts.items()))
    assert text != CANDIDATE[name]
    records = proposal.metadata["attempt_records"]
    assert records[0]["component"] == "answer"
    assert records[-1]["component"] == name
    assert len({r["action_choice"] for r in records}) == len(records)
    assert proposal.metadata["generation_error_count"] == (0 if failure == "empty" else 1)
    assert set(proposal.metadata["controller_plans"]) == {"joint"}
    assert len(requests) == int(backend == "jev")
    assert len(roles.controller_prompts) == int(backend == "verbalized")
    context = json.dumps(requests[0]["state"]) if requests else roles.controller_prompts[0]
    assert all(name + " mismatch" in context for name in CANDIDATE)
    if failure == "empty":
        assert len(records) == 1
        first, correction = roles.editor_tasks
        assert {key: correction[key] for key in first} == first
        assert "native_protocol_correction" in correction
    else:
        assert len(records) == 2
        assert records[0]["attempt_status"] == "generation_error"


@pytest.mark.parametrize("backend", ["verbalized", "jev"])
def test_joint_recovery_exhausts_both_modules_without_rescoring(backend, setup_controller):
    controller, requests, replies = setup_controller
    replies.append(jev_response)
    roles = JointRecoveryRoles(incompatible=True)
    proposal = recover_joint(joint_recovery(roles, backend, controller))
    records = proposal.metadata["attempt_records"]
    assert not proposal.new_texts and proposal.metadata["generation_exhausted"]
    assert {row["component"] for row in records} == set(CANDIDATE)
    assert len({row["action_choice"] for row in records}) == len(records)
    assert len(records) == len(proposal.metadata["controller_plans"]["joint"]["entries"])
    assert proposal.metadata["generation_error_count"] == len(records)
    assert not roles.editor_tasks
    assert len(requests) + len(roles.controller_prompts) == 1


@pytest.mark.parametrize("backend", ["verbalized", "jev"])
def test_joint_recovery_replays_failed_attempt_and_rng_without_new_controller_call(backend, setup_controller, tmp_path):
    controller, requests, replies = setup_controller
    replies.append(jev_response)
    interrupted = joint_recovery(JointRecoveryRoles(["finish"], interrupt_at=1), backend, controller)
    interrupted.recovery_planner.bind_run_dir(str(tmp_path / "recovery"))
    before = interrupted.get_state()
    with pytest.raises(KeyboardInterrupt):
        recover_joint(interrupted)
    roles = JointRecoveryRoles()
    resumed = joint_recovery(roles, backend, controller)
    resumed.recovery_planner.bind_run_dir(str(tmp_path / "recovery"))
    resumed.set_state(before)
    actual = recover_joint(resumed)
    assert not roles.controller_prompts
    assert len(requests) == int(backend == "jev")
    assert len(roles.editor_tasks) == 1
    replies.append(jev_response)
    expected_lm = joint_recovery(JointRecoveryRoles(["finish"]), backend, controller)
    expected = recover_joint(expected_lm)
    assert actual.new_texts == expected.new_texts
    actual_record = deepcopy(actual.metadata["attempt_records"][0])
    expected_record = deepcopy(expected.metadata["attempt_records"][0])
    # The uninterrupted reference makes its own physical Jev request.
    actual_record["controller_sampling"].pop("source_request_id")
    expected_record["controller_sampling"].pop("source_request_id")
    assert actual_record == expected_record
    assert resumed.rng.getstate() == expected_lm.rng.getstate()


@pytest.mark.parametrize("backend", ["verbalized", "jev"])
@pytest.mark.parametrize("child_score", [0.0, 0.25, 0.5])
def test_joint_engine_retries_generation_but_never_scored_ties_or_losses(
    backend, child_score, setup_controller, tmp_path
):
    controller, requests, replies = setup_controller
    replies.append(jev_response)
    roles = JointRecoveryRoles(["finish"])
    reflection = joint_recovery(roles, backend, controller)

    class Adapter:
        propose_new_texts = None

        def __init__(self):
            self.evaluations = []

        def evaluate(self, batch, candidate, capture_traces=False):
            self.evaluations.append(deepcopy(candidate))
            return EvaluationBatch(
                outputs=["answer"] * len(batch),
                scores=[0.25 if candidate == CANDIDATE else child_score] * len(batch),
                trajectories=[{}] * len(batch) if capture_traces else None,
            )

        def make_reflective_dataset(self, candidate, evaluation, components):
            return {name: EVIDENCE[name] for name in components}

    adapter = Adapter()
    result = optimize(
        seed_candidate=CANDIDATE,
        trainset=[1, 2, 3],
        valset=[4, 5],
        adapter=adapter,
        reflection_strategy=reflection,
        module_selector="controller",
        reflection_minibatch_size=3,
        max_metric_calls=8,
        run_dir=str(tmp_path / "run"),
        display_progress_bar=False,
        raise_on_exception=True,
    )
    assert len(adapter.evaluations) == (4 if child_score > 0.25 else 3)
    assert len(result.candidates) == (2 if child_score > 0.25 else 1)
    assert len(roles.editor_tasks) == 2
    assert len(requests) + len(roles.controller_prompts) == 1
    assert sum(adapter.evaluations[2][name] != CANDIDATE[name] for name in CANDIDATE) == 1


def test_joint_recovery_rejects_round_robin_checkpoint():
    reflection = joint_recovery(JointRecoveryRoles(), "verbalized", None)
    joint_contract = reflection.recovery_planner.get_state()
    reflection.bind_module_selector("round_robin")
    with pytest.raises(GenerationRecoveryError, match="identity mismatch"):
        reflection.recovery_planner.set_state(joint_contract)


@pytest.mark.parametrize("backend", ["verbalized", "jev"])
def test_one_joint_decision_sees_all_evidence_and_edits_only_selected_module(backend, setup_controller):
    """Let either backend override round-robin order without broadening the edit."""
    controller, requests, replies = setup_controller
    replies.append(jev_response)
    lm = ModuleChoosingLM([EDIT])
    reflection, _ = strategy(
        2,
        lm=lm,
        controller_selection=backend,
        editor_mode="single_call",
        **({"jev_controller": controller} if backend == "jev" else {}),
    )
    reflection.bind_module_selector("controller")
    before = deepcopy(CANDIDATE)
    proposal, _ = reflection.reflect(CANDIDATE, EVIDENCE, list(CANDIDATE))
    assert CANDIDATE == before
    assert set(proposal.new_texts) == {"answer"}
    assert proposal.new_texts["answer"] == PROMPT.replace("be nice", "be kind")
    assert len(proposal.metadata["attempt_records"]) == 1
    record = proposal.metadata["controller_sampling"]
    assert record["selected_component"] == "answer"
    assert record["eligible_components"] == ["query", "answer"]
    if backend == "jev":
        assert len(requests) == 1
        context = json.dumps(requests[0]["state"])
        assert lm.roles == ["manifestor", "react_v2"]
        assert proposal.metadata["controller_direction"] is None
        criteria = requests[0]["questions"]["edit"]["criteria"]
        assert {choice["component"] for choice in criteria.values()} == set(CANDIDATE)
        assert all(choice["section"] != "Task" or choice["operator"] == "INSERT_TEXT" for choice in criteria.values())
    else:
        context = lm.string_calls[0]
        assert lm.roles == ["controller", "manifestor", "react_v2"]
        assert proposal.metadata["controller_direction"] == "Repair answer wording."
    for name in CANDIDATE:
        assert name + " input" in context and name + " output" in context and name + " mismatch" in context
    editor = json.loads(lm.react_calls[0][-1]["content"])
    assert "answer mismatch" in editor["execution_traces"]
    assert "query mismatch" not in editor["execution_traces"]


def test_joint_jev_correction_preserves_all_modules_and_edits_only_selected_module(setup_controller, monkeypatch):
    """Recover an inconsistent choice without narrowing evidence or editing extra modules."""
    controller, requests, replies = setup_controller
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)

    def inconsistent_choice(request):
        payload = jev_response(request).json()
        answer = payload["answers"]["edit"]
        answer["choice"] = next(key for key in answer["probabilities"] if key != answer["choice"])
        return httpx2.Response(200, json=payload)

    replies.extend([inconsistent_choice, jev_response])
    lm = ModuleChoosingLM([EDIT])
    reflection, _ = strategy(2, lm=lm, controller_selection="jev", editor_mode="single_call", jev_controller=controller)
    reflection.bind_module_selector("controller")
    proposal, _ = reflection.reflect(CANDIDATE, EVIDENCE, list(CANDIDATE))

    assert len(requests) == 2
    assert requests[0]["state"] == requests[1]["state"]
    assert set(requests[1]["state"]["components"]) == set(CANDIDATE)
    assert requests[0]["questions"]["edit"]["criteria"] == requests[1]["questions"]["edit"]["criteria"]
    assert "argmax choice disagrees" in requests[1]["questions"]["edit"]["instructions"]
    assert "Previous response:" in requests[1]["questions"]["edit"]["instructions"]
    assert set(proposal.new_texts) == {"answer"}
    assert proposal.new_texts["answer"] == PROMPT.replace("be nice", "be kind")
    assert lm.roles == ["manifestor", "react_v2"]
    assert proposal.metadata["controller_sampling"]["physical_attempts"] == 2
    records = [json.loads(line) for line in controller._attempt_log.read_text().splitlines()]
    finished = [record for record in records if record["event"] == "finished"]
    assert [row["attempt"] for row in finished] == [1, 2]
    assert [row["outcome"] for row in finished] == ["error", "success"]
    assert finished[0]["request_id"] == finished[1]["request_id"]


def test_component_menu_ids_are_casefold_unique_and_legacy_ids_unchanged():
    """Avoid collisions even for module names containing markup and delimiters."""
    names = ["a", "A", "a::b/@<x>", "תשובה"]
    menus = [
        build_controller_menu(
            TEMPLATES["system_prompt"],
            name,
            EDIT_TOOL_SETS["broad"],
            2,
            rng=random.Random(0),
            include_component=True,
        )
        for name in names
    ]
    ids = [choice.menu_id.casefold() for menu in menus for choice in menu]
    assert len(ids) == len(set(ids)) == len(names) * len(TEMPLATES["system_prompt"].sections) * len(SEMANTIC_ACTIONS)
    legacy = build_controller_menu(TEMPLATES["system_prompt"], "a", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0))
    assert legacy[0].menu_id == "contextualize@Role/INSERT_TEXT"


def test_jev_joint_journal_replay_reuses_response_and_rejects_changed_module_evidence(setup_controller):
    """Reproduce component sampling without another API call after restore."""
    controller, requests, replies = setup_controller
    replies.append(jev_response)
    reflection, _ = strategy(2, controller_selection="jev", jev_controller=controller)
    reflection.bind_module_selector("controller")
    before = reflection.get_batch_retry_state()
    with response_journal_scope("joint:1"):
        first, record, _ = reflection._select_action(CANDIDATE, EVIDENCE, list(CANDIDATE))
    reflection.set_batch_retry_state(before)
    with response_journal_scope("joint:1"):
        replay, saved, _ = reflection._select_action(CANDIDATE, EVIDENCE, list(CANDIDATE))
    assert first == replay and record["sampled"] == saved["sampled"]
    assert saved["replayed"] and saved["physical_attempts"] == 0 and len(requests) == 1
    reflection.set_batch_retry_state(before)
    changed = deepcopy(EVIDENCE)
    changed["query"][0]["Feedback"] = "a different failure"
    with response_journal_scope("joint:1"), pytest.raises(ResponseJournalError):
        reflection._select_action(CANDIDATE, changed, list(CANDIDATE))
    assert len(requests) == 1


def test_joint_mode_cannot_resume_round_robin_contract(tmp_path):
    """Bind module choice before the saved run contract is compared."""
    reflection, _ = strategy(2)
    original = reflection.run_contract(CANDIDATE)
    ensure_reflection_run_contract(str(tmp_path), original)
    reflection.bind_module_selector("controller")
    joint = reflection.run_contract(CANDIDATE)
    assert joint["controller"]["factorization"] == "P(component, region, action)"
    with pytest.raises(ValueError, match="different reflection strategy"):
        ensure_reflection_run_contract(str(tmp_path), joint)
    reflection.bind_module_selector("round_robin")
    assert reflection.run_contract(CANDIDATE) == original


@pytest.mark.parametrize("level,backend", [(0, "verbalized"), (1, "verbalized"), (2, "uniform_random")])
def test_unsupported_controller_modes_fail_before_inference(level, backend):
    reflection, lm = strategy(level, controller_selection=backend)
    with pytest.raises(ValueError, match="requires level 2"):
        reflection.bind_module_selector("controller")
    assert lm.roles == []


@pytest.mark.parametrize("front_door", ["optimize", "launcher"])
def test_public_front_doors_supply_all_modules_without_advancing_round_robin(front_door, monkeypatch):
    """Wire both entrypoints before engine construction or metric evaluation."""
    reflection, _ = strategy(2)
    module = "gepa.api" if front_door == "optimize" else "gepa.gepa_launcher"

    class CapturedError(Exception):
        pass

    def capture(**kwargs):
        selector = kwargs["module_selector"]
        assert isinstance(selector, AllReflectionComponentSelector)
        state = SimpleNamespace(named_predictor_id_to_update_next_for_program_candidate=[0])
        assert selector(state, [], [], 0, CANDIDATE) == ["query", "answer"]
        assert state.named_predictor_id_to_update_next_for_program_candidate == [0]
        assert reflection.controller_selects_component
        raise CapturedError

    monkeypatch.setattr(module + ".ReflectiveMutationProposer", capture)
    with pytest.raises(CapturedError):
        if front_door == "optimize":
            adapter = Mock(propose_new_texts=None)
            optimize(
                seed_candidate=CANDIDATE,
                trainset=[1, 2],
                adapter=adapter,
                reflection_strategy=reflection,
                module_selector="controller",
                max_metric_calls=10,
            )
        else:
            optimize_anything(
                seed_candidate=CANDIDATE,
                evaluator=Mock(return_value=0),
                dataset=[1, 2],
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=10),
                    reflection=ReflectionConfig(
                        reflection_strategy=reflection, reflection_lm=None, module_selector="controller"
                    ),
                ),
            )


def test_jev_refuses_oversized_joint_menus_without_spending_a_request(setup_controller):
    controller, requests, _ = setup_controller
    reflection, _ = strategy(2, controller_selection="jev", jev_controller=controller)
    reflection.bind_module_selector("controller")
    template = TEMPLATES["system_prompt"]
    prompt = template.render(dict.fromkeys(template.sections, "nonempty"))
    candidate = {f"module{i}": prompt for i in range(6)}
    evidence = dict.fromkeys(candidate, EVIDENCE["query"])
    with pytest.raises(JevControllerError, match="1..255"):
        reflection.reflect(candidate, evidence, list(candidate))
    assert requests == []


def test_all_four_hotpot_modules_fit_one_jev_decision(setup_controller):
    """Keep every production component/section/action in the joint request."""
    controller, requests, replies = setup_controller
    template = TEMPLATE_FAMILIES["alibaba"]["system_prompt"]
    prompt = template.render(dict.fromkeys(template.sections, "nonempty"))
    candidate = dict.fromkeys(["query1", "summary1", "query2", "final_answer"], prompt)
    evidence = dict.fromkeys(candidate, EVIDENCE["query"])
    reflection, _ = strategy(2, controller_selection="jev", jev_controller=controller, template_family="alibaba")
    reflection.bind_module_selector("controller")

    def response(request):
        choices = request["questions"]["edit"]["criteria"]
        assert len(choices) == 200
        chosen = next(key for key, value in choices.items() if value["component"] == "summary1")
        return httpx2.Response(
            200,
            json={
                "model": JEV_MODEL,
                "usage": {"input_tokens": 2000, "output_tokens": 200},
                "answers": {
                    "edit": {
                        "type": "choice",
                        "choice": chosen,
                        "confidence": 1,
                        "probabilities": {key: float(key == chosen) for key in choices},
                    }
                },
            },
        )

    replies.append(response)
    action, _, _ = reflection._select_action(candidate, evidence, list(candidate))
    assert action.edit_target.component_name == "summary1" and len(requests) == 1


def test_joint_selection_runs_through_training_acceptance_and_validation(tmp_path):
    """Keep selection on training traces while evaluating the resulting complete candidate."""
    lm = ModuleChoosingLM([EDIT])
    reflection, _ = strategy(2, lm=lm, editor_mode="single_call")
    evaluated = []

    def evaluator(candidate, example):
        evaluated.append((dict(candidate), example))
        info = {name + "_specific_info": {**EVIDENCE[name][0], "split_marker": example} for name in candidate}
        return float("be kind" in candidate["answer"]), info

    result = optimize_anything(
        seed_candidate=CANDIDATE,
        evaluator=evaluator,
        dataset=["training-only"],
        valset=["validation-only"],
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=4, run_dir=str(tmp_path), raise_on_exception=True, parallel=False),
            reflection=ReflectionConfig(
                reflection_lm=None,
                reflection_strategy=reflection,
                module_selector="controller",
                reflection_minibatch_size=1,
            ),
        ),
    )
    assert result.total_metric_calls == 4
    assert len(result.candidates) == 2
    assert result.candidates[1]["query"] == CANDIDATE["query"]
    assert "be kind" in result.candidates[1]["answer"]
    assert result.val_aggregate_scores == [0.0, 1.0]
    assert lm.roles == ["controller", "manifestor", "react_v2"]
    assert "training-only" in lm.string_calls[0] and "validation-only" not in lm.string_calls[0]
    assert [example for _, example in evaluated] == [
        "validation-only",
        "training-only",
        "training-only",
        "validation-only",
    ]


def test_failed_joint_distribution_drops_proposal_without_editing_another_module():
    """Keep the existing bounded distribution failure behavior across the full menu."""

    class BrokenControllerLM(ThreeRoleLM):
        def __call__(self, prompt):
            self.string_calls.append(prompt)
            return "<response>not a distribution</response>"

    lm = BrokenControllerLM([])
    reflection, _ = strategy(2, lm=lm, editor_mode="single_call")
    reflection.bind_module_selector("controller")
    proposal, _ = reflection.reflect(CANDIDATE, EVIDENCE, list(CANDIDATE))
    assert proposal.new_texts == {} and not lm.react_calls
    assert len(lm.string_calls) == 2
    assert {failure["component"] for failure in proposal.metadata["controller_failures"]} == set(CANDIDATE)


@pytest.mark.parametrize("front_door", ["optimize", "launcher"])
def test_plain_reflection_cannot_silently_treat_controller_mode_as_edit_all(front_door):
    """Reject a strategy without joint selection before evaluating any examples."""
    if front_door == "optimize":
        adapter = Mock(propose_new_texts=None)
        with pytest.raises(ValueError, match="requires a supporting strategy"):
            optimize(
                seed_candidate=CANDIDATE,
                trainset=[1],
                adapter=adapter,
                reflection_lm=lambda _: "unused",
                module_selector="controller",
                max_metric_calls=4,
            )
        adapter.evaluate.assert_not_called()
    else:
        evaluator = Mock(return_value=0.0)
        with pytest.raises(ValueError, match="requires a supporting strategy"):
            optimize_anything(
                seed_candidate=CANDIDATE,
                evaluator=evaluator,
                dataset=[1],
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=4),
                    reflection=ReflectionConfig(reflection_lm=lambda _: "unused", module_selector="controller"),
                ),
            )
        evaluator.assert_not_called()
