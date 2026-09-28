"""Exercise real-edit recovery, node-scoped diversity and deterministic interruption replay."""

import json
import random
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_response_journal import completion_response, journal_lm

import gepa
from gepa.core.action_tracking import ActionDiversityCallback
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState
from gepa.lm import NativeToolCall, ToolCompletion
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.manifestor import ManifestationError, Manifestor
from gepa.proposer.reflective_mutation.sibling_policy import SIBLING_POLICY_ID, SiblingInvariantError
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.document_template import DocumentTemplate, EditTarget, MalformedDocumentError
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool
from gepa.strategies.intervention import (
    Controller,
    ControllerChoice,
    SemanticActionSpec,
    build_controller_menu,
    canonical_action_constraints,
)
from gepa.strategies.proposal_sampling import SameParentSampling

TEMPLATE = DocumentTemplate("prompt", {"Style": "Presentation", "Objective": "Task purpose"})
SEED = {"system_prompt": TEMPLATE.render({"Style": "Brief.", "Objective": "Answer."})}
DATA = {"system_prompt": [{"Inputs": {"question": "training-only"}, "Feedback": "Missing background."}]}
PAIR = "contextualize@Style/INSERT_TEXT"


class Roles:
    def __init__(self, errors=(), *, all_zero=False, incompatible=False, interrupt_at=None):
        self.errors = list(errors)
        self.all_zero = all_zero
        self.incompatible = incompatible
        self.interrupt_at = interrupt_at
        self.controller_prompts = []
        self.manifestor_prompts = []
        self.editor_tasks = []
        self.editor_systems = []

    def __call__(self, prompt):
        if prompt.startswith("Validate and manifest"):
            self.manifestor_prompts.append(prompt)
            if self.incompatible:
                return json.dumps({"status": "incompatible", "reason": "Direction adds an operative rule as context."})
            return json.dumps(
                {
                    "status": "ready",
                    "observation": "Missing background.",
                    "hypothesis": "Needs context.",
                    "general_change": "Add relevant non-operative background.",
                    "scope": "Keep output rules.",
                }
            )
        self.controller_prompts.append(prompt)
        pairs = re.findall(r"^- ([^: ]+):", prompt, re.M)
        return (
            "<response>"
            + "".join(
                f"<candidate><action>{pair}</action><reasoning>Apply {pair} within its constraints using this evidence."
                f"</reasoning><probability>{int(pair == PAIR and not self.all_zero)}</probability></candidate>"
                for pair in pairs
            )
            + "</response>"
        )

    def complete_with_tools(self, messages, tools, **kwargs):
        if self.interrupt_at is not None and len(self.editor_tasks) == self.interrupt_at:
            raise KeyboardInterrupt("Simulated allocation interruption")
        task = json.loads(messages[-1]["content"])
        self.editor_tasks.append(task)
        self.editor_systems.append(messages[0]["content"])
        body = task["section_body"]
        operator = re.search(r"Every call must use (\w+)", messages[0]["content"])[1]
        error = self.errors.pop(0) if self.errors else None
        if error == "finish":
            return ToolCompletion("<finish>No edit.</finish>", ())
        if error == "empty":
            return ToolCompletion("", ())
        if operator == "INSERT_TEXT":
            args = {"anchor": "", "text": " Background.", "where": "after"}
        elif operator == "DELETE_TEXT":
            args = {"target": body[0]}
        elif operator == "REPLACE_TEXT":
            args = {"target": body, "text": body + " revised."}
        else:
            args = {"target": body[0], "anchor": "", "where": "after"}
        if error == "invalid":
            args = {"target": "absent", "text": "changed"}
        if error == "whitespace":
            args = {"anchor": "", "text": " ", "where": "after"}
        if error == "unchanged":
            args = {"anchor": "", "text": "", "where": "after"}
        return ToolCompletion("", (NativeToolCall("edit", operator, json.dumps(args)),))


def strategy(roles=None, *, seed=0):
    roles = roles or Roles()
    return ThreeRoleReflectionLM(
        roles,
        level=2,
        rng=random.Random(seed),
        templates={"system_prompt": TEMPLATE},
        base_lm_run_identity={"test": "roles"},
        editor_mode="single_call",
    )


def reflect(lm, parent=0, iteration=1, candidate=None, data=None, components=None):
    return lm.reflect(
        candidate or SEED,
        data or DATA,
        components or ["system_prompt"],
        metadata={
            "candidate_idx": parent,
            "iteration_id": str(iteration),
            "minibatch_ids": [0, 1, 2],
        },
    )[0]


def as_child(proposal, parent=0, candidate=None):
    return CandidateProposal({**(candidate or SEED), **proposal.new_texts}, [parent], metadata=proposal.metadata)


def accept(lm, proposal, candidates, parent=0):
    child = as_child(proposal, parent, candidates[parent])
    state = SimpleNamespace(program_candidates=candidates)
    lm.sibling_planner.validate_children([child], state)
    candidates.append(child.candidate)
    lm.sibling_planner.accept_child(child, len(candidates) - 1, state)
    return child


def test_duplicate_siblings_blocked_and_different_sections_allowed():
    lm = strategy()
    candidates = [deepcopy(SEED)]
    first = reflect(lm)
    accept(lm, first, candidates)
    assert lm.sibling_planner.accepted[1] == {}
    second = reflect(lm, iteration=2)
    assert second.metadata["sibling_choice"]["pair"] != PAIR
    assert PAIR not in lm.base_lm.controller_prompts[-1].split("## Available actions")[1].split("Score ")[0]
    assert any(row["reason"] == "accepted_sibling" for row in second.metadata["planner_exclusions"])
    with pytest.raises(SiblingInvariantError, match="Duplicate"):
        lm.sibling_planner.validate_children([as_child(first)], SimpleNamespace(program_candidates=candidates))
    other = deepcopy(first)
    other.new_texts = {"system_prompt": TEMPLATE.replace_section_body(SEED["system_prompt"], "Objective", "Another.")}
    other.metadata["sibling_choice"].update(pair="contextualize@Objective/INSERT_TEXT", section="Objective")
    lm.sibling_planner.validate_children([as_child(other)], SimpleNamespace(program_candidates=candidates))


def test_vertical_repeats_across_generations_and_identical_text_nodes():
    lm = strategy()
    candidates = [deepcopy(SEED)]
    for parent in range(3):
        proposal = reflect(lm, parent, parent + 1, candidates[parent])
        assert proposal.metadata["sibling_choice"]["pair"] == PAIR
        accept(lm, proposal, candidates, parent)
    # Equal text does not merge the identities or outgoing choices.
    candidates.append(deepcopy(SEED))
    identical = reflect(lm, 4, 5, candidates[4])
    assert identical.metadata["sibling_choice"]["pair"] == PAIR
    assert lm.sibling_planner.accepted[3] == {}


def test_failed_ancestor_attempt_does_not_block_descendant():
    lm = strategy(Roles(["finish"]))
    candidates = [deepcopy(SEED)]
    parent_proposal = reflect(lm)
    assert parent_proposal.metadata["attempt_records"][0]["action_choice"] == PAIR
    assert parent_proposal.metadata["attempt_records"][0]["attempt_status"] == "generation_error"
    assert parent_proposal.metadata["sibling_choice"]["pair"] != PAIR
    accept(lm, parent_proposal, candidates)
    descendant = reflect(lm, 1, 2, candidates[1])
    assert descendant.metadata["sibling_choice"]["pair"] == PAIR
    accept(lm, descendant, candidates, 1)


@pytest.mark.parametrize("failure", ["finish", "empty", "invalid", "unchanged", "whitespace"])
def test_editor_generation_errors_replan_without_unchanged_candidate(failure):
    roles = Roles([failure])
    lm = strategy(roles)
    proposal = reflect(lm)
    assert proposal.new_texts and proposal.new_texts["system_prompt"] != SEED["system_prompt"]
    assert proposal.metadata["generation_error_count"] >= 1
    records = proposal.metadata["attempt_records"]
    assert len({(r["component"], r["action_choice"]) for r in records}) == len(records)
    assert records[0]["action_choice"] == PAIR
    assert records[1]["controller_sampling"]["phase"] == "zero_weight_fallback"
    assert all(task["execution_traces"] == roles.editor_tasks[0]["execution_traces"] for task in roles.editor_tasks)
    assert lm.sibling_planner.accepted == {}


def test_direct_reflection_rejects_noncanonical_parent_before_model_calls():
    parent = {"system_prompt": SEED["system_prompt"] + "\n\n"}
    roles = Roles(["whitespace"])
    lm = strategy(roles)
    with pytest.raises(MalformedDocumentError, match="canonical"):
        reflect(lm, candidate=parent)
    assert not roles.controller_prompts and not roles.editor_tasks
    assert not lm.sibling_planner.accepted


@pytest.mark.parametrize("editor_mode", ["react", "single_call"])
def test_sibling_policy_rejects_incompatible_edit_basis_before_model_calls(editor_mode):
    roles = Roles()
    with pytest.raises(ValueError, match="broad direct-tool basis"):
        ThreeRoleReflectionLM(roles, level=2, edit_tool_set="minimal", editor_mode=editor_mode)
    assert not roles.controller_prompts and not roles.editor_tasks


def test_required_edit_manifestor_checks_fixed_guidance_instead_of_bypassing_validation():
    roles = Roles(incompatible=True)
    spec = SemanticActionSpec("fixed_context", "Add context.", EditTool.INSERT_TEXT, fixed_text="Keep unchanged.")
    action = ControllerChoice(EditTarget("system_prompt", "Style"), spec)
    manifestor = Manifestor(roles)
    assert manifestor.manifest(action, "Brief.", "Feedback", "Traces") == "Keep unchanged."
    with pytest.raises(ManifestationError, match="incompatible"):
        manifestor.manifest(action, "Brief.", "Feedback", "Traces", require_edit=True)
    assert len(roles.manifestor_prompts) == 1
    assert spec.fixed_text in roles.manifestor_prompts[0]


def test_full_constraints_identical_in_all_roles():
    roles = Roles()
    reflect(strategy(roles))
    for prompt in [*roles.controller_prompts, *roles.manifestor_prompts, *roles.editor_systems]:
        assert canonical_action_constraints() in prompt
    assert "Legitimate no-ops" not in roles.editor_systems[0]
    assert "preserve a legitimate no-op" not in roles.editor_systems[0]


def test_manifestor_incompatibility_exhausts_pairs_without_editor_calls():
    roles = Roles(incompatible=True)
    lm = strategy(roles)
    proposal = reflect(lm)
    assert proposal.metadata["generation_exhausted"] and not proposal.new_texts
    records = proposal.metadata["attempt_records"]
    assert len(records) == 20
    assert len({r["action_choice"] for r in records}) == 20
    assert proposal.metadata["generation_error_count"] == 20
    assert not roles.editor_tasks
    assert all("operative rule" in r["manifestor_error"] for r in records)
    assert lm.sibling_planner.accepted == {}
    # New evidence opportunity releases temporary failures even on the same node.
    roles.incompatible = False
    retry = reflect(lm, iteration=2)
    assert retry.metadata["sibling_choice"]["pair"] == PAIR


def test_all_zero_distribution_is_explored_and_empty_sections_recorded():
    roles = Roles(all_zero=True)
    lm = strategy(roles)
    proposal = reflect(lm, candidate={"system_prompt": ""})
    assert proposal.new_texts
    assert proposal.metadata["attempt_records"][0]["controller_sampling"]["phase"] == "zero_weight_fallback"
    excluded = proposal.metadata["planner_exclusions"]
    assert len(excluded) == 18
    assert all(row["reason"] == "nonempty_target_required" for row in excluded)


def test_batch_reservations_and_acceptance_invariant():
    lm = strategy()
    jobs = [(SEED, DATA, ["system_prompt"])] * 3
    contexts = [{"candidate_idx": 0, "iteration_id": "1", "minibatch_ids": [0, 1, 2]}] * 3
    proposals = [p for p, _ in lm.reflect_many(jobs, metadatas=contexts)]
    pairs = [p.metadata["sibling_choice"]["pair"] for p in proposals]
    assert len(set(pairs)) == 3
    assert not lm.sibling_planner.accepted and not lm.sibling_planner.reserved
    state = SimpleNamespace(program_candidates=[SEED])
    lm.sibling_planner.validate_children([as_child(p) for p in proposals], state)
    duplicate = as_child(proposals[0])
    duplicate.candidate["system_prompt"] = duplicate.candidate["system_prompt"].replace("Background.", "Different.")
    with pytest.raises(SiblingInvariantError, match="Duplicate"):
        lm.sibling_planner.validate_children([as_child(proposals[0]), duplicate], state)
    # Reservations and failed attempts never become permanent after rejection.
    assert reflect(lm, iteration=2).metadata["sibling_choice"]["pair"] == PAIR


def test_checkpoint_restores_accepted_choices_and_rng():
    original = strategy()
    candidates = [deepcopy(SEED)]
    first = reflect(original)
    accept(original, first, candidates)
    snapshot = original.get_state()
    restored = strategy(seed=99)
    restored.set_state(snapshot)
    expected = reflect(original, iteration=2)
    actual = reflect(restored, iteration=2)
    assert expected.metadata["sibling_choice"] == actual.metadata["sibling_choice"]
    assert original.rng.getstate() == restored.rng.getstate()
    assert restored.sibling_planner.accepted == original.sibling_planner.accepted
    assert restored.sibling_planner.accepted[1] == {}


def test_durable_recovery_replays_failed_steps_without_new_calls(tmp_path):
    roles = Roles(["finish"], interrupt_at=1)
    interrupted = strategy(roles)
    interrupted.sibling_planner.bind_run_dir(str(tmp_path))
    before = interrupted.get_state()
    with pytest.raises(KeyboardInterrupt):
        reflect(interrupted)
    assert len(roles.editor_tasks) == 1
    restored_roles = Roles()
    restored = strategy(restored_roles)
    restored.sibling_planner.bind_run_dir(str(tmp_path))
    restored.set_state(before)
    resumed = reflect(restored)
    assert not restored_roles.controller_prompts  # Scored menu and first failure replay from disk.
    assert len(restored_roles.editor_tasks) == 1
    assert resumed.metadata["generation_error_count"] == 1
    reference = strategy(Roles(["finish"]))
    expected = reflect(reference)
    assert resumed.metadata["sibling_choice"] == expected.metadata["sibling_choice"]
    assert restored.rng.getstate() == reference.rng.getstate()
    assert resumed.metadata["attempt_records"][0] == expected.metadata["attempt_records"][0]


def test_exhaustion_defers_parent_but_not_its_descendant():
    lm = strategy(Roles(incompatible=True))
    reflect(lm)
    state = SimpleNamespace(get_pareto_front_mapping=lambda: {0: {0, 1}}, per_program_tracked_scores=[0.5, 0.5])
    assert lm.sibling_planner.eligible_parent(state, 0) == 1
    assert lm.sibling_planner.eligible_parent(state, 0) == 0


def test_policy_identity_rejects_old_checkpoint():
    lm = strategy()
    assert lm.run_contract(SEED)["controller"]["identity"] == SIBLING_POLICY_ID
    with pytest.raises(SiblingInvariantError):
        lm.set_state({"rng_state": random.Random(0).getstate()})


@pytest.mark.parametrize("version", [None, -1])
def test_policy_rejects_missing_or_mismatched_checkpoint_version(version):
    lm = strategy()
    snapshot = lm.get_state()
    snapshot["sibling_policy"]["contract_version"] = version
    with pytest.raises(SiblingInvariantError, match="identity mismatch"):
        strategy().set_state(snapshot)


def test_reporting_separates_siblings_vertical_repeats_and_errors():
    tracker = ActionDiversityCallback()
    for parent, child, section in [(0, 1, "Style"), (0, 2, "Objective"), (1, 3, "Style")]:
        tracker.on_candidate_accepted(
            {
                "iteration": child,
                "new_candidate_idx": child,
                "old_score": 0,
                "new_score": 1,
                "parent_ids": [parent],
                "metadata": {
                    "action": "contextualize",
                    "sibling_choice": {
                        "parent": parent,
                        "component": "sys",
                        "pair": f"contextualize@{section}",
                        "section": section,
                    },
                },
            }
        )
    tracker.on_evaluation_skipped({"reason": "all_scores_perfect"})
    summary = tracker.summary()
    assert summary["sibling_diversity"]["siblings_by_parent_component"]["0/sys"]["distinct_pairs"] == 2
    assert len(summary["sibling_diversity"]["vertical_repeats"]) == 1
    restored = ActionDiversityCallback()
    restored.set_state(tracker.get_state())
    assert restored.summary() == summary


class Adapter:
    propose_new_texts = None

    def __init__(self, child_score=0.5):
        self.child_score = child_score
        self.evaluations = []

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluations.append(deepcopy(candidate))
        score = 0.25 if candidate == SEED else self.child_score
        return EvaluationBatch(
            outputs=["answer"] * len(batch),
            scores=[score] * len(batch),
            trajectories=[{"question": ex, "feedback": "missing context"} for ex in batch] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, evaluation, components):
        return {name: [{"Feedback": trace} for trace in evaluation.trajectories] for name in components}


@pytest.mark.parametrize("child_score", [0.0, 0.25, 0.5])
def test_engine_evaluates_changed_proposal_once_and_consumes_only_acceptance(tmp_path, child_score):
    adapter = Adapter(child_score)
    lm = strategy(Roles(["finish"]))
    tracker = ActionDiversityCallback()
    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=[1, 2, 3],
        valset=[4, 5],
        adapter=adapter,
        reflection_strategy=lm,
        max_metric_calls=8,
        reflection_minibatch_size=3,
        callbacks=[tracker],
        run_dir=str(tmp_path),
        display_progress_bar=False,
        raise_on_exception=True,
    )
    assert len(adapter.evaluations) == (4 if child_score > 0.25 else 3)
    assert len(result.candidates) == (2 if child_score > 0.25 else 1)
    assert bool(lm.sibling_planner.accepted) == (child_score > 0.25)
    assert tracker.proposal_outcomes["generation_errors"] >= 1
    if child_score <= 0.25:
        assert tracker.proposal_outcomes["evaluated_tie" if child_score == 0.25 else "evaluated_loss"] == 1


def test_different_components_keep_independent_sibling_choices():
    lm = strategy()
    candidate = {"system_prompt": SEED["system_prompt"], "other": SEED["system_prompt"]}
    data = {"system_prompt": DATA["system_prompt"], "other": DATA["system_prompt"]}
    first = reflect(lm, candidate=candidate, data=data)
    candidates = [candidate]
    accept(lm, first, candidates)
    second = reflect(lm, iteration=2, candidate=candidate, data=data, components=["other"])
    assert second.metadata["sibling_choice"]["pair"] == PAIR
    assert second.metadata["sibling_choice"]["component"] == "other"
    accept(lm, second, candidates)
    # This child changed only 'other'; its identical system_prompt still has an empty outgoing ledger.
    descendant = reflect(lm, parent=2, iteration=3, candidate=candidates[2], data=data)
    assert descendant.metadata["sibling_choice"]["pair"] == PAIR
    assert descendant.metadata["sibling_choice"]["component"] == "system_prompt"


def test_component_fallback_keeps_same_trace_and_prioritizes_positive_choices():
    roles = Roles(["finish"])
    lm = strategy(roles)
    candidate = {"system_prompt": SEED["system_prompt"], "other": SEED["system_prompt"]}
    data = {"system_prompt": DATA["system_prompt"], "other": DATA["system_prompt"]}
    proposal = reflect(lm, candidate=candidate, data=data)
    assert proposal.metadata["sibling_choice"]["component"] == "other"
    assert proposal.metadata["sibling_choice"]["pair"] == PAIR
    assert len(roles.editor_tasks) == 2
    assert roles.editor_tasks[0]["execution_traces"] == roles.editor_tasks[1]["execution_traces"]


def test_engine_perfect_batch_still_skips_all_role_calls(tmp_path):
    roles = Roles()
    lm = strategy(roles)
    adapter = Adapter()
    gepa.optimize(
        seed_candidate=SEED,
        trainset=[1, 2, 3],
        valset=[4, 5],
        adapter=adapter,
        reflection_strategy=lm,
        perfect_score=0.25,
        skip_perfect_score=True,
        max_metric_calls=5,
        reflection_minibatch_size=3,
        run_dir=str(tmp_path),
        display_progress_bar=False,
        raise_on_exception=True,
    )
    assert not roles.controller_prompts and not roles.editor_tasks
    assert not lm.sibling_planner.accepted


def test_engine_resumes_mid_recovery_without_repeating_failed_generation(tmp_path):
    def run(path, lm, adapter, callback):
        return gepa.optimize(
            seed_candidate=SEED,
            trainset=[1, 2, 3],
            valset=[4, 5],
            adapter=adapter,
            reflection_strategy=lm,
            max_metric_calls=8,
            reflection_minibatch_size=3,
            callbacks=[callback],
            run_dir=str(path),
            display_progress_bar=False,
            raise_on_exception=True,
        )

    interrupted_roles = Roles(["finish"], interrupt_at=1)
    interrupted_lm = strategy(interrupted_roles)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "resume", interrupted_lm, Adapter(), ActionDiversityCallback())
    restored_roles = Roles()
    restored_lm = strategy(restored_roles)
    restored_tracker = ActionDiversityCallback()
    resumed = run(tmp_path / "resume", restored_lm, Adapter(), restored_tracker)
    reference_lm = strategy(Roles(["finish"]))
    reference_tracker = ActionDiversityCallback()
    reference = run(tmp_path / "reference", reference_lm, Adapter(), reference_tracker)
    assert resumed.candidates == reference.candidates
    assert resumed.total_metric_calls == reference.total_metric_calls
    assert restored_lm.sibling_planner.accepted == reference_lm.sibling_planner.accepted
    assert restored_lm.rng.getstate() == reference_lm.rng.getstate()
    assert not restored_roles.controller_prompts
    assert len(interrupted_roles.editor_tasks) + len(restored_roles.editor_tasks) == 2
    assert restored_tracker.proposal_outcomes == reference_tracker.proposal_outcomes


def test_engine_resumes_partially_generated_sibling_batch(tmp_path):
    def run(path, lm):
        return gepa.optimize(
            seed_candidate=SEED,
            trainset=[1, 2, 3],
            valset=[4, 5],
            adapter=Adapter(),
            reflection_strategy=lm,
            max_metric_calls=14,
            reflection_minibatch_size=3,
            sampling_strategy=SameParentSampling(n=3),
            run_dir=str(path),
            display_progress_bar=False,
            raise_on_exception=True,
        )

    interrupted_roles = Roles(interrupt_at=1)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "resume", strategy(interrupted_roles))
    restored_roles = Roles()
    restored_lm = strategy(restored_roles)
    resumed = run(tmp_path / "resume", restored_lm)
    reference_lm = strategy()
    reference = run(tmp_path / "reference", reference_lm)
    assert resumed.candidates == reference.candidates
    assert restored_lm.sibling_planner.accepted == reference_lm.sibling_planner.accepted
    assert restored_lm.rng.getstate() == reference_lm.rng.getstate()
    assert len(interrupted_roles.editor_tasks) + len(restored_roles.editor_tasks) == 3


def test_invalid_direction_never_reaches_editor_and_raw_error_is_preserved():
    lm = strategy(Roles(incompatible=True))
    proposal = reflect(lm)
    assert all(row["attempt_status"] == "generation_error" for row in proposal.metadata["attempt_records"])
    assert not proposal.new_texts
    assert all("incompatible" in row["manifestor_error"] for row in proposal.metadata["attempt_records"])


def test_generation_exhaustion_is_not_recorded_as_an_accepted_edit(tmp_path):
    roles = Roles(incompatible=True)
    lm = strategy(roles)
    tracker = ActionDiversityCallback()
    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=[1, 2, 3],
        valset=[4, 5],
        adapter=Adapter(),
        reflection_strategy=lm,
        max_metric_calls=5,
        reflection_minibatch_size=3,
        callbacks=[tracker],
        run_dir=str(tmp_path),
        display_progress_bar=False,
        raise_on_exception=True,
    )
    assert len(result.candidates) == 1
    assert tracker.proposal_outcomes["generation_exhausted"] == 1
    assert tracker.proposal_outcomes["generation_errors"] == 20
    state = GEPAState.load(str(tmp_path))
    history = state.revision_history_by_candidate[0]
    assert any("Generation error" in message["content"] for message in history)
    assert not any("Optimizer result: accepted" in message["content"] for message in history)


def test_recovery_failures_never_inherit_the_successful_edits_score():
    records = [
        {"attempt_status": "generation_error", "dropped_reason": "invalid batch", "component": "sys"},
        {"attempt_status": "completed", "component": "sys"},
    ]
    messages = GEPAState._attempt_records_to_chat(records, "accepted", score_before=0, score_after=1)
    assert "Generation error" in messages[0]["content"]
    assert "Score after" not in messages[0]["content"]
    assert "accepted" in messages[1]["content"] and "Score after: 1" in messages[1]["content"]


@pytest.mark.parametrize("raise_on_exception", [True, False])
def test_engine_guard_rejects_duplicate_siblings_before_full_validation(tmp_path, monkeypatch, raise_on_exception):
    lm = strategy()
    first = reflect(lm)
    second = deepcopy(first)
    second.new_texts["system_prompt"] = second.new_texts["system_prompt"].replace(
        "Background.", "Different background."
    )
    responses = iter([first, second])
    monkeypatch.setattr(lm.sibling_planner, "generate", lambda *args: next(responses))
    adapter = Adapter()
    with pytest.raises(SiblingInvariantError, match="Duplicate"):
        gepa.optimize(
            seed_candidate=SEED,
            trainset=[1, 2, 3],
            valset=[4, 5],
            adapter=adapter,
            reflection_strategy=lm,
            max_metric_calls=14,
            reflection_minibatch_size=3,
            sampling_strategy=SameParentSampling(n=2),
            run_dir=str(tmp_path),
            display_progress_bar=False,
            raise_on_exception=raise_on_exception,
        )
    assert not lm.sibling_planner.accepted
    assert len(GEPAState.load(str(tmp_path)).program_candidates) == 1


def test_all_zero_controller_select_is_not_a_malformed_distribution():
    controller = Controller(
        build_controller_menu(TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0)),
        Roles(all_zero=True),
        k=20,
        require_full_support=True,
        allow_zero_weights=True,
    )
    result = controller.select(1, candidate=SEED["system_prompt"], feedback_summary="Improve this.")
    assert len(result) == 1
    assert controller.history[-1]["sampling_policy"] == "zero_weight_uniform"
    assert not controller.history[-1]["fallback"]


def test_controller_raw_outputs_only_keep_the_current_scoring_attempts():
    controller = Controller(
        build_controller_menu(TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0)),
        Roles(),
        k=20,
        require_full_support=True,
    )
    for _ in range(3):
        controller.select(1, candidate=SEED["system_prompt"], feedback_summary="Improve this.")
        assert len(controller.raw_outputs) == 1
    assert len(controller.history) == 3


def test_role_journal_replays_completed_manifestor_inside_interrupted_edit_step(tmp_path, monkeypatch):
    """Use the actual LM response journal, with a mocked provider and real role boundaries."""
    original_roles = Roles(["finish"], interrupt_at=1)

    def provider_for(roles):
        def provider(**kwargs):
            if kwargs.get("tools"):
                result = roles.complete_with_tools(kwargs["messages"], kwargs["tools"])
                response = completion_response(result.content)
                response.choices[0].message.tool_calls = [
                    SimpleNamespace(
                        id=call.id, type="function", function=SimpleNamespace(name=call.name, arguments=call.arguments)
                    )
                    for call in result.tool_calls
                ]
            else:
                response = completion_response(roles(kwargs["messages"][-1]["content"]))
            return response

        return provider

    journal_path = tmp_path / "role-responses.sqlite3"
    interrupted = strategy(journal_lm(journal_path))
    interrupted.sibling_planner.bind_run_dir(str(tmp_path))
    before = interrupted.get_state()
    monkeypatch.setattr("litellm.completion", provider_for(original_roles))
    monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: 0.25)
    with pytest.raises(KeyboardInterrupt):
        reflect(interrupted)
    assert len(original_roles.manifestor_prompts) == 2
    resumed_roles = Roles()
    monkeypatch.setattr("litellm.completion", provider_for(resumed_roles))
    resumed = strategy(journal_lm(journal_path))
    resumed.sibling_planner.bind_run_dir(str(tmp_path))
    resumed.set_state(before)
    proposal = reflect(resumed)
    assert proposal.new_texts and proposal.metadata["generation_error_count"] == 1
    assert not resumed_roles.controller_prompts and not resumed_roles.manifestor_prompts
    assert len(resumed_roles.editor_tasks) == 1
    assert resumed.total_cost == 1.25  # Five completed responses, no repeated completed work.
