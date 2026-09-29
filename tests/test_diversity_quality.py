"""Verify combined planning attributes only evaluated edits and resumes exactly."""

import json
import random
from types import SimpleNamespace

import pytest
from test_sibling_diversity import DATA, PAIR, SEED, TEMPLATE, Adapter, Roles, reflect

import gepa
import gepa.strategies.edit_novelty as edit_novelty
from gepa.core.action_tracking import ActionDiversityCallback
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.sibling_policy import SiblingInvariantError
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM


def quality_strategy(roles=None, *, novelty_backend="generative"):
    """Construct the combined policy with the existing deterministic role fixture."""
    return ThreeRoleReflectionLM(
        roles or Roles(),
        level=2,
        rng=random.Random(0),
        templates={"system_prompt": TEMPLATE},
        base_lm_run_identity={"test": "roles"},
        editor_mode="single_call",
        proposal_policy="diversity_quality",
        novelty_backend=novelty_backend,
    )


def run_once(path, strategy, adapter=None, callback=None):
    """Run one scored proposal through the native engine and persistence hooks."""
    return gepa.optimize(
        seed_candidate=SEED,
        trainset=[1, 2, 3],
        valset=[4, 5],
        adapter=adapter or Adapter(),
        reflection_strategy=strategy,
        max_metric_calls=8,
        reflection_minibatch_size=3,
        callbacks=[callback or ActionDiversityCallback()],
        run_dir=str(path),
        display_progress_bar=False,
        raise_on_exception=True,
    )


def test_generation_errors_never_receive_the_successful_edits_training_credit():
    strategy = quality_strategy(Roles(["finish"]))
    proposal = reflect(strategy)
    attempts = proposal.metadata["attempt_records"]
    failed_ids = {attempt["attempt_id"] for attempt in attempts if attempt["attempt_status"] == "generation_error"}
    assert failed_ids
    assert strategy.sibling_planner.memory.get_state()["records"] == []
    child = CandidateProposal(
        candidate={**SEED, **proposal.new_texts},
        parent_program_ids=[0],
        subsample_indices=[0, 1, 2],
        subsample_scores_before=[0, 0, 0],
        subsample_scores_after=[1, 1, 1],
        metadata=proposal.metadata,
    )
    strategy.sibling_planner.observe_evaluation(child, SEED)
    [record] = strategy.sibling_planner.memory.get_state()["records"]
    assert record["attempt_id"] == proposal.metadata["evaluated_attempt_id"]
    assert record["attempt_id"] not in failed_ids
    assert record["outcome"] == "improvement"
    assert record["training_gain"] == 1
    assert record["action_pair"] == proposal.metadata["sibling_choice"]["pair"]
    strategy.sibling_planner.observe_evaluation(child, SEED)
    assert len(strategy.sibling_planner.memory.get_state()["records"]) == 1


@pytest.mark.parametrize("score,outcome,gain", [(0, "regression", -0.25), (0.25, "tie", 0), (0.5, "improvement", 0.25)])
def test_native_proposer_records_training_outcomes_before_acceptance(tmp_path, score, outcome, gain):
    strategy = quality_strategy()
    result = run_once(tmp_path, strategy, Adapter(score))
    [record] = strategy.sibling_planner.memory.get_state()["records"]
    assert record["outcome"] == outcome
    assert record["training_gain"] == gain
    assert record["parent_id"] == 0
    assert record["before"] == SEED["system_prompt"]
    assert len(record["minibatch_ids"]) == 3
    assert record["scores_before"] == [0.25] * 3
    assert record["scores_after"] == [score] * 3
    assert len(result.candidates) == (2 if score > 0.25 else 1)


def test_full_validation_cannot_replace_observed_training_gain(tmp_path):
    class DifferentValidationAdapter(Adapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            evaluation = super().evaluate(batch, candidate, capture_traces)
            if len(batch) == 2:
                evaluation.scores = [0.9 if candidate == SEED else 0.1] * len(batch)
            return evaluation

    strategy = quality_strategy()
    run_once(tmp_path, strategy, DifferentValidationAdapter(0.5))
    [record] = strategy.sibling_planner.memory.get_state()["records"]
    assert record["outcome"] == "improvement"
    assert record["training_gain"] == 0.25


def test_combined_policy_restores_history_rng_and_accepted_edges(tmp_path):
    strategy = quality_strategy()
    run_once(tmp_path, strategy)
    snapshot = strategy.get_state()
    restored = quality_strategy()
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    assert restored.rng.getstate() == strategy.rng.getstate()
    [record] = restored.sibling_planner.memory.get_state()["records"]
    assert restored.sibling_planner.memory.context(1, "system_prompt", record["before"])["recent_attempts"] == []
    assert restored.sibling_planner.accepted[1] == {}


def test_interrupted_combined_recovery_has_identical_history_and_rng(tmp_path):
    interrupted_roles = Roles(["finish"], interrupt_at=1)
    with pytest.raises(KeyboardInterrupt):
        run_once(tmp_path / "resume", quality_strategy(interrupted_roles))
    resumed_roles = Roles()
    resumed_strategy = quality_strategy(resumed_roles)
    resumed = run_once(tmp_path / "resume", resumed_strategy)
    reference_strategy = quality_strategy(Roles(["finish"]))
    reference = run_once(tmp_path / "reference", reference_strategy)
    assert resumed.candidates == reference.candidates
    assert resumed.total_metric_calls == reference.total_metric_calls
    assert resumed_strategy.rng.getstate() == reference_strategy.rng.getstate()
    assert resumed_strategy.sibling_planner.memory.get_state() == reference_strategy.sibling_planner.memory.get_state()
    assert resumed_strategy.sibling_planner.accepted == reference_strategy.sibling_planner.accepted
    assert len(interrupted_roles.editor_tasks) + len(resumed_roles.editor_tasks) == 2
    assert not resumed_roles.controller_prompts


def test_exhausted_generation_records_no_task_rewards():
    strategy = quality_strategy(Roles(incompatible=True))
    proposal = strategy.reflect(
        SEED, DATA, ["system_prompt"], metadata={"candidate_idx": 0, "minibatch_ids": [0, 1, 2]}
    )[0]
    assert proposal.metadata["generation_exhausted"]
    assert strategy.sibling_planner.memory.get_state()["records"] == []


class NoveltyRoles(Roles):
    """Answer semantic verification without changing existing deterministic edit fixtures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.verifier_prompts = []

    def __call__(self, prompt):
        if "\nCOMPARISON_DATA\n" in prompt:
            self.verifier_prompts.append(prompt)
            return json.dumps(
                {"verdict": "distinct", "matched_attempt_id": None, "reason": "A different intervention."}
            )
        return super().__call__(prompt)


def save_first_evaluation(strategy):
    """Preserve a tied first proposal without permanently consuming its sibling pair."""
    proposal = reflect(strategy)
    child = CandidateProposal(
        candidate={**SEED, **proposal.new_texts},
        parent_program_ids=[0],
        subsample_indices=[0, 1, 2],
        subsample_scores_before=[0, 0, 0],
        subsample_scores_after=[0, 0, 0],
        metadata=proposal.metadata,
    )
    strategy.sibling_planner.observe_evaluation(child, SEED)


def test_exact_evaluated_duplicate_replans_without_inheriting_exclusions_to_descendant():
    roles = NoveltyRoles()
    strategy = quality_strategy(roles)
    save_first_evaluation(strategy)
    proposal = reflect(strategy, iteration=2)
    assert proposal.metadata["duplicate_generation_count"] == 1
    blocked = proposal.metadata["attempt_records"][0]
    assert blocked["attempt_status"] == "duplicate_generation"
    assert blocked["controller_sampling"]["outcome_context"]["duplicate_feedback"] == []
    assert proposal.metadata["sibling_choice"]["pair"] != PAIR
    assert len(strategy.sibling_planner.memory.get_state()["records"]) == 1
    calls_before = len(roles.verifier_prompts)
    descendant = reflect(strategy, parent=1, iteration=3)
    assert descendant.metadata["sibling_choice"]["pair"] == PAIR
    assert descendant.metadata["duplicate_generation_count"] == 0
    assert len(roles.verifier_prompts) == calls_before


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "generation_exhausted"])
def test_blocked_duplicates_never_inherit_the_later_edits_result(outcome):
    records = [
        {"attempt_status": "duplicate_generation", "dropped_reason": "already evaluated", "component": "sys"},
        {"attempt_status": "completed", "component": "sys"},
    ]
    messages = GEPAState._attempt_records_to_chat(
        records, outcome, score_before=0, score_after=1, reason="later proposal outcome"
    )
    duplicate_feedback = messages[0]["content"]
    assert "duplicate" in duplicate_feedback.lower()
    assert "already evaluated" in duplicate_feedback
    assert "Score before" not in duplicate_feedback
    assert "Score after" not in duplicate_feedback
    assert "accepted" not in duplicate_feedback
    assert "later proposal outcome" not in duplicate_feedback
    assert "later proposal outcome" in messages[1]["content"]


def test_resume_after_duplicate_recovery_preserves_all_metadata_and_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(edit_novelty, "time", SimpleNamespace(perf_counter=lambda: 10.0))
    original = quality_strategy(NoveltyRoles())
    save_first_evaluation(original)
    snapshot = original.get_state()

    def resumed_strategy(roles, path):
        strategy = quality_strategy(roles)
        strategy.set_state(snapshot)
        strategy.sibling_planner.bind_run_dir(str(path))
        return strategy

    interrupted_roles = NoveltyRoles(interrupt_at=1)
    interrupted = resumed_strategy(interrupted_roles, tmp_path / "resume")
    with pytest.raises(KeyboardInterrupt):
        reflect(interrupted, iteration=2)
    actual_roles = NoveltyRoles()
    actual_strategy = resumed_strategy(actual_roles, tmp_path / "resume")
    actual = reflect(actual_strategy, iteration=2)
    reference_strategy = resumed_strategy(NoveltyRoles(), tmp_path / "reference")
    reference = reflect(reference_strategy, iteration=2)
    assert actual == reference
    assert actual_strategy.rng.getstate() == reference_strategy.rng.getstate()
    assert len(interrupted_roles.editor_tasks) + len(actual_roles.editor_tasks) == 2


def test_completed_novelty_step_replays_original_elapsed_time_without_new_call(tmp_path, monkeypatch):
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr(edit_novelty, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    original = quality_strategy(NoveltyRoles())
    save_first_evaluation(original)
    snapshot = original.get_state()
    interrupted = quality_strategy(NoveltyRoles())
    interrupted.set_state(snapshot)
    interrupted.sibling_planner.bind_run_dir(str(tmp_path))
    saved_step = interrupted.sibling_planner._step

    def stop_after_verifier_commit(scope, request, run):
        result = saved_step(scope, request, run)
        if "/novelty/" in scope and result.get("verifier_called"):
            raise KeyboardInterrupt("Stop after the completed verdict was journaled")
        return result

    monkeypatch.setattr(interrupted.sibling_planner, "_step", stop_after_verifier_commit)
    with pytest.raises(KeyboardInterrupt):
        reflect(interrupted, iteration=2)
    roles = NoveltyRoles()
    restored = quality_strategy(roles)
    restored.set_state(snapshot)
    restored.sibling_planner.bind_run_dir(str(tmp_path))
    proposal = reflect(restored, iteration=2)
    assert proposal.metadata["novelty_checks"][1]["elapsed_seconds"] == 2
    assert proposal.metadata["novelty_checks"][1]["verifier_called"]
    assert not roles.verifier_prompts and not roles.editor_tasks


@pytest.mark.parametrize("verdict", ["duplicate", "uncertain"])
def test_model_duplicate_replans_and_uncertain_candidate_remains_evaluable(verdict):
    class ScriptedVerifierRoles(NoveltyRoles):
        def __call__(self, prompt):
            if "\nCOMPARISON_DATA\n" not in prompt:
                return super().__call__(prompt)
            payload = json.loads(prompt.split("\nCOMPARISON_DATA\n", 1)[1])
            response = verdict if not self.verifier_prompts else "distinct"
            self.verifier_prompts.append(prompt)
            return json.dumps(
                {
                    "verdict": response,
                    "matched_attempt_id": payload["previous_records"][0]["attempt_id"]
                    if response == "duplicate"
                    else None,
                    "reason": "Same operational change."
                    if response == "duplicate"
                    else "Insufficient evidence of redundancy.",
                }
            )

    roles = ScriptedVerifierRoles()
    strategy = quality_strategy(roles)
    strategy.sibling_planner.memory.record_evaluation(
        parent_id=0,
        component="system_prompt",
        before=SEED["system_prompt"],
        after=TEMPLATE.replace_section_body(SEED["system_prompt"], "Style", "Brief. Prior background."),
        minibatch_ids=[0, 1, 2],
        action_pair=PAIR,
        action_name="contextualize",
        section="Style",
        scores_before=[0, 0, 0],
        scores_after=[0, 0, 0],
        attempt_id="prior-evaluated-edit",
        training_evidence=DATA["system_prompt"],
    )
    proposal = reflect(strategy)
    assert proposal.new_texts
    assert proposal.metadata["novelty_checks"][0]["verdict"] == f"model_{verdict}"
    assert proposal.metadata["duplicate_generation_count"] == int(verdict == "duplicate")
    assert len(roles.editor_tasks) == (2 if verdict == "duplicate" else 1)
    assert len(strategy.sibling_planner.memory.get_state()["records"]) == 1
    if verdict == "duplicate":
        assert proposal.metadata["attempt_records"][0]["attempt_status"] == "duplicate_generation"
        assert proposal.metadata["sibling_choice"]["pair"] != PAIR
    else:
        assert proposal.metadata["sibling_choice"]["pair"] == PAIR


@pytest.mark.parametrize("score", [0, 0.25, 0.5])
def test_callback_retains_training_outcome_for_accepted_and_rejected_edits(tmp_path, score):
    strategy = quality_strategy()
    callback = ActionDiversityCallback()
    run_once(tmp_path, strategy, Adapter(score), callback)
    [memory_record] = strategy.sibling_planner.memory.get_state()["records"]
    assert callback.proposal_records[0]["training_outcome"] == memory_record
    restored = ActionDiversityCallback()
    restored.set_state(callback.get_state())
    assert restored.proposal_records[0]["training_outcome"] == memory_record


@pytest.mark.parametrize("proposal_policy", ["independent", "sibling_diverse"])
def test_jev_verifier_requires_the_combined_policy(proposal_policy):
    with pytest.raises(ValueError, match="requires diversity_quality"):
        ThreeRoleReflectionLM(Roles(), level=2, proposal_policy=proposal_policy, novelty_backend="jev")


def test_verifier_backend_identity_prevents_cross_backend_checkpoint_resume():
    generative = quality_strategy()
    jev = quality_strategy(novelty_backend="jev")
    assert generative.run_contract(SEED)["controller"]["verifier"] != jev.run_contract(SEED)["controller"]["verifier"]
    for target, source in ((generative, jev), (jev, generative)):
        with pytest.raises(SiblingInvariantError, match="identity mismatch"):
            target.set_state(source.get_state())


def test_conflicting_observed_outcome_is_an_integrity_error():
    strategy = quality_strategy()
    proposal = reflect(strategy)
    child = CandidateProposal(
        candidate={**SEED, **proposal.new_texts},
        parent_program_ids=[0],
        subsample_indices=[0, 1, 2],
        subsample_scores_before=[0, 0, 0],
        subsample_scores_after=[0, 0, 0],
        metadata=proposal.metadata,
    )
    strategy.sibling_planner.observe_evaluation(child, SEED)
    child.subsample_scores_after = [1, 1, 1]
    with pytest.raises(SiblingInvariantError):
        strategy.sibling_planner.observe_evaluation(child, SEED)
    [record] = strategy.sibling_planner.memory.get_state()["records"]
    assert record["outcome"] == "tie"


def test_callback_preserves_allowed_lexical_flags_after_semantic_verification():
    callback = ActionDiversityCallback()
    callback.on_proposal_end(
        {
            "iteration": 1,
            "new_instructions": {"system_prompt": "Changed prompt"},
            "prompts": {},
            "raw_lm_outputs": {},
            "metadata": {
                "generation_outcome": "changed_candidate",
                "novelty_checks": [
                    {
                        "lexical_verdict": "suspected_similarity",
                        "verdict": "model_uncertain",
                        "blocked": False,
                    }
                ],
            },
        }
    )
    assert callback.proposal_outcomes["suspected_similar_edits_allowed"] == 1
