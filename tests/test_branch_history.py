# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for branch-local proposal-attempt history in :class:`GEPAState`."""

import json
from typing import Any

import pytest

from gepa.core.state import GEPAState, ValsetEvaluation


def evaluation(score: float = 0.5) -> ValsetEvaluation[str, int]:
    """Build one-example validation output for state updates."""
    return ValsetEvaluation(
        outputs_by_val_id={0: f"output-{score}"},
        scores_by_val_id={0: score},
        objective_scores_by_val_id=None,
    )


def accept(
    state: GEPAState[str, int],
    parents: list[int],
    marker: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Add one accepted candidate with a marker revision record."""
    proposal_metadata = metadata if metadata is not None else {"attempt_records": [attempt_record(marker)]}
    return state.update_state_with_new_program(
        parent_program_idx=parents,
        new_program={"sys": marker},
        valset_evaluation=evaluation(0.6),
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=1,
        iteration_id=f"iteration-{marker}",
        proposal_metadata=proposal_metadata,
    )


def attempt_record(marker: str, **extra: Any) -> dict[str, Any]:
    """Build proposal diagnostics with an actual chat trajectory."""
    return {
        "marker": marker,
        "chat_messages": [
            {"role": "assistant", "content": f"Assistant attempt: {marker}"},
            {"role": "user", "content": f"Tool observation: {marker}"},
        ],
        **extra,
    }


def transcript(
    marker: str,
    outcome: str = "accepted",
    *,
    score_before: float | None = None,
    score_after: float | None = None,
    reason: str | None = None,
) -> list[dict[str, str]]:
    """Build the expected persistent user/assistant transcript for one attempt."""
    outcome_feedback = {
        "accepted": (
            "Optimizer feedback: ACCEPTED. This edit passed the acceptance criterion and is retained on this branch."
        ),
        "rejected": (
            "Optimizer feedback: REJECTED. This edit did not pass the acceptance criterion; the branch document "
            "is unchanged."
        ),
        "dropped": (
            "Optimizer feedback: DROPPED. This attempt produced no completed candidate; the branch document "
            "is unchanged."
        ),
    }
    feedback = [outcome_feedback[outcome]]
    if score_before is not None:
        feedback.append(f"Score before: {score_before}.")
    if score_after is not None:
        feedback.append(f"Score after: {score_after}.")
    if reason is not None:
        feedback.append(f"Reason: {reason}")
    return [
        {"role": "assistant", "content": f"Assistant attempt: {marker}"},
        {"role": "user", "content": f"Tool observation: {marker}"},
        {"role": "user", "content": " ".join(feedback)},
    ]


def test_seed_starts_with_no_global_revision_history() -> None:
    """Keep history absent until an accepted descendant exists."""
    state = GEPAState({"sys": "seed"}, evaluation())
    assert state.revision_history_for_candidate(0) == []


def test_descendant_inherits_only_its_parent_lineage() -> None:
    """Append accepted records along one branch in seed-to-leaf order."""
    state = GEPAState({"sys": "seed"}, evaluation())
    child = accept(state, [0], "parent")
    grandchild = accept(state, [child], "child")
    assert state.revision_history_for_candidate(child) == transcript("parent")
    assert state.revision_history_for_candidate(grandchild) == transcript("parent") + transcript("child")
    assert state.revision_history_for_candidate(0) == []


def test_sibling_histories_are_isolated_and_never_become_global() -> None:
    """Prevent one accepted sibling's private trajectory from reaching another."""
    state = GEPAState({"sys": "seed"}, evaluation())
    left = accept(state, [0], "left-only")
    right = accept(state, [0], "right-only")
    assert state.revision_history_for_candidate(left) == transcript("left-only")
    assert state.revision_history_for_candidate(right) == transcript("right-only")
    assert "right-only" not in str(state.revision_history_for_candidate(left))
    assert "left-only" not in str(state.revision_history_for_candidate(right))


def test_history_accessor_returns_a_deep_copy() -> None:
    """Stop callers from mutating persistent branch context through an accessor."""
    state = GEPAState({"sys": "seed"}, evaluation())
    child = accept(state, [0], "accepted")
    exposed = state.revision_history_for_candidate(child)
    exposed[0]["content"] = "mutated"
    exposed.append({"role": "user", "content": "injected"})
    assert state.revision_history_for_candidate(child) == transcript("accepted")


def test_merge_inherits_only_the_exact_prefix_shared_by_all_parents() -> None:
    """Drop sibling-private suffixes instead of turning either into global history."""
    state = GEPAState({"sys": "seed"}, evaluation())
    common = accept(state, [0], "common")
    left = accept(state, [common], "left-only")
    right = accept(state, [common], "right-only")
    merged = accept(state, [left, right], "merge")
    assert state.revision_history_for_candidate(merged) == transcript("common") + transcript("merge")


def test_candidate_without_new_revision_records_inherits_parent_history_only() -> None:
    """Avoid inventing edit records for non-reflective accepted candidates."""
    state = GEPAState({"sys": "seed"}, evaluation())
    parent = accept(state, [0], "reflection")
    merged = accept(state, [parent], "ignored", metadata={"other_metadata": True})
    assert state.revision_history_for_candidate(merged) == transcript("reflection")


def test_rejected_attempt_is_visible_only_when_reselecting_that_candidate() -> None:
    """Keep a failed edit on its parent branch without mutating existing siblings."""
    state = GEPAState({"sys": "seed"}, evaluation())
    left = accept(state, [0], "left")
    right = accept(state, [0], "right")
    state.record_proposal_attempts(
        left,
        {"attempt_records": [attempt_record("tried-and-regressed")]},
        outcome="rejected",
        score_before=0.8,
        score_after=0.3,
        reason="did not help",
    )
    assert state.revision_history_for_candidate(left) == transcript("left") + transcript(
        "tried-and-regressed",
        "rejected",
        score_before=0.8,
        score_after=0.3,
        reason="did not help",
    )
    assert state.revision_history_for_candidate(right) == transcript("right")
    assert state.revision_history_for_candidate(0) == []


def test_accepted_descendant_retains_prior_dropped_attempt_lineage() -> None:
    """Carry a prior exhausted ReAct attempt into later accepted descendants."""
    state = GEPAState({"sys": "seed"}, evaluation())
    parent = accept(state, [0], "parent")
    state.record_proposal_attempts(
        parent,
        {"attempt_records": [attempt_record("exhausted")]},
        outcome="dropped",
        reason="Reflection attempt produced no completed text update.",
    )
    child = accept(
        state,
        [parent],
        "child",
        metadata={"attempt_records": [attempt_record("child")]},
    )
    history = state.revision_history_for_candidate(child)
    assert history == (
        transcript("parent")
        + transcript(
            "exhausted",
            "dropped",
            reason="Reflection attempt produced no completed text update.",
        )
        + transcript("child")
    )


def test_proposal_time_snapshot_prevents_same_batch_sibling_leakage() -> None:
    """Exclude attempts recorded after a concurrent proposal saw its parent."""
    state = GEPAState({"sys": "seed"}, evaluation())
    state.record_proposal_attempts(
        0,
        {"attempt_records": [attempt_record("dropped-sibling")]},
        outcome="dropped",
    )
    child = accept(
        state,
        [0],
        "accepted-sibling",
        metadata={
            "attempt_records": [attempt_record("accepted-sibling")],
            "parent_branch_history_lengths": {"0": 0},
        },
    )
    assert state.revision_history_for_candidate(0) == transcript("dropped-sibling", "dropped")
    assert state.revision_history_for_candidate(child) == transcript("accepted-sibling")


def test_accepted_multi_component_proposal_keeps_internal_drop_outcome() -> None:
    """Do not relabel an exhausted component because another component was accepted."""
    state = GEPAState({"sys": "seed"}, evaluation())
    child = accept(
        state,
        [0],
        "mixed",
        metadata={
            "attempt_records": [
                attempt_record("completed", attempt_status="completed"),
                attempt_record(
                    "exhausted",
                    attempt_status="dropped",
                    dropped_reason="invalid calls",
                ),
            ]
        },
    )
    history = state.revision_history_for_candidate(child)
    assert history == transcript("completed") + transcript("exhausted", "dropped", reason="invalid calls")


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"revision_records": "not-a-sequence"}, id="string"),
        pytest.param({"revision_records": ["not-a-mapping"]}, id="bad_entry"),
    ],
)
def test_revision_record_shape_is_validated_before_state_mutation(metadata: dict[str, Any]) -> None:
    """Reject malformed reserved metadata without adding a candidate."""
    state = GEPAState({"sys": "seed"}, evaluation())
    with pytest.raises(TypeError, match="revision[_ ]record"):
        accept(state, [0], "bad", metadata=metadata)
    assert state.program_candidates == [{"sys": "seed"}]
    assert state.revision_history_by_candidate == [[]]


def test_revision_history_survives_state_save_and_load(tmp_path) -> None:
    """Persist branch context so resumed optimization keeps the same lineage."""
    state = GEPAState({"sys": "seed"}, evaluation())
    state.total_num_evals = 1
    state.num_full_ds_evals = 1
    child = accept(state, [0], "accepted")
    state.save(str(tmp_path))
    loaded = GEPAState.load(str(tmp_path))
    assert loaded.revision_history_for_candidate(child) == transcript("accepted")
    assert loaded.is_consistent()


def test_persisted_history_contains_only_user_assistant_content_messages() -> None:
    """Keep diagnostics out of the transcript consumed by later ReAct calls."""
    state = GEPAState({"sys": "seed"}, evaluation())
    child = accept(state, [0], "accepted")
    history = state.revision_history_for_candidate(child)
    assert {message["role"] for message in history} == {"user", "assistant"}
    assert all(set(message) == {"role", "content"} for message in history)
    assert all(isinstance(message["content"], str) for message in history)


def test_persisted_chat_messages_are_bounded_and_json_serializable() -> None:
    """Bound each actual turn while leaving total-history overflow explicit to ReAct."""
    state = GEPAState({"sys": "seed"}, evaluation())
    child = accept(
        state,
        [0],
        "large",
        metadata={
            "attempt_records": [
                {
                    "chat_messages": [
                        {"role": "assistant", "content": "x" * 5000},
                        {"role": "user", "content": "tool observation"},
                    ]
                }
            ]
        },
    )
    history = state.revision_history_for_candidate(child)
    assert len(history[0]["content"]) <= 2048
    assert history[0]["content"].endswith("chars)")
    json.dumps(history)


def test_schema_seven_attempt_records_migrate_to_chat_messages() -> None:
    """Resume pre-chat-history runs without replaying opaque diagnostic objects."""
    state_dict: dict[str, Any] = {
        "validation_schema_version": 7,
        "program_candidates": [{"sys": "seed"}],
        "revision_history_by_candidate": [
            [
                {
                    "assistant": "attempted replacement",
                    "observation": "replacement applied",
                    "outcome": "rejected",
                    "subsample_score_before": 0.8,
                    "subsample_score_after": 0.2,
                    "outcome_reason": "did not help",
                }
            ]
        ],
    }
    GEPAState._upgrade_state_dict(state_dict)
    assert state_dict["revision_history_by_candidate"] == [
        [
            {"role": "assistant", "content": "attempted replacement"},
            {"role": "user", "content": "replacement applied"},
            {
                "role": "user",
                "content": (
                    "Optimizer feedback: REJECTED. This edit did not pass the acceptance criterion; the branch "
                    "document is unchanged. Score before: 0.8. Score after: 0.2. Reason: did not help"
                ),
            },
        ]
    ]
