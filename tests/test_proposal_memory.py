"""Verify node-local training attribution, recovery and gentle proposal penalties."""

from copy import deepcopy

import pytest

from gepa.strategies.proposal_memory import ProposalMemory


def record(memory, attempt_id="1-0-0", **overrides):
    """Save one changed edit evaluated on its exact two-question training batch."""
    values = {
        "parent_id": 0,
        "component": "answer",
        "before": "Be brief.",
        "after": "Be brief. Cite evidence.",
        "minibatch_ids": [8, 4],
        "action_pair": "contextualize@Style/INSERT_TEXT",
        "action_name": "contextualize",
        "section": "Style",
        "scores_before": [0, 1],
        "scores_after": [0, 1],
        "attempt_id": attempt_id,
    }
    return memory.record_evaluation(**{**values, **overrides})


@pytest.mark.parametrize(
    "scores,outcome,gain", [([1, 1], "improvement", 0.5), ([0, 1], "tie", 0), ([0, 0], "regression", -0.5)]
)
def test_matched_training_outcomes_retain_exact_batch_and_edit(scores, outcome, gain):
    memory = ProposalMemory()
    result = record(memory, scores_after=scores)
    assert result["outcome"] == outcome
    assert result["training_gain"] == gain
    assert result["minibatch_ids"] == [8, 4]
    assert result["before"] == "Be brief."
    assert result["after"] == "Be brief. Cite evidence."
    result["scores_before"][0] = 100
    assert memory.get_state()["records"][0]["scores_before"] == [0, 1]


def test_idempotent_observation_rejects_conflicting_attribution():
    memory = ProposalMemory()
    record(memory)
    record(memory)
    assert len(memory.get_state()["records"]) == 1
    for changes in ({"scores_after": [1, 1]}, {"minibatch_ids": [4, 8]}, {"parent_id": 1}):
        with pytest.raises(ValueError, match="Conflicting"):
            record(memory, **changes)


def test_identical_text_on_independent_or_descendant_nodes_has_no_inherited_history():
    memory = ProposalMemory()
    saved = record(memory)
    record(memory, "2-0-0")
    pair = saved["action_pair"]
    assert memory.multipliers(0, "answer", saved["before"], [pair]) == {pair: 0.8}
    for parent_id, component, before in (
        (1, "answer", saved["before"]),
        (0, "query", saved["before"]),
        (0, "answer", "Other text."),
    ):
        assert memory.matches(parent_id, component, before) == []
        assert memory.multipliers(parent_id, component, before, [pair]) == {}
    assert memory.context(1, "answer", saved["before"])["recent_attempts"] == []


def test_recent_window_success_reset_floor_and_unknown_pairs():
    memory = ProposalMemory()
    saved = record(memory)
    pair = saved["action_pair"]
    assert memory.multipliers(0, "answer", saved["before"], [pair]) == {}
    for index in range(1, 7):
        record(memory, str(index))
    assert memory.multipliers(0, "answer", saved["before"], [pair]) == {pair: 0.5}
    record(memory, "success", scores_after=[1, 1])
    record(memory, "next-tie")
    assert memory.multipliers(0, "answer", saved["before"], [pair]) == {}
    record(memory, "next-tie-2")
    assert memory.multipliers(0, "answer", saved["before"], [pair]) == {pair: 0.8}
    assert memory.multipliers(0, "answer", saved["before"], ["another-pair"]) == {}
    for index in range(8):
        record(memory, f"new-{index}", action_pair="different-pair")
    assert memory.multipliers(0, "answer", saved["before"], [pair]) == {}
    assert len(memory.context(0, "answer", saved["before"])["recent_attempts"]) == 8
    assert len(memory.matches(0, "answer", saved["before"])) == 18


@pytest.mark.parametrize(
    "changes",
    [
        {"after": "Be brief."},
        {"parent_id": True},
        {"attempt_id": ""},
        {"minibatch_ids": []},
        {"minibatch_ids": [True, 4]},
        {"scores_after": [1]},
        {"scores_after": [float("nan"), 1]},
        {"scores_after": [True, 1]},
    ],
)
def test_unscored_or_malformed_attempts_never_receive_task_rewards(changes):
    memory = ProposalMemory()
    with pytest.raises(ValueError):
        record(memory, **changes)
    assert memory.get_state()["records"] == []


def test_restore_is_exact_independent_and_atomic():
    memory = ProposalMemory()
    saved = record(memory)
    record(memory, "2-0-0")
    snapshot = memory.get_state()
    restored = ProposalMemory()
    restored.set_state(snapshot)
    assert restored.get_state() == memory.get_state()
    assert restored.context(0, "answer", saved["before"]) == memory.context(0, "answer", saved["before"])
    assert restored.multipliers(0, "answer", saved["before"], [saved["action_pair"]]) == {saved["action_pair"]: 0.8}
    snapshot["records"][0]["before"] = "mutated"
    assert restored.get_state() == memory.get_state()
    invalid = memory.get_state()
    invalid["records"][1]["training_gain"] = 0.9
    with pytest.raises(ValueError, match="outcome"):
        restored.set_state(invalid)
    assert restored.get_state() == memory.get_state()


def test_controller_context_bounds_edit_text_but_preserves_full_novelty_records():
    memory = ProposalMemory()
    before = "Original " * 1000
    after = "Revised " * 1000
    record(memory, before=before, after=after)
    context = memory.context(0, "answer", before)["recent_attempts"][0]
    assert "before" not in context and "after" not in context
    assert len(context["diff_excerpt"]) == 1200
    assert context["diff_truncated"] and context["same_prompt"]
    assert len(context["before_sha256"]) == len(context["after_sha256"]) == 64
    assert memory.matches(0, "answer", before)[0]["after"] == after


def test_full_training_evidence_is_preserved_for_novelty_but_not_controller_context():
    memory = ProposalMemory()
    evidence = [{"Inputs": {"question": "Which year?"}, "Feedback": "Incorrect date."}]
    saved = record(memory, training_evidence=evidence)
    evidence[0]["Feedback"] = "Changed later"
    assert memory.matches(0, "answer", saved["before"])[0]["training_evidence"][0]["Feedback"] == "Incorrect date."
    assert "training_evidence" not in memory.context(0, "answer", saved["before"])["recent_attempts"][0]
    restored = ProposalMemory()
    restored.set_state(memory.get_state())
    assert restored.get_state() == memory.get_state()
    with pytest.raises(ValueError, match="Conflicting"):
        record(memory, training_evidence=evidence)


@pytest.mark.parametrize(
    "mutation", ["version", "missing", "duplicate", "conflict", "extra", "outcome", "evaluated", "bool_gain"]
)
def test_restore_rejects_malformed_or_conflicting_history(mutation):
    memory = ProposalMemory()
    record(memory)
    snapshot = memory.get_state()
    if mutation == "version":
        snapshot["version"] = True
    elif mutation == "missing":
        del snapshot["records"][0]["before"]
    elif mutation in {"duplicate", "conflict"}:
        snapshot["records"].append(deepcopy(snapshot["records"][0]))
        if mutation == "conflict":
            snapshot["records"][-1]["scores_after"] = [1, 1]
    elif mutation == "extra":
        snapshot["records"][0]["unexpected"] = "untrusted"
    elif mutation == "evaluated":
        snapshot["records"][0]["evaluated"] = 1
    elif mutation == "bool_gain":
        snapshot["records"][0]["training_gain"] = False
    else:
        snapshot["records"][0]["outcome"] = "improvement"
    with pytest.raises(ValueError):
        ProposalMemory().set_state(snapshot)
