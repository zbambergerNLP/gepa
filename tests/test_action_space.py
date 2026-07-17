# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for action-conditioned reflection (Rev 1)."""

import random

import pytest

from gepa.core.action_tracking import ActionDiversityCallback
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionLM,
    ReflectionProposal,
    StatelessReflectionLM,
)
from gepa.strategies.action_space import (
    DEFAULT_ACTIONS,
    AllActionsSelector,
    PromptEditAction,
    RandomActionSelector,
    RoundRobinActionSelector,
    format_action_suffix,
)


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_reflection_lm.py)
# ---------------------------------------------------------------------------


class RecordingLM:
    """A fake reflection LM: records prompts, returns a fenced instruction."""

    def __init__(self, reply: str = "improved instruction"):
        self.reply = reply
        self.calls: list = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return f"Here is the update:\n```\n{self.reply}\n```"


class BatchRecordingLM(RecordingLM):
    """A fake LM that also exposes ``batch_complete`` (like ``gepa.lm.LM``)."""

    def __init__(self, reply: str = "improved instruction"):
        super().__init__(reply)
        self.batch_calls: list[list] = []

    def batch_complete(self, messages_list, max_workers: int = 10):
        self.batch_calls.append(messages_list)
        return [f"```\n{self.reply}\n```" for _ in messages_list]


def _reflective_dataset(components):
    return {name: [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "bad"}] for name in components}


# ---------------------------------------------------------------------------
# Action selector tests
# ---------------------------------------------------------------------------


class TestRoundRobinActionSelector:
    def test_cycles_all_actions(self):
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        rng = random.Random(42)
        # First cycle: should return each action exactly once.
        actions = selector.select(len(DEFAULT_ACTIONS), rng)
        assert [a.name for a in actions] == [a.name for a in DEFAULT_ACTIONS]

    def test_wraps_around(self):
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        rng = random.Random(42)
        # Request more than the number of actions.
        actions = selector.select(len(DEFAULT_ACTIONS) + 2, rng)
        assert actions[-2].name == DEFAULT_ACTIONS[0].name
        assert actions[-1].name == DEFAULT_ACTIONS[1].name

    def test_deterministic_across_calls(self):
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        rng = random.Random(42)
        first = selector.select(3, rng)
        second = selector.select(3, rng)
        # Second call continues from where first left off.
        assert first[0].name == DEFAULT_ACTIONS[0].name
        assert second[0].name == DEFAULT_ACTIONS[3].name


class TestRandomActionSelector:
    def test_returns_correct_count(self):
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(42))
        rng = random.Random(0)  # Not used by RandomActionSelector (uses its own rng)
        actions = selector.select(5, rng)
        assert len(actions) == 5

    def test_membership(self):
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(42))
        rng = random.Random(0)
        actions = selector.select(20, rng)
        for action in actions:
            assert action in DEFAULT_ACTIONS


class TestAllActionsSelector:
    def test_returns_full_set(self):
        selector = AllActionsSelector(DEFAULT_ACTIONS)
        rng = random.Random(42)
        actions = selector.select(1, rng)
        assert len(actions) == len(DEFAULT_ACTIONS)

    def test_ignores_n(self):
        selector = AllActionsSelector(DEFAULT_ACTIONS)
        rng = random.Random(42)
        assert selector.select(1, rng) == selector.select(100, rng)


class TestFormatActionSuffix:
    def test_contains_name_and_instruction(self):
        action = DEFAULT_ACTIONS[0]
        suffix = format_action_suffix(action)
        assert action.name in suffix
        assert action.instruction_suffix in suffix
        assert "--- ACTION CONSTRAINT ---" in suffix

    def test_contains_description(self):
        action = DEFAULT_ACTIONS[0]
        suffix = format_action_suffix(action)
        assert action.description in suffix


# ---------------------------------------------------------------------------
# Reflection LM integration tests
# ---------------------------------------------------------------------------


class TestActionConditionedReflection:
    def test_action_suffix_appended_to_prompt(self):
        lm = RecordingLM()
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": "old instruction"}
        ds = _reflective_dataset(["system_prompt"])

        reflection.reflect(candidate, ds, ["system_prompt"])

        # The prompt sent to the LM should contain the action suffix.
        assert len(lm.calls) == 1
        prompt_text = lm.calls[0] if isinstance(lm.calls[0], str) else str(lm.calls[0])
        assert "--- ACTION CONSTRAINT ---" in prompt_text
        assert DEFAULT_ACTIONS[0].name in prompt_text

    def test_action_recorded_in_metadata(self):
        lm = RecordingLM()
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": "old instruction"}
        ds = _reflective_dataset(["system_prompt"])

        proposal, _ = reflection.reflect(candidate, ds, ["system_prompt"])

        assert "action" in proposal.metadata
        assert proposal.metadata["action"] == DEFAULT_ACTIONS[0].name

    def test_no_action_selector_backward_compatible(self):
        lm = RecordingLM()
        reflection = StatelessReflectionLM(lm)
        candidate = {"system_prompt": "old instruction"}
        ds = _reflective_dataset(["system_prompt"])

        proposal, next_lm = reflection.reflect(candidate, ds, ["system_prompt"])

        assert isinstance(proposal, ReflectionProposal)
        assert next_lm is reflection
        # No action metadata.
        assert "action" not in proposal.metadata
        # No action suffix in prompt.
        prompt_text = lm.calls[0] if isinstance(lm.calls[0], str) else str(lm.calls[0])
        assert "--- ACTION CONSTRAINT ---" not in prompt_text

    def test_satisfies_protocol(self):
        lm = RecordingLM()
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        assert isinstance(reflection, ReflectionLM)

    def test_all_components_share_action(self):
        """Multi-component job: all components should use the same action."""
        lm = RecordingLM()
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": "sys", "user_prompt": "usr"}
        ds = _reflective_dataset(["system_prompt", "user_prompt"])

        proposal, _ = reflection.reflect(candidate, ds, ["system_prompt", "user_prompt"])

        # Both components should have been rendered with the same action.
        assert len(lm.calls) == 2
        action_name = DEFAULT_ACTIONS[0].name
        for call in lm.calls:
            prompt_text = call if isinstance(call, str) else str(call)
            assert action_name in prompt_text

        assert proposal.metadata["action"] == action_name

    def test_batch_reflect_many_different_actions(self):
        """Multiple jobs get different actions via round-robin."""
        lm = BatchRecordingLM()
        selector = RoundRobinActionSelector(DEFAULT_ACTIONS)
        reflection = StatelessReflectionLM(lm, action_selector=selector)

        jobs = [
            ({"sp": "old1"}, _reflective_dataset(["sp"]), ["sp"]),
            ({"sp": "old2"}, _reflective_dataset(["sp"]), ["sp"]),
            ({"sp": "old3"}, _reflective_dataset(["sp"]), ["sp"]),
        ]

        results = reflection.reflect_many(jobs)

        assert len(results) == 3
        # Each job should have a different action (round-robin).
        actions = [r[0].metadata["action"] for r in results]
        assert actions[0] == DEFAULT_ACTIONS[0].name
        assert actions[1] == DEFAULT_ACTIONS[1].name
        assert actions[2] == DEFAULT_ACTIONS[2].name


# ---------------------------------------------------------------------------
# ActionDiversityCallback tests
# ---------------------------------------------------------------------------


class TestActionDiversityCallback:
    def _make_proposal_end_event(self, iteration: int, action_name: str | None = None) -> dict:
        """Build a ProposalEndEvent dict with an action suffix in the prompt."""
        prompt_text = "I provided an assistant with instructions..."
        if action_name:
            prompt_text += (
                "\n\n--- ACTION CONSTRAINT ---\n"
                f"You MUST make exactly one type of edit: {action_name}\n"
            )
        return {
            "iteration": iteration,
            "new_instructions": {"system_prompt": f"instruction from {action_name or 'unconditioned'}"},
            "prompts": {"system_prompt": prompt_text},
            "raw_lm_outputs": {"system_prompt": "raw output"},
        }

    def test_counts_proposals_per_action(self):
        cb = ActionDiversityCallback()
        cb.on_proposal_end(self._make_proposal_end_event(1, "add_constraint"))
        cb.on_proposal_end(self._make_proposal_end_event(1, "add_constraint"))
        cb.on_proposal_end(self._make_proposal_end_event(2, "restructure"))

        assert cb.action_proposal_counts["add_constraint"] == 2
        assert cb.action_proposal_counts["restructure"] == 1

    def test_tracks_acceptance_rate(self):
        cb = ActionDiversityCallback()
        cb.on_proposal_end(self._make_proposal_end_event(1, "add_constraint"))
        cb.on_candidate_accepted({"iteration": 1, "new_candidate_idx": 1, "new_score": 0.8, "parent_ids": [0]})

        cb.on_proposal_end(self._make_proposal_end_event(2, "add_constraint"))
        cb.on_candidate_rejected({"iteration": 2, "old_score": 0.8, "new_score": 0.6, "reason": "no improvement"})

        assert cb.action_acceptance_counts["add_constraint"] == 1
        assert cb.action_rejection_counts["add_constraint"] == 1

    def test_summary_returns_expected_keys(self):
        cb = ActionDiversityCallback()
        cb.on_proposal_end(self._make_proposal_end_event(1, "restructure"))
        cb.on_candidate_accepted({"iteration": 1, "new_candidate_idx": 1, "new_score": 0.9, "parent_ids": [0]})

        s = cb.summary()
        assert "action_proposal_counts" in s
        assert "action_acceptance_counts" in s
        assert "action_rejection_counts" in s
        assert "action_acceptance_rates" in s
        assert "textual_diversity_per_iteration" in s
        assert "total_proposals" in s
        assert "total_accepted" in s

    def test_textual_diversity_computed(self):
        cb = ActionDiversityCallback()
        # Two different proposals in the same iteration.
        cb.on_proposal_end(self._make_proposal_end_event(1, "add_constraint"))
        cb.on_proposal_end(self._make_proposal_end_event(1, "restructure"))

        diversity = cb.textual_diversity()
        assert "1" in diversity
        # Different texts should have non-zero dissimilarity.
        assert diversity["1"] > 0.0

    def test_unconditioned_proposals_not_counted(self):
        cb = ActionDiversityCallback()
        cb.on_proposal_end(self._make_proposal_end_event(1, None))
        cb.on_candidate_accepted({"iteration": 1, "new_candidate_idx": 1, "new_score": 0.8, "parent_ids": [0]})

        assert len(cb.action_proposal_counts) == 0
        assert len(cb.action_acceptance_counts) == 0
