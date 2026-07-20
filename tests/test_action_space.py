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
    ActionDistribution,
    RandomActionSelector,
    VerbalizedActionSelector,
    _sample_from_tails,
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


class TestRandomActionSelector:
    def test_returns_correct_count(self):
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(42))
        rng = random.Random(0)  # Passed rng takes precedence over the constructor rng.
        actions = selector.select(5, rng)
        assert len(actions) == 5

    def test_membership(self):
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(42))
        rng = random.Random(0)
        actions = selector.select(20, rng)
        for action in actions:
            assert action in DEFAULT_ACTIONS

    def test_empty_actions_raises(self):
        with pytest.raises(ValueError):
            RandomActionSelector([])

    def test_passed_rng_takes_precedence(self):
        # Different constructor rngs, same passed rng -> identical sequences.
        selector_a = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(1))
        selector_b = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(2))
        actions_a = selector_a.select(20, random.Random(42))
        actions_b = selector_b.select(20, random.Random(42))
        assert [a.name for a in actions_a] == [a.name for a in actions_b]

    def test_falls_back_to_instance_rng(self):
        selector_a = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(7))
        selector_b = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(7))
        actions_a = selector_a.select(20)
        actions_b = selector_b.select(20)
        assert [a.name for a in actions_a] == [a.name for a in actions_b]


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
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": "old instruction"}
        ds = _reflective_dataset(["system_prompt"])

        reflection.reflect(candidate, ds, ["system_prompt"])

        # The prompt sent to the LM should contain the action suffix.
        assert len(lm.calls) == 1
        prompt_text = lm.calls[0] if isinstance(lm.calls[0], str) else str(lm.calls[0])
        assert "--- ACTION CONSTRAINT ---" in prompt_text

    def test_action_recorded_in_metadata(self):
        lm = RecordingLM()
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": "old instruction"}
        ds = _reflective_dataset(["system_prompt"])

        proposal, _ = reflection.reflect(candidate, ds, ["system_prompt"])

        assert "action" in proposal.metadata
        assert proposal.metadata["action"] in [a.name for a in DEFAULT_ACTIONS]

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
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        assert isinstance(reflection, ReflectionLM)

    def test_all_components_share_action(self):
        """Multi-component job: all components should use the same action."""
        lm = RecordingLM()
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": "sys", "user_prompt": "usr"}
        ds = _reflective_dataset(["system_prompt", "user_prompt"])

        proposal, _ = reflection.reflect(candidate, ds, ["system_prompt", "user_prompt"])

        # Both components should have been rendered with the same action.
        assert len(lm.calls) == 2
        action_name = proposal.metadata["action"]
        assert action_name in [a.name for a in DEFAULT_ACTIONS]
        for call in lm.calls:
            prompt_text = call if isinstance(call, str) else str(call)
            assert action_name in prompt_text

    def test_batch_reflect_many_assigns_actions(self):
        """Multiple jobs each get an action assigned."""
        lm = BatchRecordingLM()
        selector = RandomActionSelector(DEFAULT_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)

        jobs = [
            ({"sp": "old1"}, _reflective_dataset(["sp"]), ["sp"]),
            ({"sp": "old2"}, _reflective_dataset(["sp"]), ["sp"]),
            ({"sp": "old3"}, _reflective_dataset(["sp"]), ["sp"]),
        ]

        results = reflection.reflect_many(jobs)

        assert len(results) == 3
        # Each job should have an action from DEFAULT_ACTIONS.
        for r in results:
            assert "action" in r[0].metadata
            assert r[0].metadata["action"] in [a.name for a in DEFAULT_ACTIONS]


# ---------------------------------------------------------------------------
# ActionDiversityCallback tests
# ---------------------------------------------------------------------------


class TestActionDiversityCallback:
    def _metadata(self, action_name: str | None, proposal_id: str = "1-0") -> dict:
        if action_name is None:
            return {"proposal_id": proposal_id}
        return {"proposal_id": proposal_id, "action": action_name}

    def _make_proposal_end_event(self, iteration: int, action_name: str | None = None) -> dict:
        """Build a ProposalEndEvent dict carrying the action in its metadata."""
        return {
            "iteration": iteration,
            "new_instructions": {"system_prompt": f"instruction from {action_name or 'unconditioned'}"},
            "prompts": {"system_prompt": "I provided an assistant with instructions..."},
            "raw_lm_outputs": {"system_prompt": "raw output"},
            "metadata": self._metadata(action_name),
        }

    def _accepted_event(self, iteration: int, action_name: str | None, score: float = 0.8) -> dict:
        return {
            "iteration": iteration,
            "new_candidate_idx": 1,
            "new_score": score,
            "parent_ids": [0],
            "metadata": self._metadata(action_name),
        }

    def _rejected_event(self, iteration: int, action_name: str | None, old: float = 0.8, new: float = 0.6) -> dict:
        return {
            "iteration": iteration,
            "old_score": old,
            "new_score": new,
            "reason": "no improvement",
            "metadata": self._metadata(action_name),
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
        cb.on_candidate_accepted(self._accepted_event(1, "add_constraint"))

        cb.on_proposal_end(self._make_proposal_end_event(2, "add_constraint"))
        cb.on_candidate_rejected(self._rejected_event(2, "add_constraint"))

        assert cb.action_acceptance_counts["add_constraint"] == 1
        assert cb.action_rejection_counts["add_constraint"] == 1

    def test_summary_returns_expected_keys(self):
        cb = ActionDiversityCallback()
        cb.on_proposal_end(self._make_proposal_end_event(1, "restructure"))
        cb.on_candidate_accepted(self._accepted_event(1, "restructure", score=0.9))

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
        cb.on_candidate_accepted(self._accepted_event(1, None))

        assert len(cb.action_proposal_counts) == 0
        assert len(cb.action_acceptance_counts) == 0

    def test_engine_event_order_rejections_before_acceptances(self):
        """The engine fires ALL rejections before acceptances within an iteration.

        Attribution must come from each event's own metadata, not arrival order
        (a FIFO pairing would attribute B's and C's rejections to A and B here).
        """
        cb = ActionDiversityCallback()
        cb.on_proposal_end(self._make_proposal_end_event(1, "action_a"))
        cb.on_proposal_end(self._make_proposal_end_event(1, "action_b"))
        cb.on_proposal_end(self._make_proposal_end_event(1, "action_c"))

        # Engine order: rejections for B and C first, then A's acceptance.
        cb.on_candidate_rejected(self._rejected_event(1, "action_b"))
        cb.on_candidate_rejected(self._rejected_event(1, "action_c", old=0.8, new=0.5))
        cb.on_candidate_accepted(self._accepted_event(1, "action_a"))

        assert dict(cb.action_acceptance_counts) == {"action_a": 1}
        assert dict(cb.action_rejection_counts) == {"action_b": 1, "action_c": 1}
        assert cb.action_score_deltas["action_b"] == [pytest.approx(-0.2)]
        assert cb.action_score_deltas["action_c"] == [pytest.approx(-0.3)]

    def test_events_without_metadata_tolerated(self):
        """Events lacking a metadata key (legacy/synthetic) neither crash nor count."""
        cb = ActionDiversityCallback()
        cb.on_proposal_end(
            {
                "iteration": 1,
                "new_instructions": {"sp": "text"},
                "prompts": {"sp": "prompt"},
                "raw_lm_outputs": {"sp": "raw"},
            }
        )
        cb.on_candidate_accepted({"iteration": 1, "new_candidate_idx": 1, "new_score": 0.8, "parent_ids": [0]})
        cb.on_candidate_rejected({"iteration": 1, "old_score": 0.8, "new_score": 0.6, "reason": "worse"})

        assert len(cb.action_proposal_counts) == 0
        assert len(cb.action_acceptance_counts) == 0
        assert len(cb.action_rejection_counts) == 0


# ---------------------------------------------------------------------------
# Verbalized action selector tests
# ---------------------------------------------------------------------------

VALID_LM_OUTPUT = """
<response>
<candidate>
<action>add_constraint</action>
<reasoning>The feedback shows edge cases being missed</reasoning>
<probability>0.35</probability>
</candidate>
<candidate>
<action>adjust_specificity</action>
<reasoning>The prompt is too vague for multi-hop reasoning</reasoning>
<probability>0.30</probability>
</candidate>
<candidate>
<action>add_illustration</action>
<reasoning>A worked example would help</reasoning>
<probability>0.20</probability>
</candidate>
<candidate>
<action>restructure</action>
<reasoning>Reordering might improve attention</reasoning>
<probability>0.10</probability>
</candidate>
<candidate>
<action>edit_guidelines</action>
<reasoning>Could refine the persona</reasoning>
<probability>0.05</probability>
</candidate>
</response>
"""


class FakeLM:
    """A fake LM that returns a fixed response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return self.response


class TestVerbalizedActionSelector:
    def test_parse_valid_distribution(self):
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution(VALID_LM_OUTPUT, random.Random(42))
        assert len(dist.entries) == 5
        # Probabilities should be renormalized to sum to 1.
        assert abs(sum(dist.probabilities) - 1.0) < 1e-6

    def test_parse_malformed_xml_falls_back(self):
        lm = FakeLM("this is not xml at all")
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution("this is not xml at all", random.Random(42))
        # Should fall back to uniform over all actions.
        assert len(dist.entries) == len(DEFAULT_ACTIONS)
        expected_prob = 1.0 / len(DEFAULT_ACTIONS)
        for _, p, _ in dist.entries:
            assert abs(p - expected_prob) < 1e-6

    def test_parse_missing_probability_skips_entry(self):
        partial_output = """
<response>
<candidate>
<action>add_constraint</action>
<reasoning>good</reasoning>
<probability>0.60</probability>
</candidate>
<candidate>
<action>restructure</action>
<reasoning>no probability here</reasoning>
</candidate>
</response>
"""
        lm = FakeLM(partial_output)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution(partial_output, random.Random(42))
        assert len(dist.entries) == 1
        assert dist.entries[0][0].name == "add_constraint"

    def test_parse_unknown_action_ignored(self):
        bad_action_output = """
<response>
<candidate>
<action>nonexistent_action</action>
<reasoning>doesn't exist</reasoning>
<probability>0.50</probability>
</candidate>
<candidate>
<action>add_constraint</action>
<reasoning>real action</reasoning>
<probability>0.50</probability>
</candidate>
</response>
"""
        lm = FakeLM(bad_action_output)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution(bad_action_output, random.Random(42))
        assert len(dist.entries) == 1
        assert dist.entries[0][0].name == "add_constraint"

    def test_parse_strips_think_tags(self):
        output_with_think = "<think>\nLet me analyze...\n</think>\n" + VALID_LM_OUTPUT
        lm = FakeLM(output_with_think)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution(output_with_think, random.Random(42))
        assert len(dist.entries) == 5

    def test_parse_unclosed_think_falls_back_to_uniform(self):
        # An unclosed <think> swallows everything after it, so no candidates
        # remain and the parser must fall back to a uniform distribution.
        truncated_output = "<think>\nreasoning cut off mid-stream " + VALID_LM_OUTPUT
        lm = FakeLM(truncated_output)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution(truncated_output, random.Random(42))
        assert len(dist.entries) == len(DEFAULT_ACTIONS)
        expected_prob = 1.0 / len(DEFAULT_ACTIONS)
        for _, p, _ in dist.entries:
            assert abs(p - expected_prob) < 1e-6

    def test_select_returns_correct_count(self):
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        selector.set_context("You are a helpful assistant.", "The model failed on edge cases.")
        actions = selector.select(3, random.Random(42))
        assert len(actions) == 3
        for action in actions:
            assert action in DEFAULT_ACTIONS

    def test_select_calls_lm(self):
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        selector.set_context("You are a helpful assistant.", "Bad output.")
        selector.select(1, random.Random(42))
        assert len(lm.calls) == 1
        assert "You are selecting which edit action" in lm.calls[0]
        assert "You are a helpful assistant." in lm.calls[0]

    def test_select_without_context_falls_back(self):
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        # No set_context call.
        actions = selector.select(2, random.Random(42))
        assert len(actions) == 2
        # LM should NOT have been called.
        assert len(lm.calls) == 0

    def test_empty_actions_raises(self):
        with pytest.raises(ValueError):
            VerbalizedActionSelector([], lm=FakeLM(VALID_LM_OUTPUT))

    def test_select_without_rng_uses_instance_rng(self):
        # No context and no passed rng: falls back to the constructor rng deterministically.
        selector_a = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=FakeLM(VALID_LM_OUTPUT), rng=random.Random(3))
        selector_b = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=FakeLM(VALID_LM_OUTPUT), rng=random.Random(3))
        actions_a = selector_a.select(10)
        actions_b = selector_b.select(10)
        assert [a.name for a in actions_a] == [a.name for a in actions_b]

    def test_context_cleared_after_select(self):
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        selector.set_context("prompt", "feedback")
        selector.select(1, random.Random(42))
        # Context cleared, second call should fall back.
        selector.select(1, random.Random(42))
        assert len(lm.calls) == 1  # Only called once (first select).

    def test_case_insensitive_action_matching(self):
        output = """
<response>
<candidate>
<action>Add_Constraint</action>
<reasoning>test</reasoning>
<probability>1.0</probability>
</candidate>
</response>
"""
        lm = FakeLM(output)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=lm)
        dist = selector._parse_distribution(output, random.Random(42))
        assert len(dist.entries) == 1
        assert dist.entries[0][0].name == "add_constraint"


class TestSampleFromTails:
    def test_samples_only_from_tail(self):
        actions = DEFAULT_ACTIONS[:3]
        dist = ActionDistribution(
            entries=[
                (actions[0], 0.70, "high"),
                (actions[1], 0.25, "mid"),
                (actions[2], 0.05, "tail"),
            ]
        )
        rng = random.Random(42)
        # With tau=0.10, only the third entry (p=0.05) is in the tail.
        results = _sample_from_tails(dist, 10, tau=0.10, rng=rng)
        assert all(a.name == actions[2].name for a in results)

    def test_falls_back_when_no_tail(self):
        actions = DEFAULT_ACTIONS[:2]
        dist = ActionDistribution(
            entries=[
                (actions[0], 0.60, "high"),
                (actions[1], 0.40, "also high"),
            ]
        )
        rng = random.Random(42)
        # tau=0.10, but both entries are above 0.10.
        results = _sample_from_tails(dist, 20, tau=0.10, rng=rng)
        assert len(results) == 20
        # Should sample from both (full distribution fallback).
        names = {a.name for a in results}
        assert len(names) >= 1  # At least one action sampled.

    def test_respects_weights(self):
        actions = DEFAULT_ACTIONS[:2]
        dist = ActionDistribution(
            entries=[
                (actions[0], 0.01, "very low"),
                (actions[1], 0.09, "low"),
            ]
        )
        rng = random.Random(42)
        results = _sample_from_tails(dist, 1000, tau=0.10, rng=rng)
        count_0 = sum(1 for a in results if a.name == actions[0].name)
        count_1 = sum(1 for a in results if a.name == actions[1].name)
        # Actions[1] has 9x the weight, so it should be sampled much more often.
        assert count_1 > count_0


class TestVerbalizedReflectionIntegration:
    """Test that VerbalizedActionSelector integrates with StatelessReflectionLM."""

    def test_reflect_many_calls_set_context(self):
        action_lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=action_lm)
        reflection_lm = RecordingLM()
        reflection = StatelessReflectionLM(reflection_lm, action_selector=selector)

        candidate = {"system_prompt": "old instruction"}
        ds = _reflective_dataset(["system_prompt"])
        jobs = [(candidate, ds, ["system_prompt"])]

        results = reflection.reflect_many(jobs)

        # Action LM should have been called once (for set_context + select).
        assert len(action_lm.calls) == 1
        # The reflection prompt should contain an action constraint.
        prompt_text = reflection_lm.calls[0] if isinstance(reflection_lm.calls[0], str) else str(reflection_lm.calls[0])
        assert "--- ACTION CONSTRAINT ---" in prompt_text
        # Proposal should have action metadata.
        assert "action" in results[0][0].metadata

    def test_reflect_many_aggregates_feedback_across_jobs(self):
        action_lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(DEFAULT_ACTIONS, lm=action_lm)
        reflection = StatelessReflectionLM(BatchRecordingLM(), action_selector=selector)

        ds_a = {"sp": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "feedback-alpha"}]}
        ds_b = {"sp": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "feedback-beta"}]}
        jobs = [({"sp": "parent one"}, ds_a, ["sp"]), ({"sp": "parent two"}, ds_b, ["sp"])]

        reflection.reflect_many(jobs)

        # One selection call covers the batch, with feedback from ALL jobs.
        assert len(action_lm.calls) == 1
        assert "feedback-alpha" in action_lm.calls[0]
        assert "feedback-beta" in action_lm.calls[0]
        # Distinct parents are disclosed in the candidate context.
        assert "2 distinct parent candidates" in action_lm.calls[0]
