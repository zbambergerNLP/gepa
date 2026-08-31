# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for generic selection and canonical action-conditioned reflection."""

import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gepa.core.action_tracking import ActionDiversityCallback
from gepa.gepa_launcher import GEPAConfig, ReflectionConfig
from gepa.lm import LM
from gepa.optimize_anything import _from_legacy_config
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionLM,
    ReflectionProposal,
    StatelessReflectionLM,
)
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.response_journal import response_journal_scope
from gepa.strategies.action_space import (
    ActionDistribution,
    IncompleteActionDistributionError,
    RandomActionSelector,
    VerbalizedActionSelector,
    _sample_from_tails,
    stateless_selector_policy_contract,
)
from gepa.strategies.document_template import DocumentTemplate
from gepa.strategies.intervention import (
    SEMANTIC_ACTIONS,
    StatelessActionConstraint,
    format_stateless_action_constraint,
)


@dataclass(frozen=True)
class MenuItem:
    """Minimal structural option accepted by action selectors."""

    menu_id: str
    menu_description: str


TEST_ACTIONS = [
    MenuItem("action_a", "First selectable test option."),
    MenuItem("action_b", "Second selectable test option."),
    MenuItem("action_c", "Third selectable test option."),
    MenuItem("action_d", "Fourth selectable test option."),
    MenuItem("action_e", "Fifth selectable test option."),
    MenuItem("action_f", "Sixth selectable test option."),
]

TEST_TEMPLATE = DocumentTemplate("prompt", {"Role": "Assistant role.", "Task": "Task to perform."})
STATELESS_ACTIONS = [
    StatelessActionConstraint(action, section, TEST_TEMPLATE)
    for section in TEST_TEMPLATE.sections
    for action in SEMANTIC_ACTIONS
]
STRUCTURED_PARENT = TEST_TEMPLATE.render({"Role": "old instruction", "Task": "original task"})
REFLECTIVE_ROWS = ({"Inputs": "x", "Generated Outputs": "y", "Feedback": "bad"},)
STATELESS_LM_OUTPUT = (
    "<response>"
    + "".join(
        "<candidate>"
        f"<action>{action.menu_id}</action>"
        f"<reasoning>Reason {index}</reasoning>"
        f"<probability>{probability}</probability>"
        "</candidate>"
        for index, (action, probability) in enumerate(
            zip(STATELESS_ACTIONS[:5], [0.35, 0.30, 0.20, 0.10, 0.05], strict=True),
            start=1,
        )
    )
    + "</response>"
)

# ---------------------------------------------------------------------------
# Helpers (same patterns as test_reflection_lm.py)
# ---------------------------------------------------------------------------


class RecordingLM:
    """A fake reflection LM: records prompts, returns a fenced instruction."""

    def __init__(self, reply: str = "improved instruction"):
        """Store one fixed fenced reply and an empty prompt log.

        Args:
            reply: Instruction body returned by each call.
        """
        self.reply = reply
        self.calls: list = []

    def __call__(self, prompt):
        """Record one prompt and return the fixed fenced instruction.

        Args:
            prompt: Reflection prompt under test.

        Returns:
            Fixed instruction wrapped in a Markdown fence.
        """
        self.calls.append(prompt)
        return f"Here is the update:\n```\n{self.reply}\n```"


class BatchRecordingLM(RecordingLM):
    """A fake LM that also exposes ``batch_complete`` (like ``gepa.lm.LM``)."""

    def __init__(self, reply: str = "improved instruction"):
        """Initialize fixed replies and an empty batch-call log.

        Args:
            reply: Instruction body returned for every batch item.
        """
        super().__init__(reply)
        self.batch_calls: list[list] = []

    def batch_complete(self, messages_list, max_workers: int = 10):
        """Record one batch and return a fenced reply per conversation.

        Args:
            messages_list: Conversations in the requested batch.
            max_workers: Accepted concurrency limit.

        Returns:
            One fixed fenced instruction per conversation.
        """
        self.batch_calls.append(messages_list)
        return [f"```\n{self.reply}\n```" for _ in messages_list]


# ---------------------------------------------------------------------------
# Action selector tests
# ---------------------------------------------------------------------------


class TestRandomActionSelector:
    def test_returns_correct_count(self):
        """Verify random selection returns the requested count."""
        selector = RandomActionSelector(TEST_ACTIONS, rng=random.Random(42))
        rng = random.Random(0)  # Passed rng takes precedence over the constructor rng.
        actions = selector.select(5, rng)
        assert len(actions) == 5

    def test_membership(self):
        """Verify every sampled action belongs to the configured menu."""
        selector = RandomActionSelector(TEST_ACTIONS, rng=random.Random(42))
        rng = random.Random(0)
        actions = selector.select(20, rng)
        for action in actions:
            assert action in TEST_ACTIONS

    def test_empty_actions_raises(self):
        """Verify an empty action menu is rejected."""
        with pytest.raises(ValueError):
            RandomActionSelector([])

    def test_passed_rng_takes_precedence(self):
        """Verify a call-scoped RNG overrides the selector's RNG."""
        # Different constructor rngs, same passed rng -> identical sequences.
        selector_a = RandomActionSelector(TEST_ACTIONS, rng=random.Random(1))
        selector_b = RandomActionSelector(TEST_ACTIONS, rng=random.Random(2))
        actions_a = selector_a.select(20, random.Random(42))
        actions_b = selector_b.select(20, random.Random(42))
        assert [action.menu_id for action in actions_a] == [action.menu_id for action in actions_b]

    def test_falls_back_to_instance_rng(self):
        """Verify selection falls back to the selector-owned RNG."""
        selector_a = RandomActionSelector(TEST_ACTIONS, rng=random.Random(7))
        selector_b = RandomActionSelector(TEST_ACTIONS, rng=random.Random(7))
        actions_a = selector_a.select(20)
        actions_b = selector_b.select(20)
        assert [action.menu_id for action in actions_a] == [action.menu_id for action in actions_b]

    @pytest.mark.parametrize("selector_type", [RandomActionSelector, VerbalizedActionSelector])
    def test_duplicate_menu_ids_raise(self, selector_type):
        """Verify menu identifiers are unique case-insensitively.

        Args:
            selector_type: Random or verbalized selector constructor under test.
        """
        actions = [MenuItem("duplicate", "first"), MenuItem("DUPLICATE", "second")]
        if selector_type is VerbalizedActionSelector:
            with pytest.raises(ValueError, match="unique menu IDs"):
                selector_type(actions, lm=RecordingLM())
        else:
            with pytest.raises(ValueError, match="unique menu IDs"):
                selector_type(actions)

    @pytest.mark.parametrize("selector_type", [RandomActionSelector, VerbalizedActionSelector])
    def test_menu_id_with_surrounding_whitespace_raises(self, selector_type):
        """Verify surrounding whitespace in a menu identifier is rejected.

        Args:
            selector_type: Random or verbalized selector constructor under test.
        """
        actions = [MenuItem(" action_a ", "padded ID")]
        if selector_type is VerbalizedActionSelector:
            with pytest.raises(ValueError, match="without surrounding whitespace"):
                selector_type(actions, lm=RecordingLM())
        else:
            with pytest.raises(ValueError, match="without surrounding whitespace"):
                selector_type(actions)

    @pytest.mark.parametrize("selector_type", [RandomActionSelector, VerbalizedActionSelector])
    def test_empty_menu_id_raises(self, selector_type):
        """Verify blank menu identifiers are rejected.

        Args:
            selector_type: Random or verbalized selector constructor under test.
        """
        actions = [MenuItem(" ", "blank ID")]
        if selector_type is VerbalizedActionSelector:
            with pytest.raises(ValueError, match="non-empty menu_id"):
                selector_type(actions, lm=RecordingLM())
        else:
            with pytest.raises(ValueError, match="non-empty menu_id"):
                selector_type(actions)

    def test_stateless_selector_policy_contract_records_material_defaults(self):
        """Verify selector contracts record behavior-bearing defaults."""
        assert stateless_selector_policy_contract("random") == {
            "version": 1,
            "selector": "random",
            "selection_granularity": "batch_shared",
            "context": "none",
            "sampling": "uniform",
        }
        assert stateless_selector_policy_contract("verbalized") == {
            "version": 1,
            "selector": "verbalized",
            "selection_granularity": "batch_shared",
            "context": "first_parent_and_aggregated_feedback",
            "sampling": "tail",
            "k": 5,
            "tau": 0.2,
            "require_full_support": False,
            "exploration_epsilon": 0.0,
        }


# ---------------------------------------------------------------------------
# Reflection LM integration tests
# ---------------------------------------------------------------------------


class TestActionConditionedReflection:
    def test_action_suffix_appended_to_prompt(self):
        """Verify action conditioning appends its constraint to the prompt."""
        lm = RecordingLM()
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": STRUCTURED_PARENT}
        ds = {"system_prompt": REFLECTIVE_ROWS}

        reflection.reflect(candidate, ds, ["system_prompt"])

        # The prompt sent to the LM should contain the action suffix.
        assert len(lm.calls) == 1
        prompt_text = lm.calls[0] if isinstance(lm.calls[0], str) else str(lm.calls[0])
        assert "--- Edit constraint ---" in prompt_text

    def test_action_recorded_in_metadata(self):
        """Verify reflection metadata records the selected action."""
        lm = RecordingLM()
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": STRUCTURED_PARENT}
        ds = {"system_prompt": REFLECTIVE_ROWS}

        proposal, _ = reflection.reflect(candidate, ds, ["system_prompt"])

        assert "action" in proposal.metadata
        assert proposal.metadata["semantic_action"] == proposal.metadata["action"]
        assert proposal.metadata["action_choice"] in [action.menu_id for action in STATELESS_ACTIONS]
        assert proposal.metadata["action_operator"] in {action.edit_tool.value for action in STATELESS_ACTIONS}
        assert proposal.metadata["action_target_section"] in TEST_TEMPLATE.sections

    def test_no_action_selector_backward_compatible(self):
        """Verify omitting the selector preserves legacy reflection behavior."""
        lm = RecordingLM()
        reflection = StatelessReflectionLM(lm)
        candidate = {"system_prompt": "old instruction"}
        ds = {"system_prompt": REFLECTIVE_ROWS}

        proposal, next_lm = reflection.reflect(candidate, ds, ["system_prompt"])

        assert isinstance(proposal, ReflectionProposal)
        assert next_lm is reflection
        # No action metadata.
        assert "action" not in proposal.metadata
        # No action suffix in prompt.
        prompt_text = lm.calls[0] if isinstance(lm.calls[0], str) else str(lm.calls[0])
        assert "--- Edit constraint ---" not in prompt_text

    def test_custom_selector_keeps_the_original_two_argument_contract(self):
        """Avoid imposing built-in verbalized context keywords on custom selectors."""
        lm = RecordingLM()
        selector = MagicMock()
        selector.select.return_value = [STATELESS_ACTIONS[0]]
        reflection = StatelessReflectionLM(lm, action_selector=selector)

        reflection.reflect({"system_prompt": STRUCTURED_PARENT}, {"system_prompt": REFLECTIVE_ROWS}, ["system_prompt"])

        selector.select.assert_called_once_with(1, reflection.rng)

    def test_satisfies_protocol(self):
        """Verify the action-conditioned reflector satisfies the reflection protocol."""
        lm = RecordingLM()
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        assert isinstance(reflection, ReflectionLM)

    def test_action_conditioned_job_rejects_multiple_components(self):
        """Require one concrete component and section per selected action."""
        lm = RecordingLM()
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)
        candidate = {"system_prompt": STRUCTURED_PARENT, "user_prompt": STRUCTURED_PARENT}
        ds = {"system_prompt": REFLECTIVE_ROWS, "user_prompt": REFLECTIVE_ROWS}

        with pytest.raises(ValueError, match="exactly one component"):
            reflection.reflect(candidate, ds, ["system_prompt", "user_prompt"])

        assert lm.calls == []

    def test_action_conditioned_rewrite_sees_and_changes_only_selected_section(self):
        """Splice one returned body while preserving sibling sections exactly."""
        role_action = next(action for action in STATELESS_ACTIONS if action.target_section == "Role")
        lm = RecordingLM(reply="new role")
        reflection = StatelessReflectionLM(lm, action_selector=RandomActionSelector([role_action]))
        candidate = {"system_prompt": STRUCTURED_PARENT}

        proposal, _ = reflection.reflect(
            candidate,
            {"system_prompt": REFLECTIVE_ROWS},
            ["system_prompt"],
        )

        assert TEST_TEMPLATE.parse(proposal.new_texts["system_prompt"]) == {
            "Role": "new role",
            "Task": "original task",
        }
        prompt_text = lm.calls[0] if isinstance(lm.calls[0], str) else str(lm.calls[0])
        assert "old instruction" in prompt_text
        assert "original task" not in prompt_text
        assert "## Task" not in prompt_text
        assert "new role" in proposal.raw_lm_outputs["system_prompt"]

    def test_batch_reflect_many_assigns_actions(self):
        """Multiple jobs each get an action assigned."""
        lm = BatchRecordingLM()
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        reflection = StatelessReflectionLM(lm, action_selector=selector)

        jobs = [
            ({"sp": TEST_TEMPLATE.render({"Role": "old1"})}, {"sp": REFLECTIVE_ROWS}, ["sp"]),
            ({"sp": TEST_TEMPLATE.render({"Role": "old2"})}, {"sp": REFLECTIVE_ROWS}, ["sp"]),
            ({"sp": TEST_TEMPLATE.render({"Role": "old3"})}, {"sp": REFLECTIVE_ROWS}, ["sp"]),
        ]

        results = reflection.reflect_many(jobs)

        assert len(results) == 3
        action_ids = {action.menu_id for action in STATELESS_ACTIONS}
        for r in results:
            assert "action" in r[0].metadata
            assert r[0].metadata["action_choice"] in action_ids


# ---------------------------------------------------------------------------
# ActionDiversityCallback tests
# ---------------------------------------------------------------------------


class TestActionDiversityCallback:
    def _metadata(self, action_name: str | None, proposal_id: str = "1-0") -> dict:
        """Build proposal metadata with an optional action label.

        Args:
            action_name: Semantic action to record, or ``None``.
            proposal_id: Proposal identifier.

        Returns:
            Metadata containing the proposal ID and optional action.
        """
        if action_name is None:
            return {"proposal_id": proposal_id}
        return {"proposal_id": proposal_id, "action": action_name}

    def test_counts_proposals_per_action(self):
        """Verify proposal totals are attributed to their actions."""
        cb = ActionDiversityCallback()
        for iteration, action_name in ((1, "action_e"), (1, "action_e"), (2, "action_f")):
            cb.on_proposal_end(
                {
                    "iteration": iteration,
                    "new_instructions": {"system_prompt": f"instruction from {action_name}"},
                    "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                    "raw_lm_outputs": {"system_prompt": "raw output"},
                    "metadata": self._metadata(action_name),
                }
            )

        assert cb.action_proposal_counts["action_e"] == 2
        assert cb.action_proposal_counts["action_f"] == 1

    def test_length_capped_proposal_counts_but_adds_no_diversity_text(self) -> None:
        """Test that a fully length-capped attempt (empty new_instructions) still counts (#7).

        The attempt reaches on_proposal_end so it is not missing from the
        action's proposal total, but its empty text is excluded from the
        diversity metrics (an empty string would read as maximally dissimilar).
        """
        cb = ActionDiversityCallback()
        event = {
            "iteration": 1,
            "new_instructions": {"system_prompt": "instruction from action_e"},
            "prompts": {"system_prompt": "I provided an assistant with instructions..."},
            "raw_lm_outputs": {"system_prompt": "raw output"},
            "metadata": self._metadata("action_e"),
        }
        cb.on_proposal_end(event)
        capped = dict(event)
        capped["new_instructions"] = {}
        cb.on_proposal_end(capped)

        assert cb.action_proposal_counts["action_e"] == 2
        assert len(cb.action_texts["action_e"]) == 1
        assert len(cb._iteration_texts[1]) == 1

    def test_tracks_acceptance_rate(self):
        """Verify the callback computes per-action acceptance rates."""
        cb = ActionDiversityCallback()
        for iteration in (1, 2):
            cb.on_proposal_end(
                {
                    "iteration": iteration,
                    "new_instructions": {"system_prompt": "instruction from action_e"},
                    "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                    "raw_lm_outputs": {"system_prompt": "raw output"},
                    "metadata": self._metadata("action_e"),
                }
            )
        cb.on_candidate_accepted(
            {
                "iteration": 1,
                "new_candidate_idx": 1,
                "old_score": 0.5,
                "new_score": 0.8,
                "parent_ids": [0],
                "metadata": self._metadata("action_e"),
            }
        )
        cb.on_candidate_rejected(
            {
                "iteration": 2,
                "old_score": 0.8,
                "new_score": 0.6,
                "reason": "no improvement",
                "metadata": self._metadata("action_e"),
            }
        )

        assert cb.action_acceptance_counts["action_e"] == 1
        assert cb.action_rejection_counts["action_e"] == 1

    def test_summary_returns_expected_keys(self):
        """Verify the summary exposes every tracked metric."""
        cb = ActionDiversityCallback()
        cb.on_proposal_end(
            {
                "iteration": 1,
                "new_instructions": {"system_prompt": "instruction from action_f"},
                "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                "raw_lm_outputs": {"system_prompt": "raw output"},
                "metadata": self._metadata("action_f"),
            }
        )
        cb.on_candidate_accepted(
            {
                "iteration": 1,
                "new_candidate_idx": 1,
                "old_score": 0.5,
                "new_score": 0.9,
                "parent_ids": [0],
                "metadata": self._metadata("action_f"),
            }
        )

        s = cb.summary()
        assert "action_proposal_counts" in s
        assert "action_acceptance_counts" in s
        assert "action_rejection_counts" in s
        assert "action_acceptance_rates" in s
        assert "textual_diversity_per_iteration" in s
        assert "total_proposals" in s
        assert "total_accepted" in s

    def test_textual_diversity_computed(self):
        """Verify the callback computes sibling textual diversity."""
        cb = ActionDiversityCallback()
        # Two different proposals in the same iteration.
        for action_name in ("action_e", "action_f"):
            cb.on_proposal_end(
                {
                    "iteration": 1,
                    "new_instructions": {"system_prompt": f"instruction from {action_name}"},
                    "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                    "raw_lm_outputs": {"system_prompt": "raw output"},
                    "metadata": self._metadata(action_name),
                }
            )

        diversity = cb.textual_diversity()
        assert "1" in diversity
        # Different texts should have non-zero dissimilarity.
        assert diversity["1"] > 0.0

    def test_unconditioned_proposals_not_counted(self):
        """Verify unconditioned proposals are excluded from action totals."""
        cb = ActionDiversityCallback()
        cb.on_proposal_end(
            {
                "iteration": 1,
                "new_instructions": {"system_prompt": "instruction from unconditioned"},
                "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                "raw_lm_outputs": {"system_prompt": "raw output"},
                "metadata": self._metadata(None),
            }
        )
        cb.on_candidate_accepted(
            {
                "iteration": 1,
                "new_candidate_idx": 1,
                "old_score": 0.5,
                "new_score": 0.8,
                "parent_ids": [0],
                "metadata": self._metadata(None),
            }
        )

        assert len(cb.action_proposal_counts) == 0
        assert len(cb.action_acceptance_counts) == 0

    def test_engine_event_order_rejections_before_acceptances(self):
        """The engine fires ALL rejections before acceptances within an iteration.

        Attribution must come from each event's own metadata, not arrival order
        (a FIFO pairing would attribute B's and C's rejections to A and B here).
        """
        cb = ActionDiversityCallback()
        for action_name in ("action_a", "action_b", "action_c"):
            cb.on_proposal_end(
                {
                    "iteration": 1,
                    "new_instructions": {"system_prompt": f"instruction from {action_name}"},
                    "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                    "raw_lm_outputs": {"system_prompt": "raw output"},
                    "metadata": self._metadata(action_name),
                }
            )

        # Engine order: rejections for B and C first, then A's acceptance.
        cb.on_candidate_rejected(
            {
                "iteration": 1,
                "old_score": 0.8,
                "new_score": 0.6,
                "reason": "no improvement",
                "metadata": self._metadata("action_b"),
            }
        )
        cb.on_candidate_rejected(
            {
                "iteration": 1,
                "old_score": 0.8,
                "new_score": 0.5,
                "reason": "no improvement",
                "metadata": self._metadata("action_c"),
            }
        )
        cb.on_candidate_accepted(
            {
                "iteration": 1,
                "new_candidate_idx": 1,
                "old_score": 0.5,
                "new_score": 0.8,
                "parent_ids": [0],
                "metadata": self._metadata("action_a"),
            }
        )

        assert dict(cb.action_acceptance_counts) == {"action_a": 1}
        assert dict(cb.action_rejection_counts) == {"action_b": 1, "action_c": 1}
        assert cb.action_score_deltas["action_b"] == [pytest.approx(-0.2)]
        assert cb.action_score_deltas["action_c"] == [pytest.approx(-0.3)]

    def test_accepted_proposals_record_score_delta(self) -> None:
        """Test that accepted proposals feed action_score_deltas via the event's old_score.

        Without accepted deltas the field held only rejection outcomes (mostly
        <= 0 under strict acceptance), so it could not show which actions improve
        prompts. Both accept and reject now contribute a signed delta.
        """
        cb = ActionDiversityCallback()
        cb.on_proposal_end(
            {
                "iteration": 1,
                "new_instructions": {"system_prompt": "instruction from action_e"},
                "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                "raw_lm_outputs": {"system_prompt": "raw output"},
                "metadata": self._metadata("action_e"),
            }
        )
        cb.on_candidate_accepted(
            {
                "iteration": 1,
                "new_candidate_idx": 1,
                "old_score": 0.5,
                "new_score": 0.9,
                "parent_ids": [0],
                "metadata": self._metadata("action_e"),
            }
        )

        assert cb.action_acceptance_counts["action_e"] == 1
        assert cb.action_score_deltas["action_e"] == [pytest.approx(0.4)]

    def test_accepted_event_without_old_score_tolerated(self) -> None:
        """Test that a synthetic accepted event lacking old_score still counts and records no delta."""
        cb = ActionDiversityCallback()
        cb.on_proposal_end(
            {
                "iteration": 1,
                "new_instructions": {"system_prompt": "instruction from action_e"},
                "prompts": {"system_prompt": "I provided an assistant with instructions..."},
                "raw_lm_outputs": {"system_prompt": "raw output"},
                "metadata": self._metadata("action_e"),
            }
        )
        cb.on_candidate_accepted(
            {
                "iteration": 1,
                "new_candidate_idx": 1,
                "new_score": 0.9,
                "parent_ids": [0],
                "metadata": {"action": "action_e"},
            }
        )

        assert cb.action_acceptance_counts["action_e"] == 1
        assert cb.action_score_deltas["action_e"] == []

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

    def test_state_round_trip_restores_metrics_and_selector_history(self) -> None:
        """Persist complete action evidence without aliasing mutable selector history."""
        selector = SimpleNamespace(history=[{"sampled": ["action_e"]}])
        callback = ActionDiversityCallback(selector=selector)
        callback.on_proposal_end(
            {
                "iteration": 4,
                "new_instructions": {"system_prompt": "revised instruction"},
                "prompts": {"system_prompt": "proposal prompt"},
                "raw_lm_outputs": {"system_prompt": "raw output"},
                "metadata": self._metadata("action_e"),
            }
        )
        callback.on_candidate_accepted(
            {
                "iteration": 4,
                "new_candidate_idx": 1,
                "old_score": 0.25,
                "new_score": 0.75,
                "parent_ids": [0],
                "metadata": self._metadata("action_e"),
            }
        )

        checkpoint = callback.get_state()
        selector.history.append({"sampled": ["action_f"]})
        restored_selector = SimpleNamespace(history=[])
        restored = ActionDiversityCallback(selector=restored_selector)
        restored.set_state(checkpoint)

        assert restored.summary() == callback.summary()
        assert restored.action_texts == callback.action_texts
        assert restored_selector.history == [{"sampled": ["action_e"]}]

    def test_invalid_persisted_selector_history_is_rejected(self) -> None:
        """Reject a malformed selector-history snapshot instead of losing evidence silently."""
        callback = ActionDiversityCallback(selector=SimpleNamespace(history=[]))
        checkpoint = callback.get_state()
        checkpoint["selector_history"] = {"sampled": ["action_e"]}

        with pytest.raises(TypeError, match="selector_history"):
            callback.set_state(checkpoint)


# ---------------------------------------------------------------------------
# Verbalized action selector tests
# ---------------------------------------------------------------------------

VALID_LM_OUTPUT = """
<response>
<candidate>
<action>action_e</action>
<reasoning>The feedback shows edge cases being missed</reasoning>
<probability>0.35</probability>
</candidate>
<candidate>
<action>action_b</action>
<reasoning>The prompt is too vague for multi-hop reasoning</reasoning>
<probability>0.30</probability>
</candidate>
<candidate>
<action>action_a</action>
<reasoning>A worked example would help</reasoning>
<probability>0.20</probability>
</candidate>
<candidate>
<action>action_f</action>
<reasoning>Reordering might improve attention</reasoning>
<probability>0.10</probability>
</candidate>
<candidate>
<action>action_c</action>
<reasoning>Could refine the persona</reasoning>
<probability>0.05</probability>
</candidate>
</response>
"""


class FakeLM:
    """A fake LM that returns a fixed response."""

    def __init__(self, response: str):
        """Store one fixed selector response and an empty prompt log.

        Args:
            response: Text returned by every call.
        """
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt):
        """Record one prompt and return the configured response.

        Args:
            prompt: Verbalized-sampling prompt.

        Returns:
            Fixed selector response.
        """
        self.calls.append(prompt)
        return self.response


class TestVerbalizedActionSelector:
    def test_parse_valid_distribution(self):
        """Verify the selector parses a complete valid distribution."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        dist = selector._parse_distribution(VALID_LM_OUTPUT, random.Random(42))
        assert len(dist.entries) == 5
        # Probabilities should be renormalized to sum to 1.
        assert abs(sum(probability for _, probability, _ in dist.entries) - 1.0) < 1e-6

    def test_parse_malformed_xml_falls_back(self):
        """Verify malformed distribution markup triggers uniform fallback."""
        lm = FakeLM("this is not xml at all")
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        dist = selector._parse_distribution("this is not xml at all", random.Random(42))
        # Should fall back to uniform over all actions.
        assert len(dist.entries) == len(TEST_ACTIONS)
        expected_prob = 1.0 / len(TEST_ACTIONS)
        for _, p, _ in dist.entries:
            assert abs(p - expected_prob) < 1e-6

    def test_parse_missing_probability_skips_entry(self):
        """Verify entries without probabilities are ignored."""
        partial_output = """
<response>
<candidate>
<action>action_e</action>
<reasoning>good</reasoning>
<probability>0.60</probability>
</candidate>
<candidate>
<action>action_f</action>
<reasoning>no probability here</reasoning>
</candidate>
</response>
"""
        lm = FakeLM(partial_output)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        dist = selector._parse_distribution(partial_output, random.Random(42))
        assert len(dist.entries) == 1
        assert dist.entries[0][0].menu_id == "action_e"

    def test_required_full_support_falls_back_on_an_incomplete_distribution(self) -> None:
        """Never treat a shortlist as a distribution over the declared menu."""
        selector = VerbalizedActionSelector(
            TEST_ACTIONS,
            lm=FakeLM(VALID_LM_OUTPUT),
            require_full_support=True,
        )
        dist = selector._parse_distribution(VALID_LM_OUTPUT, random.Random(42))
        assert dist.is_fallback is True
        assert len(dist.entries) == len(TEST_ACTIONS)
        assert len({action.menu_id for action, _, _ in dist.entries}) == len(TEST_ACTIONS)

    def test_required_full_support_retries_an_incomplete_model_response(self) -> None:
        """Use a complete second distribution instead of sampling the parser fallback."""
        probability = 1.0 / len(TEST_ACTIONS)
        complete_output = (
            "<response>"
            + "".join(
                "<candidate>"
                f"<action>{action.menu_id}</action><reasoning>test</reasoning>"
                f"<probability>{probability}</probability>"
                "</candidate>"
                for action in TEST_ACTIONS
            )
            + "</response>"
        )
        lm = MagicMock(side_effect=["not a distribution", complete_output])
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm, require_full_support=True)

        selected = selector.select(1, candidate="component", feedback_summary="feedback")

        assert len(selected) == 1
        assert lm.call_count == 2
        assert selector.history[0]["fallback"] is False

    def test_required_full_support_rejects_two_incomplete_model_responses(self) -> None:
        """Leave selection unresolved when the model never supplies its judgment."""
        lm = MagicMock(return_value="not a distribution")
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm, require_full_support=True)

        with pytest.raises(IncompleteActionDistributionError, match="after two attempts"):
            selector.select(1, candidate="component", feedback_summary="feedback")

        assert lm.call_count == 2
        assert selector.history == []

    @pytest.mark.parametrize("probability", ["nan", "inf", "-0.1"])
    def test_invalid_numeric_probability_falls_back_uniformly(self, probability: str) -> None:
        """Reject non-finite and negative weights before sampling or logging.

        Args:
            probability: Invalid probability text placed in the model response.
        """
        output = (
            "<response><candidate><action>action_e</action><reasoning>x</reasoning>"
            f"<probability>{probability}</probability></candidate>"
            "<candidate><action>action_f</action><reasoning>y</reasoning>"
            "<probability>1.0</probability></candidate></response>"
        )
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(output))
        dist = selector._parse_distribution(output, random.Random(42))
        assert dist.is_fallback is True
        assert [probability for _, probability, _ in dist.entries] == pytest.approx(
            [1.0 / len(TEST_ACTIONS)] * len(TEST_ACTIONS)
        )

    def test_zero_mass_distribution_falls_back_uniformly(self) -> None:
        """Never pass a zero-total verbalized distribution to the sampler."""
        output = """
<response>
<candidate><action>action_e</action><probability>0</probability></candidate>
<candidate><action>action_f</action><probability>0</probability></candidate>
</response>
"""
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(output))
        dist = selector._parse_distribution(output, random.Random(42))
        assert dist.is_fallback is True

    def test_parse_unknown_action_ignored(self):
        """Verify unknown action identifiers are ignored."""
        bad_action_output = """
<response>
<candidate>
<action>nonexistent_action</action>
<reasoning>doesn't exist</reasoning>
<probability>0.50</probability>
</candidate>
<candidate>
<action>action_e</action>
<reasoning>real action</reasoning>
<probability>0.50</probability>
</candidate>
</response>
"""
        lm = FakeLM(bad_action_output)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        dist = selector._parse_distribution(bad_action_output, random.Random(42))
        assert len(dist.entries) == 1
        assert dist.entries[0][0].menu_id == "action_e"

    def test_select_returns_correct_count(self):
        """Verify selection returns the requested number of actions."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        actions = selector.select(
            3,
            random.Random(42),
            candidate="You are a helpful assistant.",
            feedback_summary="The model failed on edge cases.",
        )
        assert len(actions) == 3
        for action in actions:
            assert action in TEST_ACTIONS

    def test_select_calls_lm(self):
        """Verify selection consults the language model."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        selector.select(
            1,
            random.Random(42),
            candidate="You are a helpful assistant.",
            feedback_summary="Bad output.",
        )
        assert len(lm.calls) == 1
        assert "Choose edit actions that address" in lm.calls[0]
        assert "You are a helpful assistant." in lm.calls[0]

    def test_select_without_context_falls_back(self):
        """Verify missing selection context triggers fallback."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        actions = selector.select(2, random.Random(42))
        assert len(actions) == 2
        # LM should NOT have been called.
        assert len(lm.calls) == 0

    def test_empty_actions_raises(self):
        """Verify an empty verbalized action menu is rejected."""
        with pytest.raises(ValueError):
            VerbalizedActionSelector([], lm=FakeLM(VALID_LM_OUTPUT))

    def test_select_without_rng_uses_instance_rng(self):
        """Verify selection defaults to the selector-owned RNG."""
        # No context and no passed rng: falls back to the constructor rng deterministically.
        selector_a = VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(VALID_LM_OUTPUT), rng=random.Random(3))
        selector_b = VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(VALID_LM_OUTPUT), rng=random.Random(3))
        actions_a = selector_a.select(10)
        actions_b = selector_b.select(10)
        assert [action.menu_id for action in actions_a] == [action.menu_id for action in actions_b]

    def test_context_is_scoped_to_one_select_call(self):
        """Verify selection context is explicit and does not persist between calls."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        selector.select(
            1,
            random.Random(42),
            candidate="prompt",
            feedback_summary="feedback",
        )
        selector.select(1, random.Random(42))
        assert len(lm.calls) == 1

    def test_case_insensitive_action_matching(self):
        """Verify action identifiers are matched case-insensitively."""
        output = """
<response>
<candidate>
<action>ACTION_E</action>
<reasoning>test</reasoning>
<probability>1.0</probability>
</candidate>
</response>
"""
        lm = FakeLM(output)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm)
        dist = selector._parse_distribution(output, random.Random(42))
        assert len(dist.entries) == 1
        assert dist.entries[0][0].menu_id == "action_e"


class TestSampleFromTails:
    def test_samples_only_from_tail(self):
        """Verify weighted sampling draws only from the selected tail."""
        actions = TEST_ACTIONS[:3]
        dist = ActionDistribution(
            entries=[
                (actions[0], 0.70, "high"),
                (actions[1], 0.25, "mid"),
                (actions[2], 0.05, "tail"),
            ]
        )
        rng = random.Random(42)
        # With tau=0.10, only the third entry (p=0.05) is in the tail.
        results, stats = _sample_from_tails(dist, 10, tau=0.10, rng=rng)
        assert all(action.menu_id == actions[2].menu_id for action in results)
        assert stats.n_parsed_entries == 3
        assert stats.used_full_fallback is False
        assert stats.tail_mass == pytest.approx(0.05)

    def test_falls_back_when_no_tail(self):
        """Verify an empty tail falls back to the full distribution."""
        actions = TEST_ACTIONS[:2]
        dist = ActionDistribution(
            entries=[
                (actions[0], 0.60, "high"),
                (actions[1], 0.40, "also high"),
            ]
        )
        rng = random.Random(42)
        # tau=0.10, but both entries are above 0.10.
        results, stats = _sample_from_tails(dist, 20, tau=0.10, rng=rng)
        assert len(results) == 20
        # Should sample from both (full distribution fallback).
        menu_ids = {action.menu_id for action in results}
        assert len(menu_ids) >= 1  # At least one action sampled.
        assert stats.used_full_fallback is True
        assert stats.tail_mass == pytest.approx(0.0)

    def test_respects_weights(self):
        """Verify tail sampling respects relative weights."""
        actions = TEST_ACTIONS[:2]
        dist = ActionDistribution(
            entries=[
                (actions[0], 0.01, "very low"),
                (actions[1], 0.09, "low"),
            ]
        )
        rng = random.Random(42)
        results, _stats = _sample_from_tails(dist, 1000, tau=0.10, rng=rng)
        count_0 = sum(1 for action in results if action.menu_id == actions[0].menu_id)
        count_1 = sum(1 for action in results if action.menu_id == actions[1].menu_id)
        # Actions[1] has 9x the weight, so it should be sampled much more often.
        assert count_1 > count_0


class TestTauDefault:
    """Test cases for VerbalizedActionSelector tail-threshold defaulting."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "k",
            "expected_tau",
        ],
        # Parameter values
        [
            pytest.param(
                5,  # k
                0.2,  # expected_tau
                id="k_5_gives_one_fifth",
            ),
            pytest.param(
                4,  # k
                0.25,  # expected_tau
                id="k_4_gives_one_quarter",
            ),
        ],
    )
    def test_tau_defaults_to_reciprocal_k(self, k: int, expected_tau: float) -> None:
        """Test that tau defaults to the reciprocal of k when not given explicitly.

        Args:
            k: The number of actions the selector draws per selection.
            expected_tau: The tail threshold tau must resolve to.
        """
        assert VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(""), k=k).tau == pytest.approx(expected_tau)

    def test_explicit_tau_overrides_default(self) -> None:
        """Test that an explicit tau overrides the reciprocal-k default."""
        assert VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(""), k=5, tau=0.1).tau == pytest.approx(0.1)


class TestVerbalizedReflectionIntegration:
    """Test that VerbalizedActionSelector integrates with StatelessReflectionLM."""

    def test_reflect_many_passes_context_to_selector(self):
        """Verify batched reflection supplies context to the selector."""
        action_lm = FakeLM(STATELESS_LM_OUTPUT)
        selector = VerbalizedActionSelector(STATELESS_ACTIONS, lm=action_lm)
        reflection_lm = RecordingLM()
        reflection = StatelessReflectionLM(reflection_lm, action_selector=selector)

        candidate = {"system_prompt": STRUCTURED_PARENT}
        ds = {"system_prompt": REFLECTIVE_ROWS}
        jobs = [(candidate, ds, ["system_prompt"])]

        results = reflection.reflect_many(jobs)

        # Action LM should have been called once with the batch context.
        assert len(action_lm.calls) == 1
        # The reflection prompt should contain an action constraint.
        prompt_text = reflection_lm.calls[0] if isinstance(reflection_lm.calls[0], str) else str(reflection_lm.calls[0])
        assert "--- Edit constraint ---" in prompt_text
        # Proposal should have action metadata.
        assert "action" in results[0][0].metadata

    def test_reflect_many_aggregates_feedback_across_jobs(self):
        """Verify batch-level selection aggregates feedback across jobs."""
        action_lm = FakeLM(STATELESS_LM_OUTPUT)
        selector = VerbalizedActionSelector(STATELESS_ACTIONS, lm=action_lm)
        reflection = StatelessReflectionLM(BatchRecordingLM(), action_selector=selector)

        ds_a = {"sp": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "feedback-alpha"}]}
        ds_b = {"sp": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "feedback-beta"}]}
        jobs = [
            ({"sp": TEST_TEMPLATE.render({"Role": "parent one"})}, ds_a, ["sp"]),
            ({"sp": TEST_TEMPLATE.render({"Role": "parent two"})}, ds_b, ["sp"]),
        ]

        reflection.reflect_many(jobs)

        # One selection call covers the batch, with feedback from ALL jobs.
        assert len(action_lm.calls) == 1
        assert "feedback-alpha" in action_lm.calls[0]
        assert "feedback-beta" in action_lm.calls[0]
        # Distinct parents are disclosed in the candidate context.
        assert "2 distinct parent candidates" in action_lm.calls[0]

    def test_invalid_section_body_drops_only_that_proposal(self) -> None:
        """Keep valid siblings when one selected-action output includes a header."""

        class MixedBatchLM:
            """Return one invalid section body followed by one valid body."""

            def batch_complete(self, messages_list) -> list[str]:
                """Return one response for each of the two expected prompts.

                Args:
                    messages_list: Batched reflection prompts.

                Returns:
                    Invalid and valid section-body responses in job order.
                """
                assert len(messages_list) == 2
                return ["```\n## Role\nfull section returned\n```", "```\nvalid revision\n```"]

        selector = MagicMock()
        selector.select.return_value = [STATELESS_ACTIONS[0], STATELESS_ACTIONS[1]]
        logger = MagicMock()
        reflection = StatelessReflectionLM(MixedBatchLM(), logger=logger, action_selector=selector)
        jobs = [
            (
                {"sp": TEST_TEMPLATE.render({"Role": "parent one"})},
                {"sp": [{"Feedback": "feedback-alpha"}]},
                ["sp"],
            ),
            (
                {"sp": TEST_TEMPLATE.render({"Role": "parent two"})},
                {"sp": [{"Feedback": "feedback-beta"}]},
                ["sp"],
            ),
        ]

        results = reflection.reflect_many(jobs)

        first, second = (proposal for proposal, _strategy in results)
        assert first.new_texts == {}
        assert first.metadata["action_choice"] == STATELESS_ACTIONS[0].menu_id
        assert second.new_texts["sp"] != jobs[1][0]["sp"]
        assert second.metadata["action_choice"] == STATELESS_ACTIONS[1].menu_id
        selector.select.assert_called_once()
        assert "invalid section body" in logger.log.call_args.args[0]

    def test_per_job_selection_scopes_context_to_each_job(self) -> None:
        """Test that opt-in per-job selection scopes each selector call to its own job.

        Per-job mode makes one selector call per job, each drawn from that job's own
        candidate and feedback, with no cross-job aggregation.
        """
        action_lm = FakeLM(STATELESS_LM_OUTPUT)
        selector = VerbalizedActionSelector(STATELESS_ACTIONS, lm=action_lm)
        reflection = StatelessReflectionLM(BatchRecordingLM(), action_selector=selector, per_job_action_selection=True)

        ds_a = {"sp": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "feedback-alpha"}]}
        ds_b = {"sp": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "feedback-beta"}]}
        jobs = [
            ({"sp": TEST_TEMPLATE.render({"Role": "parent one"})}, ds_a, ["sp"]),
            ({"sp": TEST_TEMPLATE.render({"Role": "parent two"})}, ds_b, ["sp"]),
        ]

        reflection.reflect_many(jobs)

        # One selector call per job, each scoped to its own candidate + feedback.
        assert len(action_lm.calls) == 2
        assert "parent one" in action_lm.calls[0]
        assert "feedback-alpha" in action_lm.calls[0]
        assert "feedback-beta" not in action_lm.calls[0]
        assert "parent two" in action_lm.calls[1]
        assert "feedback-beta" in action_lm.calls[1]
        assert "feedback-alpha" not in action_lm.calls[1]
        # No batch-aggregation disclosure in per-job mode.
        assert "distinct parent candidates" not in action_lm.calls[0]

    def test_failed_batch_reuses_selected_actions_and_replays_after_restart(self, tmp_path: Path) -> None:
        """Keep one aggregated Controller decision through fallback and restart.

        Args:
            tmp_path: Pytest directory containing the response journal.
        """
        journal_path = tmp_path / "responses.sqlite3"

        class FallbackReflectionClient:
            """Expose a failing batch transport over a journaled plain client."""

            def __init__(self, inner: LM):
                """Store the journaled client and initialize a batch-call count.

                Args:
                    inner: LM used for individual fallback requests.
                """
                self.inner = inner
                self.batch_calls = 0

            def batch_complete(self, messages_list) -> list[str]:
                """Simulate an optional batch transport failure.

                Args:
                    messages_list: Batched messages that would have been sent.

                Raises:
                    RuntimeError: Always, so stateless reflection reuses the
                        selected actions through its individual path.
                """
                self.batch_calls += 1
                raise RuntimeError(f"batch transport failed for {len(messages_list)} requests")

            def __call__(self, prompt) -> str:
                """Complete one fallback prompt through the durable client.

                Args:
                    prompt: Rendered reflection prompt.

                Returns:
                    Journaled reflection response text.
                """
                return self.inner(prompt)

        def provider_response(content: str) -> MagicMock:
            """Build one LiteLLM-shaped response with fixed provider identity.

            Args:
                content: Assistant text returned by the fake provider.

            Returns:
                Completion response accepted by :class:`gepa.lm.LM`.
            """
            response = MagicMock()
            response.model = "runtime-model"
            response.system_fingerprint = "fp-fixed"
            response.usage = None
            response.choices = [MagicMock()]
            response.choices[0].finish_reason = "stop"
            response.choices[0].message.content = content
            return response

        def make_strategy() -> tuple[
            StatelessReflectionLM,
            VerbalizedActionSelector,
            FallbackReflectionClient,
        ]:
            """Build fresh journaled Controller and reflection clients.

            Returns:
                Stateless reflector, its selector, and batch-failing client.
            """
            controller_lm = LM(
                "test/controller",
                response_journal_path=str(journal_path),
                response_journal_namespace="stateless-controller",
            )
            selector = VerbalizedActionSelector(STATELESS_ACTIONS, lm=controller_lm)
            reflection_client = FallbackReflectionClient(
                LM(
                    "test/reflection",
                    response_journal_path=str(journal_path),
                    response_journal_namespace="reflection-proposer",
                )
            )
            return (
                StatelessReflectionLM(reflection_client, action_selector=selector),
                selector,
                reflection_client,
            )

        jobs = [
            (
                {"sp": TEST_TEMPLATE.render({"Role": "parent one"})},
                {"sp": [{"Feedback": "feedback-alpha"}]},
                ["sp"],
            ),
            (
                {"sp": TEST_TEMPLATE.render({"Role": "parent two"})},
                {"sp": [{"Feedback": "feedback-beta"}]},
                ["sp"],
            ),
        ]
        initial, initial_selector, initial_client = make_strategy()
        with (
            patch(
                "litellm.completion",
                side_effect=[
                    provider_response(STATELESS_LM_OUTPUT),
                    provider_response("```\nfirst revision\n```"),
                    provider_response("```\nsecond revision\n```"),
                ],
            ) as provider,
            response_journal_scope("optimizer-iteration-21"),
        ):
            initial_results = initial.reflect_many(jobs)

        controller_calls = [
            call for call in provider.call_args_list if call.kwargs["model"] == "test/controller"
        ]
        assert len(controller_calls) == 1
        assert provider.call_count == 3
        assert initial_client.batch_calls == 1
        assert len(initial_selector.history) == 1
        assert all(proposal.new_texts for proposal, _strategy in initial_results)

        resumed, resumed_selector, resumed_client = make_strategy()
        with (
            patch("litellm.completion") as resumed_provider,
            response_journal_scope("optimizer-iteration-21"),
        ):
            replayed_results = resumed.reflect_many(jobs)

        assert [proposal.new_texts for proposal, _strategy in replayed_results] == [
            proposal.new_texts for proposal, _strategy in initial_results
        ]
        assert [proposal.metadata for proposal, _strategy in replayed_results] == [
            proposal.metadata for proposal, _strategy in initial_results
        ]
        assert resumed_selector.history == initial_selector.history
        assert resumed_client.batch_calls == 1
        resumed_provider.assert_not_called()


class TestVerbalizedHistory:
    def test_history_records_distribution_and_samples(self):
        """Verify selector history records distributions and sampled actions."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm, rng=random.Random(0))
        picks = selector.select(3, candidate="prompt text", feedback_summary="feedback text")

        assert len(selector.history) == 1
        record = selector.history[0]
        assert record["fallback"] is False
        assert record["sampled"] == [pick.menu_id for pick in picks]
        assert len(record["sampled_probabilities"]) == len(picks)
        assert all(0.0 <= probability <= 1.0 for probability in record["sampled_probabilities"])
        assert abs(sum(record["probs"].values()) - 1.0) < 1e-6
        assert record["sampling_policy"] == "tail"
        assert record["exploration_epsilon"] == 0.0

    def test_full_support_policy_preserves_controller_zero_probabilities(self) -> None:
        """Explore within the Controller's positive support without reviving rejected actions."""
        actions = TEST_ACTIONS[:3]
        output = (
            "<response>"
            + "".join(
                "<candidate>"
                f"<action>{action.menu_id}</action><reasoning>x</reasoning><probability>{probability}</probability>"
                "</candidate>"
                for action, probability in zip(actions, [1.0, 0.0, 0.0], strict=True)
            )
            + "</response>"
        )
        selector = VerbalizedActionSelector(
            actions,
            lm=FakeLM(output),
            rng=random.Random(0),
            require_full_support=True,
        )
        selected = selector.select(100, candidate="component", feedback_summary="feedback")

        record = selector.history[0]
        assert record["sampling_policy"] == "positive_support_uniform_mixture"
        assert record["exploration_epsilon"] == pytest.approx(0.1)
        assert record["sampling_probs"] == pytest.approx(
            {
                actions[0].menu_id: 1.0,
                actions[1].menu_id: 0.0,
                actions[2].menu_id: 0.0,
            }
        )
        assert set(record["probs"]) == {action.menu_id for action in actions}
        assert {action.menu_id for action in selected} == {actions[0].menu_id}
        assert sum(record["sampling_probs"].values()) == pytest.approx(1.0)

    def test_history_records_tail_sample_stats(self) -> None:
        """Test that a history record carries the tail-sampling diagnostics."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm, rng=random.Random(0))
        selector.select(3, candidate="prompt text", feedback_summary="feedback text")

        record = selector.history[0]
        assert record["n_parsed_entries"] == len(record["probs"])
        assert record["used_full_fallback"] in (True, False)
        assert record["entropy_bits"] >= 0.0
        assert 0.0 <= record["tail_mass"] <= 1.0
        assert record["tau"] == pytest.approx(selector.tau)

    def test_history_marks_fallback_on_unparseable_output(self):
        """Verify selector history marks parse fallback explicitly."""
        lm = FakeLM("no xml here")
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm, rng=random.Random(0))
        selector.select(2, candidate="prompt text", feedback_summary="feedback text")

        assert selector.history[0]["fallback"] is True

    def test_uniform_fallback_without_context_leaves_no_history(self):
        """Verify context-free uniform fallback does not fabricate LM history."""
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=FakeLM(VALID_LM_OUTPUT), rng=random.Random(0))
        selector.select(2)
        assert selector.history == []


# ---------------------------------------------------------------------------
# Length control: soft budget, length-aware selection, hard cap
# ---------------------------------------------------------------------------


class TestLengthControl:
    def test_suffix_includes_length_budget(self):
        """Verify the action suffix states the active length budget."""
        from gepa.strategies.action_space import SOFT_PROMPT_CHAR_BUDGET

        suffix = format_stateless_action_constraint(STATELESS_ACTIONS[0])
        assert f"under {SOFT_PROMPT_CHAR_BUDGET} characters" in suffix

    def test_verbalized_prompt_includes_length_stats(self):
        """Verify verbalized sampling receives current length statistics."""
        lm = FakeLM(VALID_LM_OUTPUT)
        selector = VerbalizedActionSelector(TEST_ACTIONS, lm=lm, rng=random.Random(0))
        selector.select(1, candidate="p" * 1234, feedback_summary="some feedback")
        assert "Current component length: 1234 characters" in lm.calls[0]
        assert "favor actions that shorten or replace existing text" in lm.calls[0]


class TestConfigWiring:
    """Keep ``action_selector`` wired through the public configuration path."""

    def test_reflection_config_accepts_action_selector(self):
        """Verify reflection configuration accepts an action selector."""
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        config = ReflectionConfig(reflection_lm="m", action_selector=selector)
        assert config.action_selector is selector
        assert ReflectionConfig(reflection_lm="m").action_selector is None

    def test_action_selector_survives_legacy_config_conversion(self):
        """Verify legacy configuration conversion preserves the selector."""
        selector = RandomActionSelector(STATELESS_ACTIONS, rng=random.Random(0))
        legacy = GEPAConfig(reflection=ReflectionConfig(reflection_lm="m", action_selector=selector))
        rebuilt = GEPAConfig(**_from_legacy_config(legacy).engine_config)
        assert rebuilt.reflection.action_selector is selector

    def test_reflective_mutation_proposer_accepts_action_selector(self):
        """Verify reflective proposer construction accepts the selector."""
        assert "action_selector" in inspect.signature(ReflectiveMutationProposer.__init__).parameters

    def test_stateless_reflection_lm_bind_rng_seeds_selection(self) -> None:
        """Test that bind_rng rebinds the reflection LM's selection rng."""
        run_rng = random.Random(1234)
        reflection = StatelessReflectionLM(
            RecordingLM(reply="x"), action_selector=RandomActionSelector(STATELESS_ACTIONS)
        )
        assert reflection.rng is not run_rng
        reflection.bind_rng(run_rng)
        assert reflection.rng is run_rng

    def test_proposer_binds_run_rng_to_default_reflection_lm(self) -> None:
        """Test that the proposer binds the run rng onto its default reflection LM."""
        adapter = type("_DummyAdapter", (), {"propose_new_texts": None})()
        run_rng = random.Random(1234)
        proposer = ReflectiveMutationProposer(
            logger=None,
            trainset=[{"x": 1}],
            adapter=adapter,
            candidate_selector=None,
            module_selector=None,
            batch_sampler=None,
            perfect_score=None,
            skip_perfect_score=False,
            experiment_tracker=None,
            reflection_lm=RecordingLM(reply="x"),
            action_selector=RandomActionSelector(STATELESS_ACTIONS),
        )
        proposer.bind_reflection_rng(run_rng)
        assert proposer._reflection_lm.rng is run_rng
