# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for POSIT-style semantic-action manifestation and role routing."""

import pytest

from gepa.proposer.reflective_mutation.manifestor import (
    INJECTION_SITE_DESCRIPTIONS,
    MAX_INTERVENTION_CHARS,
    MAX_TRACES_CHARS,
    ManifestationError,
    Manifestor,
    infer_manifestor_injection_site,
)
from gepa.strategies.document_template import EditTarget
from gepa.strategies.edit_tools import EditTool
from gepa.strategies.intervention import ControllerAction, Intervention, InterventionSpec

SPEC = InterventionSpec(
    name="expand",
    description="Expand the region with missing guidance.",
    compatible_tools=(EditTool.INSERT_TEXT,),
    instruction="Name the missing guidance and where the editor should add it.",
)
FIXED_SPEC = InterventionSpec(
    name="fixed",
    description="Use fixed steering.",
    compatible_tools=(EditTool.INSERT_TEXT,),
    fixed_text="Add one concise requirement.",
    inject_as="system",
)


class RecordingLM:
    """Record manifestation prompts and return a fixed steering message."""

    def __init__(self, reply: str = "The failures omit citations, so add a citation requirement."):
        """Store the fixed reply."""
        self.reply = reply
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record and answer one Manifestor prompt."""
        self.calls.append(prompt)
        return self.reply


class SequenceLM:
    """Return one scripted reply per manifestation attempt."""

    def __init__(self, replies: list[str]):
        """Store scripted replies."""
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record the prompt and return the next reply."""
        self.calls.append(prompt)
        return self.replies[len(self.calls) - 1]


def action(spec: InterventionSpec | None = SPEC) -> ControllerAction:
    """Build a Rules-section Controller action."""
    tool = spec.edit_tool if spec is not None else None
    return ControllerAction(EditTarget("sys", "Rules"), tool, spec)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param("openai/gpt-5.6", "developer", id="openai_prefix"),
        pytest.param("gpt-4.1", "developer", id="bare_gpt"),
        pytest.param("openai/o3", "developer", id="openai_o_series"),
        pytest.param("anthropic/claude-sonnet-4-5", "user", id="claude"),
        pytest.param("claude-opus-4-1", "user", id="bare_claude"),
        pytest.param("deepseek/deepseek-v4-flash", "user", id="portable_fallback"),
        pytest.param(None, "user", id="custom_callable_fallback"),
    ],
)
def test_provider_routing_uses_developer_for_openai_and_user_elsewhere(
    model: str | None,
    expected: str,
) -> None:
    """Apply the meeting's provider-specific steering-role decision."""
    assert infer_manifestor_injection_site(model) == expected


def test_level1_action_skips_manifestation() -> None:
    """Avoid an LM call when the Controller selected only an atomic region."""
    lm = RecordingLM()
    assert Manifestor(lm).manifest(action(None), "rules", "full", "feedback", "traces") is None
    assert lm.calls == []


def test_fixed_text_skips_lm_and_honors_provider_override() -> None:
    """Route even fixed steering through the provider-resolved role."""
    lm = RecordingLM()
    result = Manifestor(lm, inject_as="developer").manifest(
        action(FIXED_SPEC),
        "rules",
        "full",
        "feedback",
        "traces",
    )
    assert result == Intervention("Add one concise requirement.", "developer")
    assert lm.calls == []


def test_instruction_spec_is_manifested_once_with_full_grounding() -> None:
    """Give the Manifestor the action, document, selected region, feedback, and traces."""
    lm = RecordingLM()
    result = Manifestor(lm, inject_as="user").manifest(
        action(),
        region_text="- be accurate",
        full_text="## Role\nhelper\n\n## Rules\n- be accurate",
        feedback_summary="The answer omitted its source.",
        traces="Output: unsupported claim",
    )
    assert result == Intervention(lm.reply, "user")
    assert len(lm.calls) == 1
    prompt = lm.calls[0]
    for expected in (
        SPEC.name,
        SPEC.description,
        SPEC.instruction,
        EditTool.INSERT_TEXT.value,
        "Rules",
        "## Role\nhelper",
        "- be accurate",
        "The answer omitted its source.",
        "Output: unsupported claim",
        INJECTION_SITE_DESCRIPTIONS["user"],
    ):
        assert expected in prompt
    assert "do not write the edited text yourself" in prompt.lower()


@pytest.mark.parametrize("site", ["assistant_reasoning", "user", "system", "developer"])
def test_prompt_explains_the_effective_injection_site(site: str) -> None:
    """Make the manifested text's voice match the role that receives it."""
    lm = RecordingLM()
    Manifestor(lm, inject_as=site).manifest(action(), "region", "full", "feedback", "traces")
    assert INJECTION_SITE_DESCRIPTIONS[site] in lm.calls[0]


def test_hidden_thoughts_and_overlong_manifestation_are_removed() -> None:
    """Keep steering concise and free of hidden-reasoning wrappers."""
    lm = RecordingLM(f"<think>private</think>{'x' * (MAX_INTERVENTION_CHARS + 50)}")
    result = Manifestor(lm).manifest(action(), "region", "full", "feedback", "traces")
    assert result is not None
    assert result.text == "x" * MAX_INTERVENTION_CHARS + "..."
    assert "private" not in result.text


def test_empty_visible_manifestation_is_retried_once() -> None:
    """Recover from a think-only first reply without sending empty steering."""
    lm = SequenceLM(["<think>private only</think>", "Use the feedback to make the rule precise."])
    result = Manifestor(lm).manifest(action(), "region", "full", "feedback", "traces")
    assert result == Intervention("Use the feedback to make the rule precise.", "assistant_reasoning")
    assert len(lm.calls) == 2
    assert "previous reply contained no visible steering text" in lm.calls[1].lower()


def test_repeated_empty_manifestation_raises_explicit_error() -> None:
    """Reject a semantic action when retrying still yields no visible text."""
    lm = SequenceLM(["<think>private</think>", "   "])
    with pytest.raises(ManifestationError, match="no visible steering text"):
        Manifestor(lm).manifest(action(), "region", "full", "feedback", "traces")
    assert len(lm.calls) == 2


def test_blank_fixed_manifestation_is_rejected() -> None:
    """Prevent fixed semantic steering from silently collapsing to empty text."""
    blank_spec = InterventionSpec(
        name="blank",
        description="Invalid blank steering.",
        compatible_tools=(EditTool.INSERT_TEXT,),
        fixed_text="   ",
    )
    lm = RecordingLM()
    with pytest.raises(ManifestationError, match="empty fixed steering text"):
        Manifestor(lm).manifest(action(blank_spec), "region", "full", "feedback", "traces")
    assert lm.calls == []


def test_only_traces_are_bounded_by_default() -> None:
    """Keep candidate and feedback whole while bounding the unbounded trace input."""
    large_state = "s" * (MAX_TRACES_CHARS + 10)
    large_traces = "t" * (MAX_TRACES_CHARS + 10)
    lm = RecordingLM()
    Manifestor(lm).manifest(action(), large_state, large_state, large_state, large_traces)
    prompt = lm.calls[0]
    assert prompt.count(large_state) == 3
    assert large_traces not in prompt
    assert "...(+10 chars)" in prompt


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        pytest.param(5, "01234\n...(+5 chars)", id="custom_bound"),
        pytest.param(None, "0123456789", id="unbounded"),
    ],
)
def test_trace_bound_is_configurable(limit: int | None, expected: str) -> None:
    """Honor explicit trace-context limits."""
    lm = RecordingLM()
    Manifestor(lm, max_traces_chars=limit).manifest(action(), "region", "full", "feedback", "0123456789")
    assert expected in lm.calls[0]
