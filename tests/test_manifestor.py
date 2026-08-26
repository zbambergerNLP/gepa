# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for POSIT-style semantic-action manifestation."""

import pytest

from gepa.proposer.reflective_mutation.manifestor import (
    MAX_STEERING_MESSAGE_CHARS,
    MAX_TRACES_CHARS,
    ManifestationError,
    Manifestor,
)
from gepa.strategies.document_template import EditTarget
from gepa.strategies.edit_tools import EditTool
from gepa.strategies.intervention import ControllerChoice, SemanticActionSpec

SPEC = SemanticActionSpec(
    name="contextualize",
    description="Add supporting context without changing the operative rule.",
    edit_tool=EditTool.INSERT_TEXT,
    instruction="Name the missing background and where the editor should add it.",
)
FIXED_SPEC = SemanticActionSpec(
    name="fixed",
    description="Use fixed steering.",
    edit_tool=EditTool.INSERT_TEXT,
    fixed_text="Add one concise requirement.",
)


class RecordingLM:
    """Record manifestation prompts and return a fixed steering message."""

    def __init__(self, reply: str = "The failures omit citations, so add a citation requirement."):
        """Configure the fixed steering reply.

        Args:
            reply: Text returned for every model call.
        """
        self.reply = reply
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record and answer one Manifestor prompt.

        Args:
            prompt: Manifestor prompt sent by the code under test.

        Returns:
            The fixed steering reply configured at initialization.
        """
        self.calls.append(prompt)
        return self.reply


class SequenceLM:
    """Return one scripted reply per manifestation attempt."""

    def __init__(self, replies: list[str]):
        """Configure replies for successive manifestation attempts.

        Args:
            replies: Steering replies returned in call order.
        """
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record one Manifestor prompt and return its scripted reply.

        Args:
            prompt: Manifestor prompt sent by the code under test.

        Returns:
            Scripted reply at the current call index.

        Raises:
            IndexError: No scripted reply exists for the current call.
        """
        self.calls.append(prompt)
        return self.replies[len(self.calls) - 1]


def test_level1_action_skips_manifestation() -> None:
    """Avoid an LM call when the Controller selected only an atomic region."""
    lm = RecordingLM()
    choice = ControllerChoice(EditTarget("sys", "Rules"), None)
    assert Manifestor(lm).manifest(choice, "rules", "feedback", "traces") is None
    assert lm.calls == []


def test_fixed_text_skips_lm() -> None:
    """Return fixed steering without calling the Manifestor LM."""
    lm = RecordingLM()
    choice = ControllerChoice(EditTarget("sys", "Rules"), FIXED_SPEC)
    result = Manifestor(lm).manifest(choice, "rules", "feedback", "traces")
    assert result == "Add one concise requirement."
    assert lm.calls == []


def test_instruction_spec_is_manifested_once_with_section_grounding() -> None:
    """Ground steering in the selected section, feedback, and traces only."""
    lm = RecordingLM()
    result = Manifestor(lm).manifest(
        ControllerChoice(EditTarget("sys", "Rules"), SPEC),
        region_text="- be accurate",
        feedback_summary="The answer omitted its source.",
        traces="Output: unsupported claim",
    )
    assert result == lm.reply
    assert len(lm.calls) == 1
    prompt = lm.calls[0]
    for expected in (
        SPEC.name,
        SPEC.description,
        SPEC.instruction,
        EditTool.INSERT_TEXT.value,
        "Rules",
        "- be accurate",
        "The answer omitted its source.",
        "Output: unsupported claim",
        "Write the next instruction",
    ):
        assert expected in prompt
    assert "## Role\nhelper" not in prompt
    assert "do not write the edit" in prompt.lower()


def test_overlong_manifestation_is_truncated() -> None:
    """Keep steering within the configured length bound."""
    lm = RecordingLM("x" * (MAX_STEERING_MESSAGE_CHARS + 50))
    choice = ControllerChoice(EditTarget("sys", "Rules"), SPEC)
    result = Manifestor(lm).manifest(choice, "region", "feedback", "traces")
    assert result is not None
    assert result == "x" * MAX_STEERING_MESSAGE_CHARS + "..."


def test_empty_manifestation_is_retried_once() -> None:
    """Recover from an empty first reply without sending empty steering."""
    lm = SequenceLM(["   ", "Use the feedback to make the rule precise."])
    choice = ControllerChoice(EditTarget("sys", "Rules"), SPEC)
    result = Manifestor(lm).manifest(choice, "region", "feedback", "traces")
    assert result == "Use the feedback to make the rule precise."
    assert len(lm.calls) == 2
    assert "previous reply contained no steering text" in lm.calls[1].lower()


def test_repeated_empty_manifestation_raises_explicit_error() -> None:
    """Raise when both Manifestor attempts return blank steering."""
    lm = SequenceLM(["   ", "\n"])
    choice = ControllerChoice(EditTarget("sys", "Rules"), SPEC)
    with pytest.raises(ManifestationError, match="no visible steering text"):
        Manifestor(lm).manifest(choice, "region", "feedback", "traces")
    assert len(lm.calls) == 2


def test_controller_action_is_authoritative_for_manifestor() -> None:
    """Realize the sampled action without a second applicability decision."""
    lm = RecordingLM("Add grounded background that addresses the observed failure.")
    choice = ControllerChoice(EditTarget("sys", "Rules"), SPEC)
    result = Manifestor(lm).manifest(choice, "region", "feedback", "traces")
    assert result == "Add grounded background that addresses the observed failure."
    assert len(lm.calls) == 1
    prompt = lm.calls[0]
    assert "do not reassess its preconditions" in prompt
    assert "not a Manifestor decision" in prompt
    assert "<not_applicable>" not in prompt


def test_blank_fixed_manifestation_is_rejected() -> None:
    """Prevent fixed semantic steering from silently collapsing to empty text."""
    blank_spec = SemanticActionSpec(
        name="blank",
        description="Invalid blank steering.",
        edit_tool=EditTool.INSERT_TEXT,
        fixed_text="   ",
    )
    lm = RecordingLM()
    choice = ControllerChoice(EditTarget("sys", "Rules"), blank_spec)
    with pytest.raises(ManifestationError, match="empty fixed steering text"):
        Manifestor(lm).manifest(choice, "region", "feedback", "traces")
    assert lm.calls == []


def test_only_traces_are_bounded_by_default() -> None:
    """Keep the section and feedback whole while bounding the trace input."""
    large_state = "s" * (MAX_TRACES_CHARS + 10)
    large_traces = "t" * (MAX_TRACES_CHARS + 10)
    lm = RecordingLM()
    choice = ControllerChoice(EditTarget("sys", "Rules"), SPEC)
    Manifestor(lm).manifest(choice, large_state, large_state, large_traces)
    prompt = lm.calls[0]
    assert prompt.count(large_state) == 2
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
    """Honor explicit trace-context limits.

    Args:
        limit: Maximum trace characters, or ``None`` for no bound.
        expected: Trace text expected in the generated Manifestor prompt.
    """
    lm = RecordingLM()
    choice = ControllerChoice(EditTarget("sys", "Rules"), SPEC)
    Manifestor(lm, max_traces_chars=limit).manifest(choice, "region", "feedback", "0123456789")
    assert expected in lm.calls[0]
