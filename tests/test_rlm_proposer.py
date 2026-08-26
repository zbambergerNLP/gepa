# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the RLM proposer's edit and history invariants."""

import json

import pytest

from gepa.proposer.reflective_mutation.rlm_environment import RLMBudget
from gepa.proposer.reflective_mutation.rlm_proposer import RLMContextError, RLMProposer
from gepa.strategies.document_template import TEMPLATES, EditTarget
from gepa.strategies.edit_tools import EditTool

TEMPLATE = TEMPLATES["system_prompt"]
PROMPT = TEMPLATE.render({"Role": "helper", "Rules": "- be nice\n- be brief"})
RULES_TEXT = TEMPLATE.parse(PROMPT)["Rules"]
FEEDBACK = "The answer was vague."
TRACES = "Input: question\nOutput: vague answer"


class ScriptedLM:
    """Replay root RLM turns while recording every prompt."""

    def __init__(self, *replies: str) -> None:
        """Store scripted root replies and an empty prompt log.

        Args:
            *replies: Responses consumed in call order.
        """
        self.replies = list(replies)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record one root prompt and consume its scripted reply.

        Args:
            prompt: Root RLM prompt.

        Returns:
            Next scripted response.

        Raises:
            AssertionError: Calls exceed the scripted replies.
        """
        self.calls.append(prompt)
        if not self.replies:
            raise AssertionError("Unexpected RLM model call")
        return self.replies.pop(0)


def propose(
    lm: ScriptedLM,
    *,
    component: str = PROMPT,
    section: str = "Rules",
    tool: EditTool = EditTool.REPLACE_TEXT,
    max_chars: int | None = None,
    history: list[dict[str, str]] | None = None,
):
    """Run a bounded proposal against one prompt section.

    Args:
        lm: Scripted root model.
        component: Canonical component document.
        section: Section selected for editing.
        tool: Controller-selected edit operator.
        max_chars: Optional edited-section length limit.
        history: Optional parent-branch chat transcript.

    Returns:
        Completed or dropped :class:`RLMResult`.
    """
    editor = RLMProposer(lm, TEMPLATE, budget=RLMBudget(max_root_iterations=max(1, len(lm.replies))))
    region_text = TEMPLATE.parse(component)[section]
    return editor.propose(
        region_text,
        EditTarget("sys", section),
        tool,
        "Make the requirement exact.",
        FEEDBACK,
        TRACES,
        max_chars,
        branch_history=history or [],
    )


def test_selected_section_is_the_only_document_context() -> None:
    """Keep sibling sections out of the RLM model's editing context."""
    component = TEMPLATE.render({"Role": "helper", "Context": "private context", "Rules": "- be nice\n- be brief"})
    lm = ScriptedLM("<edit><target>be nice</target><text>be kind</text></edit>")
    result = propose(lm, component=component)

    assert result.changed is True
    assert result.new_text == "- be kind\n- be brief"
    assert "helper" not in lm.calls[0]
    assert "private context" not in lm.calls[0]
    assert "- component: str" not in lm.calls[0]


def test_manifestor_guidance_is_included_in_the_root_prompt() -> None:
    """Pass plain Manifestor text to the RLM as planner guidance."""
    lm = ScriptedLM("<edit><target>be nice</target><text>be kind</text></edit>")

    propose(lm)

    assert "## Guidance from the planner\nMake the requirement exact." in lm.calls[0]


def test_insert_into_omitted_section_returns_the_new_body() -> None:
    """Populate an absent section body without receiving sibling content."""
    component = TEMPLATE.render({"Role": "helper"})
    insertion = "<edit><anchor></anchor><where>after</where><text>- be exact</text></edit>"

    result = propose(ScriptedLM(insertion), component=component, tool=EditTool.INSERT_TEXT)

    assert result.new_text == "- be exact"
    assert "helper" not in result.final_output


def test_header_injection_is_rejected_and_retried() -> None:
    """Keep the fixed section structure even when an edit payload adds a header."""
    lm = ScriptedLM(
        "<edit><target>be nice</target><text>be kind\n## Unapproved\nbad</text></edit>",
        "<edit><target>be nice</target><text>be kind</text></edit>",
    )

    result = propose(lm)

    assert result.changed is True
    assert result.iterations == 2
    assert "section body cannot contain" in (result.steps[0].error or "")
    assert "Try again" in result.chat_messages[1]["content"]
    assert "## Unapproved" not in result.new_text


def test_delete_last_text_returns_an_empty_section_body() -> None:
    """Return an empty body when the selected section's last text is deleted."""
    component = TEMPLATE.render({"Reasoning": "Check the answer."})
    deletion = "<edit><target>Check the answer.</target></edit>"
    result = propose(
        ScriptedLM(deletion),
        component=component,
        section="Reasoning",
        tool=EditTool.DELETE_TEXT,
    )
    assert result.changed is True
    assert result.new_text == ""


def test_insert_into_an_empty_section_returns_only_the_selected_body() -> None:
    """Populate one selected section whose current body is empty."""
    insertion = "<edit><anchor></anchor><where>after</where><text>helper</text></edit>"
    result = propose(
        ScriptedLM(insertion),
        component="",
        section="Role",
        tool=EditTool.INSERT_TEXT,
    )
    assert result.new_text == "helper"


def test_move_cannot_use_an_anchor_from_an_unselected_section() -> None:
    """Keep both ends of a MOVE inside the selected section."""
    move = "<edit><target>- be nice</target><anchor>helper</anchor><where>after</where></edit>"
    result = propose(
        ScriptedLM(move),
        tool=EditTool.MOVE_TEXT,
    )
    assert result.changed is False
    assert result.new_text == RULES_TEXT
    assert "anchor not found" in (result.steps[0].error or "")


def test_noop_replace_is_rejected_and_retried() -> None:
    """Do not report a successful revision when the forced operator changes nothing."""
    result = propose(
        ScriptedLM(
            "<edit><target>be nice</target><text>be nice</text></edit>",
            "<edit><target>be nice</target><text>be kind</text></edit>",
        )
    )

    assert result.changed is True
    assert result.iterations == 2
    assert "produced no text change" in (result.steps[0].error or "")
    assert result.executed_edit


def test_oversize_edit_is_rejected_and_retried_within_the_same_run() -> None:
    """Give the RLM a chance to fit the section budget instead of dropping immediately."""
    result = propose(
        ScriptedLM(
            f"<edit><target>be nice</target><text>{'x' * 200}</text></edit>",
            "<edit><target>be nice</target><text>be kind</text></edit>",
        ),
        max_chars=len(RULES_TEXT),
    )

    assert result.changed is True
    assert result.iterations == 2
    assert "exceeding max_chars" in (result.steps[0].error or "")


def test_branch_history_is_externalized_as_user_assistant_json_and_attempt_is_retained() -> None:
    """Expose only this branch's transcript as data and retain actual RLM messages."""
    history = [
        {"role": "user", "content": "parent-user-marker"},
        {"role": "assistant", "content": "parent-assistant-marker"},
    ]
    python_turn = "<python>print(history)</python>"
    edit_turn = "<edit><target>be nice</target><text>be kind</text></edit>"
    lm = ScriptedLM(python_turn, edit_turn)

    result = propose(lm, history=history)

    assert "- history: str" in lm.calls[0]
    assert "parent-user-marker" not in lm.calls[0]
    serialized = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    assert serialized in lm.calls[1]
    assert result.chat_messages[0] == {"role": "assistant", "content": python_turn}
    assert result.chat_messages[1]["role"] == "user"
    assert serialized in result.chat_messages[1]["content"]
    assert result.chat_messages[2] == {"role": "assistant", "content": edit_turn}
    assert result.chat_messages[3] == {
        "role": "user",
        "content": "OK: REPLACE_TEXT applied.\nExecuted edit:\nDELETE 'be nice'\nINSERT 'be kind'",
    }


def test_branch_history_over_cap_is_a_fatal_context_error_before_model_use() -> None:
    """Never silently drop a proposal when its selected lineage cannot be transported."""
    lm = ScriptedLM("<edit><target>be nice</target><text>be kind</text></edit>")
    history = [{"role": "user", "content": "x" * 13_000}]

    with pytest.raises(RLMContextError, match="history budget"):
        propose(lm, history=history)

    assert lm.calls == []


@pytest.mark.parametrize(
    "history",
    [
        pytest.param([{"role": "tool", "content": "bad"}], id="tool_role"),
        pytest.param([{"role": "user", "content": "ok", "extra": "bad"}], id="extra_field"),
    ],
)
def test_branch_history_shape_is_validated(history: list[dict[str, str]]) -> None:
    """Accept only branch-local user and assistant messages.

    Args:
        history: Invalid transcript shape under test.
    """
    with pytest.raises(ValueError):
        propose(
            ScriptedLM("<edit><target>be nice</target><text>be kind</text></edit>"),
            history=history,
        )
