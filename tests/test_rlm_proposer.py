# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Focused tests for the RLM proposer's edit and history invariants."""

import json

import pytest

from gepa.proposer.reflective_mutation.rlm_environment import RLMBudget
from gepa.proposer.reflective_mutation.rlm_proposer import RLMContextError, RLMProposer
from gepa.strategies.document_template import TEMPLATES, EditTarget
from gepa.strategies.edit_tools import EditTool

TEMPLATE = TEMPLATES["prompt"]
PROMPT = TEMPLATE.render({"Role": "helper", "Rules": "- be nice\n- be brief"})
FEEDBACK = "The answer was vague."
TRACES = "Input: question\nOutput: vague answer"


class ScriptedLM:
    """Replay root RLM turns while recording every prompt."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.replies:
            raise AssertionError("Unexpected RLM model call")
        return self.replies.pop(0)


def replace(target: str, text: str) -> str:
    """Render one forced replace action."""
    return f"<edit><target>{target}</target><text>{text}</text></edit>"


def propose(
    lm: ScriptedLM,
    *,
    component: str = PROMPT,
    section: str = "Rules",
    tool: EditTool = EditTool.REPLACE_TEXT,
    max_chars: int | None = None,
    history: list[dict[str, str]] | None = None,
):
    """Run a bounded proposal against one prompt section."""
    editor = RLMProposer(lm, TEMPLATE, budget=RLMBudget(max_root_iterations=max(1, len(lm.replies))))
    return editor.propose(
        component,
        EditTarget("sys", section),
        tool,
        "Make the requirement exact.",
        FEEDBACK,
        TRACES,
        max_chars,
        branch_history=history or [],
    )


def test_section_edit_preserves_every_byte_outside_the_selected_body() -> None:
    """Splice the edited body instead of re-rendering unselected sections."""
    component = TEMPLATE.render({"Role": "helper", "Rules": "- be nice\n- be brief"})
    component = component.replace("## Role\nhelper\n\n", "## Role  \n\n  helper  \n\n\n")
    component = component.replace(
        "## Rules\n- be nice\n- be brief\n\n",
        "## Rules\t\n\n - be nice\n- be brief \n\n\n",
    )
    old_start, old_end = TEMPLATE.section_body_span(component, "Rules")

    result = propose(ScriptedLM(replace("be nice", "be kind")), component=component)
    new_start, new_end = TEMPLATE.section_body_span(result.new_text, "Rules")

    assert result.changed is True
    assert result.new_text[:new_start] == component[:old_start]
    assert result.new_text[new_end:] == component[old_end:]
    assert TEMPLATE.parse(result.new_text)["Rules"] == "- be kind\n- be brief"


def test_insert_into_empty_section_keeps_headers_and_surrounding_bytes() -> None:
    """Handle a zero-width section body without merging it into the next header."""
    component = TEMPLATE.render({"Role": "helper"})
    old_start, old_end = TEMPLATE.section_body_span(component, "Rules")
    insertion = "<edit><anchor></anchor><where>after</where><text>- be exact</text></edit>"

    result = propose(ScriptedLM(insertion), component=component, tool=EditTool.INSERT_TEXT)
    new_start, new_end = TEMPLATE.section_body_span(result.new_text, "Rules")

    assert old_start == old_end
    assert result.new_text[:new_start] == component[:old_start]
    assert result.new_text[new_end:] == component[old_end:]
    assert TEMPLATE.parse(result.new_text)["Rules"] == "- be exact"


def test_header_injection_is_rejected_and_retried() -> None:
    """Keep the fixed section structure even when an edit payload adds a header."""
    lm = ScriptedLM(
        replace("be nice", "be kind\n## Unapproved\nbad"),
        replace("be nice", "be kind"),
    )

    result = propose(lm)

    assert result.changed is True
    assert result.iterations == 2
    assert "sections" in (result.steps[0].error or "")
    assert "Try again" in result.chat_messages[1]["content"]
    assert "## Unapproved" not in result.new_text


def test_noop_replace_is_rejected_and_retried() -> None:
    """Do not report a successful revision when the forced operator changes nothing."""
    result = propose(ScriptedLM(replace("be nice", "be nice"), replace("be nice", "be kind")))

    assert result.changed is True
    assert result.iterations == 2
    assert "produced no text change" in (result.steps[0].error or "")
    assert result.executed_edit


def test_oversize_edit_is_rejected_and_retried_within_the_same_run() -> None:
    """Give the RLM a chance to fit the component budget instead of dropping immediately."""
    result = propose(
        ScriptedLM(replace("be nice", "x" * 200), replace("be nice", "be kind")),
        max_chars=len(PROMPT),
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
    edit_turn = replace("be nice", "be kind")
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
    lm = ScriptedLM(replace("be nice", "be kind"))
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
    """Accept only branch-local user and assistant messages."""
    with pytest.raises(ValueError):
        propose(ScriptedLM(replace("be nice", "be kind")), history=history)
