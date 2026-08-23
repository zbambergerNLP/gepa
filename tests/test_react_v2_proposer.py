# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the bounded ReAct V2 text-tool loop."""

import json
from copy import deepcopy
from typing import Any

import pytest

from gepa.lm import NativeToolCall, ToolCompletion
from gepa.proposer.reflective_mutation.react_v2_proposer import (
    ReActV2ContextError,
    ReActV2Proposer,
    ReActV2ProtocolError,
    parse_tool_call,
)
from gepa.strategies.document_template import TEMPLATES, EditTarget
from gepa.strategies.edit_tools import (
    EDIT_TOOL_SETS,
    DeleteTextArgs,
    EditTool,
    InsertTextArgs,
    MoveTextArgs,
    ReplaceTextArgs,
)
from gepa.strategies.intervention import Intervention

TEMPLATE = TEMPLATES["system_prompt"]
PROMPT = TEMPLATE.render({"Role": "helper", "Rules": "- be nice\n- be brief"})
LOWERING_PROMPT = TEMPLATE.render({"Role": "helper", "Rules": "old|anchor|tail"})
RULES = EditTarget("sys", "Rules")


def tool_call(tool: EditTool, **fields: str) -> str:
    """Render one protocol tool call."""
    children = [f"<tool>{tool.value}</tool>"]
    children.extend(f"<{name}>{value}</{name}>" for name, value in fields.items())
    return f"<tool_call>{''.join(children)}</tool_call>"


class ScriptedLM:
    """Return scripted actions while recording immutable conversation snapshots."""

    def __init__(self, replies: list[str]):
        """Store the scripted assistant replies."""
        self.replies = replies
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        """Record the current messages and return the matching reply."""
        self.calls.append(deepcopy(messages))
        index = len(self.calls) - 1
        if index >= len(self.replies):
            raise AssertionError(f"Unexpected ReAct V2 turn {index + 1}")
        return self.replies[index]


class NativeScriptedLM:
    """Return provider-native tool completions and record their request contract."""

    def __init__(self, replies: list[ToolCompletion]):
        """Store scripted native completions."""
        self.replies = replies
        self.calls: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]]] = []
        self.tool_choices: list[str] = []

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str,
    ) -> ToolCompletion:
        """Record one native request and return its scripted completion."""
        self.calls.append(deepcopy(messages))
        self.tools.append(deepcopy(tools))
        self.tool_choices.append(tool_choice)
        index = len(self.calls) - 1
        if index >= len(self.replies):
            raise AssertionError(f"Unexpected ReAct V2 turn {index + 1}")
        return self.replies[index]


def native_call(tool: EditTool, call_id: str = "call-1", **fields: str) -> NativeToolCall:
    """Build one provider-native function call."""
    return NativeToolCall(call_id, tool.value, json.dumps(fields))


def run(
    lm: Any,
    *,
    allowed_tools: list[EditTool] | None = None,
    preferred_tool: EditTool | None = None,
    intervention: Intervention | None = None,
    history: list[dict[str, Any]] | None = None,
    max_iterations: int = 8,
    max_tool_calls: int = 4,
    max_history_chars: int = 12_000,
    max_initial_context_chars: int = 64_000,
    max_chars: int | None = None,
    component_text: str = PROMPT,
    edit_target: EditTarget = RULES,
    traces: str = "Input: q\nOutput: a",
):
    """Run a Rules-section proposal with concise defaults."""
    proposer = ReActV2Proposer(
        lm,
        TEMPLATE,
        allowed_tools or EDIT_TOOL_SETS["broad"],
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        max_history_chars=max_history_chars,
        max_initial_context_chars=max_initial_context_chars,
    )
    return proposer.propose(
        component_text,
        edit_target,
        preferred_tool,
        intervention,
        "The answer was vague.",
        traces,
        history or [],
        max_chars,
    )


@pytest.mark.parametrize(
    ("reply", "expected_tool", "expected_args"),
    [
        pytest.param(
            tool_call(EditTool.INSERT_TEXT, anchor="be nice", where="after", text=" and exact"),
            EditTool.INSERT_TEXT,
            InsertTextArgs(text=" and exact", anchor="be nice", where="after"),
            id="insert",
        ),
        pytest.param(
            tool_call(EditTool.DELETE_TEXT, target="be brief"),
            EditTool.DELETE_TEXT,
            DeleteTextArgs(target="be brief"),
            id="delete",
        ),
        pytest.param(
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
            EditTool.REPLACE_TEXT,
            ReplaceTextArgs(target="be nice", text="be kind"),
            id="replace",
        ),
        pytest.param(
            tool_call(EditTool.MOVE_TEXT, target="be nice", anchor="be brief", where="after"),
            EditTool.MOVE_TEXT,
            MoveTextArgs(target="be nice", anchor="be brief", where="after"),
            id="move",
        ),
    ],
)
def test_parse_tool_call_returns_typed_arguments(
    reply: str,
    expected_tool: EditTool,
    expected_args: object,
) -> None:
    """Parse every supported text operator into its typed contract."""
    assert parse_tool_call(reply) == (expected_tool, expected_args)


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        pytest.param("no action", "exactly one", id="missing_action"),
        pytest.param("<tool_call><target>x</target></tool_call>", "<tool>", id="missing_tool"),
        pytest.param(
            "<tool_call><tool>UPSERT</tool><text>x</text></tool_call>",
            "Unknown edit tool",
            id="unknown_tool",
        ),
        pytest.param(
            tool_call(EditTool.INSERT_TEXT, anchor="x", where="beside", text="y"),
            "before.*after",
            id="invalid_placement",
        ),
    ],
)
def test_parse_tool_call_rejects_invalid_protocol(reply: str, message: str) -> None:
    """Return precise protocol errors that the loop can expose as observations."""
    with pytest.raises(ReActV2ProtocolError, match=message):
        parse_tool_call(reply)


def test_invalid_call_becomes_an_observation_and_can_be_corrected() -> None:
    """Keep the document unchanged, show the error, and let the model retry."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.REPLACE_TEXT, target="not present", text="be kind"),
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT"]
    assert result.steps[0].component_text == PROMPT
    second_turn = lm.calls[1]
    assert second_turn[-1]["role"] == "user"
    assert "ERROR:" in second_turn[-1]["content"]
    assert "target not found" in second_turn[-1]["content"]
    assert "be kind" in result.new_text


def test_direct_semantic_action_completes_after_exactly_one_bound_tool_call() -> None:
    """Use a single REPLACE call for direct rephrase/summarize execution."""
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")])
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert result.iterations == 1
    assert result.tool_calls == 1
    assert len(lm.calls) == 1
    assert "Make exactly one valid REPLACE_TEXT call" in lm.calls[0][0]["content"]
    assert result.executed_edit == ["DELETE 'be nice'", "INSERT 'be kind'"]


def test_production_path_exposes_every_configured_tool_with_auto_choice() -> None:
    """Use provider-native functions instead of the compatibility text protocol."""
    lm = NativeScriptedLM([ToolCompletion("", (native_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),))])
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert lm.tool_choices == ["auto"]
    assert {definition["function"]["name"] for definition in lm.tools[0]} == {
        tool.value for tool in EDIT_TOOL_SETS["broad"]
    }
    assert result.steps[0].assistant.startswith("Tool call: REPLACE_TEXT\nArguments: ")
    assert '"target": "be nice"' in result.steps[0].assistant
    assert not result.steps[0].assistant.startswith('{"role": "assistant"')
    assert all(definition["type"] == "function" for definition in lm.tools[0])
    assert "one provider-native function call" in lm.calls[0][0]["content"]
    assert "<tool_call>" not in lm.calls[0][0]["content"]


def test_custom_callable_uses_explicit_text_tool_compatibility_protocol() -> None:
    """Retain a documented fallback for callables without native-tool support."""
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")])
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert "compatibility text schema" in lm.calls[0][0]["content"]
    assert "<tool_call>" in lm.calls[0][0]["content"]


def test_multiple_native_tool_calls_are_rejected_before_retry() -> None:
    """Return one error observation per native call and apply neither action."""
    lm = NativeScriptedLM(
        [
            ToolCompletion(
                "",
                (
                    native_call(
                        EditTool.INSERT_TEXT,
                        "call-insert",
                        anchor="be nice",
                        where="after",
                        text=" and wrong",
                    ),
                    native_call(
                        EditTool.REPLACE_TEXT,
                        "call-replace",
                        target="be nice",
                        text="be wrong",
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (native_call(EditTool.REPLACE_TEXT, "call-correct", target="be nice", text="be kind"),),
            ),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT"]
    assert "received 2" in result.steps[0].error
    assert "wrong" not in result.new_text
    retry_messages = lm.calls[1]
    assert [message["role"] for message in retry_messages[-2:]] == ["tool", "tool"]
    assert {message["tool_call_id"] for message in retry_messages[-2:]} == {"call-insert", "call-replace"}


def test_multiple_text_tool_blocks_are_rejected_before_retry() -> None:
    """Do not silently execute the first action from a multi-action text reply."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be wrong")
            + tool_call(EditTool.DELETE_TEXT, target="be brief"),
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT"]
    assert "received 2" in result.steps[0].error
    assert "wrong" not in result.new_text
    assert "be brief" in result.new_text


def test_direct_semantic_action_rejects_a_different_available_tool() -> None:
    """Enforce semantic-action/tool coupling even when all four tools are exposed."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.INSERT_TEXT, anchor="be nice", where="after", text=" and kind"),
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT"]
    assert "coupled to REPLACE_TEXT" in result.steps[0].observation
    assert "and kind" not in result.new_text


def test_minimal_basis_composes_delete_and_insert_then_finishes() -> None:
    """Faithfully lower hidden REPLACE into delete, same-location insert, and finish."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            "<finish>The rephrase is complete.</finish>",
        ]
    )
    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
    )
    assert result.changed is True
    assert result.iterations == 3
    assert result.tool_calls == 2
    assert [step.action for step in result.steps] == ["DELETE_TEXT", "INSERT_TEXT", "FINISH"]
    assert result.executed_edit == ["DELETE 'old|'", "INSERT 'new|' before 'anchor'"]
    assert "new|anchor|tail" in result.new_text
    assert "old|" not in result.new_text
    assert "exactly one DELETE_TEXT call followed by one INSERT_TEXT call" in lm.calls[0][0]["content"]


def test_minimal_replace_rejects_insert_before_delete_without_advancing_state() -> None:
    """Return a wrong-order observation while retaining the untouched parent."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.INSERT_TEXT, anchor="old", where="after", text="wrong"),
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            "<finish>Done.</finish>",
        ]
    )

    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
    )

    assert [step.action for step in result.steps] == ["INVALID", "DELETE_TEXT", "INSERT_TEXT", "FINISH"]
    assert "requires DELETE_TEXT next" in result.steps[0].observation
    assert result.steps[0].component_text == LOWERING_PROMPT
    assert "wrong" not in result.new_text


def test_hidden_replace_uses_atomic_lowering_even_with_extra_available_tools() -> None:
    """Do not let a custom basis bypass the selected high-level operator."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.MOVE_TEXT, target="old|", anchor="tail", where="after"),
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            "<finish>Done.</finish>",
        ]
    )
    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=[EditTool.INSERT_TEXT, EditTool.DELETE_TEXT, EditTool.MOVE_TEXT],
        preferred_tool=EditTool.REPLACE_TEXT,
    )

    assert [step.action for step in result.steps] == ["INVALID", "DELETE_TEXT", "INSERT_TEXT", "FINISH"]
    assert "requires DELETE_TEXT next" in result.steps[0].observation
    assert "new|anchor|tail" in result.new_text


def test_hidden_semantic_operator_requires_a_direct_or_lowerable_basis() -> None:
    """Fail before calling the model when a coupled operator cannot be realized."""
    lm = ScriptedLM([])
    with pytest.raises(ValueError, match="cannot be lowered"):
        run(
            lm,
            allowed_tools=[EditTool.INSERT_TEXT],
            preferred_tool=EditTool.REPLACE_TEXT,
        )
    assert lm.calls == []


def test_minimal_replace_rejects_wrong_insertion_location_and_allows_retry() -> None:
    """Accept only the final state produced by one broad replacement at the deleted span."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="tail", where="after", text="new|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            "<finish>Done.</finish>",
        ]
    )

    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
    )

    assert [step.action for step in result.steps] == ["DELETE_TEXT", "INVALID", "INSERT_TEXT", "FINISH"]
    assert "does not reproduce one REPLACE_TEXT" in result.steps[1].observation
    assert "new|" not in result.steps[1].component_text
    broad = run(
        ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="old|", text="new|")]),
        component_text=LOWERING_PROMPT,
        preferred_tool=EditTool.REPLACE_TEXT,
    )
    assert result.new_text == broad.new_text


def test_minimal_replace_compares_the_exact_component() -> None:
    """Reject an insertion that does not reproduce the exact broad replacement."""
    component = TEMPLATE.render({"Rules": "prefix  old  suffix"})
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old"),
            tool_call(EditTool.INSERT_TEXT, anchor="", where="after", text="new"),
        ]
    )
    result = run(
        lm,
        component_text=component,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
        max_iterations=2,
    )

    assert result.changed is False
    assert result.new_text == component
    assert [step.action for step in result.steps] == ["DELETE_TEXT", "INVALID"]
    assert "does not reproduce one REPLACE_TEXT" in result.steps[1].observation
    assert result.executed_edit == ["DELETE 'old'"]


def test_minimal_replace_rejects_finish_after_only_the_delete() -> None:
    """Keep a partial replacement open until the matching insert arrives."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            "<finish>Done.</finish>",
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            "<finish>Done.</finish>",
        ]
    )

    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
    )

    assert [step.action for step in result.steps] == ["DELETE_TEXT", "INVALID", "INSERT_TEXT", "FINISH"]
    assert "required" in result.steps[1].observation
    assert "INSERT_TEXT" in result.steps[1].observation
    assert result.tool_calls == 2


def test_minimal_replace_rejects_extra_tool_call_after_exact_lowering() -> None:
    """Require finish immediately after the two valid atomic calls."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            tool_call(EditTool.DELETE_TEXT, target="tail"),
            "<finish>Done.</finish>",
        ]
    )

    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
    )

    assert [step.action for step in result.steps] == ["DELETE_TEXT", "INSERT_TEXT", "INVALID", "FINISH"]
    assert "decomposition is complete" in result.steps[2].observation
    assert "tail" in result.new_text
    assert result.tool_calls == 2


def test_minimal_move_native_calls_match_one_broad_move() -> None:
    """Lower hidden MOVE with exact deleted bytes and the direct move destination."""
    component = TEMPLATE.render({"Role": "helper", "Rules": "- one\n- two\n- three"})
    lm = NativeScriptedLM(
        [
            ToolCompletion("", (native_call(EditTool.DELETE_TEXT, "delete", target="- two\n"),)),
            ToolCompletion(
                "",
                (
                    native_call(
                        EditTool.INSERT_TEXT,
                        "insert",
                        anchor="- one\n",
                        where="before",
                        text="- two\n",
                    ),
                ),
            ),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )

    result = run(
        lm,
        component_text=component,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.MOVE_TEXT,
    )
    broad = run(
        NativeScriptedLM(
            [
                ToolCompletion(
                    "",
                    (
                        native_call(
                            EditTool.MOVE_TEXT,
                            "move",
                            target="- two\n",
                            anchor="- one\n",
                            where="before",
                        ),
                    ),
                )
            ]
        ),
        component_text=component,
        preferred_tool=EditTool.MOVE_TEXT,
    )

    assert result.new_text == broad.new_text
    assert [step.action for step in result.steps] == ["DELETE_TEXT", "INSERT_TEXT", "FINISH"]
    assert result.tool_calls == 2
    assert lm.tool_choices == ["auto", "auto", "auto"]


def test_minimal_move_rejects_changed_bytes_and_original_destination() -> None:
    """Keep the delete pending through invalid text and no-op destination retries."""
    component = TEMPLATE.render({"Role": "helper", "Rules": "- one\n- two\n- three"})
    lm = NativeScriptedLM(
        [
            ToolCompletion("", (native_call(EditTool.DELETE_TEXT, "delete", target="- two\n"),)),
            ToolCompletion(
                "",
                (
                    native_call(
                        EditTool.INSERT_TEXT,
                        "wrong-text",
                        anchor="- one\n",
                        where="before",
                        text="- changed\n",
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (
                    native_call(
                        EditTool.INSERT_TEXT,
                        "same-place",
                        anchor="- three",
                        where="before",
                        text="- two\n",
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (
                    native_call(
                        EditTool.INSERT_TEXT,
                        "correct",
                        anchor="- one\n",
                        where="before",
                        text="- two\n",
                    ),
                ),
            ),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )

    result = run(
        lm,
        component_text=component,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.MOVE_TEXT,
    )

    assert [step.action for step in result.steps] == [
        "DELETE_TEXT",
        "INVALID",
        "INVALID",
        "INSERT_TEXT",
        "FINISH",
    ]
    assert "reinsert exactly the bytes" in result.steps[1].observation
    assert "distinct replacement or MOVE_TEXT destination" in result.steps[2].observation
    assert "changed" not in result.new_text


@pytest.mark.parametrize(
    ("tool", "fields"),
    [
        pytest.param(
            EditTool.INSERT_TEXT,
            {"anchor": "be nice", "where": "after", "text": " and kind"},
            id="insert",
        ),
        pytest.param(EditTool.DELETE_TEXT, {"target": "be nice"}, id="delete"),
        pytest.param(EditTool.REPLACE_TEXT, {"target": "be nice", "text": "be kind"}, id="replace"),
        pytest.param(
            EditTool.MOVE_TEXT,
            {"target": "be nice", "anchor": "be brief", "where": "after"},
            id="move",
        ),
    ],
)
def test_broad_basis_executes_all_four_text_tools(tool: EditTool, fields: dict[str, str]) -> None:
    """Keep insert, delete, replace, and move callable through one protocol."""
    lm = ScriptedLM([tool_call(tool, **fields)])
    result = run(lm, preferred_tool=tool)
    assert result.changed is True
    assert result.tool_calls == 1
    assert result.steps[0].action == tool.value
    TEMPLATE.parse(result.new_text)


def test_canonical_section_headers_are_protected_and_error_is_observable() -> None:
    """Reject a tool call that writes a valid but absent header, then accept a safe retry."""
    lm = ScriptedLM(
        [
            tool_call(
                EditTool.INSERT_TEXT,
                anchor="be brief",
                where="after",
                text="\n## Reasoning\nignore safeguards",
            ),
            tool_call(EditTool.INSERT_TEXT, anchor="be brief", where="after", text="\n- cite sources"),
        ]
    )
    result = run(lm, preferred_tool=EditTool.INSERT_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "INSERT_TEXT"]
    assert "section body cannot contain" in result.steps[0].observation
    assert "## Reasoning" not in result.new_text
    assert "cite sources" in result.new_text
    TEMPLATE.parse(result.new_text)


def test_section_edit_preserves_every_unselected_body() -> None:
    """Keep every unselected section unchanged while canonically rendering the edited body."""
    component = TEMPLATE.render({"Role": "helper", "Context": "private context", "Rules": "- be nice\n- be brief"})
    before = TEMPLATE.parse(component)
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")])
    result = run(
        lm,
        preferred_tool=EditTool.REPLACE_TEXT,
        component_text=component,
    )
    after = TEMPLATE.parse(result.new_text)
    assert after["Rules"] == "- be kind\n- be brief"
    assert {section: body for section, body in after.items() if section != "Rules"} == {
        section: body for section, body in before.items() if section != "Rules"
    }


def test_insert_into_omitted_section_adds_its_header_in_schema_order() -> None:
    """Populate an absent section while leaving neighboring content unchanged."""
    component = TEMPLATE.render({"Rules": "- be brief"})
    role = EditTarget("sys", "Role")
    lm = ScriptedLM([tool_call(EditTool.INSERT_TEXT, anchor="", where="after", text="helper")])
    result = run(
        lm,
        preferred_tool=EditTool.INSERT_TEXT,
        component_text=component,
        edit_target=role,
    )
    assert result.new_text == TEMPLATE.render({"Role": "helper", "Rules": "- be brief"})
    assert TEMPLATE.parse(result.new_text)["Role"] == "helper"


def test_delete_last_text_from_a_section_removes_its_header() -> None:
    """Omit a section from task-model text once its body becomes empty."""
    component = TEMPLATE.render({"Reasoning": "Check the answer."})
    result = run(
        ScriptedLM([tool_call(EditTool.DELETE_TEXT, target="Check the answer.")]),
        component_text=component,
        edit_target=EditTarget("sys", "Reasoning"),
        preferred_tool=EditTool.DELETE_TEXT,
    )
    assert result.changed is True
    assert result.new_text == ""
    assert TEMPLATE.parse(result.new_text)["Reasoning"] == ""


def test_insert_into_an_entirely_empty_document_adds_only_the_selected_section() -> None:
    """Allow the proposer to populate a schema even when no section is currently rendered."""
    result = run(
        ScriptedLM([tool_call(EditTool.INSERT_TEXT, anchor="", where="after", text="helper")]),
        component_text="",
        edit_target=EditTarget("sys", "Role"),
        preferred_tool=EditTool.INSERT_TEXT,
    )
    assert result.new_text == TEMPLATE.render({"Role": "helper"})


def test_whole_document_edit_cannot_delete_a_header() -> None:
    """Reject a whole-document operation that changes structural headers."""
    result = run(
        ScriptedLM([tool_call(EditTool.DELETE_TEXT, target="## Role\nhelper\n\n")]),
        edit_target=EditTarget("sys", None),
        preferred_tool=EditTool.DELETE_TEXT,
        max_iterations=1,
    )
    assert result.changed is False
    assert result.new_text == PROMPT
    assert result.steps[0].action == "INVALID"
    assert "preserve the existing" in result.steps[0].observation


def test_finish_before_a_changing_tool_call_is_rejected() -> None:
    """Prevent an unchanged atomic-basis proposal from being marked complete."""
    lm = ScriptedLM(
        [
            "<finish>Nothing to do.</finish>",
            tool_call(EditTool.DELETE_TEXT, target="be nice"),
            "<finish>Done.</finish>",
        ]
    )
    result = run(lm, allowed_tools=EDIT_TOOL_SETS["minimal"])
    assert [step.action for step in result.steps] == ["INVALID", "DELETE_TEXT", "FINISH"]
    assert "Cannot finish" in result.steps[0].observation


def test_manifestor_steering_is_delivered_in_the_current_user_message() -> None:
    """Keep Manifestor steering coupled to the current editing task."""
    user_lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")])
    run(
        user_lm,
        preferred_tool=EditTool.REPLACE_TEXT,
        intervention=Intervention("User steering", "user"),
    )
    user_messages = user_lm.calls[0]
    assert [message["role"] for message in user_messages] == ["system", "user"]
    assert user_messages[-1]["content"].startswith("User steering\n\n## Selected component")


def test_branch_chat_history_is_replayed_before_the_current_task() -> None:
    """Replay only user/assistant turns supplied for this selected parent."""
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")])
    history = [
        {"role": "assistant", "content": "<tool_call>parent-only</tool_call>"},
        {"role": "user", "content": "Optimizer feedback: REJECTED. That edit did not help."},
    ]
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)
    assert lm.calls[0][1:3] == history
    assert "Branch-local conversation" in lm.calls[0][-1]["content"]
    assert result.revision_record()["steps"][0]["action"] == "REPLACE_TEXT"


def test_branch_history_overflow_raises_without_calling_the_lm() -> None:
    """Fail explicitly instead of adding global memory or lossy compression."""
    lm = ScriptedLM([])
    with pytest.raises(ReActV2ContextError, match="Global history.*disabled"):
        run(
            lm,
            preferred_tool=EditTool.REPLACE_TEXT,
            history=[{"role": "user", "content": "x" * 100}],
            max_history_chars=20,
        )
    assert lm.calls == []


@pytest.mark.parametrize(
    ("history", "error"),
    [
        pytest.param([{"role": "tool", "content": "bad"}], ValueError, id="tool_role"),
        pytest.param([{"role": "user", "content": "ok", "extra": "bad"}], ValueError, id="extra_key"),
        pytest.param([{"role": "assistant", "content": 3}], TypeError, id="non_string"),
    ],
)
def test_branch_chat_history_requires_user_assistant_content_messages(
    history: list[dict[str, Any]],
    error: type[Exception],
) -> None:
    """Reject provider/tool metadata in the persisted branch transcript."""
    lm = ScriptedLM([])
    with pytest.raises(error, match="[Bb]ranch-local history"):
        run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)
    assert lm.calls == []


def test_total_initial_context_overflow_includes_traces_and_precedes_lm_call() -> None:
    """Bound the whole initial request, not only its branch-history subsection."""
    lm = ScriptedLM([])
    with pytest.raises(ReActV2ContextError, match="total context budget.*compression.*disabled"):
        run(
            lm,
            preferred_tool=EditTool.REPLACE_TEXT,
            traces="trajectory-step\n" * 1_000,
            max_initial_context_chars=4_000,
        )
    assert lm.calls == []


def test_context_growth_overflow_precedes_the_next_lm_call() -> None:
    """Recheck the full conversation after observations grow the next turn."""
    lm = ScriptedLM(["invalid response " + "x" * 5_000])
    with pytest.raises(ReActV2ContextError, match="prior ReAct turns.*compression.*disabled"):
        run(
            lm,
            preferred_tool=EditTool.REPLACE_TEXT,
            max_iterations=2,
            max_initial_context_chars=6_000,
        )
    assert len(lm.calls) == 1


def test_iteration_exhaustion_drops_partial_atomic_edits() -> None:
    """Return the untouched parent when a multi-step proposal never finishes."""
    lm = ScriptedLM([tool_call(EditTool.DELETE_TEXT, target="be nice")])
    result = run(
        lm,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        max_iterations=1,
    )
    assert result.changed is False
    assert result.new_text == PROMPT
    assert result.tool_calls == 1
    assert result.dropped_reason == "No completed revision within 1 ReAct V2 turns."


def test_max_chars_violation_is_returned_as_an_error_observation() -> None:
    """Keep an over-budget edit out of the completed candidate."""
    lm = ScriptedLM([tool_call(EditTool.INSERT_TEXT, anchor="be nice", where="after", text="x" * 100)])
    result = run(
        lm,
        preferred_tool=EditTool.INSERT_TEXT,
        max_iterations=1,
        max_chars=len(PROMPT) + 5,
    )
    assert result.changed is False
    assert result.new_text == PROMPT
    assert "exceeding max_chars" in result.steps[0].observation
