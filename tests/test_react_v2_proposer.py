# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the ReAct V2 text-tool loop and explicit completion."""

import json
from copy import deepcopy
from typing import Any

import pytest

from gepa.lm import NativeToolCall, ToolCompletion
from gepa.proposer.reflective_mutation.react_v2_proposer import (
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

TEMPLATE = TEMPLATES["system_prompt"]
PROMPT = TEMPLATE.render({"Role": "helper", "Rules": "- be nice\n- be brief"})
LOWERING_PROMPT = TEMPLATE.render({"Role": "helper", "Rules": "old|anchor|tail"})
RULES_TEXT = TEMPLATE.parse(PROMPT)["Rules"]
LOWERING_RULES = TEMPLATE.parse(LOWERING_PROMPT)["Rules"]
RULES = EditTarget("sys", "Rules")


def tool_call(tool: EditTool, **fields: str) -> str:
    """Render one compatibility-protocol tool call.

    Args:
        tool: Edit operator named in the call.
        **fields: Tool-specific child fields.

    Returns:
        XML-like tool-call block.
    """
    children = [f"<tool>{tool.value}</tool>"]
    children.extend(f"<{name}>{value}</{name}>" for name, value in fields.items())
    return f"<tool_call>{''.join(children)}</tool_call>"


class ScriptedLM:
    """Return scripted actions while recording immutable conversation snapshots."""

    def __init__(self, replies: list[str]):
        """Store scripted assistant replies and an empty call log.

        Args:
            replies: Responses returned on successive calls.
        """
        self.replies = replies
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        """Record a conversation snapshot and return its scripted reply.

        Args:
            messages: Current ReAct conversation.

        Returns:
            Reply at the matching call index.

        Raises:
            AssertionError: Calls exceed the scripted replies.
        """
        self.calls.append(deepcopy(messages))
        index = len(self.calls) - 1
        if index >= len(self.replies):
            raise AssertionError(f"Unexpected ReAct V2 turn {index + 1}")
        return self.replies[index]


class NativeScriptedLM:
    """Return provider-native tool completions and record their request contract."""

    def __init__(self, replies: list[ToolCompletion]):
        """Store scripted native completions and empty request logs.

        Args:
            replies: Native completions returned on successive calls.
        """
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
        """Record one native request and return its scripted completion.

        Args:
            messages: Current ReAct conversation.
            tools: Provider function schemas.
            tool_choice: Requested provider selection mode.

        Returns:
            Completion at the matching call index.

        Raises:
            AssertionError: Calls exceed the scripted completions.
        """
        self.calls.append(deepcopy(messages))
        self.tools.append(deepcopy(tools))
        self.tool_choices.append(tool_choice)
        index = len(self.calls) - 1
        if index >= len(self.replies):
            raise AssertionError(f"Unexpected ReAct V2 turn {index + 1}")
        return self.replies[index]


def run(
    lm: Any,
    *,
    allowed_tools: list[EditTool] | None = None,
    preferred_tool: EditTool | None = None,
    steering_message: str | None = None,
    history: list[dict[str, Any]] | None = None,
    max_iterations: int | None = None,
    max_tool_calls: int | None = None,
    max_chars: int | None = None,
    component_text: str = PROMPT,
    edit_target: EditTarget = RULES,
    traces: str = "Input: q\nOutput: a",
):
    """Run a selected-section proposal with concise test defaults.

    Args:
        lm: Scripted text or native-tool language model.
        allowed_tools: Tool basis exposed to the proposer.
        preferred_tool: Semantic action's coupled direct operator.
        steering_message: Optional Manifestor guidance.
        history: Parent-branch chat history.
        max_iterations: Assistant-turn budget.
        max_tool_calls: Valid tool-call budget.
        max_chars: Optional completed-section length limit.
        component_text: Canonical component document.
        edit_target: Selected component section.
        traces: Execution evidence included in the task.

    Returns:
        Completed or dropped :class:`ReActV2Result`.
    """
    proposer = ReActV2Proposer(
        lm,
        TEMPLATE,
        allowed_tools or EDIT_TOOL_SETS["broad"],
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )
    region_text = TEMPLATE.parse(component_text)[edit_target.section]
    return proposer.propose(
        region_text,
        edit_target,
        preferred_tool,
        steering_message,
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
    """Parse every supported text operator into its typed contract.

    Args:
        reply: Compatibility-protocol tool call.
        expected_tool: Operator expected from parsing.
        expected_args: Typed argument object expected from parsing.
    """
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
    """Return protocol errors precise enough to expose as observations.

    Args:
        reply: Invalid compatibility-protocol reply.
        message: Regular expression expected in the raised error.
    """
    with pytest.raises(ReActV2ProtocolError, match=message):
        parse_tool_call(reply)


def test_invalid_call_becomes_an_observation_and_can_be_corrected() -> None:
    """Keep the document unchanged, show the error, and let the model retry."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.REPLACE_TEXT, target="not present", text="be kind"),
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
            "<finish>Done.</finish>",
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert result.iterations == 3
    assert result.tool_calls == 1
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT", "FINISH"]
    assert result.steps[0].region_text == RULES_TEXT
    second_turn = lm.calls[1]
    assert second_turn[-1]["role"] == "user"
    assert "ERROR:" in second_turn[-1]["content"]
    assert "target not found" in second_turn[-1]["content"]
    assert "be kind" in result.new_text


def test_direct_semantic_action_waits_for_explicit_finish() -> None:
    """Continue after a successful bound edit until the editor explicitly finishes."""
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert len(lm.calls) == 2
    assert "Make as many REPLACE_TEXT calls as needed" in lm.calls[0][0]["content"]
    assert [step.action for step in result.steps] == ["REPLACE_TEXT", "FINISH"]
    assert "Latest selected region" in lm.calls[1][-1]["content"]
    assert result.executed_edit == ["DELETE 'be nice'", "INSERT 'be kind'"]


@pytest.mark.parametrize("native", [False, True], ids=["text", "native"])
@pytest.mark.parametrize("tool", EDIT_TOOL_SETS["broad"])
def test_multiple_edits_exceed_both_former_limits_and_finish(native: bool, tool: EditTool) -> None:
    """Keep all ten selected-section edits and expose updated text until finish.

    Args:
        native: Whether to use provider-native tool calls.
        tool: Controller-selected operator used for every edit.
    """
    targets = [f"item-{index}|" for index in range(10)]
    region = "END" if tool is EditTool.INSERT_TEXT else "".join(targets) + "END"
    arguments = []
    for target in targets:
        if tool is EditTool.INSERT_TEXT:
            fields = {"anchor": "END", "where": "before", "text": target}
        elif tool is EditTool.DELETE_TEXT:
            fields = {"target": target}
        elif tool is EditTool.REPLACE_TEXT:
            fields = {"target": target, "text": target.upper()}
        else:
            fields = {"target": target, "anchor": "END", "where": "after"}
        arguments.append(fields)
    expected = {
        EditTool.INSERT_TEXT: "".join(targets) + "END",
        EditTool.DELETE_TEXT: "END",
        EditTool.REPLACE_TEXT: "".join(targets).upper() + "END",
        EditTool.MOVE_TEXT: "END" + "".join(reversed(targets)),
    }[tool]
    if native:
        lm = NativeScriptedLM(
            [
                ToolCompletion("", (NativeToolCall(f"call-{index}", tool.value, json.dumps(fields)),))
                for index, fields in enumerate(arguments)
            ]
            + [ToolCompletion("<finish>Done.</finish>", ())]
        )
    else:
        lm = ScriptedLM([tool_call(tool, **fields) for fields in arguments] + ["<finish>Done.</finish>"])

    result = run(lm, preferred_tool=tool, component_text=TEMPLATE.render({"Role": "helper", "Rules": region}))

    assert result.new_text == expected
    assert result.iterations == 11
    assert result.tool_calls == 10
    assert result.steps[-1].action == "FINISH"
    assert all(step.error is None for step in result.steps)
    assert expected in lm.calls[-1][-1]["content"]
    assert "helper" not in json.dumps(lm.calls[0])


@pytest.mark.parametrize("tool", [EditTool.REPLACE_TEXT, EditTool.MOVE_TEXT])
def test_repeated_atomic_pairs_use_the_latest_region_without_a_call_limit(tool: EditTool) -> None:
    """Complete ten faithful atomic operations before explicitly submitting.

    Args:
        tool: Semantic operator decomposed into delete/insert pairs.
    """
    targets = [f"item-{index}|" for index in range(10)]
    replies = []
    for index, target in enumerate(targets):
        replies.append(tool_call(EditTool.DELETE_TEXT, target=target))
        if tool is EditTool.REPLACE_TEXT:
            anchor = targets[index + 1] if index < 9 else "END"
            replies.append(tool_call(EditTool.INSERT_TEXT, anchor=anchor, where="before", text=target.upper()))
        else:
            replies.append(tool_call(EditTool.INSERT_TEXT, anchor="END", where="after", text=target))
    replies.append("<finish>Done.</finish>")
    result = run(
        ScriptedLM(replies),
        preferred_tool=tool,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        component_text=TEMPLATE.render({"Rules": "".join(targets) + "END"}),
    )
    expected = "".join(targets).upper() + "END" if tool is EditTool.REPLACE_TEXT else "END" + "".join(reversed(targets))
    assert result.new_text == expected
    assert result.iterations == 21
    assert result.tool_calls == 20
    assert all(step.error is None for step in result.steps)


def test_successful_edit_without_finish_does_not_complete() -> None:
    """Discard a direct edit if an explicitly bounded caller runs out of turns."""
    result = run(
        ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")]),
        preferred_tool=EditTool.REPLACE_TEXT,
        max_iterations=1,
    )
    assert result.changed is False
    assert result.new_text == RULES_TEXT
    assert result.tool_calls == 1
    assert result.dropped_reason


def test_production_path_exposes_every_configured_tool_with_auto_choice() -> None:
    """Use provider-native functions instead of the compatibility text protocol."""
    lm = NativeScriptedLM(
        [
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "call-1",
                        EditTool.REPLACE_TEXT.value,
                        json.dumps({"target": "be nice", "text": "be kind"}),
                    ),
                ),
            ),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert result.changed is True
    assert lm.tool_choices == ["auto", "auto"]
    assert {definition["function"]["name"] for definition in lm.tools[0]} == {
        tool.value for tool in EDIT_TOOL_SETS["broad"]
    }
    assert result.steps[0].assistant.startswith("Tool call: REPLACE_TEXT\nArguments: ")
    assert '"target": "be nice"' in result.steps[0].assistant
    assert not result.steps[0].assistant.startswith('{"role": "assistant"')
    assert all(definition["type"] == "function" for definition in lm.tools[0])
    assert "one provider-native function call" in lm.calls[0][0]["content"]
    assert "<tool_call>" not in lm.calls[0][0]["content"]


def test_explicit_tool_budget_disables_more_calls_but_allows_finish() -> None:
    """Tell the model it must finish once a bounded diagnostic spends its edit budget."""
    lm = NativeScriptedLM(
        [
            ToolCompletion("", (NativeToolCall("edit", "REPLACE_TEXT", '{"target":"be nice","text":"be kind"}'),)),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT, max_tool_calls=1, max_iterations=2)
    assert result.changed and result.tool_calls == 1
    assert lm.tool_choices == ["auto", "none"]
    assert "budget is exhausted" in lm.calls[1][-1]["content"]


def test_custom_callable_uses_explicit_text_tool_compatibility_protocol() -> None:
    """Retain a documented fallback for callables without native-tool support."""
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])
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
                    NativeToolCall(
                        "call-insert",
                        EditTool.INSERT_TEXT.value,
                        json.dumps({"anchor": "be nice", "where": "after", "text": " and wrong"}),
                    ),
                    NativeToolCall(
                        "call-replace",
                        EditTool.REPLACE_TEXT.value,
                        json.dumps({"target": "be nice", "text": "be wrong"}),
                    ),
                ),
                "reasoning that DeepSeek requires on the retry",
            ),
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "call-correct",
                        EditTool.REPLACE_TEXT.value,
                        json.dumps({"target": "be nice", "text": "be kind"}),
                    ),
                ),
            ),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT", "FINISH"]
    assert "received 2" in result.steps[0].error
    assert "wrong" not in result.new_text
    retry_messages = lm.calls[1]
    assistant_turn = next(message for message in retry_messages if message["role"] == "assistant")
    assert assistant_turn["reasoning_content"] == "reasoning that DeepSeek requires on the retry"
    assert [message["role"] for message in retry_messages[-2:]] == ["tool", "tool"]
    assert {message["tool_call_id"] for message in retry_messages[-2:]} == {"call-insert", "call-replace"}


def test_direct_deepseek_quotes_prior_branch_turns_outside_the_native_tool_conversation() -> None:
    """Avoid replaying assistant turns whose private reasoning was not retained."""
    history = [
        {"role": "assistant", "content": "Earlier edit attempt."},
        {"role": "user", "content": "Optimizer result: accepted."},
    ]
    lm = NativeScriptedLM(
        [
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "call-replace",
                        EditTool.REPLACE_TEXT.value,
                        json.dumps({"target": "be nice", "text": "be kind"}),
                    ),
                ),
                "reasoning for the current provider turn",
            ),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )
    lm.model = "deepseek/deepseek-v4-flash"

    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)

    assert result.changed is True
    assert [message["role"] for message in lm.calls[0]] == ["system", "user"]
    assert json.dumps(history, ensure_ascii=False) in lm.calls[0][-1]["content"]
    assert "not earlier turns in this provider tool conversation" in lm.calls[0][-1]["content"]


def test_other_native_providers_keep_branch_history_as_chat_messages() -> None:
    """Preserve ordinary branch-message replay when no provider rule forbids it."""
    history = [
        {"role": "assistant", "content": "Earlier edit attempt."},
        {"role": "user", "content": "Optimizer result: accepted."},
    ]
    lm = NativeScriptedLM(
        [
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "call-replace",
                        EditTool.REPLACE_TEXT.value,
                        json.dumps({"target": "be nice", "text": "be kind"}),
                    ),
                ),
            ),
            ToolCompletion("<finish>Done.</finish>", ()),
        ]
    )
    lm.model = "hosted_vllm/Qwen/Qwen3.8-27B"

    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)

    assert result.changed is True
    assert lm.calls[0][1:3] == history


def test_multiple_text_tool_blocks_are_rejected_before_retry() -> None:
    """Do not silently execute the first action from a multi-action text reply."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be wrong")
            + tool_call(EditTool.DELETE_TEXT, target="be brief"),
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
            "<finish>Done.</finish>",
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT", "FINISH"]
    assert "received 2" in result.steps[0].error
    assert "wrong" not in result.new_text
    assert "be brief" in result.new_text


def test_direct_semantic_action_rejects_a_different_available_tool() -> None:
    """Enforce semantic-action/tool coupling even when all four tools are exposed."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.INSERT_TEXT, anchor="be nice", where="after", text=" and kind"),
            tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
            "<finish>Done.</finish>",
        ]
    )
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "REPLACE_TEXT", "FINISH"]
    assert "coupled to REPLACE_TEXT" in result.steps[0].observation
    assert "and kind" not in result.new_text


def test_minimal_basis_composes_delete_and_insert_then_finishes() -> None:
    """Faithfully lower hidden REPLACE into delete, same-location insert, and finish."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            "<finish>The replacement is complete.</finish>",
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
    assert (
        "For each replacement, use one DELETE_TEXT call followed by one INSERT_TEXT call" in lm.calls[0][0]["content"]
    )


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
    assert result.steps[0].region_text == LOWERING_RULES
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
    assert "new|" not in result.steps[1].region_text
    broad = run(
        ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="old|", text="new|"), "<finish>Done.</finish>"]),
        component_text=LOWERING_PROMPT,
        preferred_tool=EditTool.REPLACE_TEXT,
    )
    assert result.new_text == broad.new_text


def test_minimal_replace_compares_the_exact_region() -> None:
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
    assert result.new_text == TEMPLATE.parse(component)["Rules"]
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


def test_minimal_replace_allows_another_pair_but_cannot_finish_halfway() -> None:
    """Validate each replacement against its own starting region before finish."""
    lm = ScriptedLM(
        [
            tool_call(EditTool.DELETE_TEXT, target="old|"),
            tool_call(EditTool.INSERT_TEXT, anchor="anchor", where="before", text="new|"),
            tool_call(EditTool.DELETE_TEXT, target="tail"),
            "<finish>Done.</finish>",
            tool_call(EditTool.INSERT_TEXT, anchor="anchor|", where="after", text="end"),
            "<finish>Done.</finish>",
        ]
    )

    result = run(
        lm,
        component_text=LOWERING_PROMPT,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        preferred_tool=EditTool.REPLACE_TEXT,
    )

    assert [step.action for step in result.steps] == [
        "DELETE_TEXT",
        "INSERT_TEXT",
        "DELETE_TEXT",
        "INVALID",
        "INSERT_TEXT",
        "FINISH",
    ]
    assert "incomplete" in result.steps[3].observation
    assert result.new_text == "new|anchor|end"
    assert result.tool_calls == 4


def test_minimal_move_native_calls_match_one_broad_move() -> None:
    """Lower hidden MOVE with exact deleted bytes and the direct move destination."""
    component = TEMPLATE.render({"Role": "helper", "Rules": "- one\n- two\n- three"})
    lm = NativeScriptedLM(
        [
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "delete",
                        EditTool.DELETE_TEXT.value,
                        json.dumps({"target": "- two\n"}),
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "insert",
                        EditTool.INSERT_TEXT.value,
                        json.dumps({"anchor": "- one\n", "where": "before", "text": "- two\n"}),
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
                        NativeToolCall(
                            "move",
                            EditTool.MOVE_TEXT.value,
                            json.dumps({"target": "- two\n", "anchor": "- one\n", "where": "before"}),
                        ),
                    ),
                ),
                ToolCompletion("<finish>Done.</finish>", ()),
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
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "delete",
                        EditTool.DELETE_TEXT.value,
                        json.dumps({"target": "- two\n"}),
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "wrong-text",
                        EditTool.INSERT_TEXT.value,
                        json.dumps({"anchor": "- one\n", "where": "before", "text": "- changed\n"}),
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "same-place",
                        EditTool.INSERT_TEXT.value,
                        json.dumps({"anchor": "- three", "where": "before", "text": "- two\n"}),
                    ),
                ),
            ),
            ToolCompletion(
                "",
                (
                    NativeToolCall(
                        "correct",
                        EditTool.INSERT_TEXT.value,
                        json.dumps({"anchor": "- one\n", "where": "before", "text": "- two\n"}),
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
    """Keep every broad-basis operator callable through one protocol.

    Args:
        tool: Broad edit operator under test.
        fields: Valid protocol fields for that operator.
    """
    lm = ScriptedLM([tool_call(tool, **fields), "<finish>Done.</finish>"])
    result = run(lm, preferred_tool=tool)
    assert result.changed is True
    assert result.tool_calls == 1
    assert result.steps[0].action == tool.value
    assert "## " not in result.new_text


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
            "<finish>Done.</finish>",
        ]
    )
    result = run(lm, preferred_tool=EditTool.INSERT_TEXT)
    assert [step.action for step in result.steps] == ["INVALID", "INSERT_TEXT", "FINISH"]
    assert "section body cannot contain" in result.steps[0].observation
    assert "## Reasoning" not in result.new_text
    assert "cite sources" in result.new_text


def test_selected_section_is_the_only_document_context() -> None:
    """Keep sibling sections out of the ReAct model's editing context."""
    component = TEMPLATE.render({"Role": "helper", "Context": "private context", "Rules": "- be nice\n- be brief"})
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])
    result = run(
        lm,
        preferred_tool=EditTool.REPLACE_TEXT,
        component_text=component,
    )
    assert result.new_text == "- be kind\n- be brief"
    initial_context = json.dumps(lm.calls[0], ensure_ascii=False)
    assert "- be nice" in initial_context
    assert "helper" not in initial_context
    assert "private context" not in initial_context


def test_insert_into_omitted_section_returns_the_new_body() -> None:
    """Populate an absent section body without receiving neighboring content."""
    component = TEMPLATE.render({"Rules": "- be brief"})
    role = EditTarget("sys", "Role")
    lm = ScriptedLM(
        [tool_call(EditTool.INSERT_TEXT, anchor="", where="after", text="helper"), "<finish>Done.</finish>"]
    )
    result = run(
        lm,
        preferred_tool=EditTool.INSERT_TEXT,
        component_text=component,
        edit_target=role,
    )
    assert result.new_text == "helper"
    assert "- be brief" not in json.dumps(lm.calls[0], ensure_ascii=False)


def test_delete_last_text_returns_an_empty_section_body() -> None:
    """Return an empty body when the selected section's last text is deleted."""
    component = TEMPLATE.render({"Reasoning": "Check the answer."})
    result = run(
        ScriptedLM([tool_call(EditTool.DELETE_TEXT, target="Check the answer."), "<finish>Done.</finish>"]),
        component_text=component,
        edit_target=EditTarget("sys", "Reasoning"),
        preferred_tool=EditTool.DELETE_TEXT,
    )
    assert result.changed is True
    assert result.new_text == ""


def test_insert_into_an_empty_section_returns_only_the_selected_body() -> None:
    """Allow the proposer to populate a section whose current body is empty."""
    result = run(
        ScriptedLM(
            [tool_call(EditTool.INSERT_TEXT, anchor="", where="after", text="helper"), "<finish>Done.</finish>"]
        ),
        component_text="",
        edit_target=EditTarget("sys", "Role"),
        preferred_tool=EditTool.INSERT_TEXT,
    )
    assert result.new_text == "helper"


def test_move_cannot_use_an_anchor_from_an_unselected_section() -> None:
    """Keep both ends of a MOVE inside the selected section."""
    result = run(
        ScriptedLM([tool_call(EditTool.MOVE_TEXT, target="- be nice", anchor="helper", where="after")]),
        edit_target=RULES,
        preferred_tool=EditTool.MOVE_TEXT,
        max_iterations=1,
    )
    assert result.changed is False
    assert result.new_text == RULES_TEXT
    assert result.steps[0].action == "INVALID"
    assert "anchor not found" in result.steps[0].observation


def test_finish_without_a_change_drops_the_proposal() -> None:
    """End an inapplicable action without accepting an unchanged candidate."""
    lm = ScriptedLM(["<finish>Nothing to do.</finish>"])
    result = run(lm, allowed_tools=EDIT_TOOL_SETS["minimal"])
    assert [step.action for step in result.steps] == ["FINISH"]
    assert not result.changed and result.new_text == RULES_TEXT
    assert result.tool_calls == 0 and "Nothing to do" in result.dropped_reason


def test_empty_section_error_restates_only_the_editable_body() -> None:
    """Recover when an empty-section MOVE mistakenly targets feedback text."""
    lm = NativeScriptedLM([
        ToolCompletion(content="", tool_calls=[NativeToolCall(
            id="move", name="MOVE_TEXT", arguments={"target": "feedback text", "anchor": "trace text", "where": "after"}
        )]),
        ToolCompletion(content="<finish>The selected body is empty, so nothing can move.</finish>", tool_calls=[]),
    ])
    result = run(lm, component_text=TEMPLATE.render({"Rules": ""}), preferred_tool=EditTool.MOVE_TEXT)
    assert not result.changed and result.new_text == ""
    assert [step.action for step in result.steps] == ["INVALID", "FINISH"]
    assert '0 characters; JSON string)\n""' in lm.calls[0][-1]["content"]
    assert '0 characters; JSON string):\n""' in result.steps[0].observation
    assert "not feedback or traces" in result.steps[0].observation


def test_manifestor_steering_is_delivered_in_the_current_user_message() -> None:
    """Keep Manifestor steering coupled to the current editing task."""
    user_lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])
    run(
        user_lm,
        preferred_tool=EditTool.REPLACE_TEXT,
        steering_message="User steering",
    )
    user_messages = user_lm.calls[0]
    assert [message["role"] for message in user_messages] == ["system", "user"]
    assert user_messages[-1]["content"].startswith("User steering\n\n## Selected component")


def test_branch_chat_history_is_replayed_before_the_current_task() -> None:
    """Replay only user/assistant turns supplied for this selected parent."""
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])
    history = [
        {"role": "assistant", "content": "<tool_call>parent-only</tool_call>"},
        {"role": "user", "content": "Optimizer feedback: REJECTED. That edit did not help."},
    ]
    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)
    assert lm.calls[0][1:3] == history
    assert "Branch-local conversation" in lm.calls[0][-1]["content"]
    assert result.steps[0].action == "REPLACE_TEXT"


def test_large_branch_history_is_replayed_without_truncation() -> None:
    """Preserve a long parent transcript byte-for-byte for the provider."""
    content = "x" * 12_001
    history = [{"role": "user", "content": content}]
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])

    result = run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)

    assert lm.calls[0][1] == history[0]
    assert lm.calls[0][1]["content"] == content
    assert result.changed is True


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
    """Reject provider or tool metadata in persisted branch transcripts.

    Args:
        history: Invalid history shape under test.
        error: Validation exception expected for that shape.
    """
    lm = ScriptedLM([])
    with pytest.raises(error, match="[Bb]ranch-local history"):
        run(lm, preferred_tool=EditTool.REPLACE_TEXT, history=history)
    assert lm.calls == []


def test_large_combined_context_reaches_the_lm_and_completes() -> None:
    """Let the configured provider model enforce its real context window."""
    history = [{"role": "assistant", "content": "h" * 20_000}]
    traces = "trajectory-step\n" * 3_000
    lm = ScriptedLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"), "<finish>Done.</finish>"])

    result = run(
        lm,
        preferred_tool=EditTool.REPLACE_TEXT,
        history=history,
        traces=traces,
    )

    serialized = json.dumps(lm.calls[0], ensure_ascii=False)
    assert len(serialized) > 64_000
    assert history[0] in lm.calls[0]
    assert traces in lm.calls[0][-1]["content"]
    assert result.changed is True


def test_iteration_exhaustion_drops_partial_atomic_edits() -> None:
    """Return the untouched parent when a multi-step proposal never finishes."""
    lm = ScriptedLM([tool_call(EditTool.DELETE_TEXT, target="be nice")])
    result = run(
        lm,
        allowed_tools=EDIT_TOOL_SETS["minimal"],
        max_iterations=1,
    )
    assert result.changed is False
    assert result.new_text == RULES_TEXT
    assert result.tool_calls == 1
    assert result.dropped_reason == "No completed revision within 1 ReAct V2 turns."


def test_max_chars_violation_is_returned_as_an_error_observation() -> None:
    """Keep an over-budget edit out of the completed candidate."""
    lm = ScriptedLM([tool_call(EditTool.INSERT_TEXT, anchor="be nice", where="after", text="x" * 100)])
    result = run(
        lm,
        preferred_tool=EditTool.INSERT_TEXT,
        max_iterations=1,
        max_chars=len(RULES_TEXT) + 5,
    )
    assert result.changed is False
    assert result.new_text == RULES_TEXT
    assert "exceeding max_chars" in result.steps[0].observation
