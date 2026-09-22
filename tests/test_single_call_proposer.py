"""Verify atomic single-response editing without semantic correction requests."""

import json
from unittest.mock import Mock

import pytest

from gepa.lm import LM, NativeToolCall, ToolCompletion
from gepa.proposer.reflective_mutation.single_call_proposer import SingleCallProposer
from gepa.strategies.document_template import TEMPLATES, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool


def native(tool, **arguments):
    return NativeToolCall("edit", tool.value, json.dumps(arguments))


def propose(calls=(), content="", *, region="alpha beta", selected=EditTool.REPLACE_TEXT):
    lm = Mock(spec=LM)
    lm.complete_with_tools.return_value = ToolCompletion(content, tuple(calls))
    editor = SingleCallProposer(lm, TEMPLATES["system_prompt"], EDIT_TOOL_SETS["broad"])
    result = editor.propose(region, EditTarget("sys", "Rules"), selected, "Revise this section", "", "", [], None)
    assert lm.complete_with_tools.call_count == 1
    assert len(lm.complete_with_tools.call_args.args[1]) == 4
    assert result.iterations == 1
    return result


def test_batch_uses_updated_section_in_response_order():
    result = propose(
        [
            native(EditTool.REPLACE_TEXT, target="alpha", text="gamma"),
            native(EditTool.REPLACE_TEXT, target="gamma beta", text="complete"),
        ]
    )
    assert result.changed and result.new_text == "complete"
    assert result.tool_calls == 2
    assert [s.action for s in result.steps] == ["REPLACE_TEXT", "REPLACE_TEXT"]


def test_invalid_later_call_rolls_back_the_entire_batch():
    result = propose(
        [
            native(EditTool.REPLACE_TEXT, target="alpha", text="gamma"),
            native(EditTool.REPLACE_TEXT, target="missing", text="complete"),
        ]
    )
    assert not result.changed and result.new_text == "alpha beta"
    assert not result.executed_edit
    assert result.steps[-1].action == "INVALID"


@pytest.mark.parametrize(
    "calls,content",
    [
        ([native(EditTool.DELETE_TEXT, target="alpha")], ""),
        ([native(EditTool.REPLACE_TEXT, target="alpha", text="gamma")], "<finish>Done.</finish>"),
        ([], "Please try again."),
    ],
)
def test_invalid_action_is_scored_as_a_discard_without_retry(calls, content):
    result = propose(calls, content)
    assert not result.changed and result.dropped_reason


def test_empty_selected_target_stays_a_legitimate_noop():
    result = propose(content="<finish>No existing target to delete.</finish>", region="", selected=EditTool.DELETE_TEXT)
    assert not result.changed and result.tool_calls == 0
    assert result.steps[0].action == "FINISH"


def test_empty_insert_and_net_unchanged_batch_are_allowed():
    inserted = propose(
        [native(EditTool.INSERT_TEXT, anchor="", text="new", where="after")], region="", selected=EditTool.INSERT_TEXT
    )
    assert inserted.changed and inserted.new_text == "new"
    unchanged = propose([native(EditTool.REPLACE_TEXT, target="alpha", text="alpha")])
    assert not unchanged.changed and unchanged.tool_calls == 1


def test_new_document_header_cannot_escape_the_section():
    result = propose([native(EditTool.REPLACE_TEXT, target="alpha", text="\n## Role\nChanged role")])
    assert not result.changed and result.new_text == "alpha beta"


def test_text_protocol_also_completes_without_a_finish_turn():
    lm = Mock(return_value="<tool_call><tool>REPLACE_TEXT</tool><target>alpha</target><text>beta</text></tool_call>")
    del lm.complete_with_tools
    editor = SingleCallProposer(lm, TEMPLATES["system_prompt"], EDIT_TOOL_SETS["broad"])
    result = editor.propose("alpha", EditTarget("sys", "Rules"), EditTool.REPLACE_TEXT, None, "", "", [], None)
    assert result.changed and result.new_text == "beta"
    assert lm.call_count == 1
