"""Replay archived no-edit responses without making new model or metric calls."""

import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from gepa.lm import LM, NativeToolCall, ToolCompletion
from gepa.proposer.reflective_mutation.single_call_proposer import SingleCallProposer
from gepa.strategies.document_template import TEMPLATE_FAMILIES, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool

ARCHIVE = json.loads((Path(__file__).parent / "fixtures/forest_observed_noops.json").read_text())
CASES = ARCHIVE["cases"]


def run_case(case, lm, *, require_edit):
    section, operator = case["action_choice"].split("@", 1)[1].split("/")
    editor = SingleCallProposer(lm, TEMPLATE_FAMILIES["alibaba"]["system_prompt"], EDIT_TOOL_SETS["broad"])
    return editor.propose(
        case["region"],
        EditTarget(case["component"], section),
        EditTool(operator),
        case["manifestor_content"],
        "",
        "",
        [],
        None,
        require_edit=require_edit,
    )


def test_audit_counts_exclude_nonproposal_iterations_and_changed_proposals():
    assert len(CASES) == len({case["id"] for case in CASES}) == 140
    assert Counter(case["kind"] for case in CASES) == {
        "finish_without_edit": 123,
        "text_without_native_calls": 17,
    }
    for source, expected in [("jev_standard", 58), ("generative_expanded", 82)]:
        counts = ARCHIVE["sources"][source]["counts"]
        assert sum(case["id"].startswith(source) for case in CASES) == counts["no_edit"] == expected
        assert counts["proposal_cycles"] == counts["no_edit"] + counts["changed"]
        assert counts["iterations"] == counts["proposal_cycles"] + counts["nonproposal_iterations"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_archive_reproduces_no_edit_and_routes_recovery(case):
    """Use actual old responses; scripted correction measures execution, not quality."""
    lm = Mock(spec=LM)
    original = ToolCompletion(case["editor_content"], ())
    lm.complete_with_tools.return_value = original
    before = run_case(case, lm, require_edit=False)
    assert lm.complete_with_tools.call_count == 1
    assert not before.changed and before.new_text == case["region"]
    assert before.tool_calls == 0

    lm.reset_mock()
    if case["kind"] == "finish_without_edit":
        after = run_case(case, lm, require_edit=True)
        assert lm.complete_with_tools.call_count == 1
        assert not after.changed and after.steps[-1].action == "INVALID"
        assert "finish-only" in after.dropped_reason.lower()
    else:
        correction = ToolCompletion(
            "",
            tuple(
                NativeToolCall(str(i), call["name"], json.dumps(call["arguments"]))
                for i, call in enumerate(case["scripted_native_correction"])
            ),
        )
        lm.complete_with_tools.side_effect = [original, correction]
        after = run_case(case, lm, require_edit=True)
        assert lm.complete_with_tools.call_count == 2
        assert after.changed and after.new_text != case["region"]
        assert after.iterations == 2 and after.tool_calls == len(correction.tool_calls)
        assert after.steps[0].action == "INVALID" and case["editor_content"] in after.steps[0].assistant
        first, second = lm.complete_with_tools.call_args_list
        assert first.args[1] == second.args[1]
        assert len(second.args[1]) == 1
        before_task = json.loads(first.args[0][-1]["content"])
        after_task = json.loads(second.args[0][-1]["content"])
        assert "native_protocol_correction" in after_task
        assert {key: after_task[key] for key in before_task} == before_task


def test_native_correction_is_bounded_and_does_not_execute_xml_itself():
    case = next(case for case in CASES if case["kind"] == "text_without_native_calls")
    lm = Mock(spec=LM)
    lm.complete_with_tools.return_value = ToolCompletion(case["editor_content"], ())
    result = run_case(case, lm, require_edit=True)
    assert lm.complete_with_tools.call_count == 2
    assert not result.changed and result.new_text == case["region"]
    assert result.tool_calls == 0 and result.iterations == 2
    assert [step.action for step in result.steps] == ["INVALID", "INVALID"]


def test_bad_later_native_call_rolls_back_a_corrected_batch():
    case = next(case for case in CASES if case["kind"] == "text_without_native_calls")
    call = case["scripted_native_correction"][0]
    lm = Mock(spec=LM)
    lm.complete_with_tools.side_effect = [
        ToolCompletion(case["editor_content"], ()),
        ToolCompletion(
            "",
            (
                NativeToolCall("first", call["name"], json.dumps(call["arguments"])),
                NativeToolCall("second", "INSERT_TEXT", '{"anchor":"ABSENT ANCHOR","text":"x","where":"after"}'),
            ),
        ),
    ]
    result = run_case(case, lm, require_edit=True)
    assert not result.changed and result.new_text == case["region"] and not result.executed_edit
    assert result.iterations == 2
