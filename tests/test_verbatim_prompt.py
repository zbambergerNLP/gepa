"""Exercise lossless prompt editing through FOREST's real tool and contract paths."""

import json
from unittest.mock import Mock

import pytest

from gepa.lm import LM, NativeToolCall, ToolCompletion
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2Proposer
from gepa.proposer.reflective_mutation.single_call_proposer import SingleCallProposer
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM, ensure_reflection_run_contract
from gepa.strategies.document_template import DocumentTemplate, EditTarget, VerbatimPromptTemplate
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool


@pytest.mark.parametrize(
    "text", ["", " \t\r\n", "no header", "\n## Existing\r\n{{tools}}\n## Existing\n\u05e9\u05dc\u05d5\u05dd  "]
)
def test_virtual_region_round_trips_without_normalization(text):
    template = VerbatimPromptTemplate()
    assert template.render(template.parse(text)) == text
    start, end = template.section_body_span(text, "Instructions")
    assert text[start:end] == text
    assert template.replace_section_body(text, "Instructions", text) == text
    with pytest.raises(KeyError):
        template.replace_section_body(text, "instance_template", "outside scope")


@pytest.mark.parametrize("editor_type", [ReActV2Proposer, SingleCallProposer])
@pytest.mark.parametrize(
    "tool, arguments, expected",
    [
        (EditTool.INSERT_TEXT, {"anchor": "alpha", "text": "new ", "where": "before"}, "new alpha beta"),
        (EditTool.DELETE_TEXT, {"target": "alpha "}, "beta"),
        (EditTool.REPLACE_TEXT, {"target": "alpha", "text": "gamma"}, "gamma beta"),
        (EditTool.MOVE_TEXT, {"target": "alpha", "anchor": "beta", "where": "after"}, " betaalpha"),
    ],
)
def test_native_edits_preserve_existing_headings_and_unselected_bytes(editor_type, tool, arguments, expected):
    """Both editors operate on literal text rather than requiring canonical headings."""
    prefix, suffix = " \n## Existing heading\n{{fixed_placeholder}}\n", "\n\n  "
    text = prefix + "alpha beta" + suffix
    lm = Mock(spec=LM)
    lm.complete_with_tools.side_effect = [
        ToolCompletion("", (NativeToolCall("edit", tool.value, json.dumps(arguments)),)),
        ToolCompletion("<finish>Complete.</finish>", ()),
    ]
    template = VerbatimPromptTemplate()
    editor = editor_type(lm, template, EDIT_TOOL_SETS["broad"], max_iterations=2)
    result = editor.propose(text, EditTarget("system_prompt", "Instructions"), tool, None, "", "", [], None)
    assert result.changed
    assert template.replace_section_body(text, "Instructions", result.new_text) == prefix + expected + suffix
    system_message = lm.complete_with_tools.call_args_list[0].args[0][0]["content"]
    assert "virtual region" in system_message
    assert "Other messages, tools and workflow are fixed" in system_message
    assert "Never write" not in system_message


def test_representation_drift_rejects_reflection_resume_even_with_the_same_region_names(tmp_path):
    """A structured one-section schema cannot resume a verbatim prompt run."""
    verbatim = VerbatimPromptTemplate()
    lm = Mock(spec=LM)
    kwargs = {"level": 1, "base_lm": lm, "base_lm_run_identity": {"model": "offline"}}
    text = "## Instructions\nsame seed\n"
    first = ThreeRoleReflectionLM(**kwargs, templates={"system_prompt": verbatim})
    structured = ThreeRoleReflectionLM(
        **kwargs, templates={"system_prompt": DocumentTemplate("prompt", dict(verbatim.sections))}
    )
    contract = first.run_contract({"system_prompt": text})
    ensure_reflection_run_contract(str(tmp_path), contract)
    assert contract["templates"]["system_prompt"]["format"] == "verbatim_single_region_v1"
    with pytest.raises(ValueError, match="contract"):
        ensure_reflection_run_contract(str(tmp_path), structured.run_contract({"system_prompt": text}))
    lm.assert_not_called()
