# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for semantic interventions and verbalized Controller selection."""

import random

import pytest

from gepa.strategies.document_template import TEMPLATE_FAMILIES, TEMPLATES, DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool
from gepa.strategies.intervention import (
    INJECTION_SITES,
    Controller,
    ControllerAction,
    Intervention,
    InterventionSpec,
    build_controller_menu,
    intervention_specs,
    summarize_feedback,
)

PROMPT_TEMPLATE = TEMPLATES["prompt"]
PROMPT = PROMPT_TEMPLATE.render({"Role": "helper", "Rules": "- be kind\n- be brief"})


class VotingLM:
    """Select the first Controller option containing a requested substring."""

    def __init__(self, preferred: str):
        """Store the preferred menu-id fragment."""
        self.preferred = preferred
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Return a one-candidate verbalized distribution."""
        self.calls.append(prompt)
        chosen = self.preferred
        for line in prompt.splitlines():
            if line.startswith("- **") and self.preferred in line:
                chosen = line.split("**", 2)[1]
                break
        return (
            "<response><candidate>"
            f"<action>{chosen}</action><reasoning>test</reasoning><probability>1.0</probability>"
            "</candidate></response>"
        )


def test_builtin_catalog_is_exactly_three_direct_bound_semantic_actions() -> None:
    """Expose rephrase, summarize, and expand with one direct tool each."""
    expected = {
        "rephrase": EditTool.REPLACE_TEXT,
        "summarize": EditTool.REPLACE_TEXT,
        "expand": EditTool.INSERT_TEXT,
    }
    for kind in ("prompt", "skill"):
        for section in (*TEMPLATES[kind].sections, None):
            specs = intervention_specs(kind, section)
            assert {spec.name: spec.edit_tool for spec in specs} == expected
            assert all(len(spec.compatible_tools) == 1 for spec in specs)
            assert all(spec.instruction and spec.fixed_text is None for spec in specs)


def test_unknown_document_kind_has_no_semantic_catalog() -> None:
    """Avoid silently applying prompt semantics to an undeclared kind."""
    assert intervention_specs("memo", "Body") == []


@pytest.mark.parametrize("site", INJECTION_SITES)
def test_intervention_spec_accepts_every_supported_injection_site(site: str) -> None:
    """Keep the public manifestation-site contract stable."""
    spec = InterventionSpec(
        "custom",
        "Custom action.",
        (EditTool.INSERT_TEXT,),
        instruction="Manifest this action.",
        inject_as=site,
    )
    assert spec.inject_as == site


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({}, "at least one direct edit tool", id="missing_direct_tool"),
        pytest.param(
            {"compatible_tools": (EditTool.INSERT_TEXT,)},
            "exactly one of instruction or fixed_text",
            id="missing_manifestation_source",
        ),
        pytest.param(
            {
                "compatible_tools": (EditTool.INSERT_TEXT,),
                "instruction": "instruction",
                "fixed_text": "fixed",
            },
            "exactly one of instruction or fixed_text",
            id="two_manifestation_sources",
        ),
        pytest.param(
            {
                "compatible_tools": (EditTool.INSERT_TEXT,),
                "instruction": "instruction",
                "inject_as": "tool",
            },
            "inject_as must be one of",
            id="unknown_injection_site",
        ),
    ],
)
def test_intervention_spec_rejects_invalid_contracts(kwargs: dict[str, object], message: str) -> None:
    """Reject specs that cannot be executed or manifested unambiguously."""
    construction = dict(kwargs)
    compatible_tools = construction.pop("compatible_tools", ())
    with pytest.raises(ValueError, match=message):
        InterventionSpec("custom", "Custom action.", compatible_tools, **construction)


def test_intervention_remains_a_text_and_role_value_object() -> None:
    """Preserve the small public object passed from Manifestor to ReAct V2."""
    assert Intervention("Steer this edit.", "developer") == Intervention("Steer this edit.", "developer")


def test_level1_menu_selects_regions_only_and_not_tools() -> None:
    """Present each addressable region once while keeping tools in the execution basis."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "system_prompt",
        EDIT_TOOL_SETS["minimal"],
        1,
        rng=random.Random(0),
        max_menu=999,
    )
    assert [action.edit_target.section for action in menu] == [*PROMPT_TEMPLATE.sections, None]
    assert all(action.edit_tool is None for action in menu)
    assert all(action.intervention_spec is None for action in menu)
    assert all(action.menu_id == f"EDIT@{action.edit_target.name}" for action in menu)
    assert len({action.menu_id for action in menu}) == len(menu)


def test_level1_menu_is_independent_of_atomic_basis_size() -> None:
    """Do not make the Controller choose tools by creating a region/tool cross-product."""
    minimal = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["minimal"],
        1,
        rng=random.Random(0),
        max_menu=999,
    )
    broad = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        1,
        rng=random.Random(0),
        max_menu=999,
    )
    assert [action.menu_id for action in minimal] == [action.menu_id for action in broad]


def test_level2_menu_couples_each_semantic_action_directly_to_one_tool() -> None:
    """Avoid a second independent tool choice or a semantic/tool cross-product."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
        max_menu=999,
    )
    expected_count = len(PROMPT_TEMPLATE.edit_targets("sys")) * 3
    assert len(menu) == expected_count
    for action in menu:
        assert action.intervention_spec is not None
        assert action.edit_tool is action.intervention_spec.edit_tool
        assert action.menu_id.endswith(f"/{action.edit_tool.value}")
        assert "direct tool" in action.menu_description


def test_default_menu_keeps_every_gpt56_region_action_pair() -> None:
    """Do not silently discard supported built-in Controller choices."""
    template = TEMPLATE_FAMILIES["openai-gpt-5.6"]["prompt"]
    menu = build_controller_menu(
        template,
        "system_prompt",
        EDIT_TOOL_SETS["broad"],
        level=2,
        rng=random.Random(0),
    )

    assert len(menu) == (len(template.sections) + 1) * 3 == 27
    assert len({action.menu_id for action in menu}) == 27


def test_level2_semantic_menu_does_not_disappear_under_minimal_basis() -> None:
    """Let ReAct V2 compose insert/delete when a semantic direct tool is hidden."""
    minimal = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["minimal"],
        2,
        rng=random.Random(0),
        max_menu=999,
    )
    broad = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
        max_menu=999,
    )
    assert [action.menu_id for action in minimal] == [action.menu_id for action in broad]


def test_unknown_kind_level2_falls_back_to_one_whole_document_atomic_action() -> None:
    """Keep a custom template editable without inventing semantic actions."""
    template = DocumentTemplate("memo", {"Body": "memo body"})
    menu = build_controller_menu(
        template,
        "memo",
        EDIT_TOOL_SETS["minimal"],
        2,
        rng=random.Random(0),
    )
    assert menu == [ControllerAction(EditTarget("memo", None), None, None)]


def test_controller_maps_verbalized_pick_back_to_semantic_action() -> None:
    """Return the rich action selected through the stand-in action menu."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
        max_menu=999,
    )
    lm = VotingLM("summarize@Rules/")
    controller = Controller(menu, lm, rng=random.Random(0))
    controller.set_context(PROMPT, "The answer repeats itself.")
    selected = controller.select_controller(1, random.Random(0))[0]
    assert selected.edit_target == EditTarget("sys", "Rules")
    assert selected.intervention_spec is not None
    assert selected.intervention_spec.name == "summarize"
    assert selected.edit_tool is EditTool.REPLACE_TEXT
    assert len(controller.history) == 1
    assert lm.calls


def test_controller_menu_bounds_are_validated_and_deterministic() -> None:
    """Bound large menus through the supplied seeded random stream."""
    with pytest.raises(ValueError, match="at least 1"):
        build_controller_menu(
            PROMPT_TEMPLATE,
            "sys",
            EDIT_TOOL_SETS["broad"],
            2,
            rng=random.Random(0),
            max_menu=0,
        )
    first = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(7),
        max_menu=5,
    )
    second = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(7),
        max_menu=5,
    )
    assert [action.menu_id for action in first] == [action.menu_id for action in second]


def test_feedback_summary_supports_both_feedback_fields_and_a_hard_bound() -> None:
    """Ground actions in both dataset shapes without unbounded prompt growth."""
    summary = summarize_feedback(
        [{"Feedback": "too vague"}, {"execution_feedback": "wrong format"}, {"Feedback": "x" * 200}],
        max_chars=50,
    )
    assert summary.startswith("too vague\nwrong format")
    assert len(summary) == 53
    assert summary.endswith("...")
    assert summarize_feedback([]) == "(no feedback available)"
