# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for semantic interventions and verbalized Controller selection."""

import random

import pytest

from gepa.strategies.document_template import TEMPLATE_FAMILIES, TEMPLATES, DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool
from gepa.strategies.intervention import (
    INJECTION_SITES,
    SEMANTIC_ACTIONS,
    Controller,
    Intervention,
    InterventionSpec,
    build_controller_menu,
    build_semantic_action_menu,
    controller_policy_contract,
    intervention_specs,
    semantic_action_catalog,
    summarize_feedback,
)

PROMPT_TEMPLATE = TEMPLATES["system_prompt"]
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
            if line.startswith("- ") and ": " in line and self.preferred in line:
                chosen = line[2:].split(": ", 1)[0]
                break
        return (
            "<response><candidate>"
            f"<action>{chosen}</action><reasoning>test</reasoning><probability>1.0</probability>"
            "</candidate></response>"
        )


def test_builtin_catalog_is_the_curated_operator_coupled_action_space() -> None:
    """Expose each semantic action with one structural operator."""
    expected = {
        "rephrase": EditTool.REPLACE_TEXT,
        "summarize": EditTool.REPLACE_TEXT,
        "reformat": EditTool.REPLACE_TEXT,
        "correct": EditTool.REPLACE_TEXT,
        "specialize": EditTool.REPLACE_TEXT,
        "generalize": EditTool.REPLACE_TEXT,
        "strengthen_requirement": EditTool.REPLACE_TEXT,
        "relax_requirement": EditTool.REPLACE_TEXT,
        "expand": EditTool.INSERT_TEXT,
        "add_constraint": EditTool.INSERT_TEXT,
        "remove_redundancy": EditTool.DELETE_TEXT,
        "remove_harmful_content": EditTool.DELETE_TEXT,
        "relocate": EditTool.MOVE_TEXT,
    }
    assert {spec.name: spec.edit_tool for spec in SEMANTIC_ACTIONS} == expected
    guidance = {spec.name: f"{spec.description} {spec.instruction}".lower() for spec in SEMANTIC_ACTIONS}
    assert "use specialize, generalize" in guidance["correct"]
    assert "use rephrase" in guidance["specialize"]
    assert "valid rule" in guidance["add_constraint"]
    assert "use add_constraint" in guidance["expand"]
    for template in (TEMPLATES["system_prompt"], TEMPLATES["skill"]):
        kind = template.kind
        for section in template.sections:
            specs = intervention_specs(kind, section)
            assert {spec.name: spec.edit_tool for spec in specs} == expected
            assert all(spec.instruction and spec.fixed_text is None for spec in specs)
        assert [spec.name for spec in intervention_specs(kind, None)] == ["relocate"]


def test_semantic_action_catalog_persists_the_full_ordered_contract() -> None:
    """Make action-space changes part of benchmark identity and safe resume."""
    catalog = semantic_action_catalog("prompt")
    assert catalog["version"] == 1
    assert catalog["kind"] == "prompt"
    assert [(action["name"], action["operator"]) for action in catalog["actions"]] == [
        (spec.name, spec.edit_tool.value) for spec in SEMANTIC_ACTIONS
    ]
    assert all(action["description"] and action["instruction"] for action in catalog["actions"])
    assert all(action["fixed_text"] is None for action in catalog["actions"])
    assert all(action["inject_as"] == "user" for action in catalog["actions"])
    assert [action["name"] for action in catalog["actions"] if action["allow_whole_document"]] == ["relocate"]
    assert semantic_action_catalog("skill")["actions"] == catalog["actions"]
    assert controller_policy_contract()["factorization"] == "P(region, action)"
    assert controller_policy_contract()["exploration_epsilon"] == pytest.approx(0.1)


def test_unknown_document_kind_has_no_semantic_catalog() -> None:
    """Avoid silently applying prompt semantics to an undeclared kind."""
    assert intervention_specs("memo", "Body") == []


@pytest.mark.parametrize("site", INJECTION_SITES)
def test_intervention_spec_accepts_every_supported_injection_site(site: str) -> None:
    """Keep the public manifestation-site contract stable."""
    spec = InterventionSpec(
        "custom",
        "Custom action.",
        EditTool.INSERT_TEXT,
        instruction="Manifest this action.",
        inject_as=site,
    )
    assert spec.inject_as == site


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"edit_tool": EditTool.INSERT_TEXT},
            "exactly one of instruction or fixed_text",
            id="missing_manifestation_source",
        ),
        pytest.param(
            {
                "edit_tool": EditTool.INSERT_TEXT,
                "instruction": "instruction",
                "fixed_text": "fixed",
            },
            "exactly one of instruction or fixed_text",
            id="two_manifestation_sources",
        ),
        pytest.param(
            {
                "edit_tool": EditTool.INSERT_TEXT,
                "instruction": "instruction",
                "inject_as": "tool",
            },
            "inject_as must be one of",
            id="unknown_injection_site",
        ),
        pytest.param(
            {"edit_tool": (EditTool.INSERT_TEXT,), "instruction": "instruction"},
            "must be one EditTool value",
            id="plural_operator",
        ),
    ],
)
def test_intervention_spec_rejects_invalid_contracts(kwargs: dict[str, object], message: str) -> None:
    """Reject specs that cannot be executed or manifested unambiguously."""
    with pytest.raises((TypeError, ValueError), match=message):
        InterventionSpec("custom", "Custom action.", **kwargs)


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


def test_level2_jointly_selects_region_and_operator_coupled_semantic_action() -> None:
    """Expose every applicable region/action pair in one Controller menu."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
        max_menu=999,
    )
    expected_count = len(PROMPT_TEMPLATE.sections) * len(SEMANTIC_ACTIONS) + 1
    assert len(menu) == expected_count
    assert len({action.menu_id for action in menu}) == expected_count
    assert all(action.intervention_spec is not None for action in menu)
    local = [action for action in menu if action.edit_target.section == "Rules"]
    whole = [action for action in menu if action.edit_target.section is None]
    assert [action.intervention_spec.name for action in local if action.intervention_spec] == [
        spec.name for spec in SEMANTIC_ACTIONS
    ]
    assert [action.intervention_spec.name for action in whole if action.intervention_spec] == ["relocate"]

    action_menu = build_semantic_action_menu(PROMPT_TEMPLATE, EditTarget("sys", "Rules"))
    assert len(action_menu) == len(SEMANTIC_ACTIONS)
    for action in action_menu:
        assert action.intervention_spec is not None
        assert action.edit_tool is action.intervention_spec.edit_tool
        assert action.menu_id.endswith(f"/{action.edit_tool.value}")
        assert "direct tool" in action.menu_description

    whole_menu = build_semantic_action_menu(PROMPT_TEMPLATE, EditTarget("sys", None))
    assert [action.intervention_spec.name for action in whole_menu if action.intervention_spec] == ["relocate"]


def test_openai_controller_builds_the_complete_joint_menu() -> None:
    """Score every applicable region/action pair in the single Controller pass."""
    template = TEMPLATE_FAMILIES["openai"]["system_prompt"]
    menu = build_controller_menu(
        template,
        "system_prompt",
        EDIT_TOOL_SETS["broad"],
        level=2,
        rng=random.Random(0),
    )

    expected_count = len(template.sections) * len(SEMANTIC_ACTIONS) + 1
    assert len(menu) == expected_count == 53
    assert len({action.menu_id for action in menu}) == expected_count


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
    assert any(action.edit_tool is EditTool.REPLACE_TEXT for action in minimal)


def test_unknown_kind_level2_fails_before_manifestation() -> None:
    """Reject uncataloged document kinds instead of inventing semantic actions."""
    template = DocumentTemplate("memo", {"Body": "memo body"})
    with pytest.raises(ValueError, match="Document kind 'memo' has no semantic actions"):
        build_controller_menu(
            template,
            "memo",
            EDIT_TOOL_SETS["minimal"],
            2,
            rng=random.Random(0),
        )


def test_controller_maps_verbalized_pick_back_to_semantic_action() -> None:
    """Return the rich action selected through the stand-in action menu."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
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
    """Allow level-1 bounds but never remove level-2 semantic regions."""
    with pytest.raises(ValueError, match="at least 1"):
        build_controller_menu(
            PROMPT_TEMPLATE,
            "sys",
            EDIT_TOOL_SETS["broad"],
            2,
            rng=random.Random(0),
            max_menu=0,
        )
    with pytest.raises(ValueError, match="requires all"):
        build_controller_menu(
            PROMPT_TEMPLATE,
            "sys",
            EDIT_TOOL_SETS["broad"],
            2,
            rng=random.Random(7),
            max_menu=5,
        )
    first = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        1,
        rng=random.Random(7),
        max_menu=5,
    )
    second = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        1,
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
