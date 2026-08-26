# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for canonical semantic actions and verbalized Controller selection."""

import random
from dataclasses import FrozenInstanceError

import pytest

from gepa.strategies.document_template import TEMPLATE_FAMILIES, TEMPLATES, DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOG_VERSION,
    SEMANTIC_ACTION_CATALOGS,
    SEMANTIC_ACTIONS,
    STATELESS_ACTION_MENU_VERSION,
    Controller,
    ControllerChoice,
    SemanticActionSpec,
    StatelessActionConstraint,
    build_controller_menu,
    format_stateless_action_constraint,
    summarize_feedback,
)

PROMPT_TEMPLATE = TEMPLATES["system_prompt"]
PROMPT = PROMPT_TEMPLATE.render({"Role": "helper", "Rules": "- be kind\n- be brief"})


class VotingLM:
    """Select the first Controller option containing a requested substring."""

    def __init__(self, preferred: str):
        """Configure the menu-id fragment this fake model should select.

        Args:
            preferred: Substring used to find the desired Controller option.
        """
        self.preferred = preferred
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Select a matching menu item from one Controller prompt.

        Args:
            prompt: Verbalized-action prompt containing the Controller menu.

        Returns:
            XML containing a one-candidate probability distribution.
        """
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
        "contextualize": EditTool.INSERT_TEXT,
        "prune_context": EditTool.DELETE_TEXT,
        "revise_context": EditTool.REPLACE_TEXT,
        "supplant_context": EditTool.REPLACE_TEXT,
        "resequence": EditTool.MOVE_TEXT,
        "reexpress": EditTool.REPLACE_TEXT,
        "restrict_meaning": EditTool.REPLACE_TEXT,
        "relax_meaning": EditTool.REPLACE_TEXT,
        "revise_meaning": EditTool.REPLACE_TEXT,
        "supplant_meaning": EditTool.REPLACE_TEXT,
    }
    assert {spec.name: spec.edit_tool for spec in SEMANTIC_ACTIONS} == expected
    assert [spec.name for spec in SEMANTIC_ACTIONS] == list(expected)
    guidance = {spec.name: f"{spec.description} {spec.instruction}".lower() for spec in SEMANTIC_ACTIONS}
    assert "operative meaning" in guidance["contextualize"]
    assert "proper superset" in guidance["contextualize"]
    assert "operative meaning" in guidance["prune_context"]
    assert "proper subset" in guidance["prune_context"]
    assert "overlap without containment" in guidance["revise_context"]
    assert "no supporting proposition" in guidance["supplant_context"]
    for name in ("restrict_meaning", "relax_meaning", "revise_meaning", "supplant_meaning"):
        assert "supporting context" in guidance[name]
    for template in (TEMPLATES["system_prompt"], TEMPLATES["user_prompt"], TEMPLATES["skill"]):
        kind = template.kind
        assert kind in SEMANTIC_ACTION_CATALOGS
        specs = SEMANTIC_ACTIONS
        assert {spec.name: spec.edit_tool for spec in specs} == expected
        assert all(spec.instruction and spec.fixed_text is None for spec in specs)
    resequence = next(spec for spec in SEMANTIC_ACTIONS if spec.name == "resequence")
    assert "current section" in (resequence.instruction or "")


@pytest.mark.parametrize("template_name", ["system_prompt", "user_prompt", "skill"])
def test_every_document_role_derives_the_same_actions_for_each_named_section(template_name: str) -> None:
    """Apply the canonical action space to system, user, and skill sections.

    Args:
        template_name: Registry key for the document-role template under test.
    """
    template = TEMPLATES[template_name]
    menu = [
        StatelessActionConstraint(spec, section, template)
        for section in template.sections
        for spec in SEMANTIC_ACTIONS
    ]
    for section in template.sections:
        assert [choice.semantic_action for choice in menu if choice.target_section == section] == list(SEMANTIC_ACTIONS)


def test_semantic_action_catalog_persists_the_full_ordered_contract() -> None:
    """Make action-space changes part of benchmark identity and safe resume."""
    catalog = SEMANTIC_ACTION_CATALOGS["prompt"]
    assert catalog["version"] == 2
    assert catalog["kind"] == "prompt"
    assert [(action["name"], action["operator"]) for action in catalog["actions"]] == [
        (spec.name, spec.edit_tool.value) for spec in SEMANTIC_ACTIONS
    ]
    assert all(action["description"] and action["instruction"] for action in catalog["actions"])
    assert all(action["fixed_text"] is None for action in catalog["actions"])
    assert set(catalog["actions"][0]) == {
        "name",
        "operator",
        "description",
        "instruction",
        "fixed_text",
    }
    assert SEMANTIC_ACTION_CATALOGS["skill"]["actions"] == catalog["actions"]
    assert CONTROLLER_POLICY_CONTRACT["factorization"] == "P(region, action)"
    assert CONTROLLER_POLICY_CONTRACT["candidates"] == "all cataloged region/action pairs"
    assert CONTROLLER_POLICY_CONTRACT["exploration_epsilon"] == pytest.approx(0.1)


def test_stateless_menu_derives_every_binding_from_the_canonical_catalog() -> None:
    """Cross canonical actions with sections without defining a second action space."""
    menu = [
        StatelessActionConstraint(spec, section, PROMPT_TEMPLATE)
        for section in PROMPT_TEMPLATE.sections
        for spec in SEMANTIC_ACTIONS
    ]
    expected_count = len(PROMPT_TEMPLATE.sections) * len(SEMANTIC_ACTIONS)
    assert len(menu) == expected_count
    assert all(isinstance(choice, StatelessActionConstraint) for choice in menu)
    assert len({choice.menu_id for choice in menu}) == expected_count
    assert {choice.target_section for choice in menu} == set(PROMPT_TEMPLATE.sections)
    assert all(choice.document_template is PROMPT_TEMPLATE for choice in menu)

    rules = [choice for choice in menu if choice.target_section == "Rules"]
    assert [choice.semantic_action for choice in rules] == list(SEMANTIC_ACTIONS)
    assert all(choice.edit_tool is choice.semantic_action.edit_tool for choice in menu)


def test_stateless_constraint_and_contract_preserve_region_action_operator_binding() -> None:
    """Render and serialize exact bindings while retaining semantic action identity."""
    menu = [
        StatelessActionConstraint(spec, section, PROMPT_TEMPLATE)
        for section in PROMPT_TEMPLATE.sections
        for spec in SEMANTIC_ACTIONS
    ]
    choice = next(
        choice
        for choice in menu
        if choice.semantic_action.name == "reexpress" and choice.target_section == "Rules"
    )
    suffix = format_stateless_action_constraint(choice)
    assert "Make exactly one semantic edit: reexpress" in suffix
    assert "Coupled text operator: REPLACE_TEXT" in suffix
    assert "only the body of the selected 'Rules' section" in suffix
    assert "without a '## <Section>' header" in suffix
    assert "Do not reference or modify any other section" in suffix

    contract = {
        "version": STATELESS_ACTION_MENU_VERSION,
        "semantic_action_catalog_version": SEMANTIC_ACTION_CATALOG_VERSION,
        "kind": PROMPT_TEMPLATE.kind,
        "sections": list(PROMPT_TEMPLATE.sections),
        "choices": [
            {
                "id": item.menu_id,
                "semantic_action": item.semantic_action.name,
                "operator": item.edit_tool.value,
                "target_section": item.target_section,
            }
            for item in menu
        ],
    }
    assert contract["version"] == STATELESS_ACTION_MENU_VERSION
    assert contract["semantic_action_catalog_version"] == SEMANTIC_ACTION_CATALOG_VERSION
    assert contract["kind"] == "prompt"
    assert contract["sections"] == list(PROMPT_TEMPLATE.sections)
    assert contract["choices"] == [
        {
            "id": item.menu_id,
            "semantic_action": item.semantic_action.name,
            "operator": item.edit_tool.value,
            "target_section": item.target_section,
        }
        for item in menu
    ]


def test_unknown_document_kind_has_no_semantic_catalog() -> None:
    """Avoid silently applying prompt semantics to an undeclared kind."""
    assert "memo" not in SEMANTIC_ACTION_CATALOGS


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
            {"edit_tool": (EditTool.INSERT_TEXT,), "instruction": "instruction"},
            "must be one EditTool value",
            id="plural_operator",
        ),
    ],
)
def test_semantic_action_spec_rejects_invalid_contracts(kwargs: dict[str, object], message: str) -> None:
    """Reject specs that cannot be executed or manifested unambiguously.

    Args:
        kwargs: Invalid constructor arguments for the semantic action.
        message: Expected validation-error message fragment.
    """
    with pytest.raises((TypeError, ValueError), match=message):
        SemanticActionSpec("custom", "Custom action.", **kwargs)


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
    assert [action.edit_target.section for action in menu] == list(PROMPT_TEMPLATE.sections)
    assert all(action.edit_tool is None for action in menu)
    assert all(action.semantic_action is None for action in menu)
    assert all(action.menu_id == f"EDIT@{action.edit_target.section}" for action in menu)
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
    """Expose every cataloged region/action pair in one Controller menu."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
        max_menu=999,
    )
    expected_count = len(PROMPT_TEMPLATE.sections) * len(SEMANTIC_ACTIONS)
    assert len(menu) == expected_count
    assert len({action.menu_id for action in menu}) == expected_count
    assert all(action.semantic_action is not None for action in menu)
    assert {action.edit_target.section for action in menu} == set(PROMPT_TEMPLATE.sections)
    local = [action for action in menu if action.edit_target.section == "Rules"]
    assert [action.semantic_action.name for action in local if action.semantic_action] == [
        spec.name for spec in SEMANTIC_ACTIONS
    ]

    edit_target = EditTarget("sys", "Rules")
    action_menu = [ControllerChoice(edit_target, spec) for spec in SEMANTIC_ACTIONS]
    assert len(action_menu) == len(SEMANTIC_ACTIONS)
    for action in action_menu:
        assert action.semantic_action is not None
        assert action.edit_tool is action.semantic_action.edit_tool
        assert action.menu_id.endswith(f"/{action.edit_tool.value}")
        assert "direct tool" in action.menu_description


def test_action_operator_bindings_are_immutable() -> None:
    """Keep cached menu fields coupled to their canonical semantic action."""
    spec = SEMANTIC_ACTIONS[0]
    choice = ControllerChoice(EditTarget("sys", "Rules"), spec)
    constraint = StatelessActionConstraint(spec, "Rules", PROMPT_TEMPLATE)

    with pytest.raises(FrozenInstanceError):
        choice.edit_tool = EditTool.DELETE_TEXT
    with pytest.raises(FrozenInstanceError):
        constraint.menu_id = "contradictory"

def test_openai_controller_builds_the_complete_joint_menu() -> None:
    """Score every cataloged region/action pair in the single Controller pass."""
    template = TEMPLATE_FAMILIES["openai"]["system_prompt"]
    menu = build_controller_menu(
        template,
        "system_prompt",
        EDIT_TOOL_SETS["broad"],
        level=2,
        rng=random.Random(0),
    )

    expected_count = len(template.sections) * len(SEMANTIC_ACTIONS)
    assert len(menu) == expected_count == 40
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


def test_controller_rejects_a_template_without_named_sections() -> None:
    """Do not fall back to a whole-document target for an empty schema."""
    with pytest.raises(ValueError, match="no named sections"):
        build_controller_menu(
            DocumentTemplate("prompt", {}),
            "sys",
            EDIT_TOOL_SETS["minimal"],
            1,
            rng=random.Random(0),
        )


def test_controller_selects_rich_choice_directly() -> None:
    """Pass rich choices through verbalized sampling without stand-ins or lookup."""
    menu = build_controller_menu(
        PROMPT_TEMPLATE,
        "sys",
        EDIT_TOOL_SETS["broad"],
        2,
        rng=random.Random(0),
    )
    lm = VotingLM("reexpress@Rules/")
    controller = Controller(actions=menu, lm=lm, rng=random.Random(0))
    selected = controller.select(
        1,
        random.Random(0),
        candidate=PROMPT,
        feedback_summary="The answer repeats itself.",
    )[0]
    assert isinstance(selected, ControllerChoice)
    assert selected is next(action for action in menu if action.menu_id == selected.menu_id)
    assert controller.actions == menu
    assert selected.edit_target == EditTarget("sys", "Rules")
    assert selected.semantic_action is not None
    assert selected.semantic_action.name == "reexpress"
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
