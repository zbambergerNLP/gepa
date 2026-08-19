# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for semantic actions and the Controller (:mod:`gepa.strategies.intervention`).

Exercises the intervention data model, the ``intervention_specs`` catalog,
controller-menu construction, the Controller's LM-driven selection, and feedback
summarization. The Controller LM is replaced by an in-file fake.

Expected usage:
```bash
pytest tests/test_intervention.py -vv
```
"""

# Standard library imports
import random

# Third-party imports
import pytest

# Local imports
from gepa.strategies.document_template import TEMPLATES, DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool
from gepa.strategies.intervention import (
    Controller,
    ControllerAction,
    InterventionSpec,
    build_controller_menu,
    intervention_specs,
    summarize_feedback,
)

# ====================== #
# Test Fakes and Helpers #
# ====================== #


PROMPT_TEMPLATE = TEMPLATES["prompt"]
PROMPT = PROMPT_TEMPLATE.render({"Role": "you are a helper", "Rules": "- be nice\n- be brief"})


class VotingLM:
    """A fake Controller LM: votes all probability mass on one menu id substring."""

    def __init__(self, prefer: str):
        self.prefer = prefer
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        # The action menu lists options as "- **<menu_id>**: <desc>"; vote for the
        # first id that contains `prefer`.
        chosen = None
        for line in prompt.splitlines():
            if line.startswith("- **") and self.prefer in line:
                chosen = line.split("**")[1]
                break
        chosen = chosen or self.prefer
        return (
            "<response><candidate>"
            f"<action>{chosen}</action><reasoning>r</reasoning><probability>1.0</probability>"
            "</candidate></response>"
        )


class TestControllerAction:
    """Test cases for ControllerAction."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "spec",
            "edit_tool",
            "expected_menu_id",
            "expected_menu_description",
        ],
        # Parameter values
        [
            pytest.param(
                InterventionSpec("tighten_rule", "Tighten a rule.", (EditTool.REPLACE_TEXT,), ("Rules",)),  # spec
                EditTool.REPLACE_TEXT,  # edit_tool
                "tighten_rule@Rules/REPLACE_TEXT",  # expected_menu_id
                "Tighten a rule. (region 'Rules', via REPLACE_TEXT)",  # expected_menu_description
                id="with_spec",
            ),
            pytest.param(
                None,  # spec
                EditTool.INSERT_TEXT,  # edit_tool
                "INSERT_TEXT@Rules",  # expected_menu_id
                "Edit region 'Rules' using INSERT_TEXT.",  # expected_menu_description
                id="without_spec",
            ),
        ],
    )
    def test_menu_id_and_description(
        self,
        spec: InterventionSpec | None,
        edit_tool: EditTool,
        expected_menu_id: str,
        expected_menu_description: str,
    ) -> None:
        """Test that ControllerAction derives its menu id and description from target, tool, and spec.

        Args:
            spec: The semantic action attached to the action, or None for an operator-only action.
            edit_tool: The atomic edit tool the action carries.
            expected_menu_id: The stable menu id the action must expose.
            expected_menu_description: The one-line menu description the action must expose.
        """
        action = ControllerAction(EditTarget("sys", "Rules"), edit_tool, spec)
        assert action.menu_id == expected_menu_id
        assert action.menu_description == expected_menu_description


class TestInterventionSpecs:
    """Test cases for intervention_specs."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "kind",
            "section",
            "expected_present",
            "expected_exact",
        ],
        # Parameter values
        [
            pytest.param(
                "prompt",  # kind
                "Rules",  # section
                {"add_constraint", "tighten_rule", "condense"},  # expected_present (Rules-specific + global)
                None,  # expected_exact
                id="prompt_rules_section_and_global",
            ),
            pytest.param(
                "prompt",  # kind
                None,  # section
                {"condense", "reorder"},  # expected_present
                {"condense", "reorder"},  # expected_exact (whole-target gets only global specs)
                id="prompt_whole_target_global_only",
            ),
            pytest.param(
                "prompt",  # kind
                "Context",  # section
                {"add_context", "clarify_context", "remove_stale_context"},  # expected_present
                None,  # expected_exact
                id="prompt_context_section",
            ),
            pytest.param(
                "prompt",  # kind
                "Reasoning",  # section
                {"add_reasoning_step", "strengthen_reasoning", "prune_reasoning"},  # expected_present
                None,  # expected_exact
                id="prompt_reasoning_section",
            ),
            pytest.param(
                "skill",  # kind
                "Instructions",  # section
                {"add_instruction", "tighten_instruction"},  # expected_present
                None,  # expected_exact
                id="skill_instructions_section",
            ),
            pytest.param(
                "mystery",  # kind
                "Rules",  # section
                set(),  # expected_present
                set(),  # expected_exact (unknown kind has no specs)
                id="unknown_kind_has_no_specs",
            ),
        ],
    )
    def test_specs_for_kind_and_section(
        self,
        kind: str,
        section: str | None,
        expected_present: set[str],
        expected_exact: set[str] | None,
    ) -> None:
        """Test that intervention_specs offers the section-scoped and global specs for a kind.

        Args:
            kind: The document kind whose catalog is queried.
            section: The section being edited, or None for a whole-document target.
            expected_present: Spec names that must appear (subset check).
            expected_exact: The full set of spec names, or None to skip the exact-equality check.
        """
        names = {spec.name for spec in intervention_specs(kind, section)}
        assert expected_present <= names
        if expected_exact is not None:
            assert names == expected_exact


class TestBuildControllerMenu:
    """Test cases for build_controller_menu."""

    def test_level1_pairs_targets_with_tools_no_specs(self) -> None:
        """Test that level 1 offers operator-only actions across every tool in the set."""
        menu = build_controller_menu(PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["minimal"], 1, rng=random.Random(0))
        assert all(a.intervention_spec is None for a in menu)
        tools = {a.edit_tool for a in menu}
        assert tools == {EditTool.INSERT_TEXT, EditTool.DELETE_TEXT}

    def test_level2_attaches_specs(self) -> None:
        """Test that level 2 attaches semantic specs to at least some actions."""
        menu = build_controller_menu(PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0))
        assert any(a.intervention_spec is not None for a in menu)

    def test_level2_only_offers_compatible_tools(self) -> None:
        """Test that every level-2 action pairs a spec with one of its compatible tools."""
        menu = build_controller_menu(
            PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0), max_menu=999
        )
        for action in menu:
            spec = action.intervention_spec
            assert spec is not None
            assert action.edit_tool in spec.compatible_tools

    def test_minimal_toolset_exposes_fewer_specs_than_broad(self) -> None:
        """Test that the minimal 2-op tool set exposes fewer specs than the broad 4-op set."""
        minimal = build_controller_menu(
            PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["minimal"], 2, rng=random.Random(0), max_menu=999
        )
        broad = build_controller_menu(
            PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0), max_menu=999
        )
        assert len(minimal) < len(broad)

    def test_menu_falls_back_to_whole_document_edit(self) -> None:
        """Test that a kind with no semantic catalog still yields one whole-document edit."""
        # A kind with no semantic catalog yields no level-2 options; the builder
        # still offers one whole-document edit so the Controller can act.
        note = DocumentTemplate("note", {"Body": "the text"})
        menu = build_controller_menu(note, "art", EDIT_TOOL_SETS["minimal"], 2, rng=random.Random(0))
        assert [(a.edit_target.section, a.intervention_spec) for a in menu] == [(None, None)]
        assert menu[0].edit_tool in EDIT_TOOL_SETS["minimal"]

    def test_menu_capped_at_max_menu(self) -> None:
        """Test that the menu is truncated to max_menu entries."""
        menu = build_controller_menu(
            PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0), max_menu=5
        )
        assert len(menu) == 5


class TestController:
    """Test cases for Controller."""

    def test_select_controller_maps_pick_back(self) -> None:
        """Test that the Controller maps the LM's vote back to the matching menu action."""
        menu = build_controller_menu(
            PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["broad"], 1, rng=random.Random(0), max_menu=999
        )
        lm = VotingLM(prefer="DELETE_TEXT@Rules")
        controller = Controller(menu, lm, rng=random.Random(0))
        controller.set_context(PROMPT, "too verbose")
        picks = controller.select_controller(1, random.Random(0))
        assert len(picks) == 1
        assert isinstance(picks[0], ControllerAction)
        assert picks[0].edit_tool == EditTool.DELETE_TEXT
        assert picks[0].edit_target.section == "Rules"
        assert lm.calls  # the Controller actually queried the LM

    def test_controller_records_history(self) -> None:
        """Test that selecting an action appends one entry to the Controller's history."""
        menu = build_controller_menu(PROMPT_TEMPLATE, "sys", EDIT_TOOL_SETS["minimal"], 1, rng=random.Random(0))
        controller = Controller(menu, VotingLM(prefer="INSERT_TEXT@Role"), rng=random.Random(0))
        controller.set_context(PROMPT, "fb")
        controller.select_controller(1, random.Random(0))
        assert len(controller.history) == 1


class TestSummarizeFeedback:
    """Test cases for summarize_feedback."""

    def test_concatenates_feedback_fields(self) -> None:
        """Test that feedback and execution-feedback fields are concatenated into the summary."""
        entries = [{"Feedback": "too vague"}, {"execution_feedback": "wrong format"}]
        summary = summarize_feedback(entries)
        assert "too vague" in summary
        assert "wrong format" in summary

    def test_empty_returns_placeholder(self) -> None:
        """Test that an empty feedback list returns the no-feedback placeholder."""
        assert summarize_feedback([]) == "(no feedback available)"

    def test_truncates_to_budget(self) -> None:
        """Test that an oversized summary is truncated to the char budget and ellipsized."""
        entries = [{"Feedback": "x" * 10_000}]
        summary = summarize_feedback(entries, max_chars=100)
        assert len(summary) <= 103  # 100 + "..."
        assert summary.endswith("...")
