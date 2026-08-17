# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Semantic actions and the Controller of the 3-role reflection architecture.

This module holds the user-facing semantic actions (:class:`InterventionSpec`)
and their per-document-kind catalog, the Controller's joint decision object
(:class:`ControllerAction`), the menu builder that enumerates the joint
``(EditTarget, EditTool, InterventionSpec)`` options for a text, and the
:class:`Controller` itself, which picks one by reusing GEPA's verbalized
sampling machinery (:class:`~gepa.strategies.action_space.VerbalizedActionSelector`).

The pieces it composes live in sibling modules: atomic edit operations in
:mod:`gepa.strategies.edit_tools` and the canonical section format in
:mod:`gepa.strategies.document_template`. The Manifestor and RLM Proposer live
under ``proposer/reflective_mutation`` and consume the objects defined here; the
whole extension is wired behind the existing ``reflection_strategy=`` seam via
``ThreeRoleReflectionLM``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.action_space import (
    SOFT_PROMPT_CHAR_BUDGET,
    PromptEditAction,
    VerbalizedActionSelector,
)
from gepa.strategies.document_template import DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EditTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterventionSpec:
    """A user-defined semantic action describing the *type* of change to pursue.

    Steers the Proposer toward a kind of revision without specifying wording or
    the low-level edit. ``applicable_sections`` scopes the spec (``None`` =
    global); ``compatible_tools`` lists the atomic tools that can realize it —
    the intersection with the offered tool set decides availability (so the
    minimal 2-op basis naturally exposes fewer specs than the broad 4-op set).

    Attributes:
        name: Stable identifier used in menu ids and logs (e.g. ``"add_constraint"``).
        description: One-line description of the change, shown to the Controller LM.
        compatible_tools: The :class:`EditTool` s that can realize this spec; the
            Controller only offers ``(spec, tool)`` pairs whose tool is in both
            this tuple and the configured tool set.
        applicable_sections: Section names the spec applies to, or ``None`` for
            a global spec offered on every section and the whole document.
    """

    name: str
    description: str
    compatible_tools: tuple[EditTool, ...]
    applicable_sections: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ControllerAction:
    """The Controller's joint decision: ``(EditTarget, EditTool, InterventionSpec)``.

    ``intervention_spec`` is ``None`` at reflection level 1 (operator only, no
    semantic steering).

    Attributes:
        edit_target: The region of the candidate the edit will modify.
        edit_tool: The atomic operation the Proposer must use on that region.
        intervention_spec: The semantic action to pursue, or ``None`` when the
            Controller only picks a region and tool.
    """

    edit_target: EditTarget
    edit_tool: EditTool
    intervention_spec: InterventionSpec | None

    @property
    def menu_id(self) -> str:
        """Stable id shown to the Controller LM and parsed back from its output.

        Returns:
            ``"<spec>@<region>/<TOOL>"`` when a spec is attached, otherwise
            ``"<TOOL>@<region>"``; unique within one Controller menu.
        """
        if self.intervention_spec is not None:
            return f"{self.intervention_spec.name}@{self.edit_target.name}/{self.edit_tool.value}"
        return f"{self.edit_tool.value}@{self.edit_target.name}"

    @property
    def menu_description(self) -> str:
        """One-line description combining the semantic action, region, and tool.

        Returns:
            The spec description annotated with the region and tool when a spec
            is attached, otherwise a generic ``"Edit region ... using ..."`` line.
        """
        tool = self.edit_tool.value
        region = self.edit_target.name
        if self.intervention_spec is not None:
            return f"{self.intervention_spec.description} (region '{region}', via {tool})"
        return f"Edit region '{region}' using {tool}."


# Per-kind, per-section semantic action catalogs. The "_global" key holds specs
# applicable to any section / the whole document.
_INTERVENTION_CATALOG: dict[str, dict[str, list[InterventionSpec]]] = {
    "prompt": {
        "Role": [
            InterventionSpec(
                "sharpen_role",
                "Sharpen the role/persona so it constrains behavior.",
                (EditTool.REPLACE_TEXT,),
                ("Role",),
            ),
            InterventionSpec(
                "add_expertise",
                "Add a relevant area of expertise to the role.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Role",),
            ),
        ],
        "Task": [
            InterventionSpec(
                "clarify_task", "Make the task description less ambiguous.", (EditTool.REPLACE_TEXT,), ("Task",)
            ),
            InterventionSpec(
                "add_step",
                "Add a missing step to the task procedure.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Task",),
            ),
        ],
        "Context": [
            InterventionSpec(
                "add_context",
                "Add missing background the task needs.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Context",),
            ),
            InterventionSpec(
                "clarify_context", "Clarify ambiguous or misleading context.", (EditTool.REPLACE_TEXT,), ("Context",)
            ),
            InterventionSpec(
                "remove_stale_context", "Remove distracting or stale context.", (EditTool.DELETE_TEXT,), ("Context",)
            ),
        ],
        "Rules": [
            InterventionSpec(
                "add_constraint",
                "Add a targeted rule that addresses a failure.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Rules",),
            ),
            InterventionSpec(
                "tighten_rule", "Tighten an existing rule to remove loopholes.", (EditTool.REPLACE_TEXT,), ("Rules",)
            ),
            InterventionSpec(
                "remove_conflicting_rule",
                "Remove a redundant or conflicting rule.",
                (EditTool.DELETE_TEXT,),
                ("Rules",),
            ),
        ],
        "Reasoning": [
            InterventionSpec(
                "add_reasoning_step",
                "Add an explicit reasoning directive.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Reasoning",),
            ),
            InterventionSpec(
                "strengthen_reasoning", "Strengthen reasoning guidance.", (EditTool.REPLACE_TEXT,), ("Reasoning",)
            ),
            InterventionSpec(
                "prune_reasoning", "Remove redundant reasoning guidance.", (EditTool.DELETE_TEXT,), ("Reasoning",)
            ),
        ],
        "Output Format": [
            InterventionSpec(
                "specify_format",
                "Specify the output format more precisely.",
                (EditTool.REPLACE_TEXT, EditTool.INSERT_TEXT),
                ("Output Format",),
            ),
            InterventionSpec(
                "add_format_constraint",
                "Add one format constraint the outputs violate.",
                (EditTool.INSERT_TEXT,),
                ("Output Format",),
            ),
        ],
        "Examples": [
            InterventionSpec(
                "add_illustration",
                "Add a worked example for a failing pattern.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Examples",),
            ),
            InterventionSpec(
                "prune_example", "Remove a misleading or redundant example.", (EditTool.DELETE_TEXT,), ("Examples",)
            ),
            InterventionSpec(
                "refine_example",
                "Refine an example so it teaches the right lesson.",
                (EditTool.REPLACE_TEXT,),
                ("Examples",),
            ),
        ],
        "_global": [
            InterventionSpec(
                "condense",
                "Condense redundant text without losing intent.",
                (EditTool.DELETE_TEXT, EditTool.REPLACE_TEXT),
            ),
            InterventionSpec("reorder", "Reorder content so critical information leads.", (EditTool.MOVE_TEXT,)),
        ],
    },
    "skill": {
        "Name": [
            InterventionSpec(
                "rename", "Rename the skill so it signals its purpose.", (EditTool.REPLACE_TEXT,), ("Name",)
            ),
        ],
        "Description": [
            InterventionSpec(
                "clarify_description",
                "Clarify what the skill does and when to use it.",
                (EditTool.REPLACE_TEXT,),
                ("Description",),
            ),
            InterventionSpec(
                "add_trigger_cue",
                "Add a cue describing when the skill applies.",
                (EditTool.INSERT_TEXT,),
                ("Description",),
            ),
        ],
        "Instructions": [
            InterventionSpec(
                "add_instruction",
                "Add an instruction addressing a failure.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Instructions",),
            ),
            InterventionSpec(
                "tighten_instruction", "Tighten a vague instruction.", (EditTool.REPLACE_TEXT,), ("Instructions",)
            ),
            InterventionSpec(
                "remove_redundant_step",
                "Remove a redundant instruction step.",
                (EditTool.DELETE_TEXT,),
                ("Instructions",),
            ),
        ],
        "Examples": [
            InterventionSpec(
                "add_illustration",
                "Add a worked example for a failing pattern.",
                (EditTool.INSERT_TEXT, EditTool.REPLACE_TEXT),
                ("Examples",),
            ),
            InterventionSpec(
                "prune_example", "Remove a misleading or redundant example.", (EditTool.DELETE_TEXT,), ("Examples",)
            ),
        ],
        "_global": [
            InterventionSpec(
                "condense",
                "Condense redundant text without losing intent.",
                (EditTool.DELETE_TEXT, EditTool.REPLACE_TEXT),
            ),
            InterventionSpec("reorder", "Reorder content so critical information leads.", (EditTool.MOVE_TEXT,)),
        ],
    },
}


def intervention_specs(kind: str, section: str | None) -> list[InterventionSpec]:
    """List the semantic actions available for one region of a document.

    A named section gets its own specs followed by the kind's ``_global`` specs;
    the whole-document region (``section is None``) gets only the global ones.
    Unknown kinds or sections have no specs of their own, so only whatever
    global specs the kind defines (possibly none) are returned.

    Args:
        kind: Document kind, e.g. ``"prompt"`` or ``"skill"``.
        section: Section name, or ``None`` for the whole document.

    Returns:
        The applicable :class:`InterventionSpec` s, section-specific first.
    """
    catalog = _INTERVENTION_CATALOG.get(kind, {})
    specs = list(catalog.get("_global", []))
    if section is not None:
        specs = list(catalog.get(section, [])) + specs
    return specs


def build_controller_menu(
    template: DocumentTemplate,
    component_name: str,
    edit_tools: list[EditTool],
    level: int,
    *,
    rng: random.Random,
    max_menu: int = 24,
) -> list[ControllerAction]:
    """Build the pruned joint menu of ``(target, tool, spec)`` options.

    Level 1 pairs each target with each offered tool (no semantic steering).
    Level 2 additionally scopes by :class:`InterventionSpec`, keeping only
    ``spec.compatible_tools`` that are offered — so the minimal 2-op basis
    exposes fewer options than the broad 4-op set.

    Args:
        template: Document template of the component (its sections are the targets).
        component_name: Name of the candidate component being edited.
        edit_tools: The offered tool set (one of :data:`EDIT_TOOL_SETS`).
        level: Reflection level; ``<= 1`` builds operator-only options,
            ``>= 2`` adds semantic :class:`InterventionSpec` steering.
        rng: RNG used to subsample an oversized menu deterministically.
        max_menu: Cap on menu size (LLM action-selection accuracy degrades on
            long menus); any excess is dropped by RNG sampling and logged,
            never silently.

    Returns:
        At least one :class:`ControllerAction`; a whole-document option with an
        offered tool is used as the fallback when nothing else applies.
    """
    targets = template.edit_targets(component_name)
    menu: list[ControllerAction] = []
    if level <= 1:
        for target in targets:
            for tool in edit_tools:
                menu.append(ControllerAction(target, tool, None))
    else:
        for target in targets:
            for spec in intervention_specs(template.kind, target.section):
                for tool in spec.compatible_tools:
                    if tool in edit_tools:
                        menu.append(ControllerAction(target, tool, spec))

    if not menu:
        # Guarantee at least one option (whole-document edit with an offered tool).
        fallback_tool = EditTool.REPLACE_TEXT if EditTool.REPLACE_TEXT in edit_tools else edit_tools[0]
        menu.append(ControllerAction(EditTarget(component_name, None), fallback_tool, None))

    if len(menu) > max_menu:
        logger.info(
            "Controller menu for '%s' has %d options; sampling %d (dropping %d) to keep selection reliable.",
            component_name,
            len(menu),
            max_menu,
            len(menu) - max_menu,
        )
        menu = rng.sample(menu, max_menu)
    return menu


class Controller(VerbalizedActionSelector):
    """Jointly selects a :class:`ControllerAction` by reusing verbalized sampling.

    Each joint option is presented to the reflection LM as a stand-in
    :class:`~gepa.strategies.action_space.PromptEditAction` (its ``menu_id`` /
    ``menu_description``); the inherited verbalized-distribution + tail-sampling
    machinery then picks one, and :meth:`select_controller` maps the pick back to
    its rich :class:`ControllerAction`.

    Args:
        menu: The joint options to choose from (see :func:`build_controller_menu`).
        lm: Language model asked to verbalize a distribution over the menu.
        k: Number of candidate options the LM is asked to assign probabilities to.
        tau: Tail-sampling threshold (options with ``p < tau`` form the tail);
            ``None`` defaults to ``1 / k``.
        rng: RNG for tail sampling; ``random.Random(0)`` when ``None``.
    """

    def __init__(
        self,
        menu: list[ControllerAction],
        lm: LanguageModel,
        *,
        k: int = 5,
        tau: float | None = None,
        rng: random.Random | None = None,
    ):
        """Wrap each menu option in a stand-in action and hand them to the verbalized selector."""
        stand_ins = [
            PromptEditAction(name=ca.menu_id, description=ca.menu_description, instruction_suffix="") for ca in menu
        ]
        super().__init__(actions=stand_ins, lm=lm, k=k, tau=tau, rng=rng)
        self._controller_by_id = {ca.menu_id: ca for ca in menu}

    def select_controller(self, n: int, rng: random.Random | None = None) -> list[ControllerAction]:
        """Select ``n`` joint controller actions.

        Args:
            n: Number of actions to draw (with replacement, one per job).
            rng: RNG override for this call; the instance RNG is used when ``None``.

        Returns:
            The chosen :class:`ControllerAction` s, in draw order. Falls back to
            uniform sampling if ``set_context()`` was not called first.
        """
        picks = self.select(n, rng)
        return [self._controller_by_id[p.name] for p in picks]


def summarize_feedback(reflective_entries: Any, max_chars: int = SOFT_PROMPT_CHAR_BUDGET) -> str:
    """Concatenate ``Feedback``/``execution_feedback`` fields into a bounded summary.

    Args:
        reflective_entries: Reflective-dataset rows for one component; each row
            is a mapping that may carry ``Feedback`` or ``execution_feedback``.
        max_chars: Truncation cap on the joined summary (``"..."`` is appended
            when truncated).

    Returns:
        The newline-joined feedback, or ``"(no feedback available)"`` when no
        row carries any.
    """
    parts: list[str] = []
    for entry in reflective_entries:
        fb = entry.get("Feedback") or entry.get("execution_feedback") or ""
        if fb:
            parts.append(str(fb))
    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."
    return summary or "(no feedback available)"
