# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Semantic actions and verbalized Controller for three-role reflection.

The Controller makes one joint document-region/semantic-action decision at
reflection level 2. An action expresses editorial intent and owns exactly one
direct text operator. ReAct V2 uses that operator directly when the broad tool
set exposes it; the minimal insert/delete basis decomposes replace and move
into their two atomic calls.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Literal, get_args

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.action_space import (
    FULL_SUPPORT_EXPLORATION_EPSILON,
    SOFT_PROMPT_CHAR_BUDGET,
    PromptEditAction,
    VerbalizedActionSelector,
)
from gepa.strategies.document_template import DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EditTool

logger = logging.getLogger(__name__)

InjectionSite = Literal["assistant_reasoning", "user", "system", "developer"]
INJECTION_SITES: tuple[InjectionSite, ...] = get_args(InjectionSite)


@dataclass(frozen=True)
class InterventionSpec:
    """Static semantic action selected by the Controller.

    Args:
        name: Stable action identifier.
        description: Menu text describing the intended revision.
        edit_tool: The one direct text operator coupled to this action.
        applicable_sections: Optional section-name restriction.
        allow_whole_document: Whether this action may target the whole document
            for a cross-section edit.
        instruction: Instruction the Manifestor realizes against run evidence.
        fixed_text: Literal steering text that bypasses the Manifestor LM.
        inject_as: Default injection site. Built-in actions use a user message.

    Raises:
        TypeError: ``edit_tool`` is not one :class:`EditTool` value.
        ValueError: The action does not define exactly one manifestation source
            or names an invalid site.
    """

    name: str
    description: str
    edit_tool: EditTool
    applicable_sections: tuple[str, ...] | None = None
    allow_whole_document: bool = False
    instruction: str | None = None
    fixed_text: str | None = None
    inject_as: InjectionSite = "user"

    def __post_init__(self) -> None:
        """Validate the direct-tool and manifestation contracts."""
        if not isinstance(self.edit_tool, EditTool):
            raise TypeError(f"InterventionSpec {self.name!r} edit_tool must be one EditTool value")
        if (self.instruction is None) == (self.fixed_text is None):
            raise ValueError(f"InterventionSpec {self.name!r} must set exactly one of instruction or fixed_text")
        if self.inject_as not in INJECTION_SITES:
            raise ValueError(f"InterventionSpec {self.name!r}: inject_as must be one of {INJECTION_SITES}")


@dataclass(frozen=True)
class Intervention:
    """Manifested steering text and the chat role where the proposer receives it.

    Args:
        text: Concrete steering message.
        inject_as: System, developer, user, or assistant-reasoning site.
    """

    text: str
    inject_as: InjectionSite = "user"


@dataclass(frozen=True)
class ControllerAction:
    """Controller decision over a region and optional semantic action.

    Args:
        edit_target: Independently selectable document section or whole document.
        intervention_spec: Semantic action, or ``None`` at level 1.
    """

    edit_target: EditTarget
    intervention_spec: InterventionSpec | None

    @property
    def edit_tool(self) -> EditTool | None:
        """Return the semantic action's structurally coupled operator."""
        return self.intervention_spec.edit_tool if self.intervention_spec is not None else None

    @property
    def menu_id(self) -> str:
        """Return the stable identifier shown to verbalized sampling.

        Returns:
            Semantic action/region/direct-tool identifier, or an atomic-basis
            region identifier at level 1.
        """
        if self.intervention_spec is not None:
            assert self.edit_tool is not None
            return f"{self.intervention_spec.name}@{self.edit_target.name}/{self.edit_tool.value}"
        return f"EDIT@{self.edit_target.name}"

    @property
    def menu_description(self) -> str:
        """Describe the region and semantic intent in one line.

        Returns:
            Controller menu description.
        """
        region = self.edit_target.name
        if self.intervention_spec is not None:
            assert self.edit_tool is not None
            return f"{self.intervention_spec.description} (region '{region}', direct tool {self.edit_tool.value})"
        return f"Revise region '{region}' using the available edit-tool basis."


SEMANTIC_ACTION_CATALOG_VERSION = 1
CONTROLLER_POLICY_VERSION = 2

SEMANTIC_ACTIONS: tuple[InterventionSpec, ...] = (
    InterventionSpec(
        "rephrase",
        "Rewrite unclear text without changing its meaning, requirements, scope, or level of detail.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one unclear passage with a clearer equivalent. Keep its meaning, requirements, scope, level of "
            "detail, and representation. Use reformat when the representation itself is the problem."
        ),
    ),
    InterventionSpec(
        "summarize",
        "Shorten a passage without losing an operative requirement or necessary fact.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one verbose passage with a materially shorter version. Keep every unique requirement, exception, "
            "dependency, and fact."
        ),
    ),
    InterventionSpec(
        "reformat",
        "Change a passage's representation without adding, removing, or moving information.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one hard-to-use passage with a clearer checklist, procedure, table, schema, field list, or other "
            "appropriate form. Keep every proposition, qualifier, requirement, and scope boundary. Use summarize when "
            "detail should be removed."
        ),
    ),
    InterventionSpec(
        "correct",
        "Correct false, contradictory, or procedurally wrong content whose purpose is still needed.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one factually, logically, or procedurally wrong passage with the correction supported by the "
            "failures. Keep the passage's purpose. Use specialize, generalize, strengthen_requirement, or "
            "relax_requirement when scope or normative force is the main defect."
        ),
    ),
    InterventionSpec(
        "specialize",
        "Narrow a passage that applies to too many cases.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one overbroad passage with a narrower operational statement that names the cases it covers. Use "
            "rephrase for ambiguity, expand for missing detail, or add_constraint when the original rule remains valid."
        ),
    ),
    InterventionSpec(
        "generalize",
        "Broaden an overfit or unjustifiably narrow passage while keeping its safeguards.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one passage that rejects valid alternatives or overfits the observed cases. State the broader "
            "invariant supported by the failures and retain the necessary safeguards."
        ),
    ),
    InterventionSpec(
        "strengthen_requirement",
        "Make a correct requirement less permissive without changing what or where it governs.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one permissive requirement with a stronger equivalent that prevents the diagnosed failure. Keep "
            "its subject and applicability unchanged."
        ),
    ),
    InterventionSpec(
        "relax_requirement",
        "Make an otherwise useful requirement less absolute without changing what or where it governs.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one overrestrictive requirement with a justified conditional, preferred, or permissive version. "
            "Keep its subject and applicability; delete it instead when none of its intent should survive."
        ),
    ),
    InterventionSpec(
        "expand",
        "Insert one missing statement, step, example, definition, or fact without changing existing text.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Insert one self-contained unit that resolves a diagnosed failure at an exact anchor. Leave existing text "
            "unchanged and use add_constraint for a boundary or guardrail."
        ),
    ),
    InterventionSpec(
        "add_constraint",
        "Keep a valid rule and insert one missing boundary, prohibition, condition, or guardrail.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Insert one precise constraint at an exact anchor where a valid rule permits a diagnosed bad case. Leave "
            "the rule unchanged; use specialize when the rule itself is overbroad."
        ),
    ),
    InterventionSpec(
        "remove_redundancy",
        "Delete one span whose full meaning already appears elsewhere.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Delete one duplicated or equivalent span only after locating the surviving passage that preserves all of "
            "its meaning and qualifiers."
        ),
    ),
    InterventionSpec(
        "remove_harmful_content",
        "Delete one harmful, obsolete, irrelevant, or non-operative span when none of its intent should remain.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Delete one exact span only after confirming that it contains no necessary meaning and needs no corrected "
            "replacement."
        ),
    ),
    InterventionSpec(
        "relocate",
        "Move one unchanged span when its placement is the only defect.",
        EditTool.MOVE_TEXT,
        allow_whole_document=True,
        instruction=(
            "Move one correctly worded span to a distinct exact anchor when its position harms dependency order, "
            "procedure, grouping, precedence, or salience. Preserve its bytes exactly."
        ),
    ),
)

if len({spec.name for spec in SEMANTIC_ACTIONS}) != len(SEMANTIC_ACTIONS):
    raise ValueError("Built-in semantic action names must be unique")

_INTERVENTION_CATALOG: dict[str, tuple[InterventionSpec, ...]] = {
    "prompt": SEMANTIC_ACTIONS,
    "skill": SEMANTIC_ACTIONS,
}


def intervention_specs(kind: str, section: str | None) -> list[InterventionSpec]:
    """List semantic actions available for one document region.

    Args:
        kind: Document kind, currently ``"prompt"`` or ``"skill"``.
        section: Region name, or ``None`` for the whole document.

    Returns:
        Actions whose optional section restriction includes this region.
    """
    specs = _INTERVENTION_CATALOG.get(kind, ())
    if section is None:
        return [spec for spec in specs if spec.allow_whole_document]
    return [spec for spec in specs if spec.applicable_sections is None or section in spec.applicable_sections]


def semantic_action_catalog(kind: str) -> dict[str, Any]:
    """Return the ordered, JSON-serializable semantic action contract.

    Args:
        kind: Document kind, currently ``"prompt"`` or ``"skill"``.

    Returns:
        Catalog version, kind, and full action definitions used for safe run
        resumption and reproducible analysis.
    """
    return {
        "version": SEMANTIC_ACTION_CATALOG_VERSION,
        "kind": kind,
        "actions": [
            {
                "name": spec.name,
                "operator": spec.edit_tool.value,
                "description": spec.description,
                "instruction": spec.instruction,
                "fixed_text": spec.fixed_text,
                "inject_as": spec.inject_as,
                "applicable_sections": list(spec.applicable_sections) if spec.applicable_sections is not None else None,
                "allow_whole_document": spec.allow_whole_document,
            }
            for spec in _INTERVENTION_CATALOG.get(kind, ())
        ],
    }


def controller_policy_contract() -> dict[str, Any]:
    """Return the level-2 semantic Controller's reproducibility contract."""
    return {
        "version": CONTROLLER_POLICY_VERSION,
        "factorization": "P(region, action)",
        "candidates": "all applicable region/action pairs",
        "verbalized_candidates": "all",
        "sampling": "verbalized distribution mixed with uniform exploration",
        "exploration_epsilon": FULL_SUPPORT_EXPLORATION_EPSILON,
        "max_menu": None,
    }


def build_controller_menu(
    template: DocumentTemplate,
    component_name: str,
    edit_tools: list[EditTool],
    level: int,
    *,
    rng: random.Random,
    max_menu: int | None = None,
) -> list[ControllerAction]:
    """Build the Controller's region/action menu.

    At level 1, each independently addressable region appears once. At level 2,
    every applicable region/action pair appears once so one verbalized-sampling
    call makes the complete semantic decision. The action's direct operator is
    derived from its :class:`InterventionSpec`; it is never a separate choice.

    Args:
        template: Document template whose sections become targets.
        component_name: Candidate component name.
        edit_tools: Configured execution basis. It must be non-empty even though
            semantic actions remain visible when their direct broad tool is absent.
        level: Reflection level.
        rng: Seeded RNG for optional deterministic menu subsampling.
        max_menu: Optional level-1 region bound. At level 2 it may be set only
            high enough to retain every applicable region/action pair.

    Returns:
        Non-empty Controller menu.

    Raises:
        ValueError: ``edit_tools`` is empty, ``max_menu`` is below one, or a
            level-2 bound would remove an applicable region.
    """
    if not edit_tools:
        raise ValueError("Controller requires at least one edit tool.")
    if max_menu is not None and max_menu < 1:
        raise ValueError("max_menu must be at least 1.")

    targets = template.edit_targets(component_name)
    if level >= 2:
        menu = [
            action
            for target in targets
            for action in build_semantic_action_menu(template, target)
        ]
    else:
        menu = [ControllerAction(target, None) for target in targets]

    if not menu and level >= 2:
        raise ValueError(
            f"Document kind {template.kind!r} has no semantic actions; level 2 supports only cataloged kinds."
        )
    if not menu:
        menu = [ControllerAction(EditTarget(component_name, None), None)]
    if max_menu is not None and len(menu) > max_menu:
        if level >= 2:
            raise ValueError(
                f"max_menu={max_menu} would remove semantic Controller choices; level 2 requires all "
                f"{len(menu)} region/action pairs."
            )
        logger.info(
            "Controller menu for '%s' has %d options; sampling %d and dropping %d.",
            component_name,
            len(menu),
            max_menu,
            len(menu) - max_menu,
        )
        menu = rng.sample(menu, max_menu)
    return menu


def build_semantic_action_menu(template: DocumentTemplate, edit_target: EditTarget) -> list[ControllerAction]:
    """Build every semantic action applicable to one sampled region.

    Args:
        template: Document template that defines the selected region.
        edit_target: Region already sampled by the first Controller stage.

    Returns:
        Complete conditional semantic-action menu in catalog order.
    """
    return [ControllerAction(edit_target, spec) for spec in intervention_specs(template.kind, edit_target.section)]


class Controller(VerbalizedActionSelector):
    """Select one region/semantic-action option by verbalized sampling.

    Args:
        menu: Rich Controller actions.
        lm: Model that verbalizes the option distribution.
        k: Number of options assigned probabilities.
        tau: Tail-sampling threshold, or ``None`` for ``1 / k``.
        rng: Seeded sampling RNG.
        require_full_support: Require every option exactly once and mix the
            verbalized distribution with uniform exploration.
    """

    def __init__(
        self,
        menu: list[ControllerAction],
        lm: LanguageModel,
        *,
        k: int = 5,
        tau: float | None = None,
        rng: random.Random | None = None,
        require_full_support: bool = False,
    ):
        """Wrap rich actions as selector-compatible menu entries."""
        stand_ins = [
            PromptEditAction(name=action.menu_id, description=action.menu_description, instruction_suffix="")
            for action in menu
        ]
        super().__init__(
            actions=stand_ins,
            lm=lm,
            k=k,
            tau=tau,
            rng=rng,
            require_full_support=require_full_support,
        )
        self._controller_by_id = {action.menu_id: action for action in menu}

    def select_controller(self, n: int, rng: random.Random | None = None) -> list[ControllerAction]:
        """Draw rich Controller actions from the verbalized distribution.

        Args:
            n: Number of draws with replacement.
            rng: Optional call-specific RNG.

        Returns:
            Selected rich actions in draw order.
        """
        picks = self.select(n, rng)
        return [self._controller_by_id[pick.name] for pick in picks]


def summarize_feedback(reflective_entries: Any, max_chars: int = SOFT_PROMPT_CHAR_BUDGET) -> str:
    """Concatenate reflective feedback into a bounded summary.

    Args:
        reflective_entries: Rows carrying ``Feedback`` or ``execution_feedback``.
        max_chars: Maximum joined feedback length.

    Returns:
        Feedback text, or a no-feedback marker.
    """
    parts: list[str] = []
    for entry in reflective_entries:
        feedback = entry.get("Feedback") or entry.get("execution_feedback") or ""
        if feedback:
            parts.append(str(feedback))
    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."
    return summary or "(no feedback available)"
