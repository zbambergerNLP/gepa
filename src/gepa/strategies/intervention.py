# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Define the shared semantic action space and its execution-path bindings.

Every semantic action owns one direct text operator. Stateless reflection binds
the actions to template sections as prompt constraints. Three-role reflection
binds them to Controller choices before Manifestor steering and ReAct V2 or RLM
execution. The broad tool set exposes each operator directly; the minimal
insert/delete basis decomposes replace and move into atomic calls.
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
    VerbalizedActionSelector,
)
from gepa.strategies.document_template import DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EditTool

logger = logging.getLogger(__name__)

InjectionSite = Literal["assistant_reasoning", "user", "system", "developer"]
INJECTION_SITES: tuple[InjectionSite, ...] = get_args(InjectionSite)


@dataclass(frozen=True)
class SemanticActionSpec:
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
            raise TypeError(f"SemanticActionSpec {self.name!r} edit_tool must be one EditTool value")
        if (self.instruction is None) == (self.fixed_text is None):
            raise ValueError(f"SemanticActionSpec {self.name!r} must set exactly one of instruction or fixed_text")
        if self.inject_as not in INJECTION_SITES:
            raise ValueError(f"SemanticActionSpec {self.name!r}: inject_as must be one of {INJECTION_SITES}")


@dataclass(frozen=True)
class SteeringMessage:
    """Manifested steering text and the chat role where the proposer receives it.

    Args:
        text: Concrete steering message.
        inject_as: System, developer, user, or assistant-reasoning site.
    """

    text: str
    inject_as: InjectionSite = "user"


@dataclass(frozen=True)
class ControllerChoice:
    """Controller decision over a region and optional semantic action.

    Args:
        edit_target: Independently selectable document section or whole document.
        semantic_action: Semantic action, or ``None`` at level 1.
    """

    edit_target: EditTarget
    semantic_action: SemanticActionSpec | None

    @property
    def edit_tool(self) -> EditTool | None:
        """Return the semantic action's structurally coupled operator."""
        return self.semantic_action.edit_tool if self.semantic_action is not None else None

    @property
    def menu_id(self) -> str:
        """Return the stable identifier shown to verbalized sampling.

        Returns:
            Semantic action/region/direct-tool identifier, or an atomic-basis
            region identifier at level 1.
        """
        if self.semantic_action is not None:
            assert self.edit_tool is not None
            return f"{self.semantic_action.name}@{self.edit_target.name}/{self.edit_tool.value}"
        return f"EDIT@{self.edit_target.name}"

    @property
    def menu_description(self) -> str:
        """Describe the region and semantic intent in one line.

        Returns:
            Controller menu description.
        """
        region = self.edit_target.name
        if self.semantic_action is not None:
            assert self.edit_tool is not None
            return f"{self.semantic_action.description} (region '{region}', direct tool {self.edit_tool.value})"
        return f"Revise region '{region}' using the available edit-tool basis."


SEMANTIC_ACTION_CATALOG_VERSION = 1
CONTROLLER_POLICY_VERSION = 2
STATELESS_ACTION_MENU_VERSION = 1

SEMANTIC_ACTIONS: tuple[SemanticActionSpec, ...] = (
    SemanticActionSpec(
        "rephrase",
        "Rewrite unclear text without changing its meaning, requirements, scope, or level of detail.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one unclear passage with a clearer equivalent. Keep its meaning, requirements, scope, level of "
            "detail, and representation. Use reformat when the representation itself is the problem."
        ),
    ),
    SemanticActionSpec(
        "summarize",
        "Shorten a passage without losing an operative requirement or necessary fact.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one verbose passage with a materially shorter version. Keep every unique requirement, exception, "
            "dependency, and fact."
        ),
    ),
    SemanticActionSpec(
        "reformat",
        "Change a passage's representation without adding, removing, or moving information.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one hard-to-use passage with a clearer checklist, procedure, table, schema, field list, or other "
            "appropriate form. Keep every proposition, qualifier, requirement, and scope boundary. Use summarize when "
            "detail should be removed."
        ),
    ),
    SemanticActionSpec(
        "correct",
        "Correct false, contradictory, or procedurally wrong content whose purpose is still needed.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one factually, logically, or procedurally wrong passage with the correction supported by the "
            "failures. Keep the passage's purpose. Use specialize, generalize, strengthen_requirement, or "
            "relax_requirement when scope or normative force is the main defect."
        ),
    ),
    SemanticActionSpec(
        "specialize",
        "Narrow a passage that applies to too many cases.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one overbroad passage with a narrower operational statement that names the cases it covers. Use "
            "rephrase for ambiguity, expand for missing detail, or add_constraint when the original rule remains valid."
        ),
    ),
    SemanticActionSpec(
        "generalize",
        "Broaden an overfit or unjustifiably narrow passage while keeping its safeguards.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one passage that rejects valid alternatives or overfits the observed cases. State the broader "
            "invariant supported by the failures and retain the necessary safeguards."
        ),
    ),
    SemanticActionSpec(
        "strengthen_requirement",
        "Make a correct requirement less permissive without changing what or where it governs.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one permissive requirement with a stronger equivalent that prevents the diagnosed failure. Keep "
            "its subject and applicability unchanged."
        ),
    ),
    SemanticActionSpec(
        "relax_requirement",
        "Make an otherwise useful requirement less absolute without changing what or where it governs.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one overrestrictive requirement with a justified conditional, preferred, or permissive version. "
            "Keep its subject and applicability; delete it instead when none of its intent should survive."
        ),
    ),
    SemanticActionSpec(
        "expand",
        "Insert one missing statement, step, example, definition, or fact without changing existing text.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Insert one self-contained unit that resolves a diagnosed failure at an exact anchor. Leave existing text "
            "unchanged and use add_constraint for a boundary or guardrail."
        ),
    ),
    SemanticActionSpec(
        "add_constraint",
        "Keep a valid rule and insert one missing boundary, prohibition, condition, or guardrail.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Insert one precise constraint at an exact anchor where a valid rule permits a diagnosed bad case. Leave "
            "the rule unchanged; use specialize when the rule itself is overbroad."
        ),
    ),
    SemanticActionSpec(
        "remove_redundancy",
        "Delete one span whose full meaning already appears elsewhere.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Delete one duplicated or equivalent span only after locating the surviving passage that preserves all of "
            "its meaning and qualifiers."
        ),
    ),
    SemanticActionSpec(
        "remove_harmful_content",
        "Delete one harmful, obsolete, irrelevant, or non-operative span when none of its intent should remain.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Delete one exact span only after confirming that it contains no necessary meaning and needs no corrected "
            "replacement."
        ),
    ),
    SemanticActionSpec(
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

_SEMANTIC_ACTION_KINDS = frozenset({"prompt", "skill"})


def semantic_action_specs(kind: str, section: str | None) -> list[SemanticActionSpec]:
    """List semantic actions available for one document region.

    Args:
        kind: Document kind, currently ``"prompt"`` or ``"skill"``.
        section: Region name, or ``None`` for the whole document.

    Returns:
        Actions whose optional section restriction includes this region.
    """
    specs = SEMANTIC_ACTIONS if kind in _SEMANTIC_ACTION_KINDS else ()
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
            for spec in (SEMANTIC_ACTIONS if kind in _SEMANTIC_ACTION_KINDS else ())
        ],
    }


@dataclass(frozen=True)
class StatelessActionConstraint:
    """Bind one canonical semantic action to a stateless edit region.

    Stateless reflection emits a complete revised document instead of calling
    an edit tool. This value keeps that baseline on the same semantic action
    catalog and records the operator whose effect the full-document rewrite
    must realize.

    Args:
        semantic_action: Canonical semantic operation and coupled edit tool.
        target_section: Named section to revise, or ``None`` for the complete
            document when the action permits a cross-section edit.
    """

    semantic_action: SemanticActionSpec
    target_section: str | None

    @property
    def edit_tool(self) -> EditTool:
        """Return the semantic action's coupled text operator."""
        return self.semantic_action.edit_tool

    @property
    def menu_id(self) -> str:
        """Return a unique selector identifier for this action/region pair."""
        region = self.target_section if self.target_section is not None else "whole"
        return f"{self.semantic_action.name}@{region}/{self.edit_tool.value}"

    @property
    def menu_description(self) -> str:
        """Describe the semantic action, target region, and coupled operator."""
        region = f"section '{self.target_section}'" if self.target_section is not None else "the whole document"
        return f"{self.semantic_action.description} (target {region}, direct tool {self.edit_tool.value})"


def build_stateless_action_menu(template: DocumentTemplate) -> list[StatelessActionConstraint]:
    """Build the stateless baseline menu from the canonical action catalog.

    Args:
        template: Document template whose named sections become edit targets.

    Returns:
        Every applicable named-section/action pair followed by permitted
        whole-document choices, all in deterministic template/catalog order.
    """
    named_choices = [
        StatelessActionConstraint(spec, section)
        for section in template.sections
        for spec in semantic_action_specs(template.kind, section)
    ]
    whole_document_choices = [
        StatelessActionConstraint(spec, None) for spec in semantic_action_specs(template.kind, None)
    ]
    return [*named_choices, *whole_document_choices]


def format_stateless_action_constraint(action: StatelessActionConstraint) -> str:
    """Render one canonical action/region choice as a reflection constraint.

    Args:
        action: Semantic action bound to its selected target region.

    Returns:
        Constraint suffix suitable for a stateless full-document proposer.
    """
    if action.target_section is None:
        section_scope = "Apply this cross-section edit to the whole document."
    else:
        section_scope = (
            f"Apply this edit only within the '## {action.target_section}' section. "
            "If that section is currently omitted because it is empty, add its header in template order only when "
            "this edit adds content. Reproduce every other section verbatim, including its header."
        )
    manifestation = action.semantic_action.instruction or action.semantic_action.fixed_text
    assert manifestation is not None
    return (
        "\n\n--- Edit constraint ---\n"
        f"Make exactly one semantic edit: {action.semantic_action.name}\n"
        f"Description: {action.semantic_action.description}\n"
        f"Coupled text operator: {action.edit_tool.value}\n"
        f"{section_scope}\n"
        f"Guidance: {manifestation}\n\n"
        f"Length budget: the complete revised document must stay under {SOFT_PROMPT_CHAR_BUDGET} characters. "
        "If this edit would exceed the budget, replace or remove existing content instead of adding.\n\n"
        "Make no other changes."
    )


def stateless_action_menu_contract(template: DocumentTemplate) -> dict[str, Any]:
    """Return the reproducible stateless menu derived from one action catalog.

    Args:
        template: Document template whose sections define the menu regions.

    Returns:
        JSON-serializable template identity and ordered action/region choices.
    """
    return {
        "version": STATELESS_ACTION_MENU_VERSION,
        "semantic_action_catalog_version": SEMANTIC_ACTION_CATALOG_VERSION,
        "kind": template.kind,
        "sections": list(template.sections),
        "choices": [
            {
                "id": choice.menu_id,
                "semantic_action": choice.semantic_action.name,
                "operator": choice.edit_tool.value,
                "target_section": choice.target_section,
            }
            for choice in build_stateless_action_menu(template)
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
) -> list[ControllerChoice]:
    """Build the Controller's region/action menu.

    At level 1, each independently addressable region appears once. At level 2,
    every applicable region/action pair appears once so one verbalized-sampling
    call makes the complete semantic decision. The action's direct operator is
    derived from its :class:`SemanticActionSpec`; it is never a separate choice.

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
        menu = [action for target in targets for action in build_semantic_action_menu(template, target)]
    else:
        menu = [ControllerChoice(target, None) for target in targets]

    if not menu and level >= 2:
        raise ValueError(
            f"Document kind {template.kind!r} has no semantic actions; level 2 supports only cataloged kinds."
        )
    if not menu:
        menu = [ControllerChoice(EditTarget(component_name, None), None)]
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


def build_semantic_action_menu(template: DocumentTemplate, edit_target: EditTarget) -> list[ControllerChoice]:
    """Build every semantic action applicable to one sampled region.

    Args:
        template: Document template that defines the selected region.
        edit_target: Region already sampled by the first Controller stage.

    Returns:
        Complete conditional semantic-action menu in catalog order.
    """
    return [ControllerChoice(edit_target, spec) for spec in semantic_action_specs(template.kind, edit_target.section)]


class Controller(VerbalizedActionSelector[ControllerChoice]):
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
        menu: list[ControllerChoice],
        lm: LanguageModel,
        *,
        k: int = 5,
        tau: float | None = None,
        rng: random.Random | None = None,
        require_full_support: bool = False,
    ):
        """Configure verbalized sampling over the rich Controller choices."""
        super().__init__(
            actions=menu,
            lm=lm,
            k=k,
            tau=tau,
            rng=rng,
            require_full_support=require_full_support,
        )


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
