# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Semantic actions and verbalized Controller for three-role reflection.

The Controller selects a document region and, at reflection level 2, one
semantic revision action. An action expresses editorial intent and owns exactly
one direct text operator. ReAct V2 uses that operator directly when the broad
tool set exposes it; the minimal insert/delete basis faithfully lowers replace
and move into their two atomic calls.
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
        inject_as: Default injection site. The three-role strategy overrides
            this with provider routing for built-in execution.

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
    inject_as: InjectionSite = "assistant_reasoning"

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
    inject_as: InjectionSite = "assistant_reasoning"


@dataclass(frozen=True)
class ControllerAction:
    """Controller decision over a region and optional semantic action.

    Args:
        edit_target: Independently selectable document section or whole document.
        intervention_spec: Semantic action, or ``None`` at level 1.
        semantic_options: Actions disclosed while the first Controller stage
            chooses a region. Empty after a semantic action is selected.
    """

    edit_target: EditTarget
    intervention_spec: InterventionSpec | None
    semantic_options: tuple[InterventionSpec, ...] = ()

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
        if self.semantic_options:
            options = ", ".join(f"{spec.name}/{spec.edit_tool.value}" for spec in self.semantic_options)
            return f"Revise region '{region}'. Compatible semantic actions: {options}."
        return f"Revise region '{region}' using the available edit-tool basis."


SEMANTIC_ACTION_CATALOG_VERSION = 1
CONTROLLER_POLICY_VERSION = 1

SEMANTIC_ACTIONS: tuple[InterventionSpec, ...] = (
    InterventionSpec(
        "rephrase",
        "Improve wording in the same representation while preserving meaning, requirements, scope, and information "
        "density; use reformat when the representation itself is the defect.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one exact passage whose expression is ineffective, establish the meaning, requirements, scope, "
            "and level of detail that must remain unchanged, and direct the editor to replace only that wording with "
            "a clearer equivalent."
        ),
    ),
    InterventionSpec(
        "summarize",
        "Shorten a passage while preserving every unique operative requirement and necessary fact.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one verbose passage, enumerate the unique requirements, exceptions, dependencies, and facts "
            "that must survive, and direct the editor to replace it with a materially shorter equivalent."
        ),
    ),
    InterventionSpec(
        "reformat",
        "Change one passage's representation while preserving every semantic proposition, qualifier, requirement, "
        "and scope boundary; compression is not an objective, so use summarize when detail should be removed.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one exact passage whose representation makes correct content hard to follow or use, name the "
            "target representation justified by the failures, and direct the editor to replace that passage with a "
            "semantically equivalent checklist, ordered procedure, table, schema, field list, or other clearer form. "
            "Preserve every semantic proposition and qualifier; do not add, compress, remove, or relocate content."
        ),
    ),
    InterventionSpec(
        "correct",
        "Correct false, contradictory, or behaviorally wrong content whose role is still needed; use specialize or "
        "generalize for applicability and strengthen_requirement or relax_requirement for normative force.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one passage that the failures show is factually, logically, or procedurally wrong, explain the "
            "evidence-grounded correction while preserving the passage's functional role, and direct the editor to "
            "replace it. Use the dedicated scope or normative-force action when that is the primary defect."
        ),
    ),
    InterventionSpec(
        "specialize",
        "Narrow a passage whose applicability is overbroad; use rephrase for ambiguous wording, expand for missing "
        "detail, or add_constraint when the original rule remains valid.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one passage whose own applicability is too broad, name the cases it should and should not "
            "cover, and direct the editor to replace it with a narrower operational statement. Do not use this for "
            "ambiguous wording, missing detail, or an otherwise valid rule that merely needs an independent guardrail."
        ),
    ),
    InterventionSpec(
        "generalize",
        "Broaden an overfit or unjustifiably narrow passage while retaining necessary safeguards.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one passage that rejects valid alternatives or overfits the observed cases, name the broader "
            "invariant supported by the failures and the safeguards that must remain, and direct the editor to "
            "replace the passage with that broader statement."
        ),
    ),
    InterventionSpec(
        "strengthen_requirement",
        "Increase the normative force of a requirement whose content and applicability are correct but too "
        "permissive or advisory; preserve what it governs and where it applies.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one requirement whose content and scope are correct but whose permissive or advisory force "
            "allows a diagnosed failure. Name the required modality or prohibition and direct the editor to replace "
            "only that requirement with a stronger equivalent while preserving its subject and applicability."
        ),
    ),
    InterventionSpec(
        "relax_requirement",
        "Reduce the normative force of an otherwise useful requirement that is too absolute or restrictive; "
        "preserve what it governs and where it applies.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Identify one otherwise useful requirement whose absolute force causes a diagnosed overconstraint. Name "
            "the justified conditional, preferred, or permissive modality and direct the editor to replace only that "
            "requirement while preserving its subject and applicability. Delete it instead if no intent should survive."
        ),
    ),
    InterventionSpec(
        "expand",
        "Insert one missing non-guardrail semantic unit while leaving existing text unchanged; use add_constraint "
        "for a boundary or guardrail.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Identify one diagnosed failure and one independently testable proposition, single procedure step, "
            "self-contained example, definition, or fact that resolves it. Name the exact anchor and direct the editor "
            "to insert only that one unit without rewriting existing text. Use add_constraint for a guardrail."
        ),
    ),
    InterventionSpec(
        "add_constraint",
        "Preserve an otherwise valid rule and insert one missing boundary, prohibition, validation condition, or "
        "guardrail; use specialize when the rule itself is overbroad.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Identify an otherwise valid rule that permits a diagnosed bad case because an independent boundary, "
            "prohibition, applicability condition, stop condition, or validation guardrail is absent. Name the exact "
            "anchor and direct the editor to insert one precise constraint while leaving the governing rule unchanged."
        ),
    ),
    InterventionSpec(
        "remove_redundancy",
        "Delete one complete span whose useful meaning already survives elsewhere.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Identify one exact duplicated or semantically equivalent span, point to the surviving canonical content "
            "that preserves all of its meaning and qualifiers, and direct the editor to delete only the redundant "
            "occurrence."
        ),
    ),
    InterventionSpec(
        "remove_harmful_content",
        "Delete one wholly harmful, obsolete, irrelevant, vacuous, or non-operative span when none of its intent "
        "should survive.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Identify one exact span that is obsolete, irrelevant, vacuous, non-operative, conflicting, "
            "counterproductive, or overconstraining, verify that it contains no necessary semantic residue and needs "
            "no corrected successor, and direct the editor to delete only that span."
        ),
    ),
    InterventionSpec(
        "relocate",
        "Move one exact unchanged span when placement is the only defect.",
        EditTool.MOVE_TEXT,
        allow_whole_document=True,
        instruction=(
            "Identify one self-contained span whose wording is already correct but whose position harms dependency "
            "order, procedural sequence, grouping, precedence, or salience. Name a distinct exact destination anchor "
            "and direct the editor to move the span there byte-for-byte without rewriting it."
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
        "factorization": "P(region) * P(action | region)",
        "region_candidates": "all regions with at least one applicable action",
        "action_candidates": "all actions applicable to the sampled region",
        "verbalized_candidates_per_stage": "all",
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
    """Build the Controller's region-selection menu.

    Each independently addressable region appears once. At level 2, regions
    without an applicable semantic action are omitted; a second conditional
    menu is built with :func:`build_semantic_action_menu` after a region is
    sampled. No region/action Cartesian menu is materialized.

    Args:
        template: Document template whose sections become targets.
        component_name: Candidate component name.
        edit_tools: Configured execution basis. It must be non-empty even though
            semantic actions remain visible when their direct broad tool is absent.
        level: Reflection level.
        rng: Seeded RNG for optional deterministic menu subsampling.
        max_menu: Optional level-1 region bound. At level 2 it may be set only
            high enough to retain every applicable region.

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
        options_by_target = [(target, tuple(intervention_specs(template.kind, target.section))) for target in targets]
        menu = [ControllerAction(target, None, options) for target, options in options_by_target if options]
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
                f"max_menu={max_menu} would remove semantic Controller regions; level 2 requires all {len(menu)}."
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
