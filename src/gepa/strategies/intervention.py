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
from dataclasses import dataclass, field
from typing import Any

from gepa.strategies.action_space import (
    FULL_SUPPORT_EXPLORATION_EPSILON,
    SOFT_PROMPT_CHAR_BUDGET,
    VerbalizedActionSelector,
)
from gepa.strategies.document_template import DocumentTemplate, EditTarget
from gepa.strategies.edit_tools import EditTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticActionSpec:
    """Static semantic action selected by the Controller.

    Args:
        name: Stable action identifier.
        description: Menu text describing the intended revision.
        edit_tool: The one direct text operator coupled to this action.
        instruction: Instruction the Manifestor realizes against run evidence.
        fixed_text: Literal steering text that bypasses the Manifestor LM.

    Raises:
        TypeError: ``edit_tool`` is not one :class:`EditTool` value.
        ValueError: The action does not define exactly one manifestation source.
    """

    name: str
    description: str
    edit_tool: EditTool
    instruction: str | None = None
    fixed_text: str | None = None

    def __post_init__(self) -> None:
        """Validate the direct-tool and manifestation contracts.

        Raises:
            TypeError: ``edit_tool`` is not one :class:`EditTool` value.
            ValueError: The action does not define exactly one of
                ``instruction`` and ``fixed_text``.
        """
        if not isinstance(self.edit_tool, EditTool):
            raise TypeError(f"SemanticActionSpec {self.name!r} edit_tool must be one EditTool value")
        if (self.instruction is None) == (self.fixed_text is None):
            raise ValueError(f"SemanticActionSpec {self.name!r} must set exactly one of instruction or fixed_text")


@dataclass(frozen=True)
class ControllerChoice:
    """Controller decision over a region and optional semantic action.

    Args:
        edit_target: Independently selectable named document section.
        semantic_action: Semantic action, or ``None`` at level 1.
    """

    edit_target: EditTarget
    semantic_action: SemanticActionSpec | None
    edit_tool: EditTool | None = field(init=False, compare=False)
    menu_id: str = field(init=False, compare=False)
    menu_description: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Materialize the coupled operator and selector-facing menu fields.

        Level-1 choices store no operator and use a region-only identifier.
        Semantic choices copy their action's operator and include it in both
        stable menu text fields.
        """
        edit_tool = self.semantic_action.edit_tool if self.semantic_action is not None else None
        if self.semantic_action is not None:
            assert edit_tool is not None
            menu_id = f"{self.semantic_action.name}@{self.edit_target.section}/{edit_tool.value}"
            menu_description = (
                f"{self.semantic_action.description} "
                f"(region '{self.edit_target.section}', direct tool {edit_tool.value})"
            )
        else:
            menu_id = f"EDIT@{self.edit_target.section}"
            menu_description = f"Revise region '{self.edit_target.section}' using the available edit-tool basis."
        object.__setattr__(self, "edit_tool", edit_tool)
        object.__setattr__(self, "menu_id", menu_id)
        object.__setattr__(self, "menu_description", menu_description)


SEMANTIC_ACTION_CATALOG_VERSION = 2
CONTROLLER_POLICY_VERSION = 2
STATELESS_ACTION_MENU_VERSION = 1

CONTROLLER_POLICY_CONTRACT: dict[str, Any] = {
    "version": CONTROLLER_POLICY_VERSION,
    "factorization": "P(region, action)",
    "candidates": "all cataloged region/action pairs",
    "verbalized_candidates": "all",
    "sampling": "verbalized distribution mixed with uniform exploration",
    "exploration_epsilon": FULL_SUPPORT_EXPLORATION_EPSILON,
    "max_menu": None,
}

# Every action belongs to the general catalog, but an action is applicable only
# when the current section supports its precondition. For example, text with no
# supporting background cannot be pruned, and one indivisible span cannot be
# resequenced. The Controller scores the complete catalog against the current
# section and is the sole judge of applicability. Once sampled, its choice is
# final and the Manifestor realizes that action without reclassifying it.
#
# Classification order makes the ten outcomes mutually exclusive:
# 1. If operative meaning changes, classify its before/after sets as proper
#    subset, proper superset, overlapping without containment, or disjoint.
# 2. Otherwise compare supporting context with the same four set relations.
# 3. If both are unchanged, test ordering and then surface realization.
# A change on more than one axis is not one catalog action; decompose it across
# revisions.
#
# Worked witness, with each result independently edited from X:
# X: After lunch, the only activities children may choose are drawing and
#    building with blocks, because the art room is open and today is
#    indoor-play day.
# contextualize: After lunch, the only activities children may choose are
#    drawing and building with blocks, because the art room is open, today is
#    indoor-play day, and all the supplies are ready.
# prune_context: After lunch, the only activities children may choose are
#    drawing and building with blocks.
# revise_context: After lunch, the only activities children may choose are
#    drawing and building with blocks, because the art room is open and the
#    teacher has prepared the supplies. The first background fact remains; the
#    second is replaced.
# supplant_context: After lunch, the only activities children may choose are
#    drawing and building with blocks, because the playground is wet and the
#    outdoor equipment is being cleaned. The rule remains and all background is
#    replaced.
# resequence: After lunch, the only activities children may choose are drawing
#    and building with blocks, because today is indoor-play day and the art room
#    is open.
# reexpress: Once lunch is finished, children can select only drawing or
#    block-building, since today is set aside for indoor play and the art room
#    is available.
# restrict_meaning: After lunch, the only activity children may choose is
#    drawing, because the art room is open and today is indoor-play day.
# relax_meaning: After lunch, the only activities children may choose are
#    drawing, building with blocks, and solving a puzzle, because the art room
#    is open and today is indoor-play day.
# revise_meaning: After lunch, the only activities children may choose are
#    building with blocks and solving a puzzle, because the art room is open and
#    today is indoor-play day. One allowed activity remains and one is replaced.
# supplant_meaning: After lunch, the only activities children may choose are
#    singing and dancing, because the art room is open and today is indoor-play
#    day. The complete rule is replaced without overlap.
#
# Context actions preserve the operative rule. Meaning actions preserve the
# supporting context. Resequence preserves both content planes and changes only
# position. Reexpress preserves meaning, context, and order and changes wording.
SEMANTIC_ACTIONS: tuple[SemanticActionSpec, ...] = (
    SemanticActionSpec(
        "contextualize",
        "Add supporting context while preserving the complete operative meaning of the current text.",
        EditTool.INSERT_TEXT,
        instruction=(
            "Insert supporting background, explanation, rationale, or illustration at an exact anchor inside the "
            "current text. Every current supporting proposition must remain, and at least one genuinely new supporting "
            "proposition must be added, so the resulting context is a proper superset. Preserve every operative "
            "requirement, permission, prohibition, condition, exception, scope boundary, normative force, and "
            "admissible behavior, and preserve the existing text verbatim. Do not add an operative commitment. Use "
            "prune_context when the result removes context, revise_context when old and new context overlap without "
            "containment, and supplant_context when all context is replaced."
        ),
    ),
    SemanticActionSpec(
        "prune_context",
        "Remove supporting context while preserving the complete operative meaning of the current text.",
        EditTool.DELETE_TEXT,
        instruction=(
            "Delete one exact target substring containing only background, explanation, rationale, illustration, or "
            "other supporting context. At least one supporting proposition must be removed and none introduced, so the "
            "resulting context is a proper subset. Preserve every operative requirement, permission, prohibition, "
            "condition, exception, scope boundary, normative force, and admissible behavior, and preserve all remaining "
            "text verbatim. Use contextualize when the result adds context, revise_context when old and new context "
            "overlap without containment, and supplant_context when all context is replaced."
        ),
    ),
    SemanticActionSpec(
        "revise_context",
        (
            "Replace some supporting context while retaining some, preserving the complete operative meaning of the "
            "current text."
        ),
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring containing supporting context. At least one supporting proposition "
            "must remain applicable to both the current and resulting text, at least one must occur only in the current "
            "text, and at least one must occur only in the result, so the two context sets overlap without containment. "
            "Preserve all operative meaning and discourse order. "
            "Use contextualize or prune_context when one context contains the other, and supplant_context when no "
            "supporting proposition survives."
        ),
    ),
    SemanticActionSpec(
        "supplant_context",
        "Replace all supporting context with non-overlapping context while preserving the complete operative meaning.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring containing the complete supporting context. The result must retain no "
            "supporting proposition from that target and must supply new supporting context, while preserving every "
            "operative commitment and the discourse order. Use revise_context whenever any supporting proposition "
            "from the target survives."
        ),
    ),
    SemanticActionSpec(
        "resequence",
        "Change only the order of existing content while preserving operative meaning, context, and wording.",
        EditTool.MOVE_TEXT,
        instruction=(
            "Move one exact target substring to a distinct exact anchor inside the current text. Preserve the moved "
            "substring verbatim and retain exactly the same operative commitments and supporting context; only order, "
            "grouping, dependency presentation, precedence presentation, or salience may change. Both target and "
            "anchor must occur inside the current section. If the move changes admissible interpretations or behaviors, "
            "use restrict_meaning, relax_meaning, revise_meaning, or supplant_meaning instead."
        ),
    ),
    SemanticActionSpec(
        "reexpress",
        "Change only wording while preserving operative meaning, supporting context, and ordering.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring inside the current text. Preserve every operative commitment, every "
            "supporting proposition, every ordering relationship, scope boundary, normative force, and necessary "
            "detail. Change only wording, syntax, or another lossless surface realization. Do not add, remove, or move "
            "content. If admissible interpretations or behaviors change, use restrict_meaning, relax_meaning, "
            "revise_meaning, or supplant_meaning."
        ),
    ),
    SemanticActionSpec(
        "restrict_meaning",
        (
            "Change operative meaning so the result admits a proper subset of the current text's interpretations or "
            "behaviors."
        ),
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring inside the current text. Every interpretation or behavior admitted by "
            "the result must have been admitted by the current text, and at least one formerly admitted interpretation "
            "or behavior must be excluded. Preserve supporting context and discourse ordering exactly. Use "
            "relax_meaning when the result is a proper superset, revise_meaning when the meanings overlap without "
            "containment, and supplant_meaning when they are disjoint."
        ),
    ),
    SemanticActionSpec(
        "relax_meaning",
        (
            "Change operative meaning so the result admits a proper superset of the current text's interpretations or "
            "behaviors."
        ),
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring inside the current text. Every interpretation or behavior admitted by "
            "the current text must remain admitted by the result, and the result must admit at least one additional "
            "interpretation or behavior. Preserve supporting context and discourse ordering exactly. Use "
            "restrict_meaning when the result is a proper subset, revise_meaning when the meanings overlap without "
            "containment, and supplant_meaning when they are disjoint."
        ),
    ),
    SemanticActionSpec(
        "revise_meaning",
        (
            "Change operative meaning so the current and resulting meanings overlap while neither contains the other."
        ),
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring inside the current text. The current and resulting text must admit at "
            "least one common interpretation or behavior, at least one admitted only by the current text, and at least "
            "one admitted only by the result. Preserve supporting context and discourse ordering exactly. Use "
            "restrict_meaning or relax_meaning when one meaning contains the other, and supplant_meaning when no "
            "compatible interpretation or behavior remains."
        ),
    ),
    SemanticActionSpec(
        "supplant_meaning",
        "Replace the complete operative meaning with a disjoint meaning while preserving supporting context.",
        EditTool.REPLACE_TEXT,
        instruction=(
            "Replace one exact target substring containing the complete operative rule or task commitment. The current "
            "and resulting operative meanings must be disjoint: no relevant interpretation or behavior may satisfy "
            "both. Preserve supporting context and discourse ordering exactly, and keep the replacement coherent with "
            "that preserved context. Use revise_meaning whenever any compatible interpretation or behavior survives."
        ),
    ),
)

if len({spec.name for spec in SEMANTIC_ACTIONS}) != len(SEMANTIC_ACTIONS):
    raise ValueError("Built-in semantic action names must be unique")

_SEMANTIC_ACTION_KINDS = frozenset({"prompt", "skill"})

SEMANTIC_ACTION_CATALOGS: dict[str, dict[str, Any]] = {
    kind: {
        "version": SEMANTIC_ACTION_CATALOG_VERSION,
        "kind": kind,
        "actions": [
            {
                "name": spec.name,
                "operator": spec.edit_tool.value,
                "description": spec.description,
                "instruction": spec.instruction,
                "fixed_text": spec.fixed_text,
            }
            for spec in SEMANTIC_ACTIONS
        ],
    }
    for kind in _SEMANTIC_ACTION_KINDS
}


@dataclass(frozen=True)
class StatelessActionConstraint:
    """Bind one canonical semantic action to a stateless edit region.

    Action-conditioned stateless reflection emits one revised section body
    instead of calling an edit tool. This value retains the originating template
    so the integration layer can splice that body into the parent document.

    Args:
        semantic_action: Canonical semantic operation and coupled edit tool.
        target_section: Named section to revise.
        document_template: Template used to isolate and splice that section.
    """

    semantic_action: SemanticActionSpec
    target_section: str
    document_template: DocumentTemplate
    edit_tool: EditTool = field(init=False, compare=False)
    menu_id: str = field(init=False, compare=False)
    menu_description: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Materialize the action's operator and selector-facing menu fields.

        Stateless constraints always have a semantic action, so all three
        derived fields encode the same fixed action/section/operator binding.
        """
        edit_tool = self.semantic_action.edit_tool
        menu_id = f"{self.semantic_action.name}@{self.target_section}/{edit_tool.value}"
        menu_description = (
            f"{self.semantic_action.description} "
            f"(target section '{self.target_section}', direct tool {edit_tool.value})"
        )
        object.__setattr__(self, "edit_tool", edit_tool)
        object.__setattr__(self, "menu_id", menu_id)
        object.__setattr__(self, "menu_description", menu_description)


def format_stateless_action_constraint(action: StatelessActionConstraint) -> str:
    """Render one canonical action/region choice as a reflection constraint.

    Args:
        action: Semantic action bound to its selected target region.

    Returns:
        Constraint suffix suitable for a stateless selected-section proposer.
    """
    section_scope = (
        f"The instruction document above is only the body of the selected '{action.target_section}' section. "
        "Return the complete revised body for that section without a '## <Section>' header. "
        "Do not reference or modify any other section."
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
        f"Length budget: the revised section body must stay under {SOFT_PROMPT_CHAR_BUDGET} characters. "
        "If this edit would exceed the budget, replace or remove existing content instead of adding.\n\n"
        "Make no other changes."
    )


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

    Below level 2, each independently addressable region appears once. At level
    2 or above, every cataloged region/action pair appears once so one
    verbalized-sampling call makes the complete semantic decision. The action's
    direct operator is derived from its :class:`SemanticActionSpec`; it is never
    a separate choice.

    Args:
        template: Document template whose sections become targets.
        component_name: Candidate component name.
        edit_tools: Configured execution basis. It must be non-empty even though
            semantic actions remain visible when their direct broad tool is absent.
        level: Values below 2 select regions only; values of 2 or above select
            joint region/action pairs.
        rng: Seeded RNG for optional deterministic menu subsampling.
        max_menu: Optional level-1 region bound. At level 2 or above it may be
            set only high enough to retain every cataloged region/action pair.

    Returns:
        Non-empty Controller menu.

    Raises:
        ValueError: ``edit_tools`` is empty, the template has no named targets,
            ``max_menu`` is below one, a level-2-or-higher document kind has no
            semantic actions, or its bound would remove a cataloged pair.
    """
    if not edit_tools:
        raise ValueError("Controller requires at least one edit tool.")
    if max_menu is not None and max_menu < 1:
        raise ValueError("max_menu must be at least 1.")

    targets = [EditTarget(component_name, section) for section in template.sections]
    if not targets:
        raise ValueError(f"Document template {template.kind!r} has no named sections to edit.")
    if level >= 2:
        specs = SEMANTIC_ACTIONS if template.kind in _SEMANTIC_ACTION_KINDS else ()
        menu = [ControllerChoice(target, spec) for target in targets for spec in specs]
    else:
        menu = [ControllerChoice(target, None) for target in targets]

    if not menu and level >= 2:
        raise ValueError(
            f"Document kind {template.kind!r} has no semantic actions; level 2 supports only cataloged kinds."
        )
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


class Controller(VerbalizedActionSelector[ControllerChoice]):
    """Select one region/semantic-action option by verbalized sampling.

    Args:
        actions: Rich Controller actions.
        lm: Model that verbalizes the option distribution.
        k: Number of options assigned probabilities.
        tau: Tail-sampling threshold, or ``None`` for ``1 / k``.
        rng: Seeded sampling RNG.
        require_full_support: Require every option exactly once and mix the
            verbalized distribution with uniform exploration.
    """

def summarize_feedback(reflective_entries: Any, max_chars: int = SOFT_PROMPT_CHAR_BUDGET) -> str:
    """Join feedback and truncate its raw prefix before adding an ellipsis.

    Args:
        reflective_entries: Rows carrying ``Feedback`` or ``execution_feedback``.
        max_chars: Characters retained from non-empty joined feedback before
            an optional ``...`` suffix. The no-feedback marker is returned
            verbatim regardless of this limit.

    Returns:
        Joined feedback, a truncated prefix plus an ellipsis, or the
        no-feedback marker.
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
