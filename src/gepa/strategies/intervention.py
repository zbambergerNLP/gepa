# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Semantic actions and verbalized Controller for three-role reflection.

The Controller selects a document region and, at reflection level 2, one of
three semantic revision actions: rephrase, summarize, or expand. Each semantic
action is coupled to exactly one direct edit tool. The ReAct V2 proposer uses
that direct call when the broad tool set exposes it; the minimal insert/delete
basis may compose several calls to realize the same intent.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Literal, get_args

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.action_space import (
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
        compatible_tools: Tools retained for API compatibility. The first is
            the direct tool; built-in actions define exactly one.
        applicable_sections: Optional section-name restriction.
        instruction: Instruction the Manifestor realizes against run evidence.
        fixed_text: Literal steering text that bypasses the Manifestor LM.
        inject_as: Default injection site. The three-role strategy overrides
            this with provider routing for built-in execution.

    Raises:
        ValueError: The action does not define a direct tool, does not
            define exactly one manifestation source, or names an invalid site.
    """

    name: str
    description: str
    compatible_tools: tuple[EditTool, ...]
    applicable_sections: tuple[str, ...] | None = None
    instruction: str | None = None
    fixed_text: str | None = None
    inject_as: InjectionSite = "assistant_reasoning"

    def __post_init__(self) -> None:
        """Validate the direct-tool and manifestation contracts."""
        if not self.compatible_tools:
            raise ValueError(f"InterventionSpec {self.name!r} must be coupled to at least one direct edit tool")
        if (self.instruction is None) == (self.fixed_text is None):
            raise ValueError(f"InterventionSpec {self.name!r} must set exactly one of instruction or fixed_text")
        if self.inject_as not in INJECTION_SITES:
            raise ValueError(f"InterventionSpec {self.name!r}: inject_as must be one of {INJECTION_SITES}")

    @property
    def edit_tool(self) -> EditTool:
        """Return the single direct tool coupled to this action.

        Returns:
            Direct edit tool.
        """
        return self.compatible_tools[0]


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
        edit_tool: Direct tool coupled to ``intervention_spec``. ``None`` at
            level 1, where ReAct V2 operates over the configured tool basis.
        intervention_spec: Semantic action, or ``None`` at level 1.
    """

    edit_target: EditTarget
    edit_tool: EditTool | None
    intervention_spec: InterventionSpec | None

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


_SEMANTIC_ACTIONS: tuple[InterventionSpec, ...] = (
    InterventionSpec(
        "rephrase",
        "Rephrase ambiguous or ineffective text without changing its intended scope.",
        (EditTool.REPLACE_TEXT,),
        instruction=(
            "Identify the exact wording that the failures show is ambiguous or ineffective, explain the intended "
            "meaning it must preserve, and direct the editor to rephrase that wording precisely."
        ),
    ),
    InterventionSpec(
        "summarize",
        "Summarize redundant text while preserving every requirement needed for success.",
        (EditTool.REPLACE_TEXT,),
        instruction=(
            "Identify the redundant passage and the requirements within it that must survive, then direct the "
            "editor to replace it with a shorter equivalent."
        ),
    ),
    InterventionSpec(
        "expand",
        "Expand the region with missing guidance grounded in the observed failures.",
        (EditTool.INSERT_TEXT,),
        instruction=(
            "Identify the missing guidance demonstrated by the failures, name where it belongs in the selected "
            "region, and direct the editor to add only that guidance."
        ),
    ),
)

_INTERVENTION_CATALOG: dict[str, tuple[InterventionSpec, ...]] = {
    "prompt": _SEMANTIC_ACTIONS,
    "skill": _SEMANTIC_ACTIONS,
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
    return [
        spec
        for spec in specs
        if spec.applicable_sections is None or (section is not None and section in spec.applicable_sections)
    ]


def build_controller_menu(
    template: DocumentTemplate,
    component_name: str,
    edit_tools: list[EditTool],
    level: int,
    *,
    rng: random.Random,
    max_menu: int | None = None,
) -> list[ControllerAction]:
    """Build a Controller menu without a semantic/tool cross-product.

    At level 1, each independently addressable region appears once and ReAct V2
    receives the configured edit-tool basis. At level 2, each region is paired
    once with each semantic action; the action already owns its direct tool.
    Consequently the Controller never makes a second, independent tool choice.

    Args:
        template: Document template whose sections become targets.
        component_name: Candidate component name.
        edit_tools: Configured execution basis. It must be non-empty even though
            semantic actions remain visible when their direct broad tool is absent.
        level: Reflection level.
        rng: Seeded RNG for optional deterministic menu subsampling.
        max_menu: Optional maximum options shown to verbalized sampling. The
            default keeps the complete menu.

    Returns:
        Non-empty Controller menu.

    Raises:
        ValueError: ``edit_tools`` is empty or ``max_menu`` is below one.
    """
    if not edit_tools:
        raise ValueError("Controller requires at least one edit tool.")
    if max_menu is not None and max_menu < 1:
        raise ValueError("max_menu must be at least 1.")

    targets = template.edit_targets(component_name)
    if level <= 1:
        menu = [ControllerAction(target, None, None) for target in targets]
    else:
        menu = [
            ControllerAction(target, spec.edit_tool, spec)
            for target in targets
            for spec in intervention_specs(template.kind, target.section)
        ]

    if not menu:
        menu = [ControllerAction(EditTarget(component_name, None), None, None)]
    if max_menu is not None and len(menu) > max_menu:
        logger.info(
            "Controller menu for '%s' has %d options; sampling %d and dropping %d.",
            component_name,
            len(menu),
            max_menu,
            len(menu) - max_menu,
        )
        menu = rng.sample(menu, max_menu)
    return menu


class Controller(VerbalizedActionSelector):
    """Select one region/semantic-action option by verbalized sampling.

    Args:
        menu: Rich Controller actions.
        lm: Model that verbalizes the option distribution.
        k: Number of options assigned probabilities.
        tau: Tail-sampling threshold, or ``None`` for ``1 / k``.
        rng: Seeded sampling RNG.
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
        """Wrap rich actions as selector-compatible menu entries."""
        stand_ins = [
            PromptEditAction(name=action.menu_id, description=action.menu_description, instruction_suffix="")
            for action in menu
        ]
        super().__init__(actions=stand_ins, lm=lm, k=k, tau=tau, rng=rng)
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
