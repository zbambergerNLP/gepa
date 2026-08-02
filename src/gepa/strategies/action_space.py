# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Action space for action-conditioned reflection.

Each reflection job can be conditioned on a single ``PromptEditAction``,
constraining the reflection LM to make a specific type of edit (e.g., add an
illustration, adjust specificity).  This isolates the effect of each edit
operation and improves proposal diversity across siblings.

The primary selector is ``VerbalizedActionSelector``, which uses verbalized
sampling to let the reflection LM itself choose actions with explicit
probabilities, then samples from the distribution tails for diversity.
``RandomActionSelector`` is kept as a simple baseline.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Protocol

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.utils.text import strip_think_tags


@dataclass(frozen=True)
class PromptEditAction:
    """One type of prompt mutation the reflection LM is constrained to perform.

    Attributes:
        name: Short identifier (e.g. ``"add_illustration"``).
        description: Human-readable description shown to the reflection LM.
            Also serves as query text for future learned selectors (e.g. a
            POSIT-style ColBERT/CrossEncoder reranker).
        instruction_suffix: Directive appended to the reflection prompt to
            constrain the LM to this edit type.
        target_section: Optional markdown section name (e.g. ``"Rules"``) this
            action is scoped to. When set, the reflection LM is instructed to
            edit only that section of a structured prompt.
    """

    name: str
    description: str
    instruction_suffix: str
    target_section: str | None = None


logger = logging.getLogger(__name__)


class ActionSelector(Protocol):
    """Picks which action(s) to apply to a batch of reflection jobs.

    The primary implementation is ``VerbalizedActionSelector``, which asks
    the reflection LM to produce a probability distribution over actions
    conditioned on the current prompt and feedback, then samples from
    the distribution tails for diversity.  ``RandomActionSelector`` serves
    as a simple uniform baseline.

    Implementations may optionally expose a ``set_context(candidate,
    feedback_summary)`` method called by ``StatelessReflectionLM`` before
    ``select()``, providing prompt state for context-aware selection.
    """

    def select(self, n: int, rng: random.Random | None = None) -> list[PromptEditAction]: ...


class RandomActionSelector:
    """Pick actions uniformly at random from the action space."""

    def __init__(self, actions: list[PromptEditAction], rng: random.Random | None = None):
        if not actions:
            raise ValueError("RandomActionSelector requires a non-empty actions list")
        self.actions = actions
        self.rng = rng if rng is not None else random.Random(0)

    def select(self, n: int, rng: random.Random | None = None) -> list[PromptEditAction]:
        rng = rng if rng is not None else self.rng
        return [rng.choice(self.actions) for _ in range(n)]


VERBALIZED_ACTION_PROMPT = """\
You are selecting which edit action(s) to apply to improve a prompt.

## Current prompt
```
{current_prompt}
```

## Recent feedback summary
{feedback_summary}

## Available actions
{action_menu}

Generate {k} candidate actions. For each, assign a probability reflecting how \
likely that action is to improve the prompt given the feedback above. \
Probabilities must sum to 1.0.

Important: try to explore less obvious actions. Assign higher probability to \
actions that specifically address the failure patterns in the feedback, even if \
they seem unconventional.

Format your response as:
<response>
<candidate>
<action>action_name_here</action>
<reasoning>why this action fits the current failure patterns</reasoning>
<probability>0.XX</probability>
</candidate>
...repeat for {k} candidates...
</response>
"""


@dataclass
class ActionDistribution:
    """A parsed distribution over actions from the verbalized selector."""

    entries: list[tuple[PromptEditAction, float, str]]  # (action, probability, reasoning)
    is_fallback: bool = False  # True when parsing failed and a uniform fallback was used

    @property
    def actions(self) -> list[PromptEditAction]:
        return [a for a, _, _ in self.entries]

    @property
    def probabilities(self) -> list[float]:
        return [p for _, p, _ in self.entries]


def _sample_from_tails(
    distribution: ActionDistribution,
    n: int,
    tau: float,
    rng: random.Random,
) -> list[PromptEditAction]:
    """Sample n actions from the tail of the distribution (probability < tau).

    If no entries fall below tau, sample from the full distribution.
    """
    entries = distribution.entries
    tail = [(a, p, r) for a, p, r in entries if p < tau]

    # Fall back to full distribution if no tail entries exist.
    if not tail:
        tail = entries

    actions = [a for a, _, _ in tail]
    weights = [p for _, p, _ in tail]

    # Renormalize weights.
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(actions)] * len(actions)
    else:
        weights = [w / total for w in weights]

    return rng.choices(actions, weights=weights, k=n)


class VerbalizedActionSelector:
    """Use the reflection LM to generate a probability distribution over actions, then sample.

    Instead of picking actions uniformly at random, this selector asks the LM
    which action(s) are most likely to help
    given the current prompt state and feedback. It then samples from the tails
    of the distribution (p < tau) to encourage diversity.

    Call ``set_context()`` before ``select()`` to provide the current prompt and
    feedback. If ``set_context()`` is not called, falls back to uniform random.
    """

    def __init__(
        self,
        actions: list[PromptEditAction],
        lm: LanguageModel,
        k: int = 5,
        tau: float = 0.10,
        rng: random.Random | None = None,
    ):
        if not actions:
            raise ValueError("VerbalizedActionSelector requires a non-empty actions list")
        self.actions = actions
        self.lm = lm
        self.k = k
        self.tau = tau
        self.rng = rng if rng is not None else random.Random(0)
        self._context: dict[str, str] | None = None
        self._action_by_name: dict[str, PromptEditAction] = {a.name: a for a in actions}
        # One record per select() call with context: the verbalized distribution
        # and what was sampled from it. Observational only (for analysis dumps).
        self.history: list[dict] = []

    def set_context(self, candidate: str, feedback_summary: str) -> None:
        """Provide current prompt state for the next select() call."""
        self._context = {"candidate": candidate, "feedback_summary": feedback_summary}

    def select(self, n: int, rng: random.Random | None = None) -> list[PromptEditAction]:
        rng = rng if rng is not None else self.rng
        if self._context is None:
            logger.warning("VerbalizedActionSelector.select() called without set_context(); falling back to uniform.")
            return [rng.choice(self.actions) for _ in range(n)]

        distribution = self._generate_distribution(rng)
        result = _sample_from_tails(distribution, n, self.tau, rng)
        self.history.append(
            {
                "probs": {a.name: p for a, p, _ in distribution.entries},
                "sampled": [a.name for a in result],
                "fallback": distribution.is_fallback,
            }
        )
        # Clear context after use so stale context isn't reused.
        self._context = None
        return result

    def _generate_distribution(self, rng: random.Random) -> ActionDistribution:
        """Call the LM to produce a verbalized probability distribution over actions."""
        assert self._context is not None
        action_menu = "\n".join(f"- **{a.name}**: {a.description}" for a in self.actions)
        prompt = VERBALIZED_ACTION_PROMPT.format(
            current_prompt=self._context["candidate"],
            feedback_summary=self._context["feedback_summary"],
            action_menu=action_menu,
            k=self.k,
        )
        raw_output = self.lm(prompt)
        return self._parse_distribution(raw_output, rng)

    def _parse_distribution(self, raw_output: str, rng: random.Random) -> ActionDistribution:
        """Parse XML-formatted action distribution from LM output."""
        # Strip think tags if present (reasoning models).
        raw_output = strip_think_tags(raw_output)

        entries: list[tuple[PromptEditAction, float, str]] = []
        for candidate_match in re.finditer(r"<candidate>(.*?)</candidate>", raw_output, re.DOTALL):
            block = candidate_match.group(1)
            action_m = re.search(r"<action>(.*?)</action>", block, re.DOTALL)
            prob_m = re.search(r"<probability>(.*?)</probability>", block, re.DOTALL)
            reasoning_m = re.search(r"<reasoning>(.*?)</reasoning>", block, re.DOTALL)

            if not action_m or not prob_m:
                continue

            action_name = action_m.group(1).strip()
            reasoning = reasoning_m.group(1).strip() if reasoning_m else ""

            try:
                probability = float(prob_m.group(1).strip())
            except ValueError:
                continue

            action = self._action_by_name.get(action_name)
            if action is None:
                # Fuzzy match: try case-insensitive lookup.
                for name, act in self._action_by_name.items():
                    if name.lower() == action_name.lower():
                        action = act
                        break
            if action is not None:
                entries.append((action, probability, reasoning))

        is_fallback = False
        if not entries:
            logger.warning("Failed to parse verbalized action distribution; falling back to uniform.")
            n_actions = len(self.actions)
            entries = [(a, 1.0 / n_actions, "") for a in self.actions]
            is_fallback = True

        # Renormalize probabilities to sum to 1.
        total = sum(p for _, p, _ in entries)
        if total > 0:
            entries = [(a, p / total, r) for a, p, r in entries]

        return ActionDistribution(entries=entries, is_fallback=is_fallback)


def format_action_suffix(action: PromptEditAction) -> str:
    """Build the constraint text appended to the reflection prompt."""
    section_scope = ""
    if action.target_section is not None:
        section_scope = (
            f"\nApply this edit ONLY within the '## {action.target_section}' section of the prompt. "
            "Reproduce every other section verbatim, including their headers.\n"
        )
    return (
        "\n\n--- ACTION CONSTRAINT ---\n"
        f"You MUST make exactly one type of edit: {action.name}\n"
        f"Description: {action.description}\n"
        f"{section_scope}\n"
        f"{action.instruction_suffix}\n\n"
        "Do not make any other type of change. Focus exclusively on this edit action."
    )


# Canonical section names for structured (markdown-skeleton) prompts.
STRUCTURED_SECTIONS: list[str] = ["Role", "Task", "Rules", "Output Format", "Examples"]

_SECTION_OPERATIONS: list[tuple[str, str, str]] = [
    (
        "rewrite",
        "Replace the content of the '{section}' section.",
        "Rewrite the '## {section}' section from scratch so it directly addresses the failure "
        "patterns in the feedback. Replace its current content entirely; keep the section header.",
    ),
    (
        "append",
        "Add one targeted item (rule, detail, or example) to the '{section}' section.",
        "Add exactly one new item to the '## {section}' section that directly addresses a failure "
        "pattern observed in the feedback. Keep all existing content of the section unchanged; "
        "the new item should be precise and actionable, not generic advice.",
    ),
    (
        "condense",
        "Remove redundant, conflicting, or harmful items from the '{section}' section.",
        "Examine the '## {section}' section for items that are redundant, mutually conflicting, "
        "overly narrow, or likely causing the failures in the feedback. Remove or merge them. "
        "Do not add new content.",
    ),
]


def _slugify_section(section: str) -> str:
    return section.lower().replace(" ", "_")


def build_structured_actions(sections: list[str] | None = None) -> list[PromptEditAction]:
    """Build a section-scoped action space for structured (markdown) prompts.

    For each section, three operations: rewrite, append, condense. Plus one
    global restructure action. With the default five sections this yields a
    16-action menu.
    """
    sections = sections if sections is not None else STRUCTURED_SECTIONS
    actions: list[PromptEditAction] = []
    for section in sections:
        for op_name, op_desc, op_suffix in _SECTION_OPERATIONS:
            actions.append(
                PromptEditAction(
                    name=f"{op_name}_{_slugify_section(section)}",
                    description=op_desc.format(section=section),
                    instruction_suffix=op_suffix.format(section=section),
                    target_section=section,
                )
            )
    actions.append(
        PromptEditAction(
            name="restructure",
            description="Reorder or rebalance sections without adding or removing content.",
            instruction_suffix=(
                "Reorganize the prompt's sections: reorder them so the most critical information "
                "appears first, move misplaced items to the section where they belong, or adjust "
                "emphasis. Keep all existing content; do not add or remove information."
            ),
        )
    )
    return actions


DEFAULT_ACTIONS: list[PromptEditAction] = [
    PromptEditAction(
        name="add_illustration",
        description="Add an inline worked example or case illustration within the instruction text.",
        instruction_suffix=(
            "Add a concrete, illustrative example within the instruction that demonstrates "
            "correct behavior on a specific pattern observed in the feedback. Use a 'For example, ...' "
            "or 'For instance, ...' construction. The example should help the assistant generalize "
            "from the failure patterns in the feedback to similar future cases."
        ),
    ),
    PromptEditAction(
        name="adjust_specificity",
        description="Make the task description more specific or more general based on failure patterns.",
        instruction_suffix=(
            "Examine the feedback to determine whether failures stem from the instruction being "
            "too vague (causing the assistant to guess) or too narrow (causing it to miss valid "
            "approaches). Adjust the level of specificity accordingly. If too vague, add precise "
            "details. If too narrow, broaden the language to allow correct alternatives."
        ),
    ),
    PromptEditAction(
        name="edit_guidelines",
        description="Modify role definition, persona, or behavioral guidelines.",
        instruction_suffix=(
            "Revise the role description, persona framing, or behavioral guidelines in the "
            "instruction. This could mean adjusting the assistant's stated expertise, changing "
            "how it should approach problems, or modifying its priorities when trade-offs arise. "
            "Focus on guidelines that would prevent the failures shown in the feedback."
        ),
    ),
    PromptEditAction(
        name="edit_field_description",
        description="Change how input/output fields or expected formats are described.",
        instruction_suffix=(
            "Modify how the instruction describes the expected input format, output format, "
            "or field semantics. This includes clarifying what each field contains, how the "
            "output should be structured, or what format constraints apply. Focus on format "
            "issues visible in the feedback."
        ),
    ),
    PromptEditAction(
        name="add_constraint",
        description="Add a targeted constraint, rule, or edge-case handling instruction.",
        instruction_suffix=(
            "Add a specific constraint, rule, or edge-case instruction that directly addresses "
            "a failure pattern observed in the feedback. The constraint should be precise and "
            "actionable (e.g., 'Always verify X before Y', 'When Z occurs, handle it by ...'). "
            "Do not add generic advice."
        ),
    ),
    PromptEditAction(
        name="restructure",
        description="Reorganize prompt structure, ordering, or emphasis without changing content.",
        instruction_suffix=(
            "Reorganize the instruction's structure, section ordering, or emphasis without "
            "adding or removing content. This could mean reordering sections so the most "
            "critical information appears first, grouping related instructions together, "
            "adding section headers, or adjusting emphasis (e.g., bolding key rules). "
            "The goal is to make existing information more salient and easier to follow."
        ),
    ),
]
