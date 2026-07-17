# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Action space for action-conditioned reflection.

Each reflection job can be conditioned on a single ``PromptEditAction``,
constraining the reflection LM to make a specific type of edit (e.g., add an
illustration, adjust specificity).  This isolates the effect of each edit
operation and improves proposal diversity across siblings.

See Revision 1: action-conditioned reflection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


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
    """

    name: str
    description: str
    instruction_suffix: str


class ActionSelector(Protocol):
    """Picks which action(s) to apply to a batch of reflection jobs.

    The protocol is intentionally minimal so that future implementations can
    use learned selection.  For example, a POSIT-style ``ScoredActionSelector``
    could use a ColBERT or CrossEncoder reranker to score each action
    conditioned on the current prompt and reflective dataset, selecting the
    highest-scoring action rather than cycling blindly.  Such an extension
    would add an optional ``select_with_context(n, rng, candidate,
    reflective_dataset)`` method without breaking existing selectors.
    """

    def select(self, n: int, rng: random.Random) -> list[PromptEditAction]: ...


class RoundRobinActionSelector:
    """Cycle through actions deterministically, ensuring full coverage."""

    def __init__(self, actions: list[PromptEditAction]):
        assert len(actions) > 0
        self.actions = actions
        self._counter = 0

    def select(self, n: int, rng: random.Random) -> list[PromptEditAction]:
        result = []
        for _ in range(n):
            result.append(self.actions[self._counter % len(self.actions)])
            self._counter += 1
        return result


class RandomActionSelector:
    """Pick actions uniformly at random from the action space."""

    def __init__(self, actions: list[PromptEditAction], rng: random.Random | None = None):
        assert len(actions) > 0
        self.actions = actions
        self.rng = rng if rng is not None else random.Random(0)

    def select(self, n: int, rng: random.Random) -> list[PromptEditAction]:
        return [self.rng.choice(self.actions) for _ in range(n)]


class AllActionsSelector:
    """Return all actions regardless of ``n`` (for best-of-N expansion)."""

    def __init__(self, actions: list[PromptEditAction]):
        assert len(actions) > 0
        self.actions = actions

    def select(self, n: int, rng: random.Random) -> list[PromptEditAction]:
        return list(self.actions)


def format_action_suffix(action: PromptEditAction) -> str:
    """Build the constraint text appended to the reflection prompt."""
    return (
        "\n\n--- ACTION CONSTRAINT ---\n"
        f"You MUST make exactly one type of edit: {action.name}\n"
        f"Description: {action.description}\n\n"
        f"{action.instruction_suffix}\n\n"
        "Do not make any other type of change. Focus exclusively on this edit action."
    )


DEFAULT_ACTIONS: list[PromptEditAction] = [
    PromptEditAction(
        name="add_illustration",
        description="Add an inline worked example or case illustration within the instruction text.",
        instruction_suffix=(
            "Add a concrete, illustrative example within the instruction that demonstrates "
            "correct behavior on a specific pattern observed in the feedback. Use a 'For example, ...' "
            "or 'For instance, ...' construction. The example should help the assistant generalize "
            "from the observed failure to similar future cases."
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
