# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Select edit actions for action-conditioned reflection.

``VerbalizedActionSelector`` asks the reflection LM to score the available
actions, then samples from the distribution. ``RandomActionSelector`` provides
a uniform baseline.
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.utils.text import strip_think_tags


class SelectableItem(Protocol):
    """Structural contract for one option shown to an action selector."""

    @property
    def menu_id(self) -> str:
        """Return the stable identifier used in model output and logs."""
        ...

    @property
    def menu_description(self) -> str:
        """Return the description shown in the verbalized action menu."""
        ...


SelectableItemT = TypeVar("SelectableItemT", bound=SelectableItem)


logger = logging.getLogger(__name__)

# Length pressure for evolved prompts. The selector communicates this soft
# budget while the proposer paths enforce their configured hard caps.
SOFT_PROMPT_CHAR_BUDGET = 8000
MAX_PROPOSAL_CHARS = 10000
FULL_SUPPORT_EXPLORATION_EPSILON = 0.1
DEFAULT_VERBALIZED_ACTION_K = 5
STATELESS_SELECTOR_POLICY_VERSION = 1


def stateless_selector_policy_contract(
    selector: Literal["random", "verbalized"],
    *,
    per_job_action_selection: bool = False,
    k: int = DEFAULT_VERBALIZED_ACTION_K,
    tau: float | None = None,
    require_full_support: bool = False,
) -> dict[str, Any]:
    """Return the reproducibility contract for a stateless selection policy."""
    selection_granularity = "per_job" if per_job_action_selection else "batch_shared"
    if selector == "random":
        return {
            "version": STATELESS_SELECTOR_POLICY_VERSION,
            "selector": selector,
            "selection_granularity": selection_granularity,
            "context": "none",
            "sampling": "uniform",
        }
    if k < 1:
        raise ValueError("k must be at least 1")
    resolved_tau = tau if tau is not None else 1.0 / k
    return {
        "version": STATELESS_SELECTOR_POLICY_VERSION,
        "selector": selector,
        "selection_granularity": selection_granularity,
        "context": "per_job" if per_job_action_selection else "first_parent_and_aggregated_feedback",
        "sampling": "full_distribution_uniform_mixture" if require_full_support else "tail",
        "k": k,
        "tau": resolved_tau,
        "require_full_support": require_full_support,
        "exploration_epsilon": FULL_SUPPORT_EXPLORATION_EPSILON if require_full_support else 0.0,
    }


def _validate_actions(actions: list[SelectableItemT], selector_name: str) -> None:
    """Require a non-empty menu with one non-empty stable ID per item."""
    if not actions:
        raise ValueError(f"{selector_name} requires a non-empty actions list")
    menu_ids = [action.menu_id for action in actions]
    if any(not menu_id.strip() for menu_id in menu_ids):
        raise ValueError(f"{selector_name} requires every action to have a non-empty menu_id")
    if any(menu_id != menu_id.strip() for menu_id in menu_ids):
        raise ValueError(f"{selector_name} requires menu IDs without surrounding whitespace")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for menu_id in menu_ids:
        normalized_id = menu_id.casefold()
        if normalized_id in seen:
            duplicates.add(menu_id)
        seen.add(normalized_id)
    if duplicates:
        raise ValueError(f"{selector_name} requires unique menu IDs; duplicates: {sorted(duplicates)}")


class ActionSelector(Protocol[SelectableItemT]):
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

    def select(self, n: int, rng: random.Random | None = None) -> list[SelectableItemT]: ...


class RandomActionSelector(Generic[SelectableItemT]):
    """Pick actions uniformly at random from the action space."""

    def __init__(self, actions: list[SelectableItemT], rng: random.Random | None = None):
        _validate_actions(actions, type(self).__name__)
        self.actions = actions
        self.rng = rng if rng is not None else random.Random(0)

    def select(self, n: int, rng: random.Random | None = None) -> list[SelectableItemT]:
        rng = rng if rng is not None else self.rng
        return [rng.choice(self.actions) for _ in range(n)]


VERBALIZED_ACTION_PROMPT = """\
Choose edit actions that address the document's observed failures.

## Current document component
```
{current_prompt}
```
Current component length: {prompt_chars} characters (budget: ~{char_budget}).

## Recent feedback summary
{feedback_summary}

## Available actions
{action_menu}

Score {k} candidate actions by how likely each is to improve the document given \
the feedback. Probabilities must sum to 1.0.
{support_rule}

Consider less obvious actions when the feedback supports them. If the component \
is near or over its length budget, favor actions that shorten or replace existing \
text over actions that add content.

Return:
<response>
<candidate>
<action>menu_id_here</action>
<reasoning>why this action fits the current failure patterns</reasoning>
<probability>0.XX</probability>
</candidate>
...repeat for {k} candidates...
</response>
"""


@dataclass
class ActionDistribution(Generic[SelectableItemT]):
    """A parsed distribution over actions from the verbalized selector."""

    entries: list[tuple[SelectableItemT, float, str]]  # (action, probability, reasoning)
    is_fallback: bool = False  # True when parsing failed and a uniform fallback was used

    @property
    def actions(self) -> list[SelectableItemT]:
        return [a for a, _, _ in self.entries]

    @property
    def probabilities(self) -> list[float]:
        return [p for _, p, _ in self.entries]


@dataclass(frozen=True)
class TailSampleStats:
    """Diagnostics for one ``_sample_from_tails`` call, recorded in selector history.

    Lets analysis quantify how far tail sampling diverged from the LM's
    verbalized distribution without re-deriving parser state from the
    ID-keyed ``probs`` map (which collapses duplicate menu IDs).

    Attributes:
        n_parsed_entries: Number of distribution entries, counting duplicates.
        tail_mass: Probability mass the LM placed strictly below ``tau``.
        used_full_fallback: True when the tail was empty and the full
            distribution was sampled instead.
        entropy_bits: Shannon entropy (bits) of the parsed distribution.
    """

    n_parsed_entries: int
    tail_mass: float
    used_full_fallback: bool
    entropy_bits: float


def _sample_from_tails(
    distribution: ActionDistribution[SelectableItemT],
    n: int,
    tau: float,
    rng: random.Random,
) -> tuple[list[SelectableItemT], TailSampleStats]:
    """Sample ``n`` actions from the tail of the distribution (probability < ``tau``).

    Tail sampling favors options the LM rated as unlikely-but-plausible, which
    keeps successive proposals diverse instead of repeating the LM's top pick.
    Weights are renormalized over the tail; if the LM assigned zero mass to
    every tail entry the draw is uniform over the tail.

    Args:
        distribution: The parsed verbalized distribution to sample from.
        n: Number of actions to draw (with replacement).
        tau: Threshold below which an entry counts as tail.
        rng: RNG used for the weighted draw.

    Returns:
        A ``(actions, stats)`` pair: the ``n`` sampled actions in draw order, and
        a :class:`TailSampleStats` describing the draw. When no entry falls below
        ``tau`` the full distribution is sampled instead and
        ``stats.used_full_fallback`` is ``True``.
    """
    entries = distribution.entries
    tail = [(a, p, r) for a, p, r in entries if p < tau]
    used_full_fallback = not tail

    stats = TailSampleStats(
        n_parsed_entries=len(entries),
        tail_mass=sum(p for _, p, _ in tail),
        used_full_fallback=used_full_fallback,
        entropy_bits=-sum(p * math.log2(p) for _, p, _ in entries if p > 0),
    )

    if used_full_fallback:
        tail = entries

    actions = [a for a, _, _ in tail]
    weights = [p for _, p, _ in tail]

    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(actions)] * len(actions)
    else:
        weights = [w / total for w in weights]

    return rng.choices(actions, weights=weights, k=n), stats


class VerbalizedActionSelector(Generic[SelectableItemT]):
    """Use the reflection LM to generate a probability distribution over actions, then sample.

    Instead of picking actions uniformly at random, this selector asks the LM
    which action(s) are most likely to help
    given the current prompt state and feedback. It then samples from the tails
    of the distribution (p < tau) to encourage diversity.

    Call ``set_context()`` before ``select()`` to provide the current prompt and
    feedback. If ``set_context()`` is not called, falls back to uniform random.

    Args:
        actions: The menu the LM chooses from; must be non-empty.
        lm: Language model asked to verbalize a distribution over ``actions``.
        k: Number of candidate actions the LM is asked to assign probabilities to.
        tau: Tail-sampling threshold (actions with ``p < tau`` form the tail);
            ``None`` defaults to ``1 / k`` so the threshold scales with ``k``.
        rng: RNG for tail sampling; ``random.Random(0)`` when ``None``.
        require_full_support: Whether the LM must score every configured action
            exactly once and sampling must retain nonzero support for all of
            them through a uniform-exploration mixture. Invalid output falls
            back to the uniform full menu.

    Raises:
        ValueError: ``actions`` is empty.
    """

    def __init__(
        self,
        actions: list[SelectableItemT],
        lm: LanguageModel,
        k: int = DEFAULT_VERBALIZED_ACTION_K,
        tau: float | None = None,
        rng: random.Random | None = None,
        require_full_support: bool = False,
    ):
        """Store the menu, LM, and sampling parameters; no LM call is made here."""
        _validate_actions(actions, type(self).__name__)
        self.actions = actions
        self.lm = lm
        self.k = k
        # Default the tail threshold to 1/k so it scales with the number of
        # verbalized candidates: with the default k=5 a uniform draw has no
        # tail (correct), while a peaked distribution's low-probability actions
        # still fall below it. A fixed tau=0.10 collapsed to full-distribution
        # fallback for both uniform and mildly-peaked distributions.
        self.tau = tau if tau is not None else 1.0 / k
        self.rng = rng if rng is not None else random.Random(0)
        self.require_full_support = require_full_support
        self._context: dict[str, str] | None = None
        self._action_by_id: dict[str, SelectableItemT] = {action.menu_id: action for action in actions}
        self.history: list[dict] = []

    def set_context(self, candidate: str, feedback_summary: str) -> None:
        """Provide current prompt state for the next select() call."""
        self._context = {"candidate": candidate, "feedback_summary": feedback_summary}

    def select(self, n: int, rng: random.Random | None = None) -> list[SelectableItemT]:
        """Draw ``n`` actions for the context set by the last :meth:`set_context` call.

        Makes one LM call to verbalize a distribution over the menu, tail-samples
        from it, and appends one record (the distribution, the picks, and the
        :class:`TailSampleStats`) to :attr:`history`. The stored context is
        cleared afterwards so a stale prompt state is never reused.

        Args:
            n: Number of actions to draw (with replacement, one per job).
            rng: RNG override for this call; the instance RNG is used when ``None``.

        Returns:
            The sampled actions in draw order. Without a prior
            :meth:`set_context` call the draw is uniform over the menu, no LM
            call is made, and nothing is recorded in :attr:`history`.
        """
        rng = rng if rng is not None else self.rng
        if self._context is None:
            logger.warning("VerbalizedActionSelector.select() called without set_context(); falling back to uniform.")
            return [rng.choice(self.actions) for _ in range(n)]

        distribution = self._generate_distribution(rng)
        if self.require_full_support:
            epsilon = FULL_SUPPORT_EXPLORATION_EPSILON
            actions = [action for action, _, _ in distribution.entries]
            probabilities = [probability for _, probability, _ in distribution.entries]
            mixed_probabilities = [
                (1.0 - epsilon) * probability + epsilon / len(actions) for probability in probabilities
            ]
            result = rng.choices(actions, weights=mixed_probabilities, k=n)
            sampled_probability_by_id = {
                action.menu_id: probability for action, probability in zip(actions, mixed_probabilities, strict=True)
            }
            tail_mass = sum(probability for probability in probabilities if probability < self.tau)
            stats = TailSampleStats(
                n_parsed_entries=len(distribution.entries),
                tail_mass=tail_mass,
                used_full_fallback=False,
                entropy_bits=-sum(
                    probability * math.log2(probability) for probability in probabilities if probability > 0
                ),
            )
            sampling_policy = "full_distribution_uniform_mixture"
        else:
            epsilon = 0.0
            result, stats = _sample_from_tails(distribution, n, self.tau, rng)
            eligible = (
                distribution.entries
                if stats.used_full_fallback
                else [
                    (action, probability, reasoning)
                    for action, probability, reasoning in distribution.entries
                    if probability < self.tau
                ]
            )
            eligible_total = sum(probability for _, probability, _ in eligible)
            if eligible_total > 0:
                sampled_probability_by_id = {
                    menu_id: sum(probability for action, probability, _ in eligible if action.menu_id == menu_id)
                    / eligible_total
                    for menu_id in {action.menu_id for action, _, _ in eligible}
                }
            else:
                eligible_ids = [action.menu_id for action, _, _ in eligible]
                sampled_probability_by_id = {
                    menu_id: eligible_ids.count(menu_id) / len(eligible_ids) for menu_id in set(eligible_ids)
                }
            sampling_policy = "tail"
        self.history.append(
            {
                "probs": {action.menu_id: probability for action, probability, _ in distribution.entries},
                "sampling_probs": sampled_probability_by_id,
                "sampled": [action.menu_id for action in result],
                "sampled_probabilities": [sampled_probability_by_id[action.menu_id] for action in result],
                "fallback": distribution.is_fallback,
                "n_parsed_entries": stats.n_parsed_entries,
                "tail_mass": stats.tail_mass,
                "tau": self.tau,
                "sampling_policy": sampling_policy,
                "exploration_epsilon": epsilon,
                "used_full_fallback": stats.used_full_fallback,
                "entropy_bits": stats.entropy_bits,
            }
        )
        self._context = None
        return result

    def _generate_distribution(self, rng: random.Random) -> ActionDistribution[SelectableItemT]:
        """Call the LM to produce a verbalized probability distribution over actions."""
        assert self._context is not None
        action_menu = "\n".join(f"- {action.menu_id}: {action.menu_description}" for action in self.actions)
        prompt = VERBALIZED_ACTION_PROMPT.format(
            current_prompt=self._context["candidate"],
            prompt_chars=len(self._context["candidate"]),
            char_budget=SOFT_PROMPT_CHAR_BUDGET,
            feedback_summary=self._context["feedback_summary"],
            action_menu=action_menu,
            k=self.k,
            support_rule=(
                "Score every available action exactly once; do not omit or repeat an action. Assign probability 0 "
                "when an action's stated precondition is not supported by the region and feedback; the sampler "
                "reserves a small uniform exploration probability."
                if self.require_full_support
                else ""
            ),
        )
        raw_output = self.lm(prompt)
        return self._parse_distribution(raw_output, rng)

    def _parse_distribution(self, raw_output: str, rng: random.Random) -> ActionDistribution[SelectableItemT]:
        """Parse XML-formatted action distribution from LM output."""
        raw_output = strip_think_tags(raw_output)

        entries: list[tuple[SelectableItemT, float, str]] = []
        invalid_candidate = False
        invalid_probability = False
        for candidate_match in re.finditer(r"<candidate>(.*?)</candidate>", raw_output, re.DOTALL):
            block = candidate_match.group(1)
            action_m = re.search(r"<action>(.*?)</action>", block, re.DOTALL)
            prob_m = re.search(r"<probability>(.*?)</probability>", block, re.DOTALL)
            reasoning_m = re.search(r"<reasoning>(.*?)</reasoning>", block, re.DOTALL)

            if not action_m or not prob_m:
                invalid_candidate = True
                continue

            menu_id = action_m.group(1).strip()
            reasoning = reasoning_m.group(1).strip() if reasoning_m else ""

            try:
                probability = float(prob_m.group(1).strip())
            except ValueError:
                invalid_candidate = True
                invalid_probability = True
                continue
            if not math.isfinite(probability) or probability < 0:
                invalid_candidate = True
                invalid_probability = True
                continue

            action = self._action_by_id.get(menu_id)
            if action is None:
                for candidate_id, act in self._action_by_id.items():
                    if candidate_id.lower() == menu_id.lower():
                        action = act
                        break
            if action is not None:
                entries.append((action, probability, reasoning))
            else:
                invalid_candidate = True

        parsed_ids = [action.menu_id for action, _, _ in entries]
        full_support_invalid = self.require_full_support and (
            invalid_candidate or len(parsed_ids) != len(self.actions) or set(parsed_ids) != set(self._action_by_id)
        )
        total = sum(probability for _, probability, _ in entries)
        is_fallback = False
        if not entries or full_support_invalid or invalid_probability or total <= 0:
            reason = "incomplete" if full_support_invalid else "invalid"
            logger.warning("Received %s verbalized action distribution; falling back to uniform.", reason)
            n_actions = len(self.actions)
            entries = [(a, 1.0 / n_actions, "") for a in self.actions]
            is_fallback = True
            total = 1.0

        entries = [(a, p / total, r) for a, p, r in entries]

        return ActionDistribution(entries=entries, is_fallback=is_fallback)
