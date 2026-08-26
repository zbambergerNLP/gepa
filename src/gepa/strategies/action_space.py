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
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar, cast

from gepa.proposer.reflective_mutation.base import LanguageModel

SelectableItemT = TypeVar("SelectableItemT")


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
    """Return the reproducibility contract for a stateless selection policy.

    Random selection records only its uniform sampling and selection
    granularity. Verbalized selection also records its context, tail threshold,
    candidate count, and optional full-support exploration policy.

    Args:
        selector: Selection implementation represented by the contract.
        per_job_action_selection: Whether each reflection job selects its own
            action instead of sharing one batch-level action.
        k: Number of candidates requested from the verbalized selector.
        tau: Explicit tail-sampling threshold; ``None`` resolves to ``1 / k``.
        require_full_support: Whether verbalized sampling mixes the complete
            distribution with uniform exploration.

    Returns:
        JSON-serializable policy fields used for reproducible runs.

    Raises:
        ValueError: Verbalized selection is requested with ``k`` below one.
    """
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
    """Validate the stable identifiers in one selector menu.

    Args:
        actions: Selectable items exposed by the menu.
        selector_name: Selector name included in validation errors.

    Raises:
        ValueError: The menu is empty, an identifier is empty or padded with
            whitespace, or identifiers are duplicated case-insensitively.
    """
    if not actions:
        raise ValueError(f"{selector_name} requires a non-empty actions list")
    menu_ids = [cast(Any, action).menu_id for action in actions]
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

    Implementations accept the historical two-argument selection call.
    :class:`VerbalizedActionSelector` additionally accepts candidate and
    feedback context as explicit keyword arguments.
    """

    select: Callable[[int, random.Random | None], list[SelectableItemT]]


class RandomActionSelector(Generic[SelectableItemT]):
    """Pick actions uniformly at random from the action space."""

    def __init__(self, actions: list[SelectableItemT], rng: random.Random | None = None):
        """Configure uniform selection over a validated action menu.

        Args:
            actions: Non-empty menu whose stable identifiers must be unique.
            rng: Default RNG for later selections; a seed-zero RNG is created
                when omitted.

        Raises:
            ValueError: The menu is empty or contains an empty, padded, or
                case-insensitively duplicated identifier.
        """
        _validate_actions(actions, type(self).__name__)
        self.actions = actions
        self.rng = rng if rng is not None else random.Random(0)

    def select(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        candidate: str | None = None,
        feedback_summary: str | None = None,
    ) -> list[SelectableItemT]:
        """Draw actions uniformly with replacement.

        Args:
            n: Number of actions to draw.
            rng: RNG override for this call; the instance RNG is used when
                omitted.
            candidate: Context accepted for selector-interface compatibility
                and ignored by uniform selection.
            feedback_summary: Feedback accepted for selector-interface
                compatibility and ignored by uniform selection.

        Returns:
            Randomly selected actions in draw order.
        """
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

    Pass the current candidate and feedback directly to ``select()``. Calls
    without either input fall back to uniform random selection.

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
        ValueError: The menu is empty or contains an empty, padded, or
            case-insensitively duplicated identifier.
        ZeroDivisionError: ``k`` is zero and ``tau`` is omitted.
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
        """Configure verbalized selection without calling the language model.

        Args:
            actions: Non-empty menu whose stable identifiers must be unique.
            lm: Model that verbalizes an action distribution.
            k: Number of candidates requested from the model.
            tau: Tail-sampling threshold; ``None`` resolves to ``1 / k``.
            rng: Default RNG for sampling; a seed-zero RNG is created when
                omitted.
            require_full_support: Whether model output must score every action
                and sampling mixes in uniform exploration.

        Raises:
            ValueError: The menu is empty or contains an empty, padded, or
                case-insensitively duplicated identifier.
            ZeroDivisionError: ``k`` is zero and ``tau`` is omitted.
        """
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
        self._action_by_id: dict[str, SelectableItemT] = {cast(Any, action).menu_id: action for action in actions}
        self.history: list[dict] = []

    def select(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        candidate: str | None = None,
        feedback_summary: str | None = None,
    ) -> list[SelectableItemT]:
        """Draw ``n`` actions using the candidate and feedback supplied now.

        Makes one LM call to verbalize a distribution over the menu, tail-samples
        from it, and appends one record (the distribution, the picks, and the
        :class:`TailSampleStats`) to :attr:`history`. Selection context is never
        retained on the selector between calls.

        Args:
            n: Number of actions to draw (with replacement, one per job).
            rng: RNG override for this call; the instance RNG is used when ``None``.
            candidate: Current document component shown to the selector model.
            feedback_summary: Failure feedback used to score the menu.

        Returns:
            The sampled actions in draw order. When either context argument is
            omitted, the draw is uniform over the menu, no LM call is made, and
            nothing is recorded in :attr:`history`.
        """
        rng = rng if rng is not None else self.rng
        if candidate is None or feedback_summary is None:
            logger.warning("VerbalizedActionSelector.select() called without context; falling back to uniform.")
            return [rng.choice(self.actions) for _ in range(n)]

        distribution = self._generate_distribution(rng, candidate, feedback_summary)
        if self.require_full_support:
            epsilon = FULL_SUPPORT_EXPLORATION_EPSILON
            actions = [action for action, _, _ in distribution.entries]
            probabilities = [probability for _, probability, _ in distribution.entries]
            mixed_probabilities = [
                (1.0 - epsilon) * probability + epsilon / len(actions) for probability in probabilities
            ]
            result = rng.choices(actions, weights=mixed_probabilities, k=n)
            sampled_probability_by_id = {
                cast(Any, action).menu_id: probability
                for action, probability in zip(actions, mixed_probabilities, strict=True)
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
                    menu_id: sum(
                        probability
                        for action, probability, _ in eligible
                        if cast(Any, action).menu_id == menu_id
                    )
                    / eligible_total
                    for menu_id in {cast(Any, action).menu_id for action, _, _ in eligible}
                }
            else:
                eligible_ids = [cast(Any, action).menu_id for action, _, _ in eligible]
                sampled_probability_by_id = {
                    menu_id: eligible_ids.count(menu_id) / len(eligible_ids) for menu_id in set(eligible_ids)
                }
            sampling_policy = "tail"
        self.history.append(
            {
                "probs": {
                    cast(Any, action).menu_id: probability for action, probability, _ in distribution.entries
                },
                "sampling_probs": sampled_probability_by_id,
                "sampled": [cast(Any, action).menu_id for action in result],
                "sampled_probabilities": [sampled_probability_by_id[cast(Any, action).menu_id] for action in result],
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
        return result

    def _generate_distribution(
        self,
        rng: random.Random,
        candidate: str,
        feedback_summary: str,
    ) -> ActionDistribution[SelectableItemT]:
        """Ask the language model for and parse an action distribution.

        The prompt contains the current candidate, feedback, complete menu, and
        full-support rule when enabled. Parsing normalizes valid probabilities
        or supplies the parser's uniform fallback.

        Args:
            rng: Selection RNG passed through to the parser interface.
            candidate: Current document component shown to the selector model.
            feedback_summary: Failure feedback used to score the menu.

        Returns:
            Parsed and normalized action distribution.
        """
        action_menu = "\n".join(
            f"- {cast(Any, action).menu_id}: {cast(Any, action).menu_description}" for action in self.actions
        )
        prompt = VERBALIZED_ACTION_PROMPT.format(
            current_prompt=candidate,
            prompt_chars=len(candidate),
            char_budget=SOFT_PROMPT_CHAR_BUDGET,
            feedback_summary=feedback_summary,
            action_menu=action_menu,
            k=self.k,
            support_rule=(
                "Score every available action exactly once; do not omit or repeat an action. Assign probability 0 "
                "when an action's stated precondition is not supported by the region and feedback; the sampler "
                "reserves a small uniform exploration probability. This is the sole applicability judgment; "
                "downstream roles realize whichever action is sampled without reclassifying it."
                if self.require_full_support
                else ""
            ),
        )
        raw_output = self.lm(prompt)
        return self._parse_distribution(raw_output, rng)

    def _parse_distribution(self, raw_output: str, rng: random.Random) -> ActionDistribution[SelectableItemT]:
        """Parse and normalize an XML-formatted action distribution.

        Menu identifiers match exactly or case-insensitively. Without the
        full-support policy, malformed or unknown entries are skipped when the
        remaining entries define positive mass. Invalid numeric probabilities
        or non-positive total mass cause a uniform fallback. With full support,
        every configured action must appear exactly once or the complete menu
        receives the uniform fallback.

        Args:
            raw_output: Language-model response containing ``<candidate>``
                entries.
            rng: Compatibility parameter for the selector interface. Parsing
                and deterministic fallback do not consume it.

        Returns:
            Normalized parsed entries, with ``is_fallback`` set when the model
            response could not define an admissible distribution.
        """
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

        parsed_ids = [cast(Any, action).menu_id for action, _, _ in entries]
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
