"""Apply HotPotQA's stateless action ablation to every Terminal-Bench document."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, cast

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal, StatelessReflectionLM
from gepa.strategies.action_space import ActionSelector, VerbalizedActionSelector
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import SEMANTIC_ACTIONS, StatelessActionConstraint
from gepa.strategies.text_limits import TextLimitError, TextLimits, resolve_text_limits


class ComponentActionReflectionLM:
    """Combine independent action-conditioned document edits into one candidate."""

    def __init__(
        self,
        *,
        lm: LanguageModel,
        selector_lm: LanguageModel,
        component_kinds: dict[str, str],
        template_family: str,
        rng: random.Random,
        text_limits: TextLimits | None = None,
    ) -> None:
        """Build the existing HotPotQA selector and rewriter for each document kind.

        Args:
            lm: Model that rewrites the selected section body once.
            selector_lm: Same-model client that chooses a semantic action/section.
            component_kinds: Document role for every editable component.
            template_family: Provider-specific prompt and skill section templates.
            rng: Seeded selection stream, isolated from training-task sampling.
            text_limits: Optional limits shared by each selector and rewriter.
        """
        self.component_kinds = dict(component_kinds)
        self.rng = rng
        self.text_limits = resolve_text_limits(text_limits)
        self.reflectors: dict[str, StatelessReflectionLM] = {}
        for kind in sorted(set(component_kinds.values())):
            template = TEMPLATE_FAMILIES[template_family][kind]
            actions = [
                StatelessActionConstraint(spec, section, template)
                for section in template.sections
                for spec in SEMANTIC_ACTIONS
            ]
            self.reflectors[kind] = StatelessReflectionLM(
                lm,
                action_selector=cast(
                    ActionSelector[StatelessActionConstraint],
                    VerbalizedActionSelector(actions, lm=selector_lm, text_limits=self.text_limits),
                ),
                rng=rng,
                text_limits=self.text_limits,
            )

    def bind_logger(self, logger: Any) -> None:
        """Route diagnostics from both document kinds to the optimization log."""
        for reflector in self.reflectors.values():
            reflector.logger = logger

    def get_state(self) -> dict[str, Any]:
        """Checkpoint selection randomness without carrying edit conversation history."""
        return {"rng_state": self.rng.getstate()}

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Resume the next semantic action draw from a durable checkpoint."""
        self.rng.setstate(state["rng_state"])

    def get_batch_retry_state(self) -> dict[str, Any]:
        """Snapshot selector histories and LM cursors before a recoverable retry."""
        return {kind: reflector.get_batch_retry_state() for kind, reflector in self.reflectors.items()}

    def set_batch_retry_state(self, state: Mapping[str, Any]) -> None:
        """Restore the selected actions' RNG and diagnostic state before retrying."""
        for kind, reflector in self.reflectors.items():
            reflector.set_batch_retry_state(state[kind])

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ComponentActionReflectionLM]:
        """Revise one section per selected component from the same parent harness.

        Args:
            candidate: Unmodified parent shared by all document edits.
            reflective_dataset: Per-component feedback from one sampled minibatch.
            components_to_update: All documents selected for this mutation.

        Returns:
            One combined proposal with per-component action diagnostics, and self.
        """
        proposal = ReflectionProposal(new_texts={}, metadata={"component_actions": {}})
        for name in components_to_update:
            if not reflective_dataset.get(name):
                continue
            reflector = self.reflectors[self.component_kinds[name]]
            edit, _ = reflector.reflect(candidate, {name: reflective_dataset[name]}, [name])
            proposal.new_texts.update(edit.new_texts)
            proposal.prompts.update(edit.prompts)
            proposal.raw_lm_outputs.update(edit.raw_lm_outputs)
            proposal.metadata["component_actions"][name] = edit.metadata
        if proposal.new_texts:
            try:
                self.text_limits.check_candidate({**candidate, **proposal.new_texts})
            except TextLimitError as exc:
                proposal.metadata["length_capped_dropped"] = list(proposal.new_texts)
                proposal.metadata["text_limit_error"] = str(exc)
                proposal.new_texts.clear()
        return proposal, self
