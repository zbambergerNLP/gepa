# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Reflection LM abstraction (#329 Phase 1).

A ``ReflectionLM`` proposes new component texts and returns the (possibly
extended) reflection LM to use next.  The stateless default
(:class:`StatelessReflectionLM`) returns itself, so behavior is identical to
GEPA's historical single-callable reflection.  Sessions, agents, and ComBEE
become additional implementations in later phases.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.action_space import ActionSelector
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.intervention import (
    StatelessActionConstraint,
    format_stateless_action_constraint,
)

# One reflection job = (candidate, reflective_dataset, components_to_update).
ReflectionJob = tuple[dict[str, str], "Mapping[str, Sequence[Mapping[str, Any]]]", list[str]]


@dataclass
class ReflectionProposal:
    """Output of one :meth:`ReflectionLM.reflect` call.

    ``new_texts`` maps component name -> proposed text.  ``prompts`` and
    ``raw_lm_outputs`` are optional per-component diagnostics for callbacks and
    experiment trackers.
    """

    new_texts: dict[str, str]
    prompts: dict[str, str | list[dict[str, Any]]] = field(default_factory=dict)
    raw_lm_outputs: dict[str, str] = field(default_factory=dict)
    # Free-form diagnostics for multi-call reflection strategies. ``prompts``/
    # ``raw_lm_outputs`` assume ONE LM call per component; strategies that make
    # several (e.g. ComBEE's k map calls + 1 reduce call per component) should
    # record per-call intermediates here (namespaced keys, e.g.
    # "combee:level1_prompts"). Merged into CandidateProposal.metadata, so it
    # reaches on_proposal_end consumers, experiment trackers, and the run
    # manifest without a future protocol break.
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ReflectionLM(Protocol):
    """Proposes new component texts.

    ``reflect`` returns ``(proposal, next_lm)``: the reflection LM to use next.
    Stateless implementations return ``self``; stateful ones return an extended
    copy, leaving the original reusable/forkable.  See #329.
    """

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ReflectionLM]: ...


@runtime_checkable
class BatchReflectionLM(ReflectionLM, Protocol):
    """A :class:`ReflectionLM` that can reflect on many jobs in one batched call.

    ``reflect_many`` is the vectorized form of ``reflect``: given N independent
    jobs (one per candidate proposed this iteration), it returns N
    ``(proposal, next_lm)`` results in order.  Implementations issue the
    underlying LLM calls as a single batched/concurrent request (e.g.
    ``litellm.batch_completion``), so per-iteration proposal throughput comes
    from one batched call at the reflection edge rather than engine threads.

    ``reflect`` is just the N=1 case and must stay consistent with it.
    """

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ReflectionLM]]: ...


@runtime_checkable
class SeedableReflectionLM(ReflectionLM, Protocol):
    """A :class:`ReflectionLM` whose internal randomness can be bound to GEPA's run seed.

    When a reflection strategy defines ``bind_rng``, GEPA's front doors call it
    at wiring time with the engine's seeded RNG. Implementations should treat
    it as a default: a user-supplied explicit RNG must win over the bound one.
    Sharing the stream preserves legacy strategies such as #307 ComBEE, whose
    shuffles intentionally participate in subsequent engine sampling.
    """

    def bind_rng(self, rng: Any) -> None: ...


class StatelessReflectionLM:
    """Default reflection LM: one stateless LM call per component (or one batched call covering all tasks/components when the underlying LM provides ``batch_complete``).

    For each component with feedback, render the instruction-proposal prompt
    (honoring a global or per-component template) and parse the new instruction.
    ``reflect`` returns ``self`` — there is no carried state.

    Args:
        lm: The reflection language model; a ``batch_complete`` method, if
            present, is used to issue all prompts of a batch in one call.
        reflection_prompt_template: A prompt template string applied to every
            component, a component-name -> template mapping, or ``None`` for
            the default template.
        logger: Optional run logger with a ``log(message)`` method.
        action_selector: Optional selector that picks one
            :class:`StatelessActionConstraint` per job and appends the canonical
            semantic action and region constraint to the prompt; ``None``
            disables action-conditioned reflection.
        rng: RNG passed to the action selector; ``random.Random(0)`` when
            ``None``, rebound to the run RNG through :meth:`bind_rng`.
        per_job_action_selection: Choose each job's action from its own context
            (one selector call per job) instead of one selector call for the
            whole batch; see :meth:`_select_actions_per_job`.
    """

    def __init__(
        self,
        lm: LanguageModel,
        reflection_prompt_template: str | dict[str, str] | None = None,
        logger: Any | None = None,
        action_selector: ActionSelector[StatelessActionConstraint] | None = None,
        rng: random.Random | None = None,
        per_job_action_selection: bool = False,
    ):
        """Store the LM, template, logger and action-selection settings."""
        self.lm = lm
        self.reflection_prompt_template = reflection_prompt_template
        self.logger = logger
        self.action_selector = action_selector
        self.rng = rng if rng is not None else random.Random(0)
        # Opt-in (#5): choose one action per job from that job's own context
        # instead of one selector call for the whole batch. Costs one selector
        # call per job but avoids conditioning every job's action on aggregated
        # cross-job context; provided so the two can be compared empirically.
        self.per_job_action_selection = per_job_action_selection
        # Components already warned about a missing per-component template (warn once).
        self._missing_template_warnings: set[str] = set()

    def bind_rng(self, rng: random.Random) -> None:
        """Bind GEPA's seeded run RNG (:class:`SeedableReflectionLM`).

        The front doors call this at wiring time so action selection derives
        from the run seed rather than this reflector's construction-time default
        (``Random(0)``). ``reflect_many`` passes ``self.rng`` to the action
        selector, so seeding here also seeds selection.

        Args:
            rng: The run RNG to use for action selection from now on.
        """
        self.rng = rng

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    @staticmethod
    def _summarize_feedback(
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        max_chars: int = 500,
    ) -> str:
        """Extract a compact feedback summary from the reflective dataset."""
        parts: list[str] = []
        for _name, entries in reflective_dataset.items():
            for entry in entries:
                fb = entry.get("Feedback") or entry.get("execution_feedback") or ""
                if fb:
                    parts.append(str(fb))
        summary = "\n".join(parts)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."
        return summary or "(no feedback available)"

    def _resolve_template(self, name: str) -> str | None:
        if isinstance(self.reflection_prompt_template, dict):
            template = self.reflection_prompt_template.get(name)
            if template is None and name not in self._missing_template_warnings:
                self._log(f"No reflection_prompt_template found for parameter '{name}'. Using default template.")
                self._missing_template_warnings.add(name)
            return template
        return self.reflection_prompt_template

    def _render(
        self,
        current_instruction_doc: str,
        dataset_with_feedback: Any,
        prompt_template: str | None,
        action: StatelessActionConstraint | None = None,
    ):
        """Render a reflection prompt and its chat-messages form.

        When *action* is provided, append the action constraint suffix to the
        rendered prompt so the reflection LM is constrained to a single edit type.
        """
        prompt = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current_instruction_doc,
                "dataset_with_feedback": dataset_with_feedback,
                "prompt_template": prompt_template,
            }
        )

        if action is not None:
            suffix = format_stateless_action_constraint(action)
            if isinstance(prompt, str):
                prompt = prompt + suffix
            else:
                for msg in reversed(prompt):
                    if msg.get("role") == "user":
                        content = msg["content"]
                        if isinstance(content, str):
                            msg["content"] = content + suffix
                        elif isinstance(content, list):
                            content.append({"type": "text", "text": suffix})
                        break

        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return prompt, messages

    def _batch_complete(self, prompts: list[Any], messages_list: list[list[dict[str, Any]]]) -> list[str]:
        """Issue the reflection completions, batched when possible.

        A single prompt uses the plain completion path, so N=1 is byte-identical
        to the historical single reflection.  When the LM exposes
        ``batch_complete`` (``litellm.batch_completion``), all prompts go out as
        one concurrent request; a custom callable without it runs sequentially.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.lm(prompts[0])]
        batch_complete = getattr(self.lm, "batch_complete", None)
        if batch_complete is not None:
            return list(batch_complete(messages_list))
        return [self.lm(prompt) for prompt in prompts]

    def _select_actions_batch(self, jobs: list[ReflectionJob]) -> list[StatelessActionConstraint | None]:
        """Choose all jobs' actions in one selector call (default cost tradeoff).

        Verbalized selectors receive context aggregated across the batch:
        feedback from every job, and the first job's candidate text (with a note
        when parents differ). Programmatic selectors ignore the context. This
        uses one selector call, with every action conditioned on shared context.

        Args:
            jobs: The batch of ``(candidate, reflective_dataset, components)``
                triples being reflected on.

        Returns:
            One selected action per job, in job order.
        """
        assert self.action_selector is not None
        set_context = getattr(self.action_selector, "set_context", None)
        if set_context is not None and jobs:
            candidate_text = "\n\n".join(jobs[0][0].values())
            distinct_parents = any(job[0] != jobs[0][0] for job in jobs[1:])
            if distinct_parents:
                candidate_text += f"\n\n(1 of {len(jobs)} distinct parent candidates shown)"
            feedback_summary = "\n---\n".join(self._summarize_feedback(job[1]) for job in jobs)
            set_context(candidate_text, feedback_summary)
        return list(self.action_selector.select(len(jobs), self.rng))

    def _select_actions_per_job(self, jobs: list[ReflectionJob]) -> list[StatelessActionConstraint | None]:
        """Choose each job's action from its own context.

        One selector call per job, each seeing only that job's candidate text and
        feedback. Costs one selector call per job rather than one per batch, in
        exchange for per-job conditioning; exists to compare against the batch
        default. Shares ``self.rng`` across calls so selection stays seeded.

        Args:
            jobs: The batch of ``(candidate, reflective_dataset, components)``
                triples being reflected on.

        Returns:
            One selected action per job, in job order.
        """
        assert self.action_selector is not None
        set_context = getattr(self.action_selector, "set_context", None)
        actions: list[StatelessActionConstraint | None] = []
        for candidate, reflective_dataset, _components in jobs:
            if set_context is not None:
                set_context("\n\n".join(candidate.values()), self._summarize_feedback(reflective_dataset))
            actions.extend(self.action_selector.select(1, self.rng))
        return actions

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, StatelessReflectionLM]:
        return self.reflect_many([(candidate, reflective_dataset, components_to_update)])[0]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, StatelessReflectionLM]]:
        """Propose new texts for every job's components in one batched pass.

        When an action selector is configured, one action is chosen per job
        first (batched or per job, see ``per_job_action_selection``) and its
        instruction suffix conditions that job's prompts. Every
        ``(job, component)`` pair with reflective data is rendered into a
        prompt; a component with no rows is logged and skipped. All prompts are
        issued together through :meth:`_batch_complete`, and the parsed
        instructions are scattered back into one proposal per job, with the
        chosen action's name recorded under ``metadata["action"]``.

        Args:
            jobs: ``(candidate, reflective_dataset, components_to_update)``
                triples, one per proposal to make.

        Returns:
            One ``(proposal, self)`` pair per job, in job order; ``self`` is
            returned as the next reflection LM because no state is carried.
        """
        actions: list[StatelessActionConstraint | None]
        if self.action_selector is None:
            actions = [None] * len(jobs)
        elif self.per_job_action_selection:
            actions = self._select_actions_per_job(jobs)
        else:
            actions = self._select_actions_batch(jobs)

        rendered: list[tuple[int, str, Any, list[dict[str, Any]]]] = []
        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            action = actions[job_idx] if job_idx < len(actions) else None
            for name in components_to_update:
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                    continue
                prompt, messages = self._render(
                    candidate[name], reflective_dataset[name], self._resolve_template(name), action=action
                )
                rendered.append((job_idx, name, prompt, messages))

        raw_outputs = self._batch_complete([r[2] for r in rendered], [r[3] for r in rendered])

        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}) for _ in jobs]
        for (job_idx, name, prompt, _messages), raw_output in zip(rendered, raw_outputs, strict=True):
            new_instruction = InstructionProposalSignature.output_extractor(raw_output.strip())["new_instruction"]
            proposals[job_idx].new_texts[name] = new_instruction
            proposals[job_idx].prompts[name] = prompt
            proposals[job_idx].raw_lm_outputs[name] = raw_output

        for job_idx, action in enumerate(actions):
            if action is not None and job_idx < len(proposals):
                proposals[job_idx].metadata.update(
                    {
                        "action": action.semantic_action.name,
                        "semantic_action": action.semantic_action.name,
                        "action_choice": action.menu_id,
                        "action_operator": action.edit_tool.value,
                        "action_target_section": action.target_section,
                    }
                )

        return [(proposal, self) for proposal in proposals]
