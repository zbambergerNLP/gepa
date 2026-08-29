# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import inspect
import random
import traceback
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from gepa.core.adapter import (
    DataInst,
    EvaluationBatch,
    GEPAAdapter,
    ProposalFn,
    RolloutOutput,
    Trajectory,
    invoke_batch_evaluate,
)
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import TRAINSET_CACHE_SPLIT, GEPAState, _candidate_hash
from gepa.lm import LMProviderError, ProviderIdentityMismatchError
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionLM, StatelessReflectionLM
from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.action_space import ActionSelector
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.intervention import StatelessActionConstraint
from gepa.strategies.proposal_sampling import ProposalTask, SamplingStrategy, SingleMutationSampling

_FATAL_REFLECTION_EXCEPTIONS = (LMProviderError, ProviderIdentityMismatchError, ResponseJournalError)


class ReflectiveMutationProposer:
    """Implements the reflective mutation flow.

    Each iteration, the proposer:

    1. Samples one or more (parent, minibatch) tasks via ``sampling_strategy``
    2. Batch-evaluates all parents (deduplicated)
    3. For each task: builds a reflective dataset and proposes new texts
    4. Batch-evaluates all children
    5. Returns ALL evaluated proposals as :class:`CandidateProposal` objects —
       acceptance and selection are applied by the engine, which is the single
       accept+select authority

    With the default ``SingleMutationSampling``, this produces exactly one
    task per iteration — matching GEPA's original sequential behavior.
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float | None,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        custom_candidate_proposer: ProposalFn | None = None,
        callbacks: list[GEPACallback] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        reflection_strategy: ReflectionLM | None = None,
        action_selector: ActionSelector[StatelessActionConstraint] | None = None,
    ):
        """Configure reflective proposal generation and minibatch evaluation.

        Args:
            logger: Run logger for diagnostics.
            trainset: Training examples or loader sampled for reflection.
            adapter: Task adapter used for evaluation and optional proposals.
            candidate_selector: Policy selecting parent candidates.
            module_selector: Policy selecting candidate components to mutate.
            batch_sampler: Policy selecting training examples for each task.
            perfect_score: Score treated as perfect, or ``None`` if undefined.
            skip_perfect_score: Whether perfect minibatches skip reflection.
            experiment_tracker: Tracker receiving proposal diagnostics.
            reflection_lm: Model used by the default stateless reflector.
            reflection_prompt_template: Shared or per-component reflection
                template.
            custom_candidate_proposer: Optional caller-owned proposal function.
            callbacks: Proposal lifecycle observers.
            sampling_strategy: Multi-proposal task sampler, or the single-task
                default when omitted.
            reflection_strategy: Optional stateful or custom reflection owner.
            action_selector: Optional stateless semantic-action selector.

        Raises:
            ValueError: Prompt templates are invalid or a reflection strategy
                is supplied while an adapter or custom proposer already owns
                proposal generation.
        """
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.custom_candidate_proposer = custom_candidate_proposer
        self.callbacks = callbacks
        self.sampling_strategy: SamplingStrategy = sampling_strategy or SingleMutationSampling()
        self.action_selector = action_selector

        self.reflection_prompt_template = reflection_prompt_template

        if isinstance(reflection_prompt_template, dict):
            for _param_name, template in reflection_prompt_template.items():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)

        if reflection_strategy is not None and (
            adapter.propose_new_texts is not None or custom_candidate_proposer is not None
        ):
            owner = (
                "adapter.propose_new_texts" if adapter.propose_new_texts is not None else "custom_candidate_proposer"
            )
            raise ValueError(
                f"reflection_strategy was provided, but {owner} owns proposal generation "
                "and the reflection strategy would be silently ignored. Remove one of the two."
            )

        # Reflection LM (#329 Phase 1); None when an adapter/custom proposer owns
        # reflection. An injected reflection_strategy — any ReflectionLM
        # implementation, e.g. session-based or ComBEE-style aggregating
        # reflectors (#329 Phase 2/3) — takes precedence over the stateless
        # default built from the raw reflection_lm callable.
        if reflection_strategy is not None:
            _bind_template = getattr(reflection_strategy, "bind_reflection_prompt_template", None)
            if callable(_bind_template):
                _bind_template(reflection_prompt_template)
        self._reflection_lm: ReflectionLM | None = reflection_strategy or (
            StatelessReflectionLM(
                reflection_lm,
                reflection_prompt_template,
                logger,
                action_selector=self.action_selector,
            )
            if reflection_lm is not None
            else None
        )

        if self.skip_perfect_score and self.perfect_score is None:
            raise ValueError(
                "perfect_score must be provided when skip_perfect_score is True. "
                "If you do not have a perfect target score, set skip_perfect_score=False."
            )

    def bind_reflection_rng(self, rng: random.Random) -> None:
        """Bind GEPA's seeded run RNG to the effective reflection LM.

        Front doors call this after construction so the default
        ``StatelessReflectionLM`` (built here, otherwise seeded ``Random(0)``)
        derives action selection from the run seed. Idempotent for an injected
        reflection_strategy, which the front door also binds at wiring time.

        Args:
            rng: The run RNG; forwarded to the reflection LM's ``bind_rng`` when
                it has one, otherwise ignored.
        """
        bind = getattr(self._reflection_lm, "bind_rng", None)
        if callable(bind):
            bind(rng)

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
        """Propose new instruction texts for the given components.

        ``metadata`` is open-ended parent context. It is forwarded to a custom
        proposer accepting ``metadata`` and to a ``ReflectionLM.reflect`` method
        accepting the same keyword; legacy three-argument implementations remain
        unchanged. GEPA supplies on-disk iteration anchors, the selected
        ``candidate_idx``, and that candidate's accepted
        ``branch_edit_history``. The adapter-owned path keeps its legacy
        three-positional-argument signature.

        Args:
            candidate: Parent component mapping.
            reflective_dataset: Per-component feedback and execution evidence.
            components_to_update: Components selected for mutation.
            metadata: Open parent-specific context forwarded when supported.

        Returns:
            A tuple of (new_texts, prompts, raw_lm_outputs, reflection_metadata)
            where the first three are dicts keyed by component name and
            ``reflection_metadata`` is the ReflectionLM's free-form diagnostics
            (empty for single-call reflectors; multi-call strategies such as
            ComBEE record per-call intermediates here).

        Raises:
            ValueError: No adapter, custom proposer, or reflection model is
                available to generate the requested texts.
        """
        empty: dict[str, str | list[dict[str, Any]]] = {}
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update), empty, {}, {}

        if self.custom_candidate_proposer is not None:
            # Custom proposers may use the legacy 3-positional signature; only
            # pass metadata= when the signature accepts it.
            try:
                sig = inspect.signature(self.custom_candidate_proposer)
                accepts_metadata = "metadata" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_metadata = False
            new_texts = self.custom_candidate_proposer(
                candidate,
                reflective_dataset,
                components_to_update,
                **({"metadata": metadata} if accepts_metadata else {}),
            )
            return new_texts, empty, {}, {}

        if self._reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

        # Delegate to the ReflectionLM (#329 Phase 1). Stateful implementations
        # return a successor carrying accumulated context; chain it so session
        # state actually persists (stateless implementations return self,
        # making this a no-op).
        reflect = self._reflection_lm.reflect
        try:
            reflect_signature = inspect.signature(reflect)
            accepts_metadata = "metadata" in reflect_signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in reflect_signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_metadata = False
        proposal, next_lm = reflect(
            candidate,
            reflective_dataset,
            components_to_update,
            **({"metadata": metadata} if accepts_metadata else {}),
        )
        self._reflection_lm = next_lm
        return proposal.new_texts, proposal.prompts, proposal.raw_lm_outputs, proposal.metadata

    def _propose_texts_batch(
        self,
        jobs: list[tuple[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]]],
        metadatas: list[Mapping[str, Any] | None] | None = None,
    ) -> list[tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str], dict[str, Any]]]:
        """Propose new texts for many tasks, batching the reflection LM calls.

        ReflectionLM implementations that provide ``reflect_many`` (e.g. the
        stateless default, which issues one ``litellm.batch_completion``
        covering every task/component) are batched; implementations that only
        provide ``reflect()`` are called once per task. When an adapter
        proposer or custom proposer owns the call, fall back to one invocation
        per task — their batching, if any, is their concern. ``metadatas`` is
        index-aligned with ``jobs`` and forwarded per task on that fallback
        path. Custom proposers receive ``metadata=``; reflection strategies that
        accept ``metadatas=`` receive the index-aligned context in batch.

        Args:
            jobs: Candidate, reflective-dataset, and component triples.
            metadatas: Optional index-aligned parent context for each job.

        Returns:
            Proposed texts, prompts, raw outputs, and metadata in job order.

        Raises:
            ValueError: A batched reflection strategy returns a different
                number of results than jobs.
        """
        mds: list[Mapping[str, Any] | None] = metadatas if metadatas is not None else [None] * len(jobs)
        if (
            self.adapter.propose_new_texts is not None
            or self.custom_candidate_proposer is not None
            or self._reflection_lm is None
        ):
            return [
                self.propose_new_texts(cand, refds, comps, metadata=md)
                for (cand, refds, comps), md in zip(jobs, mds, strict=True)
            ]

        # SingleMutationSampling is #307's original execution path. Going
        # through reflect_many([job]) lets a batch-capable LM change its call
        # transport (and potentially its result) despite there being no PxN
        # parallelism to exploit.
        if len(jobs) == 1:
            return [self.propose_new_texts(*jobs[0], metadata=mds[0])]

        reflect_many = getattr(self._reflection_lm, "reflect_many", None)
        if reflect_many is not None:
            try:
                batch_signature = inspect.signature(reflect_many)
                accepts_metadatas = "metadatas" in batch_signature.parameters or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in batch_signature.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_metadatas = False
            results = list(reflect_many(jobs, **({"metadatas": mds} if accepts_metadatas else {})))
            if len(results) != len(jobs):
                raise ValueError(f"ReflectionLM.reflect_many returned {len(results)} results for {len(jobs)} jobs")
        else:
            return [
                self.propose_new_texts(cand, refds, comps, metadata=md)
                for (cand, refds, comps), md in zip(jobs, mds, strict=True)
            ]
        if results:
            # For batched reflection, chain to the final returned successor.
            self._reflection_lm = results[-1][1]
        return [
            (proposal.new_texts, proposal.prompts, proposal.raw_lm_outputs, proposal.metadata)
            for proposal, _next_lm in results
        ]

    def _propose_texts_batch_safe(
        self,
        jobs: list[tuple[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]]],
        metadatas: list[Mapping[str, Any] | None] | None = None,
    ) -> list[tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str], dict[str, Any]] | None]:
        """Like :meth:`_propose_texts_batch`, but isolates per-task failures.

        Returns ``None`` in the slot of any task whose reflection raised, so one
        bad task (or a failed batch) does not sink the whole iteration.

        Args:
            jobs: Candidate, reflective-dataset, and component triples.
            metadatas: Optional index-aligned parent context for each job.

        Returns:
            Job-aligned proposal payloads with ``None`` for recoverable failures.

        """
        if not jobs:
            return []
        mds: list[Mapping[str, Any] | None] = metadatas if metadatas is not None else [None] * len(jobs)
        retry_state_getter = getattr(self._reflection_lm, "get_batch_retry_state", None)
        retry_state_setter = getattr(self._reflection_lm, "set_batch_retry_state", None)
        retry_state = retry_state_getter() if callable(retry_state_getter) else None
        reflection_rng = getattr(self._reflection_lm, "rng", None)
        rng_state = None
        if retry_state is None and isinstance(reflection_rng, random.Random):
            rng_state = reflection_rng.getstate()
        try:
            return list(self._propose_texts_batch(jobs, mds))
        except Exception as e:
            if isinstance(e, _FATAL_REFLECTION_EXCEPTIONS):
                raise
            if (
                isinstance(self._reflection_lm, StatelessReflectionLM)
                and self._reflection_lm.action_selector is not None
            ):
                # The stateless reflector already retries a failed batch
                # transport with its selected actions. Retrying the whole
                # operation here would select again and change its journaled
                # Controller request at the same logical ordinal.
                raise
            if retry_state is not None:
                if not callable(retry_state_setter):
                    raise TypeError("Reflection strategy exposed retry state without a restore method.") from e
                retry_state_setter(retry_state)
            elif rng_state is not None:
                # Retry transport failures without silently changing a random
                # condition's already-sampled semantic intervention.
                reflection_rng.setstate(rng_state)
            self.logger.log(f"Batched reflection failed ({e}); retrying per task.")
            self.logger.log(traceback.format_exc())
            out: list[
                tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str], dict[str, Any]] | None
            ] = []
            for (cand, refds, comps), md in zip(jobs, mds, strict=True):
                try:
                    out.append(self.propose_new_texts(cand, refds, comps, metadata=md))
                except Exception as e2:
                    if isinstance(e2, _FATAL_REFLECTION_EXCEPTIONS):
                        raise
                    self.logger.log(f"Per-task reflection failed: {e2}")
                    out.append(None)
            return out

    # ------------------------------------------------------------------
    # Batch evaluate helper
    # ------------------------------------------------------------------

    def _batch_evaluate(self, items: list[tuple[dict[str, str], list]]) -> list[EvaluationBatch]:
        """Evaluate (candidate, batch) pairs via the adapter's batch_evaluate or fallback."""
        return invoke_batch_evaluate(self.adapter, items, capture_traces=True)

    # ------------------------------------------------------------------
    # Main proposal method
    # ------------------------------------------------------------------

    def propose(self, state: GEPAState) -> list[CandidateProposal]:
        """Run the reflective mutation pipeline and return all evaluated proposals.

        The proposer generates and minibatch-evaluates candidates; acceptance and
        selection (which to keep) are the engine's job. With the default
        ``SingleMutationSampling`` this returns at most one proposal — identical to
        the original sequential behavior. A reflection that yields no text
        updates produces no proposal; when a ReAct attempt was exhausted or
        otherwise dropped (reported through ``length_capped_dropped`` for
        callback compatibility), the attempt is persisted and still reported
        through ``on_proposal_end`` so per-action acceptance rates stay honest.

        Args:
            state: The current optimization state (candidates, scores, RNG,
                iteration counter).

        Returns:
            The evaluated child proposals for this iteration; empty when no task
            was sampled or every reflection came back empty.

        """
        i = state.i + 1

        # Stage 1: Sample (parent, minibatch) tasks
        tasks = self.sampling_strategy.sample_tasks(state, self.candidate_selector, self.batch_sampler, self.trainset)
        if not tasks:
            return []

        # Fire callbacks for each sampled task
        for task in tasks:
            notify_callbacks(
                self.callbacks,
                "on_candidate_selected",
                CandidateSelectedEvent(
                    iteration=i,
                    candidate_idx=task.parent_idx,
                    candidate=task.parent_candidate,
                    score=state.program_full_scores_val_set[task.parent_idx],
                ),
            )
            notify_callbacks(
                self.callbacks,
                "on_minibatch_sampled",
                MinibatchSampledEvent(
                    iteration=i,
                    minibatch_ids=task.minibatch_ids,
                    trainset_size=len(self.trainset),
                ),
            )

        # Stage 2: Batch evaluate parents (deduplicated)
        unique_keys: dict[tuple[str, tuple], tuple[dict[str, str], list[Any]]] = {}
        task_to_key: list[tuple[str, tuple]] = []
        for task in tasks:
            key = (_candidate_hash(task.parent_candidate), tuple(task.minibatch_ids))
            unique_keys.setdefault(key, (task.parent_candidate, task.minibatch))
            task_to_key.append(key)

        key_list = list(unique_keys.keys())
        items = [unique_keys[k] for k in key_list]

        # Fire evaluation start callbacks for each task
        for task in tasks:
            notify_callbacks(
                self.callbacks,
                "on_evaluation_start",
                EvaluationStartEvent(
                    iteration=i,
                    candidate_idx=task.parent_idx,
                    batch_size=len(task.minibatch),
                    capture_traces=True,
                    parent_ids=[p for p in state.parent_program_for_candidate[task.parent_idx] if p is not None],
                    inputs=task.minibatch,
                    is_seed_candidate=task.parent_idx == 0,
                ),
            )

        parent_evals = self._batch_evaluate(items)
        key_to_eval: dict[tuple[str, tuple], EvaluationBatch] = dict(zip(key_list, parent_evals, strict=True))

        # Fire evaluation end callbacks for each task
        for task, key in zip(tasks, task_to_key, strict=True):
            eval_curr = key_to_eval[key]
            notify_callbacks(
                self.callbacks,
                "on_evaluation_end",
                EvaluationEndEvent(
                    iteration=i,
                    candidate_idx=task.parent_idx,
                    scores=eval_curr.scores,
                    has_trajectories=bool(eval_curr.trajectories),
                    parent_ids=[p for p in state.parent_program_for_candidate[task.parent_idx] if p is not None],
                    outputs=eval_curr.outputs,
                    trajectories=eval_curr.trajectories,
                    objective_scores=eval_curr.objective_scores,
                    is_seed_candidate=task.parent_idx == 0,
                ),
            )

        total_parent_evals = sum(
            e.num_metric_calls if e.num_metric_calls is not None else len(items[idx][1])
            for idx, e in enumerate(parent_evals)
        )
        state.increment_evals(total_parent_evals)

        # Update evaluation cache for parents
        if state.evaluation_cache is not None:
            for task, key in zip(tasks, task_to_key, strict=True):
                eval_curr = key_to_eval[key]
                objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
                state.evaluation_cache.put_batch(
                    task.parent_candidate,
                    task.minibatch_ids,
                    eval_curr.outputs,
                    eval_curr.scores,
                    objective_scores_list,
                    split=TRAINSET_CACHE_SPLIT,
                )

        # Trace: legacy first-task keys (pre-#329 tooling compatibility) plus
        # full per-task records — multi-task iterations record every task, not
        # just the first. Tasks that get skipped later simply never receive
        # score keys.
        first_task = tasks[0]
        state.full_program_trace[-1]["selected_program_candidate"] = first_task.parent_idx
        state.full_program_trace[-1]["subsample_ids"] = first_task.minibatch_ids
        state.full_program_trace[-1]["n_tasks"] = len(tasks)
        state.full_program_trace[-1]["tasks"] = [
            {"parent_idx": task.parent_idx, "subsample_ids": list(task.minibatch_ids)} for task in tasks
        ]
        self.logger.log(
            f"Iteration {i}: Selected program {first_task.parent_idx} "
            f"score: {state.program_full_scores_val_set[first_task.parent_idx]}"
        )

        self.experiment_tracker.log_metrics(
            {
                "iteration": i,
                "selected_program_candidate": first_task.parent_idx,
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        # On-disk anchor (``iterations/<iteration_id>/``) for this iteration's
        # proposals, stamped on the trace entry when the engine opened the
        # slot; fall back to the legacy sequence anchor for entries that
        # predate it. Every task in the batch shares the slot, and therefore
        # the anchor.
        trace_entry = state.full_program_trace[-1]
        iteration_id = trace_entry.get("iteration_id") or str(trace_entry.get("i", 0) + 1)

        # Stage 3a: Build reflective datasets + fire pre-reflection callbacks (per task).
        # ``prepared`` holds one slot per task (None = skipped); ``jobs`` is the
        # subset that will reflect, in order, so the reflection LM can batch them.
        prepared: list[tuple[ProposalTask, EvaluationBatch, list[str], Any] | None] = []
        for task, key in zip(tasks, task_to_key, strict=True):
            eval_curr = key_to_eval[key]

            if not eval_curr.trajectories:
                self.logger.log(f"Iteration {i}: No trajectories for parent {task.parent_idx}. Skipping.")
                notify_callbacks(
                    self.callbacks,
                    "on_evaluation_skipped",
                    EvaluationSkippedEvent(
                        iteration=i,
                        candidate_idx=task.parent_idx,
                        reason="no_trajectories",
                        scores=eval_curr.scores,
                        is_seed_candidate=task.parent_idx == 0,
                    ),
                )
                prepared.append(None)
                continue

            if (
                self.skip_perfect_score
                and self.perfect_score is not None
                and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
            ):
                self.logger.log(f"Iteration {i}: All subsample scores perfect for parent {task.parent_idx}. Skipping.")
                notify_callbacks(
                    self.callbacks,
                    "on_evaluation_skipped",
                    EvaluationSkippedEvent(
                        iteration=i,
                        candidate_idx=task.parent_idx,
                        reason="all_scores_perfect",
                        scores=eval_curr.scores,
                        is_seed_candidate=task.parent_idx == 0,
                    ),
                )
                prepared.append(None)
                continue

            predictor_names = self.module_selector(
                state, eval_curr.trajectories, eval_curr.scores, task.parent_idx, task.parent_candidate
            )

            try:
                reflective_dataset = self.adapter.make_reflective_dataset(
                    task.parent_candidate, eval_curr, predictor_names
                )
                reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                    k: [dict(item) for item in v] for k, v in reflective_dataset.items()
                }
                notify_callbacks(
                    self.callbacks,
                    "on_reflective_dataset_built",
                    ReflectiveDatasetBuiltEvent(
                        iteration=i,
                        iteration_id=iteration_id,
                        candidate_idx=task.parent_idx,
                        components=predictor_names,
                        dataset=reflective_dataset_concrete,
                    ),
                )
                notify_callbacks(
                    self.callbacks,
                    "on_proposal_start",
                    ProposalStartEvent(
                        iteration=i,
                        parent_candidate=task.parent_candidate,
                        components=predictor_names,
                        reflective_dataset=reflective_dataset_concrete,
                    ),
                )
            except Exception as e:
                self.logger.log(f"Iteration {i}: Exception building reflective dataset: {e}")
                self.logger.log(traceback.format_exc())
                prepared.append(None)
                continue

            prepared.append((task, eval_curr, predictor_names, reflective_dataset))

        # Stage 3b: Reflect across all prepared tasks — one batched LM call when the
        # reflection LM supports it (litellm.batch_completion), else per task.
        # Each job carries parent-specific context. In addition to on-disk
        # anchors, reflection strategies receive only the branch-local chat
        # history of the selected parent candidate. Sibling attempts are never
        # included.
        jobs = [(p[0].parent_candidate, p[3], p[2]) for p in prepared if p is not None]
        job_metadatas: list[Mapping[str, Any] | None] = [
            {
                "iteration_id": iteration_id,
                "parent_iteration_id": state.iteration_id_for_candidate_idx(p[0].parent_idx),
                "candidate_idx": p[0].parent_idx,
                "branch_edit_history": deepcopy(state.revision_history_by_candidate[p[0].parent_idx]),
            }
            for p in prepared
            if p is not None
        ]
        with response_journal_scope(f"optimizer-iteration-{state.i}"):
            reflected_batches = self._propose_texts_batch_safe(jobs, job_metadatas)
        batch_texts = iter(reflected_batches)
        batch_contexts = iter(job_metadatas)

        # Stage 3c: Build each child candidate from its proposed texts.
        children: list[tuple[ProposalTask, dict[str, str], EvaluationBatch, dict[str, Any]] | None] = []
        for p in prepared:
            if p is None:
                children.append(None)
                continue
            task, eval_curr, _predictor_names, _reflective_dataset = p
            reflection_context = next(batch_contexts)
            texts = next(batch_texts)
            if texts is None:
                children.append(None)
                continue
            new_texts, prompts, raw_outputs, reflection_metadata = texts

            if not new_texts:
                # Do not evaluate an unchanged child; retain metadata when an
                # attempted proposal produced no completed edit.
                dropped = (reflection_metadata or {}).get("length_capped_dropped")
                attempt_records = (reflection_metadata or {}).get("attempt_records")
                if dropped or attempt_records:
                    state.record_proposal_attempts(
                        task.parent_idx,
                        reflection_metadata,
                        outcome="dropped",
                        reason="Reflection attempt produced no completed text update.",
                    )
                    capped_metadata: dict[str, Any] = {"proposal_id": f"{i}-{len(children)}"}
                    for meta_key, meta_val in reflection_metadata.items():
                        if meta_key.startswith(("prompt:", "raw_lm_output:")):
                            capped_metadata[f"reflection_meta:{meta_key}"] = meta_val
                        else:
                            capped_metadata[meta_key] = meta_val
                    notify_callbacks(
                        self.callbacks,
                        "on_proposal_end",
                        ProposalEndEvent(
                            iteration=i,
                            new_instructions={},
                            prompts=prompts,
                            raw_lm_outputs=raw_outputs,
                            metadata=capped_metadata,
                        ),
                    )
                else:
                    self.logger.log(
                        f"Iteration {i}: Reflection returned no text updates; skipping proposal for this task."
                    )
                children.append(None)
                continue

            _lm_metadata: dict[str, Any] = {}
            # Stable per-proposal identifier (iteration-taskindex): downstream
            # consumers (run manifests, #346's per-proposal state anchors) can
            # key on this instead of positional inference.
            _lm_metadata["proposal_id"] = f"{i}-{len(children)}"
            branch_history = reflection_context.get("branch_edit_history", []) if reflection_context else []
            for comp in new_texts:
                _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
                _lm_metadata[f"raw_lm_output:{comp}"] = raw_outputs.get(comp, "")
            # Multi-call reflection diagnostics (e.g. ComBEE per-call
            # intermediates) flow into the proposal metadata for callbacks,
            # trackers, and the run manifest. Keys that would collide with the
            # reserved prompt:/raw_lm_output: namespaces are remapped so a
            # reflector cannot inject phantom components into proposal tables.
            for meta_key, meta_val in (reflection_metadata or {}).items():
                if meta_key == "parent_branch_history_lengths" or meta_key.startswith(("prompt:", "raw_lm_output:")):
                    _lm_metadata[f"reflection_meta:{meta_key}"] = meta_val
                else:
                    _lm_metadata[meta_key] = meta_val
            _lm_metadata["parent_branch_history_lengths"] = {str(task.parent_idx): len(branch_history)}

            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")

            notify_callbacks(
                self.callbacks,
                "on_proposal_end",
                ProposalEndEvent(
                    iteration=i,
                    new_instructions=new_texts,
                    prompts=prompts,
                    raw_lm_outputs=raw_outputs,
                    metadata=dict(_lm_metadata),
                ),
            )

            new_candidate = task.parent_candidate.copy()
            for name, text in new_texts.items():
                assert name in new_candidate, f"{name} missing in candidate"
                new_candidate[name] = text

            children.append((task, new_candidate, eval_curr, _lm_metadata))

        # Stage 4: Batch evaluate children
        valid_children = [(idx, c) for idx, c in enumerate(children) if c is not None]
        if not valid_children:
            return []

        child_items = [(c[1], c[0].minibatch) for _, c in valid_children]

        # Fire evaluation start callbacks for each child candidate (parity with
        # the pre-batch sequential path, which emitted these around the new
        # candidate's minibatch evaluation; candidate_idx is None because the
        # child is not in the candidate pool yet)
        for _, (task, _new_candidate, _eval_curr, _meta) in valid_children:
            notify_callbacks(
                self.callbacks,
                "on_evaluation_start",
                EvaluationStartEvent(
                    iteration=i,
                    candidate_idx=None,
                    batch_size=len(task.minibatch),
                    capture_traces=True,
                    parent_ids=[task.parent_idx],
                    inputs=task.minibatch,
                    is_seed_candidate=False,
                ),
            )

        child_evals = self._batch_evaluate(child_items)

        # Fire evaluation end callbacks for each child candidate
        for (_, (task, _new_candidate, _eval_curr, _meta)), child_eval in zip(valid_children, child_evals, strict=True):
            notify_callbacks(
                self.callbacks,
                "on_evaluation_end",
                EvaluationEndEvent(
                    iteration=i,
                    candidate_idx=None,
                    scores=child_eval.scores,
                    has_trajectories=bool(child_eval.trajectories),
                    parent_ids=[task.parent_idx],
                    outputs=child_eval.outputs,
                    trajectories=child_eval.trajectories,
                    objective_scores=child_eval.objective_scores,
                    is_seed_candidate=False,
                ),
            )

        total_child_evals = sum(
            e.num_metric_calls if e.num_metric_calls is not None else len(child_items[idx][1])
            for idx, e in enumerate(child_evals)
        )
        state.increment_evals(total_child_evals)

        # Update evaluation cache for children
        if state.evaluation_cache is not None:
            for (_, (task, new_candidate, _, _)), child_eval in zip(valid_children, child_evals, strict=True):
                new_obj_scores = list(child_eval.objective_scores) if child_eval.objective_scores else None
                state.evaluation_cache.put_batch(
                    new_candidate,
                    task.minibatch_ids,
                    child_eval.outputs,
                    child_eval.scores,
                    new_obj_scores,
                    split=TRAINSET_CACHE_SPLIT,
                )

        # Trace: per-task before/after scores (children is index-aligned with tasks)
        trace_tasks = state.full_program_trace[-1].get("tasks")
        if trace_tasks is not None:
            for (child_idx, (_task, _nc, eval_curr, _md)), child_eval in zip(valid_children, child_evals, strict=True):
                trace_tasks[child_idx]["subsample_scores"] = list(eval_curr.scores)
                trace_tasks[child_idx]["new_subsample_scores"] = list(child_eval.scores)

        # Log subsample scores for first task (trace compatibility)
        if valid_children:
            first_child_idx = valid_children[0][0]
            first_child = children[first_child_idx]
            if first_child is not None:
                state.full_program_trace[-1]["subsample_scores"] = key_to_eval[task_to_key[first_child_idx]].scores
                state.full_program_trace[-1]["new_subsample_scores"] = child_evals[0].scores

                subsample_before = sum(key_to_eval[task_to_key[first_child_idx]].scores)
                subsample_after = sum(child_evals[0].scores)
                self.experiment_tracker.log_metrics(
                    {
                        # pre-#329 key names, kept for existing dashboards
                        "subsample_score": subsample_before,
                        "new_subsample_score": subsample_after,
                        "subsample/before": subsample_before,
                        "subsample/after": subsample_after,
                        "total_metric_calls": state.total_num_evals,
                    },
                    step=i,
                )

        # Stage 5: Build proposals and filter
        proposals: list[CandidateProposal] = []
        for (_, (task, new_candidate, eval_curr, _lm_metadata)), child_eval in zip(
            valid_children, child_evals, strict=True
        ):
            proposal = CandidateProposal(
                candidate=new_candidate,
                parent_program_ids=[task.parent_idx],
                subsample_indices=task.minibatch_ids,
                subsample_scores_before=eval_curr.scores,
                subsample_scores_after=child_eval.scores,
                eval_before=SubsampleEvaluation(
                    scores=eval_curr.scores,
                    outputs=eval_curr.outputs,
                    objective_scores=list(eval_curr.objective_scores) if eval_curr.objective_scores else None,
                    trajectories=eval_curr.trajectories,
                ),
                eval_after=SubsampleEvaluation(
                    scores=child_eval.scores,
                    outputs=child_eval.outputs,
                    objective_scores=list(child_eval.objective_scores) if child_eval.objective_scores else None,
                    trajectories=child_eval.trajectories,
                ),
                tag="reflective_mutation",
                metadata=_lm_metadata,
            )
            proposals.append(proposal)

        return proposals
