# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the ReflectionLM protocol and StatelessReflectionLM (#329 Phase 1)."""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import LM, LMProviderError
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionLM,
    ReflectionProposal,
    StatelessReflectionLM,
)


class RecordingLM:
    """A fake reflection LM: records prompts, returns a fenced instruction."""

    def __init__(self, reply: str = "improved instruction"):
        self.reply = reply
        self.calls: list = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return f"Here is the update:\n```\n{self.reply}\n```"


class BatchRecordingLM(RecordingLM):
    """A fake LM that also exposes ``batch_complete`` (like ``gepa.lm.LM``)."""

    def __init__(self, reply: str = "improved instruction"):
        super().__init__(reply)
        self.batch_calls: list[list] = []

    def batch_complete(self, messages_list, max_workers: int = 10):
        self.batch_calls.append(messages_list)
        return [f"```\n{self.reply}\n```" for _ in messages_list]


class CollectingLogger:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _reflective_dataset(components):
    return {name: [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "bad"}] for name in components}


def test_stateless_reflection_lm_satisfies_protocol():
    lm = StatelessReflectionLM(RecordingLM())
    assert isinstance(lm, ReflectionLM)


def test_reflect_returns_self_as_next_lm():
    """Stateless: the next LM is the same object (no carried context)."""
    lm = StatelessReflectionLM(RecordingLM())
    candidate = {"system_prompt": "old"}
    proposal, next_lm = lm.reflect(candidate, _reflective_dataset(["system_prompt"]), ["system_prompt"])
    assert isinstance(proposal, ReflectionProposal)
    assert next_lm is lm


def test_reflect_proposes_new_text_per_component():
    fake = RecordingLM(reply="brand new prompt")
    lm = StatelessReflectionLM(fake)
    candidate = {"a": "old_a", "b": "old_b"}
    proposal, _ = lm.reflect(candidate, _reflective_dataset(["a", "b"]), ["a", "b"])
    assert proposal.new_texts == {"a": "brand new prompt", "b": "brand new prompt"}
    # Diagnostics populated per component.
    assert set(proposal.prompts.keys()) == {"a", "b"}
    assert set(proposal.raw_lm_outputs.keys()) == {"a", "b"}
    assert len(fake.calls) == 2


def test_reflect_skips_components_without_feedback():
    fake = RecordingLM()
    logger = CollectingLogger()
    lm = StatelessReflectionLM(fake, logger=logger)
    candidate = {"a": "old_a", "b": "old_b"}
    # Only 'a' has data; 'b' is requested but absent from the reflective dataset.
    reflective_dataset = _reflective_dataset(["a"])
    proposal, _ = lm.reflect(candidate, reflective_dataset, ["a", "b"])
    assert set(proposal.new_texts.keys()) == {"a"}
    assert len(fake.calls) == 1
    assert any("'b' is not in reflective dataset" in m for m in logger.messages)


def test_reflect_many_batches_into_one_call():
    """N tasks with a batch-capable LM → a single batch_complete covering all prompts."""
    fake = BatchRecordingLM(reply="new text")
    lm = StatelessReflectionLM(fake)
    jobs = [
        ({"a": "old0"}, _reflective_dataset(["a"]), ["a"]),
        ({"a": "old1"}, _reflective_dataset(["a"]), ["a"]),
        ({"a": "old2"}, _reflective_dataset(["a"]), ["a"]),
    ]
    results = lm.reflect_many(jobs)
    assert len(results) == len(jobs)
    assert all(proposal.new_texts == {"a": "new text"} for proposal, _ in results)
    assert all(next_lm is lm for _, next_lm in results)
    # One batched request covering all 3 jobs — not 3 separate completions.
    assert len(fake.batch_calls) == 1
    assert len(fake.batch_calls[0]) == 3
    assert fake.calls == []  # the per-call path was not used


def test_reflect_many_scatters_results_per_job():
    """Each job's components must come back in its own proposal slot, in order."""
    fake = BatchRecordingLM(reply="X")
    lm = StatelessReflectionLM(fake)
    jobs = [
        ({"a": "0a", "b": "0b"}, _reflective_dataset(["a", "b"]), ["a", "b"]),
        ({"a": "1a"}, _reflective_dataset(["a"]), ["a"]),
    ]
    results = lm.reflect_many(jobs)
    assert set(results[0][0].new_texts.keys()) == {"a", "b"}
    assert set(results[1][0].new_texts.keys()) == {"a"}
    # 3 (job,component) prompts total, one batched call.
    assert len(fake.batch_calls) == 1
    assert len(fake.batch_calls[0]) == 3


def test_reflect_many_sequential_fallback_without_batch_complete():
    """A plain callable LM (no batch_complete) still works, one call per prompt."""
    fake = RecordingLM(reply="Y")
    lm = StatelessReflectionLM(fake)
    jobs = [
        ({"a": "0a"}, _reflective_dataset(["a"]), ["a"]),
        ({"a": "1a"}, _reflective_dataset(["a"]), ["a"]),
    ]
    results = lm.reflect_many(jobs)
    assert [p.new_texts for p, _ in results] == [{"a": "Y"}, {"a": "Y"}]
    assert len(fake.calls) == 2


def test_per_component_template_dict_and_missing_warning():
    fake = RecordingLM()
    logger = CollectingLogger()
    # Template provided for 'a' only; 'b' should warn once and fall back to default.
    template_a = "Improve <curr_param> using this feedback: <side_info>"
    lm = StatelessReflectionLM(fake, reflection_prompt_template={"a": template_a}, logger=logger)
    candidate = {"a": "old_a", "b": "old_b"}
    reflective_dataset = _reflective_dataset(["a", "b"])
    lm.reflect(candidate, reflective_dataset, ["a", "b"])
    # Warn again on a second call — but only once per component overall.
    lm.reflect(candidate, reflective_dataset, ["a", "b"])
    missing_warnings = [m for m in logger.messages if "No reflection_prompt_template found for parameter 'b'" in m]
    assert len(missing_warnings) == 1


# ---------------------------------------------------------------------------
# reflection_strategy injection seam (#329 Phase 2 entry point)
# ---------------------------------------------------------------------------


class _ReflectOnlyLM:
    """A ReflectionLM implementation that provides only reflect() — no reflect_many."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def reflect(self, candidate, reflective_dataset, components_to_update):
        self.calls.append(list(components_to_update))
        proposal = ReflectionProposal(new_texts={c: f"new_{c}" for c in components_to_update})
        return proposal, self


def _make_proposer(**overrides):
    from unittest.mock import MagicMock

    from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

    adapter = MagicMock()
    adapter.propose_new_texts = None
    kwargs = {
        "logger": MagicMock(),
        "trainset": [{"q": 1}, {"q": 2}],
        "adapter": adapter,
        "candidate_selector": MagicMock(),
        "module_selector": MagicMock(),
        "batch_sampler": MagicMock(),
        "perfect_score": None,
        "skip_perfect_score": False,
        "experiment_tracker": MagicMock(),
        "reflection_lm": None,
        "custom_candidate_proposer": None,
    }
    kwargs.update(overrides)
    return ReflectiveMutationProposer(**kwargs)


def test_injected_reflection_strategy_takes_precedence_over_stateless_default():
    stub = _ReflectOnlyLM()
    proposer = _make_proposer(reflection_strategy=stub, reflection_lm=RecordingLM())
    assert proposer._reflection_lm is stub


def test_without_injection_stateless_default_is_built():
    proposer = _make_proposer(reflection_lm=RecordingLM())
    assert isinstance(proposer._reflection_lm, StatelessReflectionLM)


def test_reflect_only_strategy_used_per_task_in_batch_path():
    """A ReflectionLM without reflect_many must be called once per job by
    _propose_texts_batch (the seam a ComBEE-style reflector plugs into)."""
    stub = _ReflectOnlyLM()
    proposer = _make_proposer(reflection_strategy=stub)
    jobs = [
        ({"c": "old1"}, {"c": [{"feedback": "f1"}]}, ["c"]),
        ({"c": "old2"}, {"c": [{"feedback": "f2"}]}, ["c"]),
    ]
    results = proposer._propose_texts_batch(jobs)
    assert stub.calls == [["c"], ["c"]]
    assert [r[0] for r in results] == [{"c": "new_c"}, {"c": "new_c"}]


def test_reflect_only_batch_fallback_forwards_aligned_metadata() -> None:
    """Keep each branch history aligned when only reflect() is available."""

    @dataclass
    class _MetadataReflectOnlyLM:
        metadatas: list = field(default_factory=list)

        def reflect(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
            """Record branch metadata and return a marked candidate.

            Args:
                candidate: Parent component mapping.
                reflective_dataset: Unused reflective rows.
                components_to_update: Unused selected components.
                metadata: Parent-specific branch context.

            Returns:
                One marked proposal and this test reflector.
            """
            self.metadatas.append(metadata)
            return ReflectionProposal(new_texts={"c": candidate["c"] + "!"}), self

    strategy = _MetadataReflectOnlyLM()
    proposer = _make_proposer(reflection_strategy=strategy)
    jobs = [
        ({"c": "left"}, {"c": [{"feedback": "f1"}]}, ["c"]),
        ({"c": "right"}, {"c": [{"feedback": "f2"}]}, ["c"]),
    ]
    metadatas = [
        {"branch_edit_history": [{"role": "user", "content": "left-only"}]},
        {"branch_edit_history": [{"role": "user", "content": "right-only"}]},
    ]
    proposer._propose_texts_batch(jobs, metadatas)
    assert strategy.metadatas == metadatas


def test_batch_retry_reuses_the_random_intervention_sequence() -> None:
    """Keep random semantic choices fixed when batch transport falls back."""

    @dataclass
    class _FailingBatchRandomLM:
        rng: random.Random = field(default_factory=lambda: random.Random(7))
        failed_choices: list[int] = field(default_factory=list)
        retried_choices: list[int] = field(default_factory=list)

        def reflect_many(self, jobs):
            """Sample every batch choice, then simulate a transport failure.

            Args:
                jobs: Reflection jobs whose interventions are sampled.

            Raises:
                RuntimeError: Always, after recording the sampled choices.
            """
            self.failed_choices = [self.rng.randrange(10_000) for _job in jobs]
            raise RuntimeError("batch transport failed")

        def reflect(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
            """Record the fallback choice and return a valid proposal.

            Args:
                candidate: Parent component mapping.
                reflective_dataset: Unused feedback rows.
                components_to_update: Selected components.
                metadata: Unused parent metadata.

            Returns:
                A proposal encoding the retried random choice and this object.
            """
            choice = self.rng.randrange(10_000)
            self.retried_choices.append(choice)
            proposal = ReflectionProposal(new_texts={components_to_update[0]: str(choice)})
            return proposal, self

    strategy = _FailingBatchRandomLM()
    proposer = _make_proposer(reflection_strategy=strategy)
    jobs = [
        ({"c": "left"}, {"c": [{"feedback": "f1"}]}, ["c"]),
        ({"c": "right"}, {"c": [{"feedback": "f2"}]}, ["c"]),
    ]

    results = proposer._propose_texts_batch_safe(jobs, [{}, {}])

    assert all(result is not None for result in results)
    assert strategy.retried_choices == strategy.failed_choices


def test_action_conditioned_stateless_post_selection_failure_does_not_reselect() -> None:
    """Abort when a whole-operation retry would change selected actions."""
    strategy = StatelessReflectionLM(RecordingLM(), action_selector=MagicMock())
    strategy.reflect_many = MagicMock(side_effect=RuntimeError("post-selection failure"))
    strategy.reflect = MagicMock()
    proposer = _make_proposer(reflection_strategy=strategy)
    jobs = [
        ({"c": "left"}, {"c": [{"feedback": "f1"}]}, ["c"]),
        ({"c": "right"}, {"c": [{"feedback": "f2"}]}, ["c"]),
    ]

    with pytest.raises(RuntimeError, match="post-selection failure"):
        proposer._propose_texts_batch_safe(jobs, [{}, {}])

    strategy.reflect.assert_not_called()


@pytest.mark.parametrize("job_count", [1, 2])
def test_batch_safe_propagates_exhausted_provider_failures(job_count: int) -> None:
    """Abort instead of turning a provider outage into missing proposals.

    Args:
        job_count: Number of reflection jobs sent through plain or batch paths.
    """
    strategy = StatelessReflectionLM(LM("test/reflection"))
    proposer = _make_proposer(reflection_strategy=strategy)
    jobs = [
        ({"c": f"parent-{index}"}, {"c": [{"feedback": f"f{index}"}]}, ["c"])
        for index in range(job_count)
    ]

    with (
        patch("litellm.completion", side_effect=RuntimeError("provider unavailable")),
        patch("litellm.batch_completion", side_effect=RuntimeError("provider unavailable")),
        pytest.raises(LMProviderError, match="Completion provider failed"),
    ):
        proposer._propose_texts_batch_safe(jobs)


def test_provider_failure_aborts_then_restart_replays_one_exact_multi_role_trajectory(tmp_path: Path) -> None:
    """Abort on provider failure, then resume the exact multi-role trajectory.

    Args:
        tmp_path: Temporary response-journal directory supplied by pytest.
    """
    from gepa.lm import LM
    from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
    from gepa.response_journal import response_journal_scope

    class JournaledThreeRoleHarness(ThreeRoleReflectionLM):
        """Exercise production retry state with a minimal two-role trajectory."""

        def __init__(self, base_lm: LM, manifestor_lm: LM):
            """Configure journaled role clients.

            Args:
                base_lm: Journaled Controller/proposer model.
                manifestor_lm: Journaled Manifestor model.
            """
            super().__init__(base_lm, level=2, manifestor_lm=manifestor_lm)

        def _run_job(self, candidate, components_to_update, metadata):
            """Run the two journaled role calls for one synthetic job.

            Args:
                candidate: Parent component mapping.
                components_to_update: Single selected component.
                metadata: Parent-specific branch history.

            Returns:
                Proposal containing both role outputs and supplied history.
            """
            controller = self.base_lm("controller request")
            manifestation = self.manifestor_lm("manifestor request")
            component = components_to_update[0]
            return ReflectionProposal(
                new_texts={component: f"{controller}|{manifestation}"},
                metadata={
                    "controller": controller,
                    "manifestor": manifestation,
                    "branch_edit_history": deepcopy((metadata or {}).get("branch_edit_history", [])),
                },
            )

        def reflect_many(self, jobs, *, metadatas=None):
            """Run all jobs through the batch path.

            Args:
                jobs: Synthetic reflection jobs.
                metadatas: Optional aligned branch histories.

            Returns:
                Proposal/strategy pairs in job order.
            """
            metadata_rows = metadatas if metadatas is not None else [None] * len(jobs)
            results = []
            for (candidate, _dataset, components), metadata in zip(jobs, metadata_rows, strict=True):
                proposal = self._run_job(candidate, components, metadata)
                results.append((proposal, self))
            return results

        def reflect(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
            """Run one fallback job with the same logical role sequence.

            Args:
                candidate: Parent component mapping.
                reflective_dataset: Unused synthetic feedback.
                components_to_update: Single selected component.
                metadata: Parent-specific branch history.

            Returns:
                Proposal and this strategy.
            """
            del reflective_dataset
            proposal = self._run_job(candidate, components_to_update, metadata)
            return proposal, self

    journal_path = tmp_path / "responses.sqlite3"

    def make_strategy() -> JournaledThreeRoleHarness:
        """Build fresh role clients sharing the condition journal.

        Returns:
            Fresh retry-aware strategy.
        """
        base = LM(
            "test/controller",
            response_journal_path=str(journal_path),
            response_journal_namespace="controller-proposer",
        )
        manifestor = LM(
            "test/manifestor",
            response_journal_path=str(journal_path),
            response_journal_namespace="manifestor",
        )
        return JournaledThreeRoleHarness(base, manifestor)

    def response(content: str, model: str) -> SimpleNamespace:
        """Build one minimal LiteLLM-shaped response.

        Args:
            content: Assistant text.
            model: Provider-returned model identity.

        Returns:
            Mock completion response.
        """
        message = SimpleNamespace(content=content, reasoning_content="", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            model=model,
            system_fingerprint="fp-fixed",
            usage=None,
        )

    controller_outputs = iter(["controller-first", "controller-second"])
    manifestor_outputs = iter(["manifestor-first", "manifestor-second"])
    manifestor_attempts = 0

    def provider_call(**kwargs):
        """Fail after the first committed Controller response, then recover.

        Args:
            **kwargs: Effective LiteLLM request.

        Returns:
            Scripted role response.

        Raises:
            RuntimeError: The first Manifestor call simulates interruption.
        """
        nonlocal manifestor_attempts
        if kwargs["model"] == "test/controller":
            return response(next(controller_outputs), "controller-runtime")
        manifestor_attempts += 1
        if manifestor_attempts == 1:
            raise RuntimeError("manifestor transport failed")
        return response(next(manifestor_outputs), "manifestor-runtime")

    jobs = [
        ({"c": "left"}, {"c": [{"feedback": "f1"}]}, ["c"]),
        ({"c": "right"}, {"c": [{"feedback": "f2"}]}, ["c"]),
    ]
    metadatas = [
        {"branch_edit_history": [{"role": "user", "content": "left"}]},
        {"branch_edit_history": [{"role": "assistant", "content": "right"}]},
    ]
    initial = _make_proposer(reflection_strategy=make_strategy())
    with (
        patch("litellm.completion", side_effect=provider_call) as provider,
        patch("litellm.completion_cost", return_value=0.0),
        response_journal_scope("optimizer-iteration-11"),
    ):
        with pytest.raises(LMProviderError, match="Completion provider failed"):
            initial._propose_texts_batch_safe(jobs, metadatas)

    assert provider.call_count == 2
    assert manifestor_attempts == 1

    resumed = _make_proposer(reflection_strategy=make_strategy())
    with (
        patch("litellm.completion", side_effect=provider_call) as resumed_provider,
        patch("litellm.completion_cost", return_value=0.0),
        response_journal_scope("optimizer-iteration-11"),
    ):
        completed = resumed._propose_texts_batch_safe(jobs, metadatas)

    assert all(result is not None for result in completed)
    assert resumed_provider.call_count == 3
    assert manifestor_attempts == 3

    replayed_strategy = _make_proposer(reflection_strategy=make_strategy())
    with (
        patch("litellm.completion") as replay_provider,
        response_journal_scope("optimizer-iteration-11"),
    ):
        replayed = replayed_strategy._propose_texts_batch_safe(jobs, metadatas)

    assert replayed == completed
    replay_provider.assert_not_called()


def test_batch_safe_propagates_response_journal_mismatch(tmp_path: Path) -> None:
    """Abort rather than convert a journal request mismatch into a dropped task.

    Args:
        tmp_path: Temporary response-journal directory supplied by pytest.
    """
    import pytest

    from gepa.lm import LM
    from gepa.response_journal import ResponseJournalError, response_journal_scope

    journal_path = tmp_path / "responses.sqlite3"
    seed_lm = LM(
        "test/reflection",
        response_journal_path=str(journal_path),
        response_journal_namespace="reflection-proposer",
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="recorded", reasoning_content="", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        model="reflection-runtime",
        system_fingerprint="fp-fixed",
        usage=None,
    )
    with (
        patch("litellm.completion", return_value=response),
        patch("litellm.completion_cost", return_value=0.0),
        response_journal_scope("optimizer-iteration-13"),
    ):
        seed_lm("different request")

    resumed_lm = LM(
        "test/reflection",
        response_journal_path=str(journal_path),
        response_journal_namespace="reflection-proposer",
    )
    proposer = _make_proposer(reflection_strategy=StatelessReflectionLM(resumed_lm))
    jobs = [
        ({"c": "left"}, _reflective_dataset(["c"]), ["c"]),
        ({"c": "right"}, _reflective_dataset(["c"]), ["c"]),
    ]
    with patch("litellm.batch_completion") as provider:
        with pytest.raises(ResponseJournalError, match="request mismatch"):
            with response_journal_scope("optimizer-iteration-13"):
                proposer._propose_texts_batch_safe(jobs)
    provider.assert_not_called()


def test_batch_safe_propagates_cached_provider_identity_drift(tmp_path: Path) -> None:
    """Abort rather than drop tasks when a cached provider identity changes.

    Args:
        tmp_path: Temporary response-journal directory supplied by pytest.
    """
    import pytest

    from gepa.lm import LM, ProviderIdentityMismatchError
    from gepa.response_journal import response_journal_scope

    journal_path = tmp_path / "responses.sqlite3"

    def make_lm(fingerprint: str) -> LM:
        """Build one journaled LM pinned to a provider fingerprint.

        Args:
            fingerprint: Expected provider system fingerprint.

        Returns:
            Configured reflection LM.
        """
        return LM(
            "test/reflection",
            expected_response_model="reflection-runtime",
            expected_system_fingerprint=fingerprint,
            response_journal_path=str(journal_path),
            response_journal_namespace="reflection-proposer",
        )

    def response(text: str) -> SimpleNamespace:
        """Build a parseable provider response with the original identity.

        Args:
            text: Proposed instruction body.

        Returns:
            LiteLLM-shaped completion response.
        """
        message = SimpleNamespace(content=f"```\n{text}\n```", reasoning_content="", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            model="reflection-runtime",
            system_fingerprint="fp-original",
            usage=None,
        )

    jobs = [
        ({"c": "left"}, _reflective_dataset(["c"]), ["c"]),
        ({"c": "right"}, _reflective_dataset(["c"]), ["c"]),
    ]
    first = _make_proposer(reflection_strategy=StatelessReflectionLM(make_lm("fp-original")))
    with (
        patch("litellm.batch_completion", return_value=[response("first"), response("second")]),
        patch("litellm.completion_cost", return_value=0.0),
        response_journal_scope("optimizer-iteration-14"),
    ):
        assert len(first._propose_texts_batch(jobs)) == 2

    resumed = _make_proposer(reflection_strategy=StatelessReflectionLM(make_lm("fp-changed")))
    with patch("litellm.batch_completion") as provider:
        with pytest.raises(ProviderIdentityMismatchError, match="Cached provider response identity"):
            with response_journal_scope("optimizer-iteration-14"):
                resumed._propose_texts_batch_safe(jobs)
    provider.assert_not_called()


def test_reflection_strategy_receives_public_prompt_template():
    """Strategies that opt in to template binding must see the public API
    value, instead of silently falling back to their own generic default."""

    class _TemplateBindingLM(_ReflectOnlyLM):
        def __init__(self):
            super().__init__()
            self.template = None

        def bind_reflection_prompt_template(self, template):
            self.template = template

    template = "PUBLIC TEMPLATE <curr_param> :: <side_info>"
    strategy = _TemplateBindingLM()
    _make_proposer(reflection_strategy=strategy, reflection_prompt_template=template)

    assert strategy.template == template


def test_single_job_batch_path_uses_reflect_not_reflect_many():
    """A one-job iteration has no PxN parallelism to exploit. Calling
    reflect_many([job]) can change a batch-capable strategy's LM transport."""

    class _BatchCapableLM(_ReflectOnlyLM):
        def __init__(self):
            super().__init__()
            self.reflect_many_calls = 0

        def reflect_many(self, jobs):
            self.reflect_many_calls += 1
            raise AssertionError("single-job path must not call reflect_many")

    strategy = _BatchCapableLM()
    proposer = _make_proposer(reflection_strategy=strategy)
    result = proposer._propose_texts_batch([({"c": "old"}, {"c": [{"feedback": "f"}]}, ["c"])])

    assert result[0][0] == {"c": "new_c"}
    assert strategy.calls == [["c"]]
    assert strategy.reflect_many_calls == 0


# ---------------------------------------------------------------------------
# H1: empty reflection output must not produce a phantom child
# ---------------------------------------------------------------------------


class _FixedProposalLM:
    """ReflectionLM stub returning a fixed new_texts dict."""

    def __init__(self, new_texts):
        self._new_texts = new_texts

    def reflect(self, candidate, reflective_dataset, components_to_update):
        return ReflectionProposal(new_texts=dict(self._new_texts)), self


def _make_propose_harness(reflection_strategy):
    from unittest.mock import MagicMock

    from gepa.core.adapter import EvaluationBatch

    parent_eval = EvaluationBatch(
        outputs=["o"], scores=[0.4], trajectories=[{"step": 1}], objective_scores=None, num_metric_calls=1
    )
    adapter = MagicMock()
    adapter.propose_new_texts = None
    adapter.batch_evaluate = MagicMock(return_value=[parent_eval])
    adapter.make_reflective_dataset = MagicMock(return_value={"c": [{"feedback": "f"}]})

    candidate_selector = MagicMock()
    candidate_selector.select_candidate_idx.return_value = 0
    batch_sampler = MagicMock()
    batch_sampler.next_minibatch_ids.return_value = [0]
    module_selector = MagicMock(return_value=["c"])

    proposer = _make_proposer(
        adapter=adapter,
        candidate_selector=candidate_selector,
        batch_sampler=batch_sampler,
        module_selector=module_selector,
        trainset=[{"q": 0}],
        reflection_strategy=reflection_strategy,
    )
    return proposer, adapter


def _make_state():
    from gepa.core.state import GEPAState, ValsetEvaluation

    base_eval = ValsetEvaluation(outputs_by_val_id={0: "o"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None)
    state = GEPAState({"c": "seed"}, base_eval, track_best_outputs=False)
    # Set by the engine's seed initialization in real runs (state.py:704).
    state.total_num_evals = 1
    # The engine appends a per-iteration trace entry before calling propose().
    state.full_program_trace.append({"i": 0})
    return state


def test_empty_new_texts_skips_child_without_evaluating_it():
    proposer, adapter = _make_propose_harness(_FixedProposalLM({}))
    proposals = proposer.propose(_make_state())
    assert proposals == []
    # Only the parent-stage batch evaluation ran; no metric calls were burned
    # evaluating a child byte-identical to its parent.
    assert adapter.batch_evaluate.call_count == 1


def test_nonempty_new_texts_still_produces_a_proposal():
    proposer, adapter = _make_propose_harness(_FixedProposalLM({"c": "improved"}))
    proposals = proposer.propose(_make_state())
    assert len(proposals) == 1
    assert proposals[0].candidate["c"] == "improved"
    assert adapter.batch_evaluate.call_count == 2  # parent stage + child stage


def test_length_capped_empty_proposal_still_fires_on_proposal_end() -> None:
    """Test that a proposal emptied by the length cap still reaches on_proposal_end (#7).

    No child is evaluated (empty new_texts), but the attempt is counted so an
    action's acceptance rate is not silently inflated by vanished attempts.
    """
    events: list[dict] = []
    recorder = MagicMock()
    recorder.on_proposal_end.side_effect = events.append

    class _CappedLM(_FixedProposalLM):
        """ReflectionLM stub whose proposal is emptied by the length cap."""

        def reflect(self, candidate, reflective_dataset, components_to_update):
            """Return a deliberately empty, length-capped proposal.

            Args:
                candidate: Unused parent candidate.
                reflective_dataset: Unused reflective rows.
                components_to_update: Unused selected components.

            Returns:
                Empty proposal with action and dropped-attempt metadata.
            """
            return (
                ReflectionProposal(
                    new_texts={},
                    metadata={
                        "action": "contextualize",
                        "length_capped_dropped": ["c"],
                        "attempt_records": [
                            {
                                "component": "c",
                                "assistant": "attempted edit",
                                "action": "INSERT_TEXT",
                                "observation": "length cap exceeded",
                                "error": "no completed edit",
                            }
                        ],
                    },
                ),
                self,
            )

    proposer, adapter = _make_propose_harness(_CappedLM({}))
    proposer.callbacks = [recorder]

    state = _make_state()
    proposals = proposer.propose(state)

    assert proposals == []  # nothing to evaluate
    assert adapter.batch_evaluate.call_count == 1  # parent stage only
    assert len(events) == 1
    assert events[0]["new_instructions"] == {}
    assert events[0]["metadata"]["action"] == "contextualize"
    assert events[0]["metadata"]["length_capped_dropped"] == ["c"]
    history = deepcopy(state.revision_history_by_candidate[0])
    assert history[0] == {"role": "assistant", "content": "attempted edit"}
    assert history[1] == {"role": "user", "content": "length cap exceeded"}
    assert history[2]["role"] == "user"
    assert "Optimizer result: dropped" in history[2]["content"]


# ---------------------------------------------------------------------------
# H2: optimize_anything exposes reflection_strategy via ReflectionConfig
# ---------------------------------------------------------------------------


def test_reflection_config_accepts_reflection_strategy():
    from gepa.gepa_launcher import ReflectionConfig

    stub = _ReflectOnlyLM()
    assert ReflectionConfig(reflection_strategy=stub).reflection_strategy is stub
    assert ReflectionConfig().reflection_strategy is None


# ---------------------------------------------------------------------------
# M2: TrackingLM must not hide batch_complete from callables that provide it
# ---------------------------------------------------------------------------


def test_tracking_lm_exposes_batch_complete_conditionally():
    from gepa.lm import TrackingLM

    plain = TrackingLM(lambda p: "x")
    assert not hasattr(plain, "batch_complete")

    class WithBatch:
        def __call__(self, prompt):
            return "x"

        def batch_complete(self, messages_list):
            return ["y"] * len(messages_list)

    tracked = TrackingLM(WithBatch())
    assert hasattr(tracked, "batch_complete")
    out = tracked.batch_complete([[{"role": "user", "content": "hi"}]] * 3)
    assert out == ["y", "y", "y"]
    assert tracked.total_tokens_out > 0


# ---------------------------------------------------------------------------
# reflection_strategy conflict guard + metadata threading + per-task trace
# ---------------------------------------------------------------------------


def test_reflection_strategy_conflicts_with_custom_proposer():
    import pytest

    with pytest.raises(ValueError, match="custom_candidate_proposer"):
        _make_proposer(
            reflection_strategy=_ReflectOnlyLM(),
            custom_candidate_proposer=lambda c, d, comps: {},
        )


def test_reflection_strategy_conflicts_with_adapter_proposer():
    from unittest.mock import MagicMock

    import pytest

    adapter = MagicMock()
    adapter.propose_new_texts = lambda c, d, comps: {}
    with pytest.raises(ValueError, match=r"adapter\.propose_new_texts"):
        _make_proposer(reflection_strategy=_ReflectOnlyLM(), adapter=adapter)


class _MetadataLM:
    """ReflectionLM stub recording multi-call diagnostics in metadata."""

    def reflect(self, candidate, reflective_dataset, components_to_update):
        proposal = ReflectionProposal(
            new_texts={c: f"new_{c}" for c in components_to_update},
            metadata={"combee:level1_outputs": ["draft-1", "draft-2"]},
        )
        return proposal, self


def test_reflection_metadata_reaches_candidate_proposal():
    proposer, _adapter = _make_propose_harness(_MetadataLM())
    proposals = proposer.propose(_make_state())
    assert len(proposals) == 1
    assert proposals[0].metadata["combee:level1_outputs"] == ["draft-1", "draft-2"]
    # Single-call diagnostics still present alongside.
    assert "prompt:c" in proposals[0].metadata


def test_trace_records_every_task_not_just_the_first():
    from unittest.mock import MagicMock

    from gepa.core.adapter import EvaluationBatch
    from gepa.strategies.proposal_sampling import ProposalTask

    class TwoTaskSampling:
        def sample_tasks(self, state, candidate_selector, batch_sampler, trainset):
            return [
                ProposalTask(
                    parent_idx=0,
                    parent_candidate=dict(state.program_candidates[0]),
                    minibatch_ids=[0],
                    minibatch=[{"q": 0}],
                ),
                ProposalTask(
                    parent_idx=0,
                    parent_candidate=dict(state.program_candidates[0]),
                    minibatch_ids=[1],
                    minibatch=[{"q": 1}],
                ),
            ]

    adapter = MagicMock()
    adapter.propose_new_texts = None
    adapter.batch_evaluate = MagicMock(
        side_effect=lambda items, **kwargs: [
            EvaluationBatch(
                outputs=["o"], scores=[0.4], trajectories=[{"step": 1}], objective_scores=None, num_metric_calls=1
            )
            for _ in items
        ]
    )
    adapter.make_reflective_dataset = MagicMock(return_value={"c": [{"feedback": "f"}]})
    module_selector = MagicMock(return_value=["c"])

    proposer = _make_proposer(
        adapter=adapter,
        candidate_selector=MagicMock(),
        batch_sampler=MagicMock(),
        module_selector=module_selector,
        trainset=[{"q": 0}, {"q": 1}],
        reflection_strategy=_FixedProposalLM({"c": "improved"}),
        sampling_strategy=TwoTaskSampling(),
    )
    state = _make_state()
    proposals = proposer.propose(state)
    assert all(call.kwargs == {"capture_traces": True} for call in adapter.batch_evaluate.call_args_list)
    assert len(proposals) == 2

    entry = state.full_program_trace[-1]
    assert entry["n_tasks"] == 2
    assert len(entry["tasks"]) == 2
    assert [t["subsample_ids"] for t in entry["tasks"]] == [[0], [1]]
    for t in entry["tasks"]:
        assert t["subsample_scores"] == [0.4]
        assert t["new_subsample_scores"] == [0.4]
    # Legacy first-task keys unchanged for older tooling.
    assert entry["selected_program_candidate"] == 0
    assert entry["subsample_ids"] == [0]


# ---------------------------------------------------------------------------
# Adversarial-review fixes: next_lm chaining, reflect_many validation,
# api entry point, tracker keys
# ---------------------------------------------------------------------------


class _ChainingLM:
    """Stateful ReflectionLM: each reflect() returns a NEW successor."""

    def __init__(self, generation=0):
        self.generation = generation

    def reflect(self, candidate, reflective_dataset, components_to_update):
        proposal = ReflectionProposal(new_texts={c: f"gen{self.generation}_{c}" for c in components_to_update})
        return proposal, _ChainingLM(self.generation + 1)


def test_stateful_reflection_lm_successor_is_chained():
    proposer = _make_proposer(reflection_strategy=_ChainingLM())
    assert proposer._reflection_lm.generation == 0
    proposer.propose_new_texts({"c": "x"}, {"c": [{"f": 1}]}, ["c"])
    assert proposer._reflection_lm.generation == 1, "next_lm was discarded"
    proposer.propose_new_texts({"c": "x"}, {"c": [{"f": 1}]}, ["c"])
    assert proposer._reflection_lm.generation == 2


class _WrongLengthLM:
    def reflect(self, candidate, reflective_dataset, components_to_update):
        return ReflectionProposal(new_texts={}), self

    def reflect_many(self, jobs):
        return []  # wrong length


def test_reflect_many_wrong_length_raises():
    import pytest

    proposer = _make_proposer(reflection_strategy=_WrongLengthLM())
    jobs = [
        ({"c": "x"}, {"c": [{"f": 1}]}, ["c"]),
        ({"c": "y"}, {"c": [{"f": 2}]}, ["c"]),
    ]
    with pytest.raises(ValueError, match="reflect_many returned 0 results for 2 jobs"):
        proposer._propose_texts_batch(jobs)


def test_reflection_metadata_reserved_prefixes_are_remapped():
    class _CollidingMetaLM:
        def reflect(self, candidate, reflective_dataset, components_to_update):
            return ReflectionProposal(
                new_texts={c: f"new_{c}" for c in components_to_update},
                metadata={"prompt:phantom": "spoof", "safe_key": 1},
            ), self

    proposer, _ = _make_propose_harness(_CollidingMetaLM())
    proposals = proposer.propose(_make_state())
    md = proposals[0].metadata
    assert "prompt:phantom" not in md
    assert md["reflection_meta:prompt:phantom"] == "spoof"
    assert md["safe_key"] == 1


def test_legacy_tracker_metric_keys_still_emitted():
    from unittest.mock import MagicMock

    tracker = MagicMock()
    proposer, _ = _make_propose_harness(_FixedProposalLM({"c": "improved"}))
    proposer.experiment_tracker = tracker
    proposer.propose(_make_state())
    logged_keys = set()
    for call in tracker.log_metrics.call_args_list:
        logged_keys |= set(call.args[0].keys())
    assert {"subsample_score", "new_subsample_score"} <= logged_keys, logged_keys


def test_gepa_optimize_accepts_reflection_strategy_without_reflection_lm():
    """Verify optimize accepts a strategy without a separate reflection LM."""
    import gepa
    from gepa.core.adapter import EvaluationBatch

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            return EvaluationBatch(
                outputs=["o"] * len(batch),
                scores=[0.5] * len(batch),
                trajectories=[{"f": 1}] * len(batch) if capture_traces else None,
                objective_scores=None,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components):
            return {c: [{"Feedback": "f"}] for c in components}

    class ValidatingStrategy(_ReflectOnlyLM):
        def __init__(self):
            """Initialize the strategy validation counter."""
            super().__init__()
            self.validated = []

        def validate_candidate(self, candidate):
            """Record the seed validated by the strategy.

            Args:
                candidate: Candidate passed by the optimizer front door.
            """
            self.validated.append(candidate)

    stub = ValidatingStrategy()
    result = gepa.optimize(
        seed_candidate={"c": "x"},
        trainset=[{"q": 1}] * 4,
        valset=[{"q": 1}] * 4,
        adapter=Adapter(),
        reflection_strategy=stub,
        max_metric_calls=25,
        display_progress_bar=False,
        seed=0,
    )
    assert stub.calls, "reflection_strategy never invoked via gepa.optimize"
    assert stub.validated == [{"c": "x"}]
    assert result is not None


def test_max_reflection_cost_with_costless_strategy_raises():
    import pytest

    import gepa

    with pytest.raises(ValueError, match="total_cost"):
        gepa.optimize(
            seed_candidate={"c": "x"},
            trainset=[{"q": 1}],
            adapter=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(propose_new_texts=None),
            reflection_strategy=_ReflectOnlyLM(),
            max_reflection_cost=5.0,
            max_metric_calls=10,
        )
