"""Verify the primary Terminal-Bench CLI runs the shared scientific lifecycle."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from examples.common.benchmark_runner import build_parser, resolve_models, validate_definition
from examples.common.experiment_models import DEFAULT_PROPOSER_MODEL, DEFAULT_SOLVER_MODEL
from examples.terminalbench import legacy_main
from examples.terminalbench import main as terminalbench
from examples.terminalbench.shared_adapter import SharedTerminusAdapter, trial_elapsed_seconds
from gepa.adapters.terminal_bench_adapter import HarborExecutionError
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (
    HarborEvaluation,
    HarborTrialResult,
)
from gepa.core.adapter import EvaluationBatch
from gepa.lm_constants import PROVIDER_RETRY_KEY


@pytest.fixture
def external_boundaries(monkeypatch, tmp_path):
    """Replace only serving discovery and Harbor's external process boundary."""
    observed = []
    runtime = Mock(return_value={"student": {"fixture": "solver"}, "proposer": {"fixture": "proposer"}})
    monkeypatch.setattr(terminalbench, "load_role_runtimes", runtime)
    requirements = Mock(return_value=("harbor", "docker"))
    monkeypatch.setattr(terminalbench.HarborCLI, "check_requirements", requirements)

    def run(harbor, task_ids, documents):
        evaluation_id = f"fixture-evaluation-{len(observed)}"
        observed.append(
            {"tasks": task_ids, "documents": deepcopy(documents), "kwargs": deepcopy(harbor.student_agent_kwargs)}
        )
        trials = {}
        for index, task_id in enumerate(task_ids):
            trials[task_id] = HarborTrialResult(
                task_id=task_id,
                reward=float(index % 2 == 0),
                rewards={"reward": float(index % 2 == 0)},
                errors=[],
                atif_trajectories=[{"fixture": True}],
                raw_result={
                    "started_at": "2026-10-03T01:00:00+00:00",
                    "finished_at": "2026-10-03T01:00:05+00:00" if index == 0 else "2026-10-03T01:00:17+00:00",
                },
                trial_dir=tmp_path / evaluation_id / task_id.rsplit("/", 1)[-1],
                verifier_logs={"test-stdout.txt": "Fixture verifier output"},
            )
        return HarborEvaluation(
            evaluation_id=evaluation_id,
            candidate_digest=harbor.manifest.candidate_digest(documents),
            config_path=tmp_path / "config.json",
            job_dir=tmp_path / evaluation_id,
            returncode=0,
            stdout_path=tmp_path / "stdout",
            stderr_path=tmp_path / "stderr",
            trials=trials,
        )

    monkeypatch.setattr(terminalbench.HarborCLI, "run", run)
    return observed, runtime, requirements


def parsed(tmp_path, *options):
    return build_parser("terminalbench", terminalbench.add_arguments).parse_args(
        [
            "--runtime-record",
            str(tmp_path / "solver.json"),
            "--proposer-runtime-record",
            str(tmp_path / "proposer.json"),
            "--run-dir",
            str(tmp_path / "run"),
            "--solver-api-base",
            "http://localhost:8000/v1",
            "--reflection-api-base",
            "http://localhost:8001/v1",
            *options,
        ]
    )


def definition(tmp_path, *options):
    args = parsed(tmp_path, *options)
    models = terminalbench.configure_models(args, resolve_models(args))
    return terminalbench.build_benchmark(args, models), args, models


def test_primary_entrypoint_is_a_direct_shared_route(monkeypatch):
    runner = Mock(return_value=0)
    monkeypatch.setattr(terminalbench, "run_cli", runner)
    monkeypatch.setattr(legacy_main, "main", Mock(side_effect=AssertionError("legacy dispatch")))
    assert terminalbench.main(["--mode", "pilot"]) == 0
    runner.assert_called_once_with(
        benchmark_name="terminalbench",
        build_benchmark=terminalbench.build_benchmark,
        add_arguments=terminalbench.add_arguments,
        configure_models=terminalbench.configure_models,
        argv=["--mode", "pilot"],
    )


def test_shared_model_profile_preserves_combined_budget_without_mutating_defaults(tmp_path):
    args = parsed(tmp_path)
    original = resolve_models(args)
    before = deepcopy(original)
    adjusted = terminalbench.configure_models(args, original)
    assert original == before
    assert (adjusted.solver_model, adjusted.proposer_model) == (DEFAULT_SOLVER_MODEL, DEFAULT_PROPOSER_MODEL)
    assert adjusted.solver_kwargs["max_tokens"] == adjusted.proposer_kwargs["max_tokens"] == 32768
    for kwargs in (adjusted.solver_kwargs, adjusted.proposer_kwargs):
        assert "thinking_token_budget" not in kwargs["extra_body"]
        assert kwargs["extra_body"]["chat_template_kwargs"]
        assert PROVIDER_RETRY_KEY in kwargs
        assert kwargs["timeout"] == 3600


@pytest.mark.parametrize("budget, cap", [("standard", 40), ("double", 80)])
def test_definition_uses_pinned_full_data_epochs_and_official_adapter(tmp_path, external_boundaries, budget, cap):
    benchmark, args, models = definition(tmp_path, "--budget", budget, "--val-limit", "1", "--test-limit", "2")
    validate_definition(benchmark)
    assert isinstance(benchmark.adapter, SharedTerminusAdapter)
    assert (len(benchmark.trainset), len(benchmark.valset), len(benchmark.testset)) == (30, 19, 40)
    assert benchmark.max_candidate_proposals == cap
    assert args.max_metric_calls is None
    assert benchmark.test_repetitions == 3
    assert benchmark.seed_candidate == legacy_main.seed_candidate(models.solver_model, "auto", "tb2.1")[0]
    assert benchmark.component_kinds == {"instruction_prompt": "user_prompt"}
    observed, runtime, requirements = external_boundaries
    assert not observed
    assert runtime.call_count == requirements.call_count == 1
    assert runtime.call_args.args[0].student_model == DEFAULT_SOLVER_MODEL
    assert runtime.call_args.args[0].proposer_model == DEFAULT_PROPOSER_MODEL
    assert benchmark.adapter.adapter.harbor.student_agent_kwargs["model_info"]["max_output_tokens"] == 32768


def test_full_identity_does_not_change_with_budget_or_training_prefix(tmp_path, external_boundaries):
    standard, _, _ = definition(tmp_path)
    expanded, _, _ = definition(tmp_path, "--budget", "double")
    limited, _, _ = definition(tmp_path, "--train-limit", "4")
    assert standard.source == expanded.source == limited.source
    assert standard.runtime == expanded.runtime == limited.runtime
    assert standard.trainset == limited.trainset
    assert limited.max_candidate_proposals == 8


def test_actual_edited_prompt_and_per_trial_latency_reach_shared_outputs(tmp_path, external_boundaries):
    benchmark, _, _ = definition(tmp_path)
    candidate = {"instruction_prompt": "Modified live instructions"}
    benchmark.adapter.set_evaluation_context(split="train", repetition=0, seed=9)
    result = benchmark.adapter.evaluate(benchmark.trainset[:2], candidate, capture_traces=True)
    assert [output["elapsed_seconds"] for output in result.outputs] == [5.0, 17.0]
    assert [output["id"] for output in result.outputs] == [record["id"] for record in benchmark.trainset[:2]]
    assert result.scores == [1.0, 0.0]
    assert external_boundaries[0][0]["documents"]["instruction_prompt"] == candidate["instruction_prompt"]
    assert external_boundaries[0][0]["kwargs"]["llm_kwargs"]["seed"] == 9
    feedback = benchmark.adapter.make_reflective_dataset(candidate, result, ["instruction_prompt"])
    assert feedback["instruction_prompt"][0]["Document"]["text"] == candidate["instruction_prompt"]
    with pytest.raises(ValueError, match="requested evaluation split"):
        benchmark.adapter.evaluate(benchmark.testset[:1], candidate)


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"started_at": "2026-10-03T01:00:00Z", "finished_at": None},
        {"started_at": "2026-10-03T01:00:00", "finished_at": "2026-10-03T01:00:01"},
        {"started_at": "2026-10-03T01:00:00Z", "finished_at": "2026-10-03T00:59:59Z"},
    ],
)
def test_missing_or_incomplete_official_task_timing_fails_closed(result):
    with pytest.raises(HarborExecutionError):
        trial_elapsed_seconds(result)


def test_adapter_rejects_ref_drift_and_incomplete_trial_evidence(tmp_path, external_boundaries, monkeypatch):
    benchmark, _, _ = definition(tmp_path)
    changed = {**benchmark.trainset[0], "task_ref": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="ref drift"):
        benchmark.adapter.evaluate([changed], benchmark.seed_candidate)
    monkeypatch.setattr(benchmark.adapter.adapter, "evaluate", lambda *_args, **_kwargs: EvaluationBatch([], [], []))
    with pytest.raises(HarborExecutionError, match="Incomplete"):
        benchmark.adapter.evaluate(benchmark.trainset[:1], benchmark.seed_candidate)


def test_heldout_repetitions_get_distinct_jobs_and_seeds(tmp_path, external_boundaries):
    benchmark, _, _ = definition(tmp_path)
    records = benchmark.testset[:2]
    evaluations = []
    for repetition in range(3):
        benchmark.adapter.set_evaluation_context(split="test", repetition=repetition, seed=10 + repetition)
        evaluations.append(benchmark.adapter.evaluate(records, benchmark.seed_candidate))
    summary = benchmark.adapter.summarize_evaluation(records, evaluations)
    assert summary["pass_at_1"] == 0.5
    assert summary["repetitions"] == 3
    assert summary["sample_standard_deviation"] == 0.0
    assert [item["kwargs"]["llm_kwargs"]["seed"] for item in external_boundaries[0]] == [10, 11, 12]
    with pytest.raises(ValueError, match="distinct official Harbor job"):
        benchmark.adapter.summarize_evaluation(records, [evaluations[0], evaluations[0]])


def test_primary_pilot_executes_real_shared_runner_and_writes_task_timings(tmp_path, external_boundaries, capsys):
    result = terminalbench.main(
        [
            "--runtime-record",
            str(tmp_path / "solver.json"),
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "pilot",
            "--pilot-size",
            "2",
        ]
    )
    assert result == 0
    summary = json.loads((tmp_path / "run" / "pilot" / "summary.json").read_text())
    assert summary["example_count"] == 2
    assert summary["timing"]["mean_seconds"] == 11.0
    assert summary["timing"]["p95_seconds"] == 17.0
    assert summary["metrics"]["pass_at_1"] == 0.5
    selected = external_boundaries[0][0]["tasks"]
    manifest = terminalbench.load_terminalbench_manifest(terminalbench.MANIFEST_PATH)
    assert selected == manifest.splits["train"][:2]
    capsys.readouterr()


def test_runtime_validation_is_required_before_any_harbor_trial(tmp_path, monkeypatch):
    fail = Mock(side_effect=ValueError("stale serving runtime"))
    monkeypatch.setattr(terminalbench, "load_role_runtimes", fail)
    launch = Mock()
    monkeypatch.setattr(terminalbench.HarborCLI, "run", launch)
    with pytest.raises(ValueError, match="stale serving runtime"):
        definition(tmp_path)
    launch.assert_not_called()
