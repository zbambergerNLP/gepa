"""Verify the primary Terminal-Bench CLI runs the shared scientific lifecycle."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from benchmark_model_fixtures import install_proposer
from terminalbench_staging_fixtures import make_offline_bundle

from examples.common.benchmark_runner import build_parser, resolve_models, validate_definition
from examples.common.experiment_models import DEFAULT_PROPOSER_MODEL, DEFAULT_SOLVER_MODEL
from examples.terminalbench import main as terminalbench
from examples.terminalbench.shared_adapter import SharedTerminusAdapter, trial_elapsed_seconds
from gepa.adapters.terminal_bench_adapter import HarborExecutionError
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (
    HarborEvaluation,
    HarborTrialResult,
)
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.lm_constants import PROVIDER_RETRY_KEY
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler


@pytest.fixture
def external_boundaries(monkeypatch, tmp_path):
    """Replace only serving discovery and Harbor's external process boundary."""
    observed = []
    runtime = Mock(
        side_effect=lambda _args, include_proposer: {
            "student": {"fixture": "solver"},
            **({"proposer": {"fixture": "proposer"}} if include_proposer else {}),
        }
    )
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
    assert terminalbench.main(["--mode", "pilot"]) == 0
    runner.assert_called_once_with(
        benchmark_name="terminalbench",
        build_benchmark=terminalbench.build_benchmark,
        add_arguments=terminalbench.add_arguments,
        configure_models=terminalbench.configure_models,
        argv=["--mode", "pilot"],
    )


@pytest.mark.parametrize("condition", ["vanilla", "random", "action", "react_v2_random", "react_v2"])
def test_optimizer_pilot_changes_actual_harbor_documents_using_training_only(
    tmp_path, external_boundaries, monkeypatch, condition
):
    """Run every optimizer arm through the maintained adapter and Harbor document boundary."""
    proposers = install_proposer(monkeypatch)
    run_dir = tmp_path / "run"
    assert (
        terminalbench.main(
            [
                "--runtime-record",
                str(tmp_path / "solver.json"),
                "--run-dir",
                str(run_dir),
                "--mode",
                "optimizer-pilot",
                "--condition",
                condition,
                "--pilot-size",
                "1",
                "--pilot-proposals",
                "1",
            ]
        )
        == 0
    )
    directory = run_dir / "optimizer-pilot" / condition
    summary = json.loads((directory / "summary.json").read_text())
    contract = json.loads((directory / "benchmark-run-contract.json").read_text())
    assert summary["winner"]["selection_split"] == "train"
    training_id = contract["optimization_data"]["train_ids"][0].removeprefix("terminalbench:")
    calls = external_boundaries[0]
    assert calls and all(call["tasks"] == [training_id] for call in calls)
    initial = calls[0]["documents"]["instruction_prompt"]
    assert any(call["documents"]["instruction_prompt"] != initial for call in calls)
    assert any(proposer.calls for proposer in proposers)
    assert "test" not in summary and "baseline" not in summary
    assert not list(run_dir.rglob("heldout"))


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
    assert benchmark.seed_candidate == benchmark.adapter.adapter.text_scope.seed_candidate()
    assert benchmark.component_kinds == {"instruction_prompt": "user_prompt"}
    observed, runtime, requirements = external_boundaries
    assert not observed
    assert runtime.call_count == requirements.call_count == 1
    assert runtime.call_args.args[0].student_model == DEFAULT_SOLVER_MODEL
    assert runtime.call_args.args[0].proposer_model == DEFAULT_PROPOSER_MODEL
    assert runtime.call_args.kwargs == {"include_proposer": True}
    assert benchmark.adapter.adapter.harbor.student_agent_kwargs["model_info"]["max_output_tokens"] == 32768


@pytest.mark.parametrize("mode", ["pilot", "baseline"])
def test_seed_only_modes_validate_only_solver_runtime(tmp_path, external_boundaries, mode):
    """Avoid allocating the unused proposer server for seed prompt evaluation."""
    definition(tmp_path, "--mode", mode)
    assert external_boundaries[1].call_args.kwargs == {"include_proposer": False}


def test_standalone_baseline_is_reused_and_optimizer_runtime_drift_is_rejected(
    tmp_path, external_boundaries, monkeypatch
):
    """Keep solver baseline identity independent of the unused optimizer server."""
    install_proposer(monkeypatch)
    base = [
        "--runtime-record",
        str(tmp_path / "solver.json"),
        "--run-dir",
        str(tmp_path / "run"),
        "--test-limit",
        "2",
        "--val-limit",
        "1",
        "--condition",
        "vanilla",
        "--max-metric-calls",
        "1",
    ]
    assert terminalbench.main([*base, "--mode", "baseline"]) == 0
    assert len(external_boundaries[0]) == 3
    assert terminalbench.main([*base, "--mode", "optimize"]) == 0
    # Three winner repetitions and one initial validation, with the baseline reused.
    assert len(external_boundaries[0]) == 7
    contract = json.loads((tmp_path / "run/vanilla/benchmark-run-contract.json").read_text())
    assert contract["proposer"]["runtime"] == {"fixture": "proposer"}
    assert contract["identity"]["runtime"]["execution_runtime"] == {"student": {"fixture": "solver"}}
    external_boundaries[1].side_effect = lambda _args, include_proposer: {
        "student": {"fixture": "solver"},
        "proposer": {"fixture": "changed optimizer runtime"},
    }
    with pytest.raises(ValueError, match="configuration or data changed"):
        terminalbench.main([*base, "--mode", "optimize"])
    assert len(external_boundaries[0]) == 7


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


def test_singularity_excludes_only_mailman_preserving_splits_models_and_resume_isolation(tmp_path, external_boundaries):
    """Prevent Docker and Apptainer results from sharing an incompatible pilot checkpoint."""
    docker, _, models = definition(tmp_path)
    singularity, _, other_models = definition(tmp_path, "--container-runtime", "singularity")
    assert singularity.source == {**docker.source, "task_exclusions": terminalbench.SINGULARITY_TASK_EXCLUSIONS}
    assert docker.trainset == singularity.trainset
    assert docker.valset == singularity.valset
    assert singularity.testset == [record for record in docker.testset if record["task_id"] != "terminal-bench/mailman"]
    assert (len(singularity.trainset), len(singularity.valset), len(singularity.testset)) == (30, 19, 39)
    assert len(docker.testset) == 40
    assert len(singularity.source["task_refs"]) == 89
    assert models == other_models
    assert singularity.runtime == {**docker.runtime, "container_runtime": "singularity"}
    argv = [
        "--mode",
        "pilot",
        "--pilot-size",
        "1",
        "--runtime-record",
        str(tmp_path / "solver.json"),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    assert terminalbench.main(argv) == 0
    calls = len(external_boundaries[0])
    with pytest.raises(ValueError, match="configuration or data changed"):
        terminalbench.main([*argv, "--container-runtime", "singularity"])
    assert len(external_boundaries[0]) == calls


@pytest.mark.parametrize("mode", ["pilot", "optimizer-pilot"])
def test_partial_offline_bundle_covers_only_requested_training_prefix(tmp_path, external_boundaries, monkeypatch, mode):
    manifest, path, _ = make_offline_bundle(tmp_path)
    monkeypatch.setattr(terminalbench, "load_terminalbench_manifest", lambda _path: manifest)
    _, args, _ = definition(
        tmp_path, "--mode", mode, "--pilot-size", "1", "--container-runtime", "singularity",
        "--offline-task-bundle", str(path),
    )
    assert args.pilot_size == 1
    with pytest.raises(ValueError, match="requested tasks"):
        definition(
            tmp_path, "--mode", mode, "--pilot-size", "3", "--container-runtime", "singularity",
            "--offline-task-bundle", str(path),
        )
    assert not external_boundaries[0]


def test_offline_runtime_change_rejects_pilot_resume_before_reusing_results(tmp_path, external_boundaries, monkeypatch):
    manifest, path, payload = make_offline_bundle(tmp_path)
    monkeypatch.setattr(terminalbench, "load_terminalbench_manifest", lambda _path: manifest)
    argv = [
        "--mode", "pilot", "--pilot-size", "1", "--runtime-record", str(tmp_path / "solver.json"),
        "--run-dir", str(tmp_path / "run"), "--container-runtime", "singularity", "--offline-task-bundle", str(path),
    ]
    assert terminalbench.main(argv) == 0
    calls = len(external_boundaries[0])
    assert terminalbench.main(argv) == 0
    assert len(external_boundaries[0]) == calls
    contract_path = tmp_path / "run" / "pilot" / "evaluation-contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["identity"]["runtime"]["offline_task_bundle"]["tasks"]
    task = next(iter(payload["tasks"].values()))
    recipe = path.parent / task["recipe"]["path"]
    recipe.write_text("Different runtime preparation")
    task["recipe"]["sha256"] = hashlib.sha256(recipe.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="configuration or data changed"):
        terminalbench.main(argv)
    assert len(external_boundaries[0]) == calls


def test_offline_campaign_requires_all_selected_splits_before_harbor_calls(tmp_path, external_boundaries, monkeypatch):
    manifest, path, _ = make_offline_bundle(tmp_path)
    monkeypatch.setattr(terminalbench, "load_terminalbench_manifest", lambda _path: manifest)
    with pytest.raises(ValueError, match="requested tasks"):
        definition(tmp_path, "--container-runtime", "singularity", "--offline-task-bundle", str(path))
    assert not external_boundaries[0]
    external_boundaries[2].assert_not_called()


@pytest.mark.parametrize("mode", ["optimize", "baseline"])
def test_singularity_campaign_preflight_omits_mailman(tmp_path, external_boundaries, mode):
    """Require infrastructure only for tasks included in the evaluation subset."""
    benchmark, _, _ = definition(tmp_path, "--mode", mode, "--container-runtime", "singularity")
    required = benchmark.testset if mode == "baseline" else [*benchmark.trainset, *benchmark.valset, *benchmark.testset]
    task_ids = [record["task_id"] for record in required]
    assert "terminal-bench/mailman" not in task_ids
    external_boundaries[2].assert_called_once_with(task_ids)


def test_singularity_baseline_scores_39_tasks_and_cannot_reuse_full_set(tmp_path, external_boundaries, monkeypatch):
    """Record the exclusion and keep subset denominators and baseline caches separate."""
    argv = [
        "--mode",
        "baseline",
        "--container-runtime",
        "singularity",
        "--runtime-record",
        str(tmp_path / "solver.json"),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    assert terminalbench.main(argv) == 0
    root = tmp_path / "benchmark-baselines" / "terminalbench"
    summary = json.loads(next(root.glob("*/summary.json")).read_text())
    contract = json.loads(next(root.glob("*/evaluation-contract.json")).read_text())
    assert summary["example_count"] == summary["metrics"]["tasks_per_repetition"] == 39
    assert summary["metrics"]["pass_at_1"] == pytest.approx(20 / 39)
    assert contract["identity"]["data"]["source"]["task_exclusions"] == terminalbench.SINGULARITY_TASK_EXCLUSIONS
    assert contract["identity"]["full_data"]["splits"]["test"]["count"] == 39
    assert "terminalbench:terminal-bench/mailman" not in contract["ids"]
    assert len(external_boundaries[0]) == 3
    assert all(len(call["tasks"]) == 39 for call in external_boundaries[0])
    assert terminalbench.main(argv) == 0
    assert len(external_boundaries[0]) == 3
    monkeypatch.setattr(terminalbench, "SINGULARITY_TASK_EXCLUSIONS", {})
    assert terminalbench.main(argv) == 0
    assert len(external_boundaries[0]) == 6
    assert len(list(root.glob("*/evaluation-contract.json"))) == 2
    assert all(len(call["tasks"]) == 40 for call in external_boundaries[0][3:])


@pytest.mark.parametrize("mode", ["pilot", "optimizer-pilot", "optimize", "baseline"])
def test_task_preflight_uses_only_the_exact_selected_mode_prefix(tmp_path, external_boundaries, mode):
    """Keep training pilots and explicit held-out prefixes independent of unselected Mailman."""
    benchmark, _, _ = definition(
        tmp_path,
        "--mode",
        mode,
        "--container-runtime",
        "singularity",
        "--pilot-size",
        "1",
        "--train-limit",
        "2",
        "--val-limit",
        "3",
        "--test-limit",
        "37",
    )
    expected = (
        benchmark.trainset[:1]
        if mode in {"pilot", "optimizer-pilot"}
        else benchmark.testset[:37]
        if mode == "baseline"
        else [*benchmark.trainset[:2], *benchmark.valset[:3], *benchmark.testset[:37]]
    )
    assert all(record["task_id"] != "terminal-bench/mailman" for record in expected)
    external_boundaries[2].assert_called_once_with([record["task_id"] for record in expected])


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


@pytest.mark.parametrize("train_size,minibatch_size", [(4, 3), (30, 4)])
def test_actual_epoch_padding_executes_every_training_occurrence(
    tmp_path, external_boundaries, train_size, minibatch_size
):
    benchmark, _, _ = definition(
        tmp_path, "--train-limit", str(train_size), "--reflection-minibatch-size", str(minibatch_size)
    )
    loader = ListDataLoader(benchmark.trainset[:train_size])
    sampler = IndependentEpochShuffledBatchSampler(minibatch_size=minibatch_size, seed=0)
    benchmark.adapter.set_evaluation_context(split="train", repetition=0, seed=0)
    expected, outputs, saw_padding = [], [], False
    for iteration in range((train_size + minibatch_size - 1) // minibatch_size):
        ids = sampler.next_minibatch_ids(loader, SimpleNamespace(i=iteration))
        batch = loader.fetch(ids)
        saw_padding |= len(set(ids)) != len(ids)
        expected.extend(record["task_id"] for record in batch)
        result = benchmark.adapter.evaluate(batch, benchmark.seed_candidate, capture_traces=True)
        outputs.extend(result.outputs)
        assert result.num_metric_calls == len(batch)
        assert [output["task_id"] for output in result.outputs] == [record["task_id"] for record in batch]
        assert [trace["task_id"] for trace in result.trajectories] == [record["task_id"] for record in batch]
        feedback = benchmark.adapter.make_reflective_dataset(benchmark.seed_candidate, result, ["instruction_prompt"])
        assert len(feedback["instruction_prompt"]) == len(batch)
    calls = external_boundaries[0]
    assert saw_padding
    assert len(expected) > train_size
    assert [task_id for call in calls for task_id in call["tasks"]] == expected
    assert all(len(set(call["tasks"])) == len(call["tasks"]) for call in calls)
    assert len({(output["task_id"], output["evaluation_id"]) for output in outputs}) == len(expected)
    assert len({output["trial_dir"] for output in outputs}) == len(expected)


@pytest.mark.parametrize("split", ["val", "test"])
def test_validation_and_test_duplicates_fail_before_harbor(tmp_path, external_boundaries, split):
    benchmark, _, _ = definition(tmp_path)
    record = (benchmark.valset if split == "val" else benchmark.testset)[0]
    with pytest.raises(ValueError, match="only for padded training"):
        benchmark.adapter.evaluate([record, record], benchmark.seed_candidate)
    assert not external_boundaries[0]
