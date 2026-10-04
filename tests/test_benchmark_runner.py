"""Exercise shared benchmark execution with a real optimizer and local task adapter."""

import json
from dataclasses import replace

import pytest

from examples.common import benchmark_runner as runner
from examples.common.benchmark_types import BenchmarkDefinition
from examples.common.experiment_models import DEFAULT_PROPOSER_MODEL, DEFAULT_SOLVER_MODEL
from examples.common.react_v2 import structured_prompt
from gepa.core.adapter import EvaluationBatch


class LocalAdapter:
    """Stand in for the external task environment while retaining real GEPA orchestration."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.contexts = []

    def set_evaluation_context(self, **context):
        self.contexts.append(context)

    def evaluate(self, batch, candidate, capture_traces=False):
        improved = "improved" in candidate["system_prompt"]
        ids = [row["id"] for row in batch]
        if "test" in ids and improved:
            assert (self.root / "vanilla" / "frozen-winner.json").exists()
        self.calls.append((ids, dict(candidate)))
        return EvaluationBatch(
            outputs=[{"elapsed_seconds": 10.0 + i * 10, "answer": "correct"} for i, _ in enumerate(batch)],
            scores=[float(improved)] * len(batch),
            trajectories=[{"feedback": "Use improved instructions"} for _ in batch] if capture_traces else None,
            objective_scores=[{"accuracy": float(improved)} for _ in batch],
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return dict.fromkeys(components_to_update, eval_batch.trajectories)


def definition(root):
    return BenchmarkDefinition(
        name="local",
        adapter=LocalAdapter(root),
        seed_candidate={"system_prompt": structured_prompt("Answer the task.", template_family="alibaba")},
        trainset=[{"id": "train1"}, {"id": "train2"}],
        valset=[{"id": "val"}],
        testset=[{"id": "test"}],
        source={"revision": "pinned"},
        runtime={"harness": "local-v1"},
        metric_name="accuracy",
    )


def test_real_optimizer_freezes_validation_winner_and_reuses_evidence(tmp_path, monkeypatch):
    benchmark = definition(tmp_path / "run")
    improved = structured_prompt("Use improved instructions.", template_family="alibaba")
    monkeypatch.setattr(runner, "LM", lambda *args, **kwargs: lambda messages: f"```\n{improved}\n```")
    argv = [
        "--run-dir",
        str(tmp_path / "run"),
        "--condition",
        "vanilla",
        "--max-metric-calls",
        "5",
        "--reflection-minibatch-size",
        "1",
    ]
    kwargs = {
        "benchmark_name": "local",
        "build_benchmark": lambda args, models: benchmark,
        "add_arguments": lambda parser: None,
        "argv": argv,
    }
    assert runner.run_cli(**kwargs) == 0
    summary = json.loads((tmp_path / "run" / "vanilla" / "summary.json").read_text())
    assert summary["winner"]["validation_score"] == 1.0
    assert summary["test"]["mean_score"] == 1.0
    assert summary["baseline"]["mean_score"] == 0.0
    assert sum(ids == ["test"] for ids, _ in benchmark.adapter.calls) == 2
    count = len(benchmark.adapter.calls)
    runner.run_cli(**kwargs)
    assert len(benchmark.adapter.calls) == count
    benchmark.testset[0]["changed"] = True
    with pytest.raises(ValueError, match="configuration or data changed"):
        runner.run_cli(**kwargs)


def test_shared_baseline_is_evaluated_once_across_run_directories(tmp_path):
    benchmark = definition(tmp_path)
    identity = {"model": "same", "data": "same"}
    first = runner._starting_baseline(benchmark, tmp_path / "budget1", identity, 0)
    second = runner._starting_baseline(benchmark, tmp_path / "budget2", identity, 0)
    assert first == second
    assert len(benchmark.adapter.calls) == 1


def test_repetitions_keep_episode_latency_separate_from_batch_throughput(tmp_path):
    benchmark = definition(tmp_path)
    kwargs = {
        "definition": benchmark,
        "candidate": benchmark.seed_candidate,
        "records": benchmark.trainset,
        "directory": tmp_path / "pilot",
        "identity": {"model": "fixed"},
        "split": "train",
        "repetitions": 2,
        "seed": 7,
    }
    summary = runner.evaluate_candidate(**kwargs)
    assert summary["timing"]["mean_seconds"] == 15.0
    assert summary["timing"]["median_seconds"] == 15.0
    assert summary["timing"]["p95_seconds"] == 20.0
    assert summary["timing"]["attempt_count"] == 4
    assert summary["timing"]["recorded_batch_seconds"] < 1.0
    assert benchmark.adapter.contexts == [
        {"split": "train", "repetition": 0, "seed": 7},
        {"split": "train", "repetition": 1, "seed": 8},
    ]
    path = tmp_path / "pilot" / "repetition-000.json"
    saved = json.loads(path.read_text())
    assert saved["payload"]["objective_scores"] == [{"accuracy": 0.0}, {"accuracy": 0.0}]
    saved["payload"]["scores"][0] = 1.0
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="evidence changed"):
        runner.evaluate_candidate(**kwargs)
    assert len(benchmark.adapter.calls) == 2


@pytest.mark.parametrize("bad_id", ["train1", "", "val"])
def test_split_overlap_and_missing_ids_fail_before_evaluation(tmp_path, bad_id):
    benchmark = definition(tmp_path)
    with pytest.raises(ValueError, match="IDs"):
        runner.validate_definition(replace(benchmark, testset=[{"id": bad_id}]))
    assert benchmark.adapter.calls == []


@pytest.mark.parametrize(
    "scores, outputs",
    [
        ([], [{"elapsed_seconds": 1.0}]),
        ([float("nan")], [{"elapsed_seconds": 1.0}]),
        ([0.0], [{"elapsed_seconds": -1.0}]),
        ([0.0], [{"elapsed_seconds": float("inf")}]),
        ([0.0], [{}]),
    ],
)
def test_incomplete_or_nonfinite_results_cannot_be_saved(scores, outputs):
    with pytest.raises(ValueError):
        runner.validate_evaluation(EvaluationBatch(outputs=outputs, scores=scores), 1)


def test_optimizer_cannot_access_test_records(tmp_path):
    benchmark = definition(tmp_path)
    observed = runner.RecordedAdapter(benchmark, tmp_path, 0)
    with pytest.raises(ValueError, match="outside"):
        observed.evaluate(benchmark.testset, benchmark.seed_candidate)
    assert benchmark.adapter.calls == []


def test_default_model_roles_and_endpoints_are_independent():
    parser = runner.build_parser("local", lambda parser: None)
    args = parser.parse_args(["--solver-api-base", "http://solver/v1", "--reflection-api-base", "http://proposer/v1"])
    models = runner.resolve_models(args)
    assert models.solver_model == DEFAULT_SOLVER_MODEL
    assert models.proposer_model == DEFAULT_PROPOSER_MODEL
    assert models.solver_kwargs["api_base"] == "http://solver/v1"
    assert models.proposer_kwargs["api_base"] == "http://proposer/v1"
    assert models.solver_kwargs["max_tokens"] == 65_536
    assert models.proposer_kwargs["max_tokens"] == 131_072
    assert models.solver_kwargs["extra_body"] is not models.proposer_kwargs["extra_body"]


def test_pilot_uses_only_training_data(tmp_path):
    benchmark = definition(tmp_path)
    assert (
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=lambda args, models: benchmark,
            add_arguments=lambda parser: None,
            argv=["--run-dir", str(tmp_path / "run"), "--mode", "pilot", "--pilot-size", "1"],
        )
        == 0
    )
    assert benchmark.adapter.calls[0][0] == ["train1"]
    assert len(benchmark.adapter.calls) == 1
