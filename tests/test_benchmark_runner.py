"""Exercise shared benchmark execution with a real optimizer and local task adapter."""

import fcntl
import importlib
import json
from dataclasses import replace

import httpx2
import pytest
from benchmark_model_fixtures import install_proposer
from test_jev_controller import setup_controller as _setup_jev_controller

from examples.common import benchmark_runner as runner
from examples.common import react_v2
from examples.common.benchmark_types import BenchmarkDefinition
from examples.common.experiment_models import DEFAULT_PROPOSER_MODEL, DEFAULT_SOLVER_MODEL
from examples.common.react_v2 import structured_prompt
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.jev_constants import JEV_MODEL

VARIANT_CASES = [
    ("vanilla", []),
    ("random", []),
    ("action", []),
    ("react_v2_random", []),
    ("react_v2", []),
    ("forest", []),
    ("react_v2", ["--reflection-level", "0"]),
    ("react_v2", ["--reflection-level", "1"]),
    ("react_v2", ["--reflection-level", "1", "--editor-mode", "single_call"]),
    ("react_v2_random", ["--reflection-level", "1"]),
    ("react_v2", ["--proposal-policy", "independent", "--editor-mode", "react"]),
    ("react_v2", ["--proposal-policy", "independent", "--editor-mode", "single_call"]),
    ("react_v2", ["--reflection-level", "1", "--edit-tool-set", "minimal"]),
    ("react_v2", ["--proposal-policy", "independent", "--edit-tool-set", "minimal"]),
    ("react_v2", ["--module-selector", "controller"]),
    ("react_v2", ["--module-selector", "all"]),
    ("vanilla", ["--candidate-selection", "current_best", "--acceptance", "improvement_or_equal", "--merge"]),
    ("vanilla", ["--candidate-selection", "epsilon_greedy"]),
    ("vanilla", ["--candidate-selection", "top_k_pareto"]),
    (
        "vanilla",
        ["--sampling-strategy", "same_parent", "--proposal-count", "2", "--proposal-selection", "best_improvement"],
    ),
    ("vanilla", ["--sampling-strategy", "independent", "--proposal-count", "2"]),
    (
        "react_v2",
        [
            "--sampling-strategy",
            "pxn",
            "--proposal-count",
            "2",
            "--parent-count",
            "2",
            "--proposal-selection",
            "top_k",
            "--proposal-top-k",
            "2",
        ],
    ),
]


class LocalAdapter:
    """Stand in for the external task environment while retaining real GEPA orchestration."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.contexts = []
        self.include_objectives = True

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
            objective_scores=[{"accuracy": float(improved)} for _ in batch] if self.include_objectives else None,
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


@pytest.mark.parametrize("budget_kind", ["metric_calls", "candidate_proposals"])
def test_real_optimizer_freezes_validation_winner_and_reuses_evidence(tmp_path, monkeypatch, budget_kind):
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
    if budget_kind == "candidate_proposals":
        benchmark = replace(benchmark, max_candidate_proposals=1)
        start = argv.index("--max-metric-calls")
        del argv[start : start + 2]

    def add_arguments(parser):
        if budget_kind == "candidate_proposals":
            parser.set_defaults(max_metric_calls=None)

    kwargs = {
        "benchmark_name": "local",
        "build_benchmark": lambda args, models: benchmark,
        "add_arguments": add_arguments,
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


@pytest.mark.parametrize(
    "module_selector, kinds",
    [("all", {}), ("round_robin", {"system_prompt": "system_prompt", "second": "user_prompt"})],
)
def test_stateless_harness_combinations_fail_before_any_condition_calls_models(
    tmp_path, monkeypatch, module_selector, kinds
):
    """Reject incompatible later arms before an all-condition run spends model calls."""
    benchmark = definition(tmp_path)
    benchmark = replace(
        benchmark, seed_candidate={**benchmark.seed_candidate, "second": "Other instructions"}, component_kinds=kinds
    )
    proposers = install_proposer(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=lambda *_: benchmark,
            add_arguments=lambda _: None,
            argv=[
                "--mode",
                "optimizer-pilot",
                "--condition",
                "all",
                "--module-selector",
                module_selector,
                "--run-dir",
                str(tmp_path / "run"),
            ],
        )
    assert exc.value.code == 2
    assert benchmark.adapter.calls == [] and proposers == []


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


def test_benchmark_role_budgets_reach_builder_and_saved_contract(tmp_path):
    seen = []

    def configure(args, models):
        return replace(models, solver_kwargs={**models.solver_kwargs, "max_tokens": 32_768})

    def build(args, models):
        seen.append(models)
        return definition(tmp_path)

    runner.run_cli(
        benchmark_name="local",
        build_benchmark=build,
        add_arguments=lambda parser: None,
        configure_models=configure,
        argv=["--run-dir", str(tmp_path / "run"), "--mode", "pilot"],
    )
    contract = json.loads((tmp_path / "run" / "pilot" / "evaluation-contract.json").read_text())
    assert seen[0].solver_kwargs["max_tokens"] == 32_768
    assert contract["identity"]["solver"]["kwargs"]["max_tokens"] == 32_768


def test_duplicate_optimizer_writer_is_rejected_before_model_work(tmp_path):
    benchmark = definition(tmp_path)
    parser = runner.build_parser("local", lambda parser: None)
    args = parser.parse_args(["--run-dir", str(tmp_path)])
    with (tmp_path / ".vanilla.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            runner._run_condition(benchmark, runner.resolve_models(args), args, {}, "vanilla")
    assert benchmark.adapter.calls == []


@pytest.mark.parametrize("condition, flags", VARIANT_CASES)
def test_real_optimizer_pilots_cover_supported_variants_without_test_leakage(tmp_path, monkeypatch, condition, flags):
    """Exercise real proposal generation, evaluation and selection on training records only."""
    benchmark = definition(tmp_path / "run")
    benchmark.adapter.include_objectives = False
    models = install_proposer(monkeypatch)
    kwargs = {
        "benchmark_name": "local",
        "build_benchmark": lambda args, models: benchmark,
        "add_arguments": lambda parser: None,
        "argv": [
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "optimizer-pilot",
            "--pilot-size",
            "1",
            "--condition",
            condition,
            "--max-metric-calls",
            "1",
            *flags,
        ],
    }
    assert runner.run_cli(**kwargs) == 0
    directory = tmp_path / "run" / "optimizer-pilot" / ("react_v2" if condition == "forest" else condition)
    contract = json.loads((directory / runner.RUN_CONTRACT_FILENAME).read_text())
    winner = json.loads((directory / "pilot-winner.json").read_text())
    assert winner["selection_split"] == "train"
    assert "validation_score" not in winner
    assert contract["optimization_data"] == {"train_ids": ["train1"], "selection_ids": ["train1"]}
    assert contract["optimizer"]["max_optimizer_iterations"] == 1
    assert contract["optimizer"]["max_candidate_proposals"] == contract["optimizer"]["proposals_per_iteration"]
    assert contract["optimizer"]["max_metric_calls"] is None
    assert all(ids == ["train1"] for ids, _ in benchmark.adapter.calls)
    assert any(candidate != benchmark.seed_candidate for _, candidate in benchmark.adapter.calls)
    assert sum(len(model.calls) for model in models) > 0
    assert not (directory / "heldout").exists()
    assert not (tmp_path / "benchmark-baselines").exists()
    calls = len(benchmark.adapter.calls)
    runner.run_cli(**kwargs)
    assert len(benchmark.adapter.calls) == calls


def test_optimizer_pilot_forces_reflection_for_a_perfect_starting_seed(tmp_path, monkeypatch):
    """Do not declare proposer readiness merely because a small training prefix is solved."""
    benchmark = definition(tmp_path)
    benchmark = replace(benchmark, seed_candidate={"system_prompt": structured_prompt("An improved seed.", "alibaba")})
    models = install_proposer(monkeypatch)
    runner.run_cli(
        benchmark_name="local",
        build_benchmark=lambda args, models: benchmark,
        add_arguments=lambda parser: None,
        argv=[
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "optimizer-pilot",
            "--condition",
            "vanilla",
            "--pilot-size",
            "1",
        ],
    )
    assert sum(len(model.calls) for model in models) > 0
    assert len(benchmark.adapter.calls) == 3
    assert all(ids == ["train1"] for ids, _ in benchmark.adapter.calls)
    evidence = json.loads(
        (tmp_path / "run" / "optimizer-pilot" / "vanilla" / "optimizer-pilot-evidence.json").read_text()
    )
    assert evidence["completed_cycles"] == 1
    assert evidence["proposals"][0]["stage"] == "rejected"
    assert evidence["proposals"][0]["old_score"] == evidence["proposals"][0]["new_score"] == 1.0


@pytest.mark.parametrize("condition", ["vanilla", "react_v2"])
def test_optimizer_pilot_caught_reflective_dataset_failure_cannot_report_success(tmp_path, monkeypatch, condition):
    """Reject a real core run that swallows a dataset error before calling its proposer."""
    benchmark = definition(tmp_path)
    models = install_proposer(monkeypatch)

    def broken_dataset(*args):
        raise ValueError("The training feedback mapper is broken")

    monkeypatch.setattr(benchmark.adapter, "make_reflective_dataset", broken_dataset)
    kwargs = {
        "benchmark_name": "local",
        "build_benchmark": lambda args, models: benchmark,
        "add_arguments": lambda parser: None,
        "argv": ["--run-dir", str(tmp_path / "run"), "--mode", "optimizer-pilot", "--condition", condition],
    }
    for _ in range(2):
        with pytest.raises(RuntimeError, match="completed no proposal/evaluation cycle"):
            runner.run_cli(**kwargs)
    directory = tmp_path / "run" / "optimizer-pilot" / condition
    assert sum(len(model.calls) for model in models) == 0
    assert not (directory / "pilot-winner.json").exists()
    assert not (directory / "summary.json").exists()
    assert not (directory / "optimizer-pilot-evidence.json").exists()


def test_optimizer_pilot_evidence_survives_checkpoint_resume_and_rejects_artifact_drift(tmp_path, monkeypatch):
    """Recover after optimization completes but winner persistence is interrupted."""
    benchmark = definition(tmp_path)
    install_proposer(monkeypatch)
    atomic_json = runner.atomic_json
    interrupted = False

    def interrupt_winner(path, value):
        nonlocal interrupted
        if path.name == "pilot-winner.json" and not interrupted:
            interrupted = True
            raise OSError("Interrupted before saving the pilot winner")
        atomic_json(path, value)

    monkeypatch.setattr(runner, "atomic_json", interrupt_winner)
    kwargs = {
        "benchmark_name": "local",
        "build_benchmark": lambda args, models: benchmark,
        "add_arguments": lambda parser: None,
        "argv": ["--run-dir", str(tmp_path / "run"), "--mode", "optimizer-pilot", "--condition", "vanilla"],
    }
    with pytest.raises(OSError, match="Interrupted"):
        runner.run_cli(**kwargs)
    directory = tmp_path / "run" / "optimizer-pilot" / "vanilla"
    path = directory / "optimizer-pilot-evidence.json"
    evidence = json.loads(path.read_text())
    calls = len(benchmark.adapter.calls)
    assert not (directory / "pilot-winner.json").exists()
    assert runner.run_cli(**kwargs) == 0
    assert json.loads(path.read_text()) == evidence
    assert len(benchmark.adapter.calls) == calls
    winner = json.loads((directory / "pilot-winner.json").read_text())
    assert winner["pilot_evidence_sha256"] == runner.digest(evidence)
    evidence["completed_cycles"] = 0
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="completion evidence"):
        runner.run_cli(**kwargs)
    assert len(benchmark.adapter.calls) == calls


def test_optimizer_pilot_records_skipped_job_when_another_proposal_completes(tmp_path, monkeypatch):
    """Allow an exercised cycle while preserving evidence of a skipped sibling opportunity."""
    benchmark = definition(tmp_path)
    install_proposer(monkeypatch)
    make_dataset = benchmark.adapter.make_reflective_dataset
    attempts = 0

    def once_broken(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("First proposal's feedback could not be built")
        return make_dataset(*args)

    monkeypatch.setattr(benchmark.adapter, "make_reflective_dataset", once_broken)
    runner.run_cli(
        benchmark_name="local",
        build_benchmark=lambda args, models: benchmark,
        add_arguments=lambda parser: None,
        argv=[
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "optimizer-pilot",
            "--condition",
            "vanilla",
            "--sampling-strategy",
            "same_parent",
            "--proposal-count",
            "2",
        ],
    )
    evidence = json.loads(
        (tmp_path / "run" / "optimizer-pilot" / "vanilla" / "optimizer-pilot-evidence.json").read_text()
    )
    assert evidence["completed_cycles"] == 1
    assert [item["stage"] for item in evidence["proposals"]] == ["sampled", "accepted"]


def test_optimizer_pilot_ignores_real_merge_evaluation_without_losing_reflective_evidence(
    tmp_path, monkeypatch, caplog
):
    """Merge complementary evaluated siblings without attributing their merge to a reflection job."""
    benchmark = definition(tmp_path)
    benchmark = replace(
        benchmark,
        seed_candidate={**benchmark.seed_candidate, "answer": benchmark.seed_candidate["system_prompt"]},
        trainset=[{"id": f"train{i}", "component": "answer" if i % 2 else "system_prompt"} for i in range(6)],
    )
    evaluate = benchmark.adapter.evaluate

    def score_complementary_modules(batch, candidate, capture_traces=False):
        result = evaluate(batch, candidate, capture_traces=capture_traces)
        result.scores = [1.0 if "improved" in candidate[row["component"]] else 0.1 for row in batch]
        result.objective_scores = None
        return result

    monkeypatch.setattr(benchmark.adapter, "evaluate", score_complementary_modules)
    install_proposer(monkeypatch)
    runner.run_cli(
        benchmark_name="local",
        build_benchmark=lambda args, models: benchmark,
        add_arguments=lambda parser: None,
        argv=[
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "optimizer-pilot",
            "--condition",
            "vanilla",
            "--pilot-size",
            "6",
            "--reflection-minibatch-size",
            "6",
            "--pilot-proposals",
            "2",
            "--sampling-strategy",
            "same_parent",
            "--proposal-count",
            "2",
            "--merge",
        ],
    )
    directory = tmp_path / "run" / "optimizer-pilot" / "vanilla"
    candidates = json.loads((directory / "candidates.json").read_text())
    assert candidates["parents"][-1] == [1, 2]
    assert candidates["val_aggregate_scores"][-1] == 1.0
    evidence = json.loads((directory / "optimizer-pilot-evidence.json").read_text())
    assert evidence["completed_cycles"] == 2
    assert [item["stage"] for item in evidence["proposals"]] == ["accepted", "accepted"]
    assert not any("failed on on_evaluation_end" in record.message for record in caplog.records)


@pytest.mark.parametrize("parents, proposals", [(1, 1), (1, 2), (2, 2)])
def test_multi_proposal_search_preserves_pinned_opportunity_budget(tmp_path, monkeypatch, parents, proposals):
    """Spend exactly the same proposal opportunities across divisible sampling groups."""
    benchmark = replace(definition(tmp_path / "run"), max_candidate_proposals=4)
    evaluate = benchmark.adapter.evaluate

    def constant_score(*args, **kwargs):
        batch = evaluate(*args, **kwargs)
        batch.scores = [0.0] * len(batch.scores)
        batch.objective_scores = None
        return batch

    monkeypatch.setattr(benchmark.adapter, "evaluate", constant_score)
    models = install_proposer(monkeypatch)
    runner.run_cli(
        benchmark_name="local",
        build_benchmark=lambda args, models: benchmark,
        add_arguments=lambda parser: parser.set_defaults(max_metric_calls=None),
        argv=[
            "--run-dir",
            str(tmp_path / "run"),
            "--condition",
            "vanilla",
            "--sampling-strategy",
            "pxn",
            "--proposal-count",
            str(proposals),
            "--parent-count",
            str(parents),
            "--reflection-minibatch-size",
            "1",
        ],
    )
    assert sum(len(model.calls) for model in models) == 4
    contract = json.loads((tmp_path / "run" / "vanilla" / runner.RUN_CONTRACT_FILENAME).read_text())
    assert contract["optimizer"]["max_candidate_proposals"] == 4
    assert contract["optimizer"]["max_optimizer_iterations"] == 4 // (parents * proposals)
    assert contract["optimizer"]["proposals_per_iteration"] == parents * proposals


def test_nondivisible_opportunity_budget_is_rejected_before_model_work(tmp_path, monkeypatch):
    """Never round a batched condition above a pinned benchmark's opportunity budget."""
    benchmark = replace(definition(tmp_path), max_candidate_proposals=3)
    models = install_proposer(monkeypatch)
    with pytest.raises(SystemExit):
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=lambda args, models: benchmark,
            add_arguments=lambda parser: None,
            argv=[
                "--run-dir",
                str(tmp_path / "run"),
                "--condition",
                "vanilla",
                "--sampling-strategy",
                "same_parent",
                "--proposal-count",
                "2",
            ],
        )
    assert not benchmark.adapter.calls
    assert not models


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--reflection-level", "1"),
        ("--controller-selection", "uniform_random"),
        ("--module-selector", "all"),
        ("--proposal-policy", "independent"),
        ("--candidate-selection", "current_best"),
        ("--acceptance", "improvement_or_equal"),
        ("--sampling-strategy", "same_parent"),
        ("--proposal-selection", "best_improvement"),
        ("--pilot-size", "2"),
        ("--pilot-proposals", "2"),
    ],
)
def test_variant_and_pilot_setting_drift_is_rejected_before_more_model_work(tmp_path, monkeypatch, flag, value):
    """Bind every effective ablation and training-pilot budget to its resume contract."""
    benchmark = definition(tmp_path)
    install_proposer(monkeypatch)
    kwargs = {
        "benchmark_name": "local",
        "build_benchmark": lambda args, models: benchmark,
        "add_arguments": lambda parser: None,
        "argv": [
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "optimizer-pilot",
            "--condition",
            "react_v2",
            "--pilot-size",
            "1",
        ],
    }
    runner.run_cli(**kwargs)
    calls = len(benchmark.adapter.calls)
    kwargs["argv"].extend([flag, value])
    with pytest.raises(ValueError, match="configuration or data changed"):
        runner.run_cli(**kwargs)
    assert len(benchmark.adapter.calls) == calls


@pytest.mark.parametrize(
    "flags",
    [
        ["--condition", "vanilla", "--module-selector", "controller"],
        ["--condition", "react_v2_random", "--module-selector", "controller"],
        ["--condition", "react_v2", "--reflection-level", "1", "--controller-selection", "jev"],
        ["--condition", "react_v2", "--reflection-level", "0", "--controller-selection", "uniform_random"],
        ["--condition", "react_v2", "--edit-tool-set", "minimal"],
        ["--condition", "react_v2", "--editor-mode", "react"],
        ["--proposal-count", "2"],
        ["--parent-count", "2"],
        ["--proposal-top-k", "2"],
        ["--pilot-proposals", "0"],
        ["--react-max-iterations", "0"],
    ],
)
def test_invalid_variant_combinations_fail_before_building_benchmark(tmp_path, flags):
    """Reject unsupported settings before loading data or constructing model-backed adapters."""
    builds = []
    with pytest.raises(SystemExit):
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=lambda *args: builds.append(args),
            add_arguments=lambda parser: None,
            argv=["--run-dir", str(tmp_path), *flags],
        )
    assert not builds


@pytest.mark.parametrize(
    "benchmark_name", ["hotpotqa", "terminalbench", "obliqbench", "decisionbench", "appworld", "taubench"]
)
def test_all_six_entrypoints_dispatch_all_conditions_through_real_optimizer(tmp_path, monkeypatch, benchmark_name):
    """Ensure every primary entrypoint exposes the same variant-aware lifecycle."""
    if benchmark_name == "decisionbench":
        pytest.importorskip("decision_bench")
    monkeypatch.setenv("DSPY_CACHEDIR", str(tmp_path / "dspy-cache"))
    entrypoint = importlib.import_module(f"examples.{benchmark_name}.main")
    benchmark = replace(definition(tmp_path), name=benchmark_name)
    install_proposer(monkeypatch)
    monkeypatch.setattr(entrypoint, "build_benchmark", lambda args, models: benchmark)
    monkeypatch.setattr(entrypoint, "add_arguments", lambda parser: None)
    assert entrypoint.main(
        ["--run-dir", str(tmp_path), "--mode", "optimizer-pilot", "--condition", "all", "--pilot-size", "1"]
    ) in (0, None)
    assert {path.name for path in (tmp_path / "optimizer-pilot").iterdir() if path.is_dir()} == {
        "vanilla",
        "random",
        "action",
        "react_v2_random",
        "react_v2",
    }
    assert all(ids == ["train1"] for ids, _ in benchmark.adapter.calls)


@pytest.fixture
def jev_setup(tmp_path):
    """Use the real typed provider client with an offline HTTP transport."""
    yield from _setup_jev_controller.__wrapped__(tmp_path)


@pytest.mark.parametrize("module_selector", ["round_robin", "all", "controller"])
@pytest.mark.parametrize("policy", ["real_edit", "independent"])
def test_jev_backend_and_adaptive_components_reach_real_optimizer(
    tmp_path, monkeypatch, jev_setup, module_selector, policy
):
    """Exercise typed Jev selection, edit execution and joint-module wiring end to end."""
    controller, requests, replies = jev_setup
    benchmark = definition(tmp_path)
    benchmark = replace(
        benchmark, seed_candidate={**benchmark.seed_candidate, "answer": benchmark.seed_candidate["system_prompt"]}
    )
    install_proposer(monkeypatch)
    monkeypatch.setattr(react_v2, "JevController", lambda **kwargs: controller)

    def response(payload):
        choices = payload["questions"]["edit"]["criteria"]
        chosen = next(key for key in choices if "contextualize@Objective/INSERT_TEXT" in key)
        return httpx2.Response(
            200,
            json={
                "model": JEV_MODEL,
                "usage": {"input_tokens": 1000, "output_tokens": 100},
                "answers": {
                    "edit": {
                        "type": "choice",
                        "choice": chosen,
                        "confidence": 1.0,
                        "probabilities": {key: float(key == chosen) for key in choices},
                    }
                },
            },
        )

    replies.extend([response, response])
    runner.run_cli(
        benchmark_name="local",
        build_benchmark=lambda args, models: benchmark,
        add_arguments=lambda parser: None,
        argv=[
            "--run-dir",
            str(tmp_path / "run"),
            "--mode",
            "optimizer-pilot",
            "--condition",
            "react_v2",
            "--controller-selection",
            "jev",
            "--module-selector",
            module_selector,
            "--proposal-policy",
            policy,
        ],
    )
    assert len(requests) == (2 if module_selector == "all" else 1)
    if module_selector == "controller":
        criteria = requests[0]["questions"]["edit"]["criteria"]
        assert {choice["component"] for choice in criteria.values()} == {"system_prompt", "answer"}
        assert all(
            sum(candidate[key] != benchmark.seed_candidate[key] for key in candidate) <= 1
            for _, candidate in benchmark.adapter.calls
        )
    assert any(candidate != benchmark.seed_candidate for _, candidate in benchmark.adapter.calls)
    assert all(set(ids) <= {"train1", "train2"} for ids, _ in benchmark.adapter.calls)
    contract = json.loads(
        (tmp_path / "run" / "optimizer-pilot" / "react_v2" / runner.RUN_CONTRACT_FILENAME).read_text()
    )
    assert contract["optimizer"]["controller_selection"] == "jev"
    assert contract["optimizer"]["module_selector"] == module_selector
