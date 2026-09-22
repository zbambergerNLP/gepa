"""Verify shared HotPotQA baselines with real contracts and offline task execution."""

import fcntl
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_wikipedia_react_v2_config import _hotpot_args

from examples.common.experiment_models import EXPERIMENT_MODELS, QWEN3_8_27B_MODEL
from examples.common.react_v2 import benchmark_data_identity
from examples.hotpotqa import main as hotpot
from examples.hotpotqa.baseline import BASELINE_CONTRACT_FILENAME, baseline_directory, build_baseline_contract
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState, ValsetEvaluation

DATASET = [
    {"id": "first", "question": "first question", "answer": "Alpha"},
    {"id": "second", "question": "second question", "answer": "Beta"},
]


def run_contract(model=QWEN3_8_27B_MODEL, condition="vanilla", budget=6871) -> dict:
    """Use the real contract builder while supplying offline data and retriever provenance."""
    return hotpot.build_run_contract(
        condition,
        _hotpot_args(
            solver_model=model,
            reflection_model=model,
            max_metric_calls=budget,
            tag=f"{condition}-{budget}",
            max_workers=1,
            data_identity=benchmark_data_identity(
                source={"revision": "fixed"}, trainset=[], valset=[], testset=DATASET
            ),
            retrieval_provenance={"corpus_sha256": "fixed-corpus", "index_sha256": "fixed-index"},
        ),
    )


def evaluate(run_dir: Path, contract: dict, dataset=DATASET) -> dict:
    """Exercise the baseline evaluator with the contract's task request settings."""
    return hotpot.evaluate_starting_baseline(
        run_dir,
        contract,
        dataset,
        SimpleNamespace(),
        "http://localhost:8000/v1",
        hotpot.resolve_hotpotqa_lm_kwargs(contract["models"]["solver"], None),
    )


@pytest.fixture(autouse=True)
def offline_task_lm(monkeypatch):
    """Prevent model construction from depending on a live serving endpoint."""
    monkeypatch.setattr(hotpot, "build_hotpotqa_task_lm", Mock(return_value=object()))


def test_all_seven_ablations_share_one_baseline_per_model(tmp_path, monkeypatch):
    """Reuse one starting-prompt evaluation across methods and budgets, separately for each model."""
    task = Mock(return_value=("query", "Alpha", {}))
    monkeypatch.setattr(hotpot, "run_program", task)
    records = []
    for index, model in enumerate(EXPERIMENT_MODELS, start=1):
        reference = None
        for budget, conditions in hotpot._SCIENTIFIC_CONDITIONS_BY_BUDGET.items():
            for condition in conditions:
                contract = run_contract(model, condition, budget)
                row = evaluate(tmp_path / f"{index}-{budget}-{condition}", contract)
                assert row["test_exact_match"] == row["test_f1"] == 0.5
                if reference is None:
                    reference = row
                assert row == reference
                assert task.call_count == len(DATASET) * index
        records.append(reference)
    assert records[0]["contract_sha256"] != records[1]["contract_sha256"]
    assert hotpot.build_hotpotqa_task_lm.call_count == len(EXPERIMENT_MODELS)
    assert len(list((tmp_path / "hotpotqa-baselines").iterdir())) == len(EXPERIMENT_MODELS)


def test_reviewed_random_addition_reuses_baseline_without_extra_calls(tmp_path, monkeypatch):
    """Share only the reviewed source change; runtime drift must still fail."""
    task = Mock(return_value=("query", "Alpha", {}))
    monkeypatch.setattr(hotpot, "run_program", task)
    monkeypatch.setenv("HOTPOTQA_CAMPAIGN_ID", "campaign")
    monkeypatch.setenv("HOTPOTQA_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("HOTPOTQA_SOURCE_MANIFEST_SHA256", "1" * 64)
    base = run_contract()
    expected = evaluate(tmp_path / "vanilla", base)
    review = {
        "schema_version": 1,
        "campaign_id": "campaign",
        "review_sha256": "3" * 64,
        "base_source_commit": "a" * 40,
        "base_source_manifest_sha256": "1" * 64,
        "source_commit": "b" * 40,
        "source_manifest_sha256": "2" * 64,
    }
    monkeypatch.setenv("HOTPOTQA_SOURCE_COMPATIBILITY_JSON", json.dumps(review))
    monkeypatch.setenv("HOTPOTQA_SOURCE_COMMIT", "b" * 40)
    monkeypatch.setenv("HOTPOTQA_SOURCE_MANIFEST_SHA256", "2" * 64)
    added = run_contract(condition="random")
    assert added["execution_runtime"]["source_commit"] == "b" * 40
    assert build_baseline_contract(added) == build_baseline_contract(base)
    assert evaluate(tmp_path / "random", added) == expected
    assert task.call_count == len(DATASET)
    for key in ("serving_env_sha256", "model_revision", "source_manifest_sha256"):
        changed = deepcopy(added)
        changed["execution_runtime"][key] = "different"
        with pytest.raises((ValueError, FileNotFoundError)):
            evaluate(tmp_path / "random", changed)
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path / "missing-archive" / "random", added)
    assert task.call_count == len(DATASET)


def test_interrupted_baseline_resumes_only_missing_questions(tmp_path, monkeypatch):
    """Keep completed predictions after a provider failure, without publishing a partial baseline."""
    contract = run_contract()
    directory = baseline_directory(tmp_path / "run", build_baseline_contract(contract))
    task = Mock(side_effect=[("query", "Alpha", {}), RuntimeError("provider unavailable")])
    monkeypatch.setattr(hotpot, "run_program", task)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        evaluate(tmp_path / "run", contract)
    saved = list(directory.glob("heldout/*/0*.json"))
    assert len(saved) == 1
    original = saved[0].read_bytes()
    assert not list(directory.glob("heldout/*/summary.json"))
    task = Mock(return_value=("query", "Beta", {}))
    monkeypatch.setattr(hotpot, "run_program", task)
    row = evaluate(tmp_path / "another-ablation", contract)
    assert task.call_count == 1
    assert row["test_exact_match"] == row["test_f1"] == 1.0
    assert saved[0].read_bytes() == original
    assert evaluate(tmp_path / "run", contract) == row
    assert task.call_count == 1


@pytest.mark.parametrize("damage", ["question", "answer", "order", "missing"])
def test_baseline_rejects_changed_test_examples_before_model_work(tmp_path, monkeypatch, damage):
    """Check content and ordering rather than trusting IDs or split counts alone."""
    task = Mock()
    monkeypatch.setattr(hotpot, "run_program", task)
    dataset = deepcopy(DATASET)
    if damage in ("question", "answer"):
        dataset[0][damage] = "changed"
    elif damage == "order":
        dataset.reverse()
    else:
        dataset.pop()
    with pytest.raises(ValueError, match="exact ordered HotPotQA test examples"):
        evaluate(tmp_path / "run", run_contract(), dataset)
    task.assert_not_called()
    assert not (tmp_path / "hotpotqa-baselines").exists()


@pytest.mark.parametrize("axis", ["candidate", "decoding", "revision", "retrieval", "runtime", "workers", "campaign"])
def test_changed_task_identity_cannot_reuse_a_baseline(tmp_path, monkeypatch, axis):
    """Use fresh baseline evidence when a material task setting changes."""
    contract = run_contract()
    task = Mock(return_value=("query", "Alpha", {}))
    monkeypatch.setattr(hotpot, "run_program", task)
    original = evaluate(tmp_path / "run", contract)
    changed = deepcopy(contract)
    if axis == "candidate":
        changed["optimizer"]["rendered_seed"]["summarize1"] += " changed"
    elif axis == "decoding":
        changed["models"]["solver_decoding"]["temperature"] = 0.0
    elif axis == "revision":
        changed["models"]["solver_version"] = "different-checkpoint"
    elif axis == "retrieval":
        changed["retrieval"]["index_sha256"] = "different-index"
    elif axis == "workers":
        changed["program"]["parallel_workers"] = 2
    elif axis == "campaign":
        changed["execution_runtime"]["campaign_id"] = "another-campaign"
    else:
        changed["execution_runtime"]["serving_env_sha256"] = "different-server"
    current = evaluate(tmp_path / "new-run", changed)
    assert task.call_count == 4
    assert original["contract_sha256"] != current["contract_sha256"]


def test_changed_frozen_baseline_is_rejected(tmp_path, monkeypatch):
    """Do not replace recorded baseline provenance in place."""
    contract = run_contract()
    task = Mock(return_value=("query", "Alpha", {}))
    monkeypatch.setattr(hotpot, "run_program", task)
    evaluate(tmp_path / "run", contract)
    directory = baseline_directory(tmp_path / "run", build_baseline_contract(contract))
    (directory / BASELINE_CONTRACT_FILENAME).write_text("{}")
    with pytest.raises(ValueError, match="baseline configuration changed"):
        evaluate(tmp_path / "run", contract)
    assert task.call_count == 2


def test_baseline_prevents_concurrent_writers(tmp_path, monkeypatch):
    """Avoid duplicate model calls when two ablations request the same baseline."""
    task = Mock()
    monkeypatch.setattr(hotpot, "run_program", task)
    contract = run_contract()
    directory = baseline_directory(tmp_path / "run", build_baseline_contract(contract))
    directory.mkdir(parents=True)
    with (directory / ".baseline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            evaluate(tmp_path / "run", contract)
    task.assert_not_called()


@pytest.mark.parametrize("unchanged_winner", [False, True])
def test_cli_reports_baseline_gains_without_feeding_test_results_into_optimization(
    tmp_path, monkeypatch, unchanged_winner
):
    """Reuse baseline evidence across separate runs while independently testing each selected winner."""
    monkeypatch.chdir(tmp_path)
    training = [{"id": "train", "question": "train question", "answer": "answer"}]
    validation = [{"id": "val", "question": "val question", "answer": "answer"}]
    monkeypatch.setattr(hotpot, "load_hotpotqa_dataset", lambda **kwargs: (training, validation, DATASET))
    retriever = SimpleNamespace(
        provenance=lambda: {"bm25s_version": "fixed", "k1": 0.9, "b": 0.4},
        search=lambda *args: ["passage"] * 7,
    )
    monkeypatch.setattr(hotpot, "Wiki17BM25Retriever", lambda _: retriever)
    monkeypatch.setattr(hotpot, "build_config", lambda *args, **kwargs: (None, None))
    events = []

    def optimize(label, seed, trainset, valset, *args, **kwargs):
        assert trainset == training and valset == validation
        events.append("optimize")
        state = GEPAState(seed, ValsetEvaluation({}, {0: 0.0}))
        if not unchanged_winner:
            winner = {key: value + " improved" for key, value in seed.items()}
            state.update_state_with_new_program(
                [0], winner, ValsetEvaluation({}, {0: 1.0}), None, 1, iteration_id="offline-winner"
            )
        return GEPAResult.from_state(state)

    def task(candidate, question, *args):
        events.append("test")
        improved = any(text.endswith(" improved") for text in candidate.values())
        answer = "Alpha" if question == "first question" else "Beta"
        return "query", answer if improved else "Wrong", {}

    monkeypatch.setattr(hotpot, "run_condition", optimize)
    task_mock = Mock(side_effect=task)
    monkeypatch.setattr(hotpot, "run_program", task_mock)
    for budget in (6871, 13742):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hotpotqa",
                "--condition",
                "vanilla",
                "--seed-style",
                "structured",
                "--max-workers",
                "1",
                "--max-metric-calls",
                str(budget),
            ],
        )
        hotpot.main()
    assert events == ["optimize", *["test"] * 4, "optimize", *["test"] * 2]
    rows = [json.loads(path.read_text()) for path in (tmp_path / "outputs").glob("*/final_metrics.json")]
    assert len(rows) == 2
    assert rows[0]["baseline"] == rows[1]["baseline"]
    for row in rows:
        assert row["baseline"]["test_exact_match"] == row["baseline"]["test_f1"] == 0.0
        assert row["test_exact_match_gain"] == row["test_f1_gain"] == (0.0 if unchanged_winner else 1.0)
        assert row["baseline"]["test_example_count"] == row["test_example_count"] == 2
