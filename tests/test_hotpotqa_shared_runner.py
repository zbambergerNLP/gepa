"""Verify the shared HotPotQA route with the real DSPy program and EM metric."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from examples.common.benchmark_runner import build_parser, resolve_models
from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL
from examples.common.provider_retries import ProviderRequestError
from examples.common.wikipedia import WikipediaPassage
from examples.hotpotqa import adapter as adapter_module
from examples.hotpotqa import main, utils
from examples.hotpotqa.adapter import HotPotQAAdapter
from examples.hotpotqa.benchmark_settings import (
    HOTPOTQA_SCIENTIFIC_SPLIT_SHA256,
    RETRIEVAL_K,
    SEED_CANDIDATE,
)


class FixtureRetriever:
    """Replace the external large corpus/index while retaining the retrieval boundary."""

    def __init__(self, root=None):
        self.queries = []
        self.verified = False

    def verify_integrity(self):
        self.verified = True

    def provenance(self):
        return {"integrity_manifest_sha256": "verified-fixture-index", "corpus_sha256": "frozen-fixture"}

    def search(self, query, limit):
        self.queries.append((query, limit))
        return [WikipediaPassage(f"Retrieved {index}", f"Visible evidence {index}") for index in range(limit)]


def example(index=0):
    return {
        "id": f"question-{index}",
        "question": f"Question {index}?",
        "answer": "Final Gold Secret",
        "context": {"title": ["Feedback-only title"], "sentences": [["Feedback-only private gold context"]]},
        "supporting_facts": {"title": ["Feedback-only title"], "sent_id": [0]},
    }


def model_settings():
    args = build_parser("hotpotqa", main.add_arguments).parse_args([])
    return resolve_models(args)


def dummy_lm():
    if utils.dspy is None:
        pytest.skip("Requires the pinned hotpotqa-task-program dependency group")
    return utils.dspy.utils.DummyLM(
        [
            {"reasoning": "First summary reasoning", "summary": "First summary"},
            {"reasoning": "Query reasoning", "query": "Second retrieval query"},
            {"reasoning": "Second summary reasoning", "summary": "Second summary"},
            {"reasoning": "Final reasoning", "answer": "The final gold secret!"},
        ]
    )


@pytest.mark.skipif(utils.dspy is None, reason="Requires the pinned hotpotqa-task-program group")
def test_real_dspy_program_executes_all_edited_components_and_keeps_gold_out():
    retriever = FixtureRetriever()
    adapter = HotPotQAAdapter(model_settings(), retriever, training_ids={"question-0"})
    adapter.task_lm = dummy_lm()
    candidate = {name: f"Edited instruction for {name}" for name in SEED_CANDIDATE}
    result = adapter.evaluate([example()], candidate, capture_traces=True)
    assert result.scores == [1.0]
    assert result.outputs[0]["prediction"] == "The final gold secret!"
    assert result.outputs[0]["elapsed_seconds"] > 0
    assert retriever.queries == [("Question 0?", RETRIEVAL_K), ("Second retrieval query", RETRIEVAL_K)]
    assert len(adapter.task_lm.history) == 4
    for history, prompt in zip(adapter.task_lm.history, candidate.values(), strict=True):
        assert prompt in history["messages"][0]["content"]
        inputs = json.dumps(history["messages"]).casefold()
        assert "final gold secret" not in inputs
        assert "feedback-only" not in inputs
    assert set(result.outputs[0]["trace"]) >= {"hop1_documents", "hop2_documents", "summary_1", "summary_2", "answer"}
    json.dumps(result.outputs, allow_nan=False)
    feedback = adapter.make_reflective_dataset(candidate, result, list(SEED_CANDIDATE))
    assert set(feedback) == set(SEED_CANDIDATE)
    assert "Final Gold Secret" in feedback["final_answer"][0]["End-to-end Outcome"]["reference_answer"]
    assert "First summary" == feedback["create_query_hop2"][0]["Inputs"]["summary_1"]
    result.trajectories[0]["record"]["id"] = "held-out"
    with pytest.raises(ValueError, match="cannot enter reflection"):
        adapter.make_reflective_dataset(candidate, result, ["final_answer"])


@pytest.mark.skipif(utils.dspy is None, reason="Requires the pinned hotpotqa-task-program group")
def test_malformed_dspy_output_is_zero_but_systemic_failures_abort():
    adapter = HotPotQAAdapter(model_settings(), FixtureRetriever(), training_ids={"question-0"})
    lm = dummy_lm()
    lm.answers = iter([{"reasoning": "no required summary field"}])
    adapter.task_lm = lm
    result = adapter.evaluate([example()], SEED_CANDIDATE, capture_traces=True)
    assert result.scores == [0]
    assert result.outputs[0]["error"] == "task_output_parse_error"
    assert result.outputs[0]["elapsed_seconds"] > 0
    feedback = adapter.make_reflective_dataset(SEED_CANDIDATE, result, ["summarize1"])
    assert feedback["summarize1"][0]["Feedback"]["error"] == "task_output_parse_error"

    class FailedProvider(type(lm)):
        def __call__(self, *args, **kwargs):
            raise ProviderRequestError("provider exhausted")

    adapter.task_lm = FailedProvider([])
    with pytest.raises(ProviderRequestError):
        adapter.evaluate([example()], SEED_CANDIDATE)


def test_shared_primary_cli_runs_a_training_pilot_and_retains_full_data_identity(tmp_path, monkeypatch, capsys):
    rows = [example(index) for index in range(20)]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    solver = dummy_lm()
    monkeypatch.setattr(main, "Wiki17BM25Retriever", FixtureRetriever)
    monkeypatch.setattr(adapter_module, "build_hotpotqa_task_lm", lambda *_: solver)
    result = main.main(
        [
            "--mode",
            "pilot",
            "--data-path",
            str(data),
            "--run-dir",
            str(tmp_path / "run"),
            "--pilot-size",
            "1",
            "--train-limit",
            "1",
            "--seed",
            "53",
        ]
    )
    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mean_score"] == 1 and summary["timing"]["attempt_count"] == 1
    contract = json.loads((tmp_path / "run/pilot/evaluation-contract.json").read_text())
    identity = contract["identity"]
    assert contract["split"] == "train" and len(contract["ids"]) == 1
    assert identity["full_data"]["splits"]["train"]["count"] == 14
    assert identity["solver"]["model"] == QWEN3_8_27B_MODEL
    assert identity["runtime"]["program"] == "2stage"
    assert len(identity["seed_candidate"]) == 4
    assert len(solver.history) == 4


@pytest.mark.skipif(utils.dspy is None, reason="Requires the pinned hotpotqa-task-program group")
def test_builder_uses_shared_models_and_ignores_optimizer_seed_for_data(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(example(index)) + "\n" for index in range(20)))
    monkeypatch.setattr(main, "Wiki17BM25Retriever", FixtureRetriever)
    monkeypatch.setattr(adapter_module, "build_hotpotqa_task_lm", lambda *_: dummy_lm())
    parser = build_parser("hotpotqa", main.add_arguments)
    first = parser.parse_args(["--data-path", str(data), "--seed", "0"])
    second = parser.parse_args(
        ["--data-path", str(data), "--seed", "123", "--max-metric-calls", "5", "--train-limit", "1"]
    )
    models = resolve_models(first)
    assert (models.solver_model, models.proposer_model) == (QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL)
    one, two = main.build_benchmark(first, models), main.build_benchmark(second, models)
    assert one.trainset == two.trainset and one.valset == two.valset and one.testset == two.testset
    assert one.seed_candidate == two.seed_candidate
    assert one.runtime == two.runtime and one.source == two.source
    assert all(text in one.seed_candidate[key] for key, text in SEED_CANDIDATE.items())
    assert one.adapter.retriever.verified


def test_small_overlapping_smoke_data_cannot_enter_shared_optimizer(tmp_path):
    data = tmp_path / "invalid.jsonl"
    data.write_text(json.dumps(example()) + "\n")
    with pytest.raises(ValueError, match="repeated question identities"):
        main.load_benchmark_data(data)


def test_canonical_data_drift_fails_before_retrieval(monkeypatch):
    monkeypatch.setattr(main, "load_hotpotqa_dataset", lambda **_: ([example(0)], [example(1)], [example(2)]))
    with pytest.raises(ValueError, match="pinned ordered train split changed"):
        main.load_benchmark_data(None)


def test_primary_and_legacy_entrypoints_are_explicit():
    from examples.hotpotqa import legacy_main

    assert main.main is not legacy_main.main
    assert main.build_benchmark.__module__ == "examples.hotpotqa.main"
    parser = build_parser("hotpotqa", main.add_arguments)
    assert parser.parse_args([]).condition == "both"
    assert legacy_main.build_parser().parse_args([]).program == "2stage"
    workload = Path("scripts/della/remote/hotpotqa_workload.sh").read_text()
    assert '"${PY}" -m examples.hotpotqa.legacy_main' in workload


@pytest.mark.smoke
def test_cached_canonical_splits_match_frozen_paper_identity():
    if os.environ.get("HOTPOTQA_VERIFY_CACHED_DATA") != "1":
        pytest.skip("Set HOTPOTQA_VERIFY_CACHED_DATA=1 with the pinned dataset already cached")
    train, val, test, _ = main.load_benchmark_data(None)
    assert (len(train), len(val), len(test)) == (150, 300, 300)
    assert set(HOTPOTQA_SCIENTIFIC_SPLIT_SHA256) == {"train", "val", "test"}
    assert len({row["id"] for row in train + val + test}) == 750
