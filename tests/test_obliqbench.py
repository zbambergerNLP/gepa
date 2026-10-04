"""Exercise OBLIQ data isolation and real retrieval/evaluation with fake model boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.benchmark_runner import build_parser, resolve_models
from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL
from examples.obliqbench import retrieval, utils
from examples.obliqbench.adapter import COMPONENT, ObliqAdapter, parse_rewrite, score_ranking
from examples.obliqbench.benchmark_settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    QUERY_COUNTS,
    RETRIEVAL_K,
    SUBSETS,
)
from examples.obliqbench.main import add_arguments, build_benchmark
from examples.obliqbench.retrieval import BM25Retriever, DenseRetriever, ranked_results
from examples.obliqbench.utils import Corpus, digest, load_data, read_qrels, split_records, verify_file


def query_record(query_id="q0", **changes):
    """Represent the evaluator boundary without supplying labels to the solver."""
    return {
        "id": f"twitter/{query_id}",
        "subset": "twitter",
        "query_id": query_id,
        "query": "blue bird",
        "gold_qrels": {"d1": 2, "d2": 1},
        "pooled_qrels": {"d1": 2, "d2": 1, "d3": 1},
        "excluded_ids": ["self"],
        "split": "train",
        "group_id": "group",
        **changes,
    }


@pytest.fixture
def corpus(tmp_path):
    """Create real JSONL corpus input for both published retrieval backends."""
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"_id": "self", "text": "blue bird blue bird"},
        {"_id": "d1", "text": "blue bird"},
        {"_id": "d2", "text": "blue sea"},
        {"_id": "d3", "text": "yellow flower"},
        {"_id": "d4", "text": "forest tree"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return Corpus(path, tuple(row["_id"] for row in rows), {"fixture_sha256": digest(rows)})


class EncoderBoundary:
    """Replace the weight download/inference boundary with deterministic vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, query):
        self.calls.append((list(texts), query))
        values = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for index, text in enumerate(texts):
            values[index, 0] = 1 + text.count("blue")
            values[index, 1] = 1 + text.count("yellow")
        return values


def test_exact_dense_retrieval_masks_before_topk_and_reuses_immutable_index(corpus, tmp_path):
    encoder = EncoderBoundary()
    retriever = DenseRetriever(corpus, encoder, tmp_path / "index", {"pinned": True}, batch_size=2)
    assert len(encoder.calls) == 3
    ranking = retriever.search("blue bird", {"self"}, 2)
    assert ranking[0][0] == "d1"
    assert "self" not in [doc for doc, _ in ranking]
    before = len(encoder.calls)
    DenseRetriever(corpus, encoder, tmp_path / "index", {"pinned": True}, batch_size=2)
    assert len(encoder.calls) == before
    cache = next((tmp_path / "index").glob("*.npy"))
    with cache.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="cache identity or content drift"):
        DenseRetriever(corpus, encoder, tmp_path / "index", {"pinned": True}, batch_size=2)


def test_incomplete_embedding_run_is_not_cached(corpus, tmp_path):
    class IncompleteEncoder(EncoderBoundary):
        def encode(self, texts, *, query):
            return super().encode(texts, query=query)[:-1]

    with pytest.raises(ValueError, match="Incomplete or malformed"):
        DenseRetriever(corpus, IncompleteEncoder(), tmp_path / "index", {}, batch_size=2)
    assert not list((tmp_path / "index").iterdir())


def test_incomplete_index_fails_closed(corpus, tmp_path):
    retriever = DenseRetriever(corpus, EncoderBoundary(), tmp_path / "index", {}, batch_size=2)
    assert len(retriever.ids) == 5
    next((tmp_path / "index").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="Incomplete OBLIQ embedding index"):
        DenseRetriever(corpus, EncoderBoundary(), tmp_path / "index", {}, batch_size=2)


def test_graded_official_metrics_and_pooled_denominator():
    record = query_record()
    metrics = score_ranking(record, [("d2", 0.9), ("d1", 0.8), ("d4", 0.1)])
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert metrics["gold_ndcg_at_10"] == pytest.approx(expected)
    assert metrics["gold_recall_at_10"] == 1.0
    assert metrics["pooled_recall_at_10"] == pytest.approx(2 / 3)
    assert set(metrics) == {
        f"{label}_{metric}_at_{k}"
        for label in ("gold", "pooled")
        for metric, cutoffs in (("ndcg", (10, 50)), ("recall", (10, 50, 100)))
        for k in cutoffs
    }


def test_actual_candidate_prompt_and_rewrite_reach_real_retrieval_without_gold(corpus):
    requests = []

    def solver(messages):
        requests.append(messages)
        return json.dumps({"query": "blue bird" if messages[0]["content"] == "blue instruction" else "yellow flower"})

    adapter = ObliqAdapter(solver, {"twitter": BM25Retriever(corpus)})
    record = query_record()
    original = deepcopy(record)
    first = adapter.evaluate([record], {COMPONENT: "blue instruction"}, capture_traces=True)
    second = adapter.evaluate([record], {COMPONENT: "yellow instruction"}, capture_traces=True)
    assert first.outputs[0]["ranking"][0] == "d1"
    assert second.outputs[0]["ranking"][0] == "d3"
    assert first.outputs[0]["error"] is None
    assert first.outputs[0]["elapsed_seconds"] >= 0
    assert record == original
    assert requests[0][0] == {"role": "system", "content": "blue instruction"}
    assert set(json.loads(requests[0][1]["content"])) == {"task", "query"}
    assert "qrels" not in json.dumps(requests) and "d1" not in json.dumps(requests)
    feedback = adapter.make_reflective_dataset({COMPONENT: "blue instruction"}, first, [COMPONENT])
    assert "qrels" not in json.dumps(feedback) and "d1" not in json.dumps(feedback)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "[]",
        '{"query": ""}',
        '{"query": ["a", "b"]}',
        '{"query":"a","gold":"d1"}',
        '{"query": "truncated',
        "<think>unfinished",
    ],
)
def test_malformed_rewrites_fail_before_retrieval(corpus, output):
    adapter = ObliqAdapter(lambda messages: output, {"twitter": BM25Retriever(corpus)})
    result = adapter.evaluate([query_record()], {COMPONENT: "rewrite"})
    assert result.scores == [0.0]
    assert result.outputs[0]["error"]
    assert "ranking" not in result.outputs[0]
    with pytest.raises(ValueError, match="Failed OBLIQ"):
        adapter.summarize_evaluation([query_record()], [result])


def test_heldout_traces_cannot_enter_reflection(corpus):
    adapter = ObliqAdapter(lambda messages: '{"query": "blue bird"}', {"twitter": BM25Retriever(corpus)})
    result = adapter.evaluate([query_record(split="test")], {COMPONENT: "rewrite"}, capture_traces=True)
    with pytest.raises(ValueError, match="Held-out"):
        adapter.make_reflective_dataset({COMPONENT: "rewrite"}, result, [COMPONENT])


def test_original_query_reference_has_no_model_call_and_summary_requires_all_records(corpus):
    def forbidden_solver(messages):
        raise AssertionError("Original-query reference must not call an LLM")

    adapter = ObliqAdapter(forbidden_solver, {"twitter": BM25Retriever(corpus)}, original_query=True)
    records = [query_record()]
    result = adapter.evaluate(records, {COMPONENT: "unused seed"})
    assert result.outputs[0]["rewritten_query"] == "blue bird"
    summary = adapter.summarize_evaluation(records, [result])
    assert summary["per_subset"]["twitter"]["gold_recall_at_10"] == 1.0
    result.outputs = []
    with pytest.raises(ValueError, match="Incomplete or reordered"):
        adapter.summarize_evaluation(records, [result])


def test_retrieval_ties_masks_and_invalid_scores():
    assert ranked_results(("z", "b", "a"), [1, 1, 1], {"z"}, 2) == [("b", 1), ("a", 1)]
    with pytest.raises(ValueError, match="invalid or incomplete"):
        ranked_results(("a", "b"), [math.nan, 1], set(), 2)


def test_groups_include_pooled_and_excluded_documents():
    records = [
        query_record(f"q{i}", query=f"query {i}", gold_qrels={f"d{i}": 1}, pooled_qrels=None, excluded_ids=[])
        for i in range(10)
    ]
    records[0]["pooled_qrels"] = {"d0": 1, "d1": 1}
    records[2]["excluded_ids"] = ["d1"]
    grouped = split_records(records)
    assert len({r["group_id"] for r in grouped[:3]}) == 1
    assert len({r["split"] for r in grouped[:3]}) == 1
    assert grouped == split_records(deepcopy(records))
    assert {r["split"] for r in grouped} == {"train", "val", "test"}
    assert [r["id"] for r in grouped] == [r["id"] for r in records]


def test_reviewed_manifest_covers_all_official_queries_and_disjoint_groups():
    rows = [json.loads(line) for line in utils.RECORD_PATH.read_text().splitlines()]
    assert Counter(row["id"].split("/")[0] for row in rows) == QUERY_COUNTS
    assert len({row["id"] for row in rows}) == 1238
    groups = {}
    for row in rows:
        assert groups.setdefault(row["group_id"], row["split"]) == row["split"]
        assert len(row["sha256"]) == 64
    assert len(EMBEDDING_REVISION) == 40
    assert EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert RETRIEVAL_K == 1000


def test_qrels_preserve_relevance_grades_and_reject_malformed_files(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("query_id\tcorpus_id\tscore\nq\td1\t2\nq\td2\t1\n")
    assert read_qrels(path, {"q"}) == {"q": {"d1": 2, "d2": 1}}
    path.write_text(path.read_text() + "q\td1\t1\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_qrels(path, {"q"})


def test_file_identity_rejects_missing_truncated_and_changed_files(tmp_path):
    path = tmp_path / "data"
    content = b"official content\n"
    identity = {
        "size": len(content),
        "algorithm": "git_blob_sha1",
        "digest": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
    }
    with pytest.raises(ValueError, match="Missing or incomplete"):
        verify_file(path, identity)
    path.write_bytes(content)
    verify_file(path, identity)
    path.write_bytes(content.replace(b"official", b"modified"))
    with pytest.raises(ValueError, match="content changed"):
        verify_file(path, identity)


@pytest.fixture
def pinned_data(tmp_path, monkeypatch):
    """Supply small pinned files at the external dataset boundary, exercising the actual loader."""
    subset_dir = tmp_path / SUBSETS["twitter"]
    (subset_dir / "queries+qrels").mkdir(parents=True)
    (subset_dir / "corpus").mkdir()
    for part in ("queries+qrels/queries.jsonl", "corpus/corpus.jsonl"):
        (subset_dir / part).write_text(
            "".join(json.dumps({"_id": f"{i}", "text": f"record {i}"}) + "\n" for i in range(9))
        )
    for name in ("qrels.tsv", "qrels_pool.tsv"):
        (subset_dir / "queries+qrels" / name).write_text(
            "query-id\tcorpus-id\tscore\n" + "".join(f"{i}\t{i}\t2\n" for i in range(9))
        )
    source = utils.load_sources()
    source["files"] = {}
    for path in subset_dir.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            source["files"][str(path.relative_to(tmp_path))] = {
                "size": len(content),
                "algorithm": "sha256",
                "digest": hashlib.sha256(content).hexdigest(),
            }
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(source))
    monkeypatch.setattr(utils, "SOURCE_PATH", source_path)
    monkeypatch.setitem(utils.QUERY_COUNTS, "twitter", 9)
    monkeypatch.setitem(utils.CORPUS_COUNTS, "twitter", 9)
    records = split_records(utils.load_records(tmp_path, "twitter"))
    record_path = tmp_path / "records.jsonl"
    record_path.write_text(
        "".join(
            json.dumps({"id": r["id"], "sha256": digest(r), "split": r["split"], "group_id": r["group_id"]}) + "\n"
            for r in records
        )
    )
    monkeypatch.setattr(utils, "RECORD_PATH", record_path)
    return tmp_path


def test_loader_freezes_order_and_rejects_record_drift(pinned_data):
    data = load_data(pinned_data, ["twitter"])
    assert sum(len(rows) for rows in data.splits.values()) == 9
    assert set(data.source["source_files"]) == set(utils.load_sources()["files"])
    lines = utils.RECORD_PATH.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    utils.RECORD_PATH.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Ordered OBLIQ records or splits drifted"):
        load_data(pinned_data, ["twitter"])


def test_loader_rejects_changed_data_before_using_manifest(pinned_data):
    query_path = pinned_data / SUBSETS["twitter"] / "queries+qrels/queries.jsonl"
    query_path.write_text(query_path.read_text().replace("record 0", "altered0"))
    with pytest.raises(ValueError, match="content changed"):
        load_data(pinned_data, ["twitter"])


@pytest.mark.smoke
def test_live_math_data_integrity():
    """Use only already-downloaded official files; never implicitly fetch data or models."""
    root = Path(__file__).parents[1] / ".cache/obliqbench/data"
    if not (root / SUBSETS["math"] / "corpus/corpus.jsonl").exists():
        pytest.skip("Run examples.obliqbench.prepare --subsets math first")
    data = load_data(root, ["math"])
    assert {split: len(rows) for split, rows in data.splits.items()} == {"train": 91, "val": 30, "test": 30}
    assert len(data.corpora["math"].ids) == 3508
    docs_by_split = {}
    for split, records in data.splits.items():
        docs_by_split[split] = set().union(
            *(set(r["gold_qrels"]) | set(r["pooled_qrels"] or {}) | set(r["excluded_ids"]) for r in records)
        )
    assert not docs_by_split["train"] & docs_by_split["test"]
    assert not docs_by_split["train"] & docs_by_split["val"]
    assert not docs_by_split["val"] & docs_by_split["test"]


def test_parse_rewrite_preserves_non_ascii_content():
    assert parse_rewrite('{"query": "  מצא מסמכים  "}') == "מצא מסמכים"


def test_duplicate_json_fields_are_not_silently_accepted():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        parse_rewrite('{"query": "one", "query": "two"}')


def test_build_uses_shared_models_budgets_and_real_retriever(pinned_data):
    args = build_parser("obliqbench", add_arguments).parse_args(
        [
            "--subsets",
            "twitter",
            "--retriever",
            "bm25",
            "--data-dir",
            str(pinned_data),
            "--run-dir",
            str(pinned_data / "outputs"),
            "--solver-api-base",
            "http://localhost:8000/v1",
            "--reflection-api-base",
            "http://localhost:8001/v1",
        ]
    )
    models = resolve_models(args)
    definition = build_benchmark(args, models)
    assert models.solver_model == QWEN3_8_27B_MODEL
    assert models.proposer_model == DEEPSEEK_V4_1_FLASH_MODEL
    assert definition.adapter.solver.model == models.solver_model
    assert definition.adapter.solver.completion_kwargs["max_tokens"] == models.solver_kwargs["max_tokens"]
    assert definition.adapter.solver.completion_kwargs["api_base"] == models.solver_api_base
    assert definition.adapter.solver.completion_kwargs["extra_body"] == models.solver_kwargs["extra_body"]
    assert definition.adapter.solver.num_retries == models.solver_kwargs["num_retries"]
    assert len(definition.trainset) + len(definition.valset) + len(definition.testset) == 9
    assert set(definition.seed_candidate) == {COMPONENT}
    assert definition.component_kinds == {COMPONENT: "system_prompt"}
    assert definition.runtime["top_k"] == 1000
    assert isinstance(definition.adapter.retrievers["twitter"], BM25Retriever)


def test_original_query_cannot_be_optimized():
    args = build_parser("obliqbench", add_arguments).parse_args(["--original-query-reference"])
    with pytest.raises(ValueError, match="only with --mode baseline"):
        build_benchmark(args, resolve_models(args))


def test_qwen_embedding_boundary_uses_exact_pinned_checkpoint(monkeypatch):
    calls = []

    class ExternalModel:
        def __init__(self, *args, **kwargs):
            self.prompts = {"query": "fixed checkpoint instruction"}
            calls.append((args, kwargs))

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)

    original_import = retrieval.importlib.import_module

    def import_external(name):
        if name == "torch":
            return SimpleNamespace(float32="float32")
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=ExternalModel)
        return original_import(name)

    monkeypatch.setattr(retrieval.importlib, "import_module", import_external)
    encoder = retrieval.QwenEncoder("cpu")
    assert calls[0][0] == (EMBEDDING_MODEL,)
    assert calls[0][1]["revision"] == EMBEDDING_REVISION
    assert calls[0][1]["trust_remote_code"] is False
    assert calls[0][1]["processor_kwargs"] == {"padding_side": "left"}
    assert encoder.model.max_seq_length == 32768
    encoder.encode(["document"], query=False)
    assert calls[-1][1]["prompt"] == ""
    encoder.encode(["question"], query=True)
    assert calls[-1][1]["prompt_name"] == "query"
