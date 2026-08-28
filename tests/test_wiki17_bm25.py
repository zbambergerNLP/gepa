"""Tests for the frozen Wiki-2017 BM25 retriever."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common import wiki17_bm25


def install_fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install lightweight dependency fakes at the artifact's pinned versions.

    Args:
        monkeypatch: Pytest fixture used to replace optional dependencies.

    Returns:
        Namespace containing the dependency mocks for call assertions.
    """
    tokenize = Mock(return_value="tokens")
    bm25_factory = Mock()
    stemmer_factory = Mock(return_value="english stemmer")
    cache_factory = Mock()
    installed_versions = {
        "bm25s": wiki17_bm25.WIKI17_BM25_VERSION,
        "PyStemmer": wiki17_bm25.WIKI17_PYSTEMMER_VERSION,
        "jax": wiki17_bm25.WIKI17_JAX_VERSION,
    }
    monkeypatch.setattr(
        wiki17_bm25,
        "bm25s",
        SimpleNamespace(tokenize=tokenize, BM25=bm25_factory),
    )
    monkeypatch.setattr(wiki17_bm25, "Stemmer", SimpleNamespace(Stemmer=stemmer_factory))
    monkeypatch.setattr(wiki17_bm25, "Cache", cache_factory)
    monkeypatch.setattr(wiki17_bm25, "version", installed_versions.__getitem__)
    return SimpleNamespace(
        tokenize=tokenize,
        bm25_factory=bm25_factory,
        stemmer_factory=stemmer_factory,
        cache_factory=cache_factory,
    )


def test_prepare_uses_artifact_bm25_settings_without_downloading(monkeypatch, tmp_path) -> None:
    """Build a fake index with the artifact's exact tokenizer and BM25 settings.

    Args:
        monkeypatch: Pytest fixture used to replace dependencies and I/O.
        tmp_path: Pytest directory used for isolated prepared state.
    """
    dependencies = install_fake_dependencies(monkeypatch)
    corpus = ["Alpha | first abstract", "Beta | second abstract"]
    index = Mock()

    def save_index(path: Path) -> None:
        """Materialize the directory created by a BM25S index save.

        Args:
            path: Temporary index directory selected by the retriever.
        """
        Path(path).mkdir(parents=True)
        (Path(path) / "index.json").write_text("{}", encoding="utf-8")

    index.save.side_effect = save_index
    dependencies.bm25_factory.return_value = index
    monkeypatch.setattr(wiki17_bm25, "WIKI17_DOCUMENT_COUNT", len(corpus))
    retriever = wiki17_bm25.Wiki17BM25Retriever(tmp_path)
    monkeypatch.setattr(retriever, "_ensure_archive", Mock())
    monkeypatch.setattr(retriever, "_ensure_corpus", Mock())
    monkeypatch.setattr(retriever, "_load_corpus", Mock(return_value=corpus))

    manifest = retriever.prepare()

    dependencies.stemmer_factory.assert_called_once_with("english")
    dependencies.tokenize.assert_called_once_with(corpus, stopwords="en", stemmer="english stemmer")
    dependencies.bm25_factory.assert_called_once_with(k1=0.9, b=0.4)
    index.index.assert_called_once_with("tokens")
    index.save.assert_called_once_with(tmp_path / f".{wiki17_bm25.WIKI17_INDEX_DIR_NAME}.building")
    assert retriever.index_path.is_dir()
    assert json.loads(retriever.manifest_path.read_text(encoding="utf-8")) == manifest


def test_technical_mini_index_uses_all_selected_contexts_and_same_bm25(monkeypatch, tmp_path) -> None:
    """Build a non-scientific selected-context index with the pinned ranker.

    Args:
        monkeypatch: Pytest fixture used to replace optional dependencies.
        tmp_path: Pytest directory used for isolated prepared state.
    """
    dependencies = install_fake_dependencies(monkeypatch)
    index = Mock()

    def save_index(path: Path) -> None:
        """Materialize the directory created by a BM25S index save.

        Args:
            path: Temporary index directory selected by the retriever.
        """
        Path(path).mkdir(parents=True)
        (Path(path) / "index.json").write_text("{}", encoding="utf-8")

    index.save.side_effect = save_index
    dependencies.bm25_factory.return_value = index
    examples = [
        {
            "id": "train-1",
            "context": {
                "title": ["Alpha", "Distractor"],
                "sentences": [["alpha evidence"], ["irrelevant text"]],
            },
        },
        {
            "id": "validation-1",
            "context": {
                "title": ["Alpha", "Bridge"],
                "sentences": [["duplicate is ignored"], ["bridge evidence"]],
            },
        },
    ]
    retriever = wiki17_bm25.HotPotQATechnicalMiniBM25Retriever(examples, tmp_path)

    manifest = retriever.prepare()

    rows = [json.loads(line) for line in retriever.corpus_path.read_text(encoding="utf-8").splitlines()]
    assert [row["title"] for row in rows] == ["Alpha", "Distractor", "Bridge"]
    assert rows[0]["text"] == ["alpha evidence"]
    assert manifest["backend"] == "hotpotqa-technical-mini-bm25s"
    assert manifest["mode"] == "technical-smoke-only"
    assert manifest["scientific_comparability"] is False
    assert manifest["contains_benchmark_context"] is True
    assert manifest["document_count"] == 3
    assert manifest["corpus_sha256"] == retriever.corpus_sha256
    dependencies.tokenize.assert_called_once_with(
        ["Alpha | alpha evidence", "Distractor | irrelevant text", "Bridge | bridge evidence"],
        stopwords="en",
        stemmer="english stemmer",
    )
    dependencies.bm25_factory.assert_called_once_with(k1=0.9, b=0.4)
    assert retriever.prepare() == manifest
    dependencies.bm25_factory.assert_called_once()


def test_technical_mini_identity_changes_with_the_selected_records(tmp_path) -> None:
    """Keep different selected benchmark examples out of the same run identity.

    Args:
        tmp_path: Pytest directory used to configure the retrievers.
    """
    context = {"title": ["Alpha"], "sentences": [["same corpus text"]]}
    first = wiki17_bm25.HotPotQATechnicalMiniBM25Retriever([{"id": "first", "context": context}], tmp_path)
    second = wiki17_bm25.HotPotQATechnicalMiniBM25Retriever([{"id": "second", "context": context}], tmp_path)

    assert first.provenance()["corpus_sha256"] == second.provenance()["corpus_sha256"]
    assert first.provenance()["selection_sha256"] != second.provenance()["selection_sha256"]


def test_search_preserves_artifact_order_dedup_threads_and_cache(monkeypatch, tmp_path) -> None:
    """Match artifact ranking semantics and avoid repeating cached retrievals.

    Args:
        monkeypatch: Pytest fixture used to replace optional dependencies.
        tmp_path: Pytest directory used for isolated prepared state.
    """
    dependencies = install_fake_dependencies(monkeypatch)
    corpus_rows = [
        {"title": "Alpha", "text": ["first abstract"]},
        {"title": "Beta", "text": ["second", "abstract"]},
        {"title": "Alpha", "text": ["first abstract"]},
    ]
    corpus_text = "".join(json.dumps(row) + "\n" for row in corpus_rows)
    retriever = wiki17_bm25.Wiki17BM25Retriever(tmp_path)
    retriever.corpus_path.write_text(corpus_text, encoding="utf-8")
    retriever.index_path.mkdir()
    (retriever.index_path / "index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(wiki17_bm25, "WIKI17_CORPUS_SIZE", retriever.corpus_path.stat().st_size)
    monkeypatch.setattr(wiki17_bm25, "WIKI17_DOCUMENT_COUNT", len(corpus_rows))
    retriever.manifest_path.write_text(json.dumps(retriever._expected_manifest()), encoding="utf-8")

    index = Mock()
    index.retrieve.return_value = ([[2, 1, 0]], [[0.9, 0.8, 0.7]])
    dependencies.bm25_factory.load.return_value = index
    cache = Mock()
    cache.get.side_effect = [None, [("Alpha", "first abstract"), ("Beta", "second abstract")]]
    dependencies.cache_factory.return_value = cache

    first = retriever.search("two hop question", 3)
    second = retriever.search("two hop question", 3)

    assert [(passage.title, passage.text) for passage in first] == [
        ("Alpha", "first abstract"),
        ("Beta", "second abstract"),
    ]
    assert second == first
    dependencies.bm25_factory.load.assert_called_once_with(retriever.index_path)
    dependencies.stemmer_factory.assert_called_once_with("english")
    dependencies.tokenize.assert_called_once_with(
        "two hop question",
        stopwords="en",
        stemmer="english stemmer",
        show_progress=False,
    )
    index.retrieve.assert_called_once_with("tokens", k=3, n_threads=1, show_progress=False)
    cache.set.assert_called_once_with(
        ("two hop question", 3),
        [("Alpha", "first abstract"), ("Beta", "second abstract")],
    )


def test_search_uses_one_stemmer_per_concurrent_evaluator_thread(monkeypatch, tmp_path) -> None:
    """Give each evaluator thread one private, reusable PyStemmer instance.

    Args:
        monkeypatch: Pytest fixture used to replace optional dependencies.
        tmp_path: Pytest directory used for isolated prepared state.
    """
    dependencies = install_fake_dependencies(monkeypatch)
    corpus_rows = [{"title": "Alpha", "text": ["first abstract"]}]
    corpus_text = "".join(json.dumps(row) + "\n" for row in corpus_rows)
    retriever = wiki17_bm25.Wiki17BM25Retriever(tmp_path)
    retriever.corpus_path.write_text(corpus_text, encoding="utf-8")
    retriever.index_path.mkdir()
    (retriever.index_path / "index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(wiki17_bm25, "WIKI17_CORPUS_SIZE", retriever.corpus_path.stat().st_size)
    monkeypatch.setattr(wiki17_bm25, "WIKI17_DOCUMENT_COUNT", len(corpus_rows))
    retriever.manifest_path.write_text(json.dumps(retriever._expected_manifest()), encoding="utf-8")

    index = Mock()
    index.retrieve.return_value = ([[0]], [[1.0]])
    dependencies.bm25_factory.load.return_value = index
    cache = Mock()
    cache.get.return_value = None
    dependencies.cache_factory.return_value = cache

    worker_count = 4
    tokenize_barrier = threading.Barrier(worker_count)
    observation_lock = threading.Lock()
    stemmers_by_thread: dict[int, list[object]] = {}

    def make_stemmer(language: str) -> object:
        """Create a distinguishable fake stemmer for one evaluator thread.

        Args:
            language: Stemmer language requested by the retriever.

        Returns:
            A unique object representing the new stemmer instance.
        """
        assert language == "english"
        return object()

    def tokenize_query(
        query: str,
        *,
        stopwords: str,
        stemmer: object,
        show_progress: bool,
    ) -> str:
        """Record the calling thread and synchronize concurrent tokenization.

        Args:
            query: Query text supplied by the evaluator thread.
            stopwords: Stopword set selected by the retriever.
            stemmer: Thread-local fake stemmer used for this query.
            show_progress: Whether BM25S progress output is enabled.

        Returns:
            The query text as a lightweight fake token sequence.
        """
        assert stopwords == "en"
        assert show_progress is False
        thread_id = threading.get_ident()
        with observation_lock:
            stemmers_by_thread.setdefault(thread_id, []).append(stemmer)
        tokenize_barrier.wait(timeout=5)
        return query

    def search_twice(worker_id: int) -> tuple[list[wiki17_bm25.WikipediaPassage], ...]:
        """Run two uncached searches from the same evaluator thread.

        Args:
            worker_id: Identifier used to make both cache keys unique.

        Returns:
            Both ranked passage lists returned to the evaluator.
        """
        first = retriever.search(f"question {worker_id} first", 1)
        second = retriever.search(f"question {worker_id} second", 1)
        return first, second

    dependencies.stemmer_factory.side_effect = make_stemmer
    dependencies.tokenize.side_effect = tokenize_query
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        search_results = list(executor.map(search_twice, range(worker_count)))

    expected = [("Alpha", "first abstract")]
    for worker_results in search_results:
        for passages in worker_results:
            assert [(passage.title, passage.text) for passage in passages] == expected
    assert len(stemmers_by_thread) == worker_count
    assert dependencies.stemmer_factory.call_count == worker_count
    assert len({id(stemmers[0]) for stemmers in stemmers_by_thread.values()}) == worker_count
    assert all(len(stemmers) == 2 and stemmers[0] is stemmers[1] for stemmers in stemmers_by_thread.values())
    assert index.retrieve.call_count == worker_count * 2


def test_prepared_state_requires_exact_manifest_index_and_corpus_size(monkeypatch, tmp_path) -> None:
    """Reject incomplete or stale prepared-state markers.

    Args:
        monkeypatch: Pytest fixture used to shrink the corpus-size invariant.
        tmp_path: Pytest directory used for isolated prepared state.
    """
    retriever = wiki17_bm25.Wiki17BM25Retriever(tmp_path)
    assert retriever._prepared_manifest() is None

    retriever.corpus_path.write_text("{}\n", encoding="utf-8")
    retriever.index_path.mkdir()
    (retriever.index_path / "index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(wiki17_bm25, "WIKI17_CORPUS_SIZE", retriever.corpus_path.stat().st_size)
    expected = retriever._expected_manifest()
    retriever.manifest_path.write_text(json.dumps(expected), encoding="utf-8")
    assert retriever._prepared_manifest() == expected

    retriever.corpus_path.write_text("{}\n{}\n", encoding="utf-8")
    assert retriever._prepared_manifest() is None


def test_provenance_locks_artifact_source_versions_and_retrieval_parameters() -> None:
    """Keep every material frozen-retrieval choice in run identity."""
    provenance = wiki17_bm25.Wiki17BM25Retriever("wiki17").provenance()

    assert provenance == {
        "backend": "wiki17-bm25s",
        "gepa_artifact_commit": "a924c2045b6f000d2d23ea3b8f8f16b2c08d9e88",
        "corpus": "wiki.abstracts.2017.jsonl",
        "huggingface_revision": "ef6a5e72a98b47cef31574a400fea8fe149559a3",
        "archive_sha256": "744183e61af986bde9b25c880b59c1502618a8b673671e189cbc0ee684fceb42",
        "archive_size": 608_448_121,
        "corpus_size": 1_780_746_240,
        "document_count": 5_233_330,
        "bm25s_version": "0.2.12",
        "pystemmer_version": "2.2.0.3",
        "jax_version": "0.6.0",
        "k1": 0.9,
        "b": 0.4,
        "stopwords": "en",
        "stemmer": "PyStemmer english",
        "retrieval_threads": 1,
    }


def test_dependency_validation_rejects_bm25_version_drift(monkeypatch, tmp_path) -> None:
    """Refuse rankings from a BM25S version other than the paper artifact's.

    Args:
        monkeypatch: Pytest fixture used to simulate dependency drift.
        tmp_path: Pytest directory used to instantiate the retriever.
    """
    install_fake_dependencies(monkeypatch)
    drifted_versions = {
        "bm25s": "0.2.13",
        "PyStemmer": wiki17_bm25.WIKI17_PYSTEMMER_VERSION,
        "jax": wiki17_bm25.WIKI17_JAX_VERSION,
    }
    monkeypatch.setattr(wiki17_bm25, "version", drifted_versions.__getitem__)

    with pytest.raises(wiki17_bm25.Wiki17PreparationError, match="bm25s==0.2.12"):
        wiki17_bm25.Wiki17BM25Retriever(tmp_path)._require_dependencies()
