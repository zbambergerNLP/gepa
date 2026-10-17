"""Verify interrupted corpus preparation preserves exact vectors and refuses drift."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.obliqbench import index_preparation
from examples.obliqbench.benchmark_settings import EMBEDDING_DIMENSION
from examples.obliqbench.index_preparation import prepare_index
from examples.obliqbench.retrieval import DenseRetriever
from examples.obliqbench.utils import Corpus, digest


class Encoder:
    """Produce stable vectors and optionally interrupt a chosen local batch."""

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def encode(self, texts, *, query):
        assert not query
        if len(self.calls) == self.fail_after:
            raise InterruptedError("local preparation interrupted")
        self.calls.append(texts)
        values = np.ones((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for index, text in enumerate(texts):
            values[index, 0] = int(text) + 1
        return values


@pytest.fixture
def corpus(tmp_path):
    rows = [{"_id": str(i), "text": str(i)} for i in range(9)]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return Corpus(path, tuple(row["_id"] for row in rows), {"sha256": digest(rows)})


def test_resume_skips_verified_prefix_and_matches_standard_index(corpus, tmp_path):
    """Retain four committed documents while recomputing an uncommitted batch."""
    directory = tmp_path / "resumed"
    with pytest.raises(InterruptedError):
        prepare_index(corpus, Encoder(fail_after=3), directory, {}, 2, checkpoint_batches=2)
    progress = json.loads(next(directory.glob("*.progress.json")).read_text())
    assert progress["completed_documents"] == 4
    encoder = Encoder()
    resumed = prepare_index(corpus, encoder, directory, {}, 2, checkpoint_batches=2)
    assert encoder.calls == [["4", "5"], ["6", "7"], ["8"]]
    standard = DenseRetriever(corpus, Encoder(), tmp_path / "standard", {}, 2)
    assert Path(resumed.embeddings.filename).read_bytes() == Path(standard.embeddings.filename).read_bytes()
    assert not list(directory.glob("*.partial.npy"))
    assert not list(directory.glob("*.progress.json"))
    prepare_index(corpus, Encoder(fail_after=0), directory, {}, 2)


@pytest.mark.parametrize("damage", ["bytes", "identity", "count"])
def test_resume_rejects_corrupt_checkpoint_before_encoding(corpus, tmp_path, damage):
    directory = tmp_path / "index"
    with pytest.raises(InterruptedError):
        prepare_index(corpus, Encoder(fail_after=2), directory, {}, 2, checkpoint_batches=2)
    progress_path = next(directory.glob("*.progress.json"))
    progress = json.loads(progress_path.read_text())
    if damage == "bytes":
        array = np.load(next(directory.glob("*.partial.npy")), mmap_mode="r+")
        array[0, 0] = 99
        array.flush()
    elif damage == "identity":
        progress["identity"]["retriever"] = {"changed": True}
    else:
        progress["completed_documents"] += 1
    progress_path.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_index(corpus, Encoder(fail_after=0), directory, {}, 2)


def test_recover_interrupted_final_publication(corpus, tmp_path, monkeypatch):
    """Recover the narrow rename-before-metadata window from the complete checkpoint."""
    directory = tmp_path / "index"
    original = index_preparation.atomic_json

    def interrupt_final_metadata(path, document):
        if "sha256" in document:
            raise InterruptedError("metadata not published")
        original(path, document)

    monkeypatch.setattr(index_preparation, "atomic_json", interrupt_final_metadata)
    with pytest.raises(InterruptedError):
        prepare_index(corpus, Encoder(), directory, {}, 2, checkpoint_batches=2)
    assert len(list(directory.glob("*.npy"))) == 1
    assert not list(directory.glob("*.partial.npy"))
    monkeypatch.setattr(index_preparation, "atomic_json", original)
    retriever = prepare_index(corpus, Encoder(fail_after=0), directory, {}, 2)
    path = Path(retriever.embeddings.filename)
    assert json.loads(path.with_suffix(".json").read_text())["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_one_preparer_can_write_an_index(corpus, tmp_path):
    """Reject a competing writer before it can change an in-progress index."""

    class CompetingEncoder(Encoder):
        def encode(self, texts, *, query):
            with pytest.raises(BlockingIOError):
                prepare_index(corpus, Encoder(fail_after=0), tmp_path / "index", {}, 2)
            return super().encode(texts, query=query)

    prepare_index(corpus, CompetingEncoder(), tmp_path / "index", {}, 2, checkpoint_batches=2)
