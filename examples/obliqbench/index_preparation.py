"""Checkpoint local corpus embedding work without changing the retrieval contract."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from examples.common.artifacts import atomic_json
from examples.obliqbench.benchmark_settings import EMBEDDING_DIMENSION
from examples.obliqbench.retrieval import DenseRetriever, file_sha256
from examples.obliqbench.utils import Corpus, digest


def prepare_index(
    corpus: Corpus,
    encoder: Any,
    cache_dir: Path,
    contract: dict[str, Any],
    batch_size: int,
    *,
    checkpoint_batches: int = 256,
) -> DenseRetriever:
    """Resume verified local embedding chunks and publish a standard immutable index."""
    if batch_size < 1 or checkpoint_batches < 1:
        raise ValueError("Embedding and checkpoint batch sizes must be positive")
    identity = {"corpus": corpus.identity, "ordered_ids": digest(corpus.ids), "retriever": contract}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (digest(identity) + ".npy")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and path.with_suffix(".json").exists():
            return DenseRetriever(corpus, encoder, cache_dir, contract, batch_size)
        _prepare(corpus, encoder, path, identity, batch_size, checkpoint_batches)
        return DenseRetriever(corpus, encoder, cache_dir, contract, batch_size)


def _prepare(corpus, encoder, path, identity, batch_size, checkpoint_batches):
    """Commit vector bytes before advancing the verified prefix in the sidecar."""
    progress_path = path.with_suffix(".progress.json")
    temporary = path.with_suffix(".partial.npy")
    shape = (len(corpus.ids), EMBEDDING_DIMENSION)
    if path.with_suffix(".json").exists():
        raise ValueError(f"Incomplete OBLIQ embedding index: {path}")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("identity") != identity or progress.get("batch_size") != batch_size:
            raise ValueError("Embedding checkpoint identity drift")
        current = path if path.exists() else temporary
        array = np.load(current, mmap_mode="r+", allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError("Embedding checkpoint shape or dtype drift")
        offset = 0
        for chunk in progress["chunks"]:
            stop = chunk["stop"]
            if not offset < stop <= len(corpus.ids):
                raise ValueError("Invalid embedding checkpoint range")
            if hashlib.sha256(array[offset:stop].tobytes()).hexdigest() != chunk["sha256"]:
                raise ValueError("Embedding checkpoint checksum drift")
            offset = stop
        if offset != progress["completed_documents"]:
            raise ValueError("Invalid embedding checkpoint count")
        if path.exists() and offset != len(corpus.ids):
            raise ValueError("Incomplete published embedding index")
    else:
        if path.exists() or temporary.exists():
            raise ValueError("Embedding bytes exist without a verified checkpoint")
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        offset = 0
        progress = {"identity": identity, "batch_size": batch_size, "completed_documents": 0, "chunks": []}
        atomic_json(progress_path, progress)

    rows = islice(corpus.documents(), offset, None)
    committed = offset
    while offset < len(corpus.ids):
        texts = [row["text"] for row in islice(rows, batch_size)]
        if not texts or offset + len(texts) > len(corpus.ids):
            raise ValueError("Incomplete corpus embedding input")
        values = np.asarray(encoder.encode(texts, query=False), dtype=np.float32)
        if values.shape != (len(texts), EMBEDDING_DIMENSION) or not np.isfinite(values).all():
            raise ValueError("Incomplete or malformed embedding response")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding model returned zero vectors")
        array[offset : offset + len(texts)] = values / norms
        offset += len(texts)
        if offset - committed >= checkpoint_batches * batch_size or offset == len(corpus.ids):
            array.flush()
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            progress["chunks"].append(
                {"stop": offset, "sha256": hashlib.sha256(array[committed:offset].tobytes()).hexdigest()}
            )
            progress["completed_documents"] = offset
            atomic_json(progress_path, progress)
            committed = offset
            print(f"{path.stem}: {offset}/{len(corpus.ids)} corpus documents checkpointed", file=sys.stderr, flush=True)
    if next(rows, None) is not None:
        raise ValueError("Corpus has more documents than its pinned IDs")
    del array
    if not path.exists():
        temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {"identity": identity, "sha256": file_sha256(path)})
    progress_path.unlink()
