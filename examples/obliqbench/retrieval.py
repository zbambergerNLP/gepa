"""Fixed single-stage retrievers; neither backend receives queries' relevance judgments."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from examples.obliqbench.benchmark_settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RUNTIME_PINS,
)
from examples.obliqbench.utils import Corpus, digest


class Retriever(Protocol):
    """Expose only retrieval text and excluded IDs to the fixed backend."""

    ids: tuple[str, ...]

    def search(self, query: str, excluded: set[str], k: int) -> list[tuple[str, float]]: ...


def require_version(package: str) -> str:
    """Fail early when a benchmark dependency differs from its reviewed pin."""
    installed = version(package)
    if installed != RUNTIME_PINS[package]:
        raise RuntimeError(f"OBLIQ requires {package}=={RUNTIME_PINS[package]}; found {installed}")
    return installed


def retrieval_contract(backend: str, device: str, batch_size: int) -> dict[str, Any]:
    """Freeze numerical runtime and retrieval settings before building an index."""
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    packages = ["numpy", "pytrec-eval-terrier"]
    if backend == "qwen":
        packages += ["sentence-transformers", "transformers", "torch"]
        config = {
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "dimension": EMBEDDING_DIMENSION,
            "max_sequence_length": EMBEDDING_MAX_LENGTH,
            "document_prompt": "",
            "query_prompt": "checkpoint_query_prompt",
            "similarity": "cosine",
            "dtype": "float32",
            "device": device,
            "batch_size": batch_size,
            "attention": "eager",
            "padding_side": "left",
        }
    elif backend == "bm25":
        packages += ["rank-bm25"]
        config = {
            "class": "BM25Okapi",
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "tokenizer": "unicode_word_regex_lower_v1",
        }
    else:
        raise ValueError(f"Unsupported retriever: {backend}")
    return {
        "backend": backend,
        "implementation_sha256": file_sha256(Path(__file__)),
        "settings": config,
        "tie_break": "corpus_order",
        "packages": {p: require_version(p) for p in packages},
    }


def ranked_results(ids: tuple[str, ...], scores: Any, excluded: set[str], k: int) -> list[tuple[str, float]]:
    """Mask before top-k selection and preserve source order for tied scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ValueError("Retriever returned invalid or incomplete document scores")
    if k < 1 or not excluded <= set(ids):
        raise ValueError("Invalid retrieval budget or unknown excluded document")
    positions = np.argsort(-scores, kind="stable")
    result = []
    for index in positions:
        doc_id = ids[index]
        if doc_id not in excluded:
            result.append((doc_id, float(scores[index])))
            if len(result) == k:
                break
    return result


class BM25Retriever:
    """Use the paper's rank-bm25 baseline with explicitly recorded tokenization."""

    def __init__(self, corpus: Corpus):
        require_version("rank-bm25")
        self.ids = corpus.ids
        self.index = importlib.import_module("rank_bm25").BM25Okapi(
            [self.tokens(row["text"]) for row in corpus.documents()]
        )

    @staticmethod
    def tokens(text: str) -> list[str]:
        """Use one fixed Unicode tokenizer for documents and queries."""
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, excluded: set[str], k: int) -> list[tuple[str, float]]:
        return ranked_results(self.ids, self.index.get_scores(self.tokens(query)), excluded, k)


class QwenEncoder:
    """Encode with the separate, immutable embedding checkpoint from the paper."""

    def __init__(self, device: str):
        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        self.model = sentence_transformers.SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device=device,
            trust_remote_code=False,
            model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
            processor_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = EMBEDDING_MAX_LENGTH
        if not self.model.prompts.get("query"):
            raise ValueError("Pinned embedding checkpoint is missing its query prompt")

    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        """Apply the checkpoint's fixed query instruction only on the query side."""
        kwargs = {"prompt_name": "query"} if query else {"prompt": ""}
        return self.model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )


def file_sha256(path: Path) -> str:
    """Hash an embedding cache without reading the entire array into RAM."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class DenseRetriever:
    """Cache normalized corpus embeddings, then perform exact cosine retrieval."""

    def __init__(self, corpus: Corpus, encoder: Any, cache_dir: Path, contract: dict[str, Any], batch_size: int):
        self.ids = corpus.ids
        self.encoder = encoder
        identity = {"corpus": corpus.identity, "ordered_ids": digest(self.ids), "retriever": contract}
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / (digest(identity) + ".npy")
        metadata_path = path.with_suffix(".json")
        if path.exists() or metadata_path.exists():
            if not path.exists() or not metadata_path.exists():
                raise ValueError(f"Incomplete OBLIQ embedding index: {path}")
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("identity") != identity or metadata.get("sha256") != file_sha256(path):
                raise ValueError(f"Embedding cache identity or content drift: {path}")
        else:
            temporary = path.with_suffix(".partial")
            try:
                array = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=np.float32, shape=(len(self.ids), EMBEDDING_DIMENSION)
                )
                offset = 0
                pending = []
                for row in corpus.documents():
                    pending.append(row["text"])
                    if len(pending) == batch_size:
                        array[offset : offset + len(pending)] = self._encode(pending, query=False)
                        offset += len(pending)
                        pending = []
                if pending:
                    array[offset : offset + len(pending)] = self._encode(pending, query=False)
                    offset += len(pending)
                if offset != len(self.ids):
                    raise ValueError("Incomplete corpus embedding run")
                array.flush()
                del array
                temporary.replace(path)
                metadata_path.write_text(
                    json.dumps({"identity": identity, "sha256": file_sha256(path)}, sort_keys=True) + "\n"
                )
            finally:
                temporary.unlink(missing_ok=True)
        self.embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.embeddings.shape != (len(self.ids), EMBEDDING_DIMENSION) or self.embeddings.dtype != np.float32:
            raise ValueError("Embedding index has an invalid shape or dtype")

    def _encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        values = np.asarray(self.encoder.encode(texts, query=query), dtype=np.float32)
        if values.shape != (len(texts), EMBEDDING_DIMENSION) or not np.isfinite(values).all():
            raise ValueError("Incomplete or malformed embedding response")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding model returned zero vectors")
        return values / norms

    def search(self, query: str, excluded: set[str], k: int) -> list[tuple[str, float]]:
        vector = self._encode([query], query=True)[0]
        return ranked_results(self.ids, self.embeddings @ vector, excluded, k)
