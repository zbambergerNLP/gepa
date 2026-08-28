"""Frozen Wiki-2017 BM25 retrieval used by the GEPA paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import threading
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from examples.common.wikipedia import WikipediaPassage

try:
    import bm25s  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by the runtime dependency check
    bm25s = None

try:
    import Stemmer  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by the runtime dependency check
    Stemmer = None

try:
    from diskcache import Cache  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by the runtime dependency check
    Cache = None


GEPA_ARTIFACT_COMMIT = "a924c2045b6f000d2d23ea3b8f8f16b2c08d9e88"
WIKI17_HF_REVISION = "ef6a5e72a98b47cef31574a400fea8fe149559a3"
WIKI17_ARCHIVE_NAME = "wiki.abstracts.2017.tar.gz"
WIKI17_CORPUS_NAME = "wiki.abstracts.2017.jsonl"
WIKI17_ARCHIVE_URL = f"https://huggingface.co/dspy/cache/resolve/{WIKI17_HF_REVISION}/{WIKI17_ARCHIVE_NAME}"
WIKI17_ARCHIVE_SHA256 = "744183e61af986bde9b25c880b59c1502618a8b673671e189cbc0ee684fceb42"
WIKI17_ARCHIVE_SIZE = 608_448_121
WIKI17_CORPUS_SIZE = 1_780_746_240
WIKI17_DOCUMENT_COUNT = 5_233_330
WIKI17_BM25_VERSION = "0.2.12"
WIKI17_PYSTEMMER_VERSION = "2.2.0.3"
WIKI17_JAX_VERSION = "0.6.0"
WIKI17_K1 = 0.9
WIKI17_B = 0.4
WIKI17_INDEX_DIR_NAME = "bm25s_retriever"
WIKI17_MANIFEST_NAME = "manifest.json"
DEFAULT_WIKI17_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gepa" / "wiki17"


class Wiki17PreparationError(RuntimeError):
    """Signal that the frozen corpus or its BM25 index is incomplete."""


class Wiki17BM25Retriever:
    """Search the paper artifact's frozen 2017 Wikipedia abstract corpus."""

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_WIKI17_ROOT) -> None:
        """Configure lazy loading from one prepared corpus directory.

        Args:
            root: Directory containing the verified corpus, BM25S index, query
                cache, and preparation manifest.
        """
        self.root = Path(root).expanduser().resolve()
        self.archive_path = self.root / WIKI17_ARCHIVE_NAME
        self.corpus_path = self.root / WIKI17_CORPUS_NAME
        self.index_path = self.root / WIKI17_INDEX_DIR_NAME
        self.manifest_path = self.root / WIKI17_MANIFEST_NAME
        self.cache_path = self.root / "retriever_cache"
        self._initialize_lock = threading.Lock()
        self._retriever: Any | None = None
        self._stemmer: Any | None = None
        self._corpus: list[str] | None = None
        self._cache: Any | None = None

    def prepare(self) -> dict[str, Any]:
        """Download, verify, extract, and index the immutable corpus.

        Returns:
            The persisted preparation manifest.

        Raises:
            Wiki17PreparationError: A dependency, archive, corpus, or index
                fails validation.
        """
        self._require_dependencies()
        assert bm25s is not None
        assert Stemmer is not None
        self.root.mkdir(parents=True, exist_ok=True)
        prepared_manifest = self._prepared_manifest()
        if prepared_manifest is not None:
            return prepared_manifest

        self._ensure_archive()
        self._ensure_corpus()
        corpus = self._load_corpus()
        if len(corpus) != WIKI17_DOCUMENT_COUNT:
            raise Wiki17PreparationError(
                f"Expected {WIKI17_DOCUMENT_COUNT} Wiki-2017 documents, found {len(corpus)} in {self.corpus_path}."
            )

        temporary_index = self.root / f".{WIKI17_INDEX_DIR_NAME}.building"
        if temporary_index.exists():
            shutil.rmtree(temporary_index)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=Stemmer.Stemmer("english"))
        retriever = bm25s.BM25(k1=WIKI17_K1, b=WIKI17_B)
        retriever.index(corpus_tokens)
        retriever.save(temporary_index)
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        temporary_index.replace(self.index_path)

        manifest = self._expected_manifest()
        temporary_manifest = self.manifest_path.with_suffix(".json.part")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_manifest.replace(self.manifest_path)
        return manifest

    def search(self, query: str, limit: int) -> list[WikipediaPassage]:
        """Return BM25-ranked abstracts with artifact-compatible deduplication.

        Args:
            query: Retrieval query passed to the artifact tokenizer unchanged.
            limit: Maximum number of unique abstracts to return.

        Returns:
            Ranked Wikipedia abstracts converted to the shared passage type.

        Raises:
            Wiki17PreparationError: The corpus has not been prepared or the
                required retrieval dependencies are unavailable.
        """
        if not query or limit <= 0:
            return []
        self._initialize()
        assert bm25s is not None
        assert self._cache is not None
        cache_key = (query, limit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return [WikipediaPassage(title=title, text=text) for title, text in cached]

        assert self._retriever is not None
        assert self._stemmer is not None
        assert self._corpus is not None
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self._stemmer, show_progress=False)
        results, scores = self._retriever.retrieve(
            query_tokens,
            k=limit,
            n_threads=1,
            show_progress=False,
        )
        ranked = {
            self._corpus[int(document_index)]: float(score)
            for document_index, score in zip(results[0], scores[0], strict=False)
        }
        passages: list[WikipediaPassage] = []
        for raw_passage in list(ranked)[:limit]:
            title, separator, text = raw_passage.partition(" | ")
            if not separator:
                title = raw_passage
                text = ""
            passages.append(WikipediaPassage(title=title, text=text))
        self._cache.set(cache_key, [(passage.title, passage.text) for passage in passages])
        return passages

    def provenance(self) -> dict[str, Any]:
        """Describe the material retrieval settings for run identity.

        Returns:
            JSON-serializable frozen-corpus, tokenizer, and BM25 metadata.
        """
        return {
            "backend": "wiki17-bm25s",
            "gepa_artifact_commit": GEPA_ARTIFACT_COMMIT,
            "corpus": WIKI17_CORPUS_NAME,
            "huggingface_revision": WIKI17_HF_REVISION,
            "archive_sha256": WIKI17_ARCHIVE_SHA256,
            "archive_size": WIKI17_ARCHIVE_SIZE,
            "corpus_size": WIKI17_CORPUS_SIZE,
            "document_count": WIKI17_DOCUMENT_COUNT,
            "bm25s_version": WIKI17_BM25_VERSION,
            "pystemmer_version": WIKI17_PYSTEMMER_VERSION,
            "jax_version": WIKI17_JAX_VERSION,
            "k1": WIKI17_K1,
            "b": WIKI17_B,
            "stopwords": "en",
            "stemmer": "PyStemmer english",
            "retrieval_threads": 1,
        }

    def _initialize(self) -> None:
        """Load the verified index, corpus strings, stemmer, and query cache.

        Raises:
            Wiki17PreparationError: The prepared state is absent or invalid.
        """
        if self._retriever is not None:
            return
        with self._initialize_lock:
            if self._retriever is not None:
                return
            self._require_dependencies()
            assert bm25s is not None
            assert Stemmer is not None
            assert Cache is not None
            if self._prepared_manifest() is None:
                raise Wiki17PreparationError(
                    f"Wiki-2017 is not prepared under {self.root}. Run "
                    f"`python -m examples.common.wiki17_bm25 prepare --root {self.root}` on an Internet-enabled host."
                )
            retriever = bm25s.BM25.load(self.index_path)
            stemmer = Stemmer.Stemmer("english")
            corpus = self._load_corpus()
            if len(corpus) != WIKI17_DOCUMENT_COUNT:
                raise Wiki17PreparationError(
                    f"Expected {WIKI17_DOCUMENT_COUNT} Wiki-2017 documents, found {len(corpus)} in {self.corpus_path}."
                )
            self._retriever = retriever
            self._stemmer = stemmer
            self._corpus = corpus
            self._cache = Cache(str(self.cache_path))

    def _require_dependencies(self) -> None:
        """Validate the benchmark-only retrieval dependency set.

        Raises:
            Wiki17PreparationError: BM25S, PyStemmer, diskcache, or a pinned
                retrieval dependency version is unavailable.
        """
        missing = []
        if bm25s is None:
            missing.append("bm25s")
        if Stemmer is None:
            missing.append("PyStemmer")
        if Cache is None:
            missing.append("diskcache")
        if missing:
            raise Wiki17PreparationError(
                "Missing Wiki-2017 dependencies: "
                + ", ".join(missing)
                + ". Install them with `uv sync --extra wiki17`."
            )
        pinned_versions = {
            "bm25s": WIKI17_BM25_VERSION,
            "PyStemmer": WIKI17_PYSTEMMER_VERSION,
            "jax": WIKI17_JAX_VERSION,
        }
        for package, expected_version in pinned_versions.items():
            try:
                installed_version = version(package)
            except PackageNotFoundError as exc:
                raise Wiki17PreparationError(f"Could not determine the installed {package} version.") from exc
            if installed_version != expected_version:
                raise Wiki17PreparationError(
                    f"Wiki-2017 requires {package}=={expected_version}; found {installed_version}."
                )

    def _prepared_manifest(self) -> dict[str, Any] | None:
        """Read a complete manifest only when every prepared artifact exists.

        Returns:
            The verified manifest, or ``None`` for missing or stale state.
        """
        if (
            not self.manifest_path.is_file()
            or not self.corpus_path.is_file()
            or not self.index_path.is_dir()
            or not any(self.index_path.iterdir())
        ):
            return None
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest != self._expected_manifest():
            return None
        if self.corpus_path.stat().st_size != WIKI17_CORPUS_SIZE:
            return None
        return manifest

    def _expected_manifest(self) -> dict[str, Any]:
        """Build the canonical manifest written after successful indexing.

        Returns:
            Stable manifest content excluding machine-specific storage paths.
        """
        return {"schema_version": 1, **self.provenance()}

    def _ensure_archive(self) -> None:
        """Download and hash-check the pinned corpus archive when absent.

        Raises:
            Wiki17PreparationError: An existing or downloaded archive has the
                wrong size or digest.
        """
        if self.archive_path.is_file():
            self._validate_archive(self.archive_path)
            return
        partial_path = self.archive_path.with_suffix(self.archive_path.suffix + ".part")
        hasher = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(WIKI17_ARCHIVE_URL) as response, open(partial_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
        except Exception as exc:
            partial_path.unlink(missing_ok=True)
            raise Wiki17PreparationError(f"Could not download {WIKI17_ARCHIVE_URL}: {exc}") from exc
        if size != WIKI17_ARCHIVE_SIZE or hasher.hexdigest() != WIKI17_ARCHIVE_SHA256:
            partial_path.unlink(missing_ok=True)
            raise Wiki17PreparationError(
                f"Downloaded Wiki-2017 archive failed integrity validation: size={size}, sha256={hasher.hexdigest()}."
            )
        partial_path.replace(self.archive_path)

    def _validate_archive(self, archive_path: Path) -> None:
        """Verify the byte size and SHA-256 of an existing archive.

        Args:
            archive_path: Candidate archive to validate.

        Raises:
            Wiki17PreparationError: The archive differs from the pinned source.
        """
        size = archive_path.stat().st_size
        hasher = hashlib.sha256()
        with open(archive_path, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if size != WIKI17_ARCHIVE_SIZE or digest != WIKI17_ARCHIVE_SHA256:
            raise Wiki17PreparationError(
                f"Existing Wiki-2017 archive failed integrity validation: size={size}, sha256={digest}."
            )

    def _ensure_corpus(self) -> None:
        """Extract only the expected JSONL member from the verified archive.

        Raises:
            Wiki17PreparationError: The member is missing or its extracted size
                differs from the pinned artifact.
        """
        if self.corpus_path.is_file() and self.corpus_path.stat().st_size == WIKI17_CORPUS_SIZE:
            return
        partial_path = self.corpus_path.with_suffix(self.corpus_path.suffix + ".part")
        try:
            with tarfile.open(self.archive_path, "r:gz") as archive:
                members = [member for member in archive.getmembers() if Path(member.name).name == WIKI17_CORPUS_NAME]
                if len(members) != 1 or not members[0].isfile():
                    raise Wiki17PreparationError(
                        f"Expected one regular {WIKI17_CORPUS_NAME} member in {self.archive_path}."
                    )
                source = archive.extractfile(members[0])
                if source is None:
                    raise Wiki17PreparationError(f"Could not read {WIKI17_CORPUS_NAME} from {self.archive_path}.")
                with source, open(partial_path, "wb") as output:
                    shutil.copyfileobj(source, output)
        except Wiki17PreparationError:
            partial_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            partial_path.unlink(missing_ok=True)
            raise Wiki17PreparationError(f"Could not extract {self.archive_path}: {exc}") from exc
        extracted_size = partial_path.stat().st_size
        if extracted_size != WIKI17_CORPUS_SIZE:
            partial_path.unlink(missing_ok=True)
            raise Wiki17PreparationError(
                f"Extracted Wiki-2017 corpus has size {extracted_size}; expected {WIKI17_CORPUS_SIZE}."
            )
        partial_path.replace(self.corpus_path)

    def _load_corpus(self) -> list[str]:
        """Load corpus rows using the artifact's exact title/text rendering.

        Returns:
            Ordered ``"title | abstract"`` strings used as BM25 documents.

        Raises:
            Wiki17PreparationError: A JSONL row is malformed.
        """
        corpus: list[str] = []
        _line_number = 0
        try:
            with open(self.corpus_path, encoding="utf-8") as source:
                for _line_number, line in enumerate(source, 1):
                    row = json.loads(line)
                    title = row["title"]
                    text = row["text"]
                    if not isinstance(title, str) or not isinstance(text, list):
                        raise TypeError("title must be a string and text must be a list")
                    corpus.append(f"{title} | {' '.join(str(sentence) for sentence in text)}")
        except Exception as exc:
            raise Wiki17PreparationError(f"Malformed Wiki-2017 corpus row near line {_line_number}: {exc}") from exc
        return corpus


def main() -> None:
    """Prepare or verify the frozen retriever from the command line.

    Raises:
        Wiki17PreparationError: The requested preparation or verification
            cannot establish the exact artifact state.
    """
    parser = argparse.ArgumentParser(description="Prepare the GEPA paper's frozen Wiki-2017 BM25 retriever")
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI17_ROOT)
    args = parser.parse_args()

    retriever = Wiki17BM25Retriever(args.root)
    if args.command == "prepare":
        manifest = retriever.prepare()
    else:
        retriever._require_dependencies()
        manifest = retriever._prepared_manifest()
        if manifest is None:
            raise Wiki17PreparationError(f"Wiki-2017 is not prepared under {retriever.root}.")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
