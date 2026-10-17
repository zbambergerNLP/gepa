"""Load the official files and enforce immutable records and group-disjoint splits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from examples.obliqbench.benchmark_settings import (
    CORPUS_COUNTS,
    DATASET,
    DATASET_REVISION,
    EXCLUSION_SUBSETS,
    POOLED_SUBSETS,
    QUERY_COUNTS,
    SPLIT_POLICY,
    SPLIT_SEED,
    SUBSETS,
)

SOURCE_PATH = Path(__file__).with_name("sources.json")
RECORD_PATH = Path(__file__).with_name("records.jsonl")


def digest(value: Any) -> str:
    """Hash canonical JSON without losing Unicode query content."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path):
    """Yield validated raw retrieval records in file order."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip() for key in ("_id", "text")
            ):
                raise ValueError(f"Malformed retrieval record at {path}:{line_number}")
            yield row


def verify_file(path: Path, identity: dict[str, Any]) -> None:
    """Reject changed, partial, or LFS-pointer files before they can enter a run."""
    if not path.is_file() or path.stat().st_size != identity["size"]:
        raise ValueError(f"Missing or incomplete pinned OBLIQ file: {path}; run examples.obliqbench.prepare.")
    algorithm = identity["algorithm"]
    if algorithm not in {"sha256", "git_blob_sha1"}:
        raise ValueError(f"Unsupported file hash: {algorithm}")
    hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git_blob_sha1":
        hasher.update(f"blob {identity['size']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != identity["digest"]:
        raise ValueError(f"Pinned OBLIQ content changed: {path}")


def load_sources() -> dict[str, Any]:
    """Read the reviewed upstream object identities."""
    source = json.loads(SOURCE_PATH.read_text())
    if source["dataset"] != DATASET or source["revision"] != DATASET_REVISION:
        raise ValueError("OBLIQ source revision drift")
    return source


def prepare_data(root: Path, subsets: list[str], *, metadata_only: bool = False) -> None:
    """Download only requested official files and atomically verify their pinned hashes."""
    source = load_sources()
    for relative, identity in source["files"].items():
        if not any(relative.startswith(SUBSETS[name] + "/") for name in subsets):
            continue
        if metadata_only and "/corpus/" in relative:
            continue
        path = root / relative
        if path.exists():
            verify_file(path, identity)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{quote(relative)}"
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
            verify_file(temporary, identity)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_qrels(path: Path, query_ids: set[str]) -> dict[str, dict[str, int]]:
    """Preserve every released relevance grade, accepting both official header spellings."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) not in (["query-id", "corpus-id", "score"], ["query_id", "corpus_id", "score"]):
            raise ValueError(f"Unexpected qrels header: {path}")
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"Malformed qrels row: {path}")
            query_id, doc_id, grade_text = row
            grade = int(grade_text)
            if query_id not in query_ids or not doc_id or grade < 0 or doc_id in qrels[query_id]:
                raise ValueError(f"Unknown query, duplicate, or invalid qrel: {path}")
            qrels[query_id][doc_id] = grade
    if set(qrels) != query_ids or any(not any(g > 0 for g in values.values()) for values in qrels.values()):
        raise ValueError(f"Every OBLIQ query must have positive qrels: {path}")
    return dict(qrels)


def load_records(root: Path, subset: str) -> list[dict[str, Any]]:
    """Read query-only inputs with evaluator-only labels and official exclusions."""
    source = load_sources()
    prefix = SUBSETS[subset] + "/queries+qrels/"
    for relative, identity in source["files"].items():
        if relative.startswith(prefix):
            verify_file(root / relative, identity)
    directory = root / prefix
    queries = list(read_jsonl(directory / "queries.jsonl"))
    query_ids = {query["_id"] for query in queries}
    if len(query_ids) != len(queries) or len(queries) != QUERY_COUNTS[subset]:
        raise ValueError(f"Duplicate queries or incorrect query count: {subset}")
    gold = read_qrels(directory / "qrels.tsv", query_ids)
    pooled = read_qrels(directory / "qrels_pool.tsv", query_ids) if subset in POOLED_SUBSETS else {}
    excluded = (
        json.loads((directory / "per_query_excluded_ids.json").read_text()) if subset in EXCLUSION_SUBSETS else {}
    )
    if subset in EXCLUSION_SUBSETS and set(excluded) != query_ids:
        raise ValueError(f"Missing or extra exclusion lists: {subset}")
    records = []
    for query in queries:
        query_id = query["_id"]
        exclusions = excluded.get(query_id, [])
        if (
            not isinstance(exclusions, list)
            or any(not isinstance(item, str) for item in exclusions)
            or len(set(exclusions)) != len(exclusions)
        ):
            raise ValueError(f"Invalid exclusions for {subset}/{query_id}")
        if any(gold[query_id].get(doc, 0) > 0 or pooled.get(query_id, {}).get(doc, 0) > 0 for doc in exclusions):
            raise ValueError(f"Excluded document is labeled relevant: {subset}/{query_id}")
        records.append(
            {
                "id": f"{subset}/{query_id}",
                "subset": subset,
                "query_id": query_id,
                "query": query["text"],
                "gold_qrels": gold[query_id],
                "pooled_qrels": pooled.get(query_id),
                "excluded_ids": sorted(exclusions),
            }
        )
    return records


def split_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group overlapping relevance/exclusion sets, then deterministically allocate whole groups."""
    parent = {record["id"]: record["id"] for record in records}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners: dict[tuple[str, str], str] = {}
    for record in records:
        documents = set(record["excluded_ids"])
        for qrels in (record["gold_qrels"], record["pooled_qrels"] or {}):
            documents.update(doc for doc, grade in qrels.items() if grade > 0)
        keys = [("document", doc) for doc in sorted(documents)] + [("query", record["query"].strip())]
        for key in keys:
            if key in owners:
                parent[root(record["id"])] = root(owners[key])
            else:
                owners[key] = record["id"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[root(record["id"])].append(record)
    if len(groups) < 3:
        raise ValueError("At least three independent query groups are required for train/val/test")
    ordered_groups = sorted(
        groups.values(), key=lambda group: (-len(group), digest([SPLIT_SEED, sorted(r["id"] for r in group)]))
    )
    targets = {"train": len(records) * 0.6, "val": len(records) * 0.2, "test": len(records) * 0.2}
    sizes = dict.fromkeys(targets, 0)
    assignments = {}
    for group in ordered_groups:
        split = max(targets, key=lambda name: targets[name] - sizes[name])
        group_id = digest(sorted(record["id"] for record in group))
        for record in group:
            assignments[record["id"]] = {**record, "group_id": group_id, "split": split}
        sizes[split] += len(group)
    if not all(sizes.values()):
        raise ValueError("Group-aware split produced an empty partition")
    return [assignments[record["id"]] for record in records]


@dataclass(frozen=True)
class Corpus:
    """Keep a verified corpus on disk instead of retaining gigabytes of text."""

    path: Path
    ids: tuple[str, ...]
    identity: dict[str, Any]

    def documents(self):
        """Stream document text in the pinned source order."""
        yield from read_jsonl(self.path)


@dataclass(frozen=True)
class ObliqData:
    """Hold exact splits, corpus handles, and the material source identity."""

    splits: dict[str, list[dict[str, Any]]]
    corpora: dict[str, Corpus]
    source: dict[str, Any]


def frozen_records(root: Path, subset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check query data against the reviewed ordered record and partition manifest."""
    expected = [json.loads(line) for line in RECORD_PATH.read_text().splitlines()]
    records = split_records(load_records(root, subset))
    actual = [{"id": r["id"], "sha256": digest(r), "split": r["split"], "group_id": r["group_id"]} for r in records]
    if actual != [r for r in expected if r["id"].startswith(subset + "/")]:
        raise ValueError(f"Ordered OBLIQ records or splits drifted: {subset}")
    return records, actual


def load_data(root: Path, subsets: list[str]) -> ObliqData:
    """Reject source, ordering, membership, grouping, or record drift before model calls."""
    if not subsets or len(set(subsets)) != len(subsets) or any(name not in SUBSETS for name in subsets):
        raise ValueError("Choose distinct official OBLIQ subsets")
    sources = load_sources()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    corpora = {}
    identities = []
    for subset in SUBSETS:
        if subset not in subsets:
            continue
        records, actual = frozen_records(root, subset)
        relative = SUBSETS[subset] + "/corpus/corpus.jsonl"
        path = root / relative
        verify_file(path, sources["files"][relative])
        ids = tuple(row["_id"] for row in read_jsonl(path))
        known_ids = set(ids)
        if len(ids) != CORPUS_COUNTS[subset] or len(known_ids) != len(ids):
            raise ValueError(f"Wrong corpus count or duplicate document IDs: {subset}")
        for record in records:
            referenced = set(record["gold_qrels"]) | set(record["pooled_qrels"] or {}) | set(record["excluded_ids"])
            if not referenced <= known_ids:
                raise ValueError(f"Judgment or exclusion references an absent corpus document: {record['id']}")
            splits[record["split"]].append(record)
        corpora[subset] = Corpus(path, ids, sources["files"][relative])
        identities.extend(actual)
    identity = {
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "split_policy": SPLIT_POLICY,
        "split_seed": SPLIT_SEED,
        "subsets": list(corpora),
        "ordered_records": identities,
        "source_files": {
            name: value
            for name, value in sources["files"].items()
            if any(name.startswith(SUBSETS[s] + "/") for s in corpora)
        },
    }
    return ObliqData(splits, corpora, identity)
