"""Load frozen rows and keep related source examples in the same derived split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from decision_bench.data import decode_storage_row
from decision_bench.schemas import DecisionExample
from huggingface_hub import hf_hub_download

from examples.decisionbench.benchmark_settings import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_ROWS,
    DATASET_SHA256,
    PINNED_SPLITS,
    SPLIT_POLICY,
    SPLIT_SEED,
)


def digest(value: Any) -> str:
    """Hash JSON values without depending on whitespace or mapping insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash an artifact without reading the complete file into memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def decode_record(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve upstream metadata and provenance while validating the official schema."""
    example = DecisionExample.model_validate(decode_storage_row(row))
    task_id = "/".join((example.domain, example.family, example.primitive.value, example.task_name))
    if row["task_id"] != task_id or row["candidate_count"] != len(example.candidates):
        raise ValueError(f"DecisionBench indexing metadata changed: {example.row_id}")
    return {
        "id": example.row_id,
        "row_id": example.row_id,
        "task_id": task_id,
        "task_name": example.task_name,
        "family": example.family,
        "domain": example.domain,
        "primitive": example.primitive.value,
        "candidate_count": len(example.candidates),
        "reasoning_required": row["reasoning_required"],
        "reasoning_type": row["reasoning_type"],
        "upstream_row_sha256": digest(row),
        "example": example.model_dump(mode="json"),
    }


def source_keys(record: dict[str, Any]) -> set[str]:
    """Link repeated states, paraphrase parents, generated requests, and source rows."""
    example = record["example"]
    source = example["source"]
    keys = {"state:" + digest(example["state"])}
    for name in ("seed_sha256", "generation_request_sha256"):
        if source.get(name):
            keys.add(name + ":" + str(source[name]))
    parent = source.get("decisionbench_paraphrase_v2", {})
    if parent.get("parent_row_sha256"):
        keys.add("parent:" + str(parent["parent_row_sha256"]))
    if source.get("source_id") is not None and source.get("repo"):
        keys.add(
            "source_id:" + digest([source["repo"], source.get("revision"), source.get("split"), source["source_id"]])
        )
    original = source.get("source", {})
    for index, source_id in enumerate(original.get("source_ids", [])):
        namespace = [original.get("dataset"), original.get("revision"), original.get("split")]
        if original.get("dataset") == "Tevatron/msmarco-passage":
            namespace.append(index)
        keys.add("original_id:" + digest([namespace, str(source_id)]))
    if original.get("raw_sha256"):
        keys.add("raw:" + digest([original.get("dataset"), original["raw_sha256"]]))
    # Text/state can recur under different targets or paraphrases with distinct raw hashes.
    raw = original.get("raw", {})
    for name in ("text", "state"):
        if raw.get(name):
            keys.add("original_" + name + ":" + digest([original.get("dataset"), raw[name]]))
    if len(keys) == 1:
        raise ValueError(f"Missing supported source lineage for {record['id']}")
    return keys


def partition_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition connected source groups deterministically, independently of run settings."""
    by_id = {record["id"]: record for record in records}
    if not records or len(by_id) != len(records):
        raise ValueError("DecisionBench requires nonempty records with unique row IDs")
    parents = {row_id: row_id for row_id in by_id}

    def find(row_id: str) -> str:
        while parents[row_id] != row_id:
            parents[row_id] = parents[parents[row_id]]
            row_id = parents[row_id]
        return row_id

    seen: dict[str, str] = {}
    for record in records:
        for key in source_keys(record):
            if key in seen:
                left, right = find(record["id"]), find(seen[key])
                parents[max(left, right)] = min(left, right)
            seen[key] = record["id"]
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id in by_id:
        groups[find(row_id)].append(row_id)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
    for ids in groups.values():
        group_id = digest(sorted(ids))
        bucket = int(digest([SPLIT_POLICY, SPLIT_SEED, group_id]), 16) % 10
        split = "train" if bucket < 6 else "val" if bucket < 8 else "test"
        splits[split].extend({**by_id[row_id], "group_id": group_id, "split": split} for row_id in ids)
    for split, rows in splits.items():
        # Prefix limits should cover small tasks instead of taking a source-file prefix.
        tasks: dict[str, deque] = defaultdict(deque)
        for row in sorted(rows, key=lambda row: (digest([SPLIT_SEED, row["id"]]), row["id"])):
            tasks[row["task_id"]].append(row)
        task_order = sorted(tasks, key=lambda task: digest([SPLIT_SEED, task]))
        ordered = []
        while any(tasks.values()):
            for task in task_order:
                if tasks[task]:
                    ordered.append(tasks[task].popleft())
        if not ordered:
            raise ValueError(f"Empty derived {split} split")
        splits[split] = ordered
    return splits


def split_identity(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Fingerprint ordered row contents and group membership for every full split."""
    return {
        name: {
            "count": len(rows),
            "ids": [row["id"] for row in rows],
            "group_ids": [row["group_id"] for row in rows],
            "sha256": digest(rows),
        }
        for name, rows in splits.items()
    }


def load_decisionbench(data_file: Path | None = None, cache_dir: Path | None = None) -> tuple[dict, dict]:
    """Verify the canonical Parquet and return complete derived splits and source identity."""
    path = data_file or Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=DATASET_FILE,
            cache_dir=cache_dir,
        )
    )
    if file_digest(path) != DATASET_SHA256:
        raise ValueError("DecisionBench dataset bytes differ from the pinned canonical artifact")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != DATASET_ROWS:
        raise ValueError("DecisionBench row count changed")
    records = [decode_record(row) for batch in parquet.iter_batches(batch_size=512) for row in batch.to_pylist()]
    splits = partition_records(records)
    identity = split_identity(splits)
    if {
        name: {key: details[key] for key in ("count", "sha256")} for name, details in identity.items()
    } != PINNED_SPLITS:
        raise ValueError("DecisionBench ordered split fingerprints changed")
    source = {
        "repository": DATASET_REPO,
        "revision": DATASET_REVISION,
        "file": DATASET_FILE,
        "sha256": DATASET_SHA256,
        "upstream_split": "eval",
        "compact_fields": False,
        "partition": {"policy": SPLIT_POLICY, "seed": SPLIT_SEED, "official_train_split": False},
        "splits": identity,
    }
    return splits, source
