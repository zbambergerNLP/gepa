"""Stage exact public data and keep evaluator-only fields outside agent records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.common.artifacts import atomic_json, digest
from examples.common.react_v2 import file_sha256
from examples.swebench_live.settings import (
    DATA_REVISION,
    DATASET,
    PARQUET_HASHES,
    PARTITION,
    PREPARED_HASHES,
    image_tag,
)


def partition(full: list[dict], test: list[dict]) -> dict[str, list[dict]]:
    """Hold out the official test IDs and split the remainder reproducibly."""
    test_ids = {row["instance_id"] for row in test}
    if len(test_ids) != len(test):
        raise ValueError("Duplicate SWE-bench Live test IDs")
    available = sorted(
        (row for row in full if row["instance_id"] not in test_ids),
        key=lambda row: hashlib.sha256(row["instance_id"].encode()).hexdigest(),
    )
    if len({row["instance_id"] for row in available}) != len(available):
        raise ValueError("Ambiguous non-test SWE-bench Live task IDs")
    midpoint = len(available) // 2
    if midpoint == 0 or not test:
        raise ValueError("SWE-bench Live requires three nonempty disjoint partitions")
    return {"train": available[:midpoint], "val": available[midpoint:], "test": test}


def prepare_data(root: Path) -> dict:
    """Download immutable Parquets and publish checksummed JSONL and split manifests."""
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    raw = {}
    for name, expected in PARQUET_HASHES.items():
        path = Path(hf_hub_download(DATASET, f"data/{name}", repo_type="dataset", revision=DATA_REVISION))
        if file_sha256(path) != expected:
            raise ValueError(f"Pinned Parquet checksum mismatch: {name}")
        raw.setdefault(name.split("-")[0], []).extend(parquet.read_table(path).to_pylist())
    if {key: len(rows) for key, rows in raw.items()} != {"full": 1888, "test": 1000, "lite": 300, "verified": 500}:
        raise ValueError("Pinned SWE-bench Live split sizes changed")
    splits = partition(raw["full"], raw["test"])
    files = {}
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in {**raw, "train": splits["train"], "val": splits["val"]}.items():
        content = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
        path = data_dir / f"{name}.jsonl"
        expected = hashlib.sha256(content.encode()).hexdigest()
        if expected != PREPARED_HASHES[name]:
            raise ValueError(f"Prepared SWE-bench Live serialization drift: {name}")
        if path.exists() and file_sha256(path) != expected:
            raise ValueError(f"Refusing to overwrite changed prepared data: {path}")
        path.write_text(content)
        files[name] = {"sha256": expected, "count": len(rows), "ids": [row["instance_id"] for row in rows]}
    manifest = {
        "dataset": DATASET,
        "revision": DATA_REVISION,
        "parquet_sha256": PARQUET_HASHES,
        "partition": PARTITION,
        "upstream_anomalies": {
            "full_duplicate_ids": ["conan-io__conan-18153"],
            "verified_duplicate_ids": ["conan-io__conan-18153"],
            "test_ids_missing_from_full": sorted(
                {row["instance_id"] for row in raw["test"]} - {row["instance_id"] for row in raw["full"]}
            ),
            "policy": "Preserve raw rows. Official test records own held-out identity; exclude all their IDs from training.",
        },
        "files": files,
    }
    target = root / "data-manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Prepared SWE-bench Live manifest drift")
    atomic_json(target, manifest)
    return manifest


def load_data(root: Path) -> tuple[dict, dict]:
    """Verify all prepared bytes, membership and order before exposing safe records."""
    manifest = json.loads((root / "data-manifest.json").read_text())
    if (
        manifest["dataset"] != DATASET
        or manifest["revision"] != DATA_REVISION
        or manifest["partition"] != PARTITION
        or manifest["parquet_sha256"] != PARQUET_HASHES
    ):
        raise ValueError("SWE-bench Live source identity drift")
    rows = {}
    if {name: entry["sha256"] for name, entry in manifest["files"].items()} != PREPARED_HASHES:
        raise ValueError("Pinned prepared SWE-bench Live data identity drift")
    for name, entry in manifest["files"].items():
        path = root / "data" / f"{name}.jsonl"
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"SWE-bench Live data drift: {name}")
        rows[name] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows[name]) != entry["count"] or [row["instance_id"] for row in rows[name]] != entry["ids"]:
            raise ValueError(f"SWE-bench Live ordered membership drift: {name}")
    expected = partition(rows["full"], rows["test"])
    if any(expected[name] != rows[name] for name in expected):
        raise ValueError("SWE-bench Live partition drift")
    public = {
        name: [
            {
                "id": row["instance_id"],
                "split": name,
                "problem_statement": row["problem_statement"],
                "image": image_tag(row["instance_id"]),
                "evaluator_sha256": digest(row),
            }
            for row in expected[name]
        ]
        for name in expected
    }
    return public, manifest
