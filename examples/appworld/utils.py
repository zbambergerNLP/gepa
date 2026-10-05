"""Validate the complete pinned corpus without exposing protected data in records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.appworld.benchmark_settings import (
    APPWORLD_REVISION,
    DATA_BUNDLE_SHA256,
    DATA_URL,
    DATA_VERSION,
    OFFICIAL_SPLITS,
)

DATA_PIN_PATH = Path(__file__).with_name("data_pin.json")


def json_digest(value: Any) -> str:
    """Hash a deterministic JSON value, rejecting nonfinite numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading the entire corpus into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(directory: Path) -> str:
    """Hash ordered relative names and contents, excluding Python's generated caches."""
    files = []
    for path in sorted(directory.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in immutable AppWorld data: {path}")
        if path.is_file():
            files.append([path.relative_to(directory).as_posix(), file_digest(path)])
    if not files:
        raise ValueError(f"Missing or empty AppWorld data directory: {directory}")
    return json_digest(files)


def read_split_ids(data: Path, split: str) -> list[str]:
    """Read the upstream ordered split file, applying its remove_tag convention."""
    lines = (data / "datasets" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    ids = [line.strip().split(":", 1)[0] for line in lines if line.strip()]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9]+_[1-9][0-9]*", id_) for id_ in ids):
        raise ValueError(f"Invalid or duplicate task IDs in official split {split}.")
    return ids


def inspect_data(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Fingerprint all four official assignments and reject task or scenario leakage."""
    data = root / "data"
    records: dict[str, list[dict[str, Any]]] = {}
    scenario_splits: dict[str, str] = {}
    task_ids: set[str] = set()
    for split in OFFICIAL_SPLITS:
        records[split] = []
        for task_id in read_split_ids(data, split):
            scenario = task_id.split("_", 1)[0]
            if task_id in task_ids or scenario_splits.get(scenario, split) != split:
                raise ValueError("AppWorld task/scenario leakage across official splits.")
            task_ids.add(task_id)
            scenario_splits[scenario] = split
            records[split].append(
                {
                    "id": f"appworld:{task_id}",
                    "task_id": task_id,
                    "scenario_id": scenario,
                    "official_split": split,
                    "record_sha256": tree_digest(data / "tasks" / task_id),
                }
            )
    identity = {
        "data_tree_sha256": tree_digest(data),
        "splits": {
            split: {
                "count": len(items),
                "scenario_count": len({item["scenario_id"] for item in items}),
                "ordered_records_sha256": json_digest(items),
            }
            for split, items in records.items()
        },
    }
    return records, identity


def load_dataset(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Require the released corpus and preserve every official split in original order."""
    records, identity = inspect_data(root)
    pin = json.loads(DATA_PIN_PATH.read_text(encoding="utf-8"))
    if identity != pin["data"]:
        raise ValueError("AppWorld data or ordered split drift; restore the pinned 0.1.0 bundle.")
    source = {
        "repository": "https://github.com/StonyBrookNLP/appworld",
        "revision": APPWORLD_REVISION,
        "data_version": DATA_VERSION,
        "bundle_url": DATA_URL,
        "bundle_sha256": DATA_BUNDLE_SHA256,
        **identity,
        "split_mapping": {"train": ["train"], "val": ["dev"], "test": ["test_normal", "test_challenge"]},
    }
    return records, source


def scenario_members(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Retain complete scenario membership for honest SGC on limited evaluations."""
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record["scenario_id"]].add(record["task_id"])
    return dict(groups)
