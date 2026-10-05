"""Verify pinned upstream files and keep task families out of other splits."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from examples.taubench.benchmark_settings import (
    DOMAIN,
    MANIFEST_PATH,
    SPLIT_SALT,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    UPSTREAM_VERSION,
)

DATA_PATH = Path("data/tau2/domains/banking_knowledge")


def digest(value: Any) -> str:
    """Hash JSON without platform-specific whitespace or nonfinite values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Fingerprint the exact bytes rather than trusting checkout metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_identity(source: Path, relative: str) -> dict[str, Any]:
    """Hash sorted source-relative paths and bytes, rejecting symlinks."""
    root = source / relative
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not files or any(p.is_symlink() for p in files):
        raise ValueError(f"Missing or symlinked upstream tree: {relative}")
    rows = [{"path": p.relative_to(source).as_posix(), "sha256": file_digest(p)} for p in files]
    return {"count": len(rows), "ordered_sha256": digest(rows)}


def _identities(value: Any) -> set[str]:
    """Find explicit customer identities used only for conservative split grouping."""
    identities = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"user_id", "customer_name"} and isinstance(item, str):
                identities.add(f"{key}:{item.casefold().strip()}")
            identities.update(_identities(item))
    elif isinstance(value, list):
        for item in value:
            identities.update(_identities(item))
    return identities


def task_groups(tasks: dict[str, dict]) -> dict[str, str]:
    """Join shared customers, identical scenarios and explicitly named task variants.

    These are local optimization groups, not an upstream family annotation.
    Evaluation fields are inspected only here, never sent to the solver.
    """
    links: dict[str, set[str]] = {}
    for task_id, task in tasks.items():
        links[task_id] = _identities(task) | {"scenario:" + digest(task["user_scenario"]), "task:" + task_id}
        notes = (task.get("description") or {}).get("notes") or ""
        links[task_id].update("task:" + ref for ref in re.findall(r"task_\d{3}", notes) if ref in tasks)
    groups = {}
    for task_id in sorted(tasks):
        if task_id in groups:
            continue
        members, identities = {task_id}, set(links[task_id])
        while True:
            joined = {other for other in tasks if identities & links[other]}
            if joined <= members:
                break
            members |= joined
            identities.update(*(links[other] for other in joined))
        group_id = min(members)
        groups.update(dict.fromkeys(members, group_id))
    return groups


def inspect_source(source: Path) -> dict[str, Any]:
    """Create the exact manifest representation, failing on malformed task data."""
    if not (source / "src/tau2").is_dir():
        raise ValueError("Missing pinned tau checkout; follow examples/taubench/README.md setup")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"tau source must be at {UPSTREAM_REVISION}, found {revision}")
    task_paths = sorted((source / DATA_PATH / "tasks").glob("task_*.json"))
    tasks = {}
    for path in task_paths:
        task = json.loads(path.read_text())
        task_id = task.get("id")
        if (
            task_id != path.stem
            or task_id in tasks
            or not task.get("evaluation_criteria")
            or not task.get("user_scenario")
        ):
            raise ValueError(f"Malformed, duplicate, or unscored task: {path}")
        tasks[task_id] = task
    if not tasks:
        raise ValueError("No banking tasks were loaded")
    groups = task_groups(tasks)
    ordered_groups = sorted(set(groups.values()), key=lambda group: digest([SPLIT_SALT, group]))
    train_end, val_end = len(ordered_groups) * 6 // 10, len(ordered_groups) * 8 // 10
    group_splits = {
        group: "train" if i < train_end else "val" if i < val_end else "test" for i, group in enumerate(ordered_groups)
    }
    ordered_ids = sorted(tasks, key=lambda task_id: (ordered_groups.index(groups[task_id]), task_id))
    records = [
        {
            "id": f"tau-banking/{task_id}",
            "task_id": task_id,
            "scenario_id": task_id,
            "group_id": groups[task_id],
            "split": group_splits[groups[task_id]],
            "task_sha256": file_digest(source / DATA_PATH / "tasks" / f"{task_id}.json"),
        }
        for task_id in ordered_ids
    ]
    return {
        "schema_version": 1,
        "repository": UPSTREAM_REPOSITORY,
        "revision": revision,
        "release": UPSTREAM_VERSION,
        "domain": DOMAIN,
        "source": tree_identity(source, "src/tau2"),
        "knowledge": tree_identity(source, str(DATA_PATH / "documents")),
        "prompts": tree_identity(source, str(DATA_PATH / "prompts")),
        "user_simulator": tree_identity(source, "data/tau2/user_simulator"),
        "database_sha256": file_digest(source / DATA_PATH / "db.json"),
        "lock_sha256": file_digest(source / "uv.lock"),
        "pyproject_sha256": file_digest(source / "pyproject.toml"),
        "split_policy": {"salt": SPLIT_SALT, "group_fractions": [0.6, 0.2, 0.2], "official_split": False},
        "records": records,
    }


def load_data(source: Path) -> tuple[dict[str, list[dict]], dict]:
    """Reject source/data drift before returning ordered, disjoint public records."""
    expected = json.loads(MANIFEST_PATH.read_text())
    actual = inspect_source(source.resolve())
    if actual != expected:
        raise ValueError("Pinned tau source, corpus, task records, or split identity changed")
    splits = {
        split: [dict(row) for row in actual["records"] if row["split"] == split] for split in ("train", "val", "test")
    }
    if any(not rows for rows in splits.values()):
        raise ValueError("All optimization splits must be nonempty")
    seen_ids, seen_groups = set(), set()
    for rows in splits.values():
        ids, groups = {row["id"] for row in rows}, {row["group_id"] for row in rows}
        if seen_ids & ids or seen_groups & groups or len(ids) != len(rows):
            raise ValueError("Task or customer-group leakage across splits")
        seen_ids |= ids
        seen_groups |= groups
    return splits, actual


def upstream_system_prompt(source: Path) -> str:
    """Read the shipped BM25 prompt without importing the paid-call runtime."""
    tree = ast.parse((source / "src/tau2/agent/llm_agent.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"AGENT_INSTRUCTION", "SYSTEM_PROMPT"}:
                if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Attribute):
                    raise ValueError("Upstream prompt constant layout changed")
                constants[name] = ast.literal_eval(node.value.func.value).strip()
    directory = source / DATA_PATH / "prompts"
    policy = (directory / "classic_rag_bm25_no_grep.md").read_text()
    policy = re.sub(r"\{\{component:(\w+)\}\}", lambda m: (directory / "components" / f"{m[1]}.md").read_text(), policy)
    return constants["SYSTEM_PROMPT"].format(agent_instruction=constants["AGENT_INSTRUCTION"], domain_policy=policy)
