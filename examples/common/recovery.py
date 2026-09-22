"""Seal durable work for allocation recovery without using metric improvement."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path

_LOCK = threading.RLock()


def file_digest(path: Path) -> str:
    """Hash a checkpoint without loading its potentially large contents at once."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write(path: Path, value: dict) -> None:
    """Atomically replace a durable recovery record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal_progress(directory: Path, completed_work: int, artifacts: list[Path]) -> None:
    """Register hash-verified saved work with the current allocation controller."""
    registry_name = os.environ.get("GEPA_RECOVERY_REGISTRY")
    if not registry_name:
        return
    if completed_work < 0 or not artifacts:
        raise ValueError("Recovery requires nonnegative completed work and durable artifacts")
    directory = directory.resolve()
    for name in ("pilot-contract.json", "terminalbench-run-contract.json", "wikipedia-run-contract.json"):
        path = directory / name
        if path.exists() and path not in artifacts:
            artifacts = [*artifacts, path]
    with _LOCK:
        stamp = directory / "recovery-checkpoint.json"
        _write(
            stamp,
            {
                "source_commit": os.environ.get("HOTPOTQA_SOURCE_COMMIT") or os.environ.get("GEPA_SOURCE_COMMIT"),
                "completed_work": completed_work,
                "artifacts": {str(path.resolve()): file_digest(path) for path in artifacts},
            },
        )
        registry = Path(registry_name)
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            saved = json.loads(registry.read_text()) if registry.exists() else {"checkpoints": []}
            if str(stamp) not in saved["checkpoints"]:
                saved["checkpoints"].append(str(stamp))
            _write(registry, saved)


def snapshot(registry: Path, source_commit: str) -> dict[str, int]:
    """Reject corrupt or incompatible recovery evidence and count saved work."""
    if not registry.exists():
        return {}
    result = {}
    for name in json.loads(registry.read_text())["checkpoints"]:
        path = Path(name)
        record = json.loads(path.read_text())
        if record["source_commit"] != source_commit or not record["artifacts"]:
            raise ValueError(f"Recovery identity mismatch: {path}")
        for artifact, expected in record["artifacts"].items():
            if file_digest(Path(artifact)) != expected:
                raise ValueError(f"Recovery checkpoint changed or is incomplete: {artifact}")
        work = record["completed_work"]
        if type(work) is not int or work < 0:
            raise ValueError(f"Invalid saved progress: {path}")
        result[name] = work
    return result


class RecoveryCallback:
    """Seal GEPA's own checkpoints at its existing save boundaries."""

    def __init__(self, directory: Path):
        """Remember the exact logical run directory."""
        self.directory = directory
        self.completed = 0

    def on_iteration_start(self, event: dict) -> None:
        """Seal the checkpoint saved immediately before this iteration."""
        state = event["state"]
        self.completed = state.total_num_evals
        seal_progress(self.directory, self.completed, [self.directory / "gepa_state.bin"])

    def on_optimization_end(self, event: dict) -> None:
        """Seal the final optimizer state before test evaluation begins."""
        seal_progress(self.directory, event["total_metric_calls"], [self.directory / "gepa_state.bin"])


def run_guarded(main) -> None:
    """Distinguish reported execution errors from scheduler termination."""
    try:
        main()
    except Exception as exc:
        destination = os.environ.get("GEPA_RECOVERY_ERROR_FILE")
        if destination:
            _write(Path(destination), {"error_type": type(exc).__name__})
        raise
