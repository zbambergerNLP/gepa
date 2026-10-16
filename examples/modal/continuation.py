"""Seal and verify allocation boundaries on the persistent results volume."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from examples.common.artifacts import atomic_json


def allocation_directory(epoch: int) -> str:
    """Keep each allocation's status, logs, runtimes and physical attempts immutable."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Invalid allocation number")
    return f"allocations/{epoch:06d}"


def file_digest(path: Path) -> str:
    """Hash large evidence streams without reading them all into memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def snapshot_state(results: Path, directory: Path) -> dict:
    """Archive mutable state after draining so earlier checkpoint seals remain true."""
    manifest = {}
    archive = directory / "state.tar"
    with tarfile.open(archive, "x") as bundle:
        for path in sorted(results.rglob("*")):
            relative = path.relative_to(results)
            if relative.parts[0] in {"allocations", "source.tar"} or not path.is_file() or path.is_symlink():
                continue
            manifest[str(relative)] = file_digest(path)
            bundle.add(path, arcname=str(relative), recursive=False)
    proof = {"files": manifest, "archive_sha256": file_digest(archive)}
    atomic_json(directory / "state-manifest.json", proof)
    return proof


def verify_state(results: Path, directory: Path) -> None:
    """Refuse drift or unknown task claims before a resumed driver can call a model."""
    proof = json.loads((directory / "state-manifest.json").read_text())
    if file_digest(directory / "state.tar") != proof["archive_sha256"]:
        raise ValueError("Continuation snapshot checksum changed")
    for relative, expected in proof["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe continuation artifact path")
        if file_digest(results / relative) != expected:
            raise ValueError(f"Continuation artifact changed: {relative}")
    for path in results.glob("**/.task-windows/**/window-*.json"):
        if json.loads(path.read_text()).get("state") != "sealed":
            raise ValueError(f"Unresolved task group: {path}")


def runtime_identity(execution: dict) -> dict:
    """Drop only allocation names and tunnel URLs; all material runtime fields stay."""
    return {
        **{key: value for key, value in execution.items() if key != "roles"},
        "roles": {
            role: {key: value for key, value in record.items() if key not in {"api_base", "allocation"}}
            for role, record in execution["roles"].items()
        },
    }
