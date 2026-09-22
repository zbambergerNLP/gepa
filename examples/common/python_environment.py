"""Freeze and verify a realized Python serving environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ENVIRONMENT_MANIFEST_SCHEMA_VERSION = 1
_DISTRIBUTION_METADATA_FILES = (
    "INSTALLER",
    "METADATA",
    "RECORD",
    "WHEEL",
    "direct_url.json",
)


class PythonEnvironmentError(RuntimeError):
    """Signal that a realized Python environment differs from its freeze."""


def _file_sha256(path: Path) -> str:
    """Hash one runtime file in bounded chunks.

    Args:
        path: Regular file to read.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_record(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    """Describe one installed distribution and its installation metadata.

    The package metadata complements the byte-level virtual-environment
    inventory with the realized wheel record, installer, wheel tags, and
    direct-source origin.

    Args:
        distribution: Installed distribution exposed by ``importlib.metadata``.

    Returns:
        Stable package name, version, location, and metadata digests.

    Raises:
        PythonEnvironmentError: The distribution omits its canonical name or
            version.
    """
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not name or not version:
        raise PythonEnvironmentError("An installed distribution omits its canonical name or version.")
    location = Path(str(distribution.locate_file(""))).resolve()
    prefix = Path(sys.prefix).resolve()
    try:
        recorded_location = location.relative_to(prefix).as_posix()
    except ValueError:
        recorded_location = location.as_posix()
    metadata_sha256 = {}
    for filename in _DISTRIBUTION_METADATA_FILES:
        content = distribution.read_text(filename)
        metadata_sha256[filename] = (
            hashlib.sha256(content.encode("utf-8")).hexdigest() if content is not None else None
        )
    return {
        "name": name,
        "normalized_name": name.lower().replace("_", "-").replace(".", "-"),
        "version": version,
        "location": recorded_location,
        "metadata_sha256": metadata_sha256,
    }


def _editable_source_record(
    distribution: importlib.metadata.Distribution,
) -> dict[str, str] | None:
    """Pin one editable distribution to a clean Git commit.

    Args:
        distribution: Installed distribution whose direct source is inspected.

    Returns:
        Repository path, package subpath, and exact commit for an editable
        install, or ``None`` for a regular wheel or immutable VCS install.

    Raises:
        PythonEnvironmentError: Editable-source metadata is malformed, the
            source is not a local Git checkout, or its tracked/untracked state
            differs from the recorded commit.
    """
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise PythonEnvironmentError("An installed distribution has malformed direct_url.json metadata.") from exc
    if not isinstance(direct_url, dict):
        raise PythonEnvironmentError("An installed distribution has malformed direct_url.json metadata.")
    directory_info = direct_url.get("dir_info")
    if not isinstance(directory_info, dict) or directory_info.get("editable") is not True:
        return None
    parsed_url = urlsplit(str(direct_url.get("url", "")))
    if parsed_url.scheme != "file" or parsed_url.netloc not in {"", "localhost"}:
        raise PythonEnvironmentError("Editable distributions must identify a local file URL.")
    source_path = Path(unquote(parsed_url.path)).resolve()
    if not source_path.is_dir():
        raise PythonEnvironmentError(f"Editable distribution source is missing: {source_path}")
    try:
        repository_root_text = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        repository_root = Path(repository_root_text).resolve()
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise PythonEnvironmentError(
            f"Editable distribution source is not a readable Git checkout: {source_path}"
        ) from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise PythonEnvironmentError(f"Editable distribution does not resolve to an exact commit: {source_path}")
    if status:
        raise PythonEnvironmentError(f"Editable distribution source has local changes: {repository_root}")
    try:
        source_subpath = source_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise PythonEnvironmentError(
            f"Editable distribution source is outside its Git repository: {source_path}"
        ) from exc
    return {
        "repository_root": repository_root.as_posix(),
        "source_subpath": source_subpath,
        "commit": commit,
    }


def _environment_file_records(root: Path) -> list[dict[str, Any]]:
    """Hash every stable file and symlink in one virtual environment.

    Python bytecode caches are derived from the recorded source and extension
    files and may appear merely because a preflight imported a module. They are
    excluded so harmless imports do not invalidate an otherwise exact runtime.

    Args:
        root: Virtual-environment prefix to inventory.

    Returns:
        Ordered relative paths with file bytes or symlink identities.

    Raises:
        PythonEnvironmentError: A path disappears or cannot be read while the
            environment is being inventoried.
    """
    digest_cache: dict[tuple[int, int, int], str] = {}
    records = []
    try:
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        for path in paths:
            relative_path = path.relative_to(root)
            if "__pycache__" in relative_path.parts or relative_path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                record: dict[str, Any] = {
                    "kind": "symlink",
                    "path": relative_path.as_posix(),
                    "target": os.readlink(path),
                }
                resolved_target = path.resolve(strict=True)
                if resolved_target.is_file():
                    target_stat = resolved_target.stat()
                    cache_key = (target_stat.st_dev, target_stat.st_ino, target_stat.st_size)
                    target_digest = digest_cache.get(cache_key)
                    if target_digest is None:
                        target_digest = _file_sha256(resolved_target)
                        digest_cache[cache_key] = target_digest
                    record["target_size"] = target_stat.st_size
                    record["target_sha256"] = target_digest
                records.append(record)
                continue
            if not path.is_file():
                continue
            file_stat = path.stat()
            cache_key = (file_stat.st_dev, file_stat.st_ino, file_stat.st_size)
            file_digest = digest_cache.get(cache_key)
            if file_digest is None:
                file_digest = _file_sha256(path)
                digest_cache[cache_key] = file_digest
            records.append(
                {
                    "kind": "file",
                    "path": relative_path.as_posix(),
                    "size": file_stat.st_size,
                    "sha256": file_digest,
                }
            )
    except OSError as exc:
        raise PythonEnvironmentError(f"Could not inventory Python environment at {root}: {exc}") from exc
    return records


def build_environment_manifest() -> dict[str, Any]:
    """Build a deterministic identity for the active Python environment.

    Returns:
        Interpreter identity and every installed distribution in stable order.

    Raises:
        PythonEnvironmentError: The interpreter executable is missing or an
            installed distribution cannot be identified.
    """
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise PythonEnvironmentError(f"Python executable is missing: {executable}")
    distributions = []
    editable_sources = []
    for distribution in importlib.metadata.distributions():
        distributions.append(_distribution_record(distribution))
        editable_source = _editable_source_record(distribution)
        if editable_source is not None and editable_source not in editable_sources:
            editable_sources.append(editable_source)
    distributions.sort(
        key=lambda record: (
            record["normalized_name"],
            record["version"],
            record["location"],
        )
    )
    editable_sources.sort(
        key=lambda record: (
            record["repository_root"],
            record["source_subpath"],
            record["commit"],
        )
    )
    return {
        "schema_version": ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "executable_sha256": _file_sha256(executable),
        },
        "distributions": distributions,
        "editable_sources": editable_sources,
        "files": _environment_file_records(Path(sys.prefix).resolve()),
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize one environment manifest canonically.

    Args:
        manifest: Environment record to serialize.

    Returns:
        Stable UTF-8 JSON bytes terminated by one newline.
    """
    rendered = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True)
    return (rendered + "\n").encode("utf-8")


def prepare_environment_manifest(path: str | Path) -> str:
    """Atomically freeze the active Python environment.

    Args:
        path: Manifest file to replace atomically.

    Returns:
        SHA-256 digest of the persisted canonical manifest.
    """
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _manifest_bytes(build_environment_manifest())
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".part")
    temporary_path.write_bytes(manifest_bytes)
    temporary_path.replace(manifest_path)
    return hashlib.sha256(manifest_bytes).hexdigest()


def verify_environment_manifest(path: str | Path) -> str:
    """Verify that the active Python environment matches a frozen manifest.

    Args:
        path: Previously prepared environment manifest.

    Returns:
        SHA-256 digest of the verified canonical manifest.

    Raises:
        PythonEnvironmentError: The manifest is unreadable, malformed, or no
            longer matches the active interpreter and installed packages.
    """
    manifest_path = Path(path).expanduser().resolve()
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PythonEnvironmentError(
            f"Python environment manifest is unreadable at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(expected, dict) or expected.get("schema_version") != ENVIRONMENT_MANIFEST_SCHEMA_VERSION:
        raise PythonEnvironmentError("Python environment manifest has an unsupported schema.")
    actual = build_environment_manifest()
    if actual != expected:
        expected_digest = hashlib.sha256(_manifest_bytes(expected)).hexdigest()
        actual_digest = hashlib.sha256(_manifest_bytes(actual)).hexdigest()
        raise PythonEnvironmentError(
            "Python environment differs from its frozen manifest "
            f"(expected {expected_digest}, found {actual_digest})."
        )
    return hashlib.sha256(_manifest_bytes(expected)).hexdigest()


def main() -> None:
    """Prepare or verify an environment manifest from the command line.

    Raises:
        PythonEnvironmentError: The selected operation cannot establish the
            exact active Python environment.
    """
    parser = argparse.ArgumentParser(description="Freeze or verify a realized Python environment")
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        digest = prepare_environment_manifest(args.path)
    else:
        digest = verify_environment_manifest(args.path)
    print(digest)


if __name__ == "__main__":
    main()
