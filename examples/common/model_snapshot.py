"""Prepare and verify exact local checkpoints used by benchmark campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download  # type: ignore[import-not-found]

from examples.common.experiment_models import (
    GLM_5_3_FLASH_REPO,
    GLM_5_3_FLASH_REVISION,
    QWEN3_8_27B_REPO,
    QWEN3_8_27B_REVISION,
)

QWEN3_8_27B_PROFILE = "qwen3.8-27b"
GLM_5_3_FLASH_PROFILE = "glm-5.3-flash"
MODEL_SNAPSHOT_SPECS = {
    QWEN3_8_27B_PROFILE: (QWEN3_8_27B_REPO, QWEN3_8_27B_REVISION),
    GLM_5_3_FLASH_PROFILE: (GLM_5_3_FLASH_REPO, GLM_5_3_FLASH_REVISION),
}
MODEL_INTEGRITY_NAME = ".gepa-model-integrity.json"


class ModelSnapshotError(RuntimeError):
    """Signal that a model snapshot is missing or differs from its pin."""


def _file_sha256(path: Path) -> str:
    """Hash one checkpoint file in bounded chunks.

    Args:
        path: Regular checkpoint file to read.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path, size: int) -> str:
    """Compute the Git object ID for one regular repository file.

    Args:
        path: Local file whose repository blob identity is checked.
        size: Expected byte length recorded by the repository.

    Returns:
        Lowercase Git blob SHA-1 digest.
    """
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_content_paths(root: Path) -> set[str]:
    """List model files that may affect loading or generation.

    Hugging Face keeps transfer metadata below ``.cache`` when downloading to
    a local directory. That metadata and GEPA's own integrity sidecar are not
    model inputs; every other regular file must belong to the pinned revision.

    Args:
        root: Local checkpoint directory to inspect.

    Returns:
        POSIX relative paths for all quality-relevant snapshot files.
    """
    paths = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if relative_path.as_posix() == MODEL_INTEGRITY_NAME or relative_path.parts[0] == ".cache":
            continue
        paths.add(relative_path.as_posix())
    return paths


def _build_manifest(
    root: Path,
    repo_files: list[dict[str, object]],
    model_profile: str,
) -> dict[str, object]:
    """Build byte-level identity for the pinned repository file list.

    Args:
        root: Local snapshot directory.
        repo_files: Authoritative paths, sizes, and content identifiers returned
            by Hugging Face for the pinned revision.
        model_profile: Named experiment profile whose repository is being
            materialized.

    Returns:
        Repository identity and ordered per-file size and SHA-256 records.

    Raises:
        ModelSnapshotError: The model profile is unknown, a repository file is
            absent or duplicated, a path is not regular, or an unpinned extra
            file is present.
    """
    try:
        repository, revision = MODEL_SNAPSHOT_SPECS[model_profile]
    except KeyError as exc:
        supported = ", ".join(MODEL_SNAPSHOT_SPECS)
        raise ModelSnapshotError(
            f"Unsupported model snapshot profile {model_profile!r}; expected one of: {supported}"
        ) from exc
    if not repo_files:
        raise ModelSnapshotError("Pinned model repository metadata contains no files.")
    expected_paths = {str(record["path"]) for record in repo_files}
    if len(expected_paths) != len(repo_files):
        raise ModelSnapshotError("Pinned model repository file list contains duplicates.")
    actual_paths = _snapshot_content_paths(root)
    missing_paths = sorted(expected_paths - actual_paths)
    unexpected_paths = sorted(actual_paths - expected_paths)
    if missing_paths:
        raise ModelSnapshotError(f"Pinned model files are missing: {missing_paths}.")
    if unexpected_paths:
        raise ModelSnapshotError(f"Model snapshot contains unpinned extra files: {unexpected_paths}.")
    files = []
    for source_record in sorted(repo_files, key=lambda record: str(record["path"])):
        relative_path = str(source_record["path"])
        path = root / relative_path
        if not path.is_file():
            raise ModelSnapshotError(f"Pinned model file is missing: {path}")
        expected_size = source_record.get("size")
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            raise ModelSnapshotError(f"Pinned model file size differs from repository metadata: {path}")
        source_oid = source_record.get("source_oid")
        source_oid_kind = source_record.get("source_oid_kind")
        local_sha256 = _file_sha256(path)
        if source_oid_kind == "lfs_sha256":
            authoritative_match = local_sha256 == source_oid
        elif source_oid_kind == "git_blob_sha1":
            authoritative_match = _git_blob_sha1(path, expected_size) == source_oid
        else:
            raise ModelSnapshotError(f"Pinned model file has unsupported repository metadata: {relative_path}")
        if not authoritative_match:
            raise ModelSnapshotError(f"Pinned model file differs from authoritative repository bytes: {path}")
        files.append(
            {
                "path": relative_path,
                "size": expected_size,
                "sha256": local_sha256,
                "source_oid_kind": source_oid_kind,
                "source_oid": source_oid,
            }
        )
    return {
        "schema_version": 3,
        "model_profile": model_profile,
        "repository": repository,
        "revision": revision,
        "files": files,
    }


def prepare_model_snapshot(root: str | Path, model_profile: str) -> dict[str, object]:
    """Download and hash one pinned experiment-model snapshot.

    Args:
        root: Local directory served by the selected inference runtime.
        model_profile: Exact profile name in :data:`MODEL_SNAPSHOT_SPECS`.

    Returns:
        Persisted byte-level model manifest.

    Raises:
        ModelSnapshotError: The profile is unknown or Hugging Face does not
            materialize the exact pinned repository contents.
    """
    try:
        repository, revision = MODEL_SNAPSHOT_SPECS[model_profile]
    except KeyError as exc:
        supported = ", ".join(MODEL_SNAPSHOT_SPECS)
        raise ModelSnapshotError(
            f"Unsupported model snapshot profile {model_profile!r}; expected one of: {supported}"
        ) from exc
    snapshot_root = Path(root).expanduser().resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=snapshot_root,
        )
        repo_files = []
        for entry in HfApi().list_repo_tree(
            repo_id=repository,
            revision=revision,
            recursive=True,
            expand=True,
        ):
            if not hasattr(entry, "blob_id"):
                continue
            if entry.lfs is None:
                source_oid_kind = "git_blob_sha1"
                source_oid = entry.blob_id
            else:
                source_oid_kind = "lfs_sha256"
                source_oid = entry.lfs.sha256
            repo_files.append(
                {
                    "path": entry.path,
                    "size": entry.size,
                    "source_oid_kind": source_oid_kind,
                    "source_oid": source_oid,
                }
            )
    except Exception as exc:
        raise ModelSnapshotError(f"Could not prepare {repository}@{revision}: {exc}") from exc
    manifest = _build_manifest(snapshot_root, repo_files, model_profile)
    manifest_path = snapshot_root / MODEL_INTEGRITY_NAME
    temporary_path = manifest_path.with_suffix(".json.part")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest


def verify_model_snapshot(root: str | Path, model_profile: str) -> dict[str, object]:
    """Verify every byte in one pinned snapshot without network access.

    Args:
        root: Local directory served by the selected inference runtime.
        model_profile: Exact profile name in :data:`MODEL_SNAPSHOT_SPECS`.

    Returns:
        Verified byte-level model manifest.

    Raises:
        ModelSnapshotError: The profile is unknown; the manifest is missing or
            malformed; or any identity, size, or digest differs from the pin.
    """
    try:
        repository, revision = MODEL_SNAPSHOT_SPECS[model_profile]
    except KeyError as exc:
        supported = ", ".join(MODEL_SNAPSHOT_SPECS)
        raise ModelSnapshotError(
            f"Unsupported model snapshot profile {model_profile!r}; expected one of: {supported}"
        ) from exc
    snapshot_root = Path(root).expanduser().resolve()
    manifest_path = snapshot_root / MODEL_INTEGRITY_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSnapshotError(f"Pinned model integrity manifest is unreadable at {manifest_path}: {exc}") from exc
    if (
        manifest.get("schema_version") != 3
        or manifest.get("model_profile") != model_profile
        or manifest.get("repository") != repository
        or manifest.get("revision") != revision
    ):
        raise ModelSnapshotError(f"Model snapshot must be {repository}@{revision} for {model_profile!r}.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ModelSnapshotError("Pinned model integrity manifest contains no files.")
    expected_paths = set()
    for record in files:
        if not isinstance(record, dict):
            raise ModelSnapshotError("Pinned model integrity manifest contains a malformed file record.")
        relative_path = Path(str(record.get("path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ModelSnapshotError("Pinned model integrity manifest contains an unsafe file path.")
        relative_path_text = relative_path.as_posix()
        if relative_path_text in expected_paths:
            raise ModelSnapshotError("Pinned model integrity manifest contains a duplicate file path.")
        expected_paths.add(relative_path_text)
        path = snapshot_root / relative_path
        if not path.is_file() or path.stat().st_size != record.get("size"):
            raise ModelSnapshotError(f"Pinned model file size differs: {path}")
        source_oid_kind = record.get("source_oid_kind")
        source_oid = record.get("source_oid")
        if source_oid_kind not in {"lfs_sha256", "git_blob_sha1"} or not isinstance(source_oid, str):
            raise ModelSnapshotError("Pinned model integrity manifest omits authoritative repository metadata.")
        digest = _file_sha256(path)
        if source_oid_kind == "lfs_sha256":
            authoritative_match = len(source_oid) == 64 and digest == source_oid
        else:
            authoritative_match = (
                len(source_oid) == 40 and _git_blob_sha1(path, int(record["size"])) == source_oid
            )
        if not authoritative_match:
            raise ModelSnapshotError(f"Pinned model file differs from authoritative repository bytes: {path}")
        if digest != record.get("sha256"):
            raise ModelSnapshotError(f"Pinned model file digest differs: {path}")
    unexpected_paths = sorted(_snapshot_content_paths(snapshot_root) - expected_paths)
    if unexpected_paths:
        raise ModelSnapshotError(f"Model snapshot contains unpinned extra files: {unexpected_paths}.")
    return manifest


def main() -> None:
    """Prepare or verify one pinned checkpoint from the command line.

    Raises:
        ModelSnapshotError: The requested snapshot operation cannot establish
            the pinned checkpoint identity.
    """
    parser = argparse.ArgumentParser(description="Prepare or verify a pinned experiment-model checkpoint")
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--model-profile", choices=tuple(MODEL_SNAPSHOT_SPECS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_model_snapshot(args.root, args.model_profile)
    else:
        manifest = verify_model_snapshot(args.root, args.model_profile)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
