"""Prepare and verify the exact Qwen checkpoint used by HotPotQA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download  # type: ignore[import-not-found]

from examples.common.experiment_models import QWEN3_8_27B_REVISION

QWEN3_8_27B_REPO = "Qwen/Qwen3.8-27B"
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


def _build_manifest(root: Path, repo_files: list[dict[str, object]]) -> dict[str, object]:
    """Build byte-level identity for the pinned repository file list.

    Args:
        root: Local snapshot directory.
        repo_files: Authoritative paths, sizes, and content identifiers returned
            by Hugging Face for the pinned revision.

    Returns:
        Repository identity and ordered per-file size and SHA-256 records.

    Raises:
        ModelSnapshotError: A repository file is absent, duplicated, is not
            regular, or an unpinned extra file is present.
    """
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
        "schema_version": 2,
        "repository": QWEN3_8_27B_REPO,
        "revision": QWEN3_8_27B_REVISION,
        "files": files,
    }


def prepare_qwen_snapshot(root: str | Path) -> dict[str, object]:
    """Download and hash the pinned Qwen3.8-27B snapshot.

    Args:
        root: Local directory served by vLLM.

    Returns:
        Persisted byte-level model manifest.

    Raises:
        ModelSnapshotError: Hugging Face does not materialize the exact pinned
            repository contents.
    """
    snapshot_root = Path(root).expanduser().resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=QWEN3_8_27B_REPO,
            revision=QWEN3_8_27B_REVISION,
            local_dir=snapshot_root,
        )
        repo_files = []
        for entry in HfApi().list_repo_tree(
            repo_id=QWEN3_8_27B_REPO,
            revision=QWEN3_8_27B_REVISION,
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
        raise ModelSnapshotError(f"Could not prepare {QWEN3_8_27B_REPO}@{QWEN3_8_27B_REVISION}: {exc}") from exc
    manifest = _build_manifest(snapshot_root, repo_files)
    manifest_path = snapshot_root / MODEL_INTEGRITY_NAME
    temporary_path = manifest_path.with_suffix(".json.part")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest


def verify_qwen_snapshot(root: str | Path) -> dict[str, object]:
    """Verify every pinned Qwen3.8-27B snapshot byte without network access.

    Args:
        root: Local directory served by vLLM.

    Returns:
        Verified byte-level model manifest.

    Raises:
        ModelSnapshotError: The manifest is missing, malformed, identifies a
            different revision, or any file differs in size or digest.
    """
    snapshot_root = Path(root).expanduser().resolve()
    manifest_path = snapshot_root / MODEL_INTEGRITY_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSnapshotError(f"Pinned model integrity manifest is unreadable at {manifest_path}: {exc}") from exc
    if (
        manifest.get("schema_version") != 2
        or manifest.get("repository") != QWEN3_8_27B_REPO
        or manifest.get("revision") != QWEN3_8_27B_REVISION
    ):
        raise ModelSnapshotError(
            f"Model snapshot must be {QWEN3_8_27B_REPO}@{QWEN3_8_27B_REVISION}."
        )
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
    """Prepare or verify the pinned Qwen checkpoint from the command line.

    Raises:
        ModelSnapshotError: The requested snapshot operation cannot establish
            the pinned checkpoint identity.
    """
    parser = argparse.ArgumentParser(description="Prepare or verify the pinned Qwen3.8-27B checkpoint")
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_qwen_snapshot(args.root)
    else:
        manifest = verify_qwen_snapshot(args.root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
