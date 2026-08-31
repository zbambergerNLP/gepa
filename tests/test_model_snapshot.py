"""Tests for exact local experiment-model snapshot manifests."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common import model_snapshot
from examples.common.experiment_models import GLM_5_3_FLASH_REVISION, QWEN3_8_27B_REVISION


@pytest.mark.parametrize(
    ("model_profile", "repository", "revision"),
    [
        (
            model_snapshot.QWEN3_8_27B_PROFILE,
            model_snapshot.QWEN3_8_27B_REPO,
            QWEN3_8_27B_REVISION,
        ),
        (
            model_snapshot.GLM_5_3_FLASH_PROFILE,
            model_snapshot.GLM_5_3_FLASH_REPO,
            GLM_5_3_FLASH_REVISION,
        ),
    ],
)
def test_prepare_and_verify_model_snapshot_use_the_exact_revision(
    monkeypatch,
    tmp_path,
    model_profile: str,
    repository: str,
    revision: str,
) -> None:
    """Download, hash, persist, and verify each pinned repository revision.

    Args:
        monkeypatch: Pytest fixture used to replace Hugging Face network calls.
        tmp_path: Pytest directory used as the local model snapshot.
        model_profile: Named model profile under test.
        repository: Expected Hugging Face repository.
        revision: Expected immutable Hugging Face revision.
    """
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
    model_sha256 = hashlib.sha256(b"weights").hexdigest()
    config_path = tmp_path / "config.json"
    snapshot_download = Mock()
    repository_api = SimpleNamespace(
        list_repo_tree=Mock(
            return_value=[
                SimpleNamespace(
                    path="model.safetensors",
                    size=len(b"weights"),
                    blob_id="lfs-pointer",
                    lfs=SimpleNamespace(sha256=model_sha256),
                ),
                SimpleNamespace(
                    path="config.json",
                    size=config_path.stat().st_size,
                    blob_id=model_snapshot._git_blob_sha1(config_path, config_path.stat().st_size),
                    lfs=None,
                ),
            ]
        ),
    )
    monkeypatch.setattr(model_snapshot, "snapshot_download", snapshot_download)
    monkeypatch.setattr(model_snapshot, "HfApi", Mock(return_value=repository_api))

    prepared = model_snapshot.prepare_model_snapshot(tmp_path, model_profile)

    snapshot_download.assert_called_once_with(
        repo_id=repository,
        revision=revision,
        local_dir=tmp_path.resolve(),
    )
    repository_api.list_repo_tree.assert_called_once_with(
        repo_id=repository,
        revision=revision,
        recursive=True,
        expand=True,
    )
    assert prepared["schema_version"] == 3
    assert prepared["model_profile"] == model_profile
    assert prepared["repository"] == repository
    assert prepared["revision"] == revision
    assert [record["path"] for record in prepared["files"]] == ["config.json", "model.safetensors"]
    assert json.loads((tmp_path / model_snapshot.MODEL_INTEGRITY_NAME).read_text()) == prepared
    assert model_snapshot.verify_model_snapshot(tmp_path, model_profile) == prepared


def test_verify_model_snapshot_rejects_file_size_and_digest_drift(tmp_path) -> None:
    """Reject truncation and same-size checkpoint-byte corruption offline.

    Args:
        tmp_path: Pytest directory containing a synthetic pinned snapshot.
    """
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights-a")
    manifest = model_snapshot._build_manifest(
        tmp_path,
        [
            {
                "path": "model.safetensors",
                "size": model_file.stat().st_size,
                "source_oid_kind": "lfs_sha256",
                "source_oid": hashlib.sha256(b"weights-a").hexdigest(),
            }
        ],
        model_snapshot.QWEN3_8_27B_PROFILE,
    )
    (tmp_path / model_snapshot.MODEL_INTEGRITY_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    model_file.write_bytes(b"short")
    with pytest.raises(model_snapshot.ModelSnapshotError, match="size differs"):
        model_snapshot.verify_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)

    model_file.write_bytes(b"weights-b")
    with pytest.raises(model_snapshot.ModelSnapshotError, match="authoritative repository bytes"):
        model_snapshot.verify_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)


def test_verify_model_snapshot_rejects_unpinned_extra_files(tmp_path) -> None:
    """Reject stale templates or generation settings outside the revision.

    Args:
        tmp_path: Pytest directory containing a synthetic pinned snapshot.
    """
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights")
    manifest = model_snapshot._build_manifest(
        tmp_path,
        [
            {
                "path": "model.safetensors",
                "size": model_file.stat().st_size,
                "source_oid_kind": "lfs_sha256",
                "source_oid": hashlib.sha256(b"weights").hexdigest(),
            }
        ],
        model_snapshot.QWEN3_8_27B_PROFILE,
    )
    (tmp_path / model_snapshot.MODEL_INTEGRITY_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "chat_template.jinja").write_text("stale template", encoding="utf-8")

    with pytest.raises(model_snapshot.ModelSnapshotError, match="unpinned extra files"):
        model_snapshot.verify_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)


def test_verify_model_snapshot_rejects_file_and_local_digest_tampering(tmp_path) -> None:
    """Keep authoritative repository identity stronger than the local sidecar.

    Args:
        tmp_path: Pytest directory containing a synthetic pinned snapshot.
    """
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights-a")
    manifest = model_snapshot._build_manifest(
        tmp_path,
        [
            {
                "path": "model.safetensors",
                "size": model_file.stat().st_size,
                "source_oid_kind": "lfs_sha256",
                "source_oid": hashlib.sha256(b"weights-a").hexdigest(),
            }
        ],
        model_snapshot.QWEN3_8_27B_PROFILE,
    )
    model_file.write_bytes(b"weights-b")
    manifest["files"][0]["sha256"] = hashlib.sha256(b"weights-b").hexdigest()
    (tmp_path / model_snapshot.MODEL_INTEGRITY_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(model_snapshot.ModelSnapshotError, match="authoritative repository bytes"):
        model_snapshot.verify_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({}, "must be"),
        (
            {
                "schema_version": 3,
                "model_profile": model_snapshot.QWEN3_8_27B_PROFILE,
                "repository": model_snapshot.QWEN3_8_27B_REPO,
                "revision": QWEN3_8_27B_REVISION,
                "files": [],
            },
            "contains no files",
        ),
    ],
)
def test_verify_model_snapshot_rejects_invalid_manifests(tmp_path, manifest: dict, message: str) -> None:
    """Reject manifests that omit the pin or contain no repository files.

    Args:
        tmp_path: Pytest directory receiving the invalid manifest.
        manifest: Invalid model identity record.
        message: Expected validation failure fragment.
    """
    (tmp_path / model_snapshot.MODEL_INTEGRITY_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(model_snapshot.ModelSnapshotError, match=message):
        model_snapshot.verify_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)


def test_verify_model_snapshot_requires_a_local_manifest(tmp_path) -> None:
    """Fail offline verification when the prepared integrity record is absent.

    Args:
        tmp_path: Empty Pytest directory used as the model root.
    """
    with pytest.raises(model_snapshot.ModelSnapshotError, match="manifest is unreadable"):
        model_snapshot.verify_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)


def test_prepare_model_snapshot_rejects_locally_corrupted_pinned_files(monkeypatch, tmp_path) -> None:
    """Refuse to bless local bytes that differ from pinned repository metadata.

    Args:
        monkeypatch: Pytest fixture used to replace Hugging Face network calls.
        tmp_path: Pytest directory containing a corrupted local checkpoint.
    """
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"corrupt")
    repository_api = SimpleNamespace(
        list_repo_tree=Mock(
            return_value=[
                SimpleNamespace(
                    path="model.safetensors",
                    size=model_file.stat().st_size,
                    blob_id="lfs-pointer",
                    lfs=SimpleNamespace(sha256=hashlib.sha256(b"correct").hexdigest()),
                )
            ]
        )
    )
    monkeypatch.setattr(model_snapshot, "snapshot_download", Mock())
    monkeypatch.setattr(model_snapshot, "HfApi", Mock(return_value=repository_api))

    with pytest.raises(model_snapshot.ModelSnapshotError, match="authoritative repository bytes"):
        model_snapshot.prepare_model_snapshot(tmp_path, model_snapshot.QWEN3_8_27B_PROFILE)


@pytest.mark.parametrize("operation", ["prepare", "verify"])
def test_model_snapshot_rejects_unknown_profiles(monkeypatch, tmp_path, operation: str) -> None:
    """Reject unknown model names before any network or filesystem work.

    Args:
        monkeypatch: Pytest fixture used to guard Hugging Face network calls.
        tmp_path: Pytest directory that must remain untouched.
        operation: Snapshot operation under test.
    """
    snapshot_download = Mock()
    monkeypatch.setattr(model_snapshot, "snapshot_download", snapshot_download)

    with pytest.raises(model_snapshot.ModelSnapshotError, match="Unsupported model snapshot profile"):
        if operation == "prepare":
            model_snapshot.prepare_model_snapshot(tmp_path, "unknown")
        else:
            model_snapshot.verify_model_snapshot(tmp_path, "unknown")

    snapshot_download.assert_not_called()
    assert list(tmp_path.iterdir()) == []
