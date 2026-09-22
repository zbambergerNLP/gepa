"""Tests for realized Python environment manifests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common import python_environment


def test_environment_manifest_is_deterministic_and_complete(monkeypatch) -> None:
    """Record the interpreter, packages, and stable runtime files.

    Args:
        monkeypatch: Pytest fixture used to avoid hashing the test environment.
    """
    file_records = [{"kind": "file", "path": "lib/runtime.so", "size": 3, "sha256": "a" * 64}]
    monkeypatch.setattr(
        python_environment,
        "_environment_file_records",
        Mock(return_value=file_records),
    )
    monkeypatch.setattr(
        python_environment,
        "_editable_source_record",
        Mock(return_value=None),
    )
    first = python_environment.build_environment_manifest()
    second = python_environment.build_environment_manifest()

    assert first == second
    assert first["schema_version"] == python_environment.ENVIRONMENT_MANIFEST_SCHEMA_VERSION
    assert len(first["python"]["executable_sha256"]) == 64
    assert first["distributions"]
    assert first["editable_sources"] == []
    assert first["files"] == file_records
    ordering = [
        (record["normalized_name"], record["version"], record["location"])
        for record in first["distributions"]
    ]
    assert ordering == sorted(ordering)
    assert all(
        set(record["metadata_sha256"]) == set(python_environment._DISTRIBUTION_METADATA_FILES)
        for record in first["distributions"]
    )


def test_environment_file_records_detect_byte_drift_and_ignore_bytecode(tmp_path: Path) -> None:
    """Hash actual package bytes while ignoring derived Python caches.

    Args:
        tmp_path: Synthetic virtual-environment prefix under test.
    """
    package_file = tmp_path / "lib" / "python3.11" / "site-packages" / "runtime.so"
    package_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"first")
    cache_file = package_file.parent / "__pycache__" / "runtime.pyc"
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"derived")
    symlink = tmp_path / "bin" / "runtime"
    symlink.parent.mkdir()
    symlink.symlink_to(package_file)

    first = python_environment._environment_file_records(tmp_path)
    package_file.write_bytes(b"other")
    second = python_environment._environment_file_records(tmp_path)

    assert first != second
    assert not any("__pycache__" in record["path"] for record in first)
    assert {record["kind"] for record in first} == {"file", "symlink"}


def test_editable_source_record_requires_a_clean_exact_git_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Record clean editable source and reject local source drift.

    Args:
        monkeypatch: Pytest fixture used to supply direct-URL metadata.
        tmp_path: Synthetic editable Git checkout under test.
    """
    subprocess_commands = [
        SimpleNamespace(stdout=f"{tmp_path}\n"),
        SimpleNamespace(stdout=f"{'a' * 40}\n"),
        SimpleNamespace(stdout=""),
    ]
    monkeypatch.setattr(
        python_environment.subprocess,
        "run",
        Mock(side_effect=subprocess_commands),
    )
    distribution = Mock()
    distribution.read_text.return_value = json.dumps(
        {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    )

    record = python_environment._editable_source_record(distribution)

    assert record == {
        "repository_root": tmp_path.as_posix(),
        "source_subpath": ".",
        "commit": "a" * 40,
    }

    python_environment.subprocess.run.side_effect = [
        SimpleNamespace(stdout=f"{tmp_path}\n"),
        SimpleNamespace(stdout=f"{'a' * 40}\n"),
        SimpleNamespace(stdout=" M runtime.py\n"),
    ]
    with pytest.raises(python_environment.PythonEnvironmentError, match="local changes"):
        python_environment._editable_source_record(distribution)


def test_prepare_and_verify_environment_manifest(monkeypatch, tmp_path: Path) -> None:
    """Persist one canonical snapshot and reject later package drift.

    Args:
        monkeypatch: Pytest fixture used to replace environment discovery.
        tmp_path: Isolated directory receiving the manifest.
    """
    manifest = {
        "schema_version": python_environment.ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        "python": {"version": "3.11.13"},
        "distributions": [{"name": "vllm", "version": "0.17.0"}],
    }
    monkeypatch.setattr(
        python_environment,
        "build_environment_manifest",
        Mock(return_value=manifest),
    )
    path = tmp_path / "posit-environment.json"

    prepared_digest = python_environment.prepare_environment_manifest(path)
    verified_digest = python_environment.verify_environment_manifest(path)

    assert prepared_digest == verified_digest
    assert json.loads(path.read_text(encoding="utf-8")) == manifest

    drifted = {**manifest, "distributions": [{"name": "vllm", "version": "0.18.0"}]}
    monkeypatch.setattr(
        python_environment,
        "build_environment_manifest",
        Mock(return_value=drifted),
    )
    with pytest.raises(python_environment.PythonEnvironmentError, match="differs from its frozen manifest"):
        python_environment.verify_environment_manifest(path)


@pytest.mark.parametrize("content", ["not-json", '{"schema_version": 0}'])
def test_verify_environment_manifest_rejects_invalid_files(tmp_path: Path, content: str) -> None:
    """Reject malformed JSON and unsupported manifest schemas.

    Args:
        tmp_path: Isolated directory receiving the invalid manifest.
        content: Invalid manifest text under test.
    """
    path = tmp_path / "posit-environment.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(python_environment.PythonEnvironmentError):
        python_environment.verify_environment_manifest(path)
