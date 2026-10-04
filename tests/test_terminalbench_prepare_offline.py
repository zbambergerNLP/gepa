"""Test package verification and failure-safe sealing with external builds simulated."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from examples.terminalbench import prepare_offline
from gepa.adapters.terminal_bench_adapter.staging import load_offline_task_bundle, singularity_image_filename


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    """Provide real package hashes and loader checks with simulated Harbor/Apptainer I/O."""
    manifest = prepare_offline.load_terminalbench_manifest(prepare_offline.MANIFEST_PATH)
    contents = {}
    refs = dict(manifest.task_refs)
    for task_id, image in prepare_offline.SUPPORTED_IMAGES.items():
        files = {
            "task.toml": f'[task]\nname = "{task_id}"\n[environment]\ndocker_image = "{image}"\n',
            "instruction.md": "Fixture task.\n",
            "tests/test.sh": "#!/bin/bash\nexit 0\n",
        }
        contents[task_id] = files
        hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
        refs[task_id] = (
            "sha256:"
            + hashlib.sha256("".join(f"{name}\0{sha}\n" for name, sha in sorted(hashes.items())).encode()).hexdigest()
        )
    manifest = replace(manifest, task_refs=refs)
    monkeypatch.setattr(prepare_offline, "load_terminalbench_manifest", lambda _: manifest)
    monkeypatch.setattr(prepare_offline.shutil, "which", lambda executable: "/tools/" + executable)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="0.22.0\n")
        if command[1:3] == ["tasks", "download"]:
            task_id, ref = command[3].split("@")
            assert ref == manifest.task_refs[task_id]
            assert command[4] == "--output-dir" and command[6:] == ["--export"]
            task_dir = Path(command[5]) / task_id.split("/")[-1]
            for name, content in contents[task_id].items():
                path = task_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        elif command[1] == "pull":
            Path(command[2]).write_bytes(b"raw image bytes")
        elif command[1:3] == ["build", "--fakeroot"]:
            Path(command[3]).write_bytes(b"prepared image bytes")
        else:
            pytest.fail(f"Unexpected subprocess: {command}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_offline.subprocess, "run", run)
    return manifest, contents, calls, run


def test_preparation_seals_real_loader_validated_packages_in_training_order(tmp_path, monkeypatch, preparation):
    manifest, _, calls, _ = preparation
    monkeypatch.setenv("PYTHONPATH", "/unrelated/python/path")
    monkeypatch.setenv("PYTHONHOME", "/unrelated/python/home")
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated/venv")
    cache = tmp_path / "raw images"
    cache.mkdir()
    first_image = next(iter(prepare_offline.SUPPORTED_IMAGES.values()))
    cached = cache / singularity_image_filename(first_image)
    cached.write_bytes(b"already cached base image")
    sealed = prepare_offline.prepare_bundle(tmp_path / "bundle", cache)
    bundle = load_offline_task_bundle(sealed, manifest)
    assert list(bundle.payload["tasks"]) == manifest.splits["train"][:2]
    assert not (sealed.parent / "bundle.pending.json").exists()
    assert cached.read_bytes() == b"already cached base image"
    pulls = [command for command, _ in calls if command[1] == "pull"]
    assert len(pulls) == 1
    assert pulls[0][-1] == "docker://alexgshaw/log-summary-date-ranges:20251031"
    for _, kwargs in calls:
        assert kwargs["check"] is True
        assert all(key not in kwargs["env"] for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"))
    for task in bundle.payload["tasks"].values():
        assert task["base_image_sha256"] == prepare_offline.file_sha256(
            cache / singularity_image_filename(task["docker_image"])
        )
        assert task["image_sha256"] == hashlib.sha256(b"prepared image bytes").hexdigest()


def test_download_pin_tampering_fails_before_pull_or_build(tmp_path, preparation):
    _, contents, calls, _ = preparation
    contents["terminal-bench/fix-ocaml-gc"]["tests/test.sh"] = "changed verifier\n"
    target = tmp_path / "bundle"
    with pytest.raises(ValueError, match="pinned package ref"):
        prepare_offline.prepare_bundle(target, tmp_path / "cache")
    assert not (target / "bundle.json").exists()
    assert all(command[1] not in ("pull", "build") for command, _ in calls)


@pytest.mark.parametrize("count", [-1, 0, 3, True])
def test_unsupported_task_counts_fail_before_external_work(tmp_path, monkeypatch, count):
    process = Mock()
    monkeypatch.setattr(prepare_offline.subprocess, "run", process)
    target = tmp_path / "bundle"
    with pytest.raises(ValueError, match="1 or 2"):
        prepare_offline.prepare_bundle(target, tmp_path / "cache", task_count=count)
    process.assert_not_called()
    assert not target.exists()


def test_unsupported_recipe_and_changed_train_order_fail_closed(tmp_path, monkeypatch, preparation):
    manifest, _, calls, _ = preparation
    with pytest.raises(ValueError, match="No verified offline recipe"):
        prepare_offline.render_recipe("terminal-bench/mteb-leaderboard", "sha256:unused", tmp_path / "base.sif")
    changed = replace(manifest, splits={**manifest.splits, "train": list(reversed(manifest.splits["train"]))})
    monkeypatch.setattr(prepare_offline, "load_terminalbench_manifest", lambda _: changed)
    with pytest.raises(ValueError, match="training prefix"):
        prepare_offline.prepare_bundle(tmp_path / "bundle", tmp_path / "cache")
    assert not calls


@pytest.mark.parametrize("stage", ["download", "pull", "build", "validate"])
def test_failed_preparation_never_seals_a_usable_bundle(tmp_path, monkeypatch, preparation, stage):
    _, _, _, real_run = preparation

    def failing_run(command, **kwargs):
        if stage in command:
            raise subprocess.CalledProcessError(1, command)
        result = real_run(command, **kwargs)
        if stage == "validate" and command[1:3] == ["build", "--fakeroot"]:
            (tmp_path / "bundle/tasks/fix-ocaml-gc/tests/test.sh").write_text("changed after build\n")
        return result

    monkeypatch.setattr(prepare_offline.subprocess, "run", failing_run)
    target = tmp_path / "bundle"
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        prepare_offline.prepare_bundle(target, tmp_path / "cache", task_count=1)
    assert not (target / "bundle.json").exists()
    with pytest.raises(FileExistsError, match="fresh bundle directory"):
        prepare_offline.prepare_bundle(target, tmp_path / "cache", task_count=1)


def test_existing_sealed_bundle_is_never_overwritten(tmp_path, preparation):
    _, _, calls, _ = preparation
    target = tmp_path / "bundle"
    target.mkdir()
    sealed = target / "bundle.json"
    sealed.write_text('{"existing": true}\n')
    with pytest.raises(FileExistsError, match="fresh bundle directory"):
        prepare_offline.prepare_bundle(target, tmp_path / "cache")
    assert json.loads(sealed.read_text()) == {"existing": True}
    assert not calls


def test_ocaml_recipe_configures_only_exact_source_url_and_shallow_tag(tmp_path):
    recipe = prepare_offline.render_recipe("terminal-bench/fix-ocaml-gc", "sha256:pinned", tmp_path / "base.sif")
    config_path = tmp_path / "gitconfig"
    for line in recipe.splitlines():
        if line.strip().startswith("git config --system "):
            command = shlex.split(line)
            subprocess.run(command[:2] + ["--file", str(config_path)] + command[3:], check=True)
    configured = subprocess.run(
        ["git", "config", "--file", str(config_path), "--list"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    assert configured == [
        "url.file:///opt/harbor-offline/ocaml.git.insteadof=https://github.com/sadiqj/ocaml/",
        "protocol.file.allow=always",
    ]
    clone = next(shlex.split(line) for line in recipe.splitlines() if line.strip().startswith("git clone "))
    assert clone == [
        "git",
        "clone",
        "--bare",
        "--depth",
        "1",
        "--single-branch",
        "--branch",
        "tag_purposefully_broken_sweeping_changes",
        "https://github.com/sadiqj/ocaml/",
        "/opt/harbor-offline/ocaml.git",
    ]
    assert prepare_offline.OCAML_REF in recipe
    assert "uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version" in recipe


@pytest.mark.parametrize("task_id", list(prepare_offline.SUPPORTED_IMAGES))
def test_recipe_keeps_build_and_runtime_python_and_cache_in_the_same_visible_paths(tmp_path, task_id):
    recipe = prepare_offline.render_recipe(task_id, "sha256:pinned", tmp_path / "base.sif")
    post, environment = recipe.split("%post\n", 1)[1].split("%environment\n", 1)
    environment = environment.split("%labels\n", 1)[0]
    for variable, path in (
        ("PATH", "/opt/harbor-offline/home/.local/bin:$PATH"),
        ("UV_CACHE_DIR", "/opt/harbor-offline/uv-cache"),
        ("UV_PYTHON_INSTALL_DIR", "/opt/harbor-offline/python"),
    ):
        assignment = f"export {variable}={path}"
        assert assignment in post and assignment in environment
        assert post.index(assignment) < post.index("    uvx ")
    assert "export HOME=/opt/harbor-offline/home" in environment
    assert "export UV_OFFLINE=1" in environment
