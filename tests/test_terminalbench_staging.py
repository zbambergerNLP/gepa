"""Verify offline task staging preserves pinned sources and runtime provenance."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
from terminalbench_staging_fixtures import make_offline_bundle

from gepa.adapters.terminal_bench_adapter import HarborCLI
from gepa.adapters.terminal_bench_adapter.staging import load_offline_task_bundle, singularity_image_filename


def runner(tmp_path: Path, manifest, bundle_path: Path, **kwargs):
    """Create the offline runner without contacting any external process."""
    return HarborCLI(
        manifest=manifest,
        student_model="hosted_vllm/Qwen/Qwen3.8-27B",
        work_dir=tmp_path / "harbor",
        agent_python_path=Path(__file__).parents[1],
        container_runtime="singularity",
        offline_task_bundle=bundle_path,
        **kwargs,
    )


def job_config(harbor, task_ids, tmp_path):
    """Build a candidate-specific job while retaining the normal benchmark settings."""
    return harbor.build_job_config(
        task_ids,
        prompt_path=tmp_path / "prompt.txt",
        bundle_path=tmp_path / "documents.json",
        jobs_dir=tmp_path / "jobs",
        job_name="offline-test",
    )


def test_offline_job_uses_verified_local_tasks_in_requested_order(tmp_path):
    manifest, path, payload = make_offline_bundle(tmp_path)
    harbor = runner(tmp_path, manifest, path)
    ordered = list(reversed(payload["tasks"]))
    config = job_config(harbor, ordered, tmp_path)
    assert "datasets" not in config
    assert config["tasks"] == [
        {"path": str(path.parent / payload["tasks"][task_id]["path"]), "source": manifest.dataset["reference"]}
        for task_id in ordered
    ]
    assert config["environment"]["kwargs"] == {"singularity_image_cache_dir": str(path.parent / "images")}
    assert config["n_attempts"] == 1
    assert config["retry"]["max_retries"] == 0
    assert config["timeout_multiplier"] == 1.0
    assert config["agents"][0]["import_path"] == "examples.terminalbench.terminus_agent:PromptedTerminus"


@pytest.mark.parametrize("artifact", ["task", "image", "recipe", "extra", "missing", "manifest"])
def test_offline_job_rechecks_task_and_runtime_bytes_after_loading(tmp_path, artifact):
    manifest, path, payload = make_offline_bundle(tmp_path)
    harbor = runner(tmp_path, manifest, path)
    task_id, task = next(iter(payload["tasks"].items()))
    if artifact == "task":
        (path.parent / task["path"] / "tests/test.sh").write_text("altered verifier")
    elif artifact == "image":
        (path.parent / "images" / singularity_image_filename(task["docker_image"])).write_bytes(b"different runtime")
    elif artifact == "recipe":
        (path.parent / task["recipe"]["path"]).write_text("different recipe")
    elif artifact == "extra":
        (path.parent / task["path"] / "extra.py").write_text("untracked runtime input")
    elif artifact == "missing":
        (path.parent / task["path"] / "instruction.md").unlink()
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        job_config(harbor, [task_id], tmp_path)


def test_resealing_changed_package_files_cannot_replace_the_pinned_source(tmp_path):
    manifest, path, payload = make_offline_bundle(tmp_path)
    task = next(iter(payload["tasks"].values()))
    changed = path.parent / task["path"] / "tests/test.sh"
    changed.write_text("altered verifier")
    task["files"]["tests/test.sh"] = hashlib.sha256(changed.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pinned package ref"):
        load_offline_task_bundle(path, manifest)


@pytest.mark.parametrize("change", ["version", "dataset", "ref", "unknown", "name", "image", "cache", "recipe"])
def test_offline_bundle_rejects_incompatible_metadata(tmp_path, change):
    manifest, path, payload = make_offline_bundle(tmp_path)
    task_id, task = next(iter(payload["tasks"].items()))
    if change == "version":
        payload["harbor_version"] = "0.21.0"
    elif change == "dataset":
        payload["dataset_ref"] = "sha256:" + "0" * 64
    elif change == "ref":
        task["ref"] = "sha256:" + "0" * 64
    elif change == "unknown":
        payload["tasks"]["terminal-bench/unknown"] = payload["tasks"].pop(task_id)
    elif change == "name":
        payload["tasks"][manifest.splits["test"][0]] = payload["tasks"].pop(task_id)
    elif change == "image":
        task["docker_image"] = "some/other:image"
    elif change == "cache":
        payload["image_cache_dir"] = "../images"
    else:
        task["recipe"]["path"] = str((path.parent / task["recipe"]["path"]).resolve())
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_offline_task_bundle(path, manifest)


@pytest.mark.parametrize("target", ["task", "image", "recipe"])
def test_offline_bundle_rejects_symlinked_artifacts(tmp_path, target):
    manifest, path, payload = make_offline_bundle(tmp_path)
    task = next(iter(payload["tasks"].values()))
    artifact = {
        "task": path.parent / task["path"] / "tests/test.sh",
        "image": path.parent / "images" / singularity_image_filename(task["docker_image"]),
        "recipe": path.parent / task["recipe"]["path"],
    }[target]
    moved = tmp_path / "outside"
    artifact.rename(moved)
    artifact.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        load_offline_task_bundle(path, manifest)


def test_runtime_identity_tracks_recipe_and_image_but_not_storage_location(tmp_path):
    manifest, path, payload = make_offline_bundle(tmp_path)
    original = load_offline_task_bundle(path, manifest).contract()
    moved = tmp_path / "moved"
    shutil.copytree(path.parent, moved)
    assert load_offline_task_bundle(moved / "bundle.json", manifest).contract() == original
    task_id, task = next(iter(payload["tasks"].items()))
    recipe = moved / task["recipe"]["path"]
    recipe.write_text("changed runtime preparation")
    task["recipe"]["sha256"] = hashlib.sha256(recipe.read_bytes()).hexdigest()
    image = moved / "images" / singularity_image_filename(task["docker_image"])
    image.write_bytes(b"rebuilt image")
    task["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    (moved / "bundle.json").write_text(json.dumps(payload))
    changed = load_offline_task_bundle(moved / "bundle.json", manifest).contract()
    assert changed["tasks"][task_id]["ref"] == original["tasks"][task_id]["ref"]
    assert changed["tasks"][task_id]["recipe_sha256"] != original["tasks"][task_id]["recipe_sha256"]
    assert changed["tasks"][task_id]["image_sha256"] != original["tasks"][task_id]["image_sha256"]


def test_offline_job_rejects_unstaged_task_before_harbor_execution(tmp_path, monkeypatch):
    manifest, path, _ = make_offline_bundle(tmp_path)
    harbor = runner(tmp_path, manifest, path)
    process = Mock()
    monkeypatch.setattr("gepa.adapters.terminal_bench_adapter.terminal_bench_adapter.subprocess.run", process)
    with pytest.raises(ValueError, match="requested tasks"):
        job_config(harbor, manifest.splits["test"][:1], tmp_path)
    process.assert_not_called()


def test_offline_profile_rejects_a_different_image_cache_or_backend(tmp_path):
    manifest, path, _ = make_offline_bundle(tmp_path)
    with pytest.raises(ValueError, match="must match"):
        runner(tmp_path, manifest, path, singularity_image_cache_dir=tmp_path / "other-images")
    with pytest.raises(ValueError, match="requires container_runtime singularity"):
        HarborCLI(
            manifest=manifest,
            student_model="fixture",
            work_dir=tmp_path,
            agent_python_path=tmp_path,
            offline_task_bundle=path,
        )
