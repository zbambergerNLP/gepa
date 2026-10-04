"""Test package verification and failure-safe sealing with external builds simulated."""

from __future__ import annotations

import hashlib
import json
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
    specs = prepare_offline.load_recipe_spec(manifest)
    contents = {}
    refs = dict(manifest.task_refs)
    for task_id, spec in specs.items():
        image = spec["docker_image"]
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
        spec["task_ref"] = refs[task_id]
        spec["verifier_script_sha256"] = hashes["tests/test.sh"]
    manifest = replace(manifest, task_refs=refs)
    monkeypatch.setattr(prepare_offline, "load_terminalbench_manifest", lambda _: manifest)
    monkeypatch.setattr(prepare_offline, "load_recipe_spec", lambda _: specs)
    monkeypatch.setattr(prepare_offline, "resource_recipe_commands", lambda task_id, task_ref: [])
    monkeypatch.setattr(prepare_offline, "uv_installer_recipe_commands", lambda version: [])
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
        elif command[1] == "exec":
            assert (
                command[2 : 2 + len(prepare_offline.PROBE_CONTRACT["flags"])] == prepare_offline.PROBE_CONTRACT["flags"]
            )
            if "/opt/harbor-server/bin/python3" not in command:
                assert "UV_OFFLINE=1" in command and "PIP_NO_INDEX=1" in command
            assert "/tests/" not in " ".join(command)
            return subprocess.CompletedProcess(command, 0, stdout="offline dependency probe passed\n", stderr="")
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
        assert isinstance(kwargs["check"], bool)
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
        prepare_offline.render_recipe("terminal-bench/unknown", "sha256:unused", tmp_path / "base.sif", {}, [], [])
    changed = replace(manifest, splits={**manifest.splits, "train": list(reversed(manifest.splits["train"]))})
    monkeypatch.setattr(prepare_offline, "load_terminalbench_manifest", lambda _: changed)
    with pytest.raises(ValueError, match="training prefix"):
        prepare_offline.prepare_bundle(tmp_path / "bundle", tmp_path / "cache")
    assert not calls


@pytest.mark.parametrize("stage", ["download", "pull", "build", "exec", "validate"])
def test_failed_preparation_never_seals_a_usable_bundle(tmp_path, monkeypatch, preparation, stage):
    _, _, _, real_run = preparation

    def failing_run(command, **kwargs):
        if stage in command:
            raise subprocess.CalledProcessError(1, command)
        result = real_run(command, **kwargs)
        if stage == "validate" and command[1:3] == ["build", "--fakeroot"]:
            (tmp_path / "bundle/parts/fix-ocaml-gc/tasks/fix-ocaml-gc/tests/test.sh").write_text(
                "changed after build\n"
            )
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


def test_resource_recipe_preserves_heredoc_syntax_and_runtime_environment(tmp_path):
    manifest = prepare_offline.load_terminalbench_manifest(prepare_offline.MANIFEST_PATH)
    task_id = manifest.splits["train"][0]
    spec = prepare_offline.load_recipe_spec(manifest)[task_id]
    heredoc = "cat > /opt/harbor-offline/resource-env.sh <<'ENV'\nexport HF_HUB_OFFLINE=1\nENV"
    recipe = prepare_offline.render_recipe(task_id, spec["task_ref"], tmp_path / "base.sif", spec, [heredoc], [])
    post = recipe.split("%post\n", 1)[1].split("%environment\n", 1)[0]
    subprocess.run(["sh", "-n"], input=post, text=True, check=True)
    assert heredoc in recipe
    assert ". /opt/harbor-offline/resource-env.sh" in recipe


@pytest.mark.parametrize("task_id", list(prepare_offline.SUPPORTED_IMAGES))
def test_recipe_keeps_build_and_runtime_python_and_cache_in_the_same_visible_paths(tmp_path, task_id):
    manifest = prepare_offline.load_terminalbench_manifest(prepare_offline.MANIFEST_PATH)
    spec = prepare_offline.load_recipe_spec(manifest)[task_id]
    recipe = prepare_offline.render_recipe(task_id, spec["task_ref"], tmp_path / "base.sif", spec, [], [])
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


def test_curated_dependency_specs_cover_all_pins_and_preserve_runtime_exceptions():
    manifest = prepare_offline.load_terminalbench_manifest(prepare_offline.MANIFEST_PATH)
    recipes = prepare_offline.load_recipe_spec(manifest)
    assert len(recipes) == 89
    assert {task_id: recipe["task_ref"] for task_id, recipe in recipes.items()} == manifest.task_refs
    assert recipes["terminal-bench/reshard-c4-data"]["uv_version"] == "0.8.15"
    assert recipes["terminal-bench/mailman"]["mode"] == "uv_venv"
    assert recipes["terminal-bench/mailman"]["python"] == "3.12"
    assert recipes["terminal-bench/sam-cell-seg"]["python"] == "3.11"
    assert recipes["terminal-bench/train-fasttext"]["python"] == "3.11"
    assert recipes["terminal-bench/kv-store-grpc"]["mode"] == "pip"
    assert "pytest==8.4.2" in recipes["terminal-bench/kv-store-grpc"]["packages"]
    for recipe in recipes.values():
        assert recipe["probe_modules"]
        assert "/tests/" not in " ".join(prepare_offline.dependency_probe_argv(recipe))


@pytest.mark.parametrize("tampering", ["task_ref", "mode", "uv_version", "python", "missing"])
def test_curated_dependency_spec_rejects_drift(tmp_path, monkeypatch, tampering):
    manifest = prepare_offline.load_terminalbench_manifest(prepare_offline.MANIFEST_PATH)
    document = json.loads(prepare_offline.RECIPE_SPEC_PATH.read_text())
    task_id = manifest.splits["train"][0]
    if tampering == "missing":
        del document["tasks"][task_id]
    else:
        document["tasks"][task_id][tampering] = "unsupported"
    altered = tmp_path / "recipes.json"
    altered.write_text(json.dumps(document))
    monkeypatch.setattr(prepare_offline, "RECIPE_SPEC_PATH", altered)
    with pytest.raises(ValueError):
        prepare_offline.load_recipe_spec(manifest)


def test_all_tasks_seals_every_split_in_manifest_order_without_grader_execution(tmp_path, preparation):
    manifest, _, calls, _ = preparation
    path = prepare_offline.prepare_bundle(tmp_path / "bundle", tmp_path / "cache", all_tasks=True)
    bundle = load_offline_task_bundle(path, manifest)
    expected = [task_id for split in ("train", "val", "test") for task_id in manifest.splits[split]]
    assert list(bundle.payload["tasks"]) == expected
    assert len([command for command, _ in calls if command[1] == "build"]) == 89
    specs = prepare_offline.load_recipe_spec(manifest)
    expected_probes = 178 + sum(bool(spec["uv_version"]) for spec in specs.values())
    assert len([command for command, _ in calls if command[1] == "exec"]) == expected_probes
    for task_id, task in bundle.payload["tasks"].items():
        part = path.parent / "parts" / task_id.split("/")[-1]
        assert load_offline_task_bundle(part / "bundle.json", manifest).payload["tasks"][task_id]["dependency_probe"][
            "offline"
        ]
        probes = 3 if specs[task_id]["uv_version"] else 2
        assert (part / "dependency-probe.log").read_text() == "offline dependency probe passed\n" * probes
        image_name = singularity_image_filename(task["docker_image"])
        assert (part / "images" / image_name).samefile(path.parent / "images" / image_name)


def test_resume_reuses_verified_parts_after_a_later_build_fails(tmp_path, monkeypatch, preparation):
    manifest, _, calls, real_run = preparation
    target, cache = tmp_path / "bundle", tmp_path / "cache"

    def fail_second_build(command, **kwargs):
        if command[1] == "build" and "log-summary-date-ranges" in command[3]:
            Path(command[3]).write_bytes(b"interrupted build")
            raise subprocess.CalledProcessError(1, command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(prepare_offline.subprocess, "run", fail_second_build)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_offline.prepare_bundle(target, cache)
    part = target / "parts/fix-ocaml-gc/bundle.json"
    first_bytes = part.read_bytes()
    assert not (target / "bundle.json").exists()
    monkeypatch.setattr(prepare_offline.subprocess, "run", real_run)
    monkeypatch.setattr(
        prepare_offline,
        "resource_recipe_commands",
        lambda task_id, task_ref: ["echo 'new resource preparation'"]
        if task_id.endswith("log-summary-date-ranges")
        else [],
    )
    calls.clear()
    sealed = prepare_offline.prepare_bundle(target, cache, resume=True)
    assert list(load_offline_task_bundle(sealed, manifest).payload["tasks"]) == manifest.splits["train"][:2]
    assert part.read_bytes() == first_bytes
    assert len([command for command, _ in calls if command[1] == "build"]) == 1
    assert not any(command[1:3] == ["tasks", "download"] for command, _ in calls)


@pytest.mark.parametrize("change", ["task", "resource", "plan"])
def test_resume_rejects_changed_parts_or_preparation_inputs(tmp_path, monkeypatch, preparation, change):
    _, _, calls, _ = preparation
    target, cache = tmp_path / "bundle", tmp_path / "cache"
    sealed = prepare_offline.prepare_bundle(target, cache)
    if change == "task":
        (target / "parts/fix-ocaml-gc/tasks/fix-ocaml-gc/tests/test.sh").write_text("changed\n")
    elif change == "resource":
        monkeypatch.setattr(prepare_offline, "resource_recipe_commands", lambda task_id, task_ref: ["different recipe"])
    else:
        (target / "preparation-plan.json").unlink()
    before = sealed.read_bytes()
    calls.clear()
    with pytest.raises(ValueError):
        prepare_offline.prepare_bundle(target, cache, resume=True)
    assert sealed.read_bytes() == before
    assert not calls


@pytest.mark.parametrize("task_id", ["terminal-bench/qemu-alpine-ssh", "terminal-bench/qemu-startup"])
def test_bullseye_server_uses_managed_python_without_replacing_task_python(tmp_path, task_id):
    manifest = prepare_offline.load_terminalbench_manifest(prepare_offline.MANIFEST_PATH)
    spec = prepare_offline.load_recipe_spec(manifest)[task_id]
    recipe = prepare_offline.render_recipe(task_id, spec["task_ref"], tmp_path / "base.sif", spec, [], [])
    server = "uv venv --managed-python -p 3.13 /opt/harbor-server"
    assert recipe.index("export UV_PYTHON_INSTALL_DIR=") < recipe.index(server)
    assert recipe.index("UV_INSTALL_DIR=") < recipe.index(server)
    assert "uv pip install --python /opt/harbor-server/bin/python3 'fastapi==0.142.2' 'uvicorn==0.54.0'" in recipe
    assert "/usr/bin/python3 -m venv /opt/harbor-server" not in recipe
    assert "ln -s" not in recipe and "update-alternatives" not in recipe


def test_targeted_preparation_preserves_manifest_order_and_skips_other_tasks(tmp_path, preparation):
    manifest, _, calls, _ = preparation
    selected = ["terminal-bench/sam-cell-seg", "terminal-bench/qemu-startup", "terminal-bench/install-windows-3.11"]
    sealed = prepare_offline.prepare_bundle(tmp_path / "bundle", tmp_path / "cache", task_ids=selected)
    bundle = load_offline_task_bundle(sealed, manifest)
    expected = [
        task_id for split in ("train", "val", "test") for task_id in manifest.splits[split] if task_id in selected
    ]
    assert list(bundle.payload["tasks"]) == expected
    assert len([command for command, _ in calls if command[1] == "build"]) == 3


@pytest.mark.parametrize("selected", [[], ["terminal-bench/not-pinned"], ["terminal-bench/fix-ocaml-gc"] * 2])
def test_targeted_preparation_rejects_unknown_or_duplicate_tasks(tmp_path, preparation, selected):
    _, _, calls, _ = preparation
    with pytest.raises(ValueError, match="unique canonical task names"):
        prepare_offline.prepare_bundle(tmp_path / "bundle", tmp_path / "cache", task_ids=selected)
    assert not calls


def test_failed_dependency_probe_retains_diagnostics_and_does_not_seal(tmp_path, monkeypatch, preparation):
    _, _, _, real_run = preparation

    def missing_dependency(command, **kwargs):
        if command[1] == "exec":
            return subprocess.CompletedProcess(command, 1, stdout="Python3.13\n", stderr="missing cached dependency\n")
        return real_run(command, **kwargs)

    monkeypatch.setattr(prepare_offline.subprocess, "run", missing_dependency)
    target = tmp_path / "bundle"
    with pytest.raises(subprocess.CalledProcessError):
        prepare_offline.prepare_bundle(target, tmp_path / "cache", task_count=1)
    part = target / "parts/fix-ocaml-gc"
    assert (part / "dependency-probe.log").read_text() == "Python3.13\nmissing cached dependency\n"
    assert (part / "dependency-probe-commands.json").is_file()
    assert not (part / "bundle.json").exists()
    assert not (target / "bundle.json").exists()
