"""Exercise public-resource preparation and genuine offline curl transport."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from examples.terminalbench import offline_resources as resources


def _run_shell(commands, *, env=None, cwd=None, check=True):
    return subprocess.run(
        ["sh", "-eu", "-c", "\n".join(commands)],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=check,
    )


@pytest.fixture
def curl_cache(tmp_path):
    content = b"original public resource\n\x00binary bytes\xff"
    artifact = tmp_path / "asset.tar.gz"
    artifact.write_bytes(content)
    url = "https://public.example/releases/v1/asset.tar.gz"
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({url: {"path": str(artifact), "sha256": hashlib.sha256(content).hexdigest()}}))
    script = tmp_path / "cached-curl.py"
    script.write_text(
        resources.CURL_WRAPPER.replace('"/opt/harbor-offline/resource-urls.json"', repr(str(mapping))).replace(
            '"/opt/harbor-offline/original-curl"', repr(shutil.which("curl"))
        )
    )

    def run(*args):
        return subprocess.run([sys.executable, str(script), *args], cwd=tmp_path, capture_output=True)

    return run, url, artifact, content, tmp_path, mapping


@pytest.mark.parametrize(
    "options",
    [[], ["-fsSL"], ["--proto", "=https", "--tlsv1.2", "--silent", "--show-error"]],
)
def test_cached_curl_preserves_stdout_bytes(curl_cache, options):
    run, url, _, content, _, _ = curl_cache
    result = run(*options, url)
    assert result.returncode == 0, result.stderr
    assert result.stdout == content


@pytest.mark.parametrize("arguments", [["-o", "result.bin"], ["-oresult.bin"], ["--output", "result.bin"]])
def test_cached_curl_preserves_file_output(curl_cache, arguments):
    run, url, _, content, directory, _ = curl_cache
    result = run("-fsSL", *arguments, "--url", url)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b""
    assert (directory / "result.bin").read_bytes() == content


def test_cached_curl_remote_name_uses_alias_basename(curl_cache):
    run, _, artifact, content, directory, mapping = curl_cache
    alias = directory / "source" / "install.sh"
    alias.parent.mkdir()
    alias.symlink_to(artifact)
    url = "https://public.example/version/install.sh"
    mapping.write_text(json.dumps({url: {"path": str(alias), "sha256": hashlib.sha256(content).hexdigest()}}))
    result = run("-sSO", "--url", url)
    assert result.returncode == 0, result.stderr
    assert (directory / "install.sh").read_bytes() == content


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_invalid_cached_resource_fails_before_output_truncation(curl_cache, damage):
    run, url, artifact, _, directory, _ = curl_cache
    destination = directory / "existing.bin"
    destination.write_bytes(b"preserve this")
    if damage == "corrupt":
        artifact.write_bytes(b"wrong bytes")
    else:
        artifact.unlink()
    result = run("-sSL", url, "-o", str(destination))
    assert result.returncode == 26
    assert b"pinned resource cache invalid" in result.stderr
    assert destination.read_bytes() == b"preserve this"


def test_unknown_url_and_http_metadata_options_delegate_to_real_curl(curl_cache):
    run, _, _, _, directory, _ = curl_cache
    other = directory / "not-cached.txt"
    other.write_text("genuine curl fallback")
    result = run("-sS", "--write-out", "\n%{url_effective}", other.as_uri())
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode() == "genuine curl fallback\n" + other.as_uri()


def test_unknown_option_for_known_url_delegates_unchanged(curl_cache):
    run, _, artifact, content, _, mapping = curl_cache
    url = artifact.as_uri()
    mapping.write_text(json.dumps({url: {"path": "/does/not/exist", "sha256": "not-used"}}))
    result = run("-sS", "--write-out", "\n%{url_effective}", url)
    assert result.returncode == 0, result.stderr
    assert result.stdout == content + b"\n" + url.encode()


def test_output_option_value_is_not_treated_as_a_resource_url(curl_cache):
    run, url, _, content, directory, _ = curl_cache
    result = run("-fsSL", "--create-dirs", "--output", url, url)
    assert result.returncode == 0, result.stderr
    assert (directory / url).read_bytes() == content


def test_genuine_curl_output_error_is_not_hidden(curl_cache):
    run, url, _, _, directory, _ = curl_cache
    result = run("-fsSL", url, "-o", str(directory))
    assert result.returncode == 23


def test_explicit_protocol_restrictions_are_preserved(curl_cache):
    run, url, _, _, _, _ = curl_cache
    result = run("--proto", "=ftp", "--silent", "--show-error", url)
    assert result.returncode == 1
    assert b"https" in result.stderr


def test_all_immutable_task_resources_are_explicit_and_generated_code_parses():
    manifest = json.loads(resources.SPEC_PATH.with_name("terminalbench-v2.1-manifest.json").read_text())
    spec = json.loads(resources.SPEC_PATH.read_text())
    assert len(spec["tasks"]) == 89
    assert {task: entry["task_ref"] for task, entry in spec["tasks"].items()} == manifest["task_refs"]
    for task, ref in manifest["task_refs"].items():
        commands = resources.resource_recipe_commands(task, ref)
        rendered = "\n".join(commands)
        result = subprocess.run(["sh", "-n"], input=rendered, text=True, capture_output=True)
        assert result.returncode == 0, (task, result.stderr)
        for program in re.findall(r"<<'FOREST_PYTHON_EOF'\n(.*?)\nFOREST_PYTHON_EOF", rendered, re.DOTALL):
            ast.parse(program, feature_version=(3, 9))
    ast.parse(resources.CURL_WRAPPER, feature_version=(3, 9))


def test_unknown_task_ref_and_installer_release_fail_early():
    with pytest.raises(ValueError, match="No audited"):
        resources.resource_recipe_commands("terminal-bench/not-a-task", "sha256:123")
    with pytest.raises(ValueError, match="No audited"):
        resources.resource_recipe_commands("terminal-bench/fix-ocaml-gc", "sha256:123")
    with pytest.raises(ValueError, match="No audited uv"):
        resources.uv_installer_recipe_commands("latest")


@pytest.mark.parametrize("version", ["0.9.5", "0.8.15"])
def test_uv_installer_recipe_pins_script_archive_and_checksum(version):
    commands = resources.uv_installer_recipe_commands(version)
    rendered = "\n".join(commands)
    assert f"https://astral.sh/uv/{version}/install.sh" in rendered
    for name, digest in resources.UV_ASSETS[version].items():
        assert f"https://github.com/astral-sh/uv/releases/download/{version}/{name}" in rendered
        assert len(digest) == 64 and digest in rendered
    assert subprocess.run(["sh", "-n"], input=rendered, text=True).returncode == 0


def test_fetch_records_real_bytes_and_checks_expected_hash(tmp_path, monkeypatch):
    stage = tmp_path / "cache"
    stage.mkdir()
    (stage / "resource-urls.json").write_text("{}")
    (stage / "original-curl").symlink_to(shutil.which("curl"))
    monkeypatch.setattr(resources, "ROOT", str(stage))
    source = tmp_path / "asset.txt"
    source.write_text("complete original source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _run_shell(resources._fetch(source.as_uri(), "public/asset.txt", revision="v1", sha256=digest))
    records = json.loads((stage / "resource-urls.json").read_text())
    assert records[source.as_uri()]["sha256"] == digest
    assert Path(records[source.as_uri()]["path"]).read_bytes() == source.read_bytes()
    failed = _run_shell(
        resources._fetch(source.as_uri(), "public/bad.txt", revision="v1", sha256="0" * 64), check=False
    )
    assert failed.returncode != 0
    assert "Pinned public resource hash mismatch" in failed.stderr


def _git_fixture(path):
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    return {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def _commit(path, env):
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, env=env)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Resource test",
            "-c",
            "user.email=resource@example.invalid",
            "commit",
            "-m",
            "Public fixture",
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], env=env, text=True).strip()


def test_mteb_sparse_cache_includes_every_model_and_all_tasks_at_pinned_commit(tmp_path, monkeypatch):
    source = tmp_path / "public-results"
    env = _git_fixture(source)
    for model in (
        "organization-a/model-a/revision1",
        "organization-b/model-b/revision2",
        "organization-c/model-c/revision3",
    ):
        directory = source / model
        directory.mkdir(parents=True)
        (directory / "model_meta.json").write_text('{"metadata": "public"}')
        for name in resources.MTEB_TASKS:
            (directory / (name + ".json")).write_text('{"fixture": "no ranking performed"}')
        (directory / "UnrelatedBenchmark.json").write_text("{}")
    revision = _commit(source, env)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(resources, "ROOT", str(stage))
    monkeypatch.setattr(resources, "MTEB_REVISION", revision)
    commands = resources._mteb_results()[1:]
    commands = [
        command.replace("https://github.com/embeddings-benchmark/results", source.as_uri()) for command in commands
    ]
    result = _run_shell(commands, env=env)
    assert result.returncode == 0
    cache = stage / "mteb/results"
    manifest = json.loads((stage / "mteb-snapshot.json").read_text())
    assert len(manifest["files"]) == 3 * 29
    assert manifest["revision"] == revision
    assert not list(cache.rglob("UnrelatedBenchmark.json"))
    assert {Path(name).name for name in manifest["files"]} == {
        "model_meta.json",
        *(name + ".json" for name in resources.MTEB_TASKS),
    }
    subprocess.run(["git", "-C", str(cache), "pull", "--ff-only"], env=env, check=True, capture_output=True)
    assert (
        subprocess.check_output(["git", "-C", str(cache), "rev-parse", "HEAD"], text=True, env=env).strip() == revision
    )


def test_resource_recipes_preserve_task_solution_boundaries():
    spec = json.loads(resources.SPEC_PATH.read_text())
    scripts = {
        name: "\n".join(
            command
            for command in resources.resource_recipe_commands(name, entry["task_ref"])
            if "cat > /opt/harbor-offline/resource-policy.json" not in command
        )
        for name, entry in spec["tasks"].items()
    }
    assert "PIP_NO_INDEX" not in scripts["terminal-bench/pypi-server"]
    assert "r-base" not in scripts["terminal-bench/rstan-to-pystan"]
    assert "cmdstan" not in scripts["terminal-bench/rstan-to-pystan"]
    assert "/app/model_cache" not in scripts["terminal-bench/hf-model-inference"]
    assert "--download-only" in scripts["terminal-bench/nginx-request-logging"]
    assert "pip install pyknotid" not in scripts["terminal-bench/build-cython-ext"]
    assert "model.safetensors" in scripts["terminal-bench/count-dataset-tokens"]
    assert "ignore_patterns" not in scripts["terminal-bench/mteb-leaderboard"]


@pytest.mark.parametrize("command", ["true", "echo ordinary-agent-command", 'exec /staging/bootstrap.sh "$@"'])
def test_windows_startup_does_not_run_for_build_or_ordinary_shell(tmp_path, command):
    script = tmp_path / "windows-startup.sh"
    script.write_text(resources.WINDOWS_STARTUP)
    log = tmp_path / "unexpected-supervisor.txt"
    supervisor = tmp_path / "supervisord"
    supervisor.write_text("#!/bin/sh\necho unexpected > " + str(log) + "\n")
    supervisor.chmod(0o755)
    env = {**os.environ, "BASH_ENV": str(script), "PATH": str(tmp_path) + ":" + os.environ["PATH"]}
    subprocess.run(["bash", "-c", command], capture_output=True, env=env)
    assert not log.exists()


def test_native_harbor_path_uses_cache_and_setup_preserves_original_curl(tmp_path, monkeypatch):
    stage = tmp_path / "offline"
    system_bin = tmp_path / "usr/bin"
    system_bin.mkdir(parents=True)
    real_curl = Path(shutil.which("curl"))
    system_curl = system_bin / "curl"
    # macOS refuses to execute relocated platform-signed binaries. This
    # entrypoint still delegates every byte and error to the genuine binary.
    system_curl.write_text("#!/bin/sh\nexec " + shlex.quote(str(real_curl)) + ' "$@"\n')
    system_curl.chmod(0o755)
    original_digest = hashlib.sha256(system_curl.read_bytes()).hexdigest()
    monkeypatch.setattr(resources, "ROOT", str(stage))
    monkeypatch.setattr(
        resources,
        "CURL_WRAPPER",
        resources.CURL_WRAPPER.replace("#!/usr/bin/python3", "#!" + sys.executable).replace(
            "/opt/harbor-offline", str(stage)
        ),
    )
    setup = [command.replace("/usr/bin/curl", str(system_curl)) for command in resources._common_commands()]
    _run_shell(setup)
    _run_shell(setup)
    assert hashlib.sha256((stage / "original-curl").read_bytes()).hexdigest() == original_digest
    cached = stage / "resources/public.txt"
    cached.write_text("complete cached public bytes")
    url = "https://public.example/pinned/public.txt"
    (stage / "resource-urls.json").write_text(
        json.dumps({url: {"path": str(cached), "sha256": hashlib.sha256(cached.read_bytes()).hexdigest()}})
    )
    # Match Harbor's /usr/bin:/usr/local/bin:<inherited PATH> ordering.
    env = {
        **os.environ,
        "PATH": str(system_bin) + ":/usr/local/bin:" + str(stage / "resource-bin") + ":" + os.environ["PATH"],
    }
    result = subprocess.run(["curl", "-fsSL", url], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == cached.read_bytes()
    uncached = tmp_path / "uncached.txt"
    uncached.write_text("genuine curl delegation")
    delegated = subprocess.run(["curl", "-fsSL", uncached.as_uri()], env=env, capture_output=True)
    assert delegated.returncode == 0, delegated.stderr
    assert delegated.stdout == uncached.read_bytes()


def test_windows_ownership_fix_preserves_task_configuration_and_other_tasks():
    spec = json.loads(resources.SPEC_PATH.read_text())
    command = "chown 0:0 /var/log/nginx /var/log/nginx/access.log /var/log/nginx/error.log"
    windows = resources.resource_recipe_commands(
        "terminal-bench/install-windows-3.11", spec["tasks"]["terminal-bench/install-windows-3.11"]["task_ref"]
    )
    assert command in windows
    nginx = resources.resource_recipe_commands(
        "terminal-bench/nginx-request-logging", spec["tasks"]["terminal-bench/nginx-request-logging"]["task_ref"]
    )
    assert command not in nginx
    assert not any(re.search(r"(?m)^\s*sed\s", entry) or "chmod 777" in entry for entry in windows)
