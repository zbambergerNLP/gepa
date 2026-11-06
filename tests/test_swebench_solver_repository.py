"""Verify Git-object isolation with real packed repositories and no task/model calls."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.swebench_live import apptainer, solver_repository
from examples.swebench_live.solver_repository import SOLVER_REPOSITORY_PROTOCOL, isolate_solver_repository


def git(repository, *args, check=True, stdin=None):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(repository.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.test",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.test",
    }
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        env=env,
        input=stdin,
        capture_output=True,
        check=check,
        timeout=30,
    )


@pytest.fixture
def packed_repository(tmp_path):
    sandbox = tmp_path / "sandbox"
    repository = sandbox / "testbed"
    repository.mkdir(parents=True)
    git(repository, "init", "--template=")
    (repository / "source.txt").write_bytes(b"ancestor\n")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "ancestor")
    ancestor = git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    (repository / "source.txt").write_bytes(b"frozen base\r\n")
    (repository / "binary").write_bytes(bytes(range(256)))
    (repository / "directory").mkdir()
    (repository / "directory/unicode-\N{HEBREW LETTER ALEF}").write_text("base content\n")
    executable = repository / "run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (repository / "link").symlink_to("source.txt")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    (repository / "source.txt").write_text("future gold repair\n")
    (repository / "gold-test.txt").write_text("future hidden test\n")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "future solution")
    future = git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    future_blob = git(repository, "rev-parse", "HEAD:gold-test.txt").stdout.decode().strip()
    git(repository, "tag", "-a", "future", "-m", "future release")
    tag = git(repository, "rev-parse", "refs/tags/future").stdout.decode().strip()
    git(repository, "checkout", "--detach", base)
    git(repository, "gc", "--prune=now")
    assert list((repository / ".git/objects/pack").glob("*.pack"))
    return sandbox, repository, base, (ancestor, future, future_blob, tag)


@pytest.mark.parametrize("branch_head", [False, True])
@pytest.mark.parametrize("storage", ["packed", "loose"])
def test_removes_future_objects_and_preserves_base_and_submission(packed_repository, branch_head, storage):
    sandbox, repository, base, hidden = packed_repository
    if storage == "loose":
        directory = repository / ".git/objects/pack"
        content = next(directory.glob("*.pack")).read_bytes()
        for path in directory.iterdir():
            path.unlink()
        git(repository, "unpack-objects", stdin=content)
    if branch_head:
        git(repository, "checkout", "-b", "pinned", base)
        git(repository, "pack-refs", "--all", "--prune")
    before = git(repository, "ls-tree", "-r", "-t", "-z", base).stdout
    for oid in hidden:
        assert git(repository, "cat-file", "-e", oid).returncode == 0
    modes = {
        str(path.relative_to(repository)): path.lstat().st_mode
        for path in repository.rglob("*")
        if ".git" not in path.parts
    }
    git(repository, "replace", base, hidden[1])
    (repository / ".git/info/grafts").write_text(base + "\n")
    (repository / ".git/objects/info/alternates").write_text("/unavailable/provider-objects\n")
    (repository / ".git/config").write_text(
        "[core]\n worktree = /unavailable\n[include]\n path = /unavailable/config\n"
    )

    receipt = isolate_solver_repository(sandbox, base)

    assert receipt["protocol"] == SOLVER_REPOSITORY_PROTOCOL
    assert receipt["base_commit"] == base and receipt["tracked_bytes_and_modes_verified"]
    assert git(repository, "rev-parse", "HEAD").stdout.decode().strip() == base
    assert git(repository, "ls-tree", "-r", "-t", "-z", base).stdout == before
    assert git(repository, "log", "--format=%H").stdout.decode().splitlines() == [base]
    assert git(repository, "status", "--porcelain").stdout == b""
    assert git(repository, "show-ref", check=False).stdout == b""
    for oid in hidden:
        assert git(repository, "cat-file", "-e", oid, check=False).returncode != 0
    assert not (repository / ".git/logs").exists()
    assert not (repository / ".git/objects/info/alternates").exists()
    assert not (repository / ".git/info/grafts").exists()
    assert not list(sandbox.parent.glob("solver-git-*"))
    for name, mode in modes.items():
        assert (repository / name).lstat().st_mode == mode
    (repository / "source.txt").write_bytes(b"candidate repair\n")
    (repository / "new.py").write_text("print('candidate')\n")
    git(repository, "add", "-A")
    patch = git(repository, "diff", "--cached").stdout
    assert b"+candidate repair" in patch and b"+print('candidate')" in patch
    git(repository, "apply", "--check", "--reverse", stdin=patch)


@pytest.mark.parametrize("value", ["HEAD", "HEAD~1", "a" * 39, "A" * 40, "a" * 40 + "\n", "--help", "", None])
def test_rejects_non_exact_base_identity(packed_repository, value):
    sandbox, repository, _, _ = packed_repository
    before = (repository / ".git/HEAD").read_bytes()
    with pytest.raises(ValueError, match="exact base_commit"):
        isolate_solver_repository(sandbox, value)
    assert (repository / ".git/HEAD").read_bytes() == before


def test_rejects_detached_head_mismatch_before_replacement(packed_repository):
    sandbox, repository, _, hidden = packed_repository
    with pytest.raises(ValueError, match="HEAD differs"):
        isolate_solver_repository(sandbox, hidden[1])
    assert git(repository, "cat-file", "-e", hidden[1]).returncode == 0


def test_rejects_tag_object_as_base_commit(packed_repository):
    sandbox, repository, _, hidden = packed_repository
    (repository / ".git/HEAD").write_text(hidden[3] + "\n")
    with pytest.raises(ValueError, match="not a commit object"):
        isolate_solver_repository(sandbox, hidden[3])


@pytest.mark.parametrize("change", ["content", "executable", "symlink", "submodule"])
def test_rejects_changed_tracked_bytes_modes_and_unsupported_gitlinks(packed_repository, change):
    sandbox, repository, base, _ = packed_repository
    if change == "content":
        (repository / "source.txt").write_text("unexpected image changes")
    elif change == "executable":
        (repository / "run.sh").chmod(0o644)
    elif change == "symlink":
        (repository / "link").unlink()
        (repository / "link").symlink_to("binary")
    else:
        (repository / "submodule").mkdir()
        git(repository, "update-index", "--add", "--cacheinfo", "160000," + base + ",submodule")
        git(repository, "commit", "-m", "gitlink fixture")
        base = git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    with pytest.raises(ValueError, match=r"differ|unsupported"):
        isolate_solver_repository(sandbox, base)
    assert not list(sandbox.parent.glob("solver-git-*"))


@pytest.mark.parametrize("location", ["sandbox", "testbed", ".git", "objects", "pack", "object-alias", "nested-git"])
def test_rejects_path_and_object_alias_bypasses(packed_repository, location):
    sandbox, repository, base, _ = packed_repository
    if location == "object-alias":
        packed = next((repository / ".git/objects/pack").glob("*.pack"))
        os.link(packed, repository / "hidden-pack")
    elif location == "nested-git":
        (repository / "nested/.git").mkdir(parents=True)
    else:
        path = {
            "sandbox": sandbox,
            "testbed": repository,
            ".git": repository / ".git",
            "objects": repository / ".git/objects",
            "pack": repository / ".git/objects/pack",
        }[location]
        moved = sandbox.parent / "escaped"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match=r"ordinary|symlink|hardlink|another Git store"):
        isolate_solver_repository(sandbox, base)


def test_missing_base_objects_do_not_follow_alternates(packed_repository, tmp_path):
    sandbox, repository, base, _ = packed_repository
    outside = tmp_path / "external-objects"
    shutil.copytree(repository / ".git/objects", outside)
    shutil.rmtree(repository / ".git/objects/pack")
    (repository / ".git/objects/pack").mkdir()
    (repository / ".git/objects/info/alternates").write_text(str(outside) + "\n")
    assert git(repository, "cat-file", "-e", base).returncode == 0
    with pytest.raises(subprocess.CalledProcessError):
        isolate_solver_repository(sandbox, base)
    assert (repository / ".git/objects/info/alternates").exists()
    assert not list(sandbox.parent.glob("solver-git-*"))


@pytest.mark.parametrize("failure", [None, "head", "permissions", "remove"])
def test_solver_preparation_finishes_before_instance_start_and_preserves_source(
    packed_repository,
    tmp_path,
    monkeypatch,
    failure,
):
    sandbox, repository, base, hidden = packed_repository
    image = tmp_path / "immutable.sif"
    image.write_bytes(b"immutable fixture image")
    monkeypatch.setattr(apptainer, "allocation_limits", lambda _: {})
    calls = []
    original_remove = shutil.rmtree

    def remove(path, *args, **kwargs):
        if failure == "remove" and Path(path).name == ".git":
            raise PermissionError("cannot remove old object store")
        return original_remove(path, *args, **kwargs)

    monkeypatch.setattr(solver_repository.shutil, "rmtree", remove)

    def run(task, argv, *args, **kwargs):
        calls.append(argv)
        if "build" in argv:
            shutil.copytree(sandbox, task.sandbox, symlinks=True)
            if failure == "permissions":
                (task.sandbox / "testbed/source.txt").chmod(0)
        if "start" in argv:
            assert task.repository_isolation["protocol"] == SOLVER_REPOSITORY_PROTOCOL
            assert git(task.sandbox / "testbed", "cat-file", "-e", hidden[1], check=False).returncode != 0
        return {"exit_code": 0, "stdout": '{"instances": []}' if "list" in argv else "", "stderr": ""}

    monkeypatch.setattr(apptainer.ApptainerTask, "_run", run)
    task = apptainer.ApptainerTask(
        {"backend": "apptainer", "image_id": str(image), "sif_sha256": apptainer.sha256(image)},
        16,
        solver_base_commit=hidden[1] if failure == "head" else base,
    )
    if failure:
        with pytest.raises((ValueError, PermissionError)):
            with task:
                pytest.fail("An unverified repository must not reach the solver")
        assert not any("start" in call for call in calls)
    else:
        with task:
            assert task.repository_isolation["base_commit"] == base
    assert not task.directory.exists()
    assert image.read_bytes() == b"immutable fixture image"
    assert git(repository, "cat-file", "-e", hidden[1]).returncode == 0


def test_isolation_version_changes_resume_identity(monkeypatch, tmp_path):
    from examples.common.benchmark_runner import ensure_contract

    monkeypatch.setattr(apptainer.shutil, "which", lambda _: "/usr/bin/apptainer")
    monkeypatch.setattr(apptainer, "sha256", lambda _: "binary-hash")
    monkeypatch.setattr(apptainer, "allocation_limits", lambda _: {"cpus": 4})
    monkeypatch.setattr(apptainer.subprocess, "check_output", lambda *_, **__: "1.5.0\n")
    current = apptainer.runtime_identity(16)
    assert current["solver_repository_protocol"] == SOLVER_REPOSITORY_PROTOCOL
    old = {key: value for key, value in current.items() if key != "solver_repository_protocol"}
    ensure_contract(tmp_path / "run", "contract.json", {"runtime": old})
    with pytest.raises(ValueError, match="use a new run directory"):
        ensure_contract(tmp_path / "run", "contract.json", {"runtime": current})
