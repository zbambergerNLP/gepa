"""Expose only the pinned base Git objects in each disposable solver sandbox."""

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

SOLVER_REPOSITORY_PROTOCOL = "swe-base-commit-and-tree-only-v1"
OBJECT_ID = re.compile(r"[0-9a-f]{40}")


def validate_base_commit(value: str) -> None:
    """Require an exact SHA-1 identity instead of a movable Git expression."""
    if not isinstance(value, str) or OBJECT_ID.fullmatch(value) is None:
        raise ValueError("Solver isolation requires an exact base_commit SHA-1")


def _git(directory: Path, env: dict, *args: str, stdin: bytes | None = None) -> bytes:
    """Run local Git plumbing without image configuration or inherited Git settings."""
    return subprocess.run(
        ["git", "--git-dir", str(directory), *args],
        cwd=directory.parent,
        env=env,
        input=stdin,
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout


def _verify_git_directory(directory: Path) -> None:
    """Reject links that could retain an accessible alias after removing the old store."""
    if directory.is_symlink() or not directory.is_dir() or (directory / "commondir").exists():
        raise ValueError("Solver Git directory must be an independent ordinary directory")
    for root, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            entry = Path(root) / name
            mode = entry.lstat()
            if stat.S_ISDIR(mode.st_mode):
                continue
            if not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1:
                raise ValueError("Solver Git storage contains a symlink, hardlink, or special file")


def _head_commit(directory: Path) -> str:
    """Resolve an ordinary branch or detached HEAD without loading repository config."""
    head = (directory / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ")
        parts = ref.split("/")
        if not ref.startswith("refs/heads/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Solver repository has an unsupported HEAD reference")
        loose = directory.joinpath(*parts)
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            packed = directory / "packed-refs"
            matches = (
                [line.split(" ", 1)[0] for line in packed.read_text().splitlines() if line.endswith(" " + ref)]
                if packed.is_file()
                else []
            )
            if len(matches) != 1:
                raise ValueError("Solver repository HEAD reference is missing or ambiguous")
            head = matches[0]
    validate_base_commit(head)
    return head


def _copy_objects(source: Path, target: Path) -> None:
    """Copy canonical object files while ignoring alternates and other old metadata."""
    for prefix in source.iterdir():
        if re.fullmatch(r"[0-9a-f]{2}", prefix.name) and prefix.is_dir():
            for entry in prefix.iterdir():
                if re.fullmatch(r"[0-9a-f]{38}", entry.name) and entry.is_file():
                    (target / prefix.name).mkdir(exist_ok=True)
                    shutil.copyfile(entry, target / prefix.name / entry.name)
        elif prefix.name == "pack" and prefix.is_dir():
            for entry in prefix.iterdir():
                if re.fullmatch(r"pack-[0-9a-f]{40}\.(pack|idx)", entry.name) and entry.is_file():
                    shutil.copyfile(entry, target / "pack" / entry.name)


def _verify_worktree(repository: Path, entries: bytes) -> set[str]:
    """Check raw tracked bytes and Git modes without filters, hooks, or checkout writes."""
    objects = set()
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split()
        name = os.fsdecode(raw_name)
        parts = name.split("/")
        if PurePosixPath(name).is_absolute() or any(
            part in {"", ".", ".."} or part.lower() == ".git" for part in parts
        ):
            raise ValueError("Solver base tree contains an unsafe path")
        path = repository.joinpath(*parts)
        if any(
            parent.is_symlink() for parent in path.parents if parent != repository and parent.is_relative_to(repository)
        ):
            raise ValueError("Solver tracked path crosses a symlink")
        info = path.lstat()
        objects.add(object_id)
        if kind == "tree" and mode == "040000":
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Solver base directory differs from the pinned tree")
            continue
        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise ValueError("Solver base tree contains an unsupported entry or submodule")
        if mode == "120000":
            if not stat.S_ISLNK(info.st_mode):
                raise ValueError("Solver base symlink differs from the pinned tree")
            content = os.fsencode(os.readlink(path))
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        else:
            if not stat.S_ISREG(info.st_mode) or bool(info.st_mode & 0o111) != (mode == "100755"):
                raise ValueError("Solver base file mode differs from the pinned tree")
            checksum = hashlib.sha1(b"blob " + str(info.st_size).encode() + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            actual = checksum.hexdigest()
        if actual != object_id:
            raise ValueError("Solver tracked bytes differ from the pinned base tree")
    return objects


def isolate_solver_repository(sandbox: Path, base_commit: str) -> dict:
    """Replace image Git storage before startup, retaining only the base commit and tree."""
    validate_base_commit(base_commit)
    if sandbox.is_symlink() or not sandbox.is_dir():
        raise ValueError("Solver sandbox must be an ordinary directory")
    sandbox = sandbox.resolve()
    repository = sandbox / "testbed"
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("Solver repository must be an ordinary /testbed directory")
    original = repository / ".git"
    _verify_git_directory(original)
    if _head_commit(original) != base_commit:
        raise ValueError("Solver repository HEAD differs from the frozen base_commit")
    for root, directories, files in os.walk(repository, followlinks=False):
        if Path(root) == repository:
            directories.remove(".git")
        elif ".git" in directories + files or {"HEAD", "objects", "refs"}.issubset(set(directories + files)):
            raise ValueError("Solver repository contains another Git store")

    with tempfile.TemporaryDirectory(prefix="solver-git-", dir=sandbox.parent) as temporary:
        control = Path(temporary)
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(control),
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_ALLOW_PROTOCOL": "",
            "GIT_TERMINAL_PROMPT": "0",
        }
        reader, fresh = control / "reader.git", control / "fresh.git"
        for directory in (reader, fresh):
            _git(directory, env, "init", "--bare", "--template=", "--object-format=sha1", str(directory))
        _copy_objects(original / "objects", reader / "objects")
        if _git(reader, env, "cat-file", "-t", base_commit).strip() != b"commit":
            raise ValueError("Solver base_commit is not a commit object")
        tree = _git(reader, env, "rev-parse", base_commit + "^{tree}").decode().strip()
        entries = _git(reader, env, "ls-tree", "-r", "-t", "-z", "--full-tree", base_commit)
        objects = _verify_worktree(repository, entries) | {base_commit, tree}
        _git(
            reader,
            env,
            "pack-objects",
            "--window=0",
            "--no-reuse-object",
            "--no-reuse-delta",
            str(fresh / "objects/pack/pack"),
            stdin="".join(value + "\n" for value in sorted(objects)).encode(),
        )
        (fresh / "HEAD").write_text(base_commit + "\n")
        (fresh / "shallow").write_text(base_commit + "\n")
        _git(fresh, env, "config", "core.bare", "false")
        _git(fresh, env, "config", "core.logAllRefUpdates", "false")
        _git(fresh, env, "read-tree", base_commit)
        actual = set(
            _git(fresh, env, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines()
        )
        if actual != objects:
            raise ValueError("Solver Git object allowlist verification failed")
        _git(fresh, env, "fsck", "--full", "--no-reflogs")
        shutil.rmtree(original)
        fresh.rename(original)
        _verify_worktree(repository, entries)
        receipt = {
            "protocol": SOLVER_REPOSITORY_PROTOCOL,
            "base_commit": base_commit,
            "base_tree": tree,
            "base_entries_sha256": hashlib.sha256(entries).hexdigest(),
            "object_count": len(objects),
            "objects_sha256": hashlib.sha256("\n".join(sorted(objects)).encode()).hexdigest(),
            "history": "shallow base commit; no ancestor objects",
            "tracked_bytes_and_modes_verified": True,
            "original_git_directory_removed": True,
        }
    return receipt
