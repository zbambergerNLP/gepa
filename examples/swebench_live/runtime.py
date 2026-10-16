"""Install and verify an isolated official grader without a cloud dependency."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from examples.common.artifacts import atomic_json, digest
from examples.common.react_v2 import file_sha256
from examples.swebench_live.settings import SOURCE_REVISION, SOURCE_URL

HERE = Path(__file__).resolve().parent


def command(argv: list[str], **kwargs) -> str:
    """Run a bounded setup/readback command and retain errors for the caller."""
    return subprocess.check_output(argv, text=True, timeout=600, **kwargs).strip()


def source_identity(source: Path) -> dict:
    """Require the pinned source with exactly the reviewed operational patch."""
    if command(["git", "rev-parse", "HEAD"], cwd=source) != SOURCE_REVISION:
        raise ValueError("SWE-bench Live harness revision drift")
    patch = (HERE / "local-runtime.patch").read_text().strip()
    if command(["git", "diff", "HEAD", "--binary", "--unified=0"], cwd=source) != patch:
        raise ValueError("SWE-bench Live harness differs from the reviewed local-only patch")
    tracked = command(["git", "ls-files"], cwd=source).splitlines()
    hashes = {name: file_sha256(source / name) for name in tracked if (source / name).is_file()}
    return {
        "revision": SOURCE_REVISION,
        "files_sha256": digest(hashes),
        "patch_sha256": file_sha256(HERE / "local-runtime.patch"),
    }


def installed_packages(python: Path) -> dict:
    """Read package versions without importing optional cloud integrations."""
    code = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"
    return json.loads(
        command(["uv", "--no-cache", "run", "--no-project", "--python", str(python), "python", "-c", code])
    )


def prepare_runtime(root: Path) -> dict:
    """Create the pinned, hash-locked evaluator environment without running Docker."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    if not source.exists():
        command(["git", "clone", "--no-checkout", SOURCE_URL, str(source)])
        command(["git", "checkout", "--detach", SOURCE_REVISION], cwd=source)
    if command(["git", "rev-parse", "HEAD"], cwd=source) != SOURCE_REVISION:
        raise ValueError("Existing SWE-bench Live source has a different revision")
    if not command(["git", "diff", "HEAD", "--binary"], cwd=source):
        command(["git", "apply", "--unidiff-zero", str(HERE / "local-runtime.patch")], cwd=source)
    identity = source_identity(source)
    python = root / "venv" / "bin" / "python"
    if not python.exists():
        command(["uv", "--no-cache", "venv", "--python", "3.12", str(python.parent.parent)])
    if (root / "runtime.json").exists():
        return verify_runtime(root)
    command(
        [
            "uv",
            "--no-cache",
            "pip",
            "sync",
            "--python",
            str(python),
            "--require-hashes",
            str(HERE / "runtime-requirements.txt"),
        ]
    )
    command(
        ["uv", "--no-cache", "pip", "install", "--python", str(python), "--no-deps", "--no-sources", "-e", str(source)]
    )
    packages = installed_packages(python)
    if any(name.lower() == "modal" for name in packages):
        raise ValueError("Unexpected cloud SDK in local evaluator runtime")
    check = "from swebench.harness.run_evaluation import run_instance; from swebench.harness.grading import get_eval_report; print('official-grader-import-ok')"
    command(["uv", "--no-cache", "run", "--no-project", "--python", str(python), "python", "-c", check])
    result = {
        "source": identity,
        "requirements_sha256": file_sha256(HERE / "runtime-requirements.txt"),
        "packages": packages,
    }
    atomic_json(root / "runtime.json", result)
    return result


def verify_runtime(root: Path) -> dict:
    """Reject source or dependency drift before an experiment starts."""
    saved = json.loads((root / "runtime.json").read_text())
    actual = {
        "source": source_identity(root / "source"),
        "requirements_sha256": file_sha256(HERE / "runtime-requirements.txt"),
        "packages": installed_packages(root / "venv" / "bin" / "python"),
    }
    if actual != saved:
        raise ValueError("Prepared SWE-bench Live runtime drift")
    return actual
