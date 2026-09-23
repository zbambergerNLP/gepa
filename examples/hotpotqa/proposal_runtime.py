"""Run pinned proposal strategies with one explicitly shared request runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import runpy
import sys
from pathlib import Path

SHARED_MODULES = {
    "gepa.lm": "src/gepa/lm.py",
    "examples.common.provider_retries": "examples/common/provider_retries.py",
}


def runtime_identity(root: Path) -> dict:
    """Identify the request modules shared by both proposal revisions."""
    return {
        "directory": str(root),
        "files": {name: hashlib.sha256((root / path).read_bytes()).hexdigest() for name, path in SHARED_MODULES.items()},
    }


class SharedRequestRuntime(importlib.abc.MetaPathFinder):
    """Override only provider dispatch, leaving strategy imports source-pinned."""

    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        relative = SHARED_MODULES.get(fullname)
        return importlib.util.spec_from_file_location(fullname, self.root / relative) if relative else None


def main() -> None:
    """Verify shared runtime bytes before importing either pinned strategy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-request", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    request = json.loads(args.proposal_request.read_text())
    if request["shared_request_runtime"] != runtime_identity(root):
        raise ValueError("Proposal request runtime differs from the reviewed shared files")
    if any(name in sys.modules for name in SHARED_MODULES):
        raise RuntimeError("Request runtime must be selected before strategy imports")
    sys.meta_path.insert(0, SharedRequestRuntime(root))
    runpy.run_path(str(root / "examples/hotpotqa/generalization_pilot.py"), run_name="__main__")


if __name__ == "__main__":
    main()
