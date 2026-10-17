"""Verify the exact official runtime used for rendering and scoring."""

import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path

import decision_bench

from examples.decisionbench.benchmark_settings import UPSTREAM_FILES, UPSTREAM_REPO, UPSTREAM_REVISION


def validate_upstream_runtime() -> dict:
    """Reject a moving, replaced, or locally modified DecisionBench installation."""
    metadata = json.loads(distribution("decision-bench").read_text("direct_url.json") or "{}")
    if metadata.get("vcs_info", {}).get("commit_id") != UPSTREAM_REVISION:
        raise ValueError(f"Install decision-bench from {UPSTREAM_REPO}.git@{UPSTREAM_REVISION}")
    root = Path(decision_bench.__file__).parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in UPSTREAM_FILES}
    if hashes != UPSTREAM_FILES:
        raise ValueError("DecisionBench runtime source changed from its pinned revision")
    return {
        "repository": UPSTREAM_REPO,
        "revision": UPSTREAM_REVISION,
        "files": hashes,
        "packages": {name: version(name) for name in ("decision-bench", "pydantic", "pyarrow", "datasets", "litellm")},
    }
