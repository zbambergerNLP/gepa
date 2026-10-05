"""Serve official AppWorld calls from its isolated, pinned Python environment.

The stdout channel contains JSON only. Evaluator details stay in local private
artifacts; neither the solver nor the reflection model receives test contents.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import traceback
from pathlib import Path
from typing import Any


def _hash_files(directory: Path) -> str:
    files = [
        [path.relative_to(directory).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()]
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".bundle"} and "__pycache__" not in path.parts
    ]
    return hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()


def runtime_identity() -> dict[str, Any]:
    """Verify every locked package and fingerprint installed AppWorld engine code."""
    lock_path = Path(__file__).with_name("runtime-requirements.txt")
    requirements = re.findall(r"^([\w.-]+)==([^\s;\\]+)", lock_path.read_text(), flags=re.MULTILINE)
    versions = {}
    for name, expected in requirements:
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"AppWorld dependency drift: {name}.")
        versions[name] = actual
    appworld = importlib.import_module("appworld")
    return {
        "python_version": platform.python_version(),
        "appworld_version": versions["appworld"],
        "requirements_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "package_sha256": _hash_files(Path(appworld.__file__).parent),
    }


def tracker_result(tracker: Any) -> dict[str, Any]:
    """Reject an empty, partial, or internally inconsistent official evaluation."""
    counts = (tracker.num_tests, tracker.pass_count, tracker.fail_count)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Invalid evaluator counts.")
    total, passed, failed = counts
    if total == 0 or total != passed + failed or type(tracker.success) is not bool:
        raise ValueError("Incomplete official evaluation.")
    if tracker.success != (passed == total):
        raise ValueError("Inconsistent official evaluation.")
    return {"success": tracker.success, "num_tests": total, "passed": passed, "failed": failed}


def dispatch(request: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Call the upstream engine or metric without interpreting generated claims."""
    operation = request["operation"]
    if operation == "inspect":
        return runtime_identity()
    if operation == "initialize":
        if "world" in state:
            raise ValueError("World already initialized.")
        appworld = importlib.import_module("appworld")
        appworld.update_root(os.environ["APPWORLD_ROOT"])
        world = appworld.AppWorld(**request["config"])
        state["world"] = world
        return {
            "instruction": world.task.instruction,
            "supervisor": {
                key: getattr(world.task.supervisor, key) for key in ("first_name", "last_name", "email", "phone_number")
            },
            "app_descriptions": world.task.app_descriptions,
        }
    if operation == "execute":
        world = state["world"]
        observation = world.execute(request["code"])
        return {"observation": observation, "task_completed": world.task_completed()}
    if operation == "evaluate":
        world = state["world"]
        tracker = world.evaluate(suppress_errors=True)
        result = tracker_result(tracker)
        path = Path(world.output_directory) / "evaluation" / "gepa-tracker.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task_id": world.task_id, "tracker": tracker.to_dict()}, allow_nan=False))
        path.chmod(0o600)
        return {
            **result,
            "evaluation_path": str(path),
            "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": world.task_id,
        }
    if operation == "aggregate":
        evaluator = importlib.import_module("appworld.evaluator")
        trackers = {}
        for task_id, evidence in request["evaluations"].items():
            path = Path(evidence["path"]).resolve()
            path.relative_to(Path(os.environ["APPWORLD_ROOT"]).resolve() / "experiments" / "outputs")
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != evidence["sha256"]:
                raise ValueError("Saved evaluator evidence changed.")
            payload = json.loads(contents)
            if payload["task_id"] != task_id:
                raise ValueError("Evaluator task identity mismatch.")
            raw = payload["tracker"]
            if raw["num_tests"] <= 0 or raw["num_tests"] != len(raw["passes"]) + len(raw["failures"]):
                raise ValueError("Incomplete saved evaluator output.")
            tracker = evaluator.TestTracker.from_dict(raw)
            if tracker_result(tracker)["success"] != raw["success"]:
                raise ValueError("Saved evaluator score mismatch.")
            trackers[task_id] = tracker
        if not trackers:
            raise ValueError("No official evaluations to aggregate.")
        # include_details=False is broken upstream in 0.1.3.post1; select the
        # aggregate after calling the normal, public compute_metrics method.
        return evaluator.Metric.compute_metrics(trackers)["aggregate"]
    if operation == "split_ids":
        appworld = importlib.import_module("appworld")
        return {split: appworld.load_task_ids(split) for split in request["splits"]}
    if operation == "close":
        world = state.pop("world", None)
        if world is not None:
            world.close()
        return {}
    raise ValueError("Unknown AppWorld runtime operation.")


def main() -> None:
    """Keep diagnostics private and release the world even on protocol failure."""
    protocol_stdout = sys.stdout
    state: dict[str, Any] = {}
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    result = dispatch(request, state)
                response = {"ok": True, "result": result}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "error": type(error).__name__}
            protocol_stdout.write(json.dumps(response, allow_nan=False) + "\n")
            protocol_stdout.flush()
    finally:
        if "world" in state:
            with contextlib.redirect_stdout(sys.stderr):
                state["world"].close()


if __name__ == "__main__":
    main()
