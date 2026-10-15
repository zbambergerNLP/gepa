"""Check every selected official Docker environment without running model or oracle solutions."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from examples.modal.gateway import atomic_json
from examples.terminalbench.benchmark_settings import MANIFEST_PATH, SINGULARITY_TASK_EXCLUSIONS
from examples.terminalbench.prepare_offline import load_recipe_spec, verify_task_package
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest


async def probe(scope: str, output: Path) -> None:
    """Retain exact task/package/image identity and stop at the first failed startup."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    recipes = load_recipe_spec(manifest)
    output.mkdir(parents=True, exist_ok=True)
    task_ids = [
        task_id
        for split in manifest.splits.values()
        for task_id in split
        if scope == "all89" or task_id not in SINGULARITY_TASK_EXCLUSIONS
    ]
    report = {
        "scope": scope,
        "expected": len(task_ids),
        "checks": [],
        "complete": False,
        "meaning": "official environment startup and shell execution; no solutions, grading, or model calls",
    }
    for index, task_id in enumerate(task_ids):
        entry = {"task_id": task_id, "task_ref": manifest.task_refs[task_id], "passed": False}
        report["checks"].append(entry)
        atomic_json(output / "report.json", report)
        subprocess.run(
            [
                "/opt/harbor/bin/harbor",
                "tasks",
                "download",
                f"{task_id}@{manifest.task_refs[task_id]}",
                "--output-dir",
                str(output / "tasks"),
                "--export",
            ],
            check=True,
        )
        task_path = output / "tasks" / task_id.split("/")[-1]
        _, image = verify_task_package(task_path, task_id, manifest.task_refs[task_id], recipes[task_id])
        task = Task(task_path)
        paths = TrialPaths(output / "trials" / str(index))
        paths.mkdir()
        environment = DockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=f"gepa-preflight-{index}",
            session_id=f"gepa-preflight-{index}",
            trial_paths=paths,
            task_env_config=task.config.environment,
        )
        try:
            await asyncio.wait_for(environment.start(force_build=False), timeout=1800)
            result = await environment.exec("true", timeout_sec=60)
            if result.return_code != 0:
                raise RuntimeError(f"Environment shell check failed: {task_id}")
            details = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
            entry.update(passed=True, image=image, image_id=details["Id"], repo_digests=details.get("RepoDigests", []))
        finally:
            await environment.stop(delete=True)
            atomic_json(output / "report.json", report)
    report["complete"] = True
    atomic_json(output / "report.json", report)


def main() -> None:
    """Run under Harbor's pinned Python environment inside the VM sandbox."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("della88", "all89"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(probe(args.scope, args.output))


if __name__ == "__main__":
    main()
