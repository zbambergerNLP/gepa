"""Build small immutable task and runtime artifacts for offline staging tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest


def make_offline_bundle(tmp_path: Path, *, task_ids: list[str] | None = None):
    """Create canonical named task fixtures using Harbor's published package digest format."""
    manifest = load_terminalbench_manifest(
        Path(__file__).parents[1] / "examples/terminalbench/terminalbench-v2.1-manifest.json"
    )
    selected = task_ids if task_ids is not None else manifest.splits["train"][:2]
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "images").mkdir(parents=True)
    (bundle_dir / "recipes").mkdir()
    payload = {
        "schema_version": 1,
        "harbor_version": "0.22.0",
        "dataset_ref": manifest.dataset["registry_content_hash"],
        "image_cache_dir": "images",
        "tasks": {},
    }
    refs = dict(manifest.task_refs)
    for index, task_id in enumerate(selected):
        short_name = task_id.split("/")[-1]
        task_dir = bundle_dir / "tasks" / short_name
        task_dir.mkdir(parents=True)
        image = f"fixture/{short_name}:pinned"
        contents = {
            "task.toml": (
                f'schema_version = "1.1"\n[task]\nname = "{task_id}"\n[environment]\ndocker_image = "{image}"\n'
            ),
            "instruction.md": f"Solve fixture {index}.\n",
            "environment/Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
            "tests/test.sh": "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
        }
        files = {}
        for name, content in contents.items():
            path = task_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        package_hash = hashlib.sha256(
            "".join(f"{name}\0{value}\n" for name, value in sorted(files.items())).encode()
        ).hexdigest()
        refs[task_id] = "sha256:" + package_hash
        image_path = bundle_dir / "images" / (image.replace("/", "_").replace(":", "_") + ".sif")
        image_path.write_bytes(f"fixture image {index}".encode())
        recipe = bundle_dir / "recipes" / f"{short_name}.def"
        recipe.write_text(f"Bootstrap: docker\nFrom: {image}\n")
        payload["tasks"][task_id] = {
            "ref": refs[task_id],
            "path": str(task_dir.relative_to(bundle_dir)),
            "files": files,
            "docker_image": image,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "recipe": {
                "path": str(recipe.relative_to(bundle_dir)),
                "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
            },
        }
    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(json.dumps(payload, indent=2))
    return replace(manifest, task_refs=refs), bundle_path, payload
