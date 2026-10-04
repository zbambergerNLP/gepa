"""Prepare the first two pinned training tasks for native offline Harbor execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.adapters.terminal_bench_adapter.staging import (
    file_sha256,
    load_offline_task_bundle,
    singularity_image_filename,
)

MANIFEST_PATH = Path(__file__).with_name("terminalbench-v2.1-manifest.json")
SUPPORTED_IMAGES = {
    "terminal-bench/fix-ocaml-gc": "alexgshaw/fix-ocaml-gc:20251031",
    "terminal-bench/log-summary-date-ranges": "alexgshaw/log-summary-date-ranges:20251031",
}
OCAML_REF = "356d558bf89552ed9229253ec4135c691234f060"


def verify_task_package(path: Path, task_id: str, ref: str) -> tuple[dict[str, str], str]:
    """Check all downloaded bytes against Harbor's canonical package digest before building."""
    files = {}
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Task package must be a regular directory: {path}")
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError(f"Task packages cannot contain symlinks or special files: {entry}")
        if entry.is_file():
            files[entry.relative_to(path).as_posix()] = file_sha256(entry)
    digest = hashlib.sha256("".join(f"{name}\0{sha}\n" for name, sha in sorted(files.items())).encode()).hexdigest()
    if f"sha256:{digest}" != ref:
        raise ValueError(f"Downloaded task {task_id!r} does not match its pinned package ref")
    try:
        toml = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        toml = importlib.import_module("tomli")
    config = toml.loads((path / "task.toml").read_text(encoding="utf-8"))
    if config.get("task", {}).get("name") != task_id:
        raise ValueError(f"Downloaded task has a different canonical name: {task_id}")
    image = config.get("environment", {}).get("docker_image")
    if task_id not in SUPPORTED_IMAGES or image != SUPPORTED_IMAGES[task_id]:
        raise ValueError(f"No verified offline recipe for task/image: {task_id} / {image}")
    return files, image


def render_recipe(task_id: str, ref: str, base_image: Path) -> str:
    """Render the proven dependency recipe without changing task files or verifier commands."""
    if task_id not in SUPPORTED_IMAGES:
        raise ValueError(f"No verified offline recipe for {task_id}")
    if "\n" in str(base_image) or "\r" in str(base_image):
        raise ValueError("Base image paths cannot contain line breaks")
    ocaml = ""
    if task_id == "terminal-bench/fix-ocaml-gc":
        ocaml = f"""    git clone --bare --depth 1 --single-branch --branch tag_purposefully_broken_sweeping_changes https://github.com/sadiqj/ocaml/ /opt/harbor-offline/ocaml.git
    test "$(git --git-dir=/opt/harbor-offline/ocaml.git rev-parse tag_purposefully_broken_sweeping_changes^{{commit}})" = "{OCAML_REF}"
    git config --system url.file:///opt/harbor-offline/ocaml.git.insteadOf https://github.com/sadiqj/ocaml/
    git config --system protocol.file.allow always
"""
    return f"""Bootstrap: localimage
From: {base_image}

%post
    set -eu
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-venv tmux asciinema curl ca-certificates
    /usr/bin/python3 -m venv /opt/harbor-server
    /opt/harbor-server/bin/python3 -m pip install 'fastapi==0.142.2' 'uvicorn==0.54.0'
    /opt/harbor-server/bin/python3 -m pip freeze > /opt/harbor-server/installed-requirements.txt
    mkdir -p /opt/harbor-offline/home /opt/harbor-offline/uv-cache
    curl --fail --location --silent --show-error https://astral.sh/uv/0.9.5/install.sh -o /opt/harbor-offline/install-uv.sh
    UV_INSTALL_DIR=/opt/harbor-offline/home/.local/bin sh /opt/harbor-offline/install-uv.sh
    export PATH=/opt/harbor-offline/home/.local/bin:$PATH
    export UV_CACHE_DIR=/opt/harbor-offline/uv-cache
    export UV_PYTHON_INSTALL_DIR=/opt/harbor-offline/python
    uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version
{ocaml}    dpkg-query -W > /opt/harbor-offline/installed-debian-packages.txt

%environment
    export HOME=/opt/harbor-offline/home
    export PATH=/opt/harbor-offline/home/.local/bin:$PATH
    export UV_CACHE_DIR=/opt/harbor-offline/uv-cache
    export UV_PYTHON_INSTALL_DIR=/opt/harbor-offline/python
    export UV_OFFLINE=1

%labels
    org.forest.harbor.version 0.22.0
    org.forest.task.ref {ref}
"""


def prepare_bundle(
    bundle_dir: Path,
    image_cache: Path,
    *,
    harbor_executable: str = "harbor",
    task_count: int = 2,
) -> Path:
    """Download, verify, build, then seal a fresh bundle for the supported training prefix."""
    if type(task_count) is not int or task_count not in (1, 2):
        raise ValueError("task_count must be 1 or 2; no verified recipes exist for later training tasks")
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    task_ids = manifest.splits["train"][:task_count]
    if task_ids != list(SUPPORTED_IMAGES)[:task_count]:
        raise ValueError("The pinned training prefix differs from the supported offline recipes")
    bundle_dir = Path(bundle_dir).expanduser().absolute()
    if bundle_dir.exists() or bundle_dir.is_symlink():
        raise FileExistsError(f"Use a fresh bundle directory; refusing to replace {bundle_dir}")
    bundle_dir = bundle_dir.resolve()
    image_cache = Path(image_cache).expanduser().resolve()
    harbor = shutil.which(harbor_executable)
    apptainer = shutil.which("apptainer")
    if harbor is None or apptainer is None:
        raise ValueError("Preparation requires the pinned Harbor executable and apptainer on PATH")
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    version = subprocess.run(
        [harbor, "--version"], check=True, capture_output=True, text=True, timeout=60, env=env
    ).stdout.strip()
    if version != manifest.dataset["harbor_version"]:
        raise ValueError(f"Harbor {manifest.dataset['harbor_version']} is required; found {version!r}")
    bundle_dir.mkdir(parents=True)
    for directory in ("tasks", "images", "recipes"):
        (bundle_dir / directory).mkdir()
    image_cache.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "harbor_version": version,
        "dataset_ref": manifest.dataset["registry_content_hash"],
        "image_cache_dir": "images",
        "tasks": {},
    }
    for task_id in task_ids:
        ref = manifest.task_refs[task_id]
        short_name = task_id.split("/")[-1]
        task_dir = bundle_dir / "tasks" / short_name
        subprocess.run(
            [harbor, "tasks", "download", f"{task_id}@{ref}", "--output-dir", str(bundle_dir / "tasks"), "--export"],
            check=True,
            env=env,
        )
        files, image = verify_task_package(task_dir, task_id, ref)
        image_name = singularity_image_filename(image)
        base_image = image_cache / image_name
        if not base_image.exists():
            partial = base_image.with_suffix(".partial.sif")
            if partial.exists():
                raise FileExistsError(f"An incomplete base image already exists: {partial}")
            subprocess.run([apptainer, "pull", str(partial), f"docker://{image}"], check=True, env=env)
            partial.rename(base_image)
        if base_image.is_symlink() or not base_image.is_file():
            raise ValueError(f"Base image cache entry must be a regular SIF file: {base_image}")
        base_sha = file_sha256(base_image)
        recipe = bundle_dir / "recipes" / f"{short_name}.def"
        recipe.write_text(render_recipe(task_id, ref, base_image), encoding="utf-8")
        prepared_image = bundle_dir / "images" / image_name
        subprocess.run([apptainer, "build", "--fakeroot", str(prepared_image), str(recipe)], check=True, env=env)
        if file_sha256(base_image) != base_sha:
            raise ValueError(f"Base image changed while preparing {task_id}")
        payload["tasks"][task_id] = {
            "ref": ref,
            "path": task_dir.relative_to(bundle_dir).as_posix(),
            "files": files,
            "docker_image": image,
            "base_image_sha256": base_sha,
            "image_sha256": file_sha256(prepared_image),
            "recipe": {"path": recipe.relative_to(bundle_dir).as_posix(), "sha256": file_sha256(recipe)},
        }
    pending = bundle_dir / "bundle.pending.json"
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_offline_task_bundle(pending, manifest)
    sealed = bundle_dir / "bundle.json"
    pending.rename(sealed)
    return sealed


def main() -> None:
    """Prepare a portable bundle on an Internet-connected Linux host without sudo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--image-cache", required=True, type=Path)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--task-count", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    path = prepare_bundle(**vars(args))
    print(f"Verified offline task bundle: {path}")


if __name__ == "__main__":
    main()
