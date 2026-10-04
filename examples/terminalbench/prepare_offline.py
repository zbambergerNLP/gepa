"""Prepare pinned task environments for offline Harbor; default to two training tasks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
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
RECIPE_SPEC_PATH = Path(__file__).with_name("terminalbench-v2.1-offline-recipes.json")
SUPPORTED_IMAGES = {
    "terminal-bench/fix-ocaml-gc": "alexgshaw/fix-ocaml-gc:20251031",
    "terminal-bench/log-summary-date-ranges": "alexgshaw/log-summary-date-ranges:20251031",
}
PROBE_CONTRACT = {
    "version": 3,
    "flags": [
        "--cleanenv",
        "--fakeroot",
        "--writable-tmpfs",
        "--containall",
        "--pid",
        "--no-mount",
        "home,tmp,bind-paths",
        "--pwd",
        "/",
    ],
    "environment": [
        "HOME=/opt/harbor-offline/home",
        "UV_OFFLINE=1",
        "PIP_NO_INDEX=1",
        "http_proxy=http://127.0.0.1:9",
        "https_proxy=http://127.0.0.1:9",
        "ALL_PROXY=http://127.0.0.1:9",
        "HTTP_PROXY=http://127.0.0.1:9",
        "HTTPS_PROXY=http://127.0.0.1:9",
        "NO_PROXY=",
        "no_proxy=",
    ],
    "harbor_exec": [
        "/bin/bash",
        "-c",
        'PATH="/usr/bin:/usr/local/bin:${PATH:-/bin}" exec "$@"',
        "harbor-offline-probe",
    ],
}


def load_recipe_spec(manifest) -> dict:
    """Require a curated recipe for every immutable task ref in this dataset."""
    document = json.loads(RECIPE_SPEC_PATH.read_text(encoding="utf-8"))
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document.get("dataset_ref") != manifest.dataset["registry_content_hash"]
    ):
        raise ValueError("Offline recipes do not match the pinned dataset")
    recipes = document.get("tasks", {})
    if set(recipes) != set(manifest.task_refs):
        raise ValueError("Offline recipes must cover exactly the pinned benchmark tasks")
    for task_id, ref in manifest.task_refs.items():
        recipe = recipes[task_id]
        if (
            recipe.get("task_ref") != ref
            or recipe.get("mode") not in ("uvx", "uv_venv", "pip")
            or recipe.get("uv_version") not in (None, "0.8.15", "0.9.5")
            or recipe.get("python") not in ("image_default", "3.11", "3.12", "3.13")
        ):
            raise ValueError(f"No matching verified dependency recipe for {task_id}")
    return recipes


def resource_recipe_commands(task_id: str, task_ref: str) -> list[str]:
    """Load separately audited resource preparation without reading any oracle content."""
    resources = importlib.import_module("examples.terminalbench.offline_resources")
    return resources.resource_recipe_commands(task_id, task_ref)


def uv_installer_recipe_commands(version: str) -> list[str]:
    """Cache the authentic uv installer and its release assets for unchanged test scripts."""
    resources = importlib.import_module("examples.terminalbench.offline_resources")
    return resources.uv_installer_recipe_commands(version)


def verify_task_package(path: Path, task_id: str, ref: str, recipe: dict) -> tuple[dict[str, str], str]:
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
    if recipe.get("task_ref") != ref or image != recipe.get("docker_image"):
        raise ValueError(f"No verified offline recipe for task/image: {task_id} / {image}")
    if files.get("tests/test.sh") != recipe.get("verifier_script_sha256"):
        raise ValueError(f"Verifier bootstrap does not match its audited recipe: {task_id}")
    return files, image


def render_recipe(
    task_id: str, ref: str, base_image: Path, spec: dict, resource_commands: list[str], installer_commands: list[str]
) -> str:
    """Render the proven dependency recipe without changing task files or verifier commands."""
    if spec.get("task_ref") != ref:
        raise ValueError(f"No verified offline recipe for {task_id}")
    if "\n" in str(base_image) or "\r" in str(base_image):
        raise ValueError("Base image paths cannot contain line breaks")
    packages = list(
        dict.fromkeys(
            ["python3", "python3-venv", "tmux", "asciinema", "curl", "ca-certificates"] + spec["system_packages"]
        )
    )
    uv_version = spec.get("uv_version") or "0.9.5"
    mode = spec["mode"]
    if mode == "uvx":
        command = spec["warm_argv"]
        if command[0] != "uvx" or command[-2:] != ["pytest", "--version"]:
            raise ValueError(f"Unsupported verifier warm command for {task_id}")
        inventory = (
            "import importlib.metadata; "
            "print('\\n'.join(sorted(d.metadata['Name'] + '==' + d.version for d in importlib.metadata.distributions())))"
        )
        warm = [
            shlex.join(command),
            shlex.join(command[:-2] + ["--from", "pytest", "python", "-c", inventory])
            + " > /opt/harbor-offline/verifier-requirements.txt",
        ]
    elif mode == "uv_venv":
        warm = [
            shlex.join(["uv", "venv", "-p", spec["python"], "/opt/harbor-offline/verifier-venv"]),
            shlex.join(
                ["uv", "pip", "install", "--python", "/opt/harbor-offline/verifier-venv/bin/python"] + spec["packages"]
            ),
            "uv pip freeze --python /opt/harbor-offline/verifier-venv/bin/python > /opt/harbor-offline/verifier-requirements.txt",
        ]
    elif mode == "pip":
        warm = [
            "mkdir -p /opt/harbor-offline/wheels",
            shlex.join(["python", "-m", "pip", "download", "--dest", "/opt/harbor-offline/wheels"] + spec["packages"]),
            "sha256sum /opt/harbor-offline/wheels/* > /opt/harbor-offline/verifier-wheel-inventory.txt",
        ]
    else:
        raise ValueError(f"Unsupported verifier mode for {task_id}: {mode}")
    extra = (
        "\n".join(resource_commands)
        + "\n    if [ -f /opt/harbor-offline/resource-env.sh ]; then . /opt/harbor-offline/resource-env.sh; fi\n"
        + "\n".join("    " + command for command in warm)
    )
    installer = "\n".join(installer_commands)
    pip_environment = (
        "    export PIP_NO_INDEX=1\n    export PIP_FIND_LINKS=/opt/harbor-offline/wheels\n" if mode == "pip" else ""
    )
    return f"""Bootstrap: localimage
From: {base_image}

%post
    set -eu
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    {shlex.join(["apt-get", "install", "-y", "--no-install-recommends"] + packages)}
    mkdir -p /opt/harbor-offline/home /opt/harbor-offline/uv-cache
{installer}
    curl --fail --location --silent --show-error https://astral.sh/uv/{uv_version}/install.sh -o /opt/harbor-offline/install-uv.sh
    UV_INSTALL_DIR=/opt/harbor-offline/home/.local/bin sh /opt/harbor-offline/install-uv.sh
    export PATH=/opt/harbor-offline/home/.local/bin:$PATH
    export UV_CACHE_DIR=/opt/harbor-offline/uv-cache
    export UV_PYTHON_INSTALL_DIR=/opt/harbor-offline/python
    uv venv --managed-python -p 3.13 /opt/harbor-server
    uv pip install --python /opt/harbor-server/bin/python3 'fastapi==0.142.2' 'uvicorn==0.54.0'
    uv pip freeze --python /opt/harbor-server/bin/python3 > /opt/harbor-server/installed-requirements.txt
{extra}
    dpkg-query -W > /opt/harbor-offline/installed-debian-packages.txt

%environment
    export HOME=/opt/harbor-offline/home
    export PATH=/opt/harbor-offline/home/.local/bin:$PATH
    export UV_CACHE_DIR=/opt/harbor-offline/uv-cache
    export UV_PYTHON_INSTALL_DIR=/opt/harbor-offline/python
    export UV_OFFLINE=1
{pip_environment}    if [ -f /opt/harbor-offline/resource-env.sh ]; then . /opt/harbor-offline/resource-env.sh; fi

%labels
    org.forest.harbor.version 0.22.0
    org.forest.task.ref {ref}
"""


def dependency_probe_argv(spec: dict) -> list[str]:
    """Probe actual imports from the cached verifier environment without loading task tests."""
    program = (
        "import importlib, importlib.metadata, json, sys; "
        f"[importlib.import_module(name) for name in {spec['probe_modules']!r}]; "
        "print(json.dumps({'python': sys.version, 'packages': "
        "{d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}, sort_keys=True))"
    )
    if spec["mode"] == "uvx":
        return spec["warm_argv"][:-2] + ["--from", "pytest", "python", "-c", program]
    setup = [
        "set -eu",
        "probe_dir=$(mktemp -d /tmp/harbor-dependency-probe.XXXXXX)",
        "trap 'rm -rf \"$probe_dir\"' EXIT",
    ]
    if spec["mode"] == "uv_venv":
        setup.extend(
            [
                f'uv venv -p {shlex.quote(spec["python"])} "$probe_dir/venv"',
                'uv pip install --python "$probe_dir/venv/bin/python" ' + shlex.join(spec["packages"]),
            ]
        )
    elif spec["mode"] == "pip":
        setup.extend(
            [
                'python -m venv "$probe_dir/venv"',
                '"$probe_dir/venv/bin/python" -m pip install --no-index --find-links /opt/harbor-offline/wheels '
                + shlex.join(spec["packages"]),
            ]
        )
    else:
        raise ValueError("Unsupported offline dependency probe")
    setup.append('"$probe_dir/venv/bin/python" -c ' + shlex.quote(program))
    return ["sh", "-c", "\n".join(setup)]


def _seal(path: Path, payload: dict, manifest) -> Path:
    """Publish a manifest only after every package and runtime artifact validates."""
    pending = path.with_name(path.stem + ".pending.json")
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_offline_task_bundle(pending, manifest)
    pending.rename(path)
    return path


def _prepare_task(
    task_id, manifest, spec, commands, installer_commands, part_dir, image_cache, harbor, apptainer, env, resume
):
    """Build one independently sealed task part, preserving successful earlier parts."""
    if part_dir.is_symlink():
        raise ValueError(f"Task preparation directories cannot be symlinks: {part_dir}")
    sealed = part_dir / "bundle.json"
    if sealed.exists():
        task = load_offline_task_bundle(sealed, manifest).payload["tasks"][task_id]
        expected = render_recipe(
            task_id,
            manifest.task_refs[task_id],
            image_cache / singularity_image_filename(task["docker_image"]),
            spec,
            commands,
            installer_commands,
        )
        if (
            hashlib.sha256(expected.encode()).hexdigest() != task["recipe"]["sha256"]
            or task["docker_image"] != spec["docker_image"]
            or task["files"].get("tests/test.sh") != spec["verifier_script_sha256"]
            or task.get("dependency_probe", {}).get("argv") != dependency_probe_argv(spec)
            or task.get("dependency_probe", {}).get("contract") != PROBE_CONTRACT
        ):
            raise ValueError(f"Sealed task has a different current preparation recipe or probe: {task_id}")
        return task
    part_dir.mkdir(parents=True, exist_ok=resume)
    for directory in ("tasks", "images", "recipes"):
        (part_dir / directory).mkdir(exist_ok=True)
    ref = manifest.task_refs[task_id]
    short_name = task_id.split("/")[-1]
    task_dir = part_dir / "tasks" / short_name
    if not task_dir.exists():
        subprocess.run(
            [harbor, "tasks", "download", f"{task_id}@{ref}", "--output-dir", str(part_dir / "tasks"), "--export"],
            check=True,
            env=env,
        )
    files, image = verify_task_package(task_dir, task_id, ref, spec)
    image_name = singularity_image_filename(image)
    base_image = image_cache / image_name
    if not base_image.exists():
        partial = base_image.with_suffix(".partial.sif")
        if partial.exists():
            if not resume:
                raise FileExistsError(f"An incomplete base image already exists: {partial}")
            partial.unlink()
        subprocess.run([apptainer, "pull", str(partial), f"docker://{image}"], check=True, env=env)
        partial.rename(base_image)
    if base_image.is_symlink() or not base_image.is_file():
        raise ValueError(f"Base image cache entry must be a regular SIF file: {base_image}")
    base_sha = file_sha256(base_image)
    recipe_path = part_dir / "recipes" / f"{short_name}.def"
    recipe_path.write_text(
        render_recipe(task_id, ref, base_image, spec, commands, installer_commands), encoding="utf-8"
    )
    prepared_image = part_dir / "images" / image_name
    partial = prepared_image.with_suffix(".partial.sif")
    if partial.exists():
        partial.unlink()
    subprocess.run(
        [apptainer, "build", "--fakeroot", "--mksquashfs-args", "-processors 2", str(partial), str(recipe_path)],
        check=True,
        env=env,
    )
    prefix = (
        [apptainer, "exec"]
        + PROBE_CONTRACT["flags"]
        + [str(partial), "env"]
        + PROBE_CONTRACT["environment"]
        + PROBE_CONTRACT["harbor_exec"]
    )
    probe_commands = [
        prefix + dependency_probe_argv(spec),
        prefix
        + [
            "/opt/harbor-server/bin/python3",
            "-c",
            "import fastapi, uvicorn, sys; print(sys.version, fastapi.__version__, uvicorn.__version__)",
        ],
    ]
    if spec.get("uv_version"):
        installer_probe = (
            "set -eu\n"
            "probe_dir=$(mktemp -d /tmp/harbor-installer-probe.XXXXXX)\n"
            "trap 'rm -rf \"$probe_dir\"' EXIT\n"
            f'curl -LsSf https://astral.sh/uv/{spec["uv_version"]}/install.sh -o "$probe_dir/install.sh"\n'
            'sh "$probe_dir/install.sh"\n'
            '. "$HOME/.local/bin/env"\n'
            f'test "$(uv --version | cut -d " " -f 2)" = {shlex.quote(spec["uv_version"])}\n'
            "uv --version"
        )
        probe_commands.insert(0, prefix + ["sh", "-c", installer_probe])
    (part_dir / "dependency-probe-commands.json").write_text(
        json.dumps(probe_commands, indent=2) + "\n", encoding="utf-8"
    )
    probe_log = part_dir / "dependency-probe.log"
    probe_log.write_text("", encoding="utf-8")
    for command in probe_commands:
        probe = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
        with probe_log.open("a", encoding="utf-8") as stream:
            stream.write(probe.stdout + probe.stderr)
        probe.check_returncode()
    partial.replace(prepared_image)
    if file_sha256(base_image) != base_sha:
        raise ValueError(f"Base image changed while preparing {task_id}")
    task = {
        "ref": ref,
        "path": task_dir.relative_to(part_dir).as_posix(),
        "files": files,
        "docker_image": image,
        "base_image_sha256": base_sha,
        "image_sha256": file_sha256(prepared_image),
        "recipe": {"path": recipe_path.relative_to(part_dir).as_posix(), "sha256": file_sha256(recipe_path)},
        "dependency_probe": {
            "argv": dependency_probe_argv(spec),
            "contract": PROBE_CONTRACT,
            "offline": True,
            "returncode": 0,
            "log_sha256": file_sha256(probe_log),
        },
    }
    _seal(
        sealed,
        {
            "schema_version": 1,
            "harbor_version": manifest.dataset["harbor_version"],
            "dataset_ref": manifest.dataset["registry_content_hash"],
            "image_cache_dir": "images",
            "tasks": {task_id: task},
        },
        manifest,
    )
    return task


def prepare_bundle(
    bundle_dir: Path,
    image_cache: Path,
    *,
    harbor_executable: str = "harbor",
    task_count: int = 2,
    all_tasks: bool = False,
    task_ids: list[str] | None = None,
    resume: bool = False,
) -> Path:
    """Prepare selected pinned tasks and seal only after offline dependency probes pass."""
    if (
        type(task_count) is not int
        or task_count not in (1, 2)
        or ((all_tasks or task_ids is not None) and task_count != 2)
        or (all_tasks and task_ids is not None)
    ):
        raise ValueError("task_count must be 1 or 2; use --all-tasks for the complete benchmark")
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    specs = load_recipe_spec(manifest)
    manifest_order = [task_id for split in ("train", "val", "test") for task_id in manifest.splits[split]]
    if task_ids is not None:
        if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids).issubset(manifest.task_refs):
            raise ValueError("task_ids must be unique canonical task names from the pinned manifest")
        selected = [task_id for task_id in manifest_order if task_id in task_ids]
    else:
        selected = (
            [task_id for split in ("train", "val", "test") for task_id in manifest.splits[split]]
            if all_tasks
            else manifest.splits["train"][:task_count]
        )
    if not all_tasks and task_ids is None and selected != list(SUPPORTED_IMAGES)[:task_count]:
        raise ValueError("The pinned training prefix differs from the supported offline recipes")
    bundle_dir = Path(bundle_dir).expanduser().absolute()
    if bundle_dir.is_symlink() or (bundle_dir.exists() and not resume):
        raise FileExistsError(f"Use a fresh bundle directory; refusing to replace {bundle_dir}")
    bundle_dir = bundle_dir.resolve()
    image_cache = Path(image_cache).expanduser().resolve()
    task_ids = selected
    commands = {task_id: resource_recipe_commands(task_id, manifest.task_refs[task_id]) for task_id in task_ids}
    installer_commands = {
        version: uv_installer_recipe_commands(version)
        for version in sorted({specs[task_id].get("uv_version") or "0.9.5" for task_id in task_ids})
    }
    plan = {
        "dataset_ref": manifest.dataset["registry_content_hash"],
        "task_ids": task_ids,
        "recipe_spec_sha256": file_sha256(RECIPE_SPEC_PATH),
        "preparer_sha256": file_sha256(Path(__file__)),
        "resource_commands_sha256": hashlib.sha256(
            json.dumps([commands, installer_commands], sort_keys=True).encode()
        ).hexdigest(),
        "image_cache": str(image_cache),
    }
    plan_path = bundle_dir / "preparation-plan.json"
    if bundle_dir.exists():
        old_plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
        if any(old_plan.get(key) != plan[key] for key in ("dataset_ref", "task_ids", "image_cache")):
            raise ValueError("Cannot resume a bundle with missing or changed preparation inputs")
        if (bundle_dir / "bundle.json").exists():
            if old_plan != plan:
                raise ValueError("Cannot replace a sealed bundle with changed preparation inputs")
            load_offline_task_bundle(bundle_dir / "bundle.json", manifest)
            return bundle_dir / "bundle.json"
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
    bundle_dir.mkdir(parents=True, exist_ok=resume)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / "images").mkdir(exist_ok=True)
    image_cache.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "harbor_version": version,
        "dataset_ref": manifest.dataset["registry_content_hash"],
        "image_cache_dir": "images",
        "tasks": {},
    }
    for task_id in task_ids:
        print(f"Preparing {task_id}", flush=True)
        short_name = task_id.split("/")[-1]
        part_dir = bundle_dir / "parts" / short_name
        task = _prepare_task(
            task_id,
            manifest,
            specs[task_id],
            commands[task_id],
            installer_commands[specs[task_id].get("uv_version") or "0.9.5"],
            part_dir,
            image_cache,
            harbor,
            apptainer,
            env,
            resume,
        )
        image_name = singularity_image_filename(task["docker_image"])
        destination = bundle_dir / "images" / image_name
        if not destination.exists():
            os.link(part_dir / "images" / image_name, destination)
        if file_sha256(destination) != task["image_sha256"]:
            raise ValueError(f"Prepared image changed: {destination}")
        payload["tasks"][task_id] = {
            **task,
            "path": (part_dir / task["path"]).relative_to(bundle_dir).as_posix(),
            "recipe": {
                **task["recipe"],
                "path": (part_dir / task["recipe"]["path"]).relative_to(bundle_dir).as_posix(),
            },
        }
    return _seal(bundle_dir / "bundle.json", payload, manifest)


def main() -> None:
    """Prepare a portable bundle on an Internet-connected Linux host without sudo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--image-cache", required=True, type=Path)
    parser.add_argument("--harbor-executable", default="harbor")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-count", type=int, choices=(1, 2), default=2)
    selection.add_argument("--all-tasks", action="store_true", help="Prepare all 89 pinned train/validation/test tasks")
    selection.add_argument(
        "--task-id", action="append", dest="task_ids", help="Prepare a canonical pinned task; repeat to select more"
    )
    parser.add_argument("--resume", action="store_true", help="Resume by reusing task parts with matching recipe bytes")
    args = parser.parse_args()
    path = prepare_bundle(**vars(args))
    print(f"Verified offline task bundle: {path}")


if __name__ == "__main__":
    main()
