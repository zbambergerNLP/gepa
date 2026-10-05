"""Validate sealed local task packages and prepared images for offline Harbor jobs."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import TerminalBenchManifest


def file_sha256(path: Path) -> str:
    """Hash artifact bytes without loading a whole container image into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, value: Any) -> Path:
    """Require an existing contained path with no symbolic-link components."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Offline bundle path must be a relative POSIX path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise ValueError(f"Offline bundle path must be contained and normalized: {value!r}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Offline bundle paths cannot contain symlinks: {path}")
    if not path.exists() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Offline bundle artifact is missing or outside its root: {path}")
    return path


def _digest(value: Any) -> str:
    """Require the lowercase SHA-256 representation used in bundle file records."""
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Offline bundle requires a SHA-256 digest: {value!r}")
    return value


def singularity_image_filename(docker_image: str) -> str:
    """Match Harbor 0.22.0's cache naming without changing the task's image reference."""
    image = docker_image if ":" in docker_image else f"{docker_image}:latest"
    return image.replace("/", "_").replace(":", "_") + ".sif"


@dataclass(frozen=True)
class OfflineTaskBundle:
    """Bind immutable package refs to local task, recipe, and prepared SIF bytes."""

    path: Path
    document_sha256: str
    payload: dict[str, Any]
    manifest: TerminalBenchManifest

    @property
    def image_cache_dir(self) -> Path:
        """Return the verified cache directory consumed by native Harbor Singularity."""
        path = _relative_path(self.path.parent, self.payload["image_cache_dir"])
        if not path.is_dir():
            raise ValueError("Offline bundle image_cache_dir must be a directory")
        return path

    def require_tasks(self, task_ids: Sequence[str]) -> None:
        """Reject incomplete staging before any requested task is evaluated."""
        missing = sorted(set(task_ids).difference(self.payload["tasks"]))
        if missing:
            raise ValueError(f"Offline task bundle does not contain requested tasks: {missing}")

    def validate(self, task_ids: Sequence[str] | None = None) -> None:
        """Recheck selected package and runtime bytes before starting a Harbor job."""
        if file_sha256(self.path) != self.document_sha256:
            raise ValueError("Offline task bundle manifest changed after loading")
        tasks = self.payload["tasks"]
        selected = list(tasks) if task_ids is None else list(task_ids)
        self.require_tasks(selected)
        image_cache = self.image_cache_dir
        checked_artifacts: dict[Path, str] = {}
        try:
            toml = importlib.import_module("tomllib")
        except ModuleNotFoundError:
            try:
                toml = importlib.import_module("tomli")
            except ModuleNotFoundError as exc:
                raise ValueError("Offline task bundles require Python 3.11+ or the tomli package") from exc
        for task_id in selected:
            task = tasks[task_id]
            if not isinstance(task, dict) or task.get("ref") != self.manifest.task_refs.get(task_id):
                raise ValueError(f"Offline task {task_id!r} does not match its pinned package ref")
            root = _relative_path(self.path.parent, task.get("path"))
            if not root.is_dir():
                raise ValueError(f"Offline task package must be a directory: {root}")
            files = task.get("files")
            if not isinstance(files, dict) or "task.toml" not in files:
                raise ValueError(f"Offline task {task_id!r} has no package file inventory")
            actual_files = set()
            for path in root.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError(f"Offline task packages require regular files without symlinks: {path}")
                if path.is_file():
                    actual_files.add(path.relative_to(root).as_posix())
            if actual_files != set(files):
                raise ValueError(f"Offline task {task_id!r} package file inventory changed")
            package_digest = hashlib.sha256()
            for name, expected in sorted(files.items()):
                path = _relative_path(root, name)
                expected = _digest(expected)
                if file_sha256(path) != expected:
                    raise ValueError(f"Offline task package file changed: {path}")
                package_digest.update(f"{name}\0{expected}\n".encode())
            if f"sha256:{package_digest.hexdigest()}" != task["ref"]:
                raise ValueError(f"Offline task {task_id!r} file hashes do not match its pinned package ref")
            config = toml.loads((root / "task.toml").read_text(encoding="utf-8"))
            image = task.get("docker_image")
            if config.get("task", {}).get("name") != task_id:
                raise ValueError(f"Offline task {task_id!r} has a different canonical task name")
            if not isinstance(image, str) or not image or config.get("environment", {}).get("docker_image") != image:
                raise ValueError(f"Offline task {task_id!r} has a different published Docker image")
            recipe = task.get("recipe")
            if not isinstance(recipe, dict):
                raise ValueError(f"Offline task {task_id!r} requires a build recipe record")
            artifacts = (
                (_relative_path(image_cache, singularity_image_filename(image)), task.get("image_sha256")),
                (_relative_path(self.path.parent, recipe.get("path")), recipe.get("sha256")),
            )
            for artifact, expected in artifacts:
                expected = _digest(expected)
                if not artifact.is_file():
                    raise ValueError(f"Offline bundle artifact must be a regular file: {artifact}")
                if artifact not in checked_artifacts:
                    checked_artifacts[artifact] = file_sha256(artifact)
                if checked_artifacts[artifact] != expected:
                    raise ValueError(f"Offline runtime artifact changed: {artifact}")

    def task_configs(self, task_ids: Sequence[str]) -> list[dict[str, str]]:
        """Return local Harbor tasks in the exact requested order after verification."""
        self.validate(task_ids)
        return [
            {
                "path": str(_relative_path(self.path.parent, self.payload["tasks"][task_id]["path"])),
                "source": self.manifest.dataset["reference"],
            }
            for task_id in task_ids
        ]

    def contract(self) -> dict[str, Any]:
        """Fingerprint execution contents independently of local artifact placement."""
        return {
            "schema_version": self.payload["schema_version"],
            "harbor_version": self.payload["harbor_version"],
            "dataset_ref": self.payload["dataset_ref"],
            "tasks": {
                task_id: {
                    "ref": task["ref"],
                    "docker_image": task["docker_image"],
                    "image_sha256": task["image_sha256"],
                    "recipe_sha256": task["recipe"]["sha256"],
                }
                for task_id, task in sorted(self.payload["tasks"].items())
            },
        }


def load_offline_task_bundle(path: str | Path, manifest: TerminalBenchManifest) -> OfflineTaskBundle:
    """Validate a bundle against the checked-in task refs before exposing its paths."""
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    payload = json.loads(content)
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("Offline task bundle requires schema_version 1")
    if (
        payload.get("harbor_version") != manifest.dataset["harbor_version"]
        or payload.get("dataset_ref") != manifest.dataset["registry_content_hash"]
    ):
        raise ValueError("Offline task bundle has a different Harbor version or pinned dataset")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks or not set(tasks).issubset(manifest.task_refs):
        raise ValueError("Offline task bundle must contain only pinned benchmark tasks")
    bundle = OfflineTaskBundle(path, hashlib.sha256(content).hexdigest(), payload, manifest)
    bundle.validate()
    return bundle
