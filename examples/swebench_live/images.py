"""Pin and stage native task images without executing containers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from examples.common.artifacts import atomic_json
from examples.swebench_live.data import load_data


def docker_json(*args: str) -> dict | list:
    """Read a bounded Docker metadata response."""
    return json.loads(subprocess.check_output(["docker", *args], text=True, timeout=120))


def resolve_image(tag: str) -> dict:
    """Resolve the published tag to its exact Linux AMD64 manifest and layer sizes."""
    repository = tag.rsplit(":", 1)[0]
    raw = subprocess.check_output(["docker", "buildx", "imagetools", "inspect", "--raw", tag], timeout=120)
    manifest = json.loads(raw)
    if "manifests" in manifest:
        matches = [
            item
            for item in manifest["manifests"]
            if item.get("platform", {}).get("os") == "linux" and item["platform"].get("architecture") == "amd64"
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one Linux AMD64 image: {tag}")
        expected = matches[0]["digest"]
        raw = subprocess.check_output(
            ["docker", "buildx", "imagetools", "inspect", "--raw", repository + "@" + expected], timeout=120
        )
        if "sha256:" + hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Registry manifest checksum mismatch: {tag}")
        manifest = json.loads(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return {"tag": tag, "immutable": repository + "@" + digest, "layers": manifest["layers"]}


def stage_images(root: Path, split: str = "all", reserve_gib: int = 10) -> dict:
    """Pull missing immutable images sequentially until complete or disk reserve is reached."""
    data, _ = load_data(root)
    records = [row for key, rows in data.items() if split in ("all", key) for row in rows]
    target = root / "images.json"
    report = json.loads(target.read_text()) if target.exists() else {"images": {}}
    report.update(containers_started=0, model_calls=0, requested_split=split, state="staging")
    for row in records:
        image = report["images"].get(row["id"])
        if image is None:
            image = resolve_image(row["image"])
            report["images"][row["id"]] = image
            atomic_json(target, report)
        found = subprocess.run(["docker", "image", "inspect", image["immutable"]], capture_output=True, text=True)
        if found.returncode == 0:
            current = json.loads(found.stdout)[0]
        else:
            free = shutil.disk_usage(root).free
            required = reserve_gib * 2**30 + 4 * sum(layer["size"] for layer in image["layers"])
            if free < required:
                report.update(
                    state="disk_space_blocked", free_bytes=free, required_free_bytes=required, next_id=row["id"]
                )
                atomic_json(target, report)
                return report
            image["state"] = "pulling"
            atomic_json(target, report)
            with (root / "image-pulls.log").open("a") as log:
                subprocess.run(
                    ["docker", "pull", "--platform", "linux/amd64", image["immutable"]],
                    check=True,
                    stdout=log,
                    stderr=log,
                    timeout=7200,
                )
            current = docker_json("image", "inspect", image["immutable"])[0]
        if current["Architecture"] != "amd64" or image["immutable"] not in current.get("RepoDigests", []):
            raise ValueError(f"Downloaded image identity mismatch: {row['id']}")
        image.update(state="verified", image_id=current["Id"], expanded_bytes=current["Size"])
        atomic_json(target, report)
    report["state"] = "requested_images_verified"
    atomic_json(target, report)
    return report


def verify_images(root: Path, records: list[dict], memory_gib: int) -> dict:
    """Require local immutable images and sufficient Docker resources before model work."""
    info = docker_json("info", "--format", "{{json .}}")
    if info["MemTotal"] < memory_gib * 2**30 or info["NCPU"] < 4:
        raise ValueError(f"SWE-bench Live needs 4 Docker CPUs and {memory_gib} GiB; change resources separately")
    report = json.loads((root / "images.json").read_text())
    selected = {}
    for row in records:
        entry = report["images"].get(row["id"], {})
        if entry.get("tag") != row["image"] or entry.get("state") != "verified":
            raise ValueError(f"SWE-bench Live image not prepared: {row['id']}")
        current = docker_json("image", "inspect", entry["image_id"])[0]
        if current["Architecture"] != "amd64" or entry["immutable"] not in current.get("RepoDigests", []):
            raise ValueError(f"SWE-bench Live image drift: {row['id']}")
        selected[row["id"]] = {key: entry[key] for key in ("tag", "immutable", "image_id")}
    return selected
