"""Pin published AMD64 OCI images and stage checksummed Apptainer SIF files."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from examples.common.artifacts import atomic_json
from examples.common.react_v2 import file_sha256
from examples.swebench_live.data import load_data

ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def resolve_image(tag: str) -> dict:
    """Resolve Docker Hub metadata without Docker or execution of image contents."""
    repository, reference = tag.rsplit(":", 1)
    if not repository.startswith("starryzhang/sweb.eval.x86_64.") or reference != "latest":
        raise ValueError("Expected an official SWE-bench Live AMD64 image")
    with urllib.request.urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull", timeout=30
    ) as response:
        token = json.load(response)["token"]

    def fetch(kind, ref):
        request = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{repository}/{kind}/{ref}",
            headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
            if ref.startswith("sha256:") and checksum != ref:
                raise ValueError("OCI descriptor checksum mismatch")
            if response.headers.get("Docker-Content-Digest", checksum) != checksum:
                raise ValueError("OCI registry content checksum mismatch")
        return json.loads(raw), checksum

    manifest, checksum = fetch("manifests", reference)
    if "manifests" in manifest:
        amd64 = [
            row
            for row in manifest["manifests"]
            if row.get("platform", {}).get("os") == "linux" and row["platform"].get("architecture") == "amd64"
        ]
        if len(amd64) != 1:
            raise ValueError("Expected exactly one Linux AMD64 image manifest")
        manifest, checksum = fetch("manifests", amd64[0]["digest"])
    config, _ = fetch("blobs", manifest["config"]["digest"])
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise ValueError("Published task image is not Linux AMD64")
    return {
        "tag": tag,
        "immutable": repository + "@" + checksum,
        "architecture": "amd64",
        "compressed_bytes": sum(layer["size"] for layer in manifest["layers"]),
    }


def stage_images(root: Path, split: str, instance_ids: list[str] | None = None) -> dict:
    """Download selected assets with an exclusive writer and bounded compression."""
    splits, _ = load_data(root)
    rows = [row for name, records in splits.items() if split in ("all", name) for row in records]
    if instance_ids:
        selected = set(instance_ids)
        if len(selected) != len(instance_ids) or not selected <= {row["id"] for row in rows}:
            raise ValueError("Image IDs must be unique members of the requested split")
        rows = [row for row in rows if row["id"] in selected]
    directory = root / "apptainer-images"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for row in rows:
            entry = manifest.get(row["id"])
            if entry and entry.get("state") == "verified":
                verify_image(entry, row["image"])
                continue
            if entry is None:
                entry = {**resolve_image(row["image"]), "state": "resolved"}
                manifest[row["id"]] = entry
                atomic_json(manifest_path, manifest)
            if entry["tag"] != row["image"]:
                raise ValueError("Prepared image tag drift")
            required = 10 * 1024**3 + 4 * entry["compressed_bytes"]
            if shutil.disk_usage(directory).free < required:
                raise RuntimeError("Insufficient space to stage this image while retaining a 10 GiB reserve")
            build_root = os.environ.get("APPTAINER_TMPDIR") or tempfile.gettempdir()
            if shutil.disk_usage(build_root).free < required:
                raise RuntimeError("Insufficient temporary build space while retaining a 10 GiB reserve")
            target = directory / (entry["immutable"].split("sha256:")[1] + ".sif")
            if target.exists():
                raise ValueError("Unverified SIF exists; inspect it before replacing or adopting it")
            # Parallel filesystems can reject valid OCI timestamps during extraction.
            # Keep publication on the cache filesystem so the final rename stays atomic.
            with (
                tempfile.TemporaryDirectory(prefix="build-", dir=directory) as publication,
                tempfile.TemporaryDirectory(prefix="gepa-apptainer-", dir=build_root) as temporary,
            ):
                output = Path(publication) / "image.sif"
                subprocess.run(
                    [
                        "apptainer",
                        "build",
                        "--disable-cache",
                        "--arch",
                        "amd64",
                        "--mksquashfs-args",
                        "-processors 4 -mem 4G",
                        str(output),
                        "docker://" + entry["immutable"],
                    ],
                    env={**os.environ, "APPTAINER_TMPDIR": temporary},
                    check=True,
                    timeout=3600,
                )
                checksum = file_sha256(output)
                output.chmod(0o444)
                output.rename(target)
            entry.update(state="verified", backend="apptainer", image_id=str(target.resolve()), sif_sha256=checksum)
            atomic_json(manifest_path, manifest)
    return {"requested": len(rows), "verified": sum(row.get("state") == "verified" for row in manifest.values())}


def verify_image(entry: dict, tag: str) -> None:
    """Reject missing, mutable, foreign or altered local images before execution."""
    if (
        entry.get("state") != "verified"
        or entry.get("backend") != "apptainer"
        or entry.get("tag") != tag
        or entry.get("architecture") != "amd64"
        or not entry.get("immutable", "").startswith(tag.rsplit(":", 1)[0] + "@sha256:")
    ):
        raise ValueError("Unverified Apptainer image identity")
    if file_sha256(Path(entry["image_id"])) != entry["sif_sha256"]:
        raise ValueError("Prepared Apptainer image checksum drift")


def verify_images(root: Path, records: list[dict]) -> dict:
    """Require every selected task image; preparation coverage never shrinks a run."""
    manifest = json.loads((root / "apptainer-images" / "manifest.json").read_text())
    result = {}
    for row in records:
        if row["id"] not in manifest:
            raise ValueError(f"Stage the Apptainer image before running task {row['id']}")
        entry = manifest[row["id"]]
        verify_image(entry, row["image"])
        result[row["id"]] = dict(entry)
    return result
