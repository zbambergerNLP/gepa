"""Stage pinned SWE-agent source and an offline SWE-ReX payload; never run a model."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from examples.common.artifacts import atomic_json
from examples.common.react_v2 import file_sha256
from examples.swebench_live.swe_agent import AGENT_REVISION, AGENT_URL, source_identity, verify_payload

HERE = Path(__file__).resolve().parent


def main():
    """Build in a new root using a relocatable uv-managed Linux Python installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--portable-python", type=Path, required=True, help="Root of uv-managed Linux CPython 3.12")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    if not source.exists():
        subprocess.run(["git", "clone", AGENT_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", AGENT_REVISION], check=True)
    identity = source_identity(source)
    payload = root / "payload"
    if payload.exists():
        print(json.dumps(verify_payload(payload)))
        return
    building = root / "payload-building"
    building.mkdir(exist_ok=False)
    shutil.copytree(
        args.portable_python, building / "python", symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    python = building / "python/bin/python3"
    packages = building / "python/lib/python3.12/site-packages"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--target",
            str(packages),
            "--require-hashes",
            "-r",
            str(HERE / "rex-requirements.txt"),
        ],
        check=True,
    )
    shutil.copytree(source / "swerex", packages / "swerex", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(HERE / "rex_worker.py", building / "rex_worker.py")
    wheels = building / "wheels"
    wheels.mkdir()
    wheel_hashes = {}
    for name, version in (("tree-sitter", "0.21.3"), ("tree-sitter-languages", "1.10.2")):
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
            metadata = json.load(response)
        for artifact in metadata["urls"]:
            filename = artifact["filename"]
            if filename.endswith(".whl") and "manylinux" in filename and "x86_64" in filename:
                with urllib.request.urlopen(artifact["url"], timeout=120) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != artifact["digests"]["sha256"]:
                    raise ValueError("Tool dependency wheel checksum mismatch")
                (wheels / filename).write_bytes(data)
                wheel_hashes[filename] = artifact["digests"]["sha256"]
    atomic_json(building / "tool-wheels.json", wheel_hashes)
    manifest = {
        "source": identity,
        "files": {str(p.relative_to(building)): file_sha256(p) for p in building.rglob("*") if p.is_file()},
    }
    atomic_json(building / "manifest.json", manifest)
    building.rename(payload)
    print(json.dumps(verify_payload(payload)))


if __name__ == "__main__":
    main()
