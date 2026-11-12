"""Install the exact missing Haystack e2e wheels in a disposable native task root."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path


def package_versions(site_packages: Path) -> dict:
    """Read only top-level distributions, excluding vendored setuptools metadata."""
    packages = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in packages:
            raise ValueError(f"Duplicate installed distribution: {name}")
        packages[name] = distribution.version
    return packages


def verified_wheels(environment: dict, wheelhouse: Path) -> list:
    """Reject changed wheels before any installation starts."""
    paths = []
    for wheel in environment["wheels"]:
        path = wheelhouse / wheel["filename"]
        if path.parent != wheelhouse or not path.is_file() or path.is_symlink():
            raise ValueError("Missing or invalid pinned wheel")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        if path.stat().st_size != wheel["size"] or checksum.hexdigest() != wheel["sha256"]:
            raise ValueError(f"Wheel checksum drift: {path.name}")
        paths.append(path)
    return paths


def install(wheelhouse: Path) -> dict:
    """Add pinned packages without resolving or replacing any existing dependency."""
    environment = json.loads(Path(__file__).with_name("environment.json").read_text())
    if sys.executable != environment["native_python"]:
        raise ValueError("Use the frozen native task interpreter inside a disposable sandbox")
    site_packages = Path(environment["site_packages"])
    before = package_versions(site_packages)
    if before != environment["existing_packages"]:
        raise ValueError("Native package inventory differs from the frozen image")
    paths = verified_wheels(environment, wheelhouse)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-index",
            "--no-deps",
            "--no-compile",
            *map(str, paths),
        ],
        check=True,
    )
    after = package_versions(site_packages)
    expected = {**before, **environment["added_packages"]}
    if after != expected:
        raise ValueError("Installed inventory differs from the proposed environment")
    return {"before": before, "after": after, "existing_package_versions_unchanged": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    with args.receipt.open("x") as receipt:
        result = install(args.wheelhouse)
        json.dump(result, receipt, indent=2)
        receipt.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
