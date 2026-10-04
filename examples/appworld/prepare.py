"""Install a reproducible isolated runtime and verify the public encrypted release."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from examples.appworld.benchmark_settings import DATA_BUNDLE_SHA256, DATA_URL, PYTHON_VERSION
from examples.appworld.runtime import inspect_runtime
from examples.appworld.utils import file_digest, load_dataset

LOCAL_RUNTIME = Path(__file__).with_name(".runtime")


def main(argv: list[str] | None = None) -> None:
    """Keep the corpus local and refuse to overwrite an existing data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LOCAL_RUNTIME / "world")
    parser.add_argument("--venv", type=Path, default=LOCAL_RUNTIME / "venv")
    args = parser.parse_args(argv)
    root, venv = args.root.resolve(), args.venv.resolve()
    root.mkdir(parents=True, exist_ok=True)
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", PYTHON_VERSION, str(venv)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(python),
            "--require-hashes",
            str(Path(__file__).with_name("runtime-requirements.txt")),
        ],
        check=True,
    )
    environment = {
        **os.environ,
        "APPWORLD_ROOT": str(root),
        "APPWORLD_CACHE": str(root / ".cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = ["uv", "run", "--no-project", "--python", str(python), "python"]
    subprocess.run([*command, "-m", "appworld.cli", "install"], env=environment, check=True)
    if not (root / "data").exists():
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            bundle = Path(temporary) / "data.bundle"
            urllib.request.urlretrieve(DATA_URL, bundle)
            if file_digest(bundle) != DATA_BUNDLE_SHA256:
                raise ValueError("AppWorld download checksum mismatch; no data were unpacked.")
            subprocess.run(
                [
                    *command,
                    "-c",
                    "import sys; from appworld.common.constants import PASSWORD, SALT; "
                    "from appworld.common.utils import unpack_bundle; "
                    "unpack_bundle(bundle_file_path=sys.argv[1], base_directory=sys.argv[2], password=PASSWORD, salt=SALT)",
                    str(bundle),
                    str(root),
                ],
                env=environment,
                check=True,
            )
    records, _ = load_dataset(root)
    inspect_runtime(root, python)
    print(
        "Verified AppWorld runtime and official splits: "
        + ", ".join(f"{split}={len(items)}" for split, items in records.items())
    )


if __name__ == "__main__":
    main()
