"""Stage only verified dependency assets in a new opt-in evaluation capsule."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from examples.swebench_live.proposals.haystack_9742_v1.install_dependencies import verified_wheels

PACKET = Path(__file__).parent


def stage_environment(wheelhouse: Path, isolated_assets: Path) -> Path:
    """Publish a separate capsule without any task, reference patch, or test result."""
    manifest = (PACKET / "environment.json").read_bytes()
    paths = verified_wheels(json.loads(manifest), wheelhouse)
    destination = isolated_assets / "evaluation-environments" / hashlib.sha256(manifest).hexdigest()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "environment.json").write_bytes(manifest)
    wheels = destination / "wheels"
    wheels.mkdir()
    for path in paths:
        shutil.copyfile(path, wheels / path.name)
    verified_wheels(json.loads(manifest), wheels)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--isolated-assets", type=Path, required=True)
    args = parser.parse_args()
    print(stage_environment(args.wheelhouse, args.isolated_assets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
