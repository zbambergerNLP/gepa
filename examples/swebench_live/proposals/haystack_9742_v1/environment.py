"""Install the opt-in Haystack environment only in a fresh grading sandbox."""

import hashlib
import json
import shlex
import shutil
from pathlib import Path

from examples.swebench_live.proposals.haystack_9742_v1.install_dependencies import verified_wheels
from examples.swebench_live.proposals.haystack_9742_v1.prepare import validate_proposed_record

PACKET = Path(__file__).parent
VARIANT = "swe-haystack-9742-offline-environment-v1"


def environment_assets(record: dict, assets_root: Path) -> Path:
    """Use the record's checked identity instead of an arbitrary asset path."""
    return assets_root / "evaluation-environments" / validate_proposed_record(record)


def install_in_grader(task, record: dict, capsule: Path) -> None:
    """Add verified missing dependencies before the existing grader applies the patch."""
    expected = validate_proposed_record(record)
    manifest = (capsule / "environment.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != expected:
        raise ValueError("Haystack environment capsule changed")
    environment = json.loads(manifest)
    if task.image["sif_sha256"] != environment["original_image_sha256"]:
        raise ValueError("Haystack grading image changed")
    wheels = verified_wheels(environment, capsule / "wheels")
    destination = task.sandbox / "opt/haystack-evaluation-environment"
    destination.mkdir()
    (destination / "environment.json").write_bytes(manifest)
    shutil.copyfile(PACKET / "install_dependencies.py", destination / "install_dependencies.py")
    (destination / "wheels").mkdir()
    for wheel in wheels:
        shutil.copyfile(wheel, destination / "wheels" / wheel.name)
    prefix = "/opt/haystack-evaluation-environment"
    result = task.execute(
        shlex.join(
            [
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                environment["native_python"],
                "-B",
                prefix + "/install_dependencies.py",
                "--wheelhouse",
                prefix + "/wheels",
                "--receipt",
                prefix + "/installation-receipt.json",
            ]
        ),
        timeout=600,
    )
    Path("evaluation-environment-installation.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["exit_code"]:
        raise RuntimeError("Haystack grading environment installation failed; inspect preserved evidence")
    Path("evaluation-environment-receipt.json").write_bytes((destination / "installation-receipt.json").read_bytes())
