"""Prepare an opt-in Haystack evaluator with explicit offline model loading."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

from examples.common.artifacts import digest

PACKET = Path(__file__).parent


def proposed_record(record: dict) -> dict:
    """Bind the unchanged frozen task to the proposed offline environment."""
    identity = json.loads((PACKET / "identity.json").read_text())
    if digest(record) != identity["original_record_digest"]:
        raise ValueError("Frozen Haystack evaluator identity changed")
    prefix = shlex.join(["env", *(f"{name}={value}" for name, value in identity["offline_environment"].items())])
    return {
        **record,
        "test_cmds": [f"{prefix} {command}" for command in record["test_cmds"]],
        "evaluation_variant": identity["variant"],
        "evaluation_variant_status": "PROPOSED_NOT_ADOPTED",
        "evaluation_environment_sha256": hashlib.sha256((PACKET / "environment.json").read_bytes()).hexdigest(),
    }


def validate_proposed_record(record: dict) -> str:
    """Reject changes to the proposed record before its environment is opened."""
    identity = json.loads((PACKET / "identity.json").read_text())
    original = {
        key: value
        for key, value in record.items()
        if key not in {"evaluation_variant", "evaluation_variant_status", "evaluation_environment_sha256"}
    }
    original["test_cmds"] = identity["original_test_commands"]
    expected = proposed_record(original)
    if record != expected:
        raise ValueError("Proposed Haystack evaluator identity changed")
    return expected["evaluation_environment_sha256"]


def prepare(record_path: Path, output: Path) -> dict:
    """Preserve the task and tests, and bind offline loading to a new evaluator identity."""
    identity = json.loads((PACKET / "identity.json").read_text())
    environment_bytes = (PACKET / "environment.json").read_bytes()
    environment_digest = hashlib.sha256(environment_bytes).hexdigest()
    original = record_path.read_bytes()
    record = json.loads(original)
    environment = identity["offline_environment"]
    proposed = proposed_record(record)
    proposed_bytes = (json.dumps(proposed, indent=2, ensure_ascii=False) + "\n").encode()
    manifest = {
        "variant": identity["variant"],
        "status": "PROPOSED_NOT_ADOPTED",
        "adoption_authorized": False,
        "scientific_protocol_change": True,
        "original_record_sha256": hashlib.sha256(original).hexdigest(),
        "original_record_digest": digest(record),
        "original_image_sha256": identity["original_image_sha256"],
        "proposed_record_sha256": hashlib.sha256(proposed_bytes).hexdigest(),
        "evaluation_contract": {"evaluation_variant": identity["variant"], "evaluator_sha256": digest(proposed)},
        "offline_environment": environment,
        "environment_manifest_sha256": environment_digest,
        "environment_asset_subdirectory": f"evaluation-environments/{environment_digest}",
        "required_container_backend": "apptainer",
        "f2p_count": len(record["FAIL_TO_PASS"]),
        "p2p_count": len(record["PASS_TO_PASS"]),
        "original_test_patch_unchanged": True,
        "original_reference_patch_unchanged": True,
        "original_required_tests_unchanged": True,
        "native_validation": "NOT_RUN_BY_PREPARATION",
        "remaining_gate": "Verify pinned native dependencies and cached assets, then grade base, reference, and saved candidate.",
        "checkpoint_compatibility": "Requires a new runtime contract; never resume or relabel an earlier evaluation.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "proposed-record.json").write_bytes(proposed_bytes)
    (output / "environment.json").write_bytes(environment_bytes)
    (output / "install_dependencies.py").write_bytes((PACKET / "install_dependencies.py").read_bytes())
    (output / "proposed-test.patch").write_text(record["test_patch"])
    (output / "proposed-reference.patch").write_text(record["patch"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.record, args.output)
    print(json.dumps({"status": manifest["status"], "adoption_authorized": False, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
