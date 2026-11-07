"""Prepare a separate proposed evaluator with one additional scoped clock input."""

import argparse
import json
from pathlib import Path

from examples.common.artifacts import digest
from examples.swebench_live.proposals.litellm_13675_v3 import prepare as previous

PACKET = Path(__file__).parent
PREVIOUS_FILE = "tests/test_litellm/_proposed_v3_offline_inputs.py"


def prepare(record_path, output):
    """Verify pinned v3 and v4 bytes, then emit a new unadopted evaluation identity."""
    identity = json.loads((PACKET / "identity.json").read_text())
    offline = (PACKET / "offline_inputs.py").read_bytes()
    old_offline = (previous.PACKET / "offline_inputs.py").read_bytes()
    reused = {
        name: previous.sha256((previous.PACKET / name).read_bytes())
        for name in ("prepare.py", "identity.json", "offline_inputs.py")
    }
    if reused != identity["reused_v3_sha256"] or previous.sha256(offline) != identity["offline_inputs_sha256"]:
        raise ValueError("Pinned evaluation variant source identity changed")
    manifest = previous.prepare(record_path, output)
    old_record = json.loads((output / "proposed-record.json").read_text())
    old_patch = previous.added_file(previous.OFFLINE_FILE, old_offline.decode())
    if old_record["test_patch"].count(old_patch) != 1:
        raise ValueError("Expected one pinned v3 fixture in the generated test patch")
    test_patch = old_record["test_patch"].replace(
        old_patch,
        previous.added_file(previous.OFFLINE_FILE, offline.decode())
        + previous.added_file(PREVIOUS_FILE, old_offline.decode()),
    )
    record = {**old_record, "test_patch": test_patch, "evaluation_variant": identity["variant"]}
    record_bytes = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode()
    manifest.update(
        variant=identity["variant"],
        reused_v3_sha256=reused,
        previous_variant=old_record["evaluation_variant"],
        previous_proposed_record_sha256=manifest["proposed_record_sha256"],
        inputs_sha256={**manifest["inputs_sha256"], "offline_inputs": previous.sha256(offline)},
        proposed_record_sha256=previous.sha256(record_bytes),
        proposed_test_patch_sha256=previous.sha256(test_patch.encode()),
        evaluation_contract={"evaluation_variant": identity["variant"], "evaluator_sha256": digest(record)},
        checkpoint_compatibility="Incompatible with v3 and all earlier evaluators; a new run contract is required.",
    )
    manifest["offline_inputs"].update(
        scope="Fifteen exact original P2P test IDs; all original assertions stay unchanged.",
        clock_input="Three exact limiter tests get same-minute clocks; all previous module clocks are restored.",
        reused_plugin=PREVIOUS_FILE,
    )
    (output / "proposed-record.json").write_bytes(record_bytes)
    (output / "proposed-test.patch").write_text(test_patch)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.record, args.output)
    print(json.dumps({"status": result["status"], "adoption_authorized": False, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
