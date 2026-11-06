"""Materialize a proposed LiteLLM evaluation variant without adopting it."""

import argparse
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).parent
PREVIOUS = PACKET.with_name("litellm_13675_v2")
TEST_FILE = "tests/test_litellm/test_groq_streaming_encoding.py"
PROBE_FILE = "tests/test_litellm/_proposed_streaming_probe.py"
OFFLINE_FILE = "tests/test_litellm/_proposed_offline_inputs.py"
ASSET_FILE = "tests/test_litellm/_proposed_asset.jpg.b64"
CASES = ("whole_body", "every_byte", "split_sse_record", "unicode_header", "duplicate_headers", "opaque_header")
TEST_SOURCE = '''"""Proposed native streaming checks; not the frozen official test fixture."""

import asyncio
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def boundary():
    path = Path(__file__).with_name("_proposed_streaming_probe.py")
    spec = importlib.util.spec_from_file_location("proposed_streaming_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return asyncio.run(module.probe(Path(__file__).resolve().parents[2]))


@pytest.mark.parametrize("case", CASES)
def test_sync_boundary(boundary, case):
    result = boundary["cases"][case]["sync"]
    assert result["outcome"] == "pass", result


@pytest.mark.parametrize("case", CASES)
def test_async_boundary(boundary, case):
    result = boundary["cases"][case]["async"]
    assert result["outcome"] == "pass", result
'''.replace("CASES", repr(CASES))


def sha256(content):
    """Return the identity of an input or generated artifact."""
    return hashlib.sha256(content).hexdigest()


def added_file(path, text):
    """Encode a new test file as a patch against the unchanged task base."""
    lines = text.splitlines()
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n" + "".join("+" + line + "\n" for line in lines)
    )


def prepare(record_path, output):
    """Pin all frozen inputs, then write a separate unadopted record and manifest."""
    identity = json.loads((PACKET / "identity.json").read_text())
    original = record_path.read_bytes()
    record = json.loads(original)
    reference = (PREVIOUS / "reference.patch").read_bytes()
    probe = (PREVIOUS / "native_probe.py").read_bytes()
    offline = (PACKET / "offline_inputs.py").read_bytes()
    asset = (PACKET / "asset.jpg.b64").read_bytes()
    actual = {
        "original_record": sha256(original),
        "original_reference": sha256(record["patch"].encode()),
        "original_test_patch": sha256(record["test_patch"].encode()),
        "proposed_reference": sha256(reference),
        "native_probe": sha256(probe),
        "offline_inputs": sha256(offline),
        "offline_image_fixture": sha256(asset),
    }
    if actual != identity["inputs_sha256"]:
        raise ValueError("Frozen record or proposed source identity changed")
    if record["instance_id"] != identity["instance_id"] or record["base_commit"] != identity["base_commit"]:
        raise ValueError("Unexpected task identity")
    test_patch = (
        added_file(TEST_FILE, TEST_SOURCE)
        + added_file(PROBE_FILE, probe.decode())
        + added_file(OFFLINE_FILE, offline.decode())
        + added_file(ASSET_FILE, asset.decode())
    )
    proposed = {
        **record,
        "patch": reference.decode(),
        "test_patch": test_patch,
        "FAIL_TO_PASS": [f"{TEST_FILE}::test_async_boundary[{case}]" for case in CASES],
        "PASS_TO_PASS": record["PASS_TO_PASS"] + [f"{TEST_FILE}::test_sync_boundary[{case}]" for case in CASES],
        "evaluation_variant": identity["variant"],
        "evaluation_variant_status": "PROPOSED_NOT_ADOPTED",
    }
    proposed_bytes = (json.dumps(proposed, indent=2, ensure_ascii=False) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    (output / "proposed-record.json").write_bytes(proposed_bytes)
    (output / "proposed-test.patch").write_text(test_patch)
    (output / "proposed-reference.patch").write_bytes(reference)
    manifest = {
        "variant": identity["variant"],
        "status": "PROPOSED_NOT_ADOPTED",
        "adoption_authorized": False,
        "scientific_protocol_change": True,
        "inputs_sha256": actual,
        "proposed_record_sha256": sha256(proposed_bytes),
        "proposed_test_patch_sha256": sha256(test_patch.encode()),
        "original_f2p_count": len(record["FAIL_TO_PASS"]),
        "proposed_f2p_count": len(proposed["FAIL_TO_PASS"]),
        "original_p2p_count": len(record["PASS_TO_PASS"]),
        "proposed_p2p_count": len(proposed["PASS_TO_PASS"]),
        "original_p2p_preserved_in_order": proposed["PASS_TO_PASS"][: len(record["PASS_TO_PASS"])]
        == record["PASS_TO_PASS"],
        "original_inputs_unchanged": record_path.read_bytes() == original,
        "native_validation": "NOT_RUN_BY_PREPARATION",
        "offline_inputs": {
            "pytest_plugin": "_proposed_offline_inputs",
            "pythonpath_entry": "/testbed/tests/test_litellm",
            "scope": "Twelve exact original P2P test IDs; all original assertions stay unchanged.",
            "image_input": "Pinned synthetic JPEG replaces the two mutable external image responses.",
            "gemini_input": "Simulated missing-key HTTP400; no successful provider response or credential supplied.",
            "ipv4_input": "Actual IPv4 HTTP transport to a private loopback server instead of example.com.",
            "unexpected_requests": "Fail locally; never return a blanket success response.",
        },
        "remaining_gate": "Run and pass every P2P requirement in the pinned native image before claiming variant validity.",
        "checkpoint_compatibility": "New record identity; prior scores and checkpoints must not be relabeled or resumed.",
    }
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
