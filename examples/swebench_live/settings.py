"""Pin SWE-bench Live independently of the older SWE-bench Verified prototype."""

from pathlib import Path

DATASET = "SWE-bench-Live/SWE-bench-Live"
DATA_REVISION = "b51a86422e10cfd403beb4773e5a2947953e36ec"
SOURCE_URL = "https://github.com/microsoft/SWE-bench-Live.git"
SOURCE_REVISION = "ad79b850f15e33992e96f03f6e97f05ddf9aa0be"
ROOT = Path(__file__).resolve().parents[2] / ".cache" / "swebench-live"
COMPONENT = "system_prompt"
PARTITION = "full-minus-official-test-sha256-id-halves-v1"
PARQUET_HASHES = {
    "full-00000-of-00002.parquet": "9371afaef65bb6e21417fb76e5cbd89a5bc5241e197db3ead4eb0353007e740c",
    "full-00001-of-00002.parquet": "12e4956a1761efaeb121afe13a5371d8d439a7708b4638397d0b3d6cce1e8407",
    "lite-00000-of-00001.parquet": "7ee0a75c41bfc954fd441b67ce738fc5c1cbae00721c4e30e7db4d893057c9ab",
    "test-00000-of-00001.parquet": "7d6e65af7708af383738eb9a5c13be97d8234b85c00116793f05ffdc8a89e5ff",
    "verified-00000-of-00001.parquet": "080e36e46198bf9c177a6b077624d4028baf6ff04d661c332cc1fe1e5dfa50b2",
}
PREPARED_HASHES = {
    "full": "3711840c07c5ea8b47c482457a784cbc05bdf70d973723fe4043f5e284c75f6c",
    "lite": "c81e43d3b254c9f36751f9fc74264491f0c976c54a3e5ac3300898821a391774",
    "test": "d05f27f0bfbaa4db07bd99bf86f41293903b88fc054ed415bbdc2c7813e420af",
    "verified": "5216073c56ba7b3419b56b8f33ea0cddb6fcdd188f30730df096431ab526f5b6",
    "train": "42bac24ad50393db61728bdc947e05c1dfd763735cd3e222719675a03d4be69d",
    "val": "dd17e86d9a95b4d3fbcad572974dbc737a432b4e8898962245a10ae197e90b7e",
}
SEED_PROMPT = """Resolve the software issue in the repository at /testbed.
Inspect the source and available tests, make a minimal correct change, and run relevant tests.
Use the shell tool to work inside the task container. Each command runs in a new shell;
use explicit paths or cd when needed. Do not change tests to hide failures. Do not fetch
solutions or access external services. When finished, explain the change in your final
response. The submitted solution is the repository diff, including staged new files."""


def image_tag(instance_id: str) -> str:
    """Return the upstream Python task's published AMD64 image name."""
    return f"starryzhang/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}:latest"
