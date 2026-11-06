"""Keep the proposed scoring correction separate from frozen official inputs."""

import ast
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import aiohttp
import httpx
import pytest

from examples.swebench_live.proposals.litellm_13675_v3 import offline_inputs
from examples.swebench_live.proposals.litellm_13675_v3.prepare import CASES, PROBE_FILE, TEST_FILE, prepare


@pytest.fixture
def record_path():
    terminal = os.environ.get("SWE_LITELLM_TERMINAL_ROOT")
    if terminal is None:
        pytest.skip("Set SWE_LITELLM_TERMINAL_ROOT to the verified attempt10 archive")
    return Path(terminal) / "files/swebench_live/native-trials/8dc83e37e1ff4d2dbbedf2bd93c1909a/evaluator-record.json"


def test_proposal_preserves_frozen_record_and_all_p2p_requirements(record_path, tmp_path):
    """Replace invalid F2P fixtures explicitly while retaining every original P2P test."""
    original_bytes = record_path.read_bytes()
    original = json.loads(original_bytes)
    output = tmp_path / "proposal"
    manifest = prepare(record_path, output)
    proposed = json.loads((output / "proposed-record.json").read_text())
    assert record_path.read_bytes() == original_bytes
    assert manifest["original_p2p_count"] == 2580
    assert manifest["proposed_p2p_count"] == 2586
    assert manifest["proposed_f2p_count"] == 6
    assert manifest["original_p2p_preserved_in_order"]
    assert len(offline_inputs.SELECTED) == 14
    assert offline_inputs.SELECTED.issubset(original["PASS_TO_PASS"])
    assert not manifest["adoption_authorized"]
    assert manifest["scientific_protocol_change"]
    assert proposed["FAIL_TO_PASS"] == [f"{TEST_FILE}::test_async_boundary[{case}]" for case in CASES]
    assert proposed["PASS_TO_PASS"][2580:] == [f"{TEST_FILE}::test_sync_boundary[{case}]" for case in CASES]
    changed = {key for key in original if original[key] != proposed[key]}
    assert changed == {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}
    assert (
        hashlib.sha256((output / "proposed-record.json").read_bytes()).hexdigest() == manifest["proposed_record_sha256"]
    )


@pytest.mark.parametrize("field", ["base_commit", "test_patch", "PASS_TO_PASS"])
def test_proposal_rejects_record_drift_before_writing(record_path, tmp_path, field):
    """Reject changed task identity, tests, or required-test membership."""
    record = json.loads(record_path.read_text())
    record[field] = [] if field == "PASS_TO_PASS" else "changed"
    altered = tmp_path / "altered-record.json"
    altered.write_text(json.dumps(record))
    output = tmp_path / "proposal"
    with pytest.raises(ValueError, match="identity changed"):
        prepare(altered, output)
    assert not output.exists()


def test_proposal_never_replaces_an_existing_packet(record_path, tmp_path):
    """Preserve the first packet on repeated preparation."""
    output = tmp_path / "proposal"
    prepare(record_path, output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(record_path, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_generated_tests_apply_and_collect_all_twelve_cases(record_path, tmp_path):
    """Check that every proposed required test ID resolves in the generated patch."""
    output = tmp_path / "proposal"
    prepare(record_path, output)
    repository = tmp_path / "task-base"
    repository.mkdir()
    patch = (output / "proposed-test.patch").read_text()
    subprocess.run(["git", "apply", "--check", "-"], cwd=repository, input=patch, text=True, check=True)
    subprocess.run(["git", "apply", "-"], cwd=repository, input=patch, text=True, check=True)
    for name in (TEST_FILE, PROBE_FILE):
        source = (repository / name).read_text()
        compile(source, name, "exec")
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", TEST_FILE],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    expected = {f"{TEST_FILE}::test_{mode}_boundary[{case}]" for mode in ("sync", "async") for case in CASES}
    assert {line for line in collected.splitlines() if "::test_" in line} == expected


def activate_inputs(nodeid, monkeypatch):
    """Enter the actual fixture around an ordinary HTTP client check."""
    fixture = offline_inputs.proposed_offline_inputs.__wrapped__(
        SimpleNamespace(node=SimpleNamespace(nodeid=nodeid)), monkeypatch
    )
    next(fixture)
    return fixture


@pytest.fixture
def frozen_image_module(monkeypatch):
    """Load unchanged frozen conversion functions with an explicit in-memory cache stub."""
    source = os.environ.get("SWE_LITELLM_IMAGE_SOURCE")
    if source is None:
        pytest.skip("Set SWE_LITELLM_IMAGE_SOURCE to the verified frozen image_handling.py")
    path = Path(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "832e91ff82ae44c009a7ffe20ea79448a52a86ac8c8608f3aff4faf542e7d97d"
    )

    class Cache:
        def __init__(self, max_size_in_memory):
            self.values = {}
            self.max_size_in_memory = max_size_in_memory

        def get_cache(self, key):
            return self.values.get(key)

        def set_cache(self, key, value):
            self.values[key] = value

    module = ModuleType("litellm.litellm_core_utils.prompt_templates.image_handling")
    module.__file__ = str(path)
    module.__dict__.update(
        base64=base64,
        Response=httpx.Response,
        InMemoryCache=Cache,
        litellm=SimpleNamespace(),
        verbose_logger=SimpleNamespace(exception=lambda _: None),
    )
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_image_inputs_keep_decoding_bytes_and_reject_unexpected_requests(monkeypatch, frozen_image_module):
    """Serve only the pinned image response through sync and async client boundaries."""
    fixture = activate_inputs(sorted(offline_inputs.IMAGE_TESTS)[0], monkeypatch)
    url = sorted(offline_inputs.IMAGE_URLS)[0]
    try:
        with httpx.Client() as client:
            response = client.get(url)
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"
            assert response.content == offline_inputs.jpeg_bytes()
            with pytest.raises(RuntimeError, match="Unexpected outbound"):
                client.post("https://unexpected.invalid/model")

        async def consume():
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                assert response.content == offline_inputs.jpeg_bytes()
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    assert await response.read() == offline_inputs.jpeg_bytes()
                with pytest.raises(RuntimeError, match="Unexpected aiohttp"):
                    session.get("https://unexpected.invalid/image")

        asyncio.run(consume())
        with pytest.raises(StopIteration):
            next(fixture)
    finally:
        fixture.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_image_cache_is_local_and_restored(frozen_image_module, tmp_path, mode):
    """Require real input consumption despite an old cache, then restore unrelated state."""
    module = frozen_image_module
    original_cache = module.in_memory_cache
    url = sorted(offline_inputs.IMAGE_URLS)[0]
    original_cache.set_cache(url, "stale-image")
    original_cache.set_cache("unrelated", "keep")
    audit = tmp_path / "audit"
    expected = "data:image/jpeg;base64," + base64.b64encode(offline_inputs.jpeg_bytes()).decode()
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setenv("SWE_VARIANT_AUDIT_DIR", str(audit))
        fixture = activate_inputs(sorted(offline_inputs.IMAGE_TESTS)[0], scoped)
        try:
            if mode == "sync":
                with httpx.Client() as client:
                    module.litellm.module_level_client = client
                    assert module.convert_url_to_base64(url) == expected
                    assert module.convert_url_to_base64(url) == expected
            else:

                async def consume():
                    async with httpx.AsyncClient() as client:
                        module.litellm.module_level_aclient = client
                        assert await module.async_convert_url_to_base64(url) == expected
                        assert await module.async_convert_url_to_base64(url) == expected

                asyncio.run(consume())
            with pytest.raises(StopIteration):
                next(fixture)
        finally:
            fixture.close()
    assert module.in_memory_cache is original_cache
    assert original_cache.values == {url: "stale-image", "unrelated": "keep"}
    receipt = json.loads(next(audit.glob("*.json")).read_text())
    assert receipt["inputs"] == [{"kind": "synthetic-jpeg", "method": "GET", "url": url}]


def test_gemini_fixture_only_returns_error_for_the_missing_key_request():
    """Reject valid-key, unknown-host, and success-path requests instead of masking them."""
    nodeid = sorted(offline_inputs.GEMINI_TESTS)[0]
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    calls = []
    response = offline_inputs.response_for(httpx.Request("POST", url), nodeid, calls)
    assert response.status_code == 400
    with pytest.raises(httpx.HTTPStatusError):
        response.raise_for_status()
    for request in (
        httpx.Request("POST", url + "?key=unexpected-value"),
        httpx.Request("POST", "https://unexpected.invalid/model"),
        httpx.Request("GET", url),
    ):
        with pytest.raises(RuntimeError, match="Unexpected outbound"):
            offline_inputs.response_for(request, nodeid, calls)
    assert len(calls) == 1


@pytest.mark.parametrize("local_address", ["0.0.0.0", None])
def test_ipv4_fixture_exercises_real_transport_and_rejects_missing_binding(monkeypatch, local_address):
    """Keep actual HTTPX transport and TCP I/O while rejecting a bad IPv4 implementation."""
    fixture = activate_inputs(offline_inputs.IPV4_TEST, monkeypatch)

    async def request():
        transport = httpx.AsyncHTTPTransport(local_address=local_address)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get("http://example.com")
            assert response.status_code == 200
            assert response.content == b"ok"

    try:
        if local_address is None:
            with pytest.raises(AssertionError, match="did not bind IPv4"):
                asyncio.run(request())
        else:
            asyncio.run(request())
            with pytest.raises(StopIteration):
                next(fixture)
    finally:
        fixture.close()
