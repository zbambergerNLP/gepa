"""Supply scoped offline inputs and a deterministic clock for thirteen tests."""

import base64
import hashlib
import importlib
import json
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import httpx
import pytest
from multidict import CIMultiDict

IMAGE_URLS = frozenset(
    {
        "https://blog.purpureus.net/assets/blog/personal_key_rotation/simplified-asset-graph.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    }
)
GEMINI_TESTS = frozenset(
    "tests/test_litellm/llms/gemini/test_gemini_client_setup.py::" + name
    for name in ("test_gemini_completion_no_api_key", "test_gemini_completion_no_api_key_with_mock")
)
IMAGE_TESTS = frozenset(
    {
        "tests/test_litellm/litellm_core_utils/test_token_counter.py::test_img_url_token_counter["
        "https://blog.purpureus.net/assets/blog/personal_key_rotation/simplified-asset-graph.jpg]",
        *(
            f"tests/test_litellm/test_main.py::test_url_with_format_param[{sync}-{model}]"
            for sync in ("False", "True")
            for model in (
                "anthropic/claude-3-5-sonnet",
                "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
                "gemini/gemini-1.5-flash",
            )
        ),
    }
)
IPV4_TEST = "tests/test_litellm/llms/custom_httpx/test_http_handler.py::test_force_ipv4_transport"
LIMITER_TEST = (
    "tests/test_litellm/proxy/hooks/test_parallel_request_limiter.py::test_pre_call_hook_rpm_limits_retry_after"
)
SELECTED = IMAGE_TESTS | GEMINI_TESTS | {IPV4_TEST, LIMITER_TEST}
IMAGE_SOURCE_SHA256 = "832e91ff82ae44c009a7ffe20ea79448a52a86ac8c8608f3aff4faf542e7d97d"
LIMITER_SOURCE_SHA256 = "d16fd09f3ebe3d062fb63804faf5398fe1ad30441cd1d18b5126e7c5ebd86553"


def jpeg_bytes():
    """Read the pinned synthetic JPEG used instead of mutable public image URLs."""
    path = Path(__file__).with_name("_proposed_asset.jpg.b64")
    if not path.exists():
        path = Path(__file__).with_name("asset.jpg.b64")
    raw = base64.b64decode(path.read_text(), validate=False)
    if hashlib.sha256(raw).hexdigest() != "123de103ab7eb361ec1a90cd1d596ccb297d5a15678ea83c7423e9f066438994":
        raise ValueError("Proposed image fixture identity changed")
    return raw


def response_for(request, nodeid, calls):
    """Return only a declared image input or a simulated missing-key error."""
    if nodeid in IMAGE_TESTS and request.method == "GET" and str(request.url) in IMAGE_URLS:
        calls.append({"kind": "synthetic-jpeg", "method": "GET", "url": str(request.url)})
        return httpx.Response(200, content=jpeg_bytes(), headers={"content-type": "image/jpeg"}, request=request)
    if (
        nodeid in GEMINI_TESTS
        and request.method == "POST"
        and request.url.host == "generativelanguage.googleapis.com"
        and request.url.path
        in ("/v1/models/gemini-1.5-flash:generateContent", "/v1beta/models/gemini-1.5-flash:generateContent")
        and request.url.params.get("key") in (None, "", "None")
    ):
        calls.append({"kind": "simulated-missing-key-error", "method": "POST", "path": request.url.path})
        return httpx.Response(
            400,
            json={"error": {"code": 400, "message": "API key is missing or invalid.", "status": "INVALID_ARGUMENT"}},
            request=request,
        )
    raise RuntimeError("Unexpected outbound request in proposed offline fixture")


class AssetResponse:
    """Model only aiohttp's public binary-response interface for the fixed image."""

    def __init__(self, url):
        self.status = 200
        self.url = url
        self.headers = CIMultiDict({"Content-Type": "image/jpeg"})
        self.content = self

    async def read(self):
        return jpeg_bytes()

    async def iter_chunked(self, size):
        raw = jpeg_bytes()
        for offset in range(0, len(raw), size):
            yield raw[offset : offset + size]

    def raise_for_status(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


def record_audit(nodeid, calls):
    """Persist fixture use without request bodies, headers, or credential values."""
    folder = os.environ.get("SWE_VARIANT_AUDIT_DIR")
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(nodeid.encode()).hexdigest() + ".json"
        with (path / name).open("x") as output:
            json.dump({"test": nodeid, "inputs": calls, "official_test_assertions_unchanged": True}, output, indent=2)


@pytest.fixture(autouse=True)
def proposed_offline_inputs(request, monkeypatch):
    """Patch only declared external inputs for exact selected test IDs."""
    nodeid = request.node.nodeid
    if nodeid not in SELECTED:
        yield
        return
    calls = []
    if nodeid == LIMITER_TEST:
        limiter = importlib.import_module("litellm.proxy.hooks.parallel_request_limiter")
        if hashlib.sha256(Path(limiter.__file__).read_bytes()).hexdigest() != LIMITER_SOURCE_SHA256:
            raise ValueError("Frozen rate limiter implementation changed")

        class SameMinuteClock:
            @classmethod
            def now(cls):
                calls.append({"kind": "deterministic-rate-limit-clock", "logical_time": "2025-01-01T12:00:10"})
                return datetime(2025, 1, 1, 12, 0, 10)

        monkeypatch.setattr(limiter, "datetime", SameMinuteClock)
        try:
            yield
        finally:
            record_audit(nodeid, calls)
        assert calls, "The selected test did not exercise its declared clock input"
        return
    if nodeid in IMAGE_TESTS:
        image_handling = importlib.import_module("litellm.litellm_core_utils.prompt_templates.image_handling")
        if hashlib.sha256(Path(image_handling.__file__).read_bytes()).hexdigest() != IMAGE_SOURCE_SHA256:
            raise ValueError("Frozen image input implementation changed")
        monkeypatch.setattr(
            image_handling,
            "in_memory_cache",
            image_handling.InMemoryCache(max_size_in_memory=image_handling.MAX_IMGS_IN_MEMORY),
        )
    if nodeid != IPV4_TEST:

        def send(client, outgoing, *args, **kwargs):
            return response_for(outgoing, nodeid, calls)

        async def async_send(client, outgoing, *args, **kwargs):
            return response_for(outgoing, nodeid, calls)

        def get(session, url, *args, **kwargs):
            if nodeid not in IMAGE_TESTS or str(url) not in IMAGE_URLS:
                raise RuntimeError("Unexpected aiohttp request in proposed offline fixture")
            calls.append({"kind": "synthetic-jpeg", "method": "GET", "url": str(url)})
            return AssetResponse(url)

        monkeypatch.setattr(httpx.Client, "send", send)
        monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
        monkeypatch.setattr(aiohttp.ClientSession, "get", get)
        try:
            yield
        finally:
            record_audit(nodeid, calls)
        assert calls, "The selected test did not exercise its declared input boundary"
        return

    class LocalHTTP(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/"
            calls.append({"kind": "real-ipv4-loopback", "peer": self.client_address[0]})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalHTTP)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = httpx.AsyncHTTPTransport.handle_async_request

    async def local_request(transport, outgoing):
        assert outgoing.method == "GET" and str(outgoing.url) == "http://example.com"
        assert transport._pool._local_address == "0.0.0.0", "The real transport did not bind IPv4"
        local = httpx.Request("GET", f"http://127.0.0.1:{server.server_port}/")
        return await original(transport, local)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", local_request)
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        record_audit(nodeid, calls)
    assert not thread.is_alive()
    assert calls == [{"kind": "real-ipv4-loopback", "peer": "127.0.0.1"}]
