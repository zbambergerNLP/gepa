"""Check proposed stream ownership and failure reporting without provider calls."""

import ast
import asyncio
import contextlib
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import httpx
import pytest

from examples.swebench_live.proposals.litellm_13675_v2 import native_probe
from examples.swebench_live.proposals.litellm_13675_v2.native_probe import check_and_close


@pytest.fixture(scope="module")
def proposed_transport():
    """Load the exact archived transport and apply the selected proposal."""
    terminal = os.environ.get("SWE_LITELLM_TERMINAL_ROOT")
    if terminal is None:
        pytest.skip("Set SWE_LITELLM_TERMINAL_ROOT to the hash-verified attempt10 archive")
    packet = Path(
        os.environ.get(
            "LITELLM_PROPOSAL_ROOT",
            Path(__file__).resolve().parents[1] / "examples/swebench_live/proposals/litellm_13675_v2",
        )
    )
    spec = importlib.util.spec_from_file_location("litellm_proposal_check", packet / "check.py")
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    _, _, source, patch, _ = checker.load_evidence(Path(terminal))
    corrected = checker.apply_patch(source, checker.TRANSPORT, patch)
    tree = ast.parse("from __future__ import annotations\n" + source)
    stream = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AiohttpResponseStream")
    namespace = {
        "httpx": httpx,
        "aiohttp": aiohttp,
        "verbose_logger": logging.getLogger(__name__),
        "map_aiohttp_exceptions": contextlib.nullcontext,
    }
    exec(
        compile(ast.Module(body=[tree.body[0], stream], type_ignores=[]), "<pinned-aiohttp-stream>", "exec"), namespace
    )
    return checker.response_constructor(corrected), namespace["AiohttpResponseStream"]


@pytest.mark.parametrize("consume_all", [False, True], ids=["early-close", "full-consumption"])
def test_public_async_client_closes_native_stream(proposed_transport, consume_all):
    """Require the public HTTPX owner to release the native response exactly once."""
    assert httpx.__version__ == "0.28.1"
    constructor, native_stream = proposed_transport

    class Upstream:
        status = 200

        def __init__(self):
            self.headers = {"content-type": "text/plain; charset=utf-8"}
            self.content = self
            self.consumed = 0
            self.close_calls = 0

        async def iter_chunked(self, size):
            for chunk in (b"first", b"second"):
                self.consumed += 1
                yield chunk

        async def __aexit__(self, *args):
            self.close_calls += 1

    upstream = Upstream()

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return eval(
                constructor,
                {"httpx": httpx, "response": upstream, "request": request, "AiohttpResponseStream": native_stream},
            )

    async def exercise():
        async with httpx.AsyncClient(transport=Transport(), trust_env=False) as client:
            async with client.stream("POST", "https://unused.invalid", content=b"{}") as response:
                if consume_all:
                    assert await response.aread() == b"firstsecond"
                else:
                    iterator = response.aiter_raw()
                    assert await anext(iterator) == b"first"
                    assert upstream.consumed == 1
            assert response.is_closed
            await response.aclose()
        assert upstream.consumed == (2 if consume_all else 1)
        assert upstream.close_calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("parser_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_diagnostic_preserves_parser_and_cleanup_results(parser_fails, cleanup_fails):
    """Report both failures and require both stages to pass."""
    calls = []

    async def parse():
        calls.append("parse")
        if parser_fails:
            raise TypeError("original parser error")
        return {"parsed_chunks": 3}

    async def close():
        calls.append("close")
        if cleanup_fails:
            raise AssertionError("Native response stream did not close the upstream response")

    result = asyncio.run(check_and_close(parse, close))
    assert calls == ["parse", "close"]
    assert result["outcome"] == ("fail" if parser_fails or cleanup_fails else "pass")
    assert result["parsing"]["outcome"] == ("fail" if parser_fails else "pass")
    assert result["cleanup"]["outcome"] == ("fail" if cleanup_fails else "pass")
    if parser_fails:
        assert result["parsing"]["exception"] == "TypeError"
        assert result["parsing"]["message"] == "original parser error"
    if cleanup_fails:
        assert result["cleanup"]["exception"] == "AssertionError"
        assert "did not close" in result["cleanup"]["message"]


def test_cancellation_runs_cleanup_and_stays_cancelled():
    """Keep cancellation distinct from a completed diagnostic result."""
    state = SimpleNamespace(closed=False)

    async def parse():
        raise asyncio.CancelledError

    async def close():
        state.closed = True

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(check_and_close(parse, close))
    assert state.closed


def test_native_case_preserves_original_error_when_closure_fails():
    """Exercise the actual diagnostic case with independent parser and close failures."""
    path = Path(os.environ.get("LITELLM_NATIVE_PROBE_SOURCE", native_probe.__file__))
    tree = ast.parse(path.read_text())
    probe = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "probe")
    consume = next(node for node in probe.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "consume")
    calls = []

    class Upstream:
        def __init__(self, *args):
            self.closed = False

    async def close():
        calls.append("close")

    class Transport:
        async def handle_async_request(self, request):
            return SimpleNamespace(headers=SimpleNamespace(raw=[]), aclose=close)

    async def fail_parser(**kwargs):
        calls.append("parse")
        raise TypeError("original parser error")

    namespace = {
        "httpx": httpx,
        "UpstreamResponse": Upstream,
        "SimpleNamespace": SimpleNamespace,
        "Mock": Mock,
        "AsyncMock": AsyncMock,
        "LiteLLMAiohttpTransport": Transport,
        "make_call": fail_parser,
        "check_and_close": check_and_close,
    }
    exec(compile(ast.Module(body=[consume], type_ignores=[]), str(path), "exec"), namespace)
    result = asyncio.run(namespace["consume"]([], [], True))
    assert calls == ["parse", "close"]
    assert result["outcome"] == "fail"
    assert result["parsing"]["exception"] == "TypeError"
    assert result["parsing"]["message"] == "original parser error"
    assert result["cleanup"]["exception"] == "AssertionError"
    assert result["cleanup"]["message"] == "Native response stream did not close the upstream response"
