"""Probe the full installed LiteLLM parser inside an offline task image."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    distributions = {}
    for distribution in importlib.metadata.distributions():
        distributions[distribution.metadata["Name"]] = distribution.version
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "distributions": dict(sorted(distributions.items())),
        "project_files": {
            name: sha256(root / name) for name in ("pyproject.toml", "poetry.lock") if (root / name).is_file()
        },
    }


async def check_and_close(check, close):
    """Keep parsing and cleanup failures without weakening either requirement."""
    result = {}
    try:
        try:
            result["parsing"] = {"outcome": "pass", **await check()}
        except Exception as error:
            result["parsing"] = {"outcome": "fail", "exception": type(error).__name__, "message": str(error)}
    finally:
        try:
            await close()
            result["cleanup"] = {"outcome": "pass"}
        except Exception as error:
            result["cleanup"] = {"outcome": "fail", "exception": type(error).__name__, "message": str(error)}
    result["outcome"] = "pass" if all(value["outcome"] == "pass" for value in result.values()) else "fail"
    return result


async def probe(root):
    import httpx
    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
    from litellm.llms.databricks.streaming_utils import ModelResponseIterator
    from litellm.llms.openai_like.chat.handler import make_call, make_sync_call
    from multidict import CIMultiDict, CIMultiDictProxy

    if httpx.__version__ != "0.28.1":
        raise ValueError("Use the frozen HTTPX 0.28.1 environment")

    sources = {}
    for symbol in (
        LiteLLMAiohttpTransport,
        ModelResponseIterator,
        make_call,
        make_sync_call,
    ):
        path = Path(inspect.getsourcefile(symbol)).resolve()
        if not path.is_relative_to(root / "litellm"):
            raise ValueError(f"Wrong native source for {symbol.__name__}: {path}")
        sources[symbol.__name__] = {"path": str(path), "sha256": sha256(path)}

    text_parts = ["µ © ™ ", "€ 北京 🚀 Ñoño"]
    records = []
    for text in text_parts:
        records.append(
            {
                "id": "offline-native-probe",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "offline-native-probe",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
        )
    records.append(
        {
            "id": "offline-native-probe",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "offline-native-probe",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    payload = (
        "\r\n\r\n".join("data: " + json.dumps(item, ensure_ascii=False) for item in records) + "\r\n\r\ndata: [DONE]"
    ).encode()
    headers = [(b"content-type", b"text/event-stream; charset=utf-8")]
    cases = {
        "whole_body": ([payload], headers),
        "every_byte": ([bytes([byte]) for byte in payload], headers),
        "split_sse_record": ([payload[:13], payload[13:]], headers),
        "unicode_header": ([payload], headers + [(b"x-reset", "6.083µs".encode())]),
        "duplicate_headers": (
            [payload],
            headers + [(b"x-reset", "µ".encode()), (b"x-reset", b"second")],
        ),
        "opaque_header": ([payload], headers + [(b"x-opaque", b"\xff\xfe")]),
    }

    class SyncChunks(httpx.SyncByteStream):
        def __init__(self, chunks):
            self.chunks, self.consumed = chunks, 0

        def __iter__(self):
            for chunk in self.chunks:
                self.consumed += 1
                yield chunk

    class UpstreamResponse:
        def __init__(self, chunks, raw_headers):
            self.status = 200
            self.headers = CIMultiDictProxy(
                CIMultiDict(
                    [
                        (
                            key.decode("utf-8", "surrogateescape"),
                            value.decode("utf-8", "surrogateescape"),
                        )
                        for key, value in raw_headers
                    ]
                )
            )
            self.content = self
            self.chunks, self.consumed, self.closed = chunks, 0, False

        async def iter_chunked(self, size):
            for chunk in self.chunks:
                self.consumed += 1
                yield chunk

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

    async def consume(chunks, raw_headers, asynchronous):
        response = None
        if asynchronous:
            upstream = UpstreamResponse(chunks, raw_headers)
            session = SimpleNamespace(request=Mock(return_value=upstream))
            transport = object.__new__(LiteLLMAiohttpTransport)
            transport._get_valid_client_session = lambda: session
            transport._get_proxy_settings = AsyncMock(return_value=None)
            response = await transport.handle_async_request(
                httpx.Request("POST", "https://unused.invalid", content=b"{}")
            )
            stream = upstream
        else:
            stream = SyncChunks(chunks)
            response = httpx.Response(
                200,
                headers=raw_headers,
                stream=stream,
                request=httpx.Request("POST", "https://unused.invalid"),
            )

        async def parse():
            names = {name for name, _ in raw_headers}
            assert [pair for pair in response.headers.raw if pair[0] in names] == raw_headers
            kwargs = {
                "api_base": "https://unused.invalid",
                "headers": {},
                "data": "{}",
                "model": "offline-native-probe",
                "messages": [],
                "logging_obj": Mock(),
            }
            if asynchronous:
                iterator = await make_call(
                    client=SimpleNamespace(post=AsyncMock(return_value=response)),
                    **kwargs,
                )
                assert isinstance(iterator, ModelResponseIterator)
                parsed = [chunk async for chunk in iterator]
            else:
                iterator = make_sync_call(client=SimpleNamespace(post=Mock(return_value=response)), **kwargs)
                assert isinstance(iterator, ModelResponseIterator)
                parsed = list(iterator)
            assert "".join(chunk["text"] for chunk in parsed) == "".join(text_parts), "Lost or split parsed text"
            finished = [chunk for chunk in parsed if chunk.get("is_finished")]
            assert len(finished) == 1 and finished[0]["finish_reason"] == "stop", "Lost finish state"
            assert stream.consumed == len(chunks), "Stream was not fully consumed"
            return {
                "parsed_chunks": len(parsed),
                "consumed_chunks": stream.consumed,
            }

        async def close():
            if asynchronous:
                await response.aclose()
                assert stream.closed, "Native response stream did not close the upstream response"
            else:
                response.close()

        return await check_and_close(parse, close)

    results = {}
    for label, (chunks, raw_headers) in cases.items():
        results[label] = {}
        for asynchronous in (False, True):
            try:
                result = await consume(chunks, raw_headers, asynchronous)
            except Exception as error:  # Preserve diagnostic failures as evidence.
                result = {
                    "outcome": "fail",
                    "exception": type(error).__name__,
                    "message": str(error),
                }
            results[label]["async" if asynchronous else "sync"] = result
    return {
        "httpx_version": httpx.__version__,
        "httpx_iter_text_signature": str(inspect.signature(httpx.Response.iter_text)),
        "httpx_aiter_text_signature": str(inspect.signature(httpx.Response.aiter_text)),
        "source_files": sources,
        "cases": results,
        "all_cases_pass": all(value["outcome"] == "pass" for case in results.values() for value in case.values()),
        "dispatch_boundary": "Actual native transport response construction, HTTPX streams, handler and parser; HTTP request/session and logging only are simulated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("base", "frozen_reference", "candidate"), required=True)
    parser.add_argument("--repo", type=Path, default=Path("/testbed"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace prior evidence")
    root = args.repo.resolve()
    if not (root / "poetry.lock").is_file() or not (root / "litellm").is_dir():
        raise ValueError("Run this probe in the immutable task image's existing Poetry environment")
    sys.path.insert(0, str(root))
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    result = {
        "variant": args.variant,
        "environment": inventory(root),
        "status": "DIAGNOSTIC_ONLY_NOT_SCORING_ADOPTION",
        "model_requests": 0,
    }
    try:
        result["boundary"] = asyncio.run(probe(root))
    except Exception as error:  # Preserve diagnostic failures as evidence.
        result["setup_error"] = {
            "exception": type(error).__name__,
            "message": str(error),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "report_written": True,
                "setup_error": "setup_error" in result,
            }
        )
    )
    return 2 if "setup_error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
