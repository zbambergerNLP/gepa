"""Check the proposed LiteLLM response boundary without running a benchmark or HTTP request."""

import argparse
import ast
import asyncio
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from multidict import CIMultiDict, CIMultiDictProxy

PACKET = Path(__file__).parent
HANDLER = "litellm/llms/openai_like/chat/handler.py"
TRANSPORT = "litellm/llms/custom_httpx/aiohttp_transport.py"
TEXT = 'data: {"content":"µ © ™ € 北京 🚀 Ñoño"}\r\n\r\ndata: second\n\ndata: [DONE]'
PAYLOAD = TEXT.encode()
EXPECTED_LINES = TEXT.splitlines()
RAW_HEADERS = [(b"content-type", b"text/event-stream; charset=utf-8")]
CASES = {
    "whole_body": ([PAYLOAD], RAW_HEADERS),
    "every_byte": ([bytes([byte]) for byte in PAYLOAD], RAW_HEADERS),
    "split_line": ([PAYLOAD[:8], PAYLOAD[8:]], RAW_HEADERS),
    "unicode_header": ([PAYLOAD], RAW_HEADERS + [(b"x-reset", "6.083µs".encode())]),
    "duplicate_headers": ([PAYLOAD], RAW_HEADERS + [(b"x-reset", "µ".encode()), (b"x-reset", b"second")]),
    "non_utf8_header": ([PAYLOAD], RAW_HEADERS + [(b"x-opaque", b"\xff\xfe")]),
}


class SyncChunks(httpx.SyncByteStream):
    """Supply fixed network chunks without a socket."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk


class AsyncChunks(httpx.AsyncByteStream):
    """Supply fixed asynchronous network chunks without a socket."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def apply_patch(source, path, patch):
    """Apply a proposed patch to a disposable source copy only."""
    with tempfile.TemporaryDirectory(prefix="litellm-reference-") as directory:
        target = Path(directory) / path
        target.parent.mkdir(parents=True)
        target.write_text(source)
        subprocess.run(["git", "apply", "--check", "-"], input=patch, text=True, cwd=directory, check=True)
        subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=directory, check=True)
        return target.read_text()


def load_evidence(terminal_root):
    """Reject changed inputs before compiling any saved source or test definitions."""
    trials = terminal_root / "files/swebench_live/native-trials"
    record_bytes = (trials / "8dc83e37e1ff4d2dbbedf2bd93c1909a/evaluator-record.json").read_bytes()
    record = json.loads(record_bytes)
    handler_trace = json.loads((trials / "1cb6def1d7834314b1569a155f27ef41/trajectory.json").read_text())
    transport_trace = json.loads((trials / "8dc83e37e1ff4d2dbbedf2bd93c1909a/trajectory.json").read_text())
    handler = handler_trace["trajectory"][11]["observation"].strip().split("\nclass OpenAILikeChatHandler(", 1)[0]
    transport = transport_trace["trajectory"][0]["observation"].strip().split("---HTTP_HANDLER---", 1)[0]
    transport = transport.split("    async def _get_proxy_settings(", 1)[0]
    handler, transport = handler.rstrip() + "\n", transport.rstrip() + "\n"
    candidate = (PACKET / "reference.patch").read_text()
    identities = json.loads((PACKET / "identity.json").read_text())
    actual = {
        "evaluator_record": sha256(record_bytes),
        "frozen_reference": sha256(record["patch"].encode()),
        "frozen_test_patch": sha256(record["test_patch"].encode()),
        "base_handler_excerpt": sha256(handler.encode()),
        "base_transport_excerpt": sha256(transport.encode()),
        "candidate_reference": sha256(candidate.encode()),
    }
    if actual != identities["sha256"]:
        raise ValueError("Evidence identity mismatch; no saved code was executed")
    for key in ("instance_id", "repo", "base_commit"):
        if record[key] != identities[key]:
            raise ValueError(f"Wrong {key}; no saved code was executed")
    if httpx.__version__ != identities["httpx_version"]:
        raise ValueError("Use HTTPX 0.28.1, matching the saved solver-shell probe")
    return record, handler, transport, candidate, identities


def handlers(source):
    """Load only the two exact saved handler functions with a transparent stream holder."""
    tree = ast.parse("from __future__ import annotations\n" + source)
    nodes = [tree.body[0]] + [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("make_call", "make_sync_call")
    ]
    if len(nodes) != 3:
        raise ValueError("Expected exactly two handler functions")
    env = {"ModelResponseIterator": lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<verified-handler-boundary>", "exec"), env)
    return env


def response_constructor(source):
    """Compile the exact response construction expression, excluding the HTTP dispatch."""
    tree = ast.parse(source)
    transport = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiteLLMAiohttpTransport"
    )
    method = next(
        node
        for node in transport.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_async_request"
    )
    returns = [node for node in method.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        raise ValueError("Expected one response constructor")
    return compile(ast.Expression(returns[0].value), "<verified-response-constructor>", "eval")


async def consume(handler, constructor, chunks, raw_headers, asynchronous):
    """Require exact headers and full line consumption through real HTTPX decoding."""
    stream = AsyncChunks(chunks) if asynchronous else SyncChunks(chunks)
    decoded = [
        (key.decode("utf-8", "surrogateescape"), value.decode("utf-8", "surrogateescape")) for key, value in raw_headers
    ]
    upstream = SimpleNamespace(status=200, headers=CIMultiDictProxy(CIMultiDict(decoded)))
    response = eval(
        constructor,
        {
            "httpx": httpx,
            "response": upstream,
            "request": httpx.Request("POST", "https://unused.invalid"),
            "AiohttpResponseStream": lambda response: stream,
        },
    )
    try:
        supplied_names = {name for name, _ in raw_headers}
        observed = [pair for pair in response.headers.raw if pair[0] in supplied_names]
        if observed != raw_headers:
            raise AssertionError("Header bytes or repeated header order changed")
        args = {
            "api_base": "https://unused.invalid",
            "headers": {},
            "data": "{}",
            "model": "offline",
            "messages": [],
            "logging_obj": Mock(),
        }
        if asynchronous:

            async def post(*args, **kwargs):
                return response

            result = await handler["make_call"](client=SimpleNamespace(post=post), **args)
            lines = [line async for line in result.streaming_response]
        else:
            result = handler["make_sync_call"](client=SimpleNamespace(post=lambda *args, **kwargs: response), **args)
            lines = list(result.streaming_response)
        if [line for line in lines if line] != [line for line in EXPECTED_LINES if line]:
            raise AssertionError("Stream lost, split, merged or reordered an SSE record")
        if lines != EXPECTED_LINES:
            raise AssertionError("Stream changed empty SSE separator lines")
        if stream.consumed != len(chunks):
            raise AssertionError("Stream was not fully consumed")
        return {"outcome": "pass", "chunks_consumed": stream.consumed, "lines": len(lines)}
    finally:
        if asynchronous:
            await response.aclose()
        else:
            response.close()


async def matrix(handler_source, transport_source):
    result = {}
    for name, (chunks, headers) in CASES.items():
        result[name] = {}
        for asynchronous in (False, True):
            try:
                outcome = await consume(
                    handlers(handler_source), response_constructor(transport_source), chunks, headers, asynchronous
                )
            except Exception as error:
                outcome = {"outcome": "fail", "exception": type(error).__name__, "message": str(error)}
            result[name]["async" if asynchronous else "sync"] = outcome
    return result


async def frozen_tests(handler_source, patch):
    """Reproduce the frozen mock tests without importing or running their paid example."""
    marker = "+++ b/tests/test_litellm/test_groq_streaming_encoding.py\n"
    section = patch.split(marker, 1)[1].split("\ndiff --git ", 1)[0]
    text = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))
    nodes = [
        node for node in ast.parse(text).body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in nodes:
        node.decorator_list = []
    env = {**handlers(handler_source), "Mock": Mock, "asyncio": asyncio}
    statements: list[ast.stmt] = list(nodes)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), "<verified-frozen-tests>", "exec"
        ),
        env,
    )
    results = {}
    for name in ("test_utf8_streaming_sync", "test_utf8_streaming_async", "test_various_unicode_characters"):
        try:
            result = env[name]()
            if inspect.isawaitable(result):
                await result
            results[name] = {"outcome": "pass"}
        except Exception as error:
            results[name] = {"outcome": "fail", "exception": type(error).__name__, "message": str(error)}
    return results


async def diagnose(terminal_root):
    record, base, transport, candidate, identities = load_evidence(terminal_root)
    reference = apply_patch(base, HANDLER, record["patch"])
    corrected_transport = apply_patch(transport, TRANSPORT, candidate)
    # This negative control isolates the per-chunk line-splitting defect after fixing the invalid argument.
    argument_only = reference.replace(".aiter_text(encoding='utf-8')", ".aiter_text()").replace(
        ".iter_text(encoding='utf-8')", ".iter_text()"
    )
    results = {}
    for label, handler_source, transport_source in (
        ("base", base, transport),
        ("frozen_reference", reference, transport),
        ("argument_only_negative_control", argument_only, corrected_transport),
        ("candidate", base, corrected_transport),
    ):
        results[label] = await matrix(handler_source, transport_source)
    mock_results = {
        "base": await frozen_tests(base, record["test_patch"]),
        "reference": await frozen_tests(reference, record["test_patch"]),
    }
    if not all(item["outcome"] == "pass" for case in results["candidate"].values() for item in case.values()):
        raise AssertionError("Proposed correction failed its boundary tests")
    if not all(
        item["outcome"] == "pass"
        for name in ("whole_body", "every_byte", "split_line")
        for item in results["base"][name].values()
    ):
        raise AssertionError("Unchanged handler failed the valid ASCII-header control")
    if not all(item.get("exception") == "UnicodeEncodeError" for item in results["base"]["unicode_header"].values()):
        raise AssertionError("Unchanged header-construction failure did not reproduce")
    if not all(
        item.get("exception") == "TypeError" and "encoding" in item["message"]
        for item in results["frozen_reference"]["every_byte"].values()
    ):
        raise AssertionError("Frozen reference failure did not reproduce")
    if not all(
        item.get("exception") == "AssertionError" and "SSE record" in item["message"]
        for item in results["argument_only_negative_control"]["every_byte"].values()
    ):
        raise AssertionError("Per-chunk line-splitting failure did not reproduce")
    if not all(item.get("exception") == "AttributeError" for item in mock_results["base"].values()):
        raise AssertionError("Frozen mock interface failure did not reproduce")
    if not all(item["outcome"] == "pass" for item in mock_results["reference"].values()):
        raise AssertionError("Frozen mocks did not accept the invalid reference")
    load_evidence(terminal_root)
    return {
        "variant": identities["variant"],
        "adopted": False,
        "official_reference_validity": "FAIL",
        "candidate_boundary_checks": "PASS",
        "native_grading": "NOT_RUN",
        "whole_task_correctness": "UNVERIFIED",
        "network_calls": 0,
        "model_calls": 0,
        "native_trials": 0,
        "httpx_version": httpx.__version__,
        "multidict_version": version("multidict"),
        "python_version": sys.version,
        "identities": identities,
        "checker_sha256": sha256(Path(__file__).read_bytes()),
        "inputs_unchanged": True,
        "frozen_mock_tests": mock_results,
        "response_boundary_tests": results,
        "limits": [
            "Only exact saved handler functions and the response constructor execute; the outer parser is a transparent holder.",
            "HTTPX 0.28.1 matches the saved solver shell. The grader's separate Poetry HTTPX version was not recorded.",
            "No native task, full LiteLLM suite, provider request or corrected benchmark score is established.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(diagnose(args.terminal_root))
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print("Official reference: FAIL. Proposed boundary correction: PASS. Native grading: NOT RUN.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
