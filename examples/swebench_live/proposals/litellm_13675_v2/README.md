# Proposed LiteLLM response-boundary correction

Status: **PROPOSED, NOT ADOPTED**. Variant: `litellm-13675-httpx-boundary-v2`.

This packet repairs Unicode-header handling and stream ownership. It supplies
offline regression checks. It does not alter the frozen SWE-bench Live record,
official tests, images, profiles, historical scores, or campaign runner.

The task is `BerriAI__litellm-13675`, at LiteLLM commit
`1b2ec16eee5070f117d401420b8cf1ce44de9e60`.
`identity.json` pins the original evaluator record, reference patch, test patch,
saved source excerpts, and proposed reference patch.

## Why the frozen reference is not valid evidence

The frozen tests replace HTTPX responses with a mock that omits `iter_lines`
and `aiter_lines`. They also stop before consuming the returned stream.
They reject the unchanged valid line-decoding boundary and accept the frozen
reference's unsupported `encoding` argument.

HTTPX 0.28.1 accepts `chunk_size` on `iter_text` and `aiter_text`, not `encoding`.
Its native line iterators buffer partial lines across chunks. Removing the
unsupported argument alone still leaves the reference's per-chunk splitting
defect. The negative control in `check.py` reproduces that separate defect.
[HTTPX 0.28.1 API](https://github.com/encode/httpx/blob/0.28.1/docs/api.md).

The reported ASCII error occurs earlier, when the aiohttp transport passes
Unicode header strings into the HTTPX response constructor. The proposed
`reference.patch` restores header bytes with UTF-8 and `surrogateescape`.
It preserves repeated header values and their order. It leaves the handler's
native `iter_lines` and `aiter_lines` calls unchanged.
This reverses aiohttp's documented header decoding.
[aiohttp header decoding](https://docs.aiohttp.org/en/v3.11.18/client_advanced.html#response-headers-and-cookies).

This proposed reference is smaller than the saved solver's header helper.
It makes no claim that the saved solver patch resolves the entire task.

## Preserve native stream ownership

The frozen transport enters an aiohttp response context and wraps it in
`AiohttpResponseStream`. That class closes the context in its `aclose` method.
Passing this object through `httpx.Response(content=...)` loses that callback.
HTTPX wraps async content in `AsyncIteratorByteStream`, whose inherited close
method does nothing. Its public client adds another wrapper that delegates to
the same ineffective close method.

The v2 patch uses `stream=AiohttpResponseStream(response)`. HTTPX retains the
native stream and forwards closure. The header repair and native line iterators
remain unchanged. This repairs an inherited transport defect. A fully consumed
aiohttp response can also release its connection at EOF, so a missing context
exit alone does not prove a real connection leak in every completed request.
[HTTPX 0.28.1 streaming lifecycle](https://github.com/encode/httpx/blob/0.28.1/docs/async.md).

`test_swebench_litellm_stream_ownership.py` exercises the public
`AsyncClient.stream` context with HTTPX 0.28.1. It uses the exact archived response
constructor and native stream class. Only the upstream response and exception
mapping are simulated. Both full consumption and early closure must close the
upstream context exactly once. No socket opens.

The tracked `native_probe.py` also preserves separate `parsing` and `cleanup`
results. The previous native probe raised a closure assertion in `finally`,
which replaced the original parser failure in its report. The v2 diagnostic
retains both failures. A case passes only when both stages pass. Cancellation
still runs cleanup and propagates. The strict closure assertion remains.

## Prior native evidence and remaining hold

The v1 diagnostic ran in the pinned image under job 16954571. The image used
HTTPX 0.28.1, aiohttp 3.10.11, and LiteLLM 1.75.7.

| Arm | Required F2P passed | Required P2P passed |
| --- | ---: | ---: |
| Unmodified base | 0 / 3 | 2,568 / 2,580 |
| Frozen reference | 3 / 3 | 2,568 / 2,580 |
| Header-only v1 proposal | 0 / 3 | 2,567 / 2,580 |

The v1 proposal passed six sync boundary cases. All six async results reported
the closure assertion. Those results do not establish parser success because
the assertion could mask an earlier error. The shared twelve P2P failures and
the candidate-only `test_failure_call_hook` failure remain unresolved. Their
presence does not establish that the header patch caused the limiter failure.

The v2 proposal did not run in the native image or official suite. The one-line
ownership repair does not establish complete validity. Preserve every previous
result and keep the SWE experiment gate closed until the evidence supports it.

## Reproduce the offline regression

Use the saved attempt10 terminal archive after its normal hash verification.
Pass its extracted root, which contains `files/swebench_live/native-trials/`.

```bash
uv run --no-project --with httpx==0.28.1 --with multidict==7.1.0 python \
  examples/swebench_live/proposals/litellm_13675_v2/check.py \
  --terminal-root /absolute/path/to/attempt10-terminal-artifacts \
  --output /absolute/path/to/new-boundary-report.json
```

The report path must not already exist. The command exits **1** because the
official reference fails. A passing candidate never converts that failure
into a readiness pass. Changed input hashes or failed controls stop the check.
The command does not fetch data, open a socket, or run a task container.

The check applies patches only to disposable source copies. It executes the
two exact saved handler functions and the exact response-construction expression.
It excludes the HTTP dispatch. It uses actual HTTPX responses, sync and async
byte streams, and aiohttp's `CIMultiDictProxy` header representation.

Each source variant receives six cases in both sync and async form:

- A complete UTF-8 body, including multiple SSE records and a final partial line.
- Single-byte chunks, splitting multibyte UTF-8, CRLF, and SSE records.
- A split inside an SSE record.
- The `6.083µs` header that triggers the saved failure.
- Repeated headers, with exact byte and order checks.
- Non-UTF-8 header bytes, recovered through `surrogateescape`.

Every case consumes the stream. It checks exact lines and consumed chunk counts.
The candidate passes 12 cases. The unchanged transport fails the Unicode header
cases. The frozen reference fails real stream consumption. The check separately
reproduces all three frozen mock tests on the unchanged and reference handlers.

Run the focused ownership and diagnostic regressions in an existing environment
with HTTPX 0.28.1, multidict, aiohttp, and pytest:

```bash
SWE_LITELLM_TERMINAL_ROOT=/absolute/path/to/attempt10-terminal-artifacts \
  PYTHONPATH=.:src uv run --offline --no-project --python /absolute/path/to/python \
  python -B -m pytest -q tests/test_swebench_litellm_stream_ownership.py
```

The two ownership cases require the preserved archive. They skip with an explicit
reason when its path is absent. Diagnostic reporting tests need no archive.
The existing environment remains unchanged. The native diagnostic must run only
inside a separately approved task image; this packet does not start it.

## Adoption and coverage limits

These checks propose a corrected boundary test, not a new official task score.
The outer `ModelResponseIterator` is a transparent holder. The complete LiteLLM
parser, task container, provider, and official P2P suite do not run here.
HTTPX 0.28.1 matches the saved solver-shell probe and the later v1 native check.
The focused ownership tests do not import the full pinned LiteLLM package or run
its HTTP session. The new native diagnostic remains unexecuted in that image.

The six original LiteLLM trial scores remain zero. The five empty patches remain
model outcomes. The saved nonempty patch still has unresolved full-suite coverage.
The v1 native results above remain separate from this proposed v2 correction.

Before adopting a corrected evaluation variant, obtain specific approval for
its record and grading changes. Validate the full unmodified and corrected
reference behavior in the pinned native environment. Keep the full P2P criteria.
Resolve the recorded environment failures before attributing them to a patch.
Use a new manifest, contract, baseline, and output directory for any adopted
variant. Never relabel historical results as corrected or official successes.
