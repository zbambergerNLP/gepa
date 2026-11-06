# Proposed LiteLLM response-boundary correction

Status: **PROPOSED, NOT ADOPTED**. Variant: `litellm-13675-httpx-boundary-v1`.

This packet repairs a confirmed Unicode-header boundary and supplies concrete,
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

## Reproduce the offline regression

Use the saved attempt10 terminal archive after its normal hash verification.
Pass its extracted root, which contains `files/swebench_live/native-trials/`.

```bash
uv run --no-project --with httpx==0.28.1 --with multidict==7.1.0 python \
  examples/swebench_live/proposals/litellm_13675_v1/check.py \
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

## Adoption and coverage limits

These checks propose a corrected boundary test, not a new official task score.
The outer `ModelResponseIterator` is a transparent holder. The complete LiteLLM
parser, task container, provider, and official P2P suite do not run here.
HTTPX 0.28.1 matches the saved solver-shell probe. The grader's separate Poetry
HTTPX version was not recorded. The aiohttp version is not inferred from the
documentation version above.

The six original LiteLLM trial scores remain zero. The five empty patches remain
model outcomes. The saved nonempty patch still has unresolved full-suite coverage.
The native smoke test covers Patroni, not this LiteLLM task.

Before adopting a corrected evaluation variant, obtain specific approval for
its record and grading changes. Validate the full unmodified and corrected
reference behavior in the pinned native environment. Keep the full P2P criteria.
Resolve the recorded environment failures before attributing them to a patch.
Use a new manifest, contract, baseline, and output directory for any adopted
variant. Never relabel historical results as corrected or official successes.
