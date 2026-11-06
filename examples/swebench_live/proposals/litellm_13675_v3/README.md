# Proposed native-streaming evaluation

Status: **PROPOSED, NOT ADOPTED**.
Variant: `litellm-13675-native-streaming-evaluation-v3`.

The frozen LiteLLM task uses incomplete response mocks. They omit HTTPX line
iterators and never consume the returned stream. They reject valid stream
handling and accept unsupported `encoding` arguments in the frozen reference.

This packet makes the proposed correction concrete. It generates a separate
record, test patch, reference patch, and identity manifest. It changes no
official record, historical test file, score, campaign runner, or checkpoint.

## Proposed test and reference changes

The generated streaming tests run the exact pinned native probe from the v2 packet.
That probe uses the installed LiteLLM parser, actual HTTPX byte streams, and
the native transport's response construction. Only the upstream HTTP response
and logging are simulated. No provider call or network connection occurs.

Each sync and async case consumes the complete stream. It checks decoded text,
finish state, byte-chunk consumption, header bytes, repeated-header order,
and response cleanup. Cases cover whole bodies, byte boundaries, split SSE
records, Unicode headers, repeated headers, and opaque header bytes.

The proposed record uses the v2 header and stream-ownership reference patch.
It replaces the three invalid F2P mock checks with six async native checks.
It preserves all **2,580 original P2P requirements in their original order**
and appends six sync controls. The proposed totals are six F2P and 2,586 P2P.

This is a scientific grading change. It is not an equivalent implementation of
the old tests. The added controls change the test population and scoring basis.
The old three mock checks remain preserved in the original test patch.

## Explicit offline inputs

The native trace-only diagnostic, job `16983378`, confirms DNS failures in the
four previously unexplained shared assertions. The earlier logs identify the
other eight. The required offline network namespace stays enabled.

The proposed plugin supplies inputs only for those twelve exact P2P test IDs:

- Nine image-dependent tests receive a pinned synthetic 8-by-6 JPEG at the HTTP
  input boundary. The JPEG format stays distinct from the requested PNG MIME
  override, so the original format assertion still checks meaningful behavior.
- Two Gemini missing-key tests receive a declared synthetic HTTP 400 response.
  This tests the real client error path without reaching a provider. The fixture
  rejects an unexpected method, host, model path, or supplied key.
- The IPv4 test uses its actual HTTPX transport against a private loopback HTTP
  server. It also requires the real transport to bind `0.0.0.0`. It does not use
  a mock transport or a fabricated successful connection.

All original P2P source and assertion bytes remain unchanged. The plugin rejects
unexpected calls rather than returning success for arbitrary requests. Its audit
records omit request headers, bodies, and credential values. Fixture scope is
per test; all other tests retain their original inputs and transport behavior.

Each selected image test receives a fresh instance of the native image cache.
The fixture restores the prior cache after the test. This forces a recorded input
read and prevents synthetic bytes from leaking through the cache to later tests.
The frozen image-conversion source hash is checked before cache isolation.

These are proposed environment fixtures, not recordings of real public servers.
They change external inputs and therefore form part of the proposed evaluation
protocol. They do not establish current provider behavior or public-site access.
Load the plugin only for the separately identified diagnostic variant.

## Evidence and remaining gate

Native validation job `16963168` ran the same v2 probe in the frozen image.
It used LiteLLM 1.75.7, HTTPX 0.28.1, aiohttp 3.10.11, and pytest 7.4.4.

| Arm | Native cases | Original F2P | Original P2P |
| --- | ---: | ---: | ---: |
| Base | 6/12 | 0/3 | 2,568/2,580 |
| Frozen reference | 0/12 | 3/3 | 2,567/2,580 |
| Proposed v2 reference | 12/12 | 0/3 | 2,568/2,580 |

The candidate has no P2P failure absent from both other arms. All twelve shared
failures show DNS-dependent inputs inside the required offline namespace.
One extra reference-only limiter assertion also failed. Its cause remains
unproven. None is waived or removed by this proposal.

The original official gate remains **FAIL**. The generated pytest wrapper and
offline inputs need a successful native validation before adoption.
Passing the streaming boundary alone does not prove
whole-task correctness or release a paid SWE experiment.

The first v3 native packet, job `16984079`, exposed an audit defect. Four image
test bodies passed, but their teardown audits failed after shared cache hits.
The first packet and its failures remain preserved. The corrected fixture keeps
the required input-use audit and isolates the native cache for each selected test.
The diagnostic candidate is a reference control, not a model-generated solution.

The local cache regressions execute the unchanged frozen conversion functions
with real HTTPX clients and an explicit cache-class stub. They fail on the first
proposal and pass with cache isolation. Native validation must also check the
actual cache class and all twelve original input-dependent tests.
Set `SWE_LITELLM_TERMINAL_ROOT` to the verified attempt10 archive and
`SWE_LITELLM_IMAGE_SOURCE` to the verified frozen `image_handling.py` for local tests.

The saved comparison is under the pilot root at
`checks/20261106-pr102-pr103-merge-validation/native-review-tools/capture-correction01/terminal-comparison/`.
Archive SHA256:
`e8c31383d12dd2235651ba04714a79fb0f0760b2455899f008edb6b0214099a1`.

## Prepare a separate review packet

Use the original, hash-verified attempt10 evaluator record. The output directory
must not exist. The command writes proposed inputs only; it does not run tests,
submit a job, update a profile, or call a provider.

```bash
uv run --offline --no-project --python /absolute/path/to/python python \
  examples/swebench_live/proposals/litellm_13675_v3/prepare.py \
  --record /absolute/path/to/original/evaluator-record.json \
  --output /absolute/path/to/new-proposal
```

`identity.json` pins the original record, original patches, proposed reference,
and complete native probe. Any input drift fails before output creation.
The generated manifest records both test populations and every input hash.
The native diagnostic sets `PYTHONPATH=/testbed/tests/test_litellm` and
`PYTEST_PLUGINS=_proposed_offline_inputs` only in the fresh task filesystem.
Set `SWE_VARIANT_AUDIT_DIR` to a fresh directory for complete fixture-use evidence.

The native comparison must use separate fresh base, frozen-reference, and
proposed-reference filesystems. Run the proposed pytest checks and every
original P2P requirement. Keep all failures, missing tests, and infrastructure
errors distinct. Preserve the original official comparison separately.

Specific approval is required before adopting a corrected record or grading
protocol. An adopted variant needs a new manifest, runtime/resume identity,
baseline, and output directory. Do not resume old checkpoints or relabel old
scores. This preparation tool does not implement campaign adoption.
