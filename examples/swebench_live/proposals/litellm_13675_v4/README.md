# Proposed normal-router clock input

This v4 proposal adds a deterministic clock for one original preservation test:
`tests/test_litellm/proxy/hooks/test_parallel_request_limiter.py::test_normal_router_call`.
It does not change the default SWE runner or adopt a new evaluation protocol.

The test records a minute bucket, waits, completes a native Router mock response,
then checks the earlier bucket. The frozen limiter computes a new minute during
cleanup. Crossing a minute can leave the test's earlier counter at one.
The original body, assertions, waits, reference patch, and limiter source stay unchanged.

Native adoption job 17006971 failed this assertion with 6/6 F2P and 2,585/2,586 P2P passes.
Its setup logs show 00:15:59; the response logs show 00:16:00.
The callback bucket and completion time were not recorded. Rollover is strongly supported,
but callback latency or a swallowed exception is not excluded for that historical run.

V4 pins both the test and limiter clocks for this exact test. It records their use
and restores both clocks afterward. It verifies the frozen module hashes first.
The previous fourteen fixtures run through the byte-identical, hash-pinned v3 plugin.
All v3 files stay unchanged. The generated patch includes that pinned dependency.

The proposed evaluator retains all 2,580 original P2P IDs and six added sync controls.
Its six async streaming tests, repaired reference, and original problem remain identical to v3.
Only the additional clock input and provenance change. This is a new scientific evaluation
variant, not an equivalent label for v3 or the original official benchmark.

The manifest includes a new evaluator digest and an `evaluation_contract` for explicit
runtime/resume binding. Existing v3 checkpoints must not resume under v4.
An adopter must bind that contract in the actual worker and grader. This builder alone
does not make a runtime use the variant.

Prepare a separate packet with:

```sh
uv run python -m examples.swebench_live.proposals.litellm_13675_v4.prepare \
  --record /path/to/unchanged/evaluator-record.json \
  --output /path/to/new/proposed-packet
```

Local regressions execute the exact frozen test and callback bodies with labeled cache,
auth, and Router-delivery stubs. They do not prove native transport or callback scheduling.
The native gate must reproduce forced rollover with the real Router, pass all fifteen
fixtures, restore every clock/cache, and pass the full reference grading gate.
No paid model call is needed for those checks. An earlier native reference pass is not
a model-generated solution or an original benchmark success.
