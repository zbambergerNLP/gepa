# JEV request sizing

The OpenRouter controller plans evidence groups before it makes a provider request.
Each group contains at most three complete records and must fit a fixed client
byte budget. Records stay in their original order. Every request retains the
complete component, section descriptions, instructions, and choice criteria.
The controller averages group probabilities with record-count weights, then
applies exploration and samples once.

## Why the policy changed

A SWE training pilot supplied three records in a single JEV request. Its logical
JSON payload measured 1,058,398 UTF-8 bytes. OpenRouter returned HTTP 400 with
`max_tokens_exceeded`. The old policy grouped only batches larger than three
records, so it sent this request unchanged.

The new policy groups that saved input into three requests, each containing one
complete record. Their logical JSON sizes are 314,021, 324,868, and 442,431 bytes.
This replay uses a mocked transport. It proves payload preservation and bounded
dispatch, not live provider acceptance or improved benchmark scores.

## Byte budgets and their limits

| Transport | Maximum logical JSON bytes |
| --- | ---: |
| OpenRouter | 524,288 |

Direct TypeSafe keeps its existing count-only grouping. This repair targets the
OpenRouter transport used by the failed pilot. It does not establish an input
admission policy for direct TypeSafe's smaller default context.

These are **client safety budgets, not token counts or guaranteed context
bounds**. The client measures `json.dumps(request, sort_keys=True,
ensure_ascii=False, allow_nan=False).encode("utf-8")`. This includes the model,
state, all question instructions, and criteria. It does not measure compressed
HTTP bytes or provider framing. No other model's tokenizer is substituted.

[TypeSafe documents](https://docs.typesafe.ai/models#context-length) separate
32k per-question and 64k per-request default token limits. Its experimental
long-context mode raises both. [OpenRouter documents a 262,144-token context](https://openrouter.ai/docs/guides/community/jev)
and automatic long-context mode for larger requests on its route. The controller
keeps the same pinned models and transport configuration; it adds no opt-in header.

The client budget uses two bytes per advertised OpenRouter context token. This is
a deliberately restrictive estimate for ordinary text and saved SWE traces,
not a documented tokenizer bound. Token-dense text can still exceed a provider
limit below these budgets. The public SDK provides usage after a request; it
does not provide an exact preflight token counter. A provider HTTP 400 remains
terminal and never causes regrouping, truncation, fallback, or a repeated call.

## Failure and replay behavior

The controller validates the entire plan before sending its first group. If a
single complete record cannot fit, it raises `JevInputSizeError` with measured
bytes and the client limit. It sends no group for that decision. Free text,
non-array evidence, and joint-component decisions must fit a single request.
The controller never splits a record, removes fields, summarizes evidence, or
shortens instructions to force a fit.

The same size check applies to OpenRouter calls and manual response recovery.
A response-correction retry must also fit in full. An
oversized correction stops after recording the prior physical response and
its available usage. It does not send another request.

The run contract records `jev_joint_action_section_v6_size_bounded_evidence_pool`,
the size metric, budget, grouping, and aggregation. Journal identities bind
groups to that contract and the complete original decision. Completed groups
replay without provider calls. Started or failed groups without a committed
response remain blocked for inspection. Older contracts cannot resume under
this policy.

## Scientific scope

Grouping changes which records JEV considers together. Preserving every record
does not make separate judgments equivalent to one joint judgment. This is a
scientific protocol change that needs approval before deployment or a new run.
It does not change the solver minibatch, Manifestor evidence, models, datasets,
acceptance rule, or trial query cap. Existing frozen runs remain unchanged.
