# Provider request retries

HotPotQA and Terminal-Bench use the same policy for every model role: at most
**three attempts per provider request**, with one-second and two-second backoff
before the second and third attempts. A successful response ends the request.

Only connection errors, transport timeouts, and HTTP 408, 429, 500, 502, 503, or
504 qualify. Authentication, invalid requests, other permanent errors, and
local programming errors stop immediately. Cancellation propagates. An explicit
numeric request timeout covers all attempts and backoff together; the existing
benchmark task deadline also continues to apply.

The shared wrapper owns the retry count. Both LiteLLM `num_retries` and the
underlying SDK `max_retries` are zero to prevent multiplication. Their zero
values in run contracts describe the disabled inner layers, not the experiment's
three-attempt policy. Harbor's two additional retry decorators are bypassed.
Exhausted requests also stop Harbor's summarization and context recovery rather
than silently continuing through another model call. Bad-request parameter
fallback cannot silently change request fields and try again.

This policy covers solver, proposer/editor, Controller, Manifestor, task-agent,
and summarization requests. Successful members of a batch remain completed;
only a failed request repeats. It does not retry tasks or low-scoring answers.
FOREST's unlimited tool-error correction and the existing context/output repair
protocols are separate. Changes to prompts during those protocols constitute
new model requests; the transport policy does not select answers or candidates.

Each run writes `provider-attempts.jsonl`; Harbor uses a separate file in each
trial's agent directory. Records contain a request ID, attempt number, model,
role, timing, error type/status, retry decision, and available token/cost counts.
Missing counts stay null. Prompts, response text, exception messages, headers,
and credentials are excluded. Direct diagnostic calls without a file destination
emit the same records to the process log.

TB also writes these attempts to its existing `token-usage.jsonl` files, so
the existing usage report includes failed attempts as well as successful calls.
These files describe the same requests and must not be added together. Journal
replay issues no provider request and adds no physical-attempt record. Costs
reported only in provider-attempt artifacts remain separate from scored metric
calls and cannot be inferred as zero when unavailable.

The policy is recorded in HotPotQA run contract version 22, TB run contract
version 23, and TB training-pilot configuration version 5. Missing or changed
policies cannot resume or enter the final TB comparison. Existing runs require
fresh directories rather than relabeling their request policy.
