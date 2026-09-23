# Provider request retries

HotPotQA and Terminal-Bench use one shared policy for model requests: an initial
attempt plus **three retries**, at most four physical attempts total. Sync, async,
batch, solver, Controller, Manifestor, Editor, proposer, and summarization calls
use this boundary. Raw HotPotQA serving/boundary probes use the same policy.
Unmarked library calls retain their caller's policy.

Retry connection errors, transport timeouts, HTTP 408/429/500/502/503/504, missing
choices, empty final output without native tool calls, and `finish_reason=length`.
Native tool calls with no prose are valid. A cutoff is rejected before partial
tool calls can execute. Refusals, valid no-ops, wrong task answers, rejected
candidates, and invalid semantic edit batches are not rerolled for a better score.
Authentication, invalid requests, other permanent errors, and programming errors
stop immediately; retrying cannot repair them. Cancellation propagates.

Transport and output failures share the four-attempt limit and the original
request deadline. Full-jitter backoff draws independently from [0,1], [0,2], and
[0,4] seconds without consuming optimizer RNG. Transport retries keep the seed.
After incomplete output, an explicit integer generation seed advances by one
(modulo 2**32) to avoid replaying the same deterministic failure. Prompts, decoding
parameters, model identities, and token caps otherwise stay fixed.

Both LiteLLM `num_retries` and the SDK `max_retries` are zero. The shared wrapper
owns retries; upper reflection/batch fallback cannot restart an exhausted budget.
Harbor's nested retry decorators are also bypassed. Successful batch members and
response-journal entries remain completed. Journal replay makes no provider call.
Existing explicit format/context protocols are separate logical requests, not an
excuse to rerun a failed provider request or select favorable candidates.

Each physical attempt is appended to `provider-attempts.jsonl`. Records separate
transport success from usable-output success and include request ID, attempt,
role, model, effective seed, retry decision, finish reason, elapsed time, and
available usage/cost. Unknown usage remains null. Incomplete response artifacts
preserve the request, reasoning, final output and native calls with permissions
0600 and without credentials. A later success does not erase earlier paid work.
The ledger itself omits prompts, responses, exception messages, and credentials.

Some workflows duplicate the same attempts in `token-usage.jsonl`; never add the
two files together. Logical metric evaluations and all physical attempts must be
reported separately. Run contracts include the complete versioned retry policy;
changed policies require fresh reviewed identities, not relabeled checkpoints.

For the generalization qualification, both source-pinned strategy arms use the
revised `gepa.lm` and provider dispatch modules and identical resolved model
kwargs. A dedicated bootstrap imports only these two modules from the reviewed
runtime, verifies their hashes, and records that mixed identity explicitly. All
Controller/Manifestor/Editor strategy code still comes from the declared arm.
Historical source directories remain untouched.
