# Recovering incomplete reasoning-model output

Role-specific output and thinking limits live in
[`model_settings.py`](model_settings.py). Shared model defaults live in
[`experiment_models.py`](../common/experiment_models.py); retry limits,
response-error codes and usage filenames live in
[`lm_constants.py`](../../src/gepa/lm_constants.py). Dispatch and
qualification checks consume the same retry limit. Independent test expectations
retain literal values so an accidental configuration change is detectable.

The September 23 qualification's revised Controller exhausted all 131,072 output
tokens in reasoning, repeatedly recalculating a probability sum, and returned
no final content. This was an HTTP-successful generation failure, not a network
failure or the earlier TextLimits serialization bug. Raising the output ceiling
or retrying seed 0 unchanged would not address the repeated calculation.

The repair has three parts:

1. Ask the Controller for finite nonnegative relative weights. Preserve the full
   action menu, legitimate zero weights, Controller direction and the existing
   90% normalized/10% uniform sampling mixture. Python normalizes the weights;
   the model need not repeatedly make their sum exactly one. Normalize very
   large finite weights safely if their direct sum overflows.
2. Leave final-output room using native `thinking_token_budget`. DeepSeek
   optimizer roles get 98,304 reasoning tokens within the unchanged 131,072 total
   ceiling. Qwen optimizer roles get 24,576 within 32,768. Solver settings remain
   32,768 within 65,536. Model effort, temperature, top-p, context and prompts
   outside the Controller's arithmetic instruction remain unchanged. These are
   explicit new qualification settings, not retroactive claims about old runs.
3. Apply the shared initial-plus-three retry policy described in
   [provider_retries.md](../common/provider_retries.md), retaining every attempt
   and incomplete response. Distinct seeds apply only after unusable output.

vLLM's pinned release documents that `thinking_token_budget` forces the reasoning
end marker when the budget is reached. It is an upper bound, not a demand to
consume every token, and does not guarantee a correct answer. The existing live
zero-budget boundary probes for both models verify actual transition behavior
before the expensive suite. The previously reviewed serving patch is retained.
[Source: vLLM 0.25.1 reasoning outputs](https://docs.vllm.ai/en/v0.25.1/features/reasoning_outputs/#thinking-budget-control).

AWS recommends choosing one retry point and using backoff with jitter: retries
at several layers multiply requests and can worsen overload. That is why inner
SDK retries and upper batch fallback do not each get three more attempts.
[Source: AWS Builders' Library](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/).

A repetition penalty or a text-based loop detector would introduce additional
decoding changes and could misclassify legitimate structured repetition. Neither
is introduced in this repair. The native boundary and deterministic arithmetic
removal target the observed failure more directly. Live qualification must still
establish that the repaired model returns usable responses.

The fixed train-only comparison retains its original proposal/transfer/synthetic
partitions and every no-op or rejection. Both strategy revisions share the new
request runtime and token limits, with hashes recorded explicitly, so a different
retry allowance cannot explain a comparison result. New traces and baselines are
computed; no earlier failed request or checkpoint is reused. Technical success
and the preregistered usefulness signal are reviewed separately. Neither releases
the production design hold automatically.

The isolated proposal runtime hashes and shares `gepa.lm_constants`,
`gepa.strategies.forest_constants`, and the worker driver's benchmark settings
alongside `gepa.lm` and `provider_retries`. Each arm's strategy remains source-pinned.
