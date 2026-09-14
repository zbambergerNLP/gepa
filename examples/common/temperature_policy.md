# Provider sampling and reasoning policy

Updated during consolidation on 2026-09-11 after the user approved replacing
DeepSeek V4 Flash with **DeepSeek V4.1 Flash** in HotPotQA and Terminal-Bench 2.1.
Use applicable task-specific model-author guidance; otherwise use the exact
checkpoint's general thinking-mode recommendation. A role name alone does not
justify different sampling.

## Current settings

| Work | Qwen3.8-27B | DeepSeek-V4.1-Flash |
| --- | --- | --- |
| HotPotQA summaries, retrieval queries, factual answers | temperature 1.0 / top-p 0.95 / xhigh | temperature 1.0 / top-p 0.95 / effort 100 |
| TB2.1 terminal execution | Same | Same |
| GEPA prompt/skill rewriting and stateless selection | Same | Same |
| FOREST Controller, Manifestor, tool-using editor | Same | Same |

These settings apply across scopes, methods, budgets, and final evaluation.
Qwen also retains top-k 20. Both models use explicit thinking mode. Random
Controller selection makes no model call.

The old V4 policy used top-p 1.0 for non-agentic roles and 0.95 for tool use.
V4.1's card reports top-p 0.95 for its instruct evaluations, including factual
and reasoning benchmarks, and for agentic evaluations. We use that consistent
published setting for all current roles. Extending it to optimizer roles is our
application of provider guidance; the provider did not test these exact roles.
The shared decoding helper still accepts the role classification argument.

## Sources

- [Pinned Qwen card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md):
  thinking-mode temperature 1.0, top-p 0.95, top-k 20, and default effort xhigh.
- [DeepSeek V4.1 card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash):
  instruct and agentic evaluation settings are temperature 1.0, top-p 0.95,
  and numeric reasoning effort 100. The checkpoint is pinned at
  `dba1be0a40aa45a94ad051997016db3960a90277`.
- [Pinned vLLM V4.1 tokenizer](https://github.com/vllm-project/vllm/blob/e77daef89e18e08321ae7b8b24827eedd5fe8673/vllm/tokenizers/deepseek_v41.py)
  and [encoder](https://github.com/vllm-project/vllm/blob/e77daef89e18e08321ae7b8b24827eedd5fe8673/vllm/tokenizers/deepseek_v41_encoding.py):
  accept numeric effort 1–100 through `chat_template_kwargs`, with `max` mapped
  to 100. Sending 100 explicitly avoids ambiguity between named effort aliases.

Qwen sends `enable_thinking=true` and `reasoning_effort=xhigh` through
`extra_body.chat_template_kwargs`. DeepSeek sends `thinking=true` and numeric
`reasoning_effort=100`. V4.1 uses its native Python prompt encoding, not a Jinja
template. Both the request fields and checkpoint identity are recorded for resume.
Hosted DeepSeek API behavior does not determine this local-vLLM experiment.

## Serving and token budgets

Qwen uses the committed vLLM 0.25.1 lock. V4.1 uses Zach's separate hash-locked
vLLM commit wheel `e77daef89e18e08321ae7b8b24827eedd5fe8673`, package version
`0.1.1.dev5+ge77daef89`, Torch 2.13, and prebuilt offline FlashInfer kernels.
Production checks both installed version and checkpoint architecture support.

The Della server context is **262,144 tokens for each model**, following the
integrated serving profiles. This is a deployment choice, not the models'
advertised maximum or a claim to reproduce their largest-context evaluations.
DeepSeek's card recommends 1M context and at least 256K output; our approved
output budgets are deliberately smaller:

- HotPotQA Qwen: **16,384 output tokens per solver call**, **32,768 per optimizer call**.
- HotPotQA DeepSeek: **32,768 output tokens per solver call**, **131,072 per optimizer call**.
- TB2.1: **32,768 output tokens per call**, every role, including reasoning.

Natural stopping remains enabled, with no minimum generation length. Review
usage and cutoffs on training tasks before freezing settings. Never raise caps
based on validation/test outcomes. The V4.1 switch does not automatically expand
output limits. Character limits remain independently configurable and unlimited
by default, subject to these model limits.

HotPotQA's logical request deadline is 3,600 seconds, shared by the fixed bounded
provider-attempt policy. This is an operational timeout, not an output budget.

## Comparability and recovery

Provider effort labels do not imply equal compute across models. Preserve each
model's configuration across methods, budgets, scopes, and final evaluation.
Identical Controller/editor sampling can share a client while retaining role
accounting; distinct sampling profiles still have separate journal identities.

Run contracts record model, revision, reasoning, decoding, and runtime. V4,
Manifestor-temperature-0.0, missing-thinking, and differently capped checkpoints
cannot silently resume under the new experiment. The AutoSaddler paper's model
choice is separate from our homogeneous local-model comparison; this policy
does not claim exact reproduction of its endpoint settings.

See [Terminal-Bench's pilot protocol](../../src/gepa/adapters/terminal_bench_adapter/README.md#output-budget-review)
and [the Della runbook](../hotpotqa/DELLA_CAMPAIGN.md).
