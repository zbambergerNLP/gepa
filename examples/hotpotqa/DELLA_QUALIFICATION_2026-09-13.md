# Della interactive measurements: September 13, 2026

DeepSeek-V4.1-Flash served successfully on four H200s. A real FOREST editor request needed **46,901 output tokens** before returning its first tool call. The same request hit the production 16,384-token cap, and its corrective next turn also hit that cap. This diagnostic does **not** qualify the complete optimizer campaign.

No production decoding, token caps, benchmark data, or optimizer settings were changed. The larger output allowance was confined to a separate measurement requested by the user.

## Allocation and source

- Job: `13806836`, node `della-i20g3`; one interactive allocation, four H200s, 32 CPUs, 512G host memory, requested time 55 minutes.
- Runtime source: `246b9cba66331e211d8ed7fca812a81969d347fd`; the complete source manifest was verified before execution. The frozen remote source was not edited.
- The allocation started at 06:19:48 UTC. Slurm still reported it running after the requested time; it was explicitly canceled at 07:16:56 UTC. Accounting records 57:08 elapsed and final state `CANCELLED`. No additional allocation or batch job was submitted. The allocation-follow-up automation was paused.
- Only training examples were used. No held-out validation or test evaluation was run. The tiny optimizer diagnostic reused three training examples as its local evaluation set; that is not the production validation split.

## Natural completion of the editor turn

The Controller and Manifestor responses from the interrupted optimizer check were replayed locally to reconstruct its first editor request. The Controller, Manifestor, and editor request hashes matched their original response-journal records. No new model inference was used to reconstruct the request.

The editor then received the same model request and decoding settings with its fixed output cap removed for this diagnostic. Its remaining context supplied an output allowance of 256,094 tokens. This was a fresh replay of the first editor request, without the subsequent cutoff-recovery history.

| Measurement | Observed value |
| --- | ---: |
| Provider-reported input tokens | 6,030 |
| Output tokens | **46,901** |
| Reasoning tokens, included in output | **46,285** |
| Other output tokens, including tool syntax | 616 |
| Total context consumed | 52,931 |
| Request elapsed time | **397.734 seconds / 6m 38s** |
| Output tokens / request elapsed time | **117.92 tokens/s** |
| Decode throughput from server token-latency counters | **118.09 tokens/s** |
| Finish reason | `tool_calls` |
| Returned action | One `INSERT_TEXT` call with JSON arguments |

The tokenizer endpoint counted 6,050 tokens for its tokenizer-facing request; the actual completion reported 6,030. The larger count was used to reserve output space conservatively. The original request hash is `e812dc1b76b7c533c55d19bf5b427426969023c8b27d61191a93aba7891c8913`.

For this observed turn, 16,384 and 32,768 tokens were insufficient; 65,536 would have covered it. This measures one editor turn ending in a tool call, not the complete edit-and-submit sequence or the largest requirement across the dataset. A universal cap remains a decision to make after more measurements.

Before the larger-cap diagnostic, the editor's first two turns each returned `finish_reason=length` with all 16,384 output tokens attributed to reasoning. The editor continued after the first cutoff, using the existing corrective protocol feedback. The optimizer was interrupted before a complete proposal and reevaluation; no complete-cycle marker was produced.

## HotPotQA execution and runtime estimates

The existing four-component HotPotQA program completed two passes over the same three training questions: six full evaluations and 24 solver calls. The seed scored 2/3 exact match on this tiny set. Metric improvement was not a qualification requirement.

| Three-question pass | Solver calls | Output tokens | Model-request span | Span / completed question |
| --- | ---: | ---: | ---: | ---: |
| Initial evaluation | 12 | 29,650 | 252.119 s | 84.040 s |
| Reflection minibatch | 12 | 12,481 | 107.926 s | 35.975 s |

Spans run from the earliest recorded request start to the final request completion in each pass, including intervening queueing and pipeline gaps. They exclude initial dataset/client setup. The second pass briefly overlapped the separate tool-compatibility diagnostic. These are two passes over **three unique questions**, not six independent examples or a full calibration.

The average observed cost is about 60.0 seconds per full evaluation under the current single-active-sequence server. The following is conditional planning arithmetic if that tiny sample represented the eventual workload; it is not a reliable production completion forecast or a confidence interval.

| Ablation | Metric-call budget | Estimated evaluation time | Range using the two observed pass speeds |
| --- | ---: | ---: | ---: |
| Vanilla GEPA | 6,871 | 114.5 h | 68.7–160.4 h |
| FOREST / `react_v2` | 6,871 | 114.5 h | 68.7–160.4 h |
| FOREST random Controller / `react_v2_random` | 6,871 | 114.5 h | 68.7–160.4 h |
| Action-only / `action` | 6,871 | 114.5 h | 68.7–160.4 h |
| Vanilla GEPA, double budget | 13,742 | 229.1 h | 137.3–320.8 h |
| FOREST, double budget | 13,742 | 229.1 h | 137.3–320.8 h |

These estimates cover full question evaluations. The metric budget already counts training and optimizer-validation evaluations; validation must not be added a second time. Add about **5.0 hours per ablation** for its 300-question held-out test at the same assumed speed, and **5.0 hours once per model** for the shared starting baseline. Those held-out evaluations have not been executed.

Optimizer-model calls and startup are additional. Observed FOREST calls took 116.237 seconds for the Controller, 74.014 seconds for the Manifestor, and 397.734 seconds for the fresh larger-cap editor turn. Their sum is about 9.8 minutes, before a subsequent finish turn. This is a partial proposal measurement, not a completed iteration. Proposal counts depend on search behavior, acceptance, and validation work; vanilla, random-Controller, and action-only proposal overheads were not measured. A full ablation estimate therefore remains:

`startup + evaluation budget × seconds per evaluation + proposal-model time + held-out test time`

The serving profile remained TP4/EP4/DP1, one active sequence, one API server, native mixed FP4/FP8 weights, FP8 KV, Engram CPU offload, context 262,144, temperature 1.0, top-p 0.95, and reasoning effort 100. No throughput-tuning restart was performed.

## Memory, startup, and completed checks

- DeepSeek reported 71.46 GiB for model loading per GPU. Its slowest worker took 493.6 seconds for model loading, followed by 142.75 seconds for engine initialization, profiling, cache setup, and warmup. Checkpoint and environment verification were separate startup costs.
- Peak sampled GPU use was **137,689 MiB / 134.46 GiB per GPU**, out of 143,771 MiB visible capacity. This includes reserved cache and runtime allocations; it is not a minimum model-memory measurement. Long-context capacity was not fully exercised.
- Host cgroup memory peaked at **511.97 GiB**, with substantial file-cache pressure and rereads during loading. No OOM was observed. The setup served successfully, but startup memory headroom was tight.
- DeepSeek passed ordinary completion, native tool-result continuation, and one real proposal through each of `INSERT_TEXT`, `DELETE_TEXT`, `REPLACE_TEXT`, and `MOVE_TEXT`. This was the standalone four-edit diagnostic, not the campaign's 20-attempt canary.
- 42 HotPotQA/Terminal-Bench pilot-logic tests passed on Della. These are CPU tests, not real Terminal-Bench task execution.
- A quoting bug in the standalone DeepSeek diagnostic's embedded package-inventory Python was fixed locally. The repair and serving-smoke regression checks passed: 22 tests, Ruff, and Bash syntax validation.
- Qwen's source, checkpoint, and environment preflight passed. After DeepSeek released its GPU memory, Qwen loaded BF16 weights on one H200, reporting **50.22 GiB** for model loading and 13.12 seconds including setup. Graph compilation also completed. It was stopped before endpoint readiness and inference were verified; its HotPotQA pilot remains uncompleted.
- A real Terminal-Bench pilot could not run: Harbor and Docker were absent. Apptainer was available, but the current GEPA Terminal-Bench adapter requires Docker.

## Evidence and remaining decisions

Remote evidence: `/scratch/gpfs/BSTEWART/gm8296/gepa/logs/hotpotqa/verify/13806836/`.

Local evidence is retained under `outputs/hardware-sizing-20260912/interactive-evidence/13806836/`, with the raw archive at `outputs/hardware-sizing-20260912/13806836-evidence.tar.gz`. Key files are `measurement-summary.json`, `deepseek-uncapped/result.json`, `deepseek-uncapped/response.json`, `deepseek-tool-verification.log`, original provider-attempt logs, GPU/host-memory samples, server metrics, and Qwen's startup log.

Still to decide or verify:

1. Select production output limits after reviewing the 46,901-token editor turn and sampling more natural finishing lengths. Production caps remain unchanged.
2. Complete a full FOREST proposal, application, reevaluation, and acceptance/rejection cycle under the selected settings.
3. Measure other optimizer ablations and a larger training sample before freezing runtime/resource estimates or concurrency.
4. Finish Qwen endpoint/inference/optimizer checks and prepare the actual Terminal-Bench container runtime.

These results establish successful DeepSeek inference and tool compatibility, plus concrete token and timing measurements. They do not mark either complete model pilot or a production campaign ready.
