# Tau banking prompt optimization

This example runs the standard upstream **text `LLMAgent` + `UserSimulator`** on
`banking_knowledge`, using the official environment and reward evaluator. It
optimizes the actual agent system prompt. The user simulator is a separate fixed
`gpt-4.1-2025-04-14` snapshot on the standard OpenAI endpoint, at temperature 0,
and is never optimized.

The solver defaults to the shared `QWEN3_8_27B_MODEL` checkpoint and the proposer
to `DEEPSEEK_V4_1_FLASH_MODEL`. The shared runner supplies their checkpoint
revisions, HotPotQA role budgets, provider retries, FOREST constants and no-op
recovery. Solver `top_k` is transported in `extra_body` for the OpenAI-compatible
vLLM endpoint; the other applicable shared request settings are preserved.

## Baseline and provenance

The pinned official repository is
[`sierra-research/tau2-bench` at `5bfa7e37b36656b37dc6d022156be6563c1007f3`](https://github.com/sierra-research/tau2-bench/tree/5bfa7e37b36656b37dc6d022156be6563c1007f3),
version **1.0.1**, including the July banking grading corrections. This repository
now contains the tau3 release, while its Python package/import namespace remains
`tau2`. The published benchmark is
[τ-Knowledge, arXiv:2603.04370v1](https://arxiv.org/abs/2603.04370v1).

We choose upstream's documented [`bm25` configuration](https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3/src/tau2/knowledge/README.md),
with its default top-k of 10. The current upstream default, `alltools`, additionally
uses dense embeddings and a shell sandbox. BM25 is an official supported sparse
retrieval baseline that avoids both services; it is **not** a reproduction of the
paper's dense-retrieval or terminal-search result, nor of the current `alltools`
leaderboard default.

The small integration changes are explicit:

- The shipped BM25 system prompt is wrapped with the shared provider-specific
  `structured_prompt` formatting. Both optimization methods and their unoptimized
  starting baseline receive exactly the same resulting text. A subclass overrides
  only `LLMAgent.system_prompt`; upstream handles every turn and tool call.
- Corpus documents are inserted in ascending document-ID order before upstream
  BM25 indexing. This makes equal-score ordering independent of filesystem order.
- Provider-level retry limits/logging come from `examples.common.provider_retries`;
  there are no whole-trajectory retries or user-hallucination rerolls.
- We use upstream `EvaluationType.ALL`, which evaluates every required reward
  basis, including the one task requiring an NL judge. Optional diagnostic NL
  assertions that do not gate a task's reward are omitted. The judge is fixed to
  upstream's `gpt-4.1-2025-04-14` default. The upstream builder's per-task read-tool
  allowlist is passed unchanged to execution **and** evaluation. Incomplete or
  mismatched required NL-assertion results abort even when upstream would treat
  an empty result list as vacuously successful.

The official reward is the product of the task's `reward_basis`: 87 tasks use DB
end-state grading, nine require ACTION grading, and one uses DB + NL_ASSERTION.
Reference actions are consumed by the official grader; they are not given to the
solver or the reflection dataset. We do not substitute matching tool-call strings
for DB evaluation. See the [official evaluation specification](https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3/docs/evaluation.md).

## Data and repetitions

`banking-manifest.json` pins all 97 ordered task-record fingerprints, the 698
knowledge documents, database, prompt files, user-simulator guidelines, runtime
source tree, dependency lock, and upstream revision. Loading rejects drift.
Upstream provides no optimization split. This example creates **57 train / 19
validation / 21 test** tasks, with stable globally unique IDs.

Local grouping joins tasks with an explicit shared `user_id` or `customer_name`,
identical user scenarios, or a task reference in the upstream notes. The resulting
69 connected groups are ordered by SHA-256 of the fixed split salt and group ID;
the first floor(60%) are training, the next groups through floor(80%) validation,
and the rest test. Tasks retain sorted IDs within each group. `scenario_id` is the
upstream task ID; `group_id` is this example's conservative local grouping, not an
upstream family label. This prevents explicit customer/variant overlap across
splits; it does not claim that all semantically similar banking workflows are
separate. The shared knowledge base is intentionally available to all splits.
Optimizer settings and CLI prefix limits never resample these groups.
When GEPA pads a training minibatch with repeated task IDs, each occurrence runs
as a separate episode with its own artifact, preserving sampled order even with
multiple workers. Validation and test batches still require unique task IDs.

Training and validation use trial 0. Held-out winners and the shared starting
baseline each use **four** independent trial seeds, drawn with upstream's
`random.Random(300).randint(0, 1000000)` policy. The explicit shared evaluation
context supplies the repetition index, including when resuming a partial test.
The optimizer's `--seed` does not alter this fixed benchmark trial schedule.
We report **pass^k = C(successful trials, k) / C(total trials, k)**, averaged over
tasks, for k=1…4. This is all-k reliability, not at-least-one-success pass@k.
Incomplete repetition matrices are rejected.

The shared runner freezes the validation-selected prompt before testing and
reuses a separately identified unoptimized baseline. Pilot timing uses only
training tasks. Per-episode `elapsed_seconds` includes environment construction,
all agent/user/tool turns, and official evaluation. Runtime setup and subprocess
startup also contribute to batch throughput, but are not mislabeled as episode
latency. Shared summaries retain count, mean, median, p95 and execution errors;
`metrics.scored_failures` additionally counts completed benchmark failures.
Step/error/context limits receive official zero reward. Infrastructure failures,
malformed worker results, missing grades and incomplete batches abort rather than
silently disappear from the denominator. Unscored failures retain elapsed time
and error type in separate failure artifacts. Actual upstream simulation JSON, the
exact optimized prompt, and provider-attempt logs are retained in `tau-episodes`.

## Setup and use

Use Python 3.12 for the optimizer environment. From this repository's root:

```sh
uv sync --extra dev --python 3.12
mkdir -p examples/taubench/.cache
git clone https://github.com/sierra-research/tau2-bench.git examples/taubench/.cache/tau2-bench
git -C examples/taubench/.cache/tau2-bench checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3
uv sync --project examples/taubench/.cache/tau2-bench --frozen --extra knowledge --python 3.12.11
uv run python -m examples.taubench.main --help
```

Do **not** add `tau2` to the optimizer environment: its upstream LiteLLM requirement
conflicts with this repository. `TauRuntime` executes a worker using the upstream
frozen `uv.lock` (LiteLLM 1.81.11) and Python 3.12.11. It supplements the lock with
`websockets==15.0.1`, the same version present in upstream's voice-extra lock,
because upstream text imports reach a voice module importing websockets. No voice
APIs, embeddings, sandbox, or model-weight downloads are needed. `--tau-source`
can select another checkout, but its revision and file fingerprints must match.

A real pilot requires the configured Qwen solver endpoint and an OpenAI key for
the fixed simulator. Optimization additionally requires the DeepSeek proposer
endpoint. These commands make model calls:

```sh
uv run python -m examples.taubench.main --mode pilot --run-dir outputs/tau-pilot \
  --solver-api-base http://SOLVER_HOST:8000/v1 --pilot-size 3
uv run python -m examples.taubench.main --condition both --run-dir outputs/tau-banking \
  --solver-api-base http://SOLVER_HOST:8000/v1 --reflection-api-base http://PROPOSER_HOST:8000/v1
```

No paid model campaign or Slurm launch is part of the offline verification.

## Offline verification

```sh
uv run python -m pytest tests/test_taubench.py -q
TAU_BANKING_OFFLINE_CHECKS=1 \
  TAU2_DATA_DIR="$PWD/examples/taubench/.cache/tau2-bench/data" \
  PYTHONPATH="$PWD:$PWD/src" \
  uv run --directory examples/taubench/.cache/tau2-bench --frozen --extra knowledge \
  --with websockets==15.0.1 --with pytest==9.0.1 --python 3.12.11 \
  python -m pytest "$PWD/tests/test_taubench_upstream.py" -q
```

The second command executes the real LLMAgent, UserSimulator, BM25 tools,
orchestrator, strict replay evaluator and pass^k function, replacing only model
API responses. It checks prompt propagation, nonleakage, real DB failure despite
a textual success claim, grading-fix configuration, and incomplete-episode
rejection. It makes no paid calls. This establishes runtime wiring, not measured
Qwen performance or live endpoint compatibility.
