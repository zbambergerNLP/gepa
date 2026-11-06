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
  `structured_prompt` formatting. Its existing Markdown headings are nested two
  levels beneath the wrapper, preserving their wording and hierarchy so FOREST
  does not misread policy headings as editable template sections. Both optimization
  methods and their unoptimized starting baseline receive exactly the same
  resulting text. A subclass overrides only `LLMAgent.system_prompt`; upstream
  handles every turn and tool call.
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

The official reward is the product of the task's `reward_basis`. In the preserved
97-task source, 87 tasks use DB end-state grading, nine require ACTION grading,
and one uses DB + NL_ASSERTION.
Reference actions are consumed by the official grader; they are not given to the
solver or the reflection dataset. We do not substitute matching tool-call strings
for DB evaluation. See the [official evaluation specification](https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3/docs/evaluation.md).

## Data and repetitions

`banking-manifest.json` pins all 97 ordered task-record fingerprints, the 698
knowledge documents, database, prompt files, user-simulator guidelines, runtime
source tree, dependency lock, and upstream revision. This original file remains
unchanged. Loading verifies its full identity before filtering any record.
Upstream provides no optimization split. The original manifest defines **57 train /
19 validation / 21 test** tasks, with stable globally unique IDs.

Future experiments use `tau-banking-ambiguity-exclusions-v2`, pinned in
`banking-exclusions-v2.json`. It excludes four training records with material
missing assumptions or conflicting policy readings:

- `task_024`: the card with the highest return changes with purchase timing.
  The scenario does not specify when the purchase must post.
- `task_075`: the scenario does not specify the comparison objective or opening
  funds needed to choose the reference account unambiguously.
- `task_084`: liability below the timing ceiling and precedence between
  provisional-credit category and merchant-contact rules have conflicting readings.
- `task_094`: the exact interest credit requires an unstated conversion convention.
  Policy specifies daily compounding but does not define the period and balance basis.

V1 excluded only `task_075` and `task_084`. Its manifest remains unchanged.
V2 adds `task_024` and `task_094` under the same material-underdetermination rule.
The retained dataset has **93 tasks: 53 train / 19 validation / 21 test**.
Each remaining record keeps its exact identity, split, and relative order.
The loader does not regroup, resplit, resample, or replace excluded records.
The exclusion manifest pins all four removed record fingerprints, plain reasons,
the original manifest checksum, retained counts, and the ordered retained digest.
Its own checksum is pinned in source. Any drift fails before an episode starts.

The additional exclusions do not depend on observed scores. `task_024` already
received successful grades in saved episodes. For its $40,000 ordinary purchase,
Bronze returns $900 within two months after an eligible opening; Silver returns $800.
After month two and within month six, Bronze returns $400 while Silver returns $800.
These cases use the frozen Bronze bonus and Silver double-rewards policies.
The reference Bronze choice fits the first case. The scenario does not require it.

For `task_094`, policy supports the 6.85% annual rate. The reference computes
`96000 * 0.0685 / 12 - 408 = 140`. Daily compounding does not establish that formula.
With a constant balance and effective annual yield, an equal-month conversion
gives $123.51; a 31-day interval in a 365-day year gives $133.73.
Those examples show the effect of missing assumptions. Neither becomes a new
reference answer. These findings do not prove either reference wrong or explain
the causes of failed episodes. They do not establish validity for every retained task.

Both methods and the worker use this same filtered manifest. The dataset version,
exclusion checksum, and derived manifest enter runtime and resume identity.
Unfiltered and v1 checkpoints cannot resume under v2, even when their selected
records all survive. V2 changes training coverage and the scientific protocol.
Use new run directories and baselines after separate run approval.
Existing v1 runs, including attempt11, finish under their original frozen contract.
Preserve every earlier attempt, grade, and original pilot record.
This change does not authorize another pilot or a full experiment.

Original grouping joins tasks with an explicit shared `user_id` or `customer_name`,
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
endpoint.

On Della, use the normal allocated-job network and load `proxy/default` for the
fixed simulator and judge. Princeton's [approved proxy services](https://researchcomputing.princeton.edu/support/knowledge-base/ports-and-proxydefault)
include `api.openai.com`. Include local solver/proposer hostnames in `NO_PROXY`
so their requests stay within the cluster. Tau requires this external API route;
the fully isolated network launcher used for offline Terminal-Bench cannot
provide it.

These commands make model calls:

```sh
uv run python -m examples.taubench.main --mode pilot --run-dir outputs/tau-pilot \
  --solver-api-base http://SOLVER_HOST:8000/v1 --pilot-size 3
uv run python -m examples.taubench.main --condition both --run-dir outputs/tau-banking \
  --solver-api-base http://SOLVER_HOST:8000/v1 --reflection-api-base http://PROPOSER_HOST:8000/v1
```

No paid model campaign or Slurm launch is part of the offline verification.

Hosted runs can explicitly select a custom simulator and judge through a
[version-3 API profile](../local_api/README.md). This is a separate Tau variant:
the user simulator can change task difficulty, and the judge can change required
NL assertion scores. Keep the same auxiliary model and decoding settings across
methods. Use fresh run directories and baselines. The default dated GPT-4.1
configuration and all pinned benchmark files remain unchanged.

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
