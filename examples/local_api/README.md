# Local benchmarks with OpenRouter

All six existing benchmark entrypoints can execute on this computer while model
requests go through one OpenRouter key. Solver and proposer models are explicit
choices: there is no assumed DeepSeek, Qwen, or OpenAI model for an API campaign.
Models may differ between roles or be identical. The existing Della entrypoints,
defaults, jobs and artifacts remain independent.

The local process has no cloud sandbox lifetime. Completed task groups, provider
responses, optimizer checkpoints and held-out repetitions are journaled on disk.
An unresolved started task or Jev request blocks replay for review; this is not an
automatic restart service. Keep the computer awake during future runs.

## Prepare without launching

From the repository root, install the local Python driver:

```bash
export DSPY_CACHEDIR="$HOME/.cache/gepa/dspy"
UV_PROJECT_ENVIRONMENT=.venv-local-api uv sync --locked --python 3.12 \
  --extra full --extra decisionbench --extra obliqbench --extra jev \
  --group hotpotqa-task-program

UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.local_api.cli doctor
```

The pinned DSPy package initializes a cache directory on import, even when task
response caching is disabled. Use a directory; `/dev/null` is not a valid value.

Choose model IDs from the [OpenRouter catalog](https://openrouter.ai/models).
This example uses GPT-4.1 mini in both roles; it is an example, not a campaign
decision or a claim about benchmark quality:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.local_api.cli plan \
  --solver-model openai/gpt-4.1-mini \
  --proposer-model openai/gpt-4.1-mini \
  --solver-parameters '{"max_tokens":32768,"temperature":1.0,"seed":0}' \
  --proposer-parameters '{"max_tokens":32768,"temperature":1.0,"seed":0}' \
  --input-output-ratio 5 --output outputs/api-profile.json
```

Planning makes public catalog requests only. It picks the cheapest available
provider for each chosen model at the stated uncached input/output token ratio,
requiring native tools, the output capacity and every requested control. Prices
for input and output both matter. It records provider, quantization, model name,
context capacity, parameters, timestamp and price caps. Cache savings, account
fees and retries are additional considerations; the ratio is an assumption, not
measured benchmark usage. Reusing an output file is refused.

DecisionBench additionally requires `response_format` and `structured_outputs`
on the selected solver route; it rejects an incompatible route before execution.
Terminal-Bench requires solver `seed` support for its fixed repetition schedule.

Only `max_tokens` defaults to 32768. Other sampling controls are omitted unless
specified, so the provider's defaults apply and that omission is recorded. Set
`reasoning_effort` in parameter JSON only for models/providers that support it.
Local vLLM thinking budgets and checkpoint hashes are never invented for APIs.

Load the key from the environment or an existing dotenv file. Only
`OPENROUTER_API_KEY` is imported from that file; it is never saved in a profile,
contract or log. Authentication and route checks do not invoke a model:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.local_api.cli check \
  --profile outputs/api-profile.json --env-file /absolute/path/to/.env \
  --output outputs/api-auth-check.json
```

Adding `--allow-check-calls` explicitly permits one tiny request for each selected
service, with no retries. Select services with `--roles solver proposer tau jev`.
These synthetic checks do not execute benchmark tasks or prove optimization,
native tool execution, long-context compatibility or held-out quality. Interrupted
and failed checks remain in the receipt. A new check requires a new receipt path.

## Local prerequisites

API hosting replaces model serving. The official task environments and data still
run locally and retain their integrity checks:

| Benchmark | Local preparation |
|---|---|
| [HotPotQA](../hotpotqa/) | Pinned DSPy, the `wiki17` extra and the verified Wiki-2017 BM25 index; `--wiki17-dir` selects an existing index. |
| [DecisionBench](../decisionbench/README.md) | Pinned `decisionbench` extra and exact canonical Parquet; `--data-file` selects a local copy. |
| [AppWorld](../appworld/README.md) | Its separate pinned Python runtime and public task bundle; `--appworld-root` and `--appworld-python` select them. |
| [OBLIQ-Bench](../obliqbench/README.md) | Frozen corpora and Qwen embedding index, built on CPU if desired. `--data-dir` and `--index-dir` select existing data. API use does not replace the retriever with BM25. |
| [Tau](../taubench/README.md) | Separate pinned upstream environment and banking data under `--tau-source`. |
| [Terminal-Bench](../terminalbench/) | Pinned Harbor, working Docker and the actual task images. Runtime records for a local GPU server are unnecessary with `--api-profile`; native task checks remain mandatory. Use `--task-scope della88` to retain that task membership. |

`doctor` reports presence and Docker resources, not verified task readiness.
Some task images need x86 emulation on Apple Silicon. Dataset/index and Docker
disk/RAM requirements remain material; do not infer full readiness from a working
API key or replace unsupported tasks silently.

Tau's official GPT-4.1 simulator and judge are separate from the chosen solver
and proposer. They use the same key through the recorded OpenRouter route, with
an explicit 32768 output ceiling. The official dated model name is checked in
endpoint metadata. Account privacy restrictions remain effective; a rejected
route is not retried through a different provider.

FOREST's optional Jev controller also uses this key, through OpenRouter's typed
decision endpoint. It requests `typesafe/jev-1.13` and validates the observed
dated response ID `typesafe/jev-1.13-20260917`. It does not upgrade Jev or turn
typed decisions into chat completions. Its price-based cost estimate remains
distinct from reported billing, which may be absent.

## Run only after authorization

Each entrypoint accepts the same two flags. An API profile alone cannot launch
a benchmark: `--allow-api-calls` is also required. This example **starts paid
benchmark work** and was not run during infrastructure setup:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.decisionbench.main \
  --api-profile outputs/api-profile.json --env-file /absolute/path/to/.env \
  --allow-api-calls --condition both --controller-selection jev \
  --run-dir outputs/decisionbench-api-new-campaign
```

Replace `decisionbench` with `hotpotqa`, `appworld`, `obliqbench`, `taubench` or
`terminalbench`, adding that benchmark's preparation arguments. All documented
[method and search variants](../BENCHMARKS.md#supported-variants) remain available.
Terminal-Bench additionally caps both roles at 32768 combined output tokens.

Every run rechecks catalog availability, parameter support, model identity and
price caps. Provider fallback is disabled and both token prices are capped on
every request. Account privacy policy is not altered. The profile is included in
the immutable run and baseline identities, and each response records the actual
reported provider, model, token usage and billing when available. Hosted model
aliases do not attest to immutable weights; use fresh campaigns for different
models, routes or settings, and do not resume Della checkpoints with an API model.
