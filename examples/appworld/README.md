# AppWorld

This integration runs a ReAct interactive Python-code agent in the **official
AppWorld environment** and scores the resulting application state with the
**official evaluator**. It is a GEPA adaptation of AppWorld's published ReAct
baseline, not an unmodified official agent or a reproduction of the paper's
original model scores.

## Baseline and pins

- Engine: `appworld==0.1.3.post1`, upstream commit
  `66ad8099e12188ece0d3fe45e661dbc01880813b` (the stable published release,
  rather than the moving `0.2.0.dev0` branch).
- Runtime: Python `3.11.14`; every environment dependency is version- and
  hash-pinned in `runtime-requirements.txt`. Installed engine code is also
  fingerprinted and checked against `data_pin.json`.
- Dataset: the official `data-0.1.0.bundle`, SHA-256
  `fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250`.
- Agent: upstream ReAct controller, public prompt, and GPT-4-Turbo baseline
  configuration provide the code/observation loop, fixed demonstration,
  first-code-block behavior, and 100-step limit.
- Solver: shared `QWEN3_8_27B_MODEL`; proposer:
  shared `DEEPSEEK_V4_1_FLASH_MODEL`. The shared runner records their immutable
  checkpoint revisions and applies the shared role budgets, decoding settings,
  provider retries, and FOREST configuration. There is no model fallback.

`upstream_react.txt` is the unchanged public upstream prompt, distributed under
the accompanying `UPSTREAM_LICENSE`. It is not extracted from a protected bundle.
Only its general instructions and final key instructions become the editable
`system_prompt`; the demonstration stays fixed. The actual candidate is the first
system message in **every model call**, including later turns of an episode.
The seed is wrapped with the shared provider-family `structured_prompt` helper.

Deliberate deviations from the published baseline are the requested model pair
and shared model budgets, relocating instructions from user messages to one
editable system message, retaining full conversation history instead of the old
character-based truncation, and requiring a closed Python fence. We omit the old
code-fence stop sequence and partial-code repair because the shared provider
policy rejects incomplete model responses. No API predictor, oracle API list,
ground-truth answer, solution, or extra agent module is used.

## Installation and a free environment check

Run from the GEPA repository root on Linux or macOS:

```sh
uv sync --extra full
uv run python -m examples.appworld.prepare
```

`prepare` creates an isolated environment under `examples/appworld/.runtime`,
runs the official `appworld install`, downloads the approximately 34 MB encrypted
release, verifies its SHA-256 before unpacking with AppWorld's own utility, and
checks the entire corpus and runtime. It reuses an existing matching corpus and
refuses data drift. Use `--root /path/to/private/world --venv /path/to/runtime`
to relocate the artifacts. It does not download model weights or call a model.

AppWorld's stable release requires Pydantic 1; the shared model clients use
Pydantic 2. The two run in separate processes/environments. Each episode gets a
fresh official `AppWorld` instance, protecting its global DB state and signal
timeouts from concurrent-task interference. The adapter executes batches
sequentially; `--max-workers` does not introduce AppWorld thread concurrency.

Install the test dependencies, then exercise the real engine and evaluator with
scripted API calls and no paid model calls:

```sh
uv sync --extra dev
APPWORLD_SMOKE_ROOT="$PWD/examples/appworld/.runtime/world" \
APPWORLD_SMOKE_PYTHON="$PWD/examples/appworld/.runtime/venv/bin/python" \
uv run python -m pytest -q tests/test_appworld.py
```

The smoke test verifies all official `load_task_ids` assignments, executes an API
discovery call and `supervisor.complete_task()`, and confirms that claiming
completion without achieving the goal scores zero. It also runs official metric
aggregation and verifies the corpus was not modified. Without the two environment
variables, only this smoke test is skipped; the remaining tests run offline.

## Runs

The CLI uses the shared benchmark runner. Once the model endpoints are prepared
and a model run is authorized:

```sh
uv run python -m examples.appworld.main --mode pilot \
  --solver-api-base http://SOLVER_HOST:PORT/v1 \
  --reflection-api-base http://PROPOSER_HOST:PORT/v1 \
  --run-dir outputs/appworld-pilot

uv run python -m examples.appworld.main --condition both \
  --solver-api-base http://SOLVER_HOST:PORT/v1 \
  --reflection-api-base http://PROPOSER_HOST:PORT/v1 \
  --run-dir outputs/appworld
```

`--mode baseline` runs the common unoptimized reference. Pilot calibration uses
training tasks only. Benchmark-specific options are `--appworld-root`,
`--appworld-python`, and `--appworld-max-steps` (default 100). Run `--help` for
shared limits, optimization budgets, model endpoints, and tracking options.

A bounded optimizer pilot exercises every variant on training tasks only:

```sh
uv run --no-sync python -m examples.appworld.main --mode optimizer-pilot \
  --condition all --pilot-size 1 --pilot-proposals 1 \
  --solver-api-base http://SOLVER_HOST:PORT/v1 \
  --reflection-api-base http://PROPOSER_HOST:PORT/v1 \
  --run-dir outputs/appworld-optimizer-pilot
```

This runs `vanilla`, `random`, `action`, `react_v2_random`, and `react_v2`
through the same adapter. Both candidate selection and scoring use
the selected training prefix. The resulting `pilot-winner.json` records a
training score; it is not a validation-selected winner or held-out result.

## Splits, evaluation, and resumption

The official ordered assignments are retained in full before shared runner
limits are applied:

| Official split | Role | Tasks | Scenarios |
| --- | --- | ---: | ---: |
| `train` | Training | 90 | 30 |
| `dev` | Validation | 57 | 19 |
| `test_normal` | Held-out test | 168 | 56 |
| `test_challenge` | Held-out test | 417 | 139 |

Test order is `test_normal` followed by `test_challenge`. Records contain a stable
task ID, official subset, scenario ID, and SHA-256 of the complete task directory.
The corpus tree and ordered record fingerprints are pinned in `data_pin.json`.
The loader rejects repeated task IDs and scenarios crossing any official split.
No resampling or task-text-based filtering occurs. Only hashes and counts from
the protected corpus are committed. Task IDs and records are reconstructed locally.

Scores are exactly `float(world.evaluate().success)`, the official Task Goal
Completion result. `complete_task()` controls episode termination; it does not
determine success. A step-limit episode is evaluated on its actual final state,
as in the upstream harness, and may receive credit if the official tests pass.
Malformed actions execute no partial code and are recorded as errors, then the
existing state is evaluated. Missing runtime responses, empty or partial evaluator
results, and provider failures abort the evaluation instead of inventing scores.

The official `Metric.compute_metrics` computes percentages for the combined set
and each official subset, separately per repetition. Scenario Goal Completion is
reported only if every variant of every included scenario is present. A limited
prefix may report task completion while its scenario completion is `null`.
The default is one attempt per task with environment seed 100. GEPA's selection
score is the mean task-success fraction; official report metrics are percentages.

The shared runner freezes the validation-selected candidate before held-out test,
keeps test feedback outside optimization, reuses an identity-matched unoptimized
baseline, and rejects data/configuration drift on resume. Every episode records
wall time covering environment startup, all model/API/code steps, official
evaluation, and cleanup. The runner reports mean, median, p95, count, failures,
and batch throughput separately. Process startup is intentionally included in
both baseline and optimized latency.

Detailed state tests and their reports stay in the private AppWorld output tree.
Only public task context, model-visible observations, completion scores, and
pass counts reach reflection; reflection rejects non-training traces. Saved
official tracker evidence has a SHA-256, checked again by the worker before
aggregation or cached-run reporting. Keep the AppWorld output tree alongside
the shared runner outputs for resumable verification.

## Data distribution and limitations

Follow [AppWorld's release restrictions](https://github.com/StonyBrookNLP/appworld/tree/66ad8099e12188ece0d3fe45e661dbc01880813b#-release-disclaimer):
do not publish unpacked bundle contents, task instructions, evaluation programs,
or generated traces containing protected data. `.runtime`, local data, and
AppWorld outputs are ignored here. When using custom roots or tracking services,
keep their artifacts private too. No corpus or protected evaluator code is
vendored by this integration.

The local engine and scoring boundary have been exercised with both the free
environment check and a training-only Qwen pilot. Pilot scores are calibration
evidence, not held-out benchmark results. Linux is the intended cluster platform;
the real environment check and pilot client were run on macOS with Qwen hosted
on Linux. Windows is not supported by this subprocess/timeout setup.

## Primary sources

- [AppWorld paper](https://arxiv.org/abs/2407.18901)
- [Pinned release and official walkthrough](https://github.com/StonyBrookNLP/appworld/tree/66ad8099e12188ece0d3fe45e661dbc01880813b)
- [Published ReAct controller](https://github.com/StonyBrookNLP/appworld/blob/66ad8099e12188ece0d3fe45e661dbc01880813b/experiments/code/recoma/react_controller.py)
- [Published ReAct prompt](https://github.com/StonyBrookNLP/appworld/blob/66ad8099e12188ece0d3fe45e661dbc01880813b/experiments/prompts/react.txt)
- [Published baseline configuration](https://github.com/StonyBrookNLP/appworld/blob/66ad8099e12188ece0d3fe45e661dbc01880813b/experiments/configs/react_gpt4turbo_test_normal.jsonnet)
- [Official environment](https://github.com/StonyBrookNLP/appworld/blob/66ad8099e12188ece0d3fe45e661dbc01880813b/src/appworld/environment.py)
- [Official evaluator and metrics](https://github.com/StonyBrookNLP/appworld/blob/66ad8099e12188ece0d3fe45e661dbc01880813b/src/appworld/evaluator.py)
