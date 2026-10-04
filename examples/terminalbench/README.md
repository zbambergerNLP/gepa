# Terminal-Bench 2.1

`python -m examples.terminalbench.main` now routes directly through
`examples.common.benchmark_runner`. It uses the existing pinned Harbor 0.22.0
transport, `PromptedTerminus` agent, immutable task manifest, and official
verifier rewards. GEPA and FOREST edit the same actual unified initial prompt
through the existing `system_prompt` text scope; auxiliary prompts and skills
remain fixed. That scope's component is named `instruction_prompt` and retains
Terminus's actual initial user-message role.

The shared runner owns optimization, validation selection, freezing, held-out
evaluation, the shared unoptimized baseline, timing reports, tracking, and resume
contracts. The benchmark supplies its adapter and scientific defaults:

- The ordered pinned split is 30 training, 19 validation, and 40 held-out tasks.
  This is the existing documented deterministic research split, not an official
  upstream training/validation assignment. Task package hashes and the complete
  manifest are included in the shared data identity before any prefix limits.
- Default solver/proposer are the shared `QWEN3_8_27B_MODEL` and
  `DEEPSEEK_V4_1_FLASH_MODEL`, with their pinned revisions and shared retry policy.
- Both roles retain Terminal-Bench's combined **32768-token** output cap.
  `configure_models` keeps shared sampling, timeout, and thinking-template
  settings, and removes HotPotQA's separate numerical thinking-token allowance.
  The actual adjusted kwargs are recorded by the shared runner.
- `--budget standard` uses four epochs; `--budget double` uses eight. The shared
  runner receives the corresponding proposal cap. `--max-metric-calls` is an
  optional additional stopping cap, rather than replacing the epoch budget.
- Held-out results and the starting baseline use three fresh repetitions, one
  official attempt per task per repetition. Solver request seeds follow the
  shared run seed plus repetition index. Pass@1 and sample standard deviation
  are reported across repetition means.

Epoch budget changes do not alter the runtime or full data identity, so matching
methods and budgets reuse the same starting baseline.

## Run

Install the pinned Harbor CLI in its own environment, then start the Docker
daemon on a Docker-capable host:

```sh
uv tool install --python 3.12 harbor==0.22.0
harbor --version
docker info
```

Princeton's clusters [do not permit Docker](https://researchcomputing.princeton.edu/get-started/guide-princeton-clusters/2-software).
On Della, use Harbor 0.22.0's official `singularity` backend with the cluster's
Apptainer installation. Run `scripts/della/remote/setup_terminalbench.sh` from a
synced checkout on the visualization host with `SCRATCH_BASE` set to the existing
GEPA directory. It installs an isolated Harbor environment, exposes Apptainer
under the `singularity` command required by Harbor, and prints the compute-job
PATH and persistent image-cache arguments.

This helper installs the CLI only. Harbor pulls uncached task images and installs
its server dependencies inside each fresh container. Della's preparation flow
uses the visualization host for internet access, so empty cache directories do
not make compute-node execution ready. Before submitting a trial, verify compute
egress or stage both the task images and the dependencies needed by Harbor's
bootstrap and the task. An offline bootstrap has not been verified on Della.

On the allocated compute node, add these arguments to the shared entrypoint:

```sh
--container-runtime singularity \
--singularity-image-cache "$SCRATCH_BASE/.cache/terminalbench/sif"
```

The explicit backend is part of the run contract, so Docker and Apptainer cannot
reuse each other's results. Task packages, images, verifier, prompts, model
settings and budgets remain pinned. The native backend requires each task's
published `docker_image`; it imports that image as a SIF and uses Apptainer's
`--fakeroot` and writable temporary filesystem. CLI installation alone does not
prove those kernel capabilities work on a particular compute node. Verify an
official training task before a campaign, and allocate the task's requested CPU
and memory through Slurm. Keep one concurrent task during the initial check.

After the pinned model servers and Harbor runtime are prepared:

```sh
uv run --no-sync python -m examples.terminalbench.main --mode pilot \
  --runtime-record /path/to/solver-runtime.json \
  --solver-api-base http://localhost:8000/v1 \
  --run-dir outputs/terminalbench-pilot

uv run --no-sync python -m examples.terminalbench.main --condition both \
  --runtime-record /path/to/solver-runtime.json \
  --proposer-runtime-record /path/to/proposer-runtime.json \
  --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --run-dir outputs/terminalbench
```

The primary route preserves `load_role_runtimes` checks for model bytes,
revisions, serving software, live process identity, and local endpoint ownership.
It also checks the exact Harbor version and selected container CLI before
evaluation. Docker additionally requires a reachable daemon. These checks are
not replaced by unchecked URL or model-name inputs.
Pilot and baseline modes validate only the solver because they do not use the
proposer. The optimizer's runtime is kept in its own run contract so a matching
standalone seed baseline is reused during optimization.
`--mode optimizer-pilot --condition all --pilot-size 1 --pilot-proposals 1`
exercises all five optimizer conditions on training tasks only, requires both
live servers, and writes no held-out result. The test suite replaces external
model and Docker I/O while executing those real optimizer and adapter paths.

The shared `--max-workers` controls Harbor's concurrent trials. Every output uses
its own official `TrialResult.finished_at - started_at` duration, including
environment setup, all tool/model execution, verification, and environment
teardown. Missing, malformed, or incomplete timing evidence is an error; batch
wall time is never divided by task count to fabricate latency. The shared runner
reports latency mean/median/p95 separately from measured batch throughput.

## Verification and source

```sh
uv run --no-sync python -m pytest -q tests/test_terminalbench_shared_runner.py \
  tests/test_terminalbench_runtime.py tests/test_terminal_bench_adapter.py \
  tests/test_benchmark_runner.py
```

The new route tests replace only external serving/Harbor boundaries and execute
the actual shared runner, adapter, candidate materialization, and reports. They
check real propagation of edited prompts, different per-task durations,
seed/repetition handling, failure on incomplete evidence, retained 4/8 epochs,
full-data identity, and required preflight.

Official timing semantics were checked against Harbor 0.22.0's
[TrialResult schema](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/models/trial/result.py)
and [trial finalization](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/trial/trial.py),
which records `finished_at` after stopping the environment. Existing adapter
documentation describes its maintained GEPA/Harbor port and benchmark source
pins. This migration adds no dependencies and does not change the official
execution/evaluation engine. No Docker trial or model-backed campaign was run
as part of this migration.
