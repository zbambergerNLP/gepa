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

Compute nodes cannot fetch the required Debian, Python, container, and task
packages. Prepare a sealed offline bundle on the visualization host, from the
synced checkout, before submitting a job:

```sh
export PYTHONPATH="$PWD/src:$PWD"
export APPTAINER_CACHEDIR="$SCRATCH_BASE/.cache/terminalbench/apptainer"
export APPTAINER_TMPDIR="$(mktemp -d /tmp/terminalbench-build.XXXXXX)"
"$SCRATCH_BASE/.tools/uv-0.9.13/uv" run --no-project \
  --python "$SCRATCH_BASE/.venv/bin/python" python \
  -m examples.terminalbench.prepare_offline \
  --bundle-dir "$SCRATCH_BASE/terminalbench-offline/pilot-v1" \
  --image-cache "$SCRATCH_BASE/.cache/terminalbench/sif" \
  --harbor-executable "$SCRATCH_BASE/.tools/terminalbench-harbor-0.22.0/venv/bin/harbor"
```

The default prepares the first two training tasks, `fix-ocaml-gc` and
`log-summary-date-ranges`. Add `--all-tasks` to prepare the complete pinned set
of 89 tasks. For an infrastructure check, repeat `--task-id terminal-bench/NAME`
to select particular environments; preparation never runs their solvers or graders.
The checked-in recipes identify each verifier's Python version, package pins,
system dependencies, and public external inputs. Authentic uv installers and
declared download URLs use a hash-checked local cache. Git mirrors, model and
dataset snapshots, and source archives retain their public provenance. The
historical MTEB cache includes every model's relevant results and metadata.
No solution, task instruction, or scoring file is changed.

The bundle records every task file, recipe, and prepared image by SHA-256. The
runner checks these bytes before evaluation and includes the runtime hashes in
its resume identity. Changed artifacts and missing tasks fail before execution.
Each task is sealed only after its installer replay and dependency imports pass
with Harbor's command search order, isolated home/tmp mounts, offline package
managers, and blocked network proxies. `--resume` verifies completed parts against
the current recipe and probe before reuse. It can continue an interrupted build
after an unbuilt task's recipe is repaired; changing an already sealed task
requires a new bundle.
A full campaign requires the completed 89-task bundle. A recipe inventory or a
two-task pilot alone does not prove that all environments have been built.

On an allocated compute node with the pinned Qwen server running:

```sh
uv run --no-sync python -m examples.terminalbench.main --mode pilot \
  --pilot-size 2 --max-workers 1 \
  --container-runtime singularity \
  --offline-task-bundle "$SCRATCH_BASE/terminalbench-offline/pilot-v1/bundle.json" \
  --runtime-record /path/to/solver-runtime.json \
  --solver-api-base http://localhost:8000/v1 \
  --run-dir outputs/terminalbench-offline-pilot
```

The explicit backend and prepared image hashes are part of the run contract, so
Docker, raw Apptainer images, and prepared offline images cannot reuse each
other's results. The native backend uses Apptainer's `--fakeroot` and writable
temporary filesystem; allocate the task's requested CPU and memory through
Slurm. Run one task at a time because native Harbor shares the job's network
namespace. Full Della jobs should start both local model servers and Harbor
inside `scripts/della/remote/with_private_network.sh`; its private loopback
supports the Windows task's port 80 without changing the host's network policy.
The Windows prepared image also starts its published `supervisord` command,
which Docker would normally start automatically. Its empty nginx logs are
recreated during image preparation for the root-mapped runtime user; the service
configuration and log paths remain unchanged.
This mode has no external network route, so complete offline staging is required.
For an online Apptainer host, `--singularity-image-cache PATH` remains available
without an offline task bundle.

`terminal-bench/mailman` requires real nonroot service users. When that task is
selected, Singularity preflight checks the staged image's kernel UID/GID mappings
and real identity changes before any model calls. Della's current one-ID mapping
cannot run this task, with or without the private-network wrapper; libfakeroot's
simulated identity changes do not satisfy the requirement. A sealed image alone
does not establish Mailman runtime readiness. Use administrator-configured
[Apptainer subordinate UID/GID mappings](https://apptainer.org/docs/user/1.5/fakeroot.html)
with a launcher that preserves them and passes this check. The current
`with_private_network.sh --map-root-user` wrapper maps only one UID/GID; assigning
subordinate ranges alone does not make that launcher compatible. Alternatively,
use a separate approved Linux Docker host with colocated, attested model servers.
The current runner requires local Linux PID/socket attestation, so a macOS Docker
engine with tunneled Della model endpoints is not a supported replacement. Docker
is not permitted on Princeton clusters. Task users, ownership, permissions and
grading remain unchanged; training-only pilots that exclude Mailman are unaffected.

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
  tests/test_benchmark_runner.py tests/test_terminalbench_staging.py \
  tests/test_terminalbench_prepare_offline.py
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
pins. The official Docker runtime completed the pinned training task
`log-summary-date-ranges` with its reference solution and verifier reward 1.0.
The prepared offline Apptainer image also completed that official reference
solution on Della's CPU partition, with reward 1.0 and no trial errors (job
14986165, `COMPLETED`, exit 0). This checks offline container execution and
verification; it is not a model quality or Qwen latency measurement.
