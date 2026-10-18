# SWE-bench Live with SWE-agent on Della

This is the seventh benchmark alongside HotPotQA, DecisionBench, AppWorld,
OBLIQ, Tau and Terminal-Bench. It uses Microsoft's **SWE-bench Live** Python
dataset and official test scripts and scorer. The solver is upstream SWE-agent,
with an Apptainer transport for Della CPU allocations. It does not use the older SWE-bench
Verified prototype or its patch-similarity metric.

## Prepare without running an experiment

From the repository root, with the [local API environment](../local_api/README.md)
installed:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.swebench_live.prepare

# Download native task images only; no containers or model calls:
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.swebench_live.prepare --images
```

Preparation stores assets in `.cache/swebench-live`; `--root` selects another
location. The data revision is `b51a86422e10cfd403beb4773e5a2947953e36ec` and the
official Python harness is `ad79b850f15e33992e96f03f6e97f05ddf9aa0be`. All five
Parquet files and their ordered JSONL exports have fixed checksums. Updating to
a newer Live dataset requires a new reviewed manifest and campaign.

The upstream `full`, `lite`, `verified` and `test` names are overlapping subsets,
not optimization partitions. The official `test` has 1,000 unique tasks and is
preserved in its original order. Excluding every test ID from `full` leaves 895
unique tasks; sorting their IDs by SHA-256 and dividing in half gives 447 training
and 448 validation tasks. These are derived research splits, not official
training/validation splits. The shared CLI defaults to the first 60 training, 120 validation, and 120
held-out tasks: 300 total, using the requested GEPA 20/40/40 proportions. Limits, ordered
records, source hashes and runtime are saved in each immutable run contract.

The pinned release has one repeated ID in `full` and `verified`, with differing
test commands/hints, and eight official test IDs absent from `full`. Raw files
retain these rows. The repeated ID is held out; its authoritative `test` record
is used. No test ID enters training or validation. `data-manifest.json` records
these facts rather than silently treating the upstream subsets as disjoint.

`local-runtime.patch` removes eager Modal imports/dependency and rejects that
execution mode. It also selects the published AMD64 images on Apple Silicon.
The grading/test logic is unchanged. Dependencies are hash-locked in
`runtime-requirements.txt`, installed in a separate environment, and verified
with the source patch before execution. No Modal SDK or credentials are needed.
To regenerate that dependency lock after a deliberate upstream update, run
`uv pip compile <prepared-source>/pyproject.toml --no-sources --python-version 3.12
--generate-hashes -o examples/swebench_live/runtime-requirements.txt` outside the
upstream workspace (its pinned workspace references an absent `launch` package).

Image preparation resolves official DockerHub tags to immutable manifests,
requests `linux/amd64`, verifies the downloaded IDs/digests, and keeps a disk
reserve. Re-running the preparation command continues existing downloads and
never replaces a recorded manifest with a newer tag. `--image-split train`,
`val` or `test` stages one partition. All partitions together need 1,895 distinct
task images; the complete collection can far exceed a laptop's free storage.
Partial staging is reported explicitly and does not imply native readiness.

The retained Docker preparation and grading helpers require four CPUs and
16 GiB RAM per sequential task. A 4 GiB Docker VM cannot pass that check.
Preparation never changes Docker settings or restarts it. Image downloads
alone are not native execution tests.

## Apptainer on Della

Stage images on an internet-connected visualization host, using a fresh scratch
directory. The Apptainer backend reads registry metadata directly, pins the
Linux AMD64 OCI manifest and verifies the resulting SIF checksum. It never
requires a Docker daemon. For example, prepare one training image without
starting a container or making a model request:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.swebench_live.prepare \
  --root /absolute/scratch/path/to/swebench-live \
  --backend apptainer --images --image-split train \
  --instance-id patroni__patroni-2998
```

Omit `--instance-id` to stage the entire selected image split. Partial staging
does not change dataset selection: the run preflight requires every image
selected by its training, validation and test limits. The default run needs
300 images; preparing one image proves no readiness for that collection.
Keep the Apptainer manifest and SIFs together in `apptainer-images/`. A changed
SIF or an existing unverified SIF fails closed and requires inspection.

Execution requires an allocated Linux AMD64 node with exactly one node, four
CPUs, `--mem=16G`, no GPUs, and sequential tasks (`--max-workers 1`). An explicit
`--task-memory-gib` must match the allocation and becomes part of the experiment
contract. Slurm enforces the allocation's CPU and memory budget. Set `TMPDIR`
to a private job directory with enough space to unpack a task image. Downloads
and image conversion belong on the visualization host, not inside task runs.

Each task gets a fresh writable sandbox and a persistent Apptainer instance.
File edits and background services survive individual shell commands. The
instance has private PID, IPC and network namespaces, local loopback, no external
network, and no host home, scratch, current-directory or configured bind mounts.
An environment allowlist excludes API keys and user-supplied Apptainer bindings.
The instance is stopped before its filesystem is removed. The evaluator starts
a separate fresh instance and uses the pinned official patch application,
evaluation script and resolution scorer. Complete test logs and infrastructure
failures are preserved.

Inside the CPU allocation, load `proxy/default` for the parent driver's
OpenRouter requests. Task containers remain offline. After paid work is
separately authorized, add these arguments to the API command below:

```text
--container-backend apptainer --swebench-live-root /absolute/scratch/path/to/swebench-live
```

The backend, Apptainer binary/version, isolation settings, allocation limits,
OCI digests and SIF checksums enter the run contract. Docker and Apptainer runs
cannot silently share checkpoints or starting-prompt baselines. Before a paid
campaign, verify native task execution and official negative/reference grading
on training records; a download or an offline unit test is insufficient.

See the [Della API runbook](../../scripts/della/API_RUNBOOK.md) and upstream
[Apptainer instances](https://apptainer.org/docs/user/latest/running_services.html)
and [fakeroot](https://apptainer.org/docs/user/latest/fakeroot.html) documentation.

## SWE-agent runtime

The [official Live evaluation guide](https://github.com/microsoft/SWE-bench-Live/blob/main/evaluation/README.md)
links [SWE-agent-for-eval](https://github.com/njukenanli/SWE-agent-for-eval).
We pin commit `824224225e1e952b48da91f2d1bde38adb2115f2` (SWE-agent 1.1.0,
with its bundled SWE-ReX 1.4.0). `DefaultAgent`, the default system/instance
prompts, function-calling parser, bash and text-editor tools, submission review,
history processing, and stopping behavior all come from that source. The old
`native-shell-v1` loop and its custom seed prompt are removed.

This pin is the Live-recommended evaluation fork as inspected on October 17,
2026, not a claim of bit-for-bit reproduction of the paper's 2025 runs. Its
configuration sets `per_instance_call_limit: 100`. The upstream implementation
checks the limit after a response and can issue a 101st call before stopping;
we retain and record that behavior rather than silently changing the agent.
The default tool timeouts are also upstream's: 30 seconds per action and
1,800 seconds of cumulative execution. The text editor's own clipping and
file-map behavior remain active; this is not the old untruncated shell tool.

Create a **new** driver environment on the visualization host. Install the
hash-locked agent dependencies in addition to the shared runner dependencies;
never install into a frozen campaign environment:

```bash
uv pip install --python /absolute/new/venv/bin/python --require-hashes \
  -r examples/swebench_live/agent-requirements.txt

# --portable-python is the root of a uv-managed Linux CPython 3.12 installation.
uv run --no-project --python /absolute/new/venv/bin/python python -m examples.swebench_live.prepare_swe_agent \
  --root /absolute/new/swe-agent \
  --portable-python /absolute/path/to/cpython-3.12-linux-x86_64-gnu
```

Preparation clones and verifies the upstream commit and builds a hash-verified,
relocatable SWE-ReX payload. It stages the upstream editor's dependency wheels
for offline installation. It makes no model calls and starts no task container.
A failed preparation directory is retained; inspect it and use a fresh root.

The only execution adaptation is transport: a private pipe carries SWE-ReX's
requests to its **unchanged LocalRuntime** inside the Apptainer instance. This
preserves the persistent PTY, command parsing, timeouts and error types. The
portable Python runtime is copied into the disposable task filesystem; no host
mount, API key, gold patch, test metadata or external network is exposed. The
upstream tools themselves are uploaded unmodified and install from staged
wheels. Transport uncertainty stops the trial without replay.

The explicitly selected OpenRouter model and request parameters replace the
fork's example Azure model. Our shared provider wrapper retains durable attempt
accounting and retries; the extra SWE-agent API retry layer is set to one attempt
to avoid retrying an exhausted or unknown outcome. Provider failures and runtime
errors are preserved as unresolved infrastructure failures, not scored as failed
tests. Raw upstream statistics remain in `.traj`; provider journals are the
accounting authority, including unknown costs rather than fabricated zeros.

The standard starting system template is used verbatim. Vanilla GEPA can
optimize that one template; the instance prompt, tools, parser and history policy
stay fixed. FOREST and the other structured-prompt conditions are blocked before
runtime work: their editor currently requires section headers absent from the
upstream prompt. Supporting those conditions needs a separately reviewed
integration; this adapter never reformats the standard starting prompt silently.
Prompt optimization, the 60/120/120 split, chosen model/provider, and Apptainer
are declared research-protocol differences from a leaderboard run. Validation
selects the winner; held-out results never guide optimization.

After the solver closes, its upstream submission patch is frozen and scored in
a separate fresh container using the official grader. One test evaluation is
performed per candidate/task. Empty submissions score zero. Full trajectories,
errors, source/config hashes, payload hashes and runtime identities are retained.
Old custom-agent checkpoints and baselines cannot match this new contract.

After separate paid-call authorization, the command is:

```bash
uv run --no-project --python /absolute/new/venv/bin/python python -m examples.swebench_live.main \
  --api-profile outputs/api-profile.json --env-file /absolute/path/to/.env \
  --allow-api-calls --condition vanilla --run-dir outputs/swebench-live-new-campaign \
  --swebench-live-root /absolute/scratch/assets --swe-agent-root /absolute/new/swe-agent \
  --container-backend apptainer --max-workers 1
```

The entrypoint now requires Apptainer. Docker image preparation and the official
Docker grading helper remain available; there is no fallback to a custom solver.
A native scripted-model check establishes agent/tool/transport compatibility,
not model quality, a completed optimization campaign, or all-task readiness.

Sources: [official Python harness](https://github.com/microsoft/SWE-bench-Live/tree/python-only),
[dataset](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live),
[pinned agent configuration](https://github.com/njukenanli/SWE-agent-for-eval/blob/824224225e1e952b48da91f2d1bde38adb2115f2/config/default.yaml).
