# Modal benchmark infrastructure

Run the six shared benchmark entrypoints on Modal with the same pinned datasets,
solver/proposer checkpoints, request parameters, optimization settings, and native
task harnesses. Della and Modal runs retain separate runtime identities, baselines,
checkpoints and receipts. This backend creates fresh experiments; it does not
migrate, restart or modify an existing Della allocation.

The implementation has offline tests. Account-specific image builds, GPU capacity,
model startup, native environments and live optimizer pilots still require a
successful run in the target Modal workspace before campaign readiness is claimed.

## Benchmark environments

| Plan benchmark | Preparation and driver |
|---|---|
| `hotpotqa` | Frozen Wiki-2017 corpus/index with deep verification; original DSPy environment and data loader; separate writable per-run query cache |
| `decisionbench` | Pinned structured-decision dataset and exact shared ordered splits |
| `obliqbench` | Full official corpus/index preparation; identical explicit embedding device, attention, dtype and batch size for indexing and evaluation; one H200 encoder when CUDA is selected |
| `appworld` | Hash-locked isolated Python 3.11 runtime and verified public release; a writable copy of the prepared world per run |
| `terminalbench` | Harbor 0.22.0 and Docker inside a CPU VM; immutable task packages; every selected environment must start and execute a shell command during preparation |
| `taubench` | Pinned upstream banking-knowledge checkout and isolated runtime, official simulator, BM25 tools and grader |

TerminalBench defaults to **the same ordered 88 tasks as Della**, excluding
Mailman. `--terminalbench-scope all89` is a distinct, explicitly selected experiment.
The full preparation probe checks environment startup, not verifier success or
task quality. Actual runs retain Harbor task artifacts and native grades. Docker
and Singularity are different runtimes; their baseline results are not shared.

## Authenticate and configure secrets

Use Python 3.12 and the pinned Modal SDK. Keep plans and receipts **outside the
checkout**: launching requires a clean Git revision, exported with `git archive`.
Untracked files, local virtual environments, credentials and caches are excluded.

```bash
uv run --no-project --python 3.12 --with modal==1.6.0 modal token new
uv run --no-project --python 3.12 --with modal==1.6.0 modal profile current
```

Select the intended workspace in Modal's authentication flow before creating
resources. Create a named Modal Secret in that workspace, for example
`gepa-research`. Supply only the credentials needed by the selected experiment:

- `TYPESAFE_API_KEY` for `--controller-selection jev`.
- `OPENAI_API_KEY` for Tau's fixed simulator and required judge.
- `HF_TOKEN` when needed to access the pinned model or dataset repositories.

Pass Secret **names**, never values, with `--secret`. Cloud workers receive no
local `.env` file or Modal account token. A private per-run bearer key protects
both model gateways; it is injected into the allocated containers and is not
written into plans, receipts, runtime contracts, or request journals.

## Plan, prepare, then run

All examples below run from the repository root. The first command is offline;
`prepare` and `launch` create billable resources. This example uses a bounded
training-only optimizer pilot on DecisionBench:

```bash
uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli plan decisionbench \
  --run-id decisionbench-modal-pilot-01 \
  --mode optimizer-pilot --timeout 43200 \
  --secret gepa-research --output ../modal-runs/decisionbench-pilot.json \
  -- --condition both --controller-selection jev \
  --pilot-size 1 --pilot-proposals 1 --seed 0 --max-workers 1

uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli prepare \
  --plan ../modal-runs/decisionbench-pilot.json \
  --receipt ../modal-runs/decisionbench-preparation.json

uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli status \
  --receipt ../modal-runs/decisionbench-preparation.json
```

Wait for both the preparation allocation's exit code **0** and its saved worker
state **completed**. Preparation downloads and verifies both pinned model
checkpoints as well as the selected benchmark's data. It makes no solver,
proposer, Jev, simulator or judge requests. OBLIQ preparation performs the required
corpus embedding computation. Then launch exactly once:

```bash
uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli launch \
  --plan ../modal-runs/decisionbench-pilot.json \
  --prepared ../modal-runs/decisionbench-preparation.json \
  --receipt ../modal-runs/decisionbench-pilot-run.json
```

Replace `decisionbench` with any benchmark in the table. For OBLIQ, include the
intended `--embedding-device`, `--embedding-attention`, `--embedding-dtype`, and
`--embedding-batch-size` in the plan; defaults are CUDA, SDPA, bfloat16, and 1.
For TerminalBench, specify `--terminalbench-scope della88` or `all89` explicitly
when comparing task scopes. Native paths, model IDs and serving URLs are owned
by this backend and cannot be overridden through passthrough arguments.

For a full experiment, create a **new plan and run ID** with `--mode optimize`
and the reviewed scientific arguments after `--`. All shared runner variants are
available, including vanilla, FOREST/Jev, reflection minibatches and original
budgets. Preparation can be reused only when its source archive, benchmark,
encoder configuration and task scope match. Training pilot outputs never seed
the full experiment. A successful pilot must show a completed generated proposal
and its training evaluation, not merely a ready server or a running allocation.

## Resources and lifetime

The Qwen solver uses **2 H200 GPUs, 8 physical CPU cores and 128 GiB RAM**. The
DeepSeek proposer uses **4 H200 GPUs, 16 physical CPU cores and 768 GiB RAM**,
including the existing CPU-offload configuration. They retain separate pinned
serving locks, checkpoint hashes, tensor/expert parallelism and decoding flags.
Solver-only pilot/baseline modes do not allocate the proposer. Each active role
is reserved for the duration of the run; this backend does not autoscale between
individual requests. Check workspace quotas and pricing before full campaigns.

The driver requests 8 physical CPU cores and 128 GiB RAM, or 16 cores for
TerminalBench. CUDA OBLIQ additionally uses one H200 for retrieval. Models run in
gVisor sandboxes; TerminalBench's Docker daemon runs in a separate CPU VM.
Optional `--region` selects the allocation region and is recorded as infrastructure.

Modal sandboxes have a [24-hour maximum lifetime](https://modal.com/docs/guide/sandboxes).
`--timeout` defaults to 12 hours and includes startup. Runs can span multiple
allocations through the same persistent results volume, frozen images, source,
data, scientific budget and optimizer checkpoint. A new allocation is part of
the same logical experiment, not another replicate or an additional budget.

Before the earliest model deadline, the driver stops admitting new task groups.
`--drain-seconds` reserves time for in-flight work and evidence sealing; the default
is two hours (or one quarter of a shorter timeout). Tasks execute in ordered groups
bounded by the adapter's existing concurrency. Each completed group, including
scored-zero errors, is journaled before advancing. Training padding keeps separate
attempts even for repeated task IDs. Long validation and held-out repetitions can
therefore continue across allocations. TerminalBench uses a fresh native Harbor
job per group with the original task seeds and one trial per requested position.
No task's active world or process is transplanted between sandboxes.

A clean drain exits with worker state **paused**, seals its checkpoint snapshot,
and shuts down the owned model workers. It never freezes an unfinished winner
or treats partial held-out work as complete. Continue once with:

```bash
uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli resume \
  --receipt ../modal-runs/decisionbench-pilot-run.json \
  --output ../modal-runs/decisionbench-allocation-000001.json
```

For unattended rollover, run the bounded local supervisor:

```bash
uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli follow \
  --receipt ../modal-runs/decisionbench-pilot-run.json --max-allocations 4
```

**Both commands can create billable allocations.** The bound includes the supplied
allocation. `follow` must remain running locally; disconnecting it leaves the
cloud workers to complete or drain and does not create replacements. Its output
identifies every new receipt. After a disconnect, inspect the latest receipt and
existing intents before following again. Keep the original source tar and all
receipts. The supervisor never resets a claim or retries an unknown creation.

Continuation requires all previous workers to exit cleanly, complete evidence
seals, matching source/data/contracts and identical freshly verified material
model runtimes. Only tunnel addresses, bearer keys and allocation identities may
rotate. Started tasks or provider calls without sealed outcomes block continuation;
timeouts, preemptions and forced stops require review. A single task group or
proposal must finish inside the drain window. Increase the operational reserve
in a new plan when needed; this does not change model timeouts or scientific
budgets. Preparation remains a separately bounded stage and must complete before
launch. Receipts from before continuation support cannot be resumed automatically.

## Observe, stop and download

```bash
uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli status --receipt ../modal-runs/decisionbench-pilot-run.json

uv run --no-project --python 3.12 --with modal==1.6.0 \
  python -m examples.modal.cli download \
  --receipt ../modal-runs/decisionbench-pilot-run.json \
  --output ../modal-runs/decisionbench-pilot-evidence
```

Receipts contain the actual Modal allocation IDs and result-volume name. The
volume retains console logs, runtime identities, complete commands, optimizer
checkpoints, frozen validation winners, held-out results and shared baselines
when those stages are reached. Files are synchronized during execution. Download
waits for all owned allocations to be terminal and checks source and artifact
SHA-256 seals. Missing final seals are explicitly reported; preemption or a forced
kill does not become a successful, complete capture.

`stop --receipt <path>` explicitly terminates only allocations recorded in that
receipt. Normal driver completion/failure requests shutdown of its model workers.
If launch fails before any driver creation is attempted, known workers are stopped
automatically. Every creation intent is recorded before contacting Modal. An
unknown creation or termination outcome is retained for inspection; it is never
blindly retried. Preserve the original receipt when diagnosing such an outcome.

## Scientific and accounting boundaries

The shared runner verifies fresh authenticated serving evidence before evaluating
any task and records actual model bytes, software, GPU properties and launch
configuration. Provider requests carry that runtime identity and one logical
request ID across transport retries. The gateway seals a physical-attempt claim
before forwarding to vLLM. It can replay an identical sealed response after a
delivery failure, but refuses to reroll an unresolved started request. Definitive
HTTP errors and output-error retries retain the existing retry/seed policy.

Count `allocations/NNNNNN/solver/deliveries` and the corresponding proposer ledgers as physical model-attempt
authorities. CPU `provider-attempts.jsonl` entries marked `modal_delivery` describe
transport attempts and can include recovered delivery of the same response.
Harbor usage summaries separate those deliveries from physical inference.
Jev external/imported records still require deduplication; aggregate controller
decisions are not additional provider attempts. Tau's simulator/judge calls are
separate external-provider work. Missing cost or reasoning-token fields stay
unknown. Baselines are reused only by the shared runner's exact matching identity
inside this new run; never add a Della baseline or failed experiment prefix.
Per-allocation snapshots contain cumulative checkpoints and journals; do not sum
their copies as new work. Allocation-specific gateway ledgers count each physical
attempt once. The latest receipt retains the full allocation history, while each
prior status, log, runtime record and snapshot stays sealed in its own directory.

Offline verification does not establish live model quality, hardware availability,
all-task native readiness, or successful held-out completion. Review the saved
terminal evidence before reporting any of those outcomes.
