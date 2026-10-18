# OpenRouter benchmarks on Della

Stage the Python driver, datasets and native task environments on an
internet-connected `della-vis` host. Run full campaigns in CPU allocations with
`module load proxy/default`: [OpenRouter is on the supported API allowlist](https://researchcomputing.princeton.edu/support/knowledge-base/ports-and-proxydefault).
OpenRouter supplies model inference. Use a
new checkout, environment and output directory: the GPU campaigns' frozen
sources, journals, baselines and queued allocations remain separate.

Princeton supports [Apptainer on Della](https://researchcomputing.princeton.edu/support/knowledge-base/apptainer).
Docker is not installed on the prepared visualization host. A working API key
does not establish container or benchmark readiness.

## Source and environments

Fetch the reviewed API branch into a fresh scratch directory through the
existing authenticated SSH session. Record the exact commit and tracked-file
hashes. Do not use the old GPU submission scripts or copy a mutable checkout
over a frozen campaign source.

Set `GEPA_API_ROOT` to the new preparation directory, containing `source/`,
`venv/`, `assets/`, `cache/` and `.secrets/`:

```bash
export GEPA_API_ROOT=/absolute/scratch/path/to/api-preparation
export UV_PROJECT_ENVIRONMENT="$GEPA_API_ROOT/venv"
export UV_CACHE_DIR="$GEPA_API_ROOT/cache/uv"
export HF_HOME="$GEPA_API_ROOT/cache/huggingface"
export DSPY_CACHEDIR="$GEPA_API_ROOT/cache/dspy"
export PYTHONDONTWRITEBYTECODE=1
cd "$GEPA_API_ROOT/source"
uv sync --locked --python 3.12 \
  --extra full --extra decisionbench --extra obliqbench --extra wiki17 --extra jev \
  --group hotpotqa-task-program
```

Keep AppWorld, Tau, Harbor and the official SWE grader in their separate pinned
environments. Unset `UV_PROJECT_ENVIRONMENT` when syncing an upstream project;
otherwise its dependencies can replace the shared driver's environment.

Provision only `OPENROUTER_API_KEY` in `.secrets/openrouter.env`, with directory
mode 700 and file mode 600. Use the entrypoint's `--env-file` argument. Do not
copy unrelated credentials or print authentication responses. Public catalog
requests and `GET /api/v1/key` suffice for transport/authentication checks.

## Asset readiness

Reuse immutable verified assets through explicit paths. Put new query caches,
task workspaces, receipts and outputs under the new preparation directory.

| Benchmark | Evidence required before an API campaign |
|---|---|
| HotPotQA | Pinned fullwiki revision, ordered 150/300/300 split, patched DSPy identity, complete Wiki-2017 BM25 integrity check and a training-query retrieval. Use `--wiki17-dir`. |
| DecisionBench | Exact canonical Parquet checksum and ordered splits. Use `--data-file`. |
| AppWorld | Complete pinned data, separate locked runtime, native training-task initialization, shell execution and official evaluator output. Use `--appworld-root` and `--appworld-python`. |
| OBLIQ | All five frozen corpora and a complete matching Qwen embedding index. Use `--data-dir` and `--index-dir`; device, attention, dtype and batch size must match its identity. A queued builder is not a ready index. Do not duplicate an existing owned builder. |
| Tau | Pinned upstream source, data and isolated runtime; pass the native offline runtime tests with the actual `TAU2_SOURCE`. Use `--tau-source`. |
| Terminal-Bench | Pinned Harbor 0.22.0, verified offline bundle, task images, compatible fakeroot helper and passing native checks for all 88 Della tasks. Use the flags below. Mailman remains excluded. |
| SWE-bench Live | Pinned data, isolated official grader, pinned upstream SWE-agent and SWE-ReX payload, verified AMD64 OCI/SIF images, and native training-task agent/tool checks plus negative/reference grading through Apptainer. Use `--container-backend apptainer --swebench-live-root <assets> --swe-agent-root <agent>`, exactly four CPUs and 16 GiB in a single-node CPU allocation, and `--max-workers 1`. The parent uses `proxy/default`; task instances stay offline. See the [preparation and isolation guide](../../examples/swebench_live/README.md#apptainer-on-della). |

Keep disk headroom when staging images. SWE Live alone references 1,895 distinct
images; do not infer that shared scratch can hold them from a successful small
download. Record preparation failures and partial caches as well as successes.

## Terminal-Bench networking

The existing `remote/with_private_network.sh` intentionally has no external
network. API runs instead use `api_network.py`. It creates private user/network namespaces so native QEMU
tasks can retain port 22, while Harbor and its containers share loopback.

An owner-only Unix socket carries an HTTPS CONNECT tunnel to `openrouter.ai:443`.
TLS stays end to end. On allocated compute nodes it forwards CONNECT through
the cluster's loaded `proxy/default`, and preserves any proxy rejection. On vis
it uses the host's direct internet connection. It rejects unallocated compute
nodes and other destinations, and does not alter host networking. The wrapper
has no key, provider fallback or model retry logic. Pre-stage task packages and dependencies; the tunnel
does not provide general internet access to task environments.

Verify public catalog access, authenticated key access, private port binding and
command exit propagation through the wrapper before any model calls. Then run
the complete native readiness probe through this same wrapper and pinned Harbor.
The older GPU allocation's readiness receipt is historical evidence, not a test
of this API transport.

After models, routes and a fresh campaign have been chosen and authorized, the
Terminal-Bench command has this shape (**it starts paid work**):

```bash
module load proxy/default  # inside the CPU allocation, never on a login/vis host
export PATH="$GEPA_API_ROOT/assets/harbor-0.22.0/bin:$PATH"
uv run --no-sync python scripts/della/api_network.py -- \
  uv run --no-sync python -m examples.terminalbench.main \
  --api-profile /absolute/path/to/approved-profile.json \
  --env-file "$GEPA_API_ROOT/.secrets/openrouter.env" --allow-api-calls \
  --container-runtime singularity --task-scope della88 \
  --harbor-executable "$GEPA_API_ROOT/assets/harbor-0.22.0/venv/bin/harbor" \
  --offline-task-bundle /absolute/path/to/verified/bundle.json \
  --singularity-image-cache /absolute/path/to/verified/image-cache \
  --condition both --max-workers 1 --run-dir /absolute/path/to/new-campaign
```

Use bounded host resources and sequential native trials. Resource ceilings must
accommodate the original task requirements; do not lower task resources to make
a probe pass. Start no model server or GPU allocation for hosted inference.

The [API profile guide](../../examples/local_api/README.md) describes explicit
model selection, provider compatibility, costs and the separate paid-call gate.
Infrastructure checks make no paid model requests and establish no optimization
or held-out quality. A future training-only live pilot must verify the chosen
routes and tools before the full campaign starts.
