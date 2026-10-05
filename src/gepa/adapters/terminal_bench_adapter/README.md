# Terminal-Bench 2.1 adapter

Run this benchmark through `python -m examples.terminalbench.main` and the
[shared benchmark runner](../../../../examples/BENCHMARKS.md). The
[benchmark guide](../../../../examples/terminalbench/README.md) documents its
commands, budgets, evaluation, and per-task timing.

## Harness and scoring

This module maintains a Harbor port of GEPA's
[upstream TerminusAdapter](https://github.com/gepa-ai/gepa/blob/4f1613773d0c13c8f1551543a801b299bd8acf73/src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py).
It runs Harbor **0.22.0** with the `PromptedTerminus` agent and a pinned
89-task Terminal-Bench 2.1 manifest: 30 training, 19 validation, and 40 test
tasks. The deterministic optimization split is recorded in
`examples/terminalbench/terminalbench-v2.1-manifest.json` along with immutable
task references and content hashes.

The suite edits one unified initial instruction prompt. The `system_prompt`
scope preserves Terminus's actual initial user-message role. Prompt documents,
context-summary instructions, and skill files are materialized by the adapter;
auxiliary instructions and skills stay fixed in this benchmark configuration.

Task execution uses the official verifier reward, ATIF messages, and verifier
diagnostics. The adapter validates task identities and complete result evidence.
Official task failures retain their official scores; infrastructure errors and
missing evidence abort execution. The shared runner freezes the validation
winner before three independent held-out repetitions and evaluates the same
unoptimized baseline. Test traces never enter reflection.

## Prepare Harbor and Docker

Install the pinned CLI into an isolated tool environment, and make sure its
executable and Docker daemon are available on the benchmark host:

```bash
uv tool install harbor==0.22.0
harbor --version
docker info
```

The runner accepts `--harbor-executable` and `--docker-executable` when explicit
paths are needed. Harbor receives the repository path so it can import the
custom `examples.terminalbench.terminus_agent.PromptedTerminus` agent.

## Record model serving runtimes

On the Linux GPU host, use the prepared serving environment's interpreter to
launch each model through `examples.terminalbench.runtime`. Pass the pinned
model identifier, verified checkpoint directory, runtime record destination,
and port. Forward the model's serving flags after `--`; dtype, KV-cache dtype,
tensor parallelism, and data parallelism must be explicit.

```bash
uv run --no-project --python "$VLLM_PY" python -m examples.terminalbench.runtime --help
```

Use Qwen3.8-27B for the solver and DeepSeek-V4.1-Flash for the proposer. Their
checkpoint revisions are defined in `examples/common/experiment_models.py`.
The [Della preparation tools](../../../../scripts/della/README.md) prepare the
pinned model and serving environments. Serving settings must match the selected
model and hardware; use a training pilot to measure the resulting performance.

The launcher verifies checkpoint bytes, records installed runtime packages,
GPU hardware, precision, parallelism, and forwarded arguments, then executes
vLLM with the same interpreter and process identity. It manages the served
model name, loopback address, and port. Wait for both servers to become ready.

Pass `--runtime-record` and `--proposer-runtime-record` plus their matching
`--solver-api-base` and `--reflection-api-base` to benchmark commands. The runner
checks the current hostname, Linux boot ID, PID, process start time, and ownership
of each listening socket before Harbor starts. Records must be regenerated when
servers restart. Material runtime changes require a fresh run directory.

## Evidence

Each official trial contributes its own start-to-finish duration, including
setup, execution, verification, and teardown. The shared runner reports mean,
median, and p95 task latency separately from batch throughput. Harbor provider
attempts retain token usage, retry, and output-limit evidence. To aggregate
those token logs:

```bash
uv run --no-sync python -m examples.terminalbench.token_usage outputs/terminalbench \
  --output outputs/terminalbench/token-usage-summary.json
```
