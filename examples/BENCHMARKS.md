# Shared benchmark experiments

The six benchmark entrypoints use `examples.common.benchmark_runner` for model
roles, optimization, validation selection, held-out evaluation, baseline reuse,
resume checks, and timing. Each adapter owns its official data, task execution,
scoring, and benchmark-specific constants.

| Benchmark | Entry module | Default task harness | Editable components |
|---|---|---|---|
| [HotPotQA](hotpotqa/) | `examples.hotpotqa.main` | Existing two-stage DSPy program and frozen Wiki-2017 BM25 retrieval | Existing program instruction modules |
| [Terminal-Bench 2.1](terminalbench/) | `examples.terminalbench.main` | Official Harbor/Terminus harness and pinned 89-task manifest | Unified initial instruction prompt |
| [OBLIQ-Bench](obliqbench/README.md) | `examples.obliqbench.main` | One query rewrite, fixed Qwen embedding retriever, top 1,000 | Query-rewriting system prompt |
| [DecisionBench](decisionbench/README.md) | `examples.decisionbench.main` | Official structured decision/probability-vector protocol | System prompt |
| [AppWorld](appworld/README.md) | `examples.appworld.main` | Published ReAct code-agent pattern and official state evaluator | System prompt |
| [Tau banking knowledge](taubench/README.md) | `examples.taubench.main` | Official text agent, user simulator, BM25 tools and reward evaluator | Agent system prompt |

The default **solver** is `hosted_vllm/Qwen/Qwen3.8-27B`; the default **proposer**
is `hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash`. Exact checkpoint revisions,
sampling parameters, thinking budgets and bounded transport retries come from
`examples/common/experiment_models.py` and `model_settings.py`. FOREST uses the
DeepSeek proposer for its Controller, Manifestor and editor roles. Fixed benchmark
services, such as the OBLIQ embedding retriever and Tau user simulator/judge, are
identified separately in each benchmark's documentation and run contract.

## Install and prepare

Use Python 3.12. Install the core environment and optional benchmark dependencies:

```bash
uv sync --locked --python 3.12 --extra full --extra decisionbench --extra obliqbench \
  --group hotpotqa-task-program
```

HotPotQA additionally needs its existing `wiki17` extra and prepared frozen index.
Terminal-Bench needs its pinned Harbor/Docker setup and verified serving-runtime
records. Follow their benchmark preparation instructions before running.

OBLIQ corpora require explicit preparation; corpus embedding/indexing is a separate
preprocessing step. AppWorld and Tau run official environments in separate pinned
Python environments because their upstream dependencies conflict with the main
model stack. Their README setup commands prepare those environments. Tau also
requires credentials for its fixed official user simulator and judge.

## Run a benchmark

With the documented benchmark prerequisites ready, the common command shape is:

```bash
uv run --no-sync python -m examples.decisionbench.main \
  --mode pilot --pilot-size 3 \
  --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --run-dir outputs/decisionbench

uv run --no-sync python -m examples.decisionbench.main \
  --condition both \
  --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --run-dir outputs/decisionbench
```

Replace the module and add the benchmark's preparation/runtime arguments as needed.
These commands make model calls. `--mode pilot` evaluates only training examples
with the unoptimized harness. `--mode optimize` is the default and runs vanilla
GEPA followed by FOREST (`react_v2`) when `--condition both` is selected.
`--mode baseline` evaluates the unoptimized seed on held-out data.

Both methods receive identical initial prompts, task order, model settings and
evaluation budgets. Selection uses validation scores; the selected prompt is saved
to `frozen-winner.json` before any held-out evaluation. The seed baseline is shared
between matched runs under the sibling `benchmark-baselines/` directory. Old
campaign artifacts are never silently interpreted as shared-runner checkpoints.

The common optimization budget is 6,871 metric calls and a reflection minibatch
of three. These are experiment settings inherited from HotPotQA, not official
defaults claimed for every benchmark. Terminal-Bench retains its epoch-based
protocol and explicit output cap. Each adapter specifies its held-out repetitions;
Tau reports all-k reliability (`pass^k`), while Terminal-Bench reports repeated
single-attempt rewards. Benchmark READMEs distinguish official evaluation splits
from the fixed derived splits needed for prompt optimization.

## Evidence and timing

Every condition records its complete data/model/runtime contract, candidate history,
frozen winner, held-out repetitions and result summary. Resume rejects changes to
ordered data, prompts, settings, source code or dependency identity. Concurrent
writers cannot run against the same optimization directory. Complete repetitions
are reused after interruption; partial repetitions must be rerun.

`task-timings.jsonl` records per-example elapsed seconds. Summary timing includes
attempt count, mean, median, p95 and failures. Batch wall time and tasks/hour are
reported separately: dividing batch time by task count under parallel execution
does not measure single-task latency. Each adapter documents whether environment
startup is included; offline corpus indexing is excluded. Solver physical-attempt
logs are retained alongside the run, including retry/backoff information.

Timing reports are measurements of the selected model, endpoint, hardware,
concurrency and task subset. No model-backed average latency or quality result is
implied by the offline integration tests. Use a training-only pilot for those
measurements before starting an optimization campaign.
