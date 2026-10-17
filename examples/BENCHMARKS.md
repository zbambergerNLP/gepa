# Shared benchmark experiments

The six benchmark entrypoints use `examples.common.benchmark_runner` for model
roles, optimization, validation selection, held-out evaluation, baseline reuse,
resume checks, and timing. Each adapter owns its official data, task execution,
scoring, and benchmark-specific constants.

| Benchmark | Entry module | Default task harness | Editable components |
|---|---|---|---|
| [HotPotQA](hotpotqa/) | `examples.hotpotqa.main` | Existing two-stage DSPy program and frozen Wiki-2017 BM25 retrieval | Existing program instruction modules |
| [Terminal-Bench 2.1](terminalbench/) | `examples.terminalbench.main` | Official Harbor/Terminus; 89 pinned tasks, 88 on Singularity/Della (Mailman excluded) | Unified initial instruction prompt |
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
Terminal-Bench needs pinned Harbor, Docker (or its official Singularity/Apptainer
backend on Della), and verified serving-runtime records. Follow the benchmark
preparation instructions before running.

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

To check the actual optimizer and proposer before a campaign, run:

```bash
uv run --no-sync python -m examples.decisionbench.main \
  --mode optimizer-pilot --condition both --pilot-size 1 --pilot-proposals 1 \
  --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --run-dir outputs/decisionbench-pilot
```

`optimizer-pilot` uses the first `--pilot-size` training examples for both
reflection and candidate selection. It forces reflection even when the seed
solves that subset and stops after `--pilot-proposals` optimization iterations
(one by default). Each iteration may contain several proposals when configured.
The campaign metric-call and epoch budgets do not apply to this bounded check.
Its `optimizer-pilot/<condition>/pilot-winner.json` records `training_score` and
`selection_split: train`. No validation/test examples or held-out baseline are
evaluated. A pilot is an execution check, not a held-out quality estimate.
Success requires at least one generated proposal to finish training evaluation
and reach acceptance or rejection. A non-improving evaluated proposal passes this
execution check; a skipped reflection or failed feedback mapper does not.
`optimizer-pilot-evidence.json` records sampled opportunities, candidate hashes,
evaluation scores and outcomes, including unfinished sibling jobs. This evidence
is checkpointed, checked on resume and bound to the saved pilot winner.

All methods receive identical initial prompts, task order, model settings and
evaluation budgets. Campaign selection uses validation scores; the selected prompt is saved
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

## Supported variants

Every entrypoint exposes this same condition matrix:

| `--condition` | Reflection method |
|---|---|
| `vanilla` | Canonical stateless GEPA reflection |
| `random` | Stateless GEPA with uniformly sampled semantic action/section constraints |
| `action` | Stateless GEPA with verbalized semantic action/section selection |
| `react_v2` or `forest` | FOREST Controller, Manifestor and editor; level 2 by default |
| `react_v2_random` | FOREST with a uniform-random Controller |
| `both` | `vanilla`, then `react_v2` |
| `all` | `vanilla`, `random`, `action`, `react_v2_random`, `react_v2` |

`all` expands the five historical method arms with the requested settings. It does
not run a Cartesian sweep of every option. Use a distinct `--run-dir` for each
additional configuration; changing a setting cannot resume a different ablation.
The `forest` alias uses the same `react_v2` directory and contract identity.

FOREST options apply to its two arms:

| Axis | Supported settings and constraints |
|---|---|
| `--reflection-level` | `0` (vanilla reflection), `1` (section selection), `2` (section and semantic action); random/Jev Controllers cannot use level 0 |
| `--controller-selection` | `verbalized`, `uniform_random`, `jev`; Jev requires level 2; `react_v2_random` always uses `uniform_random` |
| `--proposal-policy` | `real_edit` (default level 2: recover failed generation until a real edit or explicit exhaustion), `independent` (historical independent proposals) |
| `--editor-mode` | `single_call` or `react`; level-2 `real_edit` requires `single_call`; otherwise the default is `react` |
| `--edit-tool-set` | `broad` (insert/delete/replace/move), `minimal` (insert/delete); `single_call` requires `broad` |
| `--react-max-iterations`, `--react-max-tool-calls` | Optional bounds for the `react` observation loop |

The existing `react_v2` condition name is retained for artifact continuity. The
effective editor is recorded separately: the current default level-2 recovery
policy uses `single_call`. To reproduce the independent ReAct editor arm, pass
`--proposal-policy independent --editor-mode react`. Level 1 has no Manifestor;
the stateless `action` condition is a separate method, not reflection level 1.

`--module-selector round_robin` is the shared default; `all` edits all modules.
The stateless `random` and `action` arms require one module per proposal, so
multi-component benchmarks reject those arms with `--module-selector all`
before any condition starts. They use round-robin selection instead.
Adaptive selection uses `--condition react_v2 --module-selector controller` and
requires level 2 with a verbalized or Jev Controller. It supplies all modules'
training evidence to one joint component/section/action decision. Combining this
flag with `both` or `all` is rejected because the vanilla and random arms do not
implement adaptive component selection. Single-component benchmarks use the same
path but have no module choice to ablate.

Jev uses its separately pinned `jev-1.13.0` controller, while the configured
proposer still supplies the Manifestor and editor. Install `--extra jev` and set
`TYPESAFE_API_KEY` (or use the existing `GEPA_JEV_HANDOFF_DIR` protocol). Its
requests and policy are journaled alongside the optimizer run.

Shared GEPA search ablations work with every method:

| Axis | CLI |
|---|---|
| Parent selection | `--candidate-selection pareto` (default), `current_best`, `epsilon_greedy` (0.1), `top_k_pareto` (5) |
| Admission | `--acceptance strict_improvement` (default) or `improvement_or_equal` |
| Proposal sampling | `--sampling-strategy single` (default), `same_parent`, `independent`, `pxn`; set `--proposal-count N`, and `--parent-count P` for `pxn` |
| Proposal selection | `--proposal-selection all_improvements` (default), `best_improvement`, `top_k`; set `--proposal-top-k K` for `top_k` |
| Merge | `--merge` enables the established five-invocation merge configuration |

Terminal-Bench's epoch protocol caps proposal opportunities. Multi-proposal groups
must divide that pinned cap exactly; unsupported group sizes fail before model
work. The run contract records proposals per iteration and the corresponding
iteration limit, so increasing the group size does not multiply the budget.
Metric-call budgets retain the engine's existing whole-iteration stopping policy.

All six use per-example (`instance`) frontiers; objective/hybrid/cartesian
frontiers are not exposed because these benchmarks do not share a defined
multi-objective score schema. Only implemented variants are exposed. Retired RLM runners and the old
HotPotQA-specific one-stage harness are not restored by this shared runner.
Harness changes are separate benchmark definitions, not optimizer flags.

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
