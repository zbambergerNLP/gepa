# Terminal-Bench v3 through Harbor

This harness evaluates GEPA candidates with the official Terminal-Bench verifier. It has no heuristic, synthetic, Hugging Face, or command-overlap fallback: every score comes from a pinned Harbor Docker trial.

## Reproducibility contract

- Harbor: `0.22.0`, installed as a separate CLI because Harbor requires Python 3.12+ while GEPA supports Python 3.10+.
- Dataset tag: `terminal-bench/terminal-bench@3.0.0`; `@latest` is never used. Runtime jobs resolve by the tag's immutable content hash below, so a later tag change cannot alter an experiment.
- Registry content hash: `sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3`.
- Source: [`harbor-framework/terminal-bench`](https://github.com/harbor-framework/terminal-bench) tag `v3.0.0`, commit `2b0442c3c583b710ca8da14c8e601b99f2f1f244`.
- Task refs and splits: [`terminalbench-v3-manifest.json`](terminalbench-v3-manifest.json), captured from the official Harbor registry with Harbor 0.22.0.

The registry does not publish train/validation/test partitions. The checked-in manifest hash-orders all 74 full task IDs by `SHA-256("gepa-terminalbench-v3-split-v1" + NUL + task_id)` and uses Hamilton largest-remainder apportionment for a 40/30/30 allocation: 30 train, 22 validation, and 22 held-out test tasks. Harness validation recomputes the policy, rejects overlaps, and requires every pinned task exactly once.

## Initial optimization target

The candidate has exactly one component: `instruction_prompt`, the Terminus 2 instruction/system prefix. The standard Terminus JSON response contract and tmux terminal tool remain fixed. Task- and agent-provided skills and MCP servers are disabled by `PromptedTerminus`, so this first harness does not conflate prompt optimization with skill or tool optimization.

Turn count is not capped (`max_turns` is intentionally omitted). Long-horizon trajectories may continue until each pinned task's Harbor agent timeout. A whole-job subprocess timeout is optional and disabled by default.

Every candidate evaluation gets a unique immutable directory containing:

- the rendered candidate prompt;
- the exact Harbor job JSON;
- Harbor stdout/stderr;
- the complete Harbor job and trial artifacts;
- all `agent/trajectory*.json` ATIF files, rewards, and structured errors.

Results are keyed back to the requested batch by full task ID, never by directory order.

## Setup

Install the exact Harbor version and start Docker:

```bash
uv tool install --force 'harbor==0.22.0'
harbor --version
docker info
```

The harness fails before spending model calls if Harbor is absent, the version is not exactly `0.22.0`, Docker is absent, or the Docker daemon is unreachable.

## Configure a run

Student and proposer models are deliberately separate required arguments. Use the provider-accurate model IDs for the planned Qwen 3.8 student and DeepSeek V4 Flash proposer in your environment:

```bash
uv run python -m examples.terminalbench.main \
  --condition react_v2 \
  --student-model '<qwen-3.8-model-id>' \
  --proposer-model '<deepseek-v4-flash-model-id>' \
  --max-metric-calls 400 \
  --run-dir outputs/terminalbench/react-v2 \
  --harbor-work-dir outputs/terminalbench/harbor-react-v2
```

Use `--condition vanilla` for stock GEPA reflection and `--condition react_v2` for Controller → Manifestor → ReAct V2 (`action` remains a compatibility alias). Prompt sections are inferred from the student model prefix by default. Keep model IDs, manifest, split prefixes, metric-call budget, minibatch size, concurrency, and random seed identical when comparing conditions. `--edit-tool-set minimal|broad` exposes the intended text-operator ablation without changing Harbor evaluation, while `--reflection-level 1|2` controls whether the Manifestor semantic-action layer is active.

Each GEPA run directory contains `terminalbench-run-contract.json`. Resuming with a different model, split, template, tool basis, reflection level, concurrency, or other material setting fails before evaluation, preventing ablation results from sharing stale GEPA state.

For local infrastructure validation only, limit both partitions explicitly; these prefixes remain deterministic:

```bash
uv run python -m examples.terminalbench.main \
  --condition react_v2 \
  --student-model '<student-model-id>' \
  --proposer-model '<proposer-model-id>' \
  --max-metric-calls 2 \
  --train-limit 1 \
  --val-limit 1 \
  --run-dir outputs/terminalbench/smoke-gepa \
  --harbor-work-dir outputs/terminalbench/smoke-harbor
```

The runner does not score the held-out test split automatically. Test evaluation should happen once, after the experiment configuration and candidate-selection rule are frozen.

## Tests

Normal tests are offline and mock the Harbor subprocess boundary:

```bash
uv run pytest tests/test_terminal_bench_adapter.py -m 'not smoke'
```

The opt-in smoke marker launches one real pinned Harbor task only when all required environment variables are present:

```bash
GEPA_TERMINALBENCH_SMOKE=1 \
GEPA_TERMINALBENCH_STUDENT_MODEL='<student-model-id>' \
uv run pytest tests/test_terminal_bench_adapter.py -m smoke
```
