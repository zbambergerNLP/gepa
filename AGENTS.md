# GEPA

GEPA (Genetic-Pareto) is a Python framework for optimizing text components (AI prompts, code, instructions) using LLM-based reflection and Pareto-efficient evolutionary search.

## Agent Skill

This repository ships an **Agent Skill** at `.claude/skills/gepa-optimize-anything/`. If you are a coding agent and the user wants to **auto-optimize, tune, or search over any scorable text artifact** (a prompt, program/code, config, regex/SQL, or agent scaffold) with `gepa.optimize_anything`, read `.claude/skills/gepa-optimize-anything/SKILL.md` first and follow it. It is auto-discovered by Claude Code and by other agents that read `.claude/skills/` (Cursor, VS Code/Copilot, Codex, Gemini CLI). To install it standalone in another project: `/plugin marketplace add gepa-ai/gepa` then `/plugin install gepa-optimize-anything@gepa`.

## Setup

We use **uv** for dependency management. The project uses setuptools as the build backend. All python executions must be done through uv.

```bash
uv sync --extra dev
```

## Project Structure

- `src/gepa/` — main package source
  - `core/` — optimization loop, state, evaluation
  - `proposer/` — candidate proposal and mutation logic
  - `adapters/` — integration adapters (DSPy, RAG, MCP, etc.)
  - `strategies/` — batch sampling and candidate selection
  - `logging/` — experiment tracking and logging
- `tests/` — pytest test suite
- `docs/` — mkdocs documentation site

## Build & Test

```bash
uv run pytest
uv run ruff check src/
uv run ruff format src/
uv run pyright src/
```

## Benchmark comparisons

Within a benchmark, all ablations and model arms must use the same pinned data
and exact ordered train/validation/test examples. Preserve source revisions,
content hashes or immutable task refs, and split membership, not just counts.
Reject data or split drift before optimization or evaluation. Method, text scope,
and budget changes must not resample splits. Apply this requirement to future
benchmarks too; use a shared manifest or ordered-record fingerprints such as
`examples.common.react_v2.benchmark_data_identity` and enforce the recorded identity.

Evaluate each ablation's validation-selected winner after that ablation finishes.
Freeze that winner before testing; test scores must not affect later prompts,
selection, budgets, or settings. Runtime calibration uses training examples only.

## Code Style

- Linter/formatter: ruff (line length 120, double quotes, space indent)
- Type checking: pyright
- Python target: 3.10+
- No relative imports (enforced by ruff)
