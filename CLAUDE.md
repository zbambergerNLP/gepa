# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is GEPA?

GEPA (Genetic-Pareto) is a Python framework for optimizing text components (AI prompts, code, instructions) using LLM-based reflection and Pareto-efficient evolutionary search. It works by evaluating candidates on minibatches, having an LLM read execution traces to diagnose failures, then proposing targeted mutations. Through iterative reflection, mutation, and Pareto-aware selection, it evolves high-performing variants with minimal evaluations (100-500 vs 10K+ for RL).

## Setup

We use **uv** for dependency management. All python executions must be done through uv.

```bash
uv sync --extra dev
```

## Build & Test

```bash
uv run pytest                        # run all tests
uv run pytest tests/test_optimize.py # run a single test file
uv run pytest -k "test_name"         # run a specific test by name
uv run ruff check src/               # lint
uv run ruff format src/              # format
uv run pyright src/                  # type check
```

## Code Style

- Linter/formatter: ruff (line length 120, double quotes, space indent)
- Type checking: pyright
- Python target: 3.10+
- No relative imports (enforced by ruff via `ban-relative-imports = "all"`)
- isort: `gepa` is first-party, `dspy` is third-party

## Architecture

### Two Main APIs

1. **`gepa.optimize()`** (`src/gepa/api.py`): Component-driven API for prompt optimization. Takes a seed_candidate dict, trainset, task_lm, and optional evaluator. Creates a `DefaultAdapter` internally.

2. **`optimize_anything()`** (`src/gepa/optimize_anything.py`): Goal-centric API for optimizing any text artifact (code, configs, SVGs). Takes a seed_candidate, evaluator function, objective string, and `GEPAConfig`. Supports three modes: single-task, multi-task, and generalization.

### Core Optimization Loop

The evolutionary search flows through these layers:

- **`GEPAEngine`** (`core/engine.py`): Orchestrates the iteration loop. Each iteration: propose candidates via `ReflectiveMutationProposer`, batch-evaluate, apply acceptance criterion, update Pareto frontier, optionally attempt merge, check stop conditions.

- **`ReflectiveMutationProposer`** (`proposer/reflective_mutation/reflective_mutation.py`): Samples a parent from the frontier, evaluates on a minibatch capturing execution traces (Actionable Side Information), builds a reflective dataset via the adapter, calls `ReflectionLM` to propose improved texts.

- **`MergeProposer`** (`proposer/merge.py`): Combines strengths of two Pareto-optimal candidates that excel on different task subsets.

- **`GEPAState`** (`core/state.py`): Holds all optimization state including candidates, scores, Pareto frontiers, lineage tracking, and an `EvaluationCache` that memoizes candidate-example pairs using SHA256 hashing.

### Key Abstractions

- **`GEPAAdapter`** (`core/adapter.py`): Protocol that users implement to integrate GEPA with their system. Two required methods: `evaluate()` (run candidates on data, return scores + traces) and `make_reflective_dataset()` (map component names to I/O/feedback for the reflection LM).

- **`ReflectionLM`** (`proposer/reflective_mutation/reflection_lm.py`): Protocol for the LLM that reads traces and proposes mutations. `StatelessReflectionLM` wraps a single LM call; extensible for stateful strategies.

- **Strategies** (`strategies/`): Pluggable algorithm components including `CandidateSelector` (Pareto, EpsilonGreedy, TopKPareto), `AcceptanceCriterion` (StrictImprovement, ImprovementOrEqual), `SamplingStrategy`, `SelectionStrategy`, and `BatchSampler`.

- **`LM`** (`lm.py`): Wrapper over LiteLLM with retries, truncation detection, and thread-safe cost tracking.

- **Callbacks** (`core/callbacks.py`): Event-driven instrumentation with 15+ event types. Integrates with experiment trackers (W&B, MLflow) via `ExperimentTracker`.

### Adapters

Built-in adapters in `src/gepa/adapters/` provide domain-specific integrations (DefaultAdapter, DSPy, LangChain, MCP, Generic RAG, ConfidenceAdapter, etc.). Each adapter implements the `GEPAAdapter` protocol.
