"""Define the small integration boundary shared by the benchmark examples."""

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import GEPAAdapter


@dataclass(frozen=True)
class BenchmarkModels:
    """Carry the fixed solver/proposer identities and resolved request settings."""

    solver_model: str
    proposer_model: str
    solver_api_base: str | None
    proposer_api_base: str | None
    solver_kwargs: dict[str, Any]
    proposer_kwargs: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkDefinition:
    """Supply an official task adapter and the exact ordered evaluation data.

    Records must have stable, globally unique ``id`` values. Every adapter
    output is a JSON object with ``elapsed_seconds`` and optional ``error``.
    The adapter may provide ``summarize_evaluation(records, evaluations)`` to
    aggregate benchmark-specific metrics across the repeated evaluation batches.
    Optional ``set_evaluation_context(split, repetition, seed)`` receives the
    split and fixed trial seed before execution, including held-out repetitions.
    """

    name: str
    adapter: GEPAAdapter
    seed_candidate: dict[str, str]
    trainset: list[dict[str, Any]]
    valset: list[dict[str, Any]]
    testset: list[dict[str, Any]]
    source: dict[str, Any]
    runtime: dict[str, Any]
    metric_name: str
    test_repetitions: int = 1
    component_kinds: dict[str, str] = field(default_factory=dict)
    max_candidate_proposals: int | None = None


BenchmarkBuilder = Callable[[Namespace, BenchmarkModels], BenchmarkDefinition]
