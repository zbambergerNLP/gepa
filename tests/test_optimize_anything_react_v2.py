"""ReAct V2 configuration parity for the legacy optimize_anything launcher."""

from __future__ import annotations

from typing import Any

import pytest

import gepa.gepa_launcher as launcher
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.strategies.document_template import TEMPLATE_FAMILIES, MalformedDocumentError


class _StrategyCapturedError(Exception):
    """Stop after observing strategy construction without running optimization."""


def _score(_candidate: dict[str, str], _example: object | None = None) -> float:
    """Provide a network-free evaluator that should not run in construction tests."""
    return 0.0


def test_reflection_config_builds_react_v2_like_optimize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward every public ReAct V2 field and derive the deterministic Manifestor."""
    captured: dict[str, Any] = {}

    def capture_strategy(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _StrategyCapturedError

    monkeypatch.setattr(launcher, "ThreeRoleReflectionLM", capture_strategy)
    template = TEMPLATE_FAMILIES["openai"]["system_prompt"]
    seed = {"system_prompt": template.render({section: section for section in template.sections})}

    with pytest.raises(_StrategyCapturedError):
        optimize_anything(
            seed_candidate=seed,
            evaluator=_score,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2),
                reflection=ReflectionConfig(
                    reflection_lm="openai/gpt-4o-mini",
                    reflection_lm_kwargs={"max_tokens": 64, "temperature": 0.8},
                    reflection_level=2,
                    edit_tool_set="minimal",
                    component_kinds={"system_prompt": "system_prompt"},
                    template_family="auto",
                    template_model="openai/gpt-5",
                ),
            ),
        )

    assert captured["level"] == 2
    assert captured["edit_tool_set"] == "minimal"
    assert captured["component_kinds"] == {"system_prompt": "system_prompt"}
    assert captured["template_family"] == "openai"
    assert captured["proposer_model"] == "openai/gpt-4o-mini"
    assert captured["base_lm"].model == "openai/gpt-4o-mini"
    assert captured["manifestor_lm"].model == "openai/gpt-4o-mini"
    assert captured["manifestor_lm"].completion_kwargs == {"max_tokens": 64, "temperature": 0.0}


def test_auto_template_family_reads_adapter_consumer_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Infer from the adapter when template_model is not explicitly supplied."""
    captured: dict[str, Any] = {}
    adapter_class = launcher.OptimizeAnythingAdapter

    def named_adapter(**kwargs: Any) -> Any:
        adapter = adapter_class(**kwargs)
        adapter.student_model = "dashscope/qwen-max"
        return adapter

    def capture_strategy(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _StrategyCapturedError

    monkeypatch.setattr(launcher, "OptimizeAnythingAdapter", named_adapter)
    monkeypatch.setattr(launcher, "ThreeRoleReflectionLM", capture_strategy)

    with pytest.raises(_StrategyCapturedError):
        optimize_anything(
            seed_candidate={"system_prompt": "construction stops before validation"},
            evaluator=_score,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2),
                reflection=ReflectionConfig(
                    reflection_lm=lambda _prompt: "unused",
                    reflection_level=1,
                    template_family="auto",
                ),
            ),
        )

    assert captured["template_family"] == "alibaba"


def test_explicit_reflection_strategy_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an explicit strategy while honoring its own seed validator."""

    class ExplicitStrategy:
        def __init__(self) -> None:
            self.validated: list[dict[str, str]] = []

        def validate_candidate(self, candidate: dict[str, str]) -> None:
            self.validated.append(candidate)

    explicit_strategy = ExplicitStrategy()

    def reject_auto_strategy(**_kwargs: Any) -> None:
        raise AssertionError("automatic ThreeRole construction must not run")

    def capture_proposer(**kwargs: Any) -> None:
        assert kwargs["reflection_strategy"] is explicit_strategy
        raise _StrategyCapturedError

    monkeypatch.setattr(launcher, "ThreeRoleReflectionLM", reject_auto_strategy)
    monkeypatch.setattr(launcher, "ReflectiveMutationProposer", capture_proposer)

    with pytest.raises(_StrategyCapturedError):
        optimize_anything(
            seed_candidate={"system_prompt": "free-form seed owned by the explicit strategy"},
            evaluator=_score,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2),
                reflection=ReflectionConfig(
                    reflection_lm=None,
                    reflection_strategy=explicit_strategy,  # type: ignore[arg-type]
                    reflection_level=2,
                    template_family="openai",
                ),
            ),
        )

    assert explicit_strategy.validated == [{"system_prompt": "free-form seed owned by the explicit strategy"}]


def test_auto_strategy_validates_structured_seed_before_evaluation() -> None:
    """Reject malformed sectioned seeds before the evaluator spends a call."""
    evaluator_called = False

    def evaluator(_candidate: dict[str, str], _example: object | None = None) -> float:
        nonlocal evaluator_called
        evaluator_called = True
        return 0.0

    with pytest.raises(MalformedDocumentError, match="migrate_document"):
        optimize_anything(
            seed_candidate={"system_prompt": "unstructured"},
            evaluator=evaluator,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2),
                reflection=ReflectionConfig(
                    reflection_lm=lambda _prompt: "unused",
                    reflection_level=1,
                    template_family="generic",
                ),
            ),
        )

    assert evaluator_called is False


def test_auto_family_migration_error_names_adapter_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the resolved adapter model when automatic family validation fails."""
    adapter_class = launcher.OptimizeAnythingAdapter

    def named_adapter(**kwargs: Any) -> Any:
        adapter = adapter_class(**kwargs)
        adapter.student_model = "dashscope/qwen-max"
        return adapter

    monkeypatch.setattr(launcher, "OptimizeAnythingAdapter", named_adapter)

    with pytest.raises(MalformedDocumentError, match=r"'alibaba'.*dashscope/qwen-max"):
        optimize_anything(
            seed_candidate={"system_prompt": "unstructured"},
            evaluator=_score,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2),
                reflection=ReflectionConfig(
                    reflection_lm=lambda _prompt: "unused",
                    reflection_level=1,
                    template_family="auto",
                ),
            ),
        )
