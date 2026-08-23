"""Tests for task-model inference used by provider prompt templates."""

from gepa.api import _template_consumer_model
from gepa.strategies.document_template import infer_template_family


class StudentAdapter:
    """Expose the prompt consumer through a common adapter attribute."""

    student_model = "hosted_vllm/Qwen3.8"


def test_custom_adapter_exposes_consumer_for_auto_template_inference() -> None:
    model = _template_consumer_model(None, StudentAdapter(), None)

    assert infer_template_family(model) == "alibaba"


def test_explicit_template_model_overrides_adapter_metadata() -> None:
    model = _template_consumer_model(None, StudentAdapter(), "anthropic/claude-sonnet-4-6")

    assert infer_template_family(model) == "anthropic"
