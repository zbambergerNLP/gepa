"""Apply Terminal-Bench's approved output budget without expanding context."""

from typing import Any

from examples.common.experiment_models import (
    experiment_decoding,
    experiment_model_info,
    validate_experiment_model_pair,
)

MAX_OUTPUT_TOKENS = 32_768


def terminalbench_decoding(model: str, *, agentic: bool = True) -> dict[str, int | float | str]:
    """Keep role-specific sampling while setting the combined output ceiling.

    Args:
        model: Campaign model identifier.
        agentic: Whether this role iteratively uses tools.

    Returns:
        Completion settings; the cap includes reasoning and final output.
    """
    return {**experiment_decoding(model, agentic=agentic), "max_tokens": MAX_OUTPUT_TOKENS}


def terminalbench_model_info(model: str) -> dict[str, Any]:
    """Expose the output ceiling to Harbor while preserving serving capacity.

    Args:
        model: One of the two homogeneous campaign model identifiers.

    Returns:
        Context and cost metadata with the Terminal-Bench output ceiling.
    """
    validate_experiment_model_pair(model, model)
    info = experiment_model_info(model)
    assert info is not None
    return {**info, "max_output_tokens": MAX_OUTPUT_TOKENS}


def terminalbench_limits(model: str) -> dict[str, Any]:
    """Describe the practical budget and the training-only review policy.

    Args:
        model: Campaign model identifier.

    Returns:
        Serializable policy shared by every role and both optimization budgets.
    """
    return {
        "context_tokens": terminalbench_model_info(model)["max_input_tokens"],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "output_includes_reasoning": True,
        "budget_basis": "approved_practical_budget_not_provider_maximum",
        "cap_review_split": "train",
        "automatic_cap_increases": False,
        "harbor_model_names": "short_registry_alias_full_request_id",
    }
