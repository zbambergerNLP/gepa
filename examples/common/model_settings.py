"""Share the reviewed HotPotQA role budgets across benchmark model clients."""

from examples.common.experiment_models import (
    DEEPSEEK_V4_1_FLASH_MODEL,
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_SOLVER_MODEL,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_request_overrides,
    validate_experiment_model_pair,
)
from examples.common.provider_retries import provider_retry_kwargs
from gepa.strategies.forest_constants import OPTIMIZER_ROLE, SOLVER_ROLE

SCIENTIFIC_REQUEST_SEED = 0
REQUEST_TIMEOUT_SECONDS = 3600
SOLVER_MAX_TOKENS = 65_536
SOLVER_THINKING_TOKENS = 32_768
OPTIMIZER_MAX_TOKENS = {
    QWEN3_8_27B_MODEL: 32_768,
    DEEPSEEK_V4_1_FLASH_MODEL: 131_072,
}
OPTIMIZER_THINKING_TOKENS = {
    QWEN3_8_27B_MODEL: 24_576,
    DEEPSEEK_V4_1_FLASH_MODEL: 98_304,
}


def validate_benchmark_model_pair(solver_model: str, proposer_model: str) -> None:
    """Accept the shared Qwen/DeepSeek default or an existing homogeneous arm.

    Args:
        solver_model: Model executing the benchmark harness.
        proposer_model: Model editing the harness prompts.

    Raises:
        ValueError: The pair is outside the reviewed experiment profiles.
    """
    if (solver_model, proposer_model) != (DEFAULT_SOLVER_MODEL, DEFAULT_PROPOSER_MODEL):
        validate_experiment_model_pair(solver_model, proposer_model)


def resolve_benchmark_lm_kwargs(
    model: str,
    api_base: str | None,
    *,
    role: str = SOLVER_ROLE,
) -> dict[str, object]:
    """Resolve the same provider, retry and reasoning settings for every harness.

    Args:
        model: Pinned experiment model identifier.
        api_base: Role-specific OpenAI-compatible endpoint, when supplied.
        role: Solver or optimizer, selecting the role's output budget.

    Returns:
        An independent mapping suitable for the shared LM client.

    Raises:
        ValueError: The role or model is not supported.
    """
    if role not in {SOLVER_ROLE, OPTIMIZER_ROLE}:
        raise ValueError(f"Unknown benchmark model role: {role!r}")
    kwargs: dict[str, object] = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        **provider_retry_kwargs(role=role),
        **experiment_decoding(model, agentic=False),
        **experiment_request_overrides(model, explicit_reasoning=True),
        "max_tokens": SOLVER_MAX_TOKENS if role == SOLVER_ROLE else OPTIMIZER_MAX_TOKENS[model],
        "seed": SCIENTIFIC_REQUEST_SEED,
    }
    extra_body = kwargs["extra_body"]
    assert isinstance(extra_body, dict)
    kwargs["extra_body"] = {
        **extra_body,
        "thinking_token_budget": SOLVER_THINKING_TOKENS if role == SOLVER_ROLE else OPTIMIZER_THINKING_TOKENS[model],
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    return kwargs
