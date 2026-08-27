"""Model identities and request settings for the paired benchmark runs."""

from copy import deepcopy

QWEN3_8_27B_MODEL = "hosted_vllm/Qwen/Qwen3.8-27B"
QWEN3_8_27B_OPENROUTER_MODEL = "openrouter/qwen/qwen3.8-27b"
QWEN3_8_27B_MODEL_INFO = {
    "max_input_tokens": 32_768,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
DEEPSEEK_V4_FLASH_MODEL = "deepseek/deepseek-v4-flash"
DEEPSEEK_V4_FLASH_0731_OPENROUTER_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
EXPERIMENT_MODELS = (QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL)
EXPERIMENT_API_PROFILES = ("direct", "openrouter")
EXPERIMENT_NUM_RETRIES = 0

_OPENROUTER_RUNTIME_MODELS = {
    QWEN3_8_27B_MODEL: QWEN3_8_27B_OPENROUTER_MODEL,
    DEEPSEEK_V4_FLASH_MODEL: DEEPSEEK_V4_FLASH_0731_OPENROUTER_MODEL,
}

# Qwen3.8-27B thinking mode and DeepSeek V4 Flash's published agent runs both
# use temperature 1.0 and top-p 0.95. Qwen additionally specifies top-k 20;
# DeepSeek's published agent setting uses maximum reasoning effort.
# Sources: https://huggingface.co/Qwen/Qwen3.8-27B
#          https://api-docs.deepseek.com/updates/
_EXPERIMENT_DECODING = {
    QWEN3_8_27B_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16_384,
    },
    QWEN3_8_27B_OPENROUTER_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16_384,
    },
    DEEPSEEK_V4_FLASH_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
        "reasoning_effort": "max",
    },
    DEEPSEEK_V4_FLASH_0731_OPENROUTER_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    },
}

# OpenRouter's native request body keeps the provider constraint attached to
# ordinary completions and native tool calls. Fallbacks are disabled so a run
# cannot silently change inference providers after it starts.
_EXPERIMENT_REQUEST_OVERRIDES = {
    QWEN3_8_27B_OPENROUTER_MODEL: {
        "extra_body": {
            "provider": {
                "only": ["akashml"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["bf16"],
                "max_price": {"prompt": 0.40, "completion": 2.55},
            },
            "reasoning": {"effort": "xhigh"},
        }
    },
    DEEPSEEK_V4_FLASH_0731_OPENROUTER_MODEL: {
        "extra_body": {
            "provider": {
                "only": ["deepseek"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "max_price": {"prompt": 0.44, "completion": 1.32},
            },
            "reasoning": {"effort": "max"},
        }
    },
}


def experiment_decoding(model: str) -> dict[str, int | float | str]:
    """Return the fixed decoding settings for one experiment model.

    Qwen3.8-27B uses its recommended thinking-mode sampling parameters.
    DeepSeek V4 Flash uses the settings published for its agent evaluations.
    Both retain the same output-token ceiling used by the benchmark harnesses.

    Args:
        model: Exact LiteLLM model identifier used by a benchmark run.

    Returns:
        Independent decoding-parameter mapping for the requested model.

    Raises:
        ValueError: The model is not a supported experiment runtime.
    """
    try:
        return dict(_EXPERIMENT_DECODING[model])
    except KeyError as exc:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}") from exc


def experiment_request_overrides(model: str) -> dict[str, object]:
    """Return provider-specific request fields for one runtime model.

    Local and direct-provider model identifiers need no transport override.
    OpenRouter identifiers receive a native request body that fixes the exact
    inference provider, disables fallback routing, requires every requested
    parameter, and fixes the reasoning mode. A deep copy prevents one client
    from mutating the routing policy used by later calls.

    Args:
        model: Exact LiteLLM model identifier used by a benchmark run.

    Returns:
        Independent provider-request mapping, or an empty mapping when the
        selected runtime does not need a transport override.

    Raises:
        ValueError: The model is not a supported experiment runtime.
    """
    if model not in _EXPERIMENT_DECODING:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    return deepcopy(_EXPERIMENT_REQUEST_OVERRIDES.get(model, {}))


def resolve_experiment_model(model: str, api_profile: str) -> str:
    """Resolve a scientific model identity to its API runtime identifier.

    The direct profile preserves the configured identifier used by Della and
    provider-native runs. The OpenRouter profile maps the same experimental
    arm to an exact OpenRouter model slug; request routing remains separate in
    :func:`experiment_request_overrides`.

    Args:
        model: Canonical student/proposer identity for the experiment arm.
        api_profile: Runtime route, either ``"direct"`` or ``"openrouter"``.

    Returns:
        LiteLLM model identifier used for actual completion requests.

    Raises:
        ValueError: The model or API profile is not supported.
    """
    if model not in EXPERIMENT_MODELS:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}")
    if api_profile == "direct":
        return model
    if api_profile == "openrouter":
        return _OPENROUTER_RUNTIME_MODELS[model]
    supported_profiles = ", ".join(EXPERIMENT_API_PROFILES)
    raise ValueError(f"Unsupported API profile {api_profile!r}; expected one of: {supported_profiles}")


def validate_experiment_model_pair(student_model: str, proposer_model: str) -> None:
    """Require a homogeneous student/proposer experiment profile.

    Args:
        student_model: Model that executes the benchmark program.
        proposer_model: Model that reflects on traces and proposes revisions.

    Raises:
        ValueError: The roles use different models or an unrecognized model.
    """
    if student_model != proposer_model:
        raise ValueError(
            "Benchmark runs require the same model for the student and proposer within each arm; "
            f"received student={student_model!r} and proposer={proposer_model!r}."
        )
    if student_model not in EXPERIMENT_MODELS:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {student_model!r}; expected one of: {supported}")
