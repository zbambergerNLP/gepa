"""Model identities and decoding settings for the paired benchmark runs."""

QWEN3_8_27B_MODEL = "hosted_vllm/Qwen/Qwen3.8-27B"
QWEN3_8_27B_MODEL_INFO = {
    "max_input_tokens": 32_768,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
DEEPSEEK_V4_FLASH_MODEL = "deepseek/deepseek-v4-flash"
EXPERIMENT_MODELS = (QWEN3_8_27B_MODEL, DEEPSEEK_V4_FLASH_MODEL)
EXPERIMENT_NUM_RETRIES = 0

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
    DEEPSEEK_V4_FLASH_MODEL: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
        "reasoning_effort": "max",
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
        ValueError: The model is not one of the two experiment configurations.
    """
    try:
        return dict(_EXPERIMENT_DECODING[model])
    except KeyError as exc:
        supported = ", ".join(EXPERIMENT_MODELS)
        raise ValueError(f"Unsupported experiment model {model!r}; expected one of: {supported}") from exc


def validate_experiment_model_pair(student_model: str, proposer_model: str) -> None:
    """Require one of the two homogeneous student/proposer profiles.

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
    experiment_decoding(student_model)
