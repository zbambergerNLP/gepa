"""Tests for shared benchmark model identities and request settings."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    DEEPSEEK_V4_1_FLASH_MODEL,
    DEEPSEEK_V4_1_FLASH_MODEL_INFO,
    DEEPSEEK_V4_1_FLASH_REVISION,
    EXPERIMENT_MODELS,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
    validate_experiment_model_pair,
    validate_experiment_vllm_version,
)


def test_deepseek_profile_uses_the_pinned_local_identity() -> None:
    """Map the DeepSeek arm to its exact local checkpoint."""
    assert EXPERIMENT_MODELS == (QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL)
    assert DEEPSEEK_V4_1_FLASH_MODEL == "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
    assert DEEPSEEK_V4_1_FLASH_MODEL_INFO == {
        "max_input_tokens": 262_144,
        "max_output_tokens": 16_384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert experiment_model_version(DEEPSEEK_V4_1_FLASH_MODEL) == DEEPSEEK_V4_1_FLASH_REVISION


def test_deepseek_profile_uses_fixed_sampling_and_maximum_reasoning() -> None:
    """Keep the local run fixed on decoding and reasoning effort."""
    expected_decoding = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    }
    assert experiment_decoding(DEEPSEEK_V4_1_FLASH_MODEL) == expected_decoding
    assert experiment_request_overrides(DEEPSEEK_V4_1_FLASH_MODEL) == {
        "extra_body": {
            "chat_template_kwargs": {
                "reasoning_effort": 100,
                "thinking": True,
            },
        }
    }


@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
@pytest.mark.parametrize("agentic", [False, True])
def test_provider_sampling_depends_on_the_work_without_mutating_other_profiles(model: str, agentic: bool) -> None:
    """Keep Qwen fixed while applying DeepSeek's general and agentic recommendations."""
    decoding = experiment_decoding(model, agentic=agentic)
    assert decoding["top_p"] == 0.95
    assert decoding["temperature"] == 1.0
    decoding["top_p"] = 0.5
    assert experiment_decoding(model)["top_p"] == 0.95
    assert experiment_decoding(model, agentic=agentic)["top_p"] != 0.5


@pytest.mark.parametrize(
    ("model", "template_kwargs"),
    [
        (QWEN3_8_27B_MODEL, {"enable_thinking": True, "reasoning_effort": "xhigh"}),
        (DEEPSEEK_V4_1_FLASH_MODEL, {"thinking": True, "reasoning_effort": 100}),
    ],
)
def test_explicit_reasoning_uses_provider_template_fields_without_shared_mutation(
    model: str, template_kwargs: dict[str, object]
) -> None:
    """Pin thinking and effort while isolating every client and preserving legacy defaults."""
    expected = {"extra_body": {"chat_template_kwargs": template_kwargs}}
    request = experiment_request_overrides(model, explicit_reasoning=True)
    assert request == expected
    request["extra_body"]["chat_template_kwargs"]["reasoning_effort"] = "low"
    assert experiment_request_overrides(model, explicit_reasoning=True) == expected
    assert experiment_request_overrides(model) == ({} if model == QWEN3_8_27B_MODEL else expected)


@pytest.mark.parametrize("version", ["0.1.1.dev5+ge77daef89"])
def test_deepseek_accepts_supported_vllm_versions(version: str) -> None:
    """Accept only the reviewed commit wheel for the V4.1 checkpoint."""
    validate_experiment_vllm_version(DEEPSEEK_V4_1_FLASH_MODEL, version)


@pytest.mark.parametrize("version", ["0.17.0", "0.25.1", "0.29.1.dev1", "0.1.1.dev5+gwrong", "", "unknown"])
def test_deepseek_rejects_missing_or_old_vllm_versions(version: str) -> None:
    """Reject old Qwen-only environments before a DeepSeek GPU launch."""
    with pytest.raises(ValueError):
        validate_experiment_vllm_version(DEEPSEEK_V4_1_FLASH_MODEL, version)


@pytest.mark.parametrize(
    "model",
    [
        "hosted_vllm/zai-org/GLM-5.3-Flash",
        "deepseek/deepseek-v4.1-flash",
        "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731",
    ],
)
def test_removed_model_routes_cannot_start_or_resume_a_campaign(model: str) -> None:
    """Reject the removed GLM arm and the unpinned direct-provider alias."""
    with pytest.raises(ValueError, match="Unsupported experiment model"):
        validate_experiment_model_pair(model, model)
