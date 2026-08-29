"""Tests for shared benchmark model identities and request settings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    GLM_5_3_FLASH_MODEL,
    GLM_5_3_FLASH_MODEL_INFO,
    GLM_5_3_FLASH_OPENROUTER_MODEL,
    GLM_5_3_FLASH_REVISION,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
    resolve_experiment_model,
)


def test_glm_profile_uses_the_pinned_local_and_openrouter_identities() -> None:
    """Map the GLM arm to its exact checkpoint and smoke-test route."""
    assert EXPERIMENT_MODELS == (QWEN3_8_27B_MODEL, GLM_5_3_FLASH_MODEL)
    assert GLM_5_3_FLASH_MODEL == "hosted_vllm/zai-org/GLM-5.3-Flash"
    assert GLM_5_3_FLASH_OPENROUTER_MODEL == "openrouter/z-ai/glm-5.3-flash"
    assert GLM_5_3_FLASH_MODEL_INFO == {
        "max_input_tokens": 262_144,
        "max_output_tokens": 16_384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert resolve_experiment_model(GLM_5_3_FLASH_MODEL, "direct") == GLM_5_3_FLASH_MODEL
    assert resolve_experiment_model(GLM_5_3_FLASH_MODEL, "openrouter") == GLM_5_3_FLASH_OPENROUTER_MODEL
    assert experiment_model_version(GLM_5_3_FLASH_MODEL) == GLM_5_3_FLASH_REVISION
    assert experiment_model_version(GLM_5_3_FLASH_OPENROUTER_MODEL) == GLM_5_3_FLASH_REVISION


def test_glm_profile_uses_fixed_sampling_and_maximum_reasoning() -> None:
    """Keep local and smoke runs aligned on decoding and reasoning effort."""
    expected_decoding = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    }
    assert experiment_decoding(GLM_5_3_FLASH_MODEL) == expected_decoding
    assert experiment_decoding(GLM_5_3_FLASH_OPENROUTER_MODEL) == expected_decoding
    assert experiment_request_overrides(GLM_5_3_FLASH_MODEL) == {
        "extra_body": {
            "chat_template_kwargs": {
                "reasoning_effort": "max",
                "clear_thinking": True,
            },
        }
    }
    assert experiment_request_overrides(GLM_5_3_FLASH_OPENROUTER_MODEL) == {
        "extra_body": {
            "provider": {
                "only": ["z-ai"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
            "reasoning": {"effort": "max"},
        }
    }
