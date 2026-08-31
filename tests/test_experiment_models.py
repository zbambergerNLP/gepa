"""Tests for shared benchmark model identities and request settings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    GLM_5_3_FLASH_MODEL,
    GLM_5_3_FLASH_MODEL_INFO,
    GLM_5_3_FLASH_REVISION,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
)


def test_glm_profile_uses_the_pinned_local_identity() -> None:
    """Map the GLM arm to its exact local checkpoint."""
    assert EXPERIMENT_MODELS == (QWEN3_8_27B_MODEL, GLM_5_3_FLASH_MODEL)
    assert GLM_5_3_FLASH_MODEL == "hosted_vllm/zai-org/GLM-5.3-Flash"
    assert GLM_5_3_FLASH_MODEL_INFO == {
        "max_input_tokens": 262_144,
        "max_output_tokens": 16_384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert experiment_model_version(GLM_5_3_FLASH_MODEL) == GLM_5_3_FLASH_REVISION


def test_glm_profile_uses_fixed_sampling_and_maximum_reasoning() -> None:
    """Keep the local run fixed on decoding and reasoning effort."""
    expected_decoding = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16_384,
    }
    assert experiment_decoding(GLM_5_3_FLASH_MODEL) == expected_decoding
    assert experiment_request_overrides(GLM_5_3_FLASH_MODEL) == {
        "extra_body": {
            "chat_template_kwargs": {
                "reasoning_effort": "max",
                "clear_thinking": True,
            },
        }
    }
