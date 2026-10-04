"""Keep the official user simulator fixed and separate from optimized roles."""

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL

SOLVER_MODEL = QWEN3_8_27B_MODEL
PROPOSER_MODEL = DEEPSEEK_V4_1_FLASH_MODEL
USER_MODEL = "gpt-4.1-2025-04-14"
USER_KWARGS = {"temperature": 0.0, "timeout": 3600, "api_base": "https://api.openai.com/v1"}
JUDGE_MODEL = "gpt-4.1-2025-04-14"
