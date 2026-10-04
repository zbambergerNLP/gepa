"""Keep the official user simulator fixed and separate from optimized roles."""

USER_MODEL = "gpt-4.1-2025-04-14"
USER_KWARGS = {"temperature": 0.0, "timeout": 3600, "api_base": "https://api.openai.com/v1"}
JUDGE_MODEL = "gpt-4.1-2025-04-14"
