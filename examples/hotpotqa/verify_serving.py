"""Manually verify that a locally served HotPotQA model handles ReAct V2 tools.

Run this diagnostic on a GPU node through
``scripts/della/verify_deepseek_serving.sh`` or the Qwen qualification pilot.
The diagnostic writes no marker or lock files; the Qwen pilot records successful
verification for its exact runtime. It exercises an ordinary completion, a native tool call followed by
its tool-result continuation, and one real ReAct V2 proposal per broad edit tool
(DELETE_TEXT, INSERT_TEXT, MOVE_TEXT, REPLACE_TEXT), then prints a PASS/FAIL
report and exits non-zero when any check failed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL
from examples.common.provider_retries import PROVIDER_RETRY_POLICY, provider_retry_kwargs
from examples.hotpotqa.model_settings import HOTPOTQA_REQUEST_TIMEOUT_SECONDS
from examples.hotpotqa.runtime_canary import (
    RuntimeCanaryError as ServingVerificationError,
)
from examples.hotpotqa.runtime_canary import (
    _edit_probe,
    _ordinary_completion_probe,
    _tool_continuation_probe,
    _validate_loopback_api_base,
)
from examples.hotpotqa.runtime_canary import (
    _require_healthy_text as _require_healthy_text,
)
from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs
from gepa.lm import LM
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.forest_constants import BROAD_EDIT_TOOL_SET, OPTIMIZER_ROLE

DEFAULT_TIMEOUT_SECONDS = HOTPOTQA_REQUEST_TIMEOUT_SECONDS


def run_serving_verification(
    model: str,
    api_base: str,
    attempts: int,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    attempt_log: Path | None = None,
) -> dict[str, object]:
    """Run every verification check and collect a PASS/FAIL report.

    Every check runs even after an earlier one fails, so one report shows the
    complete picture of what the served model can and cannot do.

    Args:
        model: Exact local LiteLLM model identifier served on the node.
        api_base: Local OpenAI-compatible /v1 endpoint.
        attempts: Number of ReAct V2 edit attempts, cycled over the four broad
            edit tools; at least one attempt per tool is required.
        timeout: Deadline shared by the bounded provider attempts.
        attempt_log: Optional JSONL path for physical provider attempts.

    Returns:
        JSON-serializable report with an overall status, per-check outcomes,
        the list of failed checks, and per-tool attempt counts.

    Raises:
        ServingVerificationError: The endpoint is non-local or fewer attempts
            than broad edit tools were requested.
        ValueError: The model identifier is outside the scientific catalog.
    """
    _validate_loopback_api_base(api_base)
    tools = EDIT_TOOL_SETS[BROAD_EDIT_TOOL_SET]
    if attempts < len(tools):
        raise ServingVerificationError(
            f"At least {len(tools)} edit attempts are needed to exercise every broad edit tool once; "
            f"received {attempts}."
        )
    lm_kwargs: dict[str, Any] = dict(resolve_hotpotqa_lm_kwargs(model, api_base, role=OPTIMIZER_ROLE))
    lm_kwargs.update(provider_retry_kwargs(attempt_log, "serving_verification"))
    lm_kwargs["timeout"] = timeout
    lm = LM(model, **lm_kwargs)

    checks: dict[str, str] = {}

    def record(name: str, probe: Callable[[], object]) -> None:
        """Run one probe and record PASS or the failure message.

        Args:
            name: Report key for the probe.
            probe: Zero-argument callable that raises on failure.
        """
        try:
            probe()
        except (RuntimeError, ValueError, TypeError) as exc:
            checks[name] = f"FAIL: {exc}"
        else:
            checks[name] = "PASS"

    record("ordinary_completion", lambda: _ordinary_completion_probe(lm))
    record("tool_result_continuation", lambda: _tool_continuation_probe(lm))
    tool_counts: Counter[str] = Counter()
    for offset in range(attempts):
        tool = tools[offset % len(tools)]
        tool_counts[tool.value] += 1
        attempt = offset + 1
        record(
            f"edit:{tool.value}#{tool_counts[tool.value]}",
            lambda tool=tool, attempt=attempt: _edit_probe(lm, tool, attempt),
        )
    failures = sorted(name for name, status in checks.items() if status != "PASS")
    return {
        "status": "FAIL" if failures else "PASS",
        "provider_retry_policy": PROVIDER_RETRY_POLICY,
        "model": model,
        "api_base": api_base,
        "attempts": attempts,
        "tool_attempts": dict(sorted(tool_counts.items())),
        "checks": checks,
        "failures": failures,
    }


def main() -> None:
    """Parse the verification CLI, print the report, and exit non-zero on failure."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a locally served HotPotQA model supports ordinary completions, native tool-result "
            "continuation, and every ReAct V2 edit tool."
        )
    )
    parser.add_argument(
        "--model",
        default=DEEPSEEK_V4_1_FLASH_MODEL,
        help="Exact local LiteLLM model identifier (default: the DeepSeek-V4.1-Flash arm)",
    )
    parser.add_argument("--api-base", required=True, help="Local OpenAI-compatible /v1 endpoint")
    parser.add_argument(
        "--attempts",
        type=int,
        default=len(EDIT_TOOL_SETS[BROAD_EDIT_TOOL_SET]),
        help="ReAct V2 edit attempts cycled over the four broad edit tools (default: one per tool)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds",
    )
    parser.add_argument("--attempt-log", type=Path, help="JSONL destination for provider attempts")
    args = parser.parse_args()
    try:
        report = run_serving_verification(args.model, args.api_base, args.attempts, args.timeout, args.attempt_log)
    except (RuntimeError, ValueError, TypeError) as exc:
        parser.exit(2, f"Serving verification could not start: {exc}\n")
    checks = report["checks"]
    assert isinstance(checks, dict)
    print("Serving verification report")
    for name, status in checks.items():
        print(f"  {'PASS' if status == 'PASS' else 'FAIL'}  {name}" + ("" if status == "PASS" else f"  ({status[6:]})"))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"RESULT: {report['status']}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
