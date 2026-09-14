"""Verify native editor recovery with one explicitly injected tool-argument error."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from examples.common.pilot_checks import atomic_json
from examples.common.provider_retries import PROVIDER_RETRY_KEY, provider_retry_kwargs
from examples.hotpotqa.runtime_canary import _EDIT_REGION, _EDIT_STEERING, _validate_loopback_api_base
from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs
from gepa.lm import LM, ToolCompletion
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2Proposer
from gepa.strategies.document_template import TEMPLATE_FAMILIES, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool


class FaultInjectingLM:
    """Change one returned target while preserving native calls and reasoning state."""

    def __init__(self, lm: Any) -> None:
        """Wrap the real local model and retain evidence of the controlled fault."""
        self.lm = lm
        self.model = lm.model
        self.injected_fault: dict[str, Any] | None = None
        self.delivered_errors: list[str] = []

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Preserve the wrapped client's ordinary completion interface."""
        return self.lm(prompt)

    def complete_with_tools(self, messages, tools, *, tool_choice=None) -> ToolCompletion:
        """Forward error observations unchanged and corrupt only the first native target."""
        for message in messages:
            content = message.get("content", "")
            if message.get("role") == "tool" and isinstance(content, str) and content.startswith("ERROR:"):
                if content not in self.delivered_errors:
                    self.delivered_errors.append(content)
        completion = self.lm.complete_with_tools(messages, tools, tool_choice=tool_choice)
        if self.injected_fault is not None:
            return completion
        if len(completion.tool_calls) != 1 or completion.tool_calls[0].name != "REPLACE_TEXT":
            raise RuntimeError("Recovery probe requires one initial native REPLACE_TEXT call")
        original = completion.tool_calls[0]
        arguments = json.loads(original.arguments)
        arguments["target"] = "__QUALIFICATION_MISSING_TARGET__"
        injected = replace(original, arguments=json.dumps(arguments))
        self.injected_fault = {
            "kind": "controlled_argument_injection",
            "original": asdict(original),
            "injected": asdict(injected),
            "spontaneous_model_error": False,
        }
        return replace(completion, tool_calls=(injected,))


def verify_error_recovery(lm: Any) -> dict[str, Any]:
    """Require a native error observation, a corrected edit, and explicit completion."""
    wrapped = FaultInjectingLM(lm)
    proposer = ReActV2Proposer(
        wrapped,
        TEMPLATE_FAMILIES["generic"]["system_prompt"],
        EDIT_TOOL_SETS["broad"],
        max_iterations=4,
        max_tool_calls=3,
    )
    result = proposer.propose(
        region_text=_EDIT_REGION,
        edit_target=EditTarget("final_answer", "Task"),
        preferred_tool=EditTool.REPLACE_TEXT,
        steering_message=_EDIT_STEERING[EditTool.REPLACE_TEXT] + " Then explicitly finish.",
        feedback_summary="The answer omitted inline source attribution. Preserve all unrelated instructions.",
        traces_text="The answer listed sources separately from the claims they supported.",
        branch_history=[],
        max_chars=None,
    )
    expected = _EDIT_REGION.replace("Cite primary sources.", "Cite primary sources inline.")
    if (
        not wrapped.injected_fault
        or not result.steps
        or not result.steps[0].error
        or not any(result.steps[0].error in error and _EDIT_REGION in error for error in wrapped.delivered_errors)
        or result.new_text != expected
        or not result.changed
        or result.dropped_reason is not None
        or result.steps[-1].action != "FINISH"
    ):
        raise RuntimeError(f"Native editor failed the controlled error-recovery check: {asdict(result)}")
    return {
        "status": "passed",
        "model": lm.model,
        "injected_fault": wrapped.injected_fault,
        "delivered_errors": wrapped.delivered_errors,
        "result": asdict(result),
        "probe_limits": {"iterations": 4, "tool_calls": 3},
        "production_editor_limits_changed": False,
    }


def run_probe(model: str, api_base: str, attempt_log: Path) -> dict[str, Any]:
    """Use the approved optimizer settings and reject any truncated diagnostic call."""
    _validate_loopback_api_base(api_base)
    kwargs: dict[str, Any] = dict(resolve_hotpotqa_lm_kwargs(model, api_base, role="optimizer"))
    kwargs.update(provider_retry_kwargs(attempt_log, "error_recovery_probe"))
    retry = kwargs[PROVIDER_RETRY_KEY]
    assert isinstance(retry, dict)
    retry.update(
        token_limits={"max_output_tokens": kwargs["max_tokens"], "context_tokens": 262144},
        token_usage_log=str(attempt_log.with_name("tokens-" + attempt_log.name)),
    )
    result = verify_error_recovery(LM(model, **kwargs))
    attempts = [json.loads(line) for line in attempt_log.read_text().splitlines()]
    if not attempts or any(
        row.get("length_finish") or row.get("output_cap_reached") or "length" in (row.get("finish_reasons") or [])
        for row in attempts
    ):
        raise RuntimeError("Controlled error-recovery probe has missing usage or truncated model output")
    result.update(
        source_commit=os.environ.get("HOTPOTQA_SOURCE_COMMIT"),
        campaign_id=os.environ.get("HOTPOTQA_CAMPAIGN_ID"),
        allocation_job_id=os.environ.get("SLURM_JOB_ID"),
        optimizer_output_cap=kwargs["max_tokens"],
    )
    return result


def main() -> None:
    """Write an auditable probe result and fail before the pilot if recovery fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--attempt-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_probe(args.model, args.api_base, args.attempt_log)
    except Exception as exc:
        atomic_json(args.output, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    atomic_json(args.output, result)
    print("Native editor controlled error-recovery probe passed.")


if __name__ == "__main__":
    main()
