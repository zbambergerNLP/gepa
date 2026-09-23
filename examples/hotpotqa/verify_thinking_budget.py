"""Check the server's reasoning boundary before expensive qualification calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.hotpotqa.runtime_canary import _validate_loopback_api_base
from examples.hotpotqa.smoke_serving import _post_json, build_chat_request


def verify_thinking_budget(model: str, api_base: str, output_dir: Path) -> dict[str, Any]:
    """Force an immediate reasoning boundary and retain the bounded exchange.

    Args:
        model: Scientific model identifier, including the local provider prefix.
        api_base: Loopback OpenAI-compatible endpoint on the allocated node.
        output_dir: New directory for this startup's request, response and report.

    Returns:
        Probe status and the observed finish reason and token usage.
    """
    _validate_loopback_api_base(api_base)
    request = build_chat_request(model, model.removeprefix("hosted_vllm/"), api_base)
    request.update(
        messages=[{"role": "user", "content": "Think carefully before answering. Return the word ready."}],
        thinking_token_budget=0,
        max_tokens=256,
        return_token_ids=True,
    )
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    request_path = output_dir / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    request_path.chmod(0o600)
    end_tokens = _post_json(
        f"{api_base.removesuffix('/v1')}/tokenize",
        {"model": request["model"], "prompt": "</think>", "add_special_tokens": False},
        120,
    )["tokens"]
    response = _post_json(
        f"{api_base.rstrip('/')}/chat/completions", request, 120,
        attempt_log=output_dir / "provider-attempts.jsonl",
    )
    response_path = output_dir / "response.json"
    response_path.write_text(json.dumps(response, indent=2) + "\n")
    response_path.chmod(0o600)
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    content = choice["message"].get("content")
    tokens = choice.get("token_ids") or []
    # The pinned Qwen server omits reasoning usage; verify the emitted boundary
    # directly instead of treating missing telemetry as a broken budget.
    boundary_verified = len(end_tokens) == 1 and tokens[:1] == end_tokens
    healthy = (
        boundary_verified
        and reasoning_tokens in (None, 0)
        and choice.get("finish_reason") == "stop"
        and bool(content and content.strip())
    )
    report = {
        "status": "PASS" if healthy else "FAIL",
        "response_id": response.get("id"),
        "model": model,
        "thinking_token_budget": 0,
        "max_tokens": 256,
        "finish_reason": choice.get("finish_reason"),
        "boundary_verified": boundary_verified,
        "first_generated_token": tokens[0] if tokens else None,
        "expected_boundary_tokens": end_tokens,
        "usage": usage,
        "final_characters": len(content or ""),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Fail startup when the native reasoning budget is broken or ignored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = verify_thinking_budget(args.model, args.api_base, args.output_dir)
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise SystemExit("The native reasoning-budget boundary failed; inspect the retained exchange.")


if __name__ == "__main__":
    main()
