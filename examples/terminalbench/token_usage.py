"""Record provider token counts and limit evidence without storing prompt content."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.common.provider_retries import PROVIDER_RETRY_KEY, provider_retry_kwargs

TOKEN_USAGE_POLICY = {
    "schema_version": 2,
    "counts": "provider_reported_only",
    "missing_counts": "null",
    "length_finish": "provider_length_finish_not_inferred_from_text",
    "scope": "every_live_provider_attempt_including_errors",
    "journal_replays": "excluded",
    "raw_text": False,
}


def _field(value: Any, key: str, default: Any = None) -> Any:
    """Read a provider field from either dictionary or object responses."""
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def record_usage(
    path: Path, role: str, model: str, limits: dict[str, Any], response: Any = None, error: BaseException | None = None
) -> None:
    """Durably append usage before a caller parses or salvages a model response.

    Args:
        path: Run- or trial-local JSONL destination.
        role: Model role; task-agent records include its summarization calls.
        model: Requested model identity.
        limits: Effective context and output policy.
        response: Raw provider completion, when available.
        error: Provider exception, recorded by type without its potentially private message.
    """
    usage = _field(response, "usage")
    prompt = _field(usage, "prompt_tokens")
    output = _field(usage, "completion_tokens")
    reasoning = _field(_field(usage, "completion_tokens_details"), "reasoning_tokens")
    reasons = [_field(choice, "finish_reason") for choice in _field(response, "choices", [])]
    record = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "requested_model": model,
        "response_model": _field(response, "model"),
        "response_id": _field(response, "id"),
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "reasoning_tokens": reasoning,
        "finish_reasons": reasons,
        "length_finish": "length" in reasons if any(reason is not None for reason in reasons) else None,
        "output_cap_reached": output >= limits["max_output_tokens"] if output is not None else None,
        "context_cap_reached": (
            prompt + output >= limits["context_tokens"] if prompt is not None and output is not None else None
        ),
        "error_type": type(error).__name__ if error is not None else None,
        "limits": limits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def observe_optimizer(lm: Any, path: Path, role: str, limits: dict[str, Any]) -> Any:
    """Record every optimizer transport attempt while preserving journal replay.

    Args:
        lm: Existing GEPA LM; shared role clients must be attached only once.
        path: Run-local usage log.
        role: Stable role label.
        limits: Serving and generation limits for the model arm.

    Returns:
        The same client with the approved retry and raw-usage policy.
    """
    settings = provider_retry_kwargs(path.with_name("provider-attempts.jsonl"), role)
    settings[PROVIDER_RETRY_KEY].update(token_usage_log=str(path), token_limits=limits)
    lm.completion_kwargs.update(settings)
    return lm


def observe_harbor(llm: Any, path: Path, limits: dict[str, Any]) -> None:
    """Record each Harbor transport attempt before parsing or length recovery.

    Args:
        llm: Harbor 0.22.0 LiteLLM instance shared by main and summary calls.
        path: Trial-local usage log.
        limits: Effective model limits.
    """
    settings = provider_retry_kwargs(path.with_name("provider-attempts.jsonl"), "task_agent")
    settings[PROVIDER_RETRY_KEY].update(token_usage_log=str(path), token_limits=limits)
    llm._llm_kwargs.update(settings)


def summarize_usage(paths: list[Path]) -> dict[str, Any]:
    """Aggregate original call logs, retaining missing-usage and truncation counts.

    Args:
        paths: Files or run roots; overlapping roots are deduplicated.

    Returns:
        Counts by model and role across observed physical calls, including failed jobs.
    """
    files = sorted(
        {file.resolve() for path in paths for file in ([path] if path.is_file() else path.rglob("token-usage.jsonl"))}
    )
    models: dict[str, Any] = {}
    for file in files:
        for line in file.read_text().splitlines():
            record = json.loads(line)
            if record["schema_version"] != 1:
                raise ValueError(f"Unsupported token-usage schema in {file}")
            roles = models.setdefault(record["requested_model"], {})
            totals = roles.setdefault(record["role"], {"calls": 0, "errors": 0})
            totals["calls"] += 1
            totals["errors"] += record["error_type"] is not None
            for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                totals.setdefault(field, 0)
                totals.setdefault(f"{field}_unreported_calls", 0)
                if record[field] is None:
                    totals[f"{field}_unreported_calls"] += 1
                else:
                    totals[field] += record[field]
            if record["completion_tokens"] is not None:
                totals["max_observed_completion_tokens"] = max(
                    totals.get("max_observed_completion_tokens", 0), record["completion_tokens"]
                )
            for field in ("length_finish", "output_cap_reached", "context_cap_reached"):
                totals[field] = totals.get(field, 0) + (record[field] is True)
                totals[f"{field}_unreported_calls"] = totals.get(f"{field}_unreported_calls", 0) + (
                    record[field] is None
                )
    return {"schema_version": 1, "files": [str(file) for file in files], "models": models}


def main() -> None:
    """Write a usage report for one or more optimization or held-out evaluation roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize_usage(args.paths), indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
