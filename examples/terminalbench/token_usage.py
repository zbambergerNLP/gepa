"""Record provider token counts and limit evidence without storing prompt content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.common.provider_retries import PROVIDER_RETRY_KEY, provider_retry_kwargs
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG, TOKEN_USAGE_LOG

TOKEN_USAGE_POLICY = {
    "schema_version": 2,
    "counts": "provider_reported_only",
    "missing_counts": "null",
    "length_finish": "provider_length_finish_not_inferred_from_text",
    "scope": "every_live_provider_attempt_including_errors",
    "journal_replays": "excluded",
    "raw_text": False,
}


def observe_harbor(llm: Any, path: Path, limits: dict[str, Any]) -> None:
    """Record each Harbor transport attempt before parsing or length recovery.

    Args:
        llm: Harbor 0.22.0 LiteLLM instance shared by main and summary calls.
        path: Trial-local usage log.
        limits: Effective model limits.
    """
    settings = provider_retry_kwargs(path.with_name(PROVIDER_ATTEMPT_LOG), "task_agent")
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
        {file.resolve() for path in paths for file in ([path] if path.is_file() else path.rglob(TOKEN_USAGE_LOG))}
    )
    models: dict[str, Any] = {}
    for file in files:
        for line in file.read_text().splitlines():
            record = json.loads(line)
            if record["schema_version"] not in {1, 2}:
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
