"""Verify practical caps and usage evidence without making model requests."""

import json
from pathlib import Path
from unittest.mock import Mock

import litellm
import pytest

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    experiment_decoding,
    experiment_model_info,
    experiment_request_overrides,
)
from examples.common import provider_retries
from examples.terminalbench.model_settings import (
    terminalbench_decoding,
    terminalbench_limits,
    terminalbench_model_info,
)
from examples.terminalbench.token_usage import observe_harbor, summarize_usage
from types import SimpleNamespace
from gepa.lm import LM
from gepa.response_journal import response_journal_scope


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
def test_output_budget_does_not_expand_context_or_change_qa(model: str) -> None:
    """Keep shared defaults and context unchanged while TB roles receive a 32K ceiling."""
    original_info = experiment_model_info(model)
    for agentic in (False, True):
        qa = experiment_decoding(model, agentic=agentic)
        tb = terminalbench_decoding(model, agentic=agentic)
        assert qa["max_tokens"] == 16_384
        assert tb == {**qa, "max_tokens": 32_768}
    assert terminalbench_model_info(model) == {**original_info, "max_output_tokens": 32_768}
    assert terminalbench_limits(model)["context_tokens"] == original_info["max_input_tokens"]
    assert "thinking_token_budget" not in experiment_request_overrides(model, explicit_reasoning=True)["extra_body"]


def response(model: str, output: int, finish: str = "stop", reasoning: int | None = None) -> litellm.ModelResponse:
    """Build a realistic provider response with optional reported reasoning usage."""
    usage = {"prompt_tokens": 123, "completion_tokens": output, "total_tokens": 123 + output}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return litellm.ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": "private model text"}, "finish_reason": finish}],
        usage=usage,
    )


def test_report_keeps_unknown_usage_and_distinguishes_caps_from_cutoffs(tmp_path: Path) -> None:
    """Aggregate physical logs once per file without merging different model arms."""
    first, second = EXPERIMENT_MODELS
    path = tmp_path / "trial" / "token-usage.jsonl"

    def append_usage(raw, model):
        usage = raw.usage if raw is not None else None
        output = usage.completion_tokens if usage is not None else None
        finish = raw.choices[0].finish_reason if raw is not None else None
        details = getattr(usage, "completion_tokens_details", None)
        record = {
            "schema_version": 2,
            "requested_model": model,
            "role": "task_agent",
            "error_type": "RuntimeError" if raw is None else None,
            "prompt_tokens": usage.prompt_tokens if usage is not None else None,
            "completion_tokens": output,
            "reasoning_tokens": details.reasoning_tokens if details else None,
            "length_finish": finish == "length" if finish else None,
            "output_cap_reached": output >= 32768 if output is not None else None,
            "context_cap_reached": False if output is not None else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    for output, finish in [(32_768, "length"), (32_768, "stop"), (100, "length")]:
        append_usage(response(first, output, finish, 50), first)
    append_usage(None, first)
    append_usage(response(second, 10), second)
    report = summarize_usage([path, tmp_path])
    assert report["files"] == [str(path)]
    totals = report["models"][first]["task_agent"]
    assert totals["calls"] == 4 and totals["errors"] == 1
    assert totals["length_finish"] == totals["output_cap_reached"] == 2
    assert totals["length_finish_unreported_calls"] == totals["completion_tokens_unreported_calls"] == 1
    assert totals["completion_tokens"] == 65_636
    assert totals["reasoning_tokens"] == 150
    assert totals["max_observed_completion_tokens"] == 32_768
    assert report["models"][second]["task_agent"]["reasoning_tokens_unreported_calls"] == 1
    assert "private" not in path.read_text()


@pytest.mark.parametrize("mode", ["plain", "tools", "batch"])
@pytest.mark.parametrize("failures", [0, 2])
def test_optimizer_usage_records_live_responses_once_and_excludes_journal_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, failures: int
) -> None:
    """Exercise real GEPA clients, request ceilings, usage accounting, and replay."""
    model = EXPERIMENT_MODELS[0]
    cutoff = response(model, 32_768, "length", 30_000)
    raw = response(model, 100, "stop", 50)
    provider = Mock(side_effect=[ConnectionError("temporary")] * failures + [cutoff, raw])
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", lambda _: None)
    monkeypatch.setattr(litellm, "completion_cost", Mock(return_value=0.25))
    path = tmp_path / "token-usage.jsonl"
    for _ in range(2):
        transport = SimpleNamespace(_llm_kwargs={})
        observe_harbor(transport, path, terminalbench_limits(model))
        lm = LM(
            model,
            response_journal_path=tmp_path / "responses.sqlite3",
            response_journal_namespace="proposer",
            **terminalbench_decoding(model),
            **transport._llm_kwargs,
        )
        with response_journal_scope("iteration-0"):
            if mode == "plain":
                assert lm("input") == "private model text"
            elif mode == "tools":
                assert (
                    lm.complete_with_tools([{"role": "user", "content": "input"}], tools=[]).content
                    == "private model text"
                )
            else:
                assert lm.batch_complete([[{"role": "user", "content": "input"}]]) == ["private model text"]
        assert lm.total_tokens_out == 100
        assert lm.total_cost == 0.25
    assert provider.call_count == failures + 2
    assert provider.call_args.kwargs["max_tokens"] == 32_768
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == failures + 2
    assert records[-2]["length_finish"] and records[-2]["reasoning_tokens"] == 30_000
    assert records[-2]["response_error"] == "output_length"
    assert not records[-1]["length_finish"] and records[-1]["reasoning_tokens"] == 50
    assert all(record["completion_tokens"] is None for record in records[:failures])
    totals = summarize_usage([path])["models"][model]["task_agent"]
    assert totals["completion_tokens"] == 32_868 and totals["errors"] == failures + 1
