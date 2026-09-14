"""Verify practical caps and usage evidence without making model requests."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import litellm
import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    experiment_decoding,
    experiment_model_info,
    experiment_request_overrides,
)
from examples.terminalbench import canary
from examples.terminalbench.model_settings import (
    terminalbench_decoding,
    terminalbench_limits,
    terminalbench_model_info,
)
from examples.terminalbench.token_usage import observe_optimizer, record_usage, summarize_usage
from gepa.adapters.terminal_bench_adapter import TERMINUS_ADAPTER_CONTRACT
from gepa.adapters.terminal_bench_adapter.text_scope import OPTIMIZATION_SCOPES
from gepa.core.adapter import EvaluationBatch
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
    for output, finish in [(32_768, "length"), (32_768, "stop"), (100, "length")]:
        record_usage(path, "task_agent", first, terminalbench_limits(first), response(first, output, finish, 50))
    record_usage(path, "task_agent", first, terminalbench_limits(first), error=RuntimeError("private error"))
    record_usage(path, "task_agent", second, terminalbench_limits(second), response(second, 10))
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
    raw = response(model, 32_768, "length", 30_000)
    provider = Mock(side_effect=[ConnectionError("temporary")] * failures + [raw])
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(litellm, "completion_cost", Mock(return_value=0.25))
    path = tmp_path / "token-usage.jsonl"
    for _ in range(2):
        lm = observe_optimizer(
            LM(
                model,
                response_journal_path=tmp_path / "responses.sqlite3",
                response_journal_namespace="proposer",
                **terminalbench_decoding(model),
            ),
            path,
            "proposer",
            terminalbench_limits(model),
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
        assert lm.total_tokens_out == 32_768
        assert lm.total_cost == 0.25
    assert provider.call_count == failures + 1
    assert provider.call_args.kwargs["max_tokens"] == 32_768
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == failures + 1
    assert records[-1]["length_finish"] and records[-1]["reasoning_tokens"] == 30_000
    assert all(record["completion_tokens"] is None for record in records[:-1])


@pytest.mark.parametrize("experiment", [None, "tb2.1"])
@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("n_concurrent", [None, 2])
@pytest.mark.parametrize("optimization_scope", [None, *OPTIMIZATION_SCOPES])
def test_canary_uses_only_training_tasks_and_saves_usage_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment: str | None,
    model: str,
    fails: bool,
    n_concurrent: int | None,
    optimization_scope: str | None,
) -> None:
    """Retain pilot evidence while excluding validation and held-out test tasks."""
    factory = Mock(wraps=canary.HarborCLI)
    monkeypatch.setattr(canary.HarborCLI, "check_requirements", Mock())
    monkeypatch.setattr(canary, "HarborCLI", factory)
    output_dir = tmp_path / "pilot"

    def evaluate(adapter, tasks, candidate):
        """Simulate the model boundary and verify the real manifest's selected split."""
        assert [task.task_id for task in tasks] == adapter.manifest.splits["train"][:3]
        assert not {task.task_id for task in tasks}.intersection(adapter.manifest.splits["val"])
        assert not {task.task_id for task in tasks}.intersection(adapter.manifest.splits["test"])
        assert adapter.text_scope.name == (optimization_scope or "system_prompt")
        assert set(candidate) == set(adapter.text_scope.component_kinds)
        record_usage(
            output_dir / "harbor" / "token-usage.jsonl",
            "task_agent",
            model,
            terminalbench_limits(model),
            response(model, 32_768, "length"),
        )
        if fails:
            raise RuntimeError("failed pilot")
        return EvaluationBatch(
            outputs=[{"task_id": task.task_id, "reward": 0.0, "errors": []} for task in tasks],
            scores=[0.0] * len(tasks),
        )

    monkeypatch.setattr(canary.TerminusAdapter, "evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "canary",
            *(["--optimization-scope", optimization_scope] if optimization_scope is not None else []),
            *(["--experiment", experiment] if experiment is not None else []),
            "--model",
            model,
            "--api-base",
            "http://localhost:8000/v1",
            "--output-dir",
            str(output_dir),
            *(["--n-concurrent", str(n_concurrent)] if n_concurrent is not None else []),
        ],
    )
    if fails:
        with pytest.raises(RuntimeError, match="failed pilot"):
            canary.main()
    else:
        canary.main()
    config = json.loads((output_dir / "canary-config.json").read_text())
    assert config["schema_version"] == 9
    assert config["stage"] == "smoke"
    assert (output_dir / "pilot-complete.json").exists() is not fails
    assert config["optimization_scope"] == (optimization_scope or "system_prompt")
    assert config["adapter"] == TERMINUS_ADAPTER_CONTRACT
    assert config["experiment"] == "tb2.1"
    assert config["task_context_settings"] == {
        "enable_summarize": True,
        "proactive_summarization_threshold": 8_000,
    }
    assert config["split"] == "train"
    assert config["n_concurrent"] == factory.call_args.kwargs["n_concurrent"] == (n_concurrent or 1)
    settings = factory.call_args.kwargs["student_agent_kwargs"]
    assert settings == config["student_agent_kwargs"]
    assert settings["llm_kwargs"]["max_tokens"] == settings["model_info"]["max_output_tokens"] == 32_768
    assert (
        settings["llm_kwargs"]["extra_body"]
        == experiment_request_overrides(model, explicit_reasoning=True)["extra_body"]
    )
    report = json.loads((output_dir / "token-usage-summary.json").read_text())
    assert report["models"][model]["task_agent"]["length_finish"] == 1


@pytest.mark.parametrize("n_concurrent", [0, -1])
def test_canary_rejects_invalid_concurrency_before_starting_harbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_concurrent: int
) -> None:
    """Reject invalid calibration settings before creating a pilot directory."""
    harbor = Mock()
    monkeypatch.setattr(canary, "HarborCLI", harbor)
    output_dir = tmp_path / "pilot"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "canary",
            "--experiment",
            "tb2.1",
            "--api-base",
            "http://localhost:8000/v1",
            "--output-dir",
            str(output_dir),
            "--n-concurrent",
            str(n_concurrent),
        ],
    )
    with pytest.raises(SystemExit):
        canary.main()
    harbor.assert_not_called()
    assert not output_dir.exists()
