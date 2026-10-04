"""Exercise the real DecisionBench schemas/scorer with fakes only at I/O boundaries."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmark_model_fixtures import install_proposer

pytest.importorskip("decision_bench")

import litellm
from decision_bench.prompt import SYSTEM_PROMPT, render_user_prompt
from decision_bench.schemas import DecisionExample

from examples.common import benchmark_runner
from examples.common.benchmark_runner import build_parser, evaluate_candidate, resolve_models, validate_definition
from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL
from examples.common.provider_retries import ProviderRequestError, provider_retry_kwargs
from examples.decisionbench import main as entrypoint
from examples.decisionbench import upstream, utils
from examples.decisionbench.adapter import DecisionBenchAdapter, parse_prediction
from examples.decisionbench.benchmark_settings import COMPONENT, UPSTREAM_FILES
from examples.decisionbench.utils import decode_record, partition_records, source_keys, split_identity
from gepa import optimize


def stored_row(index=0, *, primitive="candidate_selection", candidates=3):
    """Create a storage row conforming to the published Parquet schema."""
    task = "choose"
    family = "reasoning"
    domain = "testing"
    return {
        "row_id": f"row-{index}",
        "task_id": f"{domain}/{family}/{primitive}/{task}",
        "task_name": task,
        "primitive": primitive,
        "family": family,
        "domain": domain,
        "candidate_count": candidates,
        "reasoning_required": True,
        "reasoning_type": "deduction",
        "instruction": "Choose the appropriate action.",
        "state_json": json.dumps({"context": f"Input {index}"}),
        "candidates_json": json.dumps(
            [
                {
                    "id": f"option-{i}",
                    "label": f"Action {i}",
                    "description": None,
                    "ordinal_value": float(20 - 10 * i) if primitive == "ordinal_scoring" else None,
                }
                for i in range(candidates)
            ]
        ),
        "gold_candidate_id": "option-1",
        "gold_probabilities": [float(i == 1) for i in range(candidates)],
        "source_json": json.dumps({"repo": "source/repo", "revision": "pinned", "source_id": str(index)}),
        "compact_instruction": "unused compact input",
        "compact_state_json": '"unused compact state"',
        "compact_candidates_json": "[]",
    }


def record(index=0, **kwargs):
    """Return a record with explicit training membership for adapter contract tests."""
    return {**decode_record(stored_row(index, **kwargs)), "split": "train", "group_id": f"group-{index}"}


def response(probabilities=(0.1, 0.8, 0.1), *, finish_reason="stop", content=None):
    """Build a provider completion without replacing any parsing or scoring code."""
    return {
        "model": "Qwen/Qwen3.8-27B",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": content if content is not None else json.dumps({"probabilities": probabilities})
                },
            }
        ],
    }


def models(**kwargs):
    """Supply the documented shared model transport fields without network access."""
    return SimpleNamespace(
        solver_model=QWEN3_8_27B_MODEL,
        proposer_model=DEEPSEEK_V4_1_FLASH_MODEL,
        solver_api_base="http://localhost:8000/v1",
        proposer_api_base="http://localhost:8001/v1",
        solver_kwargs={"temperature": 1.0, "max_tokens": 65536, "seed": 0, **kwargs},
        proposer_kwargs={},
    )


def test_runtime_is_exact_official_source():
    assert upstream.validate_upstream_runtime()["files"] == UPSTREAM_FILES


def test_runtime_rejects_revision_and_source_drift(monkeypatch):
    monkeypatch.setattr(upstream, "distribution", lambda _: SimpleNamespace(read_text=lambda _: "{}"))
    with pytest.raises(ValueError, match="Install decision-bench"):
        upstream.validate_upstream_runtime()
    monkeypatch.undo()
    monkeypatch.setattr(upstream, "UPSTREAM_FILES", {"prompt.py": "0" * 64})
    with pytest.raises(ValueError, match="source changed"):
        upstream.validate_upstream_runtime()


def test_loader_rejects_unpinned_artifact_before_decoding(tmp_path):
    path = tmp_path / "wrong.parquet"
    path.write_bytes(b"a changed dataset")
    with pytest.raises(ValueError, match="dataset bytes"):
        utils.load_decisionbench(path)


def test_storage_schema_preserves_metadata_and_uses_standard_input():
    row = stored_row()
    decoded = decode_record(row)
    assert decoded["id"] == decoded["row_id"] == row["row_id"]
    assert decoded["task_id"] == row["task_id"]
    assert decoded["family"] == row["family"]
    assert decoded["reasoning_required"] is True
    rendered = render_user_prompt(DecisionExample.model_validate(decoded["example"]))
    assert "Input 0" in rendered and "compact" not in rendered
    for forbidden in ("gold", "source_id", "row-0", "task_id"):
        assert forbidden not in rendered
    row["candidate_count"] = 2
    with pytest.raises(ValueError, match="metadata changed"):
        decode_record(row)


def test_partition_is_order_independent_and_keeps_transitive_source_groups_together():
    rows = [decode_record(stored_row(i)) for i in range(100)]
    rows[1]["example"]["source"]["source_id"] = "0"
    rows[2]["example"]["state"] = rows[1]["example"]["state"]
    rows[3]["example"]["source"] = {
        "source": {"dataset": "Team-ACE/ToolACE", "source_ids": ["conversation-1"], "raw_sha256": "a"}
    }
    rows[4]["example"]["source"] = {
        "source": {"dataset": "Team-ACE/ToolACE", "source_ids": ["conversation-1"], "raw_sha256": "b"}
    }
    splits = partition_records(rows)
    assert split_identity(splits) == split_identity(partition_records(list(reversed(rows))))
    flat = {row["id"]: row for values in splits.values() for row in values}
    assert len(flat) == len(rows)
    assert len({flat[f"row-{i}"]["group_id"] for i in range(3)}) == 1
    assert flat["row-3"]["group_id"] == flat["row-4"]["group_id"]
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not {row["group_id"] for row in splits[left]} & {row["group_id"] for row in splits[right]}
        assert not set().union(*(source_keys(row) for row in splits[left])) & set().union(
            *(source_keys(row) for row in splits[right])
        )
    with pytest.raises(ValueError, match="unique row IDs"):
        partition_records(rows + [rows[0]])


def test_partition_fails_without_source_lineage():
    row = record()
    row["example"]["source"] = {}
    with pytest.raises(ValueError, match="lineage"):
        partition_records([row])


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '{"probabilities":[0.2,0.8]}',
        '{"probabilities":[0,0,0]}',
        '{"probabilities":[-0.1,1,0.1]}',
        '{"probabilities":[NaN,0,1]}',
        '{"probabilities":[Infinity,0,1]}',
        '{"probabilities":[0,2,0]}',
        '{"probabilities":[true,false,false]}',
        '{"probabilities":["0.1", "0.8", "0.1"]}',
        '{"probabilities":[0.1,0.8,0.1],"selected":"option-1"}',
        '{"probabilities":[0,1,0],"probabilities":[1,0,0]}',
        '{"selected_candidate_id":"option-1"}',
    ],
)
def test_invalid_output_is_a_miss_with_no_fabricated_distribution(content):
    adapter = DecisionBenchAdapter(models(), training_ids={"row-0"}, completion=lambda **_: response(content=content))
    result = adapter.evaluate([record()], {COMPONENT: SYSTEM_PROMPT}, capture_traces=True)
    assert result.scores == [0.0]
    assert result.outputs[0]["error"]
    assert "scored" not in result.outputs[0]
    assert result.outputs[0]["elapsed_seconds"] >= 0
    summary = adapter.summarize_evaluation([record()], [result])["metrics"]["overall"]
    assert summary["accuracy"] == 0 and summary["coverage"] == 0
    assert summary["mean_negative_log_likelihood"] is None


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None, "tool_calls"])
def test_incomplete_response_cannot_succeed(finish_reason):
    with pytest.raises(ValueError, match="complete"):
        parse_prediction(response(finish_reason=finish_reason), 3)


def test_missing_completion_cannot_succeed():
    with pytest.raises(ValueError):
        parse_prediction({"choices": []}, 3)


@pytest.mark.parametrize(
    ("primitive", "count", "values", "expected_score", "expected_ordinal"),
    [
        ("binary_classification", 2, [0.2, 0.8], 1.0, None),
        ("candidate_selection", 3, [0.5, 0.5, 0.0], 0.0, None),
        ("ordinal_scoring", 3, [0.1, 0.6, 0.1], 1.0, 10.0),
        ("candidate_selection", 255, [1 / 255] * 255, 0.0, None),
    ],
)
def test_official_probability_accuracy_tie_and_ordinal_semantics(
    primitive, count, values, expected_score, expected_ordinal
):
    row = record(primitive=primitive, candidates=count)
    calls = []

    def send(**kwargs):
        calls.append(kwargs)
        return response(values)

    adapter = DecisionBenchAdapter(models(), training_ids={row["id"]}, completion=send)
    candidate = {COMPONENT: "Edited runtime instruction"}
    result = adapter.evaluate([row], candidate, capture_traces=True)
    assert result.scores == [expected_score]
    scored = result.outputs[0]["scored"]
    assert scored["probabilities"] == pytest.approx([value / sum(values) for value in values])
    assert scored["expected_ordinal_score"] == expected_ordinal
    assert calls[0]["messages"][0]["content"] == candidate[COMPONENT]
    assert calls[0]["model"] == QWEN3_8_27B_MODEL
    assert calls[0]["api_base"] == models().solver_api_base
    assert calls[0]["max_tokens"] == 65536
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["probabilities"]["minItems"] == count
    assert "gold" not in json.dumps(calls)
    summary = adapter.summarize_evaluation([row], [result])["metrics"]["overall"]
    assert summary["mean_negative_log_likelihood"] == pytest.approx(-math.log(scored["probabilities"][1]))
    assert summary["expected_calibration_error"] == pytest.approx(abs(expected_score - max(scored["probabilities"])))
    assert summary["ece_bins"] == 15


def test_errors_count_in_accuracy_and_coverage_but_not_successful_probability_metrics():
    rows = [record(0), record(1)]
    responses = iter([response(), response(content="invalid")])
    adapter = DecisionBenchAdapter(
        models(), training_ids={row["id"] for row in rows}, completion=lambda **_: next(responses)
    )
    result = adapter.evaluate(rows, {COMPONENT: SYSTEM_PROMPT})
    summary = adapter.summarize_evaluation(rows, [result])["metrics"]["overall"]
    assert summary["accuracy"] == 0.5
    assert summary["coverage"] == 0.5
    assert summary["supported_row_accuracy"] == 1
    assert summary["mean_negative_log_likelihood"] == pytest.approx(-math.log(0.8))


def test_prompt_mutations_reach_every_rollout_and_feedback_is_training_only():
    calls = []

    def send(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        return response()

    adapter = DecisionBenchAdapter(models(), training_ids={"row-0"}, completion=send)
    for text in (SYSTEM_PROMPT, "Changed prompt"):
        result = adapter.evaluate([record()], {COMPONENT: text}, capture_traces=True)
        assert adapter.make_reflective_dataset({COMPONENT: text}, result, [COMPONENT])[COMPONENT][0]["Feedback"][
            "correct"
        ]
    assert calls == [SYSTEM_PROMPT, "Changed prompt"]
    heldout = {**record(1), "split": "test"}
    result = adapter.evaluate([heldout], {COMPONENT: SYSTEM_PROMPT}, capture_traces=True)
    with pytest.raises(ValueError, match="cannot be used"):
        adapter.make_reflective_dataset({COMPONENT: SYSTEM_PROMPT}, result, [COMPONENT])


def test_summary_rejects_incomplete_misaligned_and_tampered_results():
    row = record()
    adapter = DecisionBenchAdapter(models(), training_ids={"row-0"}, completion=lambda **_: response())
    result = adapter.evaluate([row], {COMPONENT: SYSTEM_PROMPT})
    with pytest.raises(ValueError, match="Incomplete"):
        adapter.summarize_evaluation([row, record(1)], [result])
    with pytest.raises(ValueError, match="row order"):
        adapter.summarize_evaluation([record(1)], [result])
    with pytest.raises(ValueError, match="one attempt"):
        adapter.summarize_evaluation([row], [result, result])
    changed = deepcopy(result)
    changed.outputs[0]["scored"]["correct"] = False
    with pytest.raises(ValueError, match="official evaluator"):
        adapter.summarize_evaluation([row], [changed])
    changed = deepcopy(result)
    changed.outputs[0]["elapsed_seconds"] = math.nan
    with pytest.raises(ValueError, match="episode timing"):
        adapter.summarize_evaluation([row], [changed])


def test_shared_provider_retries_apply_to_incomplete_responses(monkeypatch, tmp_path):
    calls = []

    def send(**kwargs):
        calls.append(kwargs)
        return response(finish_reason="length" if len(calls) == 1 else "stop")

    monkeypatch.setattr(litellm, "completion", send)
    monkeypatch.setattr("examples.common.provider_retries.time.sleep", lambda _: None)
    kwargs = provider_retry_kwargs(tmp_path / "attempts.jsonl", role="solver")
    adapter = DecisionBenchAdapter(models(**kwargs), training_ids={"row-0"})
    result = adapter.evaluate([record()], {COMPONENT: SYSTEM_PROMPT})
    assert result.scores == [1.0]
    assert len(calls) == 2
    assert calls[0]["seed"] == 0 and calls[1]["seed"] == 1
    assert all(call["max_retries"] == 0 for call in calls)
    assert result.outputs[0]["elapsed_seconds"] >= 0
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 2


def test_exhausted_provider_aborts_instead_of_fabricating_a_completed_run():
    def send(**_):
        raise ProviderRequestError("exhausted")

    adapter = DecisionBenchAdapter(models(), training_ids={"row-0"}, completion=send)
    with pytest.raises(ProviderRequestError):
        adapter.evaluate([record()], {COMPONENT: SYSTEM_PROMPT})


def test_real_gepa_optimizes_the_executed_prompt_without_test_feedback(tmp_path):
    calls = []
    reflections = []

    def send(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        calls.append(prompt)
        return response((0.1, 0.8, 0.1) if prompt == "Improved instruction" else (0.8, 0.1, 0.1))

    def reflect(prompt):
        reflections.append(str(prompt))
        return "```\nImproved instruction\n```"

    adapter = DecisionBenchAdapter(models(), training_ids={"row-0"}, completion=send)
    result = optimize(
        adapter=adapter,
        seed_candidate={COMPONENT: "Initial instruction"},
        trainset=[record(0)],
        valset=[{**record(1), "split": "val"}],
        reflection_lm=reflect,
        reflection_minibatch_size=1,
        max_metric_calls=4,
        run_dir=str(tmp_path),
        use_merge=False,
        seed=0,
    )
    assert {COMPONENT: "Improved instruction"} in result.candidates
    assert "Improved instruction" in calls
    assert any("Input 0" in prompt for prompt in reflections)
    assert all("Input 1" not in prompt for prompt in reflections)


def test_shared_builder_preserves_complete_splits_and_model_defaults(monkeypatch):
    splits = partition_records([decode_record(stored_row(i)) for i in range(100)])
    monkeypatch.setattr(entrypoint, "load_decisionbench", lambda *_: (splits, {"fixture": "dataset I/O boundary"}))
    parser = build_parser("decisionbench", entrypoint.add_arguments)
    defaults = parser.parse_args([])
    assert (defaults.train_limit, defaults.val_limit, defaults.test_limit) == (150, 300, 300)
    args = parser.parse_args(["--train-limit", "1", "--val-limit", "1", "--test-limit", "1"])
    resolved = resolve_models(args)
    benchmark = entrypoint.build_benchmark(args, resolved)
    validate_definition(benchmark)
    assert resolved.solver_model == QWEN3_8_27B_MODEL
    assert resolved.proposer_model == DEEPSEEK_V4_1_FLASH_MODEL
    assert len(benchmark.trainset) > args.train_limit
    assert len(benchmark.valset) > args.val_limit
    assert len(benchmark.testset) > args.test_limit
    assert SYSTEM_PROMPT in benchmark.seed_candidate[COMPONENT]
    assert benchmark.component_kinds == {COMPONENT: "system_prompt"}
    assert benchmark.runtime["official_runtime"]["files"] == UPSTREAM_FILES


def test_shared_evaluator_reuses_verified_results_and_rejects_prompt_drift(monkeypatch, tmp_path):
    splits = partition_records([decode_record(stored_row(i)) for i in range(100)])
    monkeypatch.setattr(entrypoint, "load_decisionbench", lambda *_: (splits, {"fixture": "dataset I/O boundary"}))
    args = build_parser("decisionbench", entrypoint.add_arguments).parse_args([])
    benchmark = entrypoint.build_benchmark(args, resolve_models(args))
    calls = []

    def send(**kwargs):
        calls.append(kwargs)
        return response()

    benchmark.adapter.completion = send
    candidate = benchmark.seed_candidate
    kwargs = {
        "definition": benchmark,
        "candidate": candidate,
        "records": benchmark.testset[:2],
        "directory": tmp_path,
        "identity": {"source": benchmark.source, "runtime": benchmark.runtime},
        "split": "test",
        "repetitions": 1,
        "seed": 0,
    }
    first = evaluate_candidate(**kwargs)
    second = evaluate_candidate(**kwargs)
    assert len(calls) == 2 and first == second
    assert first["mean_score"] == 1
    assert first["metrics"]["metrics"]["overall"]["accuracy"] == 1
    timing = first["timing"]
    assert timing["attempt_count"] == 2 and timing["failed_attempts"] == 0
    assert timing["p95_seconds"] >= timing["median_seconds"] > 0
    assert timing["tasks_per_hour"] == pytest.approx(7200 / timing["recorded_batch_seconds"])
    with pytest.raises(ValueError, match="configuration or data changed"):
        evaluate_candidate(**{**kwargs, "candidate": {COMPONENT: "different"}})


@pytest.mark.parametrize("condition", ["vanilla", "random", "action", "react_v2_random", "react_v2"])
def test_real_optimizer_pilot_scores_changed_prompt_using_training_only(monkeypatch, tmp_path, condition):
    """Run every real optimizer variant through official decision rendering and scoring."""
    splits = partition_records([decode_record(stored_row(i)) for i in range(100)])
    monkeypatch.setattr(entrypoint, "load_decisionbench", lambda *_: (splits, {"fixture": "dataset I/O boundary"}))
    requests = []
    seeds = []

    def build(args, resolved):
        benchmark = entrypoint.build_benchmark(args, resolved)
        initial = benchmark.seed_candidate[COMPONENT]
        seeds.append(initial)

        def send(**kwargs):
            requests.append(kwargs)
            changed = kwargs["messages"][0]["content"] != initial
            return response((0.1, 0.8, 0.1) if changed else (0.8, 0.1, 0.1))

        benchmark.adapter.completion = send
        return benchmark

    proposers = install_proposer(monkeypatch)
    run_dir = tmp_path / "run"
    assert (
        benchmark_runner.run_cli(
            benchmark_name="decisionbench",
            build_benchmark=build,
            add_arguments=entrypoint.add_arguments,
            argv=[
                "--mode",
                "optimizer-pilot",
                "--condition",
                condition,
                "--pilot-size",
                "1",
                "--pilot-proposals",
                "1",
                "--run-dir",
                str(run_dir),
            ],
        )
        == 0
    )
    summary = json.loads((run_dir / "optimizer-pilot" / condition / "summary.json").read_text())
    expected_user = render_user_prompt(DecisionExample.model_validate(splits["train"][0]["example"]))
    assert summary["winner"]["selection_split"] == "train"
    assert summary["winner"]["training_score"] == 1.0
    assert requests and all(request["messages"][1]["content"] == expected_user for request in requests)
    assert any(request["messages"][0]["content"] != seeds[0] for request in requests)
    assert all(request["response_format"]["json_schema"]["strict"] for request in requests)
    assert any(proposer.calls for proposer in proposers)
    assert "test" not in summary and "baseline" not in summary
    assert not list(run_dir.rglob("heldout"))


@pytest.mark.smoke
def test_pinned_canonical_parquet_and_real_builder_without_inference():
    path = os.environ.get("DECISIONBENCH_DATA_FILE")
    if not path:
        pytest.skip("Set DECISIONBENCH_DATA_FILE to the pinned local Parquet; this test never calls a model")
    args = build_parser("decisionbench", entrypoint.add_arguments).parse_args(["--data-file", path])
    benchmark = entrypoint.build_benchmark(args, resolve_models(args))
    validate_definition(benchmark)
    assert [len(benchmark.trainset), len(benchmark.valset), len(benchmark.testset)] == [14331, 4752, 4817]
    for rows, limit in ((benchmark.trainset, 150), (benchmark.valset, 300), (benchmark.testset, 300)):
        assert len({row["task_id"] for row in rows[:limit]}) == 43
    keys = {
        name: set().union(*(source_keys(row) for row in rows))
        for name, rows in (("train", benchmark.trainset), ("val", benchmark.valset), ("test", benchmark.testset))
    }
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert Path(args.data_file).is_file()
