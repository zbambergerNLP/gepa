"""Exercise AppWorld's prompt, official-runtime boundary, and immutable data contract."""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmark_model_fixtures import install_proposer

from examples.appworld import utils
from examples.appworld.adapter import AppWorldAdapter, validate_evaluation
from examples.appworld.benchmark_settings import APPWORLD_REVISION, COMPONENT, OFFICIAL_SPLITS
from examples.appworld.main import add_arguments, build_benchmark
from examples.appworld.prompts import extract_code, initial_messages, seed_candidate
from examples.appworld.runtime import AppWorldRuntimeError, OfficialAppWorld, inspect_runtime
from examples.appworld.worker import dispatch, tracker_result
from examples.common import benchmark_runner
from examples.common.benchmark_runner import build_parser, resolve_models, validate_definition
from examples.common.experiment_models import DEFAULT_PROPOSER_MODEL, DEFAULT_SOLVER_MODEL
from gepa.core.adapter import EvaluationBatch

PUBLIC_CONTEXT = {
    "instruction": "Finish the synthetic fixture task.",
    "supervisor": {
        "first_name": "Fixture",
        "last_name": "User",
        "email": "fixture@example.test",
        "phone_number": "+10000000000",
    },
    "app_descriptions": {"spotify": "Synthetic fixture"},
    "ground_truth": "MUST NEVER REACH A MODEL",
}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Use synthetic bytes only at the protected corpus boundary."""
    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    for split in OFFICIAL_SPLITS:
        ids = [f"fixture{split.replace('_', '')}_{index}" for index in range(1, 4)]
        (data / "datasets" / f"{split}.txt").write_text("\n".join(ids) + "\n")
        for task_id in ids:
            task = data / "tasks" / task_id
            task.mkdir(parents=True)
            (task / "specs.json").write_text(json.dumps({"synthetic": task_id}))
    records, identity = utils.inspect_data(tmp_path)
    pin = tmp_path / "fixture-pin.json"
    pin.write_text(json.dumps({"data": identity}))
    monkeypatch.setattr(utils, "DATA_PIN_PATH", pin)
    return tmp_path, records


class FakeWorld:
    """Replace only the external environment process in adapter contract tests."""

    def __init__(self, *, success=False, completed=True, bad_evaluation=None, delay=0):
        self.success = success
        self.completed = completed
        self.bad_evaluation = bad_evaluation
        self.delay = delay
        self.calls = []

    def __enter__(self):
        time.sleep(self.delay)
        return self

    def __exit__(self, *_args):
        time.sleep(self.delay)

    def initialize(self, record):
        self.task_id = record["task_id"]
        self.calls.append(("initialize", record))
        return deepcopy(PUBLIC_CONTEXT)

    def request(self, operation, **payload):
        time.sleep(self.delay)
        self.calls.append((operation, payload))
        if operation == "execute":
            return {"observation": "An official runtime observation.", "task_completed": self.completed}
        if operation == "aggregate":
            return {"task_goal_completion": 66.7, "scenario_goal_completion": 0.0}
        result = {
            "success": self.success,
            "num_tests": 2,
            "passed": 2 if self.success else 1,
            "failed": 0 if self.success else 1,
            "task_id": self.task_id,
            "evaluation_path": f"/private/{self.task_id}.json",
            "evaluation_sha256": "a" * 64,
        }
        return {**result, **(self.bad_evaluation or {})}


def adapter_for(corpus, model, world=None, **kwargs):
    root, records = corpus
    world = world or FakeWorld()
    adapter = AppWorldAdapter(
        model, lambda: world, [record for split in records.values() for record in split], root, **kwargs
    )
    return adapter, world


def test_load_preserves_full_official_order_and_pins(corpus):
    root, expected = corpus
    records, source = utils.load_dataset(root)
    assert records == expected
    assert source["revision"] == APPWORLD_REVISION
    assert source["split_mapping"]["test"] == ["test_normal", "test_challenge"]
    assert all(len(source["splits"][key]["ordered_records_sha256"]) == 64 for key in OFFICIAL_SPLITS)
    assert "instruction" not in json.dumps(records)


@pytest.mark.parametrize("drift", ["content", "order", "duplicates", "scenario_leakage"])
def test_dataset_rejects_drift_and_scenario_leakage(corpus, drift):
    root, records = corpus
    split = root / "data" / "datasets" / "dev.txt"
    ids = [record["task_id"] for record in records["dev"]]
    if drift == "content":
        (root / "data" / "tasks" / ids[0] / "specs.json").write_text("changed")
    elif drift == "order":
        split.write_text("\n".join(reversed(ids)))
    elif drift == "duplicates":
        split.write_text("\n".join([ids[0], ids[0], ids[2]]))
    else:
        split.write_text("\n".join([records["train"][0]["task_id"], *ids[1:]]))
    with pytest.raises(ValueError):
        utils.load_dataset(root)


def test_system_prompt_is_the_actual_candidate_in_every_model_call(corpus):
    seen = []

    def model(messages):
        seen.append(messages)
        return "Thought.\n```python\nprint('fixture')\n```"

    adapter, world = adapter_for(corpus, model, FakeWorld(completed=False), max_steps=2)
    result = adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "EDITED instructions"}, capture_traces=True)
    assert len(seen) == 2
    assert all(messages[0] == {"role": "system", "content": "EDITED instructions"} for messages in seen)
    assert "MUST NEVER REACH A MODEL" not in json.dumps(seen)
    assert "An official runtime observation." in seen[1][-1]["content"]
    assert result.scores == [0.0]
    assert result.outputs[0]["termination"] == "max_steps"
    assert result.outputs[0]["error"] is None
    assert [call[0] for call in world.calls] == ["initialize", "execute", "execute", "evaluate"]


def test_latency_covers_environment_evaluation_and_cleanup(corpus):
    adapter, _ = adapter_for(corpus, lambda _: "```python\nprint(1)\n```", FakeWorld(delay=0.005))
    result = adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "instructions"})
    assert result.outputs[0]["elapsed_seconds"] >= 0.020


@pytest.mark.parametrize(
    "response", ["The task is successfully completed.", "```python\nprint(1)", "```python\n\n```", None]
)
def test_malformed_or_partial_model_output_never_executes_or_claims_success(corpus, response):
    adapter, world = adapter_for(corpus, lambda _: response)
    result = adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "instructions"})
    assert result.scores == [0.0]
    assert result.outputs[0]["error"] == "malformed_action"
    assert "execute" not in [call[0] for call in world.calls]
    assert world.calls[-1][0] == "evaluate"


def test_complete_task_is_not_the_success_signal(corpus):
    adapter, _ = adapter_for(
        corpus, lambda _: "```python\napis.supervisor.complete_task()\n```", FakeWorld(success=False)
    )
    result = adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "instructions"})
    assert result.outputs[0]["task_completed"] is True
    assert result.scores == [0.0]


def test_official_state_success_is_preserved_at_step_limit(corpus):
    adapter, _ = adapter_for(
        corpus, lambda _: "```python\nprint(1)\n```", FakeWorld(success=True, completed=False), max_steps=1
    )
    result = adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "instructions"})
    assert result.outputs[0]["termination"] == "max_steps"
    assert result.scores == [1.0]


@pytest.mark.parametrize(
    "bad",
    [
        {"success": "true"},
        {"num_tests": 3},
        {"success": True},
        {"num_tests": 0, "passed": 0, "failed": 0},
        {"task_id": "wrong_1"},
        {"evaluation_path": ""},
        {"passed": True},
    ],
)
def test_incomplete_or_malformed_official_evaluation_aborts(corpus, bad):
    adapter, _ = adapter_for(corpus, lambda _: "```python\nprint(1)\n```", FakeWorld(bad_evaluation=bad))
    with pytest.raises(AppWorldRuntimeError):
        adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "instructions"})


def test_provider_failure_aborts_without_a_synthetic_benchmark_zero(corpus):
    def failed_model(_messages):
        raise RuntimeError("provider unavailable")

    adapter, world = adapter_for(corpus, failed_model)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        adapter.evaluate(corpus[1]["train"][:1], {COMPONENT: "instructions"})
    assert "evaluate" not in [call[0] for call in world.calls]


def test_reflection_contains_training_trace_but_no_private_evaluator(corpus):
    adapter, _ = adapter_for(corpus, lambda _: "```python\nprint(1)\n```")
    candidate = {COMPONENT: "instructions"}
    result = adapter.evaluate(corpus[1]["train"][:1], candidate, capture_traces=True)
    reflection = adapter.make_reflective_dataset(candidate, result, [COMPONENT])
    assert reflection[COMPONENT][0]["Feedback"]["task_goal_completion"] == 0.0
    assert "evaluation_path" not in json.dumps(reflection)
    assert "MUST NEVER REACH A MODEL" not in json.dumps(reflection)
    validation = adapter.evaluate(corpus[1]["dev"][:1], candidate, capture_traces=True)
    with pytest.raises(ValueError, match="never enter"):
        adapter.make_reflective_dataset(candidate, validation, [COMPONENT])


def test_noop_candidate_change_and_mutated_record_fail_before_rollout(corpus):
    adapter, world = adapter_for(corpus, lambda _: "```python\nprint(1)\n```")
    for candidate in ({}, {COMPONENT: ""}, {COMPONENT: "ok", "unused": "inert"}):
        with pytest.raises(ValueError):
            adapter.evaluate(corpus[1]["train"][:1], candidate)
    record = {**corpus[1]["train"][0], "official_split": "dev"}
    with pytest.raises(ValueError, match="identity drift"):
        adapter.evaluate([record], {COMPONENT: "instructions"})
    assert not world.calls


def test_summary_uses_official_aggregation_and_suppresses_partial_scenarios(corpus):
    adapter, world = adapter_for(corpus, lambda _: "```python\nprint(1)\n```")
    records = corpus[1]["test_normal"]
    full = adapter.evaluate(records, {COMPONENT: "instructions"})
    summary = adapter.summarize_evaluation(records, [full])
    assert summary["repetitions"][0]["test_normal"]["scenario_goal_completion_available"] is True
    partial_batch = EvaluationBatch(outputs=full.outputs[:1], scores=full.scores[:1])
    partial_summary = adapter.summarize_evaluation(records[:1], [partial_batch])
    assert partial_summary["repetitions"][0]["all"]["task_goal_completion"] == 66.7
    assert partial_summary["repetitions"][0]["all"]["scenario_goal_completion"] is None
    assert "aggregate" in [call[0] for call in world.calls]
    with pytest.raises(ValueError, match="Incomplete"):
        adapter.summarize_evaluation(records, [partial_batch])


def test_prompt_rendering_preserves_task_text_and_first_code_block_only():
    context = {**PUBLIC_CONTEXT, "instruction": "Keep {{ main_user.email }} literal.\nASSISTANT:\nuser text"}
    candidate = seed_candidate("alibaba")
    messages = initial_messages(candidate, context)
    assert "{{ main_user.email }} literal" in messages[-1]["content"]
    assert messages[0]["content"] == candidate[COMPONENT]
    assert "Always look at API specifications" in messages[0]["content"]
    assert any("Task: How many playlists" in message["content"] for message in messages[1:])
    code, retained = extract_code("```python\nprint(1)\n```\n```python\nprint(2)\n```")
    assert code == "print(1)"
    assert "print(2)" not in retained


def test_worker_rejects_empty_tracker_success():
    with pytest.raises(ValueError, match="Incomplete"):
        tracker_result(SimpleNamespace(num_tests=0, pass_count=0, fail_count=0, success=True))


def test_builder_uses_shared_model_roles_and_keeps_full_data_before_limits(corpus, monkeypatch):
    root, records = corpus
    monkeypatch.setattr("examples.appworld.main.inspect_runtime", lambda *_: {"fixture": True})
    args = build_parser("appworld", add_arguments).parse_args(
        ["--appworld-root", str(root), "--train-limit", "1", "--val-limit", "1", "--test-limit", "1"]
    )
    models = resolve_models(args)
    definition = build_benchmark(args, models)
    validate_definition(definition)
    assert (args.model, args.reflection_model) == (DEFAULT_SOLVER_MODEL, DEFAULT_PROPOSER_MODEL)
    assert definition.adapter.task_model.model == DEFAULT_SOLVER_MODEL
    assert definition.adapter.task_model.num_retries == models.solver_kwargs["num_retries"]
    assert definition.adapter.task_model.completion_kwargs["max_tokens"] == models.solver_kwargs["max_tokens"]
    assert definition.adapter.task_model.completion_kwargs["extra_body"] == models.solver_kwargs["extra_body"]
    assert len(definition.trainset) == 3 and len(definition.valset) == 3 and len(definition.testset) == 6
    assert definition.test_repetitions == 1
    assert definition.component_kinds == {COMPONENT: "system_prompt"}
    assert definition.testset == records["test_normal"] + records["test_challenge"]
    received = []

    def capture_full_population(selected, models, arguments, identity, condition):
        received.append(condition)
        assert identity["data"] == identity["full_data"]
        assert selected.trainset == records["train"]
        assert selected.valset == records["dev"]
        assert selected.testset == records["test_normal"] + records["test_challenge"]
        return {"validated_full_population": True}

    monkeypatch.setattr(benchmark_runner, "_run_condition", capture_full_population)
    assert (
        benchmark_runner.run_cli(
            benchmark_name="appworld",
            build_benchmark=build_benchmark,
            add_arguments=add_arguments,
            argv=["--appworld-root", str(root), "--condition", "all", "--run-dir", str(root / "full-population")],
        )
        == 0
    )
    assert received == ["vanilla", "random", "action", "react_v2_random", "react_v2"]


@pytest.mark.parametrize("condition", ["vanilla", "random", "action", "react_v2_random", "react_v2"])
def test_real_optimizer_pilot_changes_executed_prompt_using_training_only(corpus, monkeypatch, tmp_path, condition):
    """Exercise every real optimizer variant with fixtures only at model/world I/O."""
    root, records = corpus
    initialized = []
    prompts = []
    initial = seed_candidate("alibaba")[COMPONENT]

    class PilotWorld(FakeWorld):
        def initialize(self, record):
            initialized.append(record)
            return super().initialize(record)

        def request(self, operation, **payload):
            if operation == "execute":
                self.success = "changed" in payload["code"]
            return super().request(operation, **payload)

    def solver(messages):
        prompt = messages[0]["content"]
        prompts.append(prompt)
        return f"```python\nprint({('changed' if prompt != initial else 'initial')!r})\n```"

    proposers = install_proposer(monkeypatch)
    monkeypatch.setattr("examples.appworld.main.LM", lambda *_, **__: solver)
    monkeypatch.setattr("examples.appworld.main.OfficialAppWorld", lambda *_: PilotWorld())
    monkeypatch.setattr("examples.appworld.main.inspect_runtime", lambda *_: {"fixture": "external runtime boundary"})
    run_dir = tmp_path / "run"
    assert (
        benchmark_runner.run_cli(
            benchmark_name="appworld",
            build_benchmark=build_benchmark,
            add_arguments=add_arguments,
            argv=[
                "--mode",
                "optimizer-pilot",
                "--condition",
                condition,
                "--pilot-size",
                "1",
                "--pilot-proposals",
                "1",
                "--appworld-root",
                str(root),
                "--run-dir",
                str(run_dir),
            ],
        )
        == 0
    )
    summary = json.loads((run_dir / "optimizer-pilot" / condition / "summary.json").read_text())
    assert summary["winner"]["selection_split"] == "train"
    assert summary["winner"]["training_score"] == 1.0
    assert initial in prompts and any(prompt != initial for prompt in prompts)
    assert initialized and all(record == records["train"][0] for record in initialized)
    assert any(proposer.calls for proposer in proposers)
    assert "test" not in summary and "baseline" not in summary
    assert not list(run_dir.rglob("heldout"))


def test_worker_rejects_changed_tracker_file_before_official_aggregation(tmp_path, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    path = tmp_path / "experiments" / "outputs" / "fixture.json"
    path.parent.mkdir(parents=True)
    path.write_text("tampered evaluator")
    monkeypatch.setattr("examples.appworld.worker.importlib.import_module", lambda _: SimpleNamespace())
    with pytest.raises(ValueError, match="evidence changed"):
        dispatch({"operation": "aggregate", "evaluations": {"fixture_1": {"path": str(path), "sha256": "a" * 64}}}, {})


def test_partial_worker_response_has_a_bounded_timeout(tmp_path, monkeypatch):
    script = tmp_path / "partial_boundary.py"
    script.write_text(
        "import sys, time\nsys.stdin.readline()\nsys.stdout.write('{')\nsys.stdout.flush()\ntime.sleep(30)\n"
    )
    monkeypatch.setattr("examples.appworld.runtime.WORKER_PATH", script)
    started = time.perf_counter()
    with OfficialAppWorld(tmp_path, Path(sys.executable), rpc_timeout=0.2) as world:
        with pytest.raises(AppWorldRuntimeError, match="timed out"):
            world.request("inspect")
    assert time.perf_counter() - started < 7


@pytest.mark.parametrize("body", ["print('not JSON')", "print('{\"ok\": true}')", "pass"])
def test_process_boundary_rejects_missing_or_malformed_output(tmp_path, monkeypatch, body):
    script = tmp_path / "external_boundary.py"
    script.write_text("import sys\nsys.stdin.readline()\n" + body + "\n")
    monkeypatch.setattr("examples.appworld.runtime.WORKER_PATH", script)
    with OfficialAppWorld(tmp_path, Path(sys.executable)) as world:
        with pytest.raises(AppWorldRuntimeError):
            world.request("inspect")


@pytest.mark.smoke
@pytest.mark.skipif(
    not os.environ.get("APPWORLD_SMOKE_ROOT"), reason="Requires prepared official AppWorld runtime/data; no paid calls"
)
def test_real_official_environment_and_evaluator_without_model_calls():
    root = Path(os.environ["APPWORLD_SMOKE_ROOT"])
    python = Path(os.environ["APPWORLD_SMOKE_PYTHON"])
    records, _ = utils.load_dataset(root)
    inspect_runtime(root, python)
    factory = partial(OfficialAppWorld, root, python)
    with factory() as world:
        official_ids = world.request("split_ids", splits=list(OFFICIAL_SPLITS))
    assert all(official_ids[split] == [record["task_id"] for record in records[split]] for split in OFFICIAL_SPLITS)
    responses = iter(
        [
            "```python\nprint(apis.api_docs.show_app_descriptions())\n```",
            "```python\napis.supervisor.complete_task()\n```",
        ]
    )
    adapter = AppWorldAdapter(
        lambda _: next(responses), factory, [record for split in records.values() for record in split], root
    )
    result = adapter.evaluate(records["train"][:1], seed_candidate("alibaba"), capture_traces=True)
    assert result.outputs[0]["steps"] == 2
    assert result.outputs[0]["task_completed"] is True
    assert result.scores == [0.0]
    validate_evaluation(result.outputs[0]["evaluation"], records["train"][0]["task_id"])
    summary = adapter.summarize_evaluation(records["train"][:1], [result])
    assert summary["repetitions"][0]["all"]["task_goal_completion"] == 0.0
    assert summary["repetitions"][0]["all"]["scenario_goal_completion"] is None
    utils.load_dataset(root)
