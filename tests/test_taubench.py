"""Contract tests at the external tau worker boundary; no paid calls."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.taubench.adapter import TauBankingAdapter, trial_seed, validate_outputs
from examples.taubench.benchmark_settings import MANIFEST_PATH, UPSTREAM_REVISION
from examples.taubench.runtime import TauRuntime, worker_command
from examples.taubench.utils import digest, load_data, task_groups
from gepa.core.adapter import EvaluationBatch


@pytest.fixture
def manifest():
    return json.loads(MANIFEST_PATH.read_text())


class FakeRuntime:
    """Replace only the external simulation process in adapter contract tests."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.calls = []

    def run(self, records, candidate, trial):
        self.calls.append((deepcopy(records), deepcopy(candidate), trial))
        return {
            "manifest_sha256": digest(self.manifest),
            "outputs": [
                {
                    "id": r["id"],
                    "task_id": r["task_id"],
                    "trial": trial,
                    "seed": trial_seed(trial),
                    "elapsed_seconds": 4.5,
                    "reward": 0.0,
                    "termination_reason": "user_stop",
                    "error": None,
                    "candidate_sha256": digest(candidate),
                    "messages": [{"role": "assistant", "content": "test"}],
                }
                for r in records
            ],
        }


def test_pinned_manifest_and_group_separation(manifest):
    assert manifest["revision"] == UPSTREAM_REVISION
    assert manifest["knowledge"]["count"] == 698
    assert len(manifest["records"]) == 97
    assert {s: sum(r["split"] == s for r in manifest["records"]) for s in ("train", "val", "test")} == {
        "train": 57,
        "val": 19,
        "test": 21,
    }
    groups = {}
    for r in manifest["records"]:
        assert not {"user_scenario", "evaluation_criteria", "required_documents", "initial_state"} & r.keys()
        assert groups.setdefault(r["group_id"], r["split"]) == r["split"]
    lookup = {r["task_id"]: r for r in manifest["records"]}
    assert lookup["task_018"]["group_id"] == lookup["task_022"]["group_id"]
    assert lookup["task_080"]["group_id"] == lookup["task_081"]["group_id"]


def test_groups_connect_customer_and_explicit_variant_transitively():
    tasks = {f"task_00{i}": {"user_scenario": str(i), "description": {"notes": None}} for i in range(1, 5)}
    tasks["task_001"]["evaluation_criteria"] = {"user_id": "same-user"}
    tasks["task_002"]["initial_state"] = {"user_id": "same-user"}
    tasks["task_003"]["description"]["notes"] = "Variant of task_002"
    groups = task_groups(tasks)
    assert groups["task_001"] == groups["task_002"] == groups["task_003"]
    assert groups["task_004"] != groups["task_001"]


@pytest.mark.parametrize(
    "field", ["knowledge", "source", "database_sha256", "records", "user_simulator", "lock_sha256"]
)
def test_loader_rejects_every_identity_drift(monkeypatch, manifest, field):
    changed = deepcopy(manifest)
    changed[field] = "changed"
    monkeypatch.setattr("examples.taubench.utils.inspect_source", lambda source: changed)
    with pytest.raises(ValueError, match="identity changed"):
        load_data(Path("unused"))


def test_every_candidate_reaches_runtime_and_reflection_uses_only_actual_training_dialogue(manifest):
    runtime = FakeRuntime(manifest)
    adapter = TauBankingAdapter(runtime, manifest, max_workers=2)
    train = [r for r in manifest["records"] if r["split"] == "train"][:3]
    for prompt in ("first actual system prompt", "changed actual system prompt"):
        candidate = {"system_prompt": prompt}
        batch = adapter.evaluate(train, candidate, capture_traces=True)
        assert all(call[1] == candidate for call in runtime.calls[-2:])
        assert [o["id"] for o in batch.outputs] == [r["id"] for r in train]
        assert batch.scores == [0, 0, 0]
        data = adapter.make_reflective_dataset(candidate, batch, ["system_prompt"])
        serialized = json.dumps(data)
        assert "official_reward" in serialized
        assert "evaluation_criteria" not in serialized and "required_documents" not in serialized
    test = [r for r in manifest["records"] if r["split"] == "test"][:1]
    with pytest.raises(ValueError, match="Held-out traces"):
        adapter.evaluate(test, {"system_prompt": "valid"}, capture_traces=True)
    batch.trajectories[0]["id"] = test[0]["id"]
    with pytest.raises(ValueError, match="cannot enter reflection"):
        adapter.make_reflective_dataset(candidate, batch, ["system_prompt"])


def test_repetition_context_is_explicit_and_resumable(manifest):
    runtime = FakeRuntime(manifest)
    adapter = TauBankingAdapter(runtime, manifest)
    records = [r for r in manifest["records"] if r["split"] == "test"][:2]
    adapter.set_evaluation_context(split="test", repetition=3, seed=12345)
    result = adapter.evaluate(records, {"system_prompt": "a"})
    assert all(output["trial"] == 3 for output in result.outputs)
    assert all(output["seed"] == trial_seed(3) for output in result.outputs)
    with pytest.raises(ValueError, match="context"):
        adapter.evaluate([manifest["records"][0]], {"system_prompt": "a"})


@pytest.mark.parametrize(
    "change",
    [
        {"reward": float("nan")},
        {"reward": True},
        {"reward": 1.1},
        {"elapsed_seconds": -1},
        {"elapsed_seconds": float("inf")},
        {"trial": 5},
        {"id": "substituted"},
        {"candidate_sha256": "other"},
        {"messages": []},
        {"termination_reason": "infrastructure_error"},
        {"termination_reason": "max_steps", "reward": 1, "error": "max_steps"},
    ],
)
def test_malformed_or_incomplete_rewards_fail_closed(manifest, change):
    records = manifest["records"][:1]
    candidate = {"system_prompt": "real"}
    outputs = FakeRuntime(manifest).run(records, candidate, 0)["outputs"]
    outputs[0].update(change)
    with pytest.raises(ValueError):
        validate_outputs(records, outputs, digest(candidate), 0)


def test_missing_attempt_not_silently_scored(manifest):
    with pytest.raises(ValueError, match="Incomplete"):
        validate_outputs(manifest["records"][:1], [], digest({"system_prompt": "a"}), 0)


def test_tau_pass_hat_k_is_probability_all_k_succeed(manifest):
    adapter = TauBankingAdapter(FakeRuntime(manifest), manifest)
    records = [r for r in manifest["records"] if r["split"] == "test"][:2]
    evaluations = []
    for trial in range(4):
        result = FakeRuntime(manifest).run(records, {"system_prompt": "p"}, trial)
        scores = [1.0, float(trial < 2)]
        for score, output in zip(scores, result["outputs"], strict=True):
            output["reward"] = score
        evaluations.append(EvaluationBatch(outputs=result["outputs"], scores=scores))
    summary = adapter.summarize_evaluation(records, evaluations)
    assert summary["pass_hat_k"] == {"1": 0.75, "2": pytest.approx(7 / 12), "3": 0.5, "4": 0.5}
    with pytest.raises(ValueError, match="complete"):
        adapter.summarize_evaluation(records, evaluations[:3])
    evaluations[2].outputs[0]["seed"] = 0
    with pytest.raises(ValueError, match="repetition mismatch"):
        adapter.summarize_evaluation(records, evaluations)


def test_worker_isolation_and_malformed_process_output(monkeypatch, tmp_path):
    command = worker_command(tmp_path)
    assert "--frozen" in command and "websockets==15.0.1" in command and "3.12.11" in command
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/bad-parent-environment")

    def fake_process(command, **kwargs):
        assert "UV_PROJECT_ENVIRONMENT" not in kwargs["env"]
        assert kwargs["env"]["TAU2_DATA_DIR"] == str(tmp_path / "data")
        return SimpleNamespace(returncode=0, stdout='{"outputs":', stderr="")

    monkeypatch.setattr("examples.taubench.runtime.subprocess.run", fake_process)
    with pytest.raises(ValueError, match="malformed"):
        TauRuntime(tmp_path, tmp_path / "artifacts", "solver", None, {}).invoke({"mode": "inspect"})


def test_validation_traces_can_be_archived_but_never_reflected(manifest):
    adapter = TauBankingAdapter(FakeRuntime(manifest), manifest)
    records = [r for r in manifest["records"] if r["split"] == "val"][:1]
    batch = adapter.evaluate(records, {"system_prompt": "a"}, capture_traces=True)
    assert batch.trajectories
    with pytest.raises(ValueError, match="cannot enter reflection"):
        adapter.make_reflective_dataset({"system_prompt": "a"}, batch, ["system_prompt"])


def test_shared_cli_models_full_data_and_real_optimizer_cycle(manifest, tmp_path, monkeypatch):
    pytest.importorskip("litellm")
    from examples.common import benchmark_runner as runner
    from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL
    from examples.common.react_v2 import structured_prompt
    from examples.taubench import main

    splits = {s: [r for r in manifest["records"] if r["split"] == s] for s in ("train", "val", "test")}
    monkeypatch.setattr(main, "load_data", lambda source: (deepcopy(splits), deepcopy(manifest)))
    monkeypatch.setattr(main, "upstream_system_prompt", lambda source: "Upstream instruction content")
    runtime = FakeRuntime(manifest)
    ordinary_run = runtime.run

    def run(records, candidate, trial):
        if records[0]["split"] == "test":
            assert (tmp_path / "run/vanilla/frozen-winner.json").exists()
        result = ordinary_run(records, candidate, trial)
        for output in result["outputs"]:
            output["reward"] = float("improved" in candidate["system_prompt"])
        return result

    runtime.run = run
    monkeypatch.setattr(main, "TauRuntime", lambda *args: runtime)
    parser = runner.build_parser("taubench", main.add_arguments)
    args = parser.parse_args(["--train-limit", "1", "--run-dir", str(tmp_path / "run")])
    models = runner.resolve_models(args)
    benchmark = main.build_benchmark(args, models)
    assert models.solver_model == QWEN3_8_27B_MODEL and models.proposer_model == DEEPSEEK_V4_1_FLASH_MODEL
    assert len(benchmark.trainset) == 57 and len(benchmark.valset) == 19 and len(benchmark.testset) == 21
    assert models.solver_kwargs["max_tokens"] == 65536
    assert "Upstream instruction content" in benchmark.seed_candidate["system_prompt"]
    assert benchmark.test_repetitions == 4
    improved = structured_prompt("Use improved instructions.", "alibaba")
    monkeypatch.setattr(runner, "LM", lambda *args, **kwargs: lambda messages: f"```\n{improved}\n```")
    argv = [
        "--run-dir",
        str(tmp_path / "run"),
        "--condition",
        "vanilla",
        "--max-metric-calls",
        "5",
        "--reflection-minibatch-size",
        "1",
        "--train-limit",
        "1",
        "--val-limit",
        "1",
        "--test-limit",
        "2",
    ]
    assert main.main(argv) == 0
    summary = json.loads((tmp_path / "run/vanilla/summary.json").read_text())
    assert summary["winner"]["validation_score"] == 1
    assert summary["test"]["metrics"]["pass_hat_k"]["4"] == 1
    assert summary["baseline"]["metrics"]["pass_hat_k"]["4"] == 0
    assert summary["test"]["timing"]["attempt_count"] == 8
    count = len(runtime.calls)
    assert main.main(argv) == 0
    assert len(runtime.calls) == count
    (tmp_path / "run/vanilla/heldout/repetition-002.json").unlink()
    assert main.main(argv) == 0
    assert len(runtime.calls) == count + 1 and runtime.calls[-1][2] == 2
