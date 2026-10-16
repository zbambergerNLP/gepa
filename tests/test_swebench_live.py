"""Check real data boundaries, native rollout isolation and launch guards without model calls."""

import json
from types import SimpleNamespace

import pytest

from examples.common.artifacts import atomic_json, digest
from examples.common.benchmark_types import BenchmarkModels
from examples.swebench_live import adapter, images
from examples.swebench_live.data import partition


def row(name, **kwargs):
    return {"instance_id": name, "problem_statement": "Fix the public issue", **kwargs}


def test_partition_preserves_official_test_and_excludes_all_its_versions():
    full = [row("a"), row("b"), row("c"), row("d"), row("held", test_cmds="old"), row("held", test_cmds="new")]
    official = [row("test-only"), row("held", test_cmds="authoritative")]
    splits = partition(full, official)
    assert splits["test"] == official
    assert len(splits["train"]) == len(splits["val"]) == 2
    assert {r["instance_id"] for r in splits["train"] + splits["val"]} == {"a", "b", "c", "d"}
    assert partition(list(reversed(full)), official) == splits


def test_partition_refuses_ambiguous_training_or_test_tasks():
    with pytest.raises(ValueError, match="Ambiguous"):
        partition([row("a"), row("a"), row("b")], [row("held")])
    with pytest.raises(ValueError, match="Duplicate"):
        partition([row("a"), row("b")], [row("held"), row("held")])


@pytest.fixture
def native(tmp_path):
    raw = row("safe", patch="GOLD SOLUTION", test_patch="HIDDEN TEST", hints_text="SECRET HINT")
    (tmp_path / "data").mkdir()
    for name in ("train", "val", "test"):
        (tmp_path / "data" / f"{name}.jsonl").write_text(json.dumps(raw) + "\n" if name == "train" else "")
    record = {
        "id": "safe",
        "split": "train",
        "problem_statement": raw["problem_statement"],
        "evaluator_sha256": digest(raw),
    }
    model = BenchmarkModels("explicit/solver", "explicit/proposer", None, None, {}, {})
    runner = adapter.SWEBenchLiveAdapter(
        model,
        tmp_path,
        tmp_path / "run",
        [record],
        {"safe": {"image_id": "sha256:image", "immutable": "repo@sha256:pin"}},
    )
    events = []

    class Task:
        def __init__(self, image, memory):
            assert image == "sha256:image"

        def __enter__(self):
            events.append("start")
            return self

        def execute(self, command):
            assert not list(tmp_path.rglob("evaluator-record.json"))
            events.append(command)
            return {"exit_code": 0, "stdout": "PATCH" if command.startswith("git ") else "tool result", "stderr": ""}

        def __exit__(self, *_):
            events.append("stop")

    runner.task_factory = Task

    def grader(root, trial, record, image, memory, timeout):
        assert events[-1] == "stop"
        assert json.loads((trial / "prediction.json").read_text())["model_patch"] == "PATCH"
        assert json.loads((trial / "evaluator-record.json").read_text())["patch"] == "GOLD SOLUTION"
        report = {"safe": {"resolved": True}}
        atomic_json(trial / "official-result.json", report)
        return report

    runner.grader = grader
    return runner, record, events


def response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda **_: message))])


def test_native_rollout_keeps_gold_private_and_grades_only_after_patch_freeze(native):
    runner, record, events = native
    requests = []

    def completion(**kwargs):
        requests.append(kwargs)
        assert not any(secret in json.dumps(kwargs) for secret in ("GOLD SOLUTION", "HIDDEN TEST", "SECRET HINT"))
        if len(requests) == 1:
            return response(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                        }
                    ],
                }
            )
        return response({"role": "assistant", "content": "Done"})

    runner.completion = completion
    candidate = {"system_prompt": "Generic repair instructions"}
    result = runner.evaluate([record], candidate, True)
    assert result.scores == [1.0]
    assert events == ["start", "ls", "git --no-pager diff HEAD --text", "stop"]
    assert requests[0]["model"] == "explicit/solver"
    feedback = runner.make_reflective_dataset(candidate, result, ["system_prompt"])
    assert feedback["system_prompt"][0]["Feedback"]["resolved"] is True
    result.trajectories[0]["record"] = {**record, "split": "test"}
    with pytest.raises(ValueError, match="training"):
        runner.make_reflective_dataset(candidate, result, ["system_prompt"])


def test_provider_failure_stays_unresolved_and_is_not_retried_or_scored(native):
    runner, record, events = native
    calls = []

    def failure(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("provider outcome unknown")

    runner.completion = failure
    with pytest.raises(RuntimeError, match="outcome unknown"):
        runner.evaluate([record], {"system_prompt": "Generic instructions"})
    assert len(calls) == 1 and events == ["start", "stop"]
    assert not list(runner.run_dir.rglob("result.json"))
    assert len(list(runner.run_dir.rglob("intent.json"))) == 1
    assert len(list(runner.run_dir.rglob("trajectory.json"))) == 1


def test_task_record_drift_is_rejected_before_a_container_or_model(native):
    runner, record, events = native
    with pytest.raises(ValueError, match="identity drift"):
        runner.evaluate([{**record, "problem_statement": "changed"}], {"system_prompt": "Generic instructions"})
    assert events == []


def test_insufficient_docker_memory_blocks_readiness_before_image_or_model_work(monkeypatch, tmp_path):
    monkeypatch.setattr(images, "docker_json", lambda *args: {"MemTotal": 4 * 2**30, "NCPU": 11})
    with pytest.raises(ValueError, match="16 GiB"):
        images.verify_images(tmp_path, [], 16)
