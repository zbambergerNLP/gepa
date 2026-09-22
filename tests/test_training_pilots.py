"""Verify training-only pilot completion independently of score improvements."""

import json

import pytest

from examples.common.pilot_checks import METHODS, CycleEvidence, atomic_json, load_cycle, require_contract
from examples.common.recovery import RecoveryCallback, snapshot
from examples.hotpotqa import main as hotpotqa_main
from examples.hotpotqa import pilot
from examples.hotpotqa.pilot import run_calibration, strict_evaluator, validate_calibration
from examples.hotpotqa.pilot_report import overlap_seconds, report, request_intervals
from gepa import optimize
from gepa.core.adapter import EvaluationBatch


def test_calibration_accepts_wrong_answers_and_resumes_completed_work(tmp_path):
    """A complete all-zero pilot passes and never reruns saved questions."""
    examples = [{"id": str(i), "question": "question", "answer": "correct"} for i in range(3)]
    calls = []

    def execute(candidate, example):
        calls.append(example["id"])
        return "wrong", {"answer": "wrong"}

    result = run_calibration(
        tmp_path / "pilot", examples, {"prompt": "initial"}, execute, contract={"stage": "smoke"}, workers=2
    )
    assert result["qualified"] and result["exact_match"] == 0
    assert len(calls) == 3
    run_calibration(
        tmp_path / "pilot", examples, {"prompt": "initial"}, execute, contract={"stage": "smoke"}, workers=2
    )
    assert len(calls) == 3
    with pytest.raises(ValueError, match="changed"):
        run_calibration(
            tmp_path / "pilot", examples, {"prompt": "different"}, execute, contract={"stage": "smoke"}, workers=2
        )


def test_missing_prediction_never_qualifies(tmp_path):
    """Execution failures remain distinct from ordinary incorrect answers."""
    with pytest.raises(RuntimeError, match="usable prediction"):
        run_calibration(
            tmp_path, [{"id": "q", "answer": "gold"}], {}, lambda *_: ("", {}), contract={"stage": "smoke"}, workers=1
        )
    assert not (tmp_path / "pilot-complete.json").exists()


def test_partial_calibration_resumes_only_unfinished_questions(tmp_path):
    """Keep saved predictions after an interrupted request without counting it twice."""
    examples = [{"id": str(i), "answer": "gold"} for i in range(3)]
    calls = []

    def interrupted(candidate, example):
        calls.append(example["id"])
        if example["id"] == "1":
            raise ConnectionError("simulated allocation interruption")
        return "wrong", {}

    with pytest.raises(ConnectionError):
        run_calibration(tmp_path, examples, {}, interrupted, contract={"stage": "smoke"}, workers=1)
    assert not (tmp_path / "pilot-complete.json").exists()
    saved = {path.name: path.read_bytes() for path in (tmp_path / "records").glob("*.json")}
    resumed = []

    def complete(candidate, example):
        resumed.append(example["id"])
        return "wrong", {}

    result = run_calibration(tmp_path, examples, {}, complete, contract={"stage": "smoke"}, workers=1)
    assert result["completed_questions"] == 3
    assert "0" not in resumed
    assert "1" in resumed
    for name, data in saved.items():
        assert (tmp_path / "records" / name).read_bytes() == data
    assert validate_calibration(tmp_path, 3)["qualified"]


@pytest.mark.parametrize("new_score,accepted", [(0.0, False), (0.5, False), (1.0, True)])
@pytest.mark.parametrize("drop_first", [False, True])
def test_real_engine_cycle_accepts_any_metric_direction(tmp_path, new_score, accepted, drop_first, monkeypatch):
    """Run the actual mutation, evaluation, and strict acceptance path."""
    registry = tmp_path / "registry.json"
    monkeypatch.setenv("GEPA_RECOVERY_REGISTRY", str(registry))
    monkeypatch.setenv("HOTPOTQA_SOURCE_COMMIT", "a" * 40)

    class Adapter:
        proposals = 0

        def evaluate(self, batch, candidate, capture_traces=False):
            score = new_score if candidate["prompt"] == "revised" else 0.5
            return EvaluationBatch(
                outputs=["answer"] * len(batch),
                scores=[score] * len(batch),
                trajectories=[{"question": q, "feedback": "real evaluation"} for q in batch],
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {"prompt": eval_batch.trajectories}

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            self.proposals += 1
            if drop_first and self.proposals == 1:
                return {}
            return {"prompt": "revised"}

    cycle = CycleEvidence(tmp_path)
    optimize(
        seed_candidate={"prompt": "initial"},
        trainset=[1, 2, 3],
        valset=[1, 2, 3],
        adapter=Adapter(),
        reflection_lm=lambda _: "unused",
        reflection_minibatch_size=3,
        run_dir=str(tmp_path),
        stop_callbacks=cycle.completed_cycle,
        callbacks=[RecoveryCallback(tmp_path), cycle],
        acceptance_criterion="strict_improvement",
        cache_evaluation=False,
        use_merge=False,
    )
    summary = cycle.verify()
    assert summary["completed_cycles"] == 1
    assert summary["decision"]["accepted"] is accepted
    assert len(json.loads((tmp_path / "optimizer-cycle.json").read_text())["reevaluation"]["scores"]) == 3
    assert next(iter(snapshot(registry, "a" * 40).values())) > 0
    assert (tmp_path / "incomplete-iterations" / "1.json").exists() is drop_first


def test_skipped_cycle_is_not_misrepresented_as_covered(tmp_path):
    """Skipping all work cannot stand in for exercising the pipeline."""
    cycle = CycleEvidence(tmp_path)
    cycle.events = {"finished": True}
    with pytest.raises(RuntimeError, match="complete cycle"):
        cycle.verify()


@pytest.mark.parametrize(
    "name", ["records/0000.json", "pilot-summary.json", "token-usage-summary.json", "pilot-contract.json"]
)
def test_calibration_rejects_modified_completed_evidence(tmp_path, name):
    """Neither stage advancement nor resume can certify changed evidence."""
    examples = [{"id": str(i), "answer": "gold"} for i in range(3)]
    execute = lambda *_: ("wrong", {})
    run_calibration(tmp_path, examples, {}, execute, contract={"stage": "smoke"}, workers=1)
    path = tmp_path / name
    value = json.loads(path.read_text())
    value["changed"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed|Incomplete"):
        validate_calibration(tmp_path, 3)
    with pytest.raises(ValueError, match="changed|Incomplete"):
        run_calibration(tmp_path, examples, {}, execute, contract={"stage": "smoke"}, workers=1)


def test_optimizer_parse_error_is_not_a_valid_zero_score():
    """The production parse-error fallback must fail the process pilot."""
    assert strict_evaluator(lambda *_: (0.0, {}))({}, {}) == (0.0, {})
    with pytest.raises(RuntimeError, match="malformed"):
        strict_evaluator(lambda *_: (0.0, {"evaluation_error": "missing output"}))({}, {})


def test_usage_overlap_excludes_idle_gaps_and_does_not_double_count(tmp_path):
    """Queued or intermittently active jobs alone cannot prove simultaneous calls."""
    log = tmp_path / "provider-attempts.jsonl"
    rows = [
        {"timestamp": "2026-09-12T00:00:10+00:00", "elapsed_seconds": 10},
        {"timestamp": "2026-09-12T00:00:12+00:00", "elapsed_seconds": 7},
        {"timestamp": "2026-09-12T00:00:30+00:00", "elapsed_seconds": 10},
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows))
    intervals = request_intervals(log)
    start = intervals[0][0]
    assert intervals == [(start, start + 12), (start + 20, start + 30)]
    assert overlap_seconds(intervals, [(start + 13, start + 19)]) == 0
    assert overlap_seconds(intervals, [(start + 5, start + 25)]) == 12
    assert report(tmp_path)["complete"] is False


def completed_cycle():
    """Provide callback-shaped observations for orchestration-only tests."""
    return {
        "reflection": {"feedback": "wrong answer"},
        "proposal": {"new_instructions": {"prompt": "revised"}},
        "reevaluation": {"scores": [0.0, 0.0, 0.0]},
        "decision": {"accepted": False},
        "finished": True,
    }


@pytest.mark.parametrize(
    "signal", [{"length_finish": True}, {"output_cap_reached": True}, {"finish_reasons": ["length"]}]
)
def test_optimizer_cutoff_cannot_be_sealed_or_reused(tmp_path, signal):
    """An evaluated candidate can still contain a truncated proposer response."""
    require_contract(tmp_path, {"method": "vanilla"})
    cycle = CycleEvidence(tmp_path)
    cycle.events = completed_cycle()
    cycle._save()
    attempts = tmp_path / "provider-attempts.jsonl"
    attempts.write_text(json.dumps({"role": "optimizer", "outcome": "success", **signal}) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        cycle.verify()
    assert not (tmp_path / "optimizer-pilot-complete.json").exists()

    attempts.write_text(json.dumps({"role": "optimizer", "outcome": "success", "finish_reasons": ["stop"]}) + "\n")
    cycle.verify()
    marker = tmp_path / "optimizer-pilot-complete.json"
    legacy = json.loads(marker.read_text())
    del legacy["provider_attempts_sha256"]
    atomic_json(marker, legacy)
    attempts.write_text(json.dumps({"role": "optimizer", "outcome": "success", **signal}) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        load_cycle(tmp_path)


@pytest.mark.parametrize("remove", [False, True])
def test_completed_optimizer_binds_provider_evidence(tmp_path, remove):
    """Removing or changing request evidence invalidates a completed pilot."""
    require_contract(tmp_path, {"method": "vanilla"})
    cycle = CycleEvidence(tmp_path)
    cycle.events = completed_cycle()
    cycle._save()
    attempts = tmp_path / "provider-attempts.jsonl"
    row = {"role": "optimizer", "outcome": "success", "finish_reasons": ["stop"], "completion_tokens": 10}
    attempts.write_text(json.dumps(row) + "\n")
    cycle.verify()
    assert load_cycle(tmp_path)["completed_cycles"] == 1
    if remove:
        attempts.unlink()
    else:
        attempts.write_text(json.dumps({**row, "completion_tokens": 11}) + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_cycle(tmp_path)


def test_completed_cycle_cannot_be_reused_after_contract_change(tmp_path):
    """Bind coverage evidence to the actual data and optimizer configuration."""
    require_contract(tmp_path, {"method": "vanilla"})
    cycle = CycleEvidence(tmp_path)
    cycle.events = completed_cycle()
    cycle._save()
    cycle.verify()
    assert load_cycle(tmp_path)["completed_cycles"] == 1
    atomic_json(tmp_path / "pilot-contract.json", {"method": "react_v2"})
    with pytest.raises(ValueError, match="changed"):
        load_cycle(tmp_path)


@pytest.mark.parametrize("stage", ["all", "preliminary", "throughput"])
def test_hotpotqa_pilot_stage_order_and_calibration_recovery(tmp_path, monkeypatch, stage):
    """Use real config builders and isolate only dataset, model transport, and GPU validation."""
    training = [{"id": f"train-{i}", "question": f"question-{i}", "answer": "gold"} for i in range(150)]
    heldout = [{"id": "never-execute", "question": "heldout", "answer": "gold"}]
    monkeypatch.setattr(pilot, "load_hotpotqa_dataset", lambda **_: (training, heldout, heldout))
    monkeypatch.setattr(pilot, "_validate_scientific_data_identity", lambda _: None)

    def validate(args):
        assert args.enforce_scientific_contract

    monkeypatch.setattr(hotpotqa_main, "_validate_scientific_contract", validate)
    monkeypatch.setattr(pilot, "_verify_scientific_retriever_integrity", lambda _: None)

    class Retriever:
        def __init__(self, path):
            pass

        def provenance(self):
            return {"fixture": True}

    monkeypatch.setattr(pilot, "Wiki17BM25Retriever", Retriever)
    monkeypatch.setattr(pilot, "build_hotpotqa_task_lm", lambda *_: object())
    executed = []

    def program(seed, question, program, model, api_base, retriever, k, lm, kwargs):
        assert program == "2stage" and k == 7 and question != "heldout"
        executed.append(question)
        return "query", "wrong", {}

    monkeypatch.setattr(pilot, "run_program", program)
    monkeypatch.setattr(pilot, "make_evaluator", lambda *a, **kw: lambda *_: (0, {}))
    methods = []

    def optimize(method, candidate, train, val, config, evaluator, callbacks):
        assert len(executed) == 3
        assert train == val == training[:3]
        assert config.engine.max_candidate_proposals is None and config.engine.max_metric_calls is None
        assert config.stop_callbacks == callbacks[-1].completed_cycle
        assert config.reflection.reflection_lm_kwargs["_gepa_provider_retry"]["token_limits"] == {
            **pilot.LIMITS,
            "max_output_tokens": 32_768,
        }
        assert config.reflection.reflection_strategy is not None if method.startswith("react_v2") else True
        methods.append(method)
        callbacks[-1].events = completed_cycle()
        callbacks[-1]._save()

    monkeypatch.setattr(pilot, "run_condition", optimize)
    args = [
        "--model",
        "hosted_vllm/Qwen/Qwen3.8-27B",
        "--api-base",
        "http://127.0.0.1:8000/v1",
        "--wiki17-dir",
        str(tmp_path / "wiki"),
        "--workers",
        "12",
        "--output-dir",
        str(tmp_path / "pilot"),
        "--stage",
        stage,
    ]
    pilot.main(args)
    assert methods == ([] if stage == "throughput" else list(METHODS))
    assert validate_calibration(tmp_path / "pilot" / "smoke", 3)["qualified"]
    assert validate_calibration(tmp_path / "pilot" / "throughput", 12)["qualified"]
    if stage == "all":
        assert len(executed) == 165
        assert validate_calibration(tmp_path / "pilot" / "full", 150)["qualified"]
    else:
        assert len(executed) == 15
        assert not (tmp_path / "pilot" / "full").exists()
    if stage == "throughput":
        assert not (tmp_path / "pilot" / "optimizer").exists()
        saved = {path: path.read_bytes() for path in (tmp_path / "pilot").rglob("*.json")}
        pilot.main(args)
        assert len(executed) == 15
        assert all(path.read_bytes() == contents for path, contents in saved.items())
