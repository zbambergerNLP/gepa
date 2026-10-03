"""Exercise persistent paired policy decisions and keep transfer evidence out of search."""

import json
import random
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.hotpotqa import diversity_quality_pilot as pilot
from examples.hotpotqa import main as production
from examples.hotpotqa import utils


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    train = [{"id": str(i), "question": f"Question {i}", "answer": "answer"} for i in range(150)]
    monkeypatch.setattr(pilot, "load_hotpotqa_dataset", lambda seed: (train, [], []))
    for name in ("_validate_scientific_data_identity", "_verify_scientific_retriever_integrity"):
        monkeypatch.setattr(pilot, name, lambda *args: None)
    monkeypatch.setattr(pilot, "benchmark_data_identity", lambda **kw: {})
    monkeypatch.setattr(pilot, "source_identity", lambda path: {"commit": "test-source"})
    monkeypatch.setattr(
        pilot, "build_run_contract", lambda *args: {"program": {"reported_supplemental_metric": "token_f1"}}
    )
    monkeypatch.setattr(pilot, "seed_candidate", lambda *args: dict.fromkeys(pilot.COMPONENTS, "original"))
    monkeypatch.setattr(pilot, "Wiki17BM25Retriever", lambda path: SimpleNamespace(provenance=lambda: {}))
    monkeypatch.setattr(pilot, "observed_kwargs", lambda *args: {})
    events = []
    controls = {"interrupt": None, "changed_only": False}

    def evaluate(candidate, example):
        index = int(example["id"])
        changed = [value for value in candidate.values() if value != "original"]
        events.append(("evaluate", index, bool(changed)))
        if index == controls["interrupt"] and (changed or not controls["changed_only"]):
            controls["interrupt"] = None
            raise RuntimeError("Interrupted evaluation")
        score = int(index < 3 or index % 3 == 0)
        if changed:
            opportunity = int(changed[0].split("|")[1])
            score = 1 if opportunity % 3 == 1 else 0 if opportunity % 3 == 2 else score
        feedback = {component + "_specific_info": {"id": example["id"]} for component in pilot.COMPONENTS}
        if index == 10 and not changed:
            feedback = {"evaluation_error": {"type": "task_output_parse_error"}}
        return score, feedback

    monkeypatch.setattr(pilot, "make_evaluator", lambda *args, **kw: evaluate)

    class Planner:
        def __init__(self, owner):
            self.owner = owner
            self.accepted = []
            self.observed = []

        def bind_run_dir(self, directory):
            self.directory = directory

        def observe_evaluation(self, proposal, parent):
            assert proposal.parent_program_ids == [0]
            assert (
                len(proposal.subsample_indices)
                == len(proposal.subsample_scores_before)
                == len(proposal.subsample_scores_after)
                == 3
            )
            self.observed.append(deepcopy(proposal.subsample_indices))
            proposal.metadata["training_outcome"] = {"observed": True}

        def validate_children(self, proposals, state):
            assert len(proposals) == 1
            assert all(value == "original" for value in state.program_candidates[0].values())
            assert not hasattr(state, "per_program_tracked_scores")
            events.append(("validate", self.owner.arm))

        def accept_child(self, proposal, child, state):
            assert state.program_candidates[child] == proposal.candidate
            self.accepted.append(child)
            events.append(("accept", self.owner.arm, child))

    class Strategy:
        def __init__(self, arm):
            self.arm = arm
            self.rng = random.Random(0)
            self.cursor = 0
            self.sibling_planner = Planner(self)

        def run_contract(self, parent):
            return {"policy": pilot.ARMS[self.arm], "parent": parent, "controller": "verbalized"}

        def get_batch_retry_state(self):
            return {
                "rng_state": self.rng.getstate(),
                "cursor": self.cursor,
                "accepted": deepcopy(self.sibling_planner.accepted),
                "observed": deepcopy(self.sibling_planner.observed),
            }

        def set_batch_retry_state(self, state):
            self.rng.setstate(state["rng_state"])
            self.cursor = state["cursor"]
            self.sibling_planner.accepted = deepcopy(state["accepted"])
            self.sibling_planner.observed = deepcopy(state["observed"])

        def reflect(self, parent, records, components, *, metadata):
            assert metadata["candidate_idx"] == 0
            assert list(records) == components
            assert all(value == "original" for value in parent.values())
            opportunity = metadata["optimizer_iteration"] - 1
            self.cursor += 1
            nonce = self.rng.random()
            events.append(("reflect", self.arm, opportunity, deepcopy(records), nonce, self.cursor))
            exhausted = self.arm == "sibling_diverse" and opportunity == 2
            text = {} if exhausted else {components[0]: f"{self.arm}|{opportunity}|{nonce}"}
            return SimpleNamespace(
                new_texts=text,
                metadata={"generation_outcome": "generation_exhausted" if exhausted else "changed_candidate"},
                prompts={"controller": "actual fake prompt"},
                raw_lm_outputs={"editor": "actual fake output"},
            ), self

    monkeypatch.setattr(pilot, "build_strategy", lambda settings, directory, arm: Strategy(arm))
    args = Namespace(
        model="hosted_vllm/Qwen/Qwen3.8-27B",
        reflection_model="hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash",
        api_base="http://solver/v1",
        reflection_api_base="http://teacher/v1",
        wiki17_dir=tmp_path,
        output_dir=tmp_path / "run",
        workers=1,
    )
    return args, events, controls


def test_persistent_real_training_admission_precedes_every_transfer_evaluation(harness):
    args, events, _ = harness
    summary = pilot.run(args)
    reflections = [event for event in events if event[0] == "reflect"]
    assert len(reflections) == 33
    assert len(summary["comparisons"]) == 36
    assert sum(row.get("perfect_batch_skip", False) for row in summary["comparisons"]) == 3
    assert summary["logical_evaluations"] == 528
    assert len([event for event in events if event[0] == "evaluate"]) == 528
    first_transfer = next(i for i, event in enumerate(events) if event[0] == "evaluate" and event[1] >= 36)
    assert all(event[0] not in {"reflect", "accept", "validate"} for event in events[first_transfer:])
    assert {event[1] for event in events if event[0] == "evaluate"} == set(range(48))
    for arm in pilot.ARMS:
        arm_calls = [event for event in reflections if event[1] == arm]
        assert [event[5] for event in arm_calls] == list(range(1, 12))
        assert len(summary["populations"][arm]) == 5
        assert all(node["parent"] == 0 for node in summary["populations"][arm][1:])
        for row in summary["comparisons"]:
            if row["arm"] == arm and row.get("changed"):
                assert row["accepted"] == (row["training"]["delta"] > 0)
                assert row["proposal_metadata"]["training_outcome"] == {"observed": True}
    assert summary["opportunities_with_missing_component_trace"] == [3]
    for event in reflections:
        if event[2] == 3:
            fallback = next(iter(event[3].values()))[1]
            assert fallback == {
                "Inputs": {"question": "Question 10"},
                "Feedback": {"evaluation_error": {"type": "task_output_parse_error"}},
                "whole_program_score": 0,
                "component_trace_available": False,
            }
    contract = json.loads((args.output_dir / "pilot-contract.json").read_text())
    assert contract["runtime"]["program"]["reported_supplemental_metric"] is None
    assert not summary["heldout_complete"]
    assert all("f1" not in json.loads(path.read_text())["record"] for path in args.output_dir.glob("**/records/*.json"))


def test_completed_resume_reuses_proposals_scores_and_real_populations(harness):
    args, events, _ = harness
    first = pilot.run(args)
    events.clear()
    second = pilot.run(args)
    assert first == second
    assert events == []


def test_offline_stage_resume_preserves_population_and_skips_completed_work(harness, monkeypatch):
    args, events, _ = harness
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setenv("GEPA_JEV_HANDOFF_DIR", str(args.output_dir.parent / "handoff"))
    monkeypatch.setattr(
        pilot,
        "build_run_contract",
        lambda _, settings: {
            "models": {
                "solver_api_base": settings.solver_api_base,
                "reflection_api_base": settings.reflection_api_base,
            },
        },
    )
    args.api_base = "http://localhost:1234/v1"
    args.reflection_api_base = "http://localhost:1235/v1"
    original_builder = pilot.build_strategy
    interrupted = []

    def builder(settings, directory, arm):
        strategy = original_builder(settings, directory, arm)
        reflect = strategy.reflect

        def pause(*values, **kwargs):
            if arm == "diversity_quality_jev" and kwargs["metadata"]["optimizer_iteration"] == 5 and not interrupted:
                interrupted.append(True)
                raise SystemExit(75)
            return reflect(*values, **kwargs)

        strategy.reflect = pause
        return strategy

    monkeypatch.setattr(pilot, "build_strategy", builder)
    with pytest.raises(SystemExit) as exc:
        pilot.run(args)
    assert exc.value.code == 75
    completed = [event for event in events if event[0] == "reflect"]
    args.api_base = "http://localhost:2234/v1"
    args.reflection_api_base = "http://localhost:2235/v1"
    summary = pilot.run(args)
    reflections = [event for event in events if event[0] == "reflect"]
    assert reflections[: len(completed)] == completed
    assert len(reflections) == 33
    assert len([event for event in events if event[0] == "evaluate"]) == 528
    assert len(summary["populations"]["diversity_quality_jev"]) == 5
    assert summary["execution_completed"]


def test_interruption_after_generation_reuses_generated_candidate_and_rng(harness):
    args, events, controls = harness
    controls["interrupt"] = 36
    with pytest.raises(RuntimeError, match="Interrupted evaluation"):
        pilot.run(args)
    generated = [event for event in events if event[0] == "reflect"]
    assert len(generated) == 33
    events.clear()
    summary = pilot.run(args)
    assert not any(event[0] in {"reflect", "accept"} for event in events)
    assert summary["execution_completed"]
    assert len(summary["populations"]["diversity_quality_jev"]) == 5


def test_interruption_between_generated_candidate_and_admission_restores_exact_state(harness):
    args, events, controls = harness
    controls.update(interrupt=3, changed_only=True)
    with pytest.raises(RuntimeError, match="Interrupted evaluation"):
        pilot.run(args)
    first_directory = args.output_dir / "diversity_quality_generative" / "opportunity-1"
    assert (first_directory / "generated.json").exists()
    assert not (first_directory / "decision.json").exists()
    assert len([event for event in events if event[0] == "reflect"]) == 1
    assert not any(event[0] == "accept" for event in events)
    resumed = pilot.run(args)
    assert len([event for event in events if event[0] == "reflect"]) == 33
    assert len([event for event in events if event[0] == "accept"]) == 12
    resumed_state = pilot._load(args.output_dir / "generation-complete.json")["final_states"]
    fresh_args = deepcopy(args)
    fresh_args.output_dir = args.output_dir.parent / "uninterrupted"
    uninterrupted = pilot.run(fresh_args)
    fresh_state = pilot._load(fresh_args.output_dir / "generation-complete.json")["final_states"]
    assert resumed_state == fresh_state
    assert resumed["populations"] == uninterrupted["populations"]


def test_changed_checkpoint_is_rejected_before_more_model_work(harness):
    args, events, _ = harness
    pilot.run(args)
    path = args.output_dir / "sibling_diverse" / "opportunity-1" / "decision.json"
    value = json.loads(path.read_text())
    value["record"]["comparison"]["accepted"] = False
    path.write_text(json.dumps(value))
    events.clear()
    with pytest.raises(ValueError, match="hash mismatch"):
        pilot.run(args)
    assert events == []


@pytest.mark.parametrize("prediction", ["The Eiffel Tower!", "An unrelated place"])
def test_em_only_evaluator_preserves_production_task_scores_and_traces_without_f1(monkeypatch, prediction):
    calls = []
    lm = object()
    retriever = object()
    kwargs = {"configured": True}

    def program(*args):
        calls.append(args)
        return "query", prediction, {"trace": "actual task trace"}

    def records(example, trace, score, *, include_diagnostics):
        return {"final_answer": {"trace": trace, "score": score, "diagnostics": include_diagnostics}}

    for module in (pilot, production):
        monkeypatch.setattr(module, "build_hotpotqa_task_lm", lambda *args: lm)
        monkeypatch.setattr(module, "run_program", program)
        monkeypatch.setattr(module, "artifact_component_records", records)
    arguments = ("solver", retriever, "http://solver/v1")
    options = {"solver_lm_kwargs": kwargs, "reflection_diagnostics": True}
    parent = {"final_answer": "prompt"}
    example = {"question": "Where?", "answer": "Eiffel Tower"}
    expected = production.make_evaluator(*arguments, **options)(parent, example)

    def forbidden_f1(*args):
        raise AssertionError("The EM-only pilot must not calculate F1")

    monkeypatch.setattr(utils, "f1_score", forbidden_f1, raising=False)
    actual = pilot.make_evaluator(*arguments, **options)(parent, example)
    assert actual == expected
    assert calls[0] == calls[1]
    assert calls[1][2] == "2stage" and calls[1][6] == 7


def test_em_only_evaluator_preserves_task_format_zero_and_propagates_systemic_errors(monkeypatch):
    monkeypatch.setattr(pilot, "build_hotpotqa_task_lm", lambda *args: object())
    error = ValueError("Failed to parse response as per signature: answer missing")

    def program(*args):
        raise error

    monkeypatch.setattr(pilot, "run_program", program)
    evaluate = pilot.make_evaluator(
        "solver", object(), "http://solver/v1", solver_lm_kwargs={}, reflection_diagnostics=True
    )
    score, feedback = evaluate({}, {"question": "Question", "answer": "Answer"})
    assert score == 0.0
    assert feedback["evaluation_error"]["type"] == "task_output_parse_error"
    error = ValueError("retrieval configuration invalid")
    with pytest.raises(ValueError, match="retrieval configuration invalid"):
        evaluate({}, {"question": "Question", "answer": "Answer"})
