"""Test the paired qualification protocol without model requests."""

import json
from argparse import Namespace
from pathlib import Path
from threading import get_ident

import pytest

from examples.hotpotqa import generalization_pilot as pilot
from examples.hotpotqa import proposal_runtime
from examples.hotpotqa.generalization_pilot import (
    COMPONENTS,
    DiagnosticRetriever,
    evaluate_records,
    paired_outcomes,
    partition_training,
)
from gepa.strategies.text_limits import TextLimits


@pytest.mark.parametrize("module", list(proposal_runtime.SHARED_MODULES))
def test_request_runtime_cannot_override_strategy_or_load_changed_files(tmp_path, monkeypatch, module):
    """Keep strategy source pinned and fail before inference on runtime identity drift."""
    root = Path(proposal_runtime.__file__).resolve().parents[2]
    finder = proposal_runtime.SharedRequestRuntime(root)
    assert finder.find_spec("gepa.strategies.action_space") is None
    assert finder.find_spec("gepa.lm").origin == str(root / "src/gepa/lm.py")
    assert finder.find_spec("gepa.lm_constants").origin == str(
        root / "src/gepa/lm_constants.py"
    )
    request = tmp_path / "request.json"
    runtime = proposal_runtime.runtime_identity(root)
    runtime["files"][module] = "changed"
    request.write_text(json.dumps({"shared_request_runtime": runtime}))
    monkeypatch.setattr(proposal_runtime.sys, "argv", ["worker", "--proposal-request", str(request)])
    with pytest.raises(ValueError, match="reviewed shared files"):
        proposal_runtime.main()


def test_transfer_membership_is_fixed_and_never_enters_proposal_batches() -> None:
    """Reserve ordered training examples outside every three-example proposal batch."""
    train = [{"id": str(i)} for i in range(150)]
    batches, transfer = partition_training(train)
    assert list(batches) == list(COMPONENTS)
    proposal_ids = [row["id"] for batch in batches.values() for row in batch]
    assert proposal_ids == [str(i) for i in range(12)]
    assert [row["id"] for row in transfer] == [str(i) for i in range(12, 36)]
    assert not set(proposal_ids) & {row["id"] for row in transfer}
    with pytest.raises(ValueError, match="distinct"):
        partition_training(train[:35] + [train[0]])


def test_paired_summary_retains_regressions_and_refuses_misaligned_cases() -> None:
    """Count a gain, a regression, and an unchanged result without filtering."""
    before = [{"id": str(i), "score": score} for i, score in enumerate([0, 1, 1])]
    after = [{"id": str(i), "score": score} for i, score in enumerate([1, 0, 1])]
    report = paired_outcomes(before, after)
    assert report["wins"] == report["losses"] == report["unchanged"] == 1
    assert report["delta"] == 0 and report["already_correct_regressions"] == ["1"]
    with pytest.raises(ValueError, match="ordered"):
        paired_outcomes(before, list(reversed(after)))


def test_evaluation_reuses_exact_records_and_detects_drift(tmp_path: Path) -> None:
    """Retain task-format zeros and reject changed predictions or case identity."""
    calls = []

    def evaluate(candidate, example):
        calls.append(example["id"])
        return 0, {"evaluation_error": {"type": "task_output_parse_error"}}

    examples = [{"id": "a", "answer": "x"}]
    candidate = {"sys": "original"}
    records = evaluate_records(tmp_path, candidate, examples, evaluate, 1)
    assert records[0]["score"] == 0 and records[0]["feedback"]["evaluation_error"]
    assert evaluate_records(tmp_path, candidate, examples, evaluate, 1) == records
    assert calls == ["a"]
    with pytest.raises(ValueError, match="configuration changed"):
        evaluate_records(tmp_path, {"sys": "changed"}, examples, evaluate, 1)
    path = tmp_path / "records/0000.json"
    saved = json.loads(path.read_text())
    saved["record"]["score"] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="record changed"):
        evaluate_records(tmp_path, candidate, examples, evaluate, 1)


def test_synthetic_diagnostics_are_distinct_from_natural_transfer() -> None:
    """Freeze eight controlled cases covering transfer and overrestriction risks."""
    cases = json.loads((Path(__file__).parents[1] / "examples/hotpotqa/generalization_cases.json").read_text())
    assert len({case["id"] for case in cases}) == 8
    assert {case["category"] for case in cases} == {
        "description_without_identity",
        "explicit_identity",
        "cross_passage_relation",
        "direct_answer_control",
    }
    retriever = DiagnosticRetriever(cases[0]["passages"])
    assert retriever.search("question") == retriever.search("a generated query")


@pytest.mark.parametrize("changed", [True, False])
def test_complete_paired_protocol_scores_every_proposal_without_transfer_leakage(tmp_path, monkeypatch, changed):
    """Exercise both revisions and no-op accounting through the complete driver without inference."""
    root = Path(pilot.__file__).resolve().parents[2]
    control = tmp_path / "control-source"
    output = tmp_path / "comparison"
    args = Namespace(
        control_source=control,
        control_commit="old",
        model="hosted_vllm/Qwen/Qwen3.8-27B",
        api_base="solver-api",
        reflection_model="hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash",
        reflection_api_base="teacher-api",
        wiki17_dir=tmp_path,
        workers=4,
        output_dir=output,
        wandb_project=None,
        wandb_entity=None,
    )
    monkeypatch.setattr(
        pilot, "source_identity", lambda p: {"commit": "old" if p == control else "new", "directory": str(p)}
    )
    train = [{"id": str(i), "question": f"Question {i}", "answer": "yes"} for i in range(150)]
    monkeypatch.setattr(pilot, "load_hotpotqa_dataset", lambda seed: (train, [], []))
    monkeypatch.setattr(pilot, "benchmark_data_identity", lambda **kw: {})
    monkeypatch.setattr(pilot, "_validate_scientific_data_identity", lambda _: None)
    monkeypatch.setattr(pilot, "_verify_scientific_retriever_integrity", lambda _: None)
    monkeypatch.setattr(pilot, "Wiki17BM25Retriever", lambda _: Namespace(provenance=lambda: {}))
    monkeypatch.setattr(pilot, "build_run_contract", lambda *a: {})
    monkeypatch.setattr(pilot, "resolve_template_family", lambda *a: "alibaba")
    seed = dict.fromkeys(COMPONENTS, "original")
    monkeypatch.setattr(pilot, "seed_candidate", lambda *a: dict(seed))
    monkeypatch.setattr(pilot, "observed_kwargs", lambda *a: {})
    evaluated = []

    def evaluator(candidate, example):
        evaluated.append((dict(candidate), example["id"]))
        score = int("revised" in candidate.values() or not example["id"].isdigit() or int(example["id"]) % 2 == 0)
        feedback = {
            f"{name}_specific_info": {
                "Inputs": {"id": example["id"]},
                "Generated Outputs": {"answer": "yes" if score else "no"},
                "Feedback": "Reference-only guidance",
                "End-to-end Outcome": {"score": score},
            }
            for name in COMPONENTS
        }
        return score, feedback

    owner_thread = get_ident()
    constructed = []

    def make_evaluator(model, retriever, api_base, **kwargs):
        assert get_ident() == owner_thread, "DSPy evaluators must be configured on the owning thread"
        constructed.append(retriever)

        def evaluate(candidate, example):
            assert get_ident() != owner_thread
            if "passages" in example:
                assert isinstance(retriever, DiagnosticRetriever)
                assert [(p.title, p.text) for p in retriever.passages] == [tuple(p) for p in example["passages"]]
            return evaluator(candidate, example)

        return evaluate

    monkeypatch.setattr(pilot, "make_evaluator", make_evaluator)
    requests = []

    def worker(command, **kwargs):
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text())
        settings = Namespace(**request["settings"])
        settings.enforce_scientific_contract = False
        config, _ = pilot.build_config("react_v2", settings, {}, str(request_path.parent))
        assert config.reflection.text_limits == TextLimits()
        requests.append(request)
        variant = request_path.parent.parent.name
        candidate = {**request["candidate"], request["component"]: variant} if changed else request["candidate"]
        pilot.atomic_json(
            request_path.parent / "proposal.json",
            {
                "request_sha256": pilot.digest(request),
                "candidate": candidate,
                "changed": changed,
                "imported_source": str((control if variant == "control" else root) / "src/gepa/__init__.py"),
                "shared_request_runtime": request["shared_request_runtime"],
            },
        )

    monkeypatch.setattr(pilot.subprocess, "run", worker)
    summary = pilot.run_comparison(args)
    assert len(constructed) == 9
    assert len(requests) == len(summary["comparisons"]) == 8
    assert all(request["shared_request_runtime"] == requests[0]["shared_request_runtime"] for request in requests)
    assert all(request["candidate"] == seed for request in requests)
    for request in requests:
        records = request["reflection_records"][request["component"]]
        assert len(records) == 3
        assert all(int(row["Inputs"]["id"]) < 12 for row in records)
        assert all(("End-to-end Outcome" in row) == (request["source"]["commit"] == "new") for row in records)
    assert summary["actual_metric_evaluations"] == len(evaluated) == (324 if changed else 44)
    assert summary["usefulness_signal"] == changed
    assert summary["execution_completed"] and not summary["heldout_complete"] and not summary["completed_ablation"]
