"""Exercise staged pilot coverage and campaign admission without model calls."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from terminalbench_pilot_helpers import runtime_fixture, write_pilot_fixture

from examples.common.experiment_models import EXPERIMENT_MODELS
from examples.terminalbench import canary
from examples.terminalbench import main as campaign
from examples.terminalbench.pilot import load_completed_pilot, review_pilot, validate_review
from examples.terminalbench.token_usage import record_usage
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.core.adapter import EvaluationBatch


def run_contract(root: Path, model: str = EXPERIMENT_MODELS[0], scope: str = "system_prompt", concurrency: int = 1):
    """Build the actual campaign contract used to compare runtime settings."""
    args = campaign.build_parser().parse_args(
        [
            "--condition",
            "vanilla",
            "--student-model",
            model,
            "--proposer-model",
            model,
            "--student-api-base",
            "http://localhost:8000/v1",
            "--optimization-scope",
            scope,
            "--n-concurrent",
            str(concurrency),
            "--run-dir",
            str(root),
            "--harbor-work-dir",
            str(root / "harbor"),
        ]
    )
    manifest = load_terminalbench_manifest(campaign.EXPERIMENT_MANIFESTS["tb2.1"])
    _, family = campaign.seed_candidate(model, "auto", "tb2.1", scope)
    args.execution_runtime = {role: runtime_fixture(model) for role in ("student", "proposer")}
    contract = campaign.build_run_contract(
        args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", family
    )
    return manifest, contract


def offline_adapter(adapter, tasks, candidate):
    """Save realistic usage and a verified task timeout without invoking Harbor."""
    model = adapter.harbor.student_model
    limits = adapter.harbor.student_agent_kwargs["token_limits"]
    for _ in tasks:
        record_usage(
            adapter.harbor.work_dir / "token-usage.jsonl",
            "task_agent",
            model,
            limits,
            {
                "model": model,
                "choices": [{"finish_reason": "length"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 32768},
            },
        )
    return EvaluationBatch(
        outputs=[
            {"task_id": task.task_id, "reward": 0.0, "errors": ["AgentTimeoutError: verified"] if index == 0 else []}
            for index, task in enumerate(tasks)
        ],
        scores=[0.0] * len(tasks),
    )


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
@pytest.mark.parametrize("full_scope", ["system_prompt", "all_text"])
def test_two_stage_pilot_covers_training_and_qualifies_both_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, full_scope: str
) -> None:
    """Cover 3 then 30 tasks and reuse one reviewed full pilot across text scopes."""
    monkeypatch.setattr(canary.HarborCLI, "check_requirements", Mock())
    evaluated = []

    def evaluate(adapter, tasks, candidate):
        evaluated.append([task.task_id for task in tasks])
        return offline_adapter(adapter, tasks, candidate)

    monkeypatch.setattr(canary.TerminusAdapter, "evaluate", evaluate)
    common = ["--model", model, "--api-base", "http://localhost:8000/v1"]
    smoke_dir, full_dir = tmp_path / "smoke", tmp_path / "full"
    canary.main([*common, "--output-dir", str(smoke_dir)])
    canary.main(
        [
            *common,
            "--stage",
            "full",
            "--smoke-dir",
            str(smoke_dir),
            "--output-dir",
            str(full_dir),
            "--optimization-scope",
            full_scope,
            "--n-concurrent",
            "4",
        ]
    )
    manifest, _ = run_contract(tmp_path / "campaign", model)
    assert evaluated == [manifest.splits["train"][:3], manifest.splits["train"]]
    assert not set(manifest.splits["val"] + manifest.splits["test"]).intersection(
        task_id for batch in evaluated for task_id in batch
    )
    summary = json.loads((full_dir / "pilot-summary.json").read_text())
    assert summary["task_count"] == 30
    assert summary["timed_out_task_ids"] == manifest.splits["train"][:1]
    assert summary["tasks_per_hour"] > 0 and summary["elapsed_seconds"] > 0
    usage = json.loads((full_dir / "token-usage-summary.json").read_text())
    assert usage["models"][model]["task_agent"]["length_finish"] == 30
    reviews = []
    for scope in ("system_prompt", "all_text"):
        _, contract = run_contract(tmp_path / "campaign", model, scope, 4)
        review = review_pilot(full_dir, contract, manifest)
        validate_review(review, contract, manifest)
        reviews.append(review)
    assert reviews[0] == reviews[1]


@pytest.mark.parametrize(
    "options",
    [
        ["--stage", "full"],
        ["--stage", "full", "--train-limit", "3"],
        ["--stage", "smoke", "--train-limit", "30"],
        ["--stage", "smoke", "--n-concurrent", "4"],
        ["--stage", "calibration", "--train-limit", "31"],
    ],
)
def test_invalid_stage_coverage_stops_before_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    """Reject missing prerequisites and undersized or non-training coverage early."""
    harbor = Mock()
    monkeypatch.setattr(canary, "HarborCLI", harbor)
    output = tmp_path / "pilot"
    with pytest.raises(SystemExit):
        canary.main(["--api-base", "http://localhost:8000/v1", "--output-dir", str(output), *options])
    harbor.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("damage", ["missing_completion", "changed_artifact", "wrong_endpoint"])
def test_full_pilot_rejects_unusable_smoke_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Stop before trials when the smoke check is absent, altered, or for another runtime."""
    manifest, contract = run_contract(tmp_path / "run")
    write_pilot_fixture(tmp_path / "pilot", contract, manifest)
    smoke = tmp_path / "pilot" / "smoke"
    endpoint = contract["student_api_base"]
    if damage == "missing_completion":
        (smoke / "pilot-complete.json").unlink()
    elif damage == "changed_artifact":
        (smoke / "task-results.json").write_text("[]")
    else:
        endpoint = "http://localhost:9000/v1"
    harbor = Mock()
    monkeypatch.setattr(canary, "HarborCLI", harbor)
    with pytest.raises(SystemExit):
        canary.main(
            [
                "--api-base",
                endpoint,
                "--stage",
                "full",
                "--smoke-dir",
                str(smoke),
                "--output-dir",
                str(tmp_path / "full"),
            ]
        )
    harbor.assert_not_called()
    assert not (tmp_path / "full").exists()


@pytest.mark.parametrize("failure", ["exception", "partial", "invalid_reward"])
def test_incomplete_full_pilot_preserves_usage_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Never certify a failed, partial, or invalidly scored full stage."""
    manifest, contract = run_contract(tmp_path / "run")
    write_pilot_fixture(tmp_path / "pilot", contract, manifest)
    monkeypatch.setattr(canary.HarborCLI, "check_requirements", Mock())

    def evaluate(adapter, tasks, candidate):
        batch = offline_adapter(adapter, tasks, candidate)
        if failure == "exception":
            raise RuntimeError("provider failed")
        if failure == "partial":
            batch.outputs.pop()
        else:
            batch.outputs[0]["reward"] = float("nan")
        return batch

    monkeypatch.setattr(canary.TerminusAdapter, "evaluate", evaluate)
    output = tmp_path / "failed-full"
    with pytest.raises((RuntimeError, ValueError)):
        canary.main(
            [
                "--api-base",
                contract["student_api_base"],
                "--stage",
                "full",
                "--smoke-dir",
                str(tmp_path / "pilot" / "smoke"),
                "--output-dir",
                str(output),
            ]
        )
    assert (output / "token-usage-summary.json").exists()
    assert not (output / "pilot-complete.json").exists()


@pytest.mark.parametrize(
    "damage", ["unreviewed", "smoke_only", "partial", "changed_cap", "changed_concurrency", "changed_seed"]
)
def test_review_requires_full_coverage_and_matching_runtime(tmp_path: Path, damage: str) -> None:
    """Reject pilot shortcuts and runtime drift before campaign admission."""
    manifest, contract = run_contract(tmp_path / "run")
    full_dir = write_pilot_fixture(tmp_path / "pilot", contract, manifest)
    review = review_pilot(full_dir, contract, manifest)
    if damage == "unreviewed":
        review["reviewed_metrics"] = []
    elif damage == "smoke_only":
        review["full_pilot"] = load_completed_pilot(full_dir.parent / "smoke", manifest, "smoke")
    elif damage == "partial":
        review["full_pilot"]["config"]["task_ids"].pop()
    elif damage == "changed_cap":
        contract["student_decoding"]["max_tokens"] = 16384
    elif damage == "changed_concurrency":
        contract["n_concurrent"] = 2
    else:
        contract["reference_seed_digest"] = "different-initial-harness"
    with pytest.raises(ValueError):
        validate_review(review, contract, manifest)


def test_campaign_cannot_start_before_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a full-data optimization before creating output or contacting Harbor."""
    harbor = Mock()
    monkeypatch.setattr(campaign, "HarborCLI", harbor)
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--condition",
            "vanilla",
            "--run-dir",
            str(output),
            "--harbor-work-dir",
            str(output / "harbor"),
        ],
    )
    with pytest.raises(SystemExit):
        campaign.main()
    harbor.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize(
    "artifact", ["canary-config.json", "task-results.json", "token-usage-summary.json", "pilot-summary.json"]
)
def test_review_rejects_artifacts_changed_after_completion(tmp_path: Path, artifact: str) -> None:
    """Bind the reviewed results and measurements to the actual completed pilot."""
    manifest, contract = run_contract(tmp_path / "run")
    full_dir = write_pilot_fixture(tmp_path / "pilot", contract, manifest)
    value = json.loads((full_dir / artifact).read_text())
    value = value[:-1] if isinstance(value, list) else {**deepcopy(value), "modified": True}
    (full_dir / artifact).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifacts changed"):
        review_pilot(full_dir, contract, manifest)
