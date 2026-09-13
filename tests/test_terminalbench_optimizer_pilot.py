"""Exercise the optimizer-pilot CLI while replacing only external runtimes."""

import json
from unittest.mock import Mock

import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from terminalbench_pilot_helpers import write_pilot_fixture

from examples.common.pilot_checks import METHODS, CycleEvidence, load_cycle
from examples.terminalbench import main as benchmark
from examples.terminalbench.optimizer_pilot import main as run_pilots
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest


def test_both_scopes_exercise_every_method_without_heldout_tasks(tmp_path, monkeypatch):
    """Require calibrated runtime, preserve production builders, and stop before testing."""
    manifest = load_terminalbench_manifest(benchmark.EXPERIMENT_MANIFESTS["tb2.1"])
    args = benchmark.build_parser().parse_args(
        [
            "--condition",
            "vanilla",
            "--run-dir",
            str(tmp_path / "reference"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )
    _, family = benchmark.seed_candidate(
        args.student_model, args.template_family, args.experiment, args.optimization_scope
    )
    contract = benchmark.build_run_contract(
        args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", family
    )
    full = write_pilot_fixture(tmp_path / "calibration", contract, manifest)
    monkeypatch.setattr(benchmark, "HarborCLI", Mock())
    monkeypatch.setattr(benchmark, "TerminusAdapter", Mock())
    observed = []

    def optimize(**kwargs):
        assert kwargs["trainset"] == kwargs["valset"] == manifest.tasks("train", 3)
        assert kwargs["max_metric_calls"] is None
        assert kwargs["skip_perfect_score"] is True
        assert kwargs["acceptance_criterion"] == "strict_improvement"
        callback = next(cb for cb in kwargs["callbacks"] if isinstance(cb, CycleEvidence))
        stored = json.loads((callback.directory / benchmark.RUN_CONTRACT_FILENAME).read_text())
        assert stored["optimization_budget"]["max_iterations"] == 1
        observed.append((stored["optimization_scope"], stored["condition"]))
        callback.events = {
            "reflection": {"feedback": "zero reward"},
            "proposal": {"new_instructions": {"prompt": "revised"}},
            "reevaluation": {"scores": [0, 0, 0]},
            "decision": {"accepted": False},
            "finished": True,
        }
        callback._save()

    monkeypatch.setattr(benchmark, "optimize", optimize)
    run_pilots(["--output-dir", str(tmp_path / "checks"), "--optimizer-pilot-calibration", str(full)])
    assert observed == [(scope, method) for scope in ("system_prompt", "all_text") for method in METHODS]
    for scope, method in observed:
        assert load_cycle(tmp_path / "checks" / scope / method)["decision"]["accepted"] is False


def test_optimizer_checks_cannot_skip_training_calibration(tmp_path):
    """A pilot invocation must identify its completed initial-harness stages."""
    with pytest.raises(SystemExit):
        run_pilots(["--output-dir", str(tmp_path)])
    assert not list(tmp_path.rglob("optimizer-pilot-complete.json"))
