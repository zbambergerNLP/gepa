"""Verify prompt-only campaign priority without launching model or Docker work."""

import json
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from terminalbench_pilot_helpers import write_pilot_fixture

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import EXPERIMENT_MODELS
from examples.terminalbench import evaluate, run_ablations
from examples.terminalbench import main as campaign
from examples.terminalbench.main import REPO_ROOT, build_parser
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.core.state import GEPAState, ValsetEvaluation


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
@pytest.mark.parametrize("dry_run", [False, True])
def test_campaign_runs_prompt_only_first_and_forwards_shared_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str, dry_run: bool
) -> None:
    """Exercise the launcher with twelve separate cells and six prompt-only runs first."""
    runner = Mock()
    monkeypatch.setattr(run_ablations.subprocess, "run", runner)
    root = tmp_path / "campaign with spaces"
    run_ablations.main(
        [
            "--run-root",
            str(root),
            *(["--dry-run"] if dry_run else []),
            "--student-model",
            model,
            "--proposer-model",
            model,
            "--student-api-base",
            "http://localhost:8000/v1",
            "--proposer-api-base",
            "http://localhost:8000/v1",
            "--seed",
            "17",
            "--n-concurrent",
            "3",
            "--reviewed-pilot",
            str(tmp_path / "reviewed full pilot"),
            "--runtime-record",
            str(tmp_path / "task server.json"),
            "--proposer-runtime-record",
            str(tmp_path / "optimizer server.json"),
        ]
    )
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 12
    assert all(
        command[:6] == ["uv", "run", "--no-sync", "python", "-m", "examples.terminalbench.main"] for command in commands
    )
    cells = [build_parser().parse_args(command[6:]) for command in commands]
    assert [cell.optimization_scope for cell in cells] == ["system_prompt"] * 6 + ["all_text"] * 6
    assert [(cell.condition, cell.budget) for cell in cells[:6]] == [
        ("vanilla", "standard"),
        ("react_v2", "standard"),
        ("react_v2_random", "standard"),
        ("action", "standard"),
        ("vanilla", "double"),
        ("react_v2", "double"),
    ]
    assert [(cell.condition, cell.budget) for cell in cells[6:]] == [
        (cell.condition, cell.budget) for cell in cells[:6]
    ]
    assert len({cell.run_dir for cell in cells}) == 12
    for cell in cells:
        assert cell.run_dir.parent == root / cell.optimization_scope
        assert cell.harbor_work_dir == cell.run_dir / "harbor"
        assert cell.test_output_dir == root / "test"
        assert cell.student_model == cell.proposer_model == model
        assert cell.runtime_record == tmp_path / "task server.json"
        assert cell.proposer_runtime_record == tmp_path / "optimizer server.json"
        assert cell.student_api_base == cell.proposer_api_base == "http://localhost:8000/v1"
        assert cell.seed == 17 and cell.n_concurrent == 3
        assert cell.reviewed_pilot == tmp_path / "reviewed full pilot"
    assert not root.exists()
    if dry_run:
        runner.assert_not_called()
    else:
        assert [call.args[0] for call in runner.call_args_list] == commands
        assert all(call.kwargs == {"cwd": REPO_ROOT, "check": True} for call in runner.call_args_list)


def test_campaign_stops_before_later_cells_when_a_run_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep later ablations from launching after a failed optimization run."""
    runner = Mock(side_effect=[None, subprocess.CalledProcessError(1, "optimization")])
    monkeypatch.setattr(run_ablations.subprocess, "run", runner)
    with pytest.raises(subprocess.CalledProcessError):
        run_ablations.main(["--run-root", str(tmp_path)])
    assert runner.call_count == 2


@pytest.mark.parametrize(
    "option",
    [
        "--optimization-scope=all_text",
        "--condition=action",
        "--budget=double",
        "--run-dir=/tmp",
        "--test-output-dir=/tmp",
        "--train-limit=1",
        "--val-limit=1",
    ],
)
def test_campaign_rejects_overrides_to_its_order_and_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    """Reject conflicting campaign flags before any cell can launch."""
    runner = Mock()
    monkeypatch.setattr(run_ablations.subprocess, "run", runner)
    with pytest.raises(SystemExit):
        run_ablations.main(["--run-root", str(tmp_path), option])
    runner.assert_not_called()


@pytest.mark.parametrize("fail_test", [False, True])
def test_campaign_optimizes_then_tests_each_cell_before_starting_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_test: bool
) -> None:
    """Run the real orchestration and checkpoint validation with offline optimization and test execution."""
    root = tmp_path / "campaign"
    events = []
    manifest = load_terminalbench_manifest(campaign.EXPERIMENT_MANIFESTS["tb2.1"])
    args = build_parser().parse_args(
        ["--condition", "vanilla", "--run-dir", str(root), "--harbor-work-dir", str(root / "harbor")]
    )
    _, family = campaign.seed_candidate(args.student_model, "auto", "tb2.1")
    contract = campaign.build_run_contract(
        args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", family
    )
    pilot_dir = write_pilot_fixture(tmp_path / "pilot", contract, manifest)

    def optimize(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        contract = json.loads((run_dir / campaign.RUN_CONTRACT_FILENAME).read_text())
        cell = f"{contract['optimization_scope']}__{run_dir.name}"
        events.append(("optimize", cell))
        assert kwargs["trainset"] == manifest.tasks("train")
        assert kwargs["valset"] == manifest.tasks("val")
        state = GEPAState(
            kwargs["seed_candidate"], ValsetEvaluation({}, dict.fromkeys(range(len(manifest.splits["val"])), 0.0))
        )
        state.i = contract["optimization_budget"]["max_iterations"] - 1
        state.save(str(run_dir))

    def test(manifest_arg, comparison, output_dir, harbor):
        cell = next(iter(comparison["source_runs"]))
        events.append(("test", cell))
        assert output_dir == root / "test"
        assert manifest_arg == manifest
        assert comparison["shared_configuration"]["test_task_ids"] == manifest.splits["test"]
        if fail_test:
            raise RuntimeError("interrupted test execution")
        return {"harnesses": {}}

    def run(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["terminalbench", *command[6:]])
        campaign.main()

    monkeypatch.setattr(run_ablations.subprocess, "run", run)
    monkeypatch.setattr(campaign, "optimize", optimize)
    monkeypatch.setattr(campaign.HarborCLI, "check_requirements", Mock())
    monkeypatch.setattr(evaluate, "evaluate_comparison", test)
    options = ["--run-root", str(root), "--reviewed-pilot", str(pilot_dir)]
    if fail_test:
        with pytest.raises(RuntimeError, match="interrupted test"):
            run_ablations.main(options)
        assert events == [("optimize", "system_prompt__vanilla"), ("test", "system_prompt__vanilla")]
    else:
        run_ablations.main(options)
        assert events == [(phase, cell) for cell in campaign.SCOPE_CAMPAIGN_CELLS for phase in ("optimize", "test")]
