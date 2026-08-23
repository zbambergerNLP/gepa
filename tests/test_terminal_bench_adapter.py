"""Offline contract tests for the pinned Harbor Terminal-Bench adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import gepa.adapters.terminal_bench_adapter.terminal_bench_adapter as terminalbench_module
from gepa.adapters.terminal_bench_adapter import (
    HarborCLI,
    HarborExecutionError,
    HarborRequirementError,
    TerminalBenchAdapter,
    TerminusAdapter,
    derive_terminalbench_splits,
    load_terminalbench_manifest,
    render_terminus_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "examples" / "terminalbench" / "terminalbench-v3-manifest.json"
SEED_PROMPT = """## Role
Terminal agent.

## Task
Solve the task.

## Context
Use the terminal state.

## Rules
Verify the result.

## Reasoning
Iterate on command output.

## Examples


## Output Format
Use the fixed Terminus JSON format."""


def test_terminus_adapter_alias_preserves_public_api() -> None:
    """Keep the documented pre-existing adapter name importable."""
    assert TerminusAdapter is TerminalBenchAdapter


def _runner(tmp_path: Path) -> HarborCLI:
    """Build a runner whose external calls can be mocked by each test.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Runner with fixed fake model and executable names.
    """
    return HarborCLI(
        student_model="provider/qwen3-8b",
        work_dir=tmp_path / "harbor",
        agent_python_path=REPO_ROOT,
        harbor_executable="harbor",
        docker_executable="docker",
        n_concurrent=2,
    )


def _write_job_result(job_dir: Path, task_count: int, *, errored_trials: int = 0) -> None:
    """Write the pinned Harbor job-status fields consumed by the adapter."""
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-08-22T12:00:00Z",
                "n_total_trials": task_count,
                "stats": {
                    "n_completed_trials": task_count,
                    "n_errored_trials": errored_trials,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                },
            }
        )
    )


def _write_trial_result(
    job_dir: Path,
    task_id: str,
    *,
    reward: float | None = 0.0,
    emit_atif: bool = True,
    trial_exception: bool = False,
    step_exception: bool = False,
) -> None:
    """Write one minimal trial with configurable verifier and ATIF evidence."""
    trial_dir = job_dir / "trial-0"
    trial_dir.mkdir(parents=True)
    rewards = {} if reward is None else {"reward": reward}
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_id,
                "verifier_result": {"rewards": rewards},
                "exception_info": (
                    {"exception_type": "RuntimeError", "exception_message": "trial failed"} if trial_exception else None
                ),
                "step_results": (
                    [
                        {
                            "step_name": "agent",
                            "exception_info": {
                                "exception_type": "RuntimeError",
                                "exception_message": "step failed",
                            },
                        }
                    ]
                    if step_exception
                    else None
                ),
            }
        )
    )
    if emit_atif:
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir()
        trajectory = {
            "schema_version": "ATIF-v1.7",
            "session_id": "session-0",
            "agent": {"name": "terminus-2", "version": "2.0.0"},
            "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
        }
        (agent_dir / "trajectory.json").write_text(json.dumps(trajectory))


def test_manifest_is_exactly_pinned_complete_and_disjoint() -> None:
    """Require all registry refs exactly once in deterministic splits."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    assert manifest.dataset["reference"] == "terminal-bench/terminal-bench@3.0.0"
    assert manifest.dataset["harbor_version"] == "0.22.0"
    assert manifest.dataset["registry_content_hash"] == (
        "sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3"
    )
    assert manifest.dataset["source_commit"] == "2b0442c3c583b710ca8da14c8e601b99f2f1f244"
    assert len(manifest.task_refs) == 74
    assert {split: len(ids) for split, ids in manifest.splits.items()} == {
        "train": 30,
        "val": 22,
        "test": 22,
    }
    split_sets = {name: set(task_ids) for name, task_ids in manifest.splits.items()}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert set().union(*split_sets.values()) == set(manifest.task_refs)
    assert derive_terminalbench_splits(list(manifest.task_refs), manifest.split_policy["seed"]) == manifest.splits


def test_job_config_fixes_dataset_agent_tools_skills_and_turn_policy(tmp_path: Path) -> None:
    """Keep all non-prompt benchmark axes fixed in generated Harbor jobs."""
    runner = _runner(tmp_path)
    task_ids = ["terminal-bench/cad-model", "terminal-bench/music-harmony"]
    config = runner.build_job_config(
        task_ids,
        prompt_path=tmp_path / "prompt.txt",
        jobs_dir=tmp_path / "jobs",
        job_name="candidate-abc",
    )

    assert config["datasets"] == [
        {
            "name": "terminal-bench/terminal-bench",
            "ref": "sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3",
            "task_names": task_ids,
        }
    ]
    assert "version" not in config["datasets"][0]
    agent = config["agents"][0]
    assert agent["import_path"] == "examples.terminalbench.terminus_agent:PromptedTerminus"
    assert agent["model_name"] == "provider/qwen3-8b"
    assert agent["skills"] == []
    assert agent["kwargs"]["disable_skills"] is True
    assert "max_turns" not in agent["kwargs"]
    assert "max_episodes" not in agent["kwargs"]
    assert config["environment"]["type"] == "docker"

    with pytest.raises(ValueError, match="max_turns"):
        HarborCLI(
            student_model="provider/qwen3-8b",
            work_dir=tmp_path / "invalid",
            agent_python_path=REPO_ROOT,
            student_agent_kwargs={"max_turns": 5},
        )


def test_rendered_prompt_preserves_candidate_braces_and_runtime_fields() -> None:
    """Escape candidate braces without breaking Harbor's task/state formatting."""
    rendered = render_terminus_prompt("Use {literal} syntax.")
    formatted = rendered.format(instruction="TASK", terminal_state="STATE")

    assert "Use {literal} syntax." in formatted
    assert "Task Description:\nTASK" in formatted
    assert "Current terminal state:\nSTATE" in formatted


def test_empty_candidate_renders_only_the_fixed_terminus_contract() -> None:
    """Keep an all-empty user template out of the task text while retaining the adapter contract."""
    rendered = render_terminus_prompt("")
    assert rendered == terminalbench_module.TERMINUS_JSON_CONTRACT
    assert "Task Description:\nTASK" in rendered.format(instruction="TASK", terminal_state="STATE")


def test_requirements_fail_when_harbor_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a missing Harbor CLI before checking Docker or launching jobs."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(terminalbench_module.shutil, "which", lambda _executable: None)

    with pytest.raises(HarborRequirementError, match="Harbor executable"):
        runner.check_requirements()


def test_requirements_fail_when_docker_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a missing Docker CLI before any benchmark launch."""
    runner = _runner(tmp_path)

    def fake_which(executable: str) -> str | None:
        """Resolve only the Harbor executable."""
        return "/mock/harbor" if executable == "harbor" else None

    monkeypatch.setattr(terminalbench_module.shutil, "which", fake_which)

    with pytest.raises(HarborRequirementError, match="Docker executable"):
        runner.check_requirements()


def test_requirements_enforce_exact_harbor_and_running_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept only Harbor 0.22.0 and a successful Docker daemon query."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(terminalbench_module.shutil, "which", lambda executable: f"/mock/{executable}")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return deterministic version/readiness results for subprocess calls."""
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="0.22.0\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout='"26.1"\n', stderr="")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)

    assert runner.check_requirements() == ("/mock/harbor", "/mock/docker")
    assert commands == [
        ["/mock/harbor", "--version"],
        ["/mock/docker", "info", "--format", "{{json .ServerVersion}}"],
    ]


def test_runner_isolates_candidates_and_adapter_maps_complete_evidence_by_task_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock Harbor end to end and retain exact task ordering and ATIF evidence."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "check_requirements", lambda: ("/mock/harbor", "/mock/docker"))
    captured_configs: list[Path] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Materialize the Harbor output layout consumed by the parser."""
        config_path = Path(command[command.index("--config") + 1])
        captured_configs.append(config_path)
        config = json.loads(config_path.read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, len(config["datasets"][0]["task_names"]))
        for index, task_id in enumerate(reversed(config["datasets"][0]["task_names"])):
            trial_dir = job_dir / f"trial-{index}"
            agent_dir = trial_dir / "agent"
            agent_dir.mkdir(parents=True)
            result = {
                "task_name": task_id,
                "verifier_result": {"rewards": {"reward": float(index % 2), "aux": 0.5}},
                "exception_info": None,
                "step_results": None,
            }
            (trial_dir / "result.json").write_text(json.dumps(result))
            trajectory = {
                "schema_version": "ATIF-v1.7",
                "session_id": f"session-{index}",
                "agent": {"name": "terminus-2", "version": "2.0.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": task_id}],
            }
            (agent_dir / "trajectory.json").write_text(json.dumps(trajectory))
        return subprocess.CompletedProcess(command, 0, stdout="mock Harbor complete", stderr="")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)
    batch = [manifest.tasks("train")[1], manifest.tasks("train")[0]]
    adapter = TerminalBenchAdapter(manifest, runner)
    evaluated = adapter.evaluate(batch, {"instruction_prompt": SEED_PROMPT}, capture_traces=True)

    assert [output["task_id"] for output in evaluated.outputs] == [task.task_id for task in batch]
    assert evaluated.scores == [1.0, 0.0]
    assert evaluated.num_metric_calls == 2
    assert evaluated.trajectories is not None
    assert [trajectory["task_id"] for trajectory in evaluated.trajectories] == [task.task_id for task in batch]
    assert all(
        trajectory["atif_trajectories"][0]["schema_version"] == "ATIF-v1.7" for trajectory in evaluated.trajectories
    )
    assert all(not trajectory["errors"] for trajectory in evaluated.trajectories)

    reflective = adapter.make_reflective_dataset(
        {"instruction_prompt": SEED_PROMPT},
        evaluated,
        ["instruction_prompt"],
    )
    assert [row["Inputs"]["task_id"] for row in reflective["instruction_prompt"]] == [task.task_id for task in batch]
    assert reflective["instruction_prompt"][0]["Generated Outputs"]["atif_trajectories"]

    runner.run([batch[0].task_id], SEED_PROMPT)
    assert len(captured_configs) == 2
    assert captured_configs[0] != captured_configs[1]
    first_config = json.loads(captured_configs[0].read_text())
    second_config = json.loads(captured_configs[1].read_text())
    assert (
        first_config["agents"][0]["kwargs"]["prompt_template_path"]
        != second_config["agents"][0]["kwargs"]["prompt_template_path"]
    )


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("process", "exited with status 1"),
        ("job", "not a clean completed job"),
        ("missing_task", "result/task mismatch"),
        ("missing_reward", "canonical verifier reward"),
        ("missing_atif", "ATIF trajectory"),
        ("trial_exception", "reported execution errors"),
        ("step_exception", "reported execution errors"),
    ],
)
def test_runner_rejects_incomplete_or_failed_harbor_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    match: str,
) -> None:
    """Never convert Harbor infrastructure or evidence failures into scores."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "check_requirements", lambda: ("/mock/harbor", "/mock/docker"))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Materialize exactly one invalid Harbor boundary condition."""
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1, errored_trials=1 if failure == "job" else 0)
        if failure != "missing_task":
            _write_trial_result(
                job_dir,
                task.task_id,
                reward=None if failure == "missing_reward" else 0.0,
                emit_atif=failure != "missing_atif",
                trial_exception=failure == "trial_exception",
                step_exception=failure == "step_exception",
            )
        return subprocess.CompletedProcess(
            command,
            1 if failure == "process" else 0,
            stdout="",
            stderr="Harbor failed" if failure == "process" else "",
        )

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)
    adapter = TerminalBenchAdapter(manifest, runner)

    with pytest.raises(HarborExecutionError, match=match):
        adapter.evaluate([task], {"instruction_prompt": SEED_PROMPT}, capture_traces=True)


def test_runner_preserves_valid_verified_zero_reward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a complete verifier-produced zero distinct from infrastructure failure."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "check_requirements", lambda: ("/mock/harbor", "/mock/docker"))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Write one clean completed trial whose canonical reward is zero."""
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1)
        _write_trial_result(job_dir, task.task_id, reward=0.0, emit_atif=True)
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)
    evaluated = TerminalBenchAdapter(manifest, runner).evaluate(
        [task],
        {"instruction_prompt": SEED_PROMPT},
        capture_traces=True,
    )

    assert evaluated.scores == [0.0]
    assert evaluated.outputs[0]["errors"] == []
    assert evaluated.trajectories is not None
    assert evaluated.trajectories[0]["atif_trajectories"]


@pytest.mark.parametrize(
    ("raw_trajectory", "match"),
    [
        ("{", "unreadable or invalid JSON"),
        ("null", "not a JSON object"),
        ("[]", "not a JSON object"),
        ("{}", "agent object"),
        (
            json.dumps({"schema_version": "ATIF-v1.7", "agent": {"name": "terminus-2", "version": "2.0.0"}}),
            "non-empty steps array",
        ),
        (
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": "terminus-2", "version": "2.0.0"},
                    "steps": [],
                }
            ),
            "non-empty steps array",
        ),
        (
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": "terminus-2", "version": "2.0.0"},
                    "steps": ["not-a-step"],
                }
            ),
            "steps\\[0\\] is not an object",
        ),
        (
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": "terminus-2", "version": "2.0.0"},
                    "steps": [{"step_id": 1, "source": "agent"}],
                }
            ),
            "message must be text or content parts",
        ),
    ],
)
def test_runner_rejects_malformed_or_structurally_invalid_atif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_trajectory: str,
    match: str,
) -> None:
    """Reject malformed JSON and parseable documents that are not ATIF trajectories."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "check_requirements", lambda: ("/mock/harbor", "/mock/docker"))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Write one successful Harbor trial with invalid trajectory evidence."""
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1)
        _write_trial_result(job_dir, task.task_id, emit_atif=False)
        agent_dir = job_dir / "trial-0" / "agent"
        agent_dir.mkdir()
        (agent_dir / "trajectory.json").write_text(raw_trajectory)
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)

    with pytest.raises(HarborExecutionError, match=match):
        TerminalBenchAdapter(manifest, runner).evaluate(
            [task],
            {"instruction_prompt": SEED_PROMPT},
            capture_traces=True,
        )


def test_runner_wraps_atif_file_read_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate trajectory filesystem errors into the Harbor boundary error."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "check_requirements", lambda: ("/mock/harbor", "/mock/docker"))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Create a directory where Harbor's trajectory file should be."""
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1)
        _write_trial_result(job_dir, task.task_id, emit_atif=False)
        trajectory_path = job_dir / "trial-0" / "agent" / "trajectory.json"
        trajectory_path.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)

    with pytest.raises(HarborExecutionError, match="unreadable or invalid JSON"):
        TerminalBenchAdapter(manifest, runner).evaluate(
            [task],
            {"instruction_prompt": SEED_PROMPT},
            capture_traces=True,
        )


@pytest.mark.smoke
def test_real_harbor_terminalbench_single_task_smoke(tmp_path: Path) -> None:
    """Run one official task only when the costly smoke test is explicitly enabled."""
    if os.environ.get("GEPA_TERMINALBENCH_SMOKE") != "1":
        pytest.skip("set GEPA_TERMINALBENCH_SMOKE=1 to run Harbor/Docker")
    student_model = os.environ.get("GEPA_TERMINALBENCH_STUDENT_MODEL")
    if not student_model:
        pytest.skip("set GEPA_TERMINALBENCH_STUDENT_MODEL to a valid Harbor/LiteLLM model ID")

    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(
        student_model=student_model,
        work_dir=tmp_path / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=1,
    )
    adapter = TerminalBenchAdapter(manifest, runner)
    evaluated = adapter.evaluate(
        manifest.tasks("val", 1),
        {"instruction_prompt": SEED_PROMPT},
        capture_traces=True,
    )

    assert len(evaluated.scores) == 1
    assert evaluated.trajectories is not None
    assert evaluated.trajectories[0]["atif_trajectories"]
