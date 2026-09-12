"""Offline contract tests for the pinned Harbor Terminal-Bench adapter."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

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
from gepa.adapters.terminal_bench_adapter.documents import (
    COMMAND_FORMAT_SEED,
    COMPONENT_KINDS,
    TASK_FIELDS,
    render_initial_instructions,
    seed_documents,
)
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM
from gepa.strategies.intervention import summarize_feedback
from gepa.strategies.text_limits import TextLimits, clip_text

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "examples" / "terminalbench" / "terminalbench-v2.1-manifest.json"
_QWEN3_8_27B_MODEL = "hosted_vllm/Qwen/Qwen3.8-27B"
_QWEN3_8_27B_LM_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 16_384,
    "num_retries": 0,
}
_QWEN3_8_27B_MODEL_INFO = {
    "max_input_tokens": 32_768,
    "max_output_tokens": 16_384,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
_RUNNER_OPTIONS = {
    "manifest": load_terminalbench_manifest(MANIFEST_PATH),
    "student_model": _QWEN3_8_27B_MODEL,
    "agent_python_path": REPO_ROOT,
    "harbor_executable": "harbor",
    "docker_executable": "docker",
    "n_concurrent": 2,
    "student_agent_kwargs": {
        "llm_kwargs": dict(_QWEN3_8_27B_LM_KWARGS),
        "model_info": dict(_QWEN3_8_27B_MODEL_INFO),
    },
}
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


def _candidate(**documents: str) -> dict[str, str]:
    """Build a complete bundle with selected test documents replaced.

    Args:
        **documents: Component values overridden by the test.

    Returns:
        Complete generic document bundle.
    """
    return {**seed_documents("generic"), **documents}


def test_terminus_adapter_alias_preserves_public_api() -> None:
    """Keep the documented pre-existing adapter name importable."""
    assert TerminusAdapter is TerminalBenchAdapter
    assert TerminusAdapter.__name__ == "TerminusAdapter"


def _write_job_result(job_dir: Path, task_count: int, *, errored_trials: int = 0) -> None:
    """Write the pinned Harbor job-status fields consumed by the adapter.

    Args:
        job_dir: Harbor job directory receiving ``result.json``.
        task_count: Expected and completed trial count.
        errored_trials: Reported trial exceptions, including verified timeouts.
    """
    payload = {
        "finished_at": "2026-08-22T12:00:00Z",
        "n_total_trials": task_count,
        "stats": {
            "n_completed_trials": task_count,
            "n_errored_trials": errored_trials,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
        },
    }
    (job_dir / "result.json").write_text(json.dumps(payload))


def _write_trial_result(
    job_dir: Path,
    task_id: str,
    *,
    reward: float | None = 0.0,
    emit_atif: bool = True,
    trial_exception: bool = False,
    step_exception: bool = False,
) -> None:
    """Write one minimal trial with configurable verifier and ATIF evidence.

    Args:
        job_dir: Harbor job directory receiving the trial.
        task_id: Fully qualified task identity stored in the result.
        reward: Canonical verifier reward, or ``None`` to omit it.
        emit_atif: Write a structurally valid trajectory when true.
        trial_exception: Attach a trial-level execution error.
        step_exception: Attach an agent-step execution error.
    """
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

    assert manifest.dataset["reference"] == "terminal-bench/terminal-bench-2-1"
    assert manifest.dataset["harbor_version"] == "0.22.0"
    assert manifest.dataset["registry_content_hash"] == (
        "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
    )
    assert manifest.dataset["version"] == "2.1"
    assert manifest.dataset["registry_version_id"] == "f92eea12-ff70-4d30-ace0-003abf294998"
    assert len(manifest.task_refs) == 89
    assert {split: len(ids) for split, ids in manifest.splits.items()} == {
        "train": 30,
        "val": 19,
        "test": 40,
    }
    split_sets = {name: set(task_ids) for name, task_ids in manifest.splits.items()}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert set().union(*split_sets.values()) == set(manifest.task_refs)
    assert (
        derive_terminalbench_splits(
            list(manifest.task_refs), manifest.split_policy["seed"], manifest.split_policy["counts"]
        )
        == manifest.splits
    )


def test_tb21_preserves_every_approved_task_name_split_assignment() -> None:
    """Keep the approved 30/19/40 memberships when switching to revised Hub tasks."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    task_names = {
        split: [task.removeprefix("terminal-bench/") for task in tasks] for split, tasks in manifest.splits.items()
    }
    assert hashlib.sha256(json.dumps(task_names, sort_keys=True).encode()).hexdigest() == (
        "0b83d1fba1b18f53b1dd7ce2a5c715980d43274aa619a414ae57be380a9b6d45"
    )


def test_job_config_fixes_dataset_agent_tools_skills_and_turn_policy(tmp_path: Path) -> None:
    """Keep all non-prompt benchmark axes fixed in generated Harbor jobs.

    Args:
        tmp_path: Pytest directory used for isolated Harbor artifacts.
    """
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    task_ids = ["terminal-bench/bn-fit-modify", "terminal-bench/cancel-async-tasks"]
    config = runner.build_job_config(
        task_ids,
        prompt_path=tmp_path / "prompt.txt",
        bundle_path=tmp_path / "document-bundle.json",
        jobs_dir=tmp_path / "jobs",
        job_name="candidate-abc",
    )

    assert config["datasets"] == [
        {
            "name": "terminal-bench/terminal-bench-2-1",
            "ref": "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a",
            "task_names": task_ids,
        }
    ]
    assert "version" not in config["datasets"][0]
    agent = config["agents"][0]
    assert agent["import_path"] == "examples.terminalbench.terminus_agent:PromptedTerminus"
    assert agent["model_name"] == _QWEN3_8_27B_MODEL
    assert agent["skills"] == []
    assert agent["kwargs"]["document_bundle_path"] == str(tmp_path / "document-bundle.json")
    assert "disable_skills" not in agent["kwargs"]
    assert agent["kwargs"]["llm_kwargs"] == _QWEN3_8_27B_LM_KWARGS
    assert agent["kwargs"]["model_info"] == _QWEN3_8_27B_MODEL_INFO
    assert agent["kwargs"]["enable_summarize"] is True
    assert agent["kwargs"]["proactive_summarization_threshold"] == 8_000
    assert "max_turns" not in agent["kwargs"]
    assert "max_episodes" not in agent["kwargs"]
    assert config["environment"]["type"] == "docker"

    with pytest.raises(ValueError, match="max_turns"):
        HarborCLI(
            manifest=load_terminalbench_manifest(MANIFEST_PATH),
            student_model=_QWEN3_8_27B_MODEL,
            work_dir=tmp_path / "invalid",
            agent_python_path=REPO_ROOT,
            student_agent_kwargs={"max_turns": 5},
        )


@pytest.mark.parametrize("settings", [{"enable_summarize": False}, {"proactive_summarization_threshold": 0}])
def test_task_context_settings_cannot_be_overridden(tmp_path: Path, settings: dict) -> None:
    """Reject agent overrides that would change the campaign's context management."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    with pytest.raises(ValueError, match="cannot override fixed harness keys"):
        HarborCLI(
            manifest=manifest,
            student_model=_QWEN3_8_27B_MODEL,
            work_dir=tmp_path,
            agent_python_path=REPO_ROOT,
            student_agent_kwargs=settings,
        )


def test_rendered_prompt_preserves_candidate_braces_and_runtime_fields() -> None:
    """Escape candidate braces without breaking Harbor's task/state formatting."""
    rendered = render_terminus_prompt(_candidate(instruction_prompt="Use {literal} syntax."))
    formatted = rendered.format(instruction="TASK", terminal_state="STATE")

    assert "Use {literal} syntax." in formatted
    assert "Task Description:\nTASK" in formatted
    assert "Current terminal state:\nSTATE" in formatted


def test_empty_candidate_preserves_only_observed_task_and_terminal_inputs() -> None:
    """Keep runtime inputs even when all candidate-authored instructions are empty."""
    rendered = render_terminus_prompt(dict.fromkeys(COMPONENT_KINDS, ""))
    assert rendered == TASK_FIELDS
    assert "Task Description:\nTASK" in rendered.format(instruction="TASK", terminal_state="STATE")


def test_command_instructions_are_editable_without_reintroducing_the_seed() -> None:
    """Let edits replace tool-format guidance while preserving literal braces."""
    seed = render_terminus_prompt(_candidate()).format(instruction="TASK", terminal_state="STATE")
    assert COMMAND_FORMAT_SEED.strip() in seed
    edited = render_terminus_prompt(_candidate(command_format="Changed tool instructions {literal}."))
    formatted = edited.format(instruction="TASK", terminal_state="STATE")
    assert "Changed tool instructions {literal}." in formatted
    assert COMMAND_FORMAT_SEED.strip() not in formatted


@pytest.mark.parametrize("path", [MANIFEST_PATH])
def test_manifest_pins_reject_task_ref_changes_and_split_overlap(path: Path, tmp_path: Path) -> None:
    """Reject altered task content and accidental training/test leakage for the pinned dataset."""
    payload = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    task_id = next(iter(changed["task_refs"]))
    changed["task_refs"][task_id] = "different-source"
    invalid = tmp_path / "manifest.json"
    invalid.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="pinned official task set"):
        load_terminalbench_manifest(invalid)
    payload["splits"]["test"][0] = payload["splits"]["train"][0]
    invalid.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="splits overlap"):
        load_terminalbench_manifest(invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("identifier", "terminal-bench"),
        ("version", "2.0"),
        ("registry_content_hash", "latest"),
        ("task_refs_digest", "ad453479e7854db2737c4ff246fbfdcd26b7dbd02df285f03c19f51aefec7efc"),
    ],
)
def test_tb21_manifest_rejects_legacy_or_mutable_dataset_pins(tmp_path: Path, field: str, value: str) -> None:
    """Prevent a version label from hiding old task contents or a moving registry ref."""
    payload = json.loads(MANIFEST_PATH.read_text())
    payload["dataset"][field] = value
    path = tmp_path / "old-dataset.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=f"manifest dataset.{field} must be"):
        load_terminalbench_manifest(path)


def test_tb21_jobs_pin_registry_contents_and_require_matching_tasks_and_manifest(tmp_path: Path) -> None:
    """Use the immutable Hub dataset while rejecting foreign task and manifest inputs."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path, **_RUNNER_OPTIONS)
    config = runner.build_job_config(
        ["terminal-bench/bn-fit-modify"],
        prompt_path=tmp_path / "prompt.txt",
        bundle_path=tmp_path / "document-bundle.json",
        jobs_dir=tmp_path / "jobs",
        job_name="tb21",
    )
    assert "tasks" not in config
    assert config["datasets"] == [
        {
            "name": "terminal-bench/terminal-bench-2-1",
            "ref": manifest.dataset["registry_content_hash"],
            "task_names": ["terminal-bench/bn-fit-modify"],
        }
    ]
    agent = config["agents"][0]
    assert agent["import_path"].endswith(":PromptedTerminus")
    assert agent["kwargs"]["document_bundle_path"] == str(tmp_path / "document-bundle.json")
    other_path = tmp_path / "other-manifest.json"
    other_path.write_text(MANIFEST_PATH.read_text())
    with pytest.raises(ValueError, match="same Terminal-Bench manifest"):
        TerminalBenchAdapter(load_terminalbench_manifest(other_path), runner)
    with pytest.raises(ValueError, match="not in pinned"):
        runner.build_job_config(
            ["terminal-bench/not-a-task"],
            prompt_path=tmp_path / "prompt.txt",
            bundle_path=tmp_path / "document-bundle.json",
            jobs_dir=tmp_path,
            job_name="wrong-task",
        )


def test_tb21_evaluation_keeps_literal_prompt_and_reports_its_own_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow TB2.1's full text bundle through materialization, scoring, and reflection."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path, **{**_RUNNER_OPTIONS, "manifest": manifest})
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    candidate = _candidate(instruction_prompt="Inspect {literal} and {instruction}; שלום.")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Write verifier evidence while checking the actual rendered model input."""
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        prompt = (
            Path(config["agents"][0]["kwargs"]["prompt_template_path"])
            .read_text()
            .format(instruction="REAL_TASK", terminal_state="REAL_STATE")
        )
        assert candidate["instruction_prompt"] in prompt
        assert render_initial_instructions(candidate) in prompt
        assert "Task Description:\nREAL_TASK" in prompt
        assert "Current terminal state:\nREAL_STATE" in prompt
        assert (config_path.parent / "skills/skill_debugging/SKILL.md").exists()
        assert (config_path.parent / "skills/skill_verification/SKILL.md").exists()
        bundle = json.loads(Path(config["agents"][0]["kwargs"]["document_bundle_path"]).read_text())
        assert bundle["documents"] == candidate
        saved = json.loads((config_path.parent / "candidate.json").read_text())
        assert saved["documents"] == candidate
        assert saved["experiment"] == "tb2.1"
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1)
        _write_trial_result(job_dir, config["datasets"][0]["task_names"][0], reward=1.0)
        return subprocess.CompletedProcess(command, 0, "complete", "")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", run)
    adapter = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))
    with pytest.raises(ValueError, match="complete document bundle"):
        adapter.evaluate(manifest.tasks("train", 1), {"system_prompt": "old experiment"})
    result = adapter.evaluate(manifest.tasks("train", 1), candidate, capture_traces=True)
    assert result.scores == [1.0] and result.num_metric_calls == 1
    rows = adapter.make_reflective_dataset(candidate, result, list(candidate))
    assert set(rows) == set(COMPONENT_KINDS)
    assert all(entries[0]["Inputs"]["dataset"] == "terminal-bench/terminal-bench-2-1" for entries in rows.values())
    assert all(entries[0]["Document"]["kind"] == COMPONENT_KINDS[name] for name, entries in rows.items())
    with pytest.raises(ValueError, match="Unknown Terminal Bench document selection"):
        adapter.make_reflective_dataset(candidate, result, ["system_prompt"])


def test_requirements_fail_when_harbor_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a missing Harbor CLI before checking Docker or launching jobs.

    Args:
        tmp_path: Pytest directory used to configure the runner.
        monkeypatch: Pytest fixture used to hide both executables.
    """
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(terminalbench_module.shutil, "which", Mock(return_value=None))

    with pytest.raises(HarborRequirementError, match="Harbor executable"):
        runner.check_requirements()


def test_requirements_fail_when_docker_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a missing Docker CLI before any benchmark launch.

    Args:
        tmp_path: Pytest directory used to configure the runner.
        monkeypatch: Pytest fixture used to expose only Harbor.
    """
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(terminalbench_module.shutil, "which", Mock(side_effect=["/mock/harbor", None]))

    with pytest.raises(HarborRequirementError, match="Docker executable"):
        runner.check_requirements()


def test_requirements_enforce_exact_harbor_and_running_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept only Harbor 0.22.0 and a successful Docker daemon query.

    Args:
        tmp_path: Pytest directory used to configure the runner.
        monkeypatch: Pytest fixture used to replace executable lookup and runs.
    """
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(
        terminalbench_module.shutil,
        "which",
        Mock(side_effect=["/mock/harbor", "/mock/docker"]),
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return deterministic version and readiness results.

        Args:
            command: Harbor version or Docker readiness command.
            **_kwargs: Subprocess options irrelevant to the fixed response.

        Returns:
            Successful process result for the requested command.
        """
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
    """Mock Harbor end to end and retain exact task ordering and ATIF evidence.

    Args:
        tmp_path: Pytest directory receiving candidate-isolated jobs.
        monkeypatch: Pytest fixture used to replace requirements and subprocess.
    """
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    captured_configs: list[Path] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Materialize the Harbor output layout consumed by the parser.

        Args:
            command: Generated Harbor invocation containing the config path.
            **_kwargs: Subprocess options irrelevant to the mock.

        Returns:
            Successful Harbor process result.
        """
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
    adapter = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))
    evaluated = adapter.evaluate(batch, _candidate(instruction_prompt=SEED_PROMPT), capture_traces=True)

    assert [output["task_id"] for output in evaluated.outputs] == [task.task_id for task in batch]
    assert evaluated.scores == [1.0, 0.0]
    assert evaluated.num_metric_calls == 2
    for output in evaluated.outputs:
        assert output["candidate_digest"] == manifest.candidate_digest(_candidate(instruction_prompt=SEED_PROMPT))
        config = json.loads(Path(output["config_path"]).read_text())
        assert Path(output["job_dir"]) == Path(config["jobs_dir"]) / config["job_name"]
        assert output["evaluation_id"] == Path(output["config_path"]).parent.name
    assert evaluated.trajectories is not None
    assert [trajectory["task_id"] for trajectory in evaluated.trajectories] == [task.task_id for task in batch]
    assert all(
        trajectory["atif_trajectories"][0]["schema_version"] == "ATIF-v1.7" for trajectory in evaluated.trajectories
    )
    assert all(not trajectory["errors"] for trajectory in evaluated.trajectories)

    reflective = adapter.make_reflective_dataset(
        _candidate(instruction_prompt=SEED_PROMPT),
        evaluated,
        ["instruction_prompt"],
    )
    assert [row["Inputs"]["task_id"] for row in reflective["instruction_prompt"]] == [task.task_id for task in batch]
    assert reflective["instruction_prompt"][0]["Generated Outputs"]["atif_trajectories"]
    for component in COMPONENT_KINDS:
        rows = adapter.make_reflective_dataset(_candidate(instruction_prompt=SEED_PROMPT), evaluated, [component])
        assert rows[component][0]["Document"]["kind"] == COMPONENT_KINDS[component]
        assert rows[component][0]["Generated Outputs"]["atif_trajectories"]
        assert rows[component][0]["Generated Outputs"]["trial_result"] == evaluated.trajectories[0]["trial_result"]
        assert rows[component][0]["Generated Outputs"]["harbor_process"]["returncode"] == 0
        assert rows[component][0]["Document"]["text"] == _candidate(instruction_prompt=SEED_PROMPT)[component]

    runner.run([batch[0].task_id], _candidate(instruction_prompt=SEED_PROMPT))
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
        ("job", "not a complete single-attempt job"),
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
    """Never convert Harbor infrastructure or evidence failures into scores.

    Args:
        tmp_path: Pytest directory receiving the invalid job.
        monkeypatch: Pytest fixture used to replace requirements and subprocess.
        failure: Boundary failure to materialize.
        match: Expected adapter error text.
    """
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Materialize exactly one invalid Harbor boundary condition.

        Args:
            command: Generated Harbor invocation containing the config path.
            **_kwargs: Subprocess options irrelevant to the mock.

        Returns:
            Process result whose status matches the selected failure.
        """
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
    adapter = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))

    with pytest.raises(HarborExecutionError, match=match):
        adapter.evaluate([task], _candidate(instruction_prompt=SEED_PROMPT), capture_traces=True)


def test_runner_preserves_valid_verified_zero_reward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a complete verifier-produced zero distinct from infrastructure failure.

    Args:
        tmp_path: Pytest directory receiving the successful zero-score job.
        monkeypatch: Pytest fixture used to replace requirements and subprocess.
    """
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Write one clean completed trial whose canonical reward is zero.

        Args:
            command: Generated Harbor invocation containing the config path.
            **_kwargs: Subprocess options irrelevant to the mock.

        Returns:
            Successful Harbor process result.
        """
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1)
        _write_trial_result(job_dir, task.task_id, reward=0.0, emit_atif=True)
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", fake_run)
    evaluated = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text")).evaluate(
        [task],
        _candidate(instruction_prompt=SEED_PROMPT),
        capture_traces=True,
    )

    assert evaluated.scores == [0.0]
    assert evaluated.outputs[0]["errors"] == []
    assert evaluated.trajectories is not None
    assert evaluated.trajectories[0]["atif_trajectories"]


@pytest.mark.parametrize("manifest_path", [MANIFEST_PATH], ids=["tb2.1"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("scope", ["trial", "step"])
def test_verified_agent_timeout_counts_once_and_reaches_reflection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_path: Path, reward: float, scope: str
) -> None:
    """Keep the actual timeout reward and diagnostics without retrying the task."""
    manifest = load_terminalbench_manifest(manifest_path)
    runner = HarborCLI(work_dir=tmp_path, **{**_RUNNER_OPTIONS, "manifest": manifest})
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = manifest.tasks("train", 1)[0]

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Emit a verified task whose agent reached the official time limit."""
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert config["n_attempts"] == 1
        assert config["retry"]["max_retries"] == 0
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1, errored_trials=int(scope == "trial"))
        _write_trial_result(job_dir, task.task_id, reward=reward)
        result_path = job_dir / "trial-0/result.json"
        result = json.loads(result_path.read_text())
        exception = {"exception_type": "AgentTimeoutError", "exception_message": "Official agent time limit reached"}
        if scope == "trial":
            result["exception_info"] = exception
        else:
            result["step_results"] = [
                {"step_name": "solve", "exception_info": exception, "verifier_result": {"rewards": {"reward": reward}}}
            ]
            step_dir = result_path.parent / "steps/solve"
            step_dir.mkdir(parents=True)
            (result_path.parent / "agent").rename(step_dir / "agent")
        result_path.write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    process = Mock(side_effect=run)
    monkeypatch.setattr(terminalbench_module.subprocess, "run", process)
    adapter = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))
    candidate = _candidate()
    evaluated = adapter.evaluate([task], candidate, capture_traces=True)

    process.assert_called_once()
    assert evaluated.scores == [reward]
    assert evaluated.num_metric_calls == 1
    assert "AgentTimeoutError" in evaluated.outputs[0]["errors"][0]
    rows = adapter.make_reflective_dataset(candidate, evaluated, list(COMPONENT_KINDS))
    for entries in rows.values():
        feedback = json.loads(entries[0]["Feedback"])
        assert feedback["reward"] == reward
        assert "Official agent time limit reached" in feedback["errors"][0]


@pytest.mark.parametrize(
    "failure",
    ["unverified_timeout", "unverified_step_timeout", "provider", "verifier", "nan_reward", "retried", "error_count"],
)
def test_timeout_policy_rejects_unverified_infrastructure_and_extra_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Never let a timeout exception conceal missing evidence or an infrastructure error."""
    runner = HarborCLI(work_dir=tmp_path, **_RUNNER_OPTIONS)
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = runner.manifest.tasks("train", 1)[0]

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Write a nominally completed job with one invalid timeout-policy condition."""
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1, errored_trials=0 if failure in {"unverified_step_timeout", "error_count"} else 1)
        _write_trial_result(job_dir, task.task_id, reward=None if failure == "unverified_timeout" else 1.0)
        result_path = job_dir / "trial-0/result.json"
        result = json.loads(result_path.read_text())
        exception_type = {"provider": "ConnectionError", "verifier": "VerifierTimeoutError"}.get(
            failure, "AgentTimeoutError"
        )
        exception = {"exception_type": exception_type, "exception_message": "failed"}
        result["exception_info"] = exception
        if failure == "unverified_step_timeout":
            result["exception_info"] = None
            result["step_results"] = [{"step_name": "solve", "exception_info": exception, "verifier_result": None}]
        if failure == "nan_reward":
            result["verifier_result"]["rewards"]["reward"] = float("nan")
        result_path.write_text(json.dumps(result))
        if failure == "retried":
            job_path = job_dir / "result.json"
            job = json.loads(job_path.read_text())
            job["stats"]["n_retries"] = 1
            job_path.write_text(json.dumps(job))
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    process = Mock(side_effect=run)
    monkeypatch.setattr(terminalbench_module.subprocess, "run", process)
    with pytest.raises(HarborExecutionError):
        TerminalBenchAdapter(runner.manifest, runner, text_scope=TerminalBenchTextScope("all_text")).evaluate(
            [task], _candidate(), capture_traces=True
        )
    process.assert_called_once()
    assert len(list(tmp_path.glob("evaluations/*/jobs/*/trial-0/result.json"))) == 1


@pytest.mark.parametrize("manifest_path", [MANIFEST_PATH], ids=["tb2.1"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("maximum", [None, 128])
def test_verifier_console_output_is_textual_feedback_for_every_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_path: Path, reward: float, maximum: int | None
) -> None:
    """Carry actual verifier text through Harbor parsing and reflection without changing the score."""
    manifest = load_terminalbench_manifest(manifest_path)
    runner = HarborCLI(
        work_dir=tmp_path,
        text_limits=TextLimits(verifier_log_chars=maximum),
        **{**_RUNNER_OPTIONS, "manifest": manifest},
    )
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = manifest.tasks("train", 1)[0]
    diagnostics = {
        "verifier/test-stdout.txt": (
            "FAILED test_output: expected result.json to contain all records.\n" + "x" * 300 + "\nשלום\n"  # noqa: RUF001
        ),
        "verifier/test-stderr.txt": "Warning: optional diagnostic message\n",
    }
    expected_logs = {path: clip_text(content, maximum, head_and_tail=True) for path, content in diagnostics.items()}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Materialize the pinned Harbor layout with verifier log-file contents."""
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        job_dir = Path(config["jobs_dir"]) / config["job_name"]
        job_dir.mkdir(parents=True)
        _write_job_result(job_dir, 1)
        _write_trial_result(job_dir, task.task_id, reward=reward)
        trial_dir = job_dir / "trial-0"
        for relative_path, content in diagnostics.items():
            log = trial_dir / relative_path
            log.parent.mkdir(exist_ok=True)
            log.write_text(content, encoding="utf-8")
        (trial_dir / "verifier/test_source.py").write_text("private verifier implementation")
        (trial_dir / "verifier/reward.txt").write_text("999")
        return subprocess.CompletedProcess(command, 0, "complete", "")

    monkeypatch.setattr(terminalbench_module.subprocess, "run", run)
    adapter = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))
    candidate = _candidate()
    evaluated = adapter.evaluate([task], candidate, capture_traces=True)
    assert evaluated.scores == [reward]
    assert evaluated.num_metric_calls == 1
    assert evaluated.trajectories is not None
    assert evaluated.trajectories[0]["verifier_logs"] == expected_logs
    rows = adapter.make_reflective_dataset(candidate, evaluated, list(candidate))
    feedback = {entries[0]["Feedback"] for entries in rows.values()}
    assert len(feedback) == 1
    actual = json.loads(feedback.pop())
    assert actual["reward"] == reward
    assert actual["verifier_log_status"] == "available"
    assert actual["verifier_logs"] == expected_logs
    assert "private verifier implementation" not in json.dumps(rows)
    assert "שלום" in rows["instruction_prompt"][0]["Feedback"]
    reflection_lm = Mock(return_value="```Revised instruction```")
    StatelessReflectionLM(reflection_lm).reflect(candidate, rows, list(candidate))
    assert reflection_lm.call_count == 16
    assert all("FAILED test_output" in call.args[0] for call in reflection_lm.call_args_list)
    assert "FAILED test_output" in summarize_feedback(rows["instruction_prompt"])

    evaluated.trajectories[0]["verifier_logs"] = {}
    missing = adapter.make_reflective_dataset(candidate, evaluated, ["instruction_prompt"])
    assert json.loads(missing["instruction_prompt"][0]["Feedback"])["verifier_log_status"] == "unavailable"
    assert evaluated.scores == [reward]

    for split in ("val", "test"):
        evaluated.trajectories[0]["task_id"] = manifest.splits[split][0]
        with pytest.raises(ValueError, match="restricted to training tasks"):
            adapter.make_reflective_dataset(candidate, evaluated, ["instruction_prompt"])


@pytest.mark.parametrize("maximum", [None, 8192])
def test_verifier_log_reader_preserves_ends_step_identity_and_original_files(tmp_path: Path, maximum: int | None) -> None:
    """Preserve full logs by default, or marked excerpts under an explicit cap."""
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    stdout = verifier / "test-stdout.txt"
    content = b"TEST HEADER\r\n" + ("\u00e9" * 10000).encode() + b"\r\nFAILED final assertion"
    stdout.write_bytes(content)
    (verifier / "test-stderr.txt").write_bytes(b"diagnostic: \xff")
    step_dir = tmp_path / "steps" / "verify-output" / "verifier"
    step_dir.mkdir(parents=True)
    (step_dir / "test-stdout.txt").write_text("Step-specific failure", encoding="utf-8")

    logs = terminalbench_module._read_verifier_logs(tmp_path, max_chars=maximum)
    shortened = logs["verifier/test-stdout.txt"]
    assert shortened.startswith("TEST HEADER\r\n")
    assert shortened.endswith("FAILED final assertion")
    if maximum is None:
        assert shortened == content.decode()
    else:
        assert f"{len(content.decode()) - maximum} characters omitted" in shortened
        assert len(shortened) < maximum + 100
        assert "\ufffd" not in shortened
    assert logs["verifier/test-stderr.txt"] == "diagnostic: \ufffd"
    assert logs["steps/verify-output/verifier/test-stdout.txt"] == "Step-specific failure"
    assert stdout.read_bytes() == content
    assert terminalbench_module._read_verifier_logs(tmp_path / "missing-trial") == {}


@pytest.mark.parametrize("invalid_path", ["directory", "external_symlink"])
def test_verifier_log_reader_rejects_unreadable_or_external_artifacts(tmp_path: Path, invalid_path: str) -> None:
    """Do not silently drop present diagnostics or read outside the current trial."""
    trial = tmp_path / "trial"
    verifier = trial / "verifier"
    verifier.mkdir(parents=True)
    stdout = verifier / "test-stdout.txt"
    if invalid_path == "directory":
        stdout.mkdir()
    else:
        outside = tmp_path / "outside.txt"
        outside.write_text("unrelated artifact")
        stdout.symlink_to(outside)
    with pytest.raises(HarborExecutionError, match=r"unreadable|outside"):
        terminalbench_module._read_verifier_logs(trial)


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
    """Reject malformed JSON and parseable documents that are not ATIF trajectories.

    Args:
        tmp_path: Pytest directory receiving the malformed trajectory.
        monkeypatch: Pytest fixture used to replace requirements and subprocess.
        raw_trajectory: Exact trajectory text written by the mock.
        match: Expected structural-validation error text.
    """
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Write one successful Harbor trial with invalid trajectory evidence.

        Args:
            command: Generated Harbor invocation containing the config path.
            **_kwargs: Subprocess options irrelevant to the mock.

        Returns:
            Successful Harbor process result.
        """
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
        TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text")).evaluate(
            [task],
            _candidate(instruction_prompt=SEED_PROMPT),
            capture_traces=True,
        )


def test_runner_wraps_atif_file_read_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate trajectory filesystem errors into the Harbor boundary error.

    Args:
        tmp_path: Pytest directory receiving the unreadable trajectory path.
        monkeypatch: Pytest fixture used to replace requirements and subprocess.
    """
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(work_dir=tmp_path / "harbor", **_RUNNER_OPTIONS)
    monkeypatch.setattr(runner, "check_requirements", Mock(return_value=("/mock/harbor", "/mock/docker")))
    task = manifest.tasks("val", 1)[0]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Create a directory where Harbor's trajectory file should be.

        Args:
            command: Generated Harbor invocation containing the config path.
            **_kwargs: Subprocess options irrelevant to the mock.

        Returns:
            Successful Harbor process result.
        """
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
        TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text")).evaluate(
            [task],
            _candidate(instruction_prompt=SEED_PROMPT),
            capture_traces=True,
        )


@pytest.mark.smoke
def test_real_harbor_terminalbench_single_task_smoke(tmp_path: Path) -> None:
    """Run one official task only when the costly smoke test is explicitly enabled.

    Args:
        tmp_path: Pytest directory receiving the live Harbor job.
    """
    if os.environ.get("GEPA_TERMINALBENCH_SMOKE") != "1":
        pytest.skip("set GEPA_TERMINALBENCH_SMOKE=1 to run Harbor/Docker")
    student_model = os.environ.get("GEPA_TERMINALBENCH_STUDENT_MODEL")
    if not student_model:
        pytest.skip("set GEPA_TERMINALBENCH_STUDENT_MODEL to a valid Harbor/LiteLLM model ID")

    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    runner = HarborCLI(
        manifest=load_terminalbench_manifest(MANIFEST_PATH),
        student_model=student_model,
        work_dir=tmp_path / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=1,
    )
    adapter = TerminalBenchAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))
    evaluated = adapter.evaluate(
        manifest.tasks("val", 1),
        _candidate(instruction_prompt=SEED_PROMPT),
        capture_traces=True,
    )

    assert len(evaluated.scores) == 1
    assert evaluated.trajectories is not None
    assert evaluated.trajectories[0]["atif_trajectories"]
