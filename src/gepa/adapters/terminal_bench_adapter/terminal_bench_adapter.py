"""Terminal-Bench v3 adapter backed by the pinned Harbor CLI.

Harbor runs in a separate Python environment through a subprocess, so GEPA
retains Python 3.10+ support. Harbor supplies the official Docker verifier and
ATIF trajectories.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

PINNED_HARBOR_VERSION = "0.22.0"
PINNED_DATASET_IDENTIFIER = "terminal-bench/terminal-bench"
PINNED_DATASET_VERSION = "3.0.0"
PINNED_DATASET_REFERENCE = f"{PINNED_DATASET_IDENTIFIER}@{PINNED_DATASET_VERSION}"
PINNED_DATASET_CONTENT_HASH = "sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3"
PINNED_SOURCE_REPOSITORY = "https://github.com/harbor-framework/terminal-bench"
PINNED_SOURCE_TAG = "v3.0.0"
PINNED_SOURCE_COMMIT = "2b0442c3c583b710ca8da14c8e601b99f2f1f244"
PINNED_TASK_COUNT = 74
PROMPTED_TERMINUS_IMPORT_PATH = "examples.terminalbench.terminus_agent:PromptedTerminus"
INSTRUCTION_COMPONENT = "instruction_prompt"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_WEIGHTS = {"train": 0.40, "val": 0.30, "test": 0.30}
SUPPORTED_ATIF_SCHEMA_VERSIONS = {f"ATIF-v1.{minor}" for minor in range(8)}

# This is the Terminus 2 JSON interaction contract from Harbor v0.22.0. GEPA
# evolves only the instruction prefix placed above it. The braces remain
# doubled because Harbor applies ``str.format`` with the task instruction and
# current tmux state at runtime.
TERMINUS_JSON_CONTRACT = r"""Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- You must end every command with a newline (\n) or it will not execute.
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it is part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid; use proper escaping for quotes and special characters within strings
- The commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""


class TerminalBenchOutput(TypedDict):
    """Opaque per-task output retained by GEPA."""

    task_id: str
    reward: float
    rewards: dict[str, float]
    errors: list[str]
    evaluation_id: str
    harbor_returncode: int
    harbor_stdout_path: str
    harbor_stderr_path: str
    trial_dir: str


class TerminalBenchTrajectory(TypedDict):
    """Complete Harbor evidence used to construct reflection records."""

    task_id: str
    candidate_prompt: str
    reward: float
    rewards: dict[str, float]
    errors: list[str]
    atif_trajectories: list[dict[str, Any]]
    trial_result: dict[str, Any]
    evaluation_id: str
    harbor_returncode: int
    harbor_stdout_path: str
    harbor_stderr_path: str
    trial_dir: str


@dataclass(frozen=True)
class TerminalBenchTask:
    """One pinned Terminal-Bench task selected from the checked-in manifest.

    Args:
        task_id: Fully qualified Harbor task ID, such as
            ``terminal-bench/cad-model``.
    """

    task_id: str


@dataclass(frozen=True)
class TerminalBenchManifest:
    """Validated task refs and deterministic train/validation/test splits."""

    path: Path
    dataset: dict[str, Any]
    split_policy: dict[str, Any]
    task_refs: dict[str, str]
    splits: dict[str, list[str]]

    def tasks(self, split: str, limit: int | None = None) -> list[TerminalBenchTask]:
        """Return tasks from one split without changing manifest order.

        Args:
            split: One of ``train``, ``val``, or ``test``.
            limit: Optional non-negative prefix length.

        Returns:
            Task records in the checked-in deterministic order.

        Raises:
            ValueError: The split or limit is invalid.
        """
        if split not in self.splits:
            raise ValueError(f"split must be one of {sorted(self.splits)}; got {split!r}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative; got {limit}")
        task_ids = self.splits[split] if limit is None else self.splits[split][:limit]
        return [TerminalBenchTask(task_id) for task_id in task_ids]


@dataclass(frozen=True)
class HarborTrialResult:
    """Parsed evidence for one Harbor trial."""

    task_id: str
    reward: float
    rewards: dict[str, float]
    errors: list[str]
    atif_trajectories: list[dict[str, Any]]
    raw_result: dict[str, Any]
    trial_dir: Path


@dataclass(frozen=True)
class HarborEvaluation:
    """One isolated Harbor job produced for a GEPA candidate evaluation."""

    evaluation_id: str
    candidate_digest: str
    config_path: Path
    job_dir: Path
    returncode: int
    stdout_path: Path
    stderr_path: Path
    trials: dict[str, HarborTrialResult]


class HarborRequirementError(RuntimeError):
    """Raised when the pinned Harbor CLI or a running Docker daemon is absent."""


class HarborExecutionError(RuntimeError):
    """Raised when a Harbor job fails before producing complete task results."""


def _validate_job_result(raw_result: Any, expected_trials: int, result_path: Path) -> None:
    """Require Harbor's completed-job counters to describe a clean run.

    Args:
        raw_result: Decoded Harbor job result.
        expected_trials: Exact requested trial count.
        result_path: Result path included in boundary errors.

    Raises:
        HarborExecutionError: The result shape, completion marker, or trial
            counters do not describe a clean finished job.
    """
    if not isinstance(raw_result, dict):
        raise HarborExecutionError(f"Harbor job result {result_path} is not a JSON object")
    stats = raw_result.get("stats")
    if not isinstance(stats, dict):
        raise HarborExecutionError(f"Harbor job result {result_path} has no stats object")

    counts = {
        name: stats.get(name)
        for name in (
            "n_completed_trials",
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
        )
    }
    if (
        raw_result.get("finished_at") is None
        or raw_result.get("n_total_trials") != expected_trials
        or counts["n_completed_trials"] != expected_trials
        or any(counts[name] != 0 for name in counts if name != "n_completed_trials")
    ):
        raise HarborExecutionError(
            f"Harbor job result {result_path} is not a clean completed job: "
            f"finished_at={raw_result.get('finished_at')!r}, "
            f"n_total_trials={raw_result.get('n_total_trials')!r}, stats={counts!r}"
        )


def _load_atif_trajectory(trajectory_path: Path) -> dict[str, Any]:
    """Load one ATIF document and require Harbor's core trajectory shape.

    Args:
        trajectory_path: Harbor trajectory JSON path.

    Returns:
        Validated ATIF trajectory object with ordered steps.

    Raises:
        HarborExecutionError: The file is unreadable, malformed, uses an
            unsupported schema, or lacks required agent and step fields.
    """
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} is unreadable or invalid JSON") from exc

    if not isinstance(trajectory, dict):
        raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} is not a JSON object")

    schema_version = trajectory.get("schema_version", "ATIF-v1.7")
    if schema_version not in SUPPORTED_ATIF_SCHEMA_VERSIONS:
        raise HarborExecutionError(
            f"Harbor ATIF trajectory {trajectory_path} has unsupported schema_version {schema_version!r}"
        )

    agent = trajectory.get("agent")
    if not isinstance(agent, dict):
        raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} has no agent object")
    for field in ("name", "version"):
        if not isinstance(agent.get(field), str):
            raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} has no string agent.{field}")

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} has no non-empty steps array")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} steps[{index}] is not an object")
        expected_step_id = index + 1
        if type(step.get("step_id")) is not int or step["step_id"] != expected_step_id:
            raise HarborExecutionError(
                f"Harbor ATIF trajectory {trajectory_path} steps[{index}].step_id must be {expected_step_id}"
            )
        if step.get("source") not in {"system", "user", "agent"}:
            raise HarborExecutionError(f"Harbor ATIF trajectory {trajectory_path} steps[{index}].source is invalid")
        message = step.get("message")
        if not isinstance(message, str | list):
            raise HarborExecutionError(
                f"Harbor ATIF trajectory {trajectory_path} steps[{index}].message must be text or content parts"
            )
        if isinstance(message, list) and not all(isinstance(part, dict) for part in message):
            raise HarborExecutionError(
                f"Harbor ATIF trajectory {trajectory_path} steps[{index}].message has invalid content parts"
            )

    return trajectory


def derive_terminalbench_splits(task_ids: Sequence[str], seed: str) -> dict[str, list[str]]:
    """Derive stable 40/30/30 splits with hash ordering and Hamilton allocation.

    Args:
        task_ids: Unique fully qualified task IDs.
        seed: Versioned text seed recorded in the manifest.

    Returns:
        ``train``, ``val``, and ``test`` lists in deterministic hash order.

    Raises:
        ValueError: Task IDs are empty or contain duplicates.
    """
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")

    ordered = [
        task_id
        for _, task_id in sorted(
            (hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest(), task_id) for task_id in task_ids
        )
    ]
    quotas = {name: len(ordered) * SPLIT_WEIGHTS[name] for name in SPLIT_NAMES}
    counts = {name: int(quotas[name]) for name in SPLIT_NAMES}
    unassigned = len(ordered) - sum(counts.values())
    remainder_order = [
        name
        for _, name in sorted((-(quotas[name] - counts[name]), name) for name in SPLIT_NAMES)
    ]
    for name in remainder_order[:unassigned]:
        counts[name] += 1

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def load_terminalbench_manifest(path: str | Path) -> TerminalBenchManifest:
    """Load and verify the pinned v3 manifest before any benchmark work begins.

    Args:
        path: JSON manifest generated from the official Harbor registry.

    Returns:
        A validated manifest.

    Raises:
        ValueError: Pin metadata, task refs, counts, or splits are inconsistent.
    """
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Terminal-Bench manifest schema_version must be 1")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Terminal-Bench manifest must contain a dataset object")
    expected_pins = {
        "identifier": PINNED_DATASET_IDENTIFIER,
        "version": PINNED_DATASET_VERSION,
        "reference": PINNED_DATASET_REFERENCE,
        "registry_content_hash": PINNED_DATASET_CONTENT_HASH,
        "source_repository": PINNED_SOURCE_REPOSITORY,
        "source_tag": PINNED_SOURCE_TAG,
        "source_commit": PINNED_SOURCE_COMMIT,
        "task_count": PINNED_TASK_COUNT,
        "harbor_version": PINNED_HARBOR_VERSION,
    }
    for field, expected in expected_pins.items():
        if dataset.get(field) != expected:
            raise ValueError(f"manifest dataset.{field} must be {expected!r}; got {dataset.get(field)!r}")
    task_refs = payload.get("task_refs")
    if not isinstance(task_refs, dict) or not task_refs:
        raise ValueError("Terminal-Bench manifest task_refs must be a non-empty object")
    normalized_refs: dict[str, str] = {}
    for task_id, ref in task_refs.items():
        if not isinstance(task_id, str) or not task_id.startswith("terminal-bench/"):
            raise ValueError(f"invalid Terminal-Bench task ID: {task_id!r}")
        if not isinstance(ref, str) or not ref.startswith("sha256:"):
            raise ValueError(f"task {task_id!r} must have a sha256 registry ref")
        normalized_refs[task_id] = ref
    if dataset.get("task_count") != len(normalized_refs):
        raise ValueError(
            f"manifest task_count is {dataset.get('task_count')!r}, but task_refs contains {len(normalized_refs)} tasks"
        )

    split_policy = payload.get("split_policy")
    if not isinstance(split_policy, dict) or not isinstance(split_policy.get("seed"), str):
        raise ValueError("Terminal-Bench manifest split_policy.seed must be a string")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Terminal-Bench manifest splits must be exactly {list(SPLIT_NAMES)}")
    normalized_splits: dict[str, list[str]] = {}
    for split in SPLIT_NAMES:
        values = splits[split]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"manifest split {split!r} must be a list of task IDs")
        normalized_splits[split] = values

    flattened = [task_id for split in SPLIT_NAMES for task_id in normalized_splits[split]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Terminal-Bench manifest splits overlap")
    if set(flattened) != set(normalized_refs):
        raise ValueError("Terminal-Bench manifest splits must contain every task ref exactly once")
    expected_splits = derive_terminalbench_splits(list(normalized_refs), split_policy["seed"])
    if normalized_splits != expected_splits:
        raise ValueError("Terminal-Bench manifest splits do not match the recorded deterministic split policy")
    expected_counts = {name: len(expected_splits[name]) for name in SPLIT_NAMES}
    if split_policy.get("counts") != expected_counts:
        raise ValueError(f"manifest split counts must be {expected_counts}; got {split_policy.get('counts')!r}")

    return TerminalBenchManifest(
        path=manifest_path,
        dataset=dataset,
        split_policy=split_policy,
        task_refs=normalized_refs,
        splits=normalized_splits,
    )


def render_terminus_prompt(candidate_prompt: str) -> str:
    """Combine an evolvable instruction prefix with the fixed Terminus contract.

    Args:
        candidate_prompt: GEPA's current ``instruction_prompt`` component.

    Returns:
        A complete Terminus template ready for Harbor's later ``str.format``.
        When ``candidate_prompt`` is blank, only the fixed output contract
        remains.
    """
    if not candidate_prompt.strip():
        return TERMINUS_JSON_CONTRACT
    escaped_candidate = candidate_prompt.replace("{", "{{").replace("}", "}}")
    return f"{escaped_candidate.rstrip()}\n\n{TERMINUS_JSON_CONTRACT}"


class HarborCLI:
    """Create isolated candidate jobs and execute them with pinned Harbor.

    Args:
        student_model: Model used by Terminus to solve benchmark tasks.
        work_dir: Root for immutable candidate prompt/config/job artifacts.
        agent_python_path: Directory added to ``PYTHONPATH`` so Harbor can load
            the checked-in ``PromptedTerminus`` wrapper.
        n_concurrent: Maximum trials Harbor may run concurrently.
        harbor_executable: Harbor CLI name or path.
        docker_executable: Docker CLI name or path used for readiness checks.
        student_api_base: Optional LiteLLM API base for the student model.
        student_agent_kwargs: Extra Terminus kwargs that do not alter the fixed
            prompt, tmux tool, skill policy, or unbounded-turn default.
        process_timeout_sec: Optional whole-job subprocess timeout. ``None``
            leaves long-horizon completion governed by each pinned task's
            Harbor agent/verifier timeouts.
    """

    def __init__(
        self,
        *,
        student_model: str,
        work_dir: str | Path,
        agent_python_path: str | Path,
        n_concurrent: int = 1,
        harbor_executable: str = "harbor",
        docker_executable: str = "docker",
        student_api_base: str | None = None,
        student_agent_kwargs: Mapping[str, Any] | None = None,
        process_timeout_sec: float | None = None,
    ) -> None:
        """Configure the pinned Harbor subprocess boundary.

        Args:
            student_model: Model used by Terminus to solve benchmark tasks.
            work_dir: Root for candidate prompt, config, and job artifacts.
            agent_python_path: Directory added to ``PYTHONPATH`` for the
                checked-in Terminus wrapper.
            n_concurrent: Maximum trials Harbor may run concurrently.
            harbor_executable: Harbor CLI name or explicit path.
            docker_executable: Docker CLI name or explicit path.
            student_api_base: Optional LiteLLM endpoint for the student model.
            student_agent_kwargs: Additional Terminus settings that do not
                override fixed harness behavior.
            process_timeout_sec: Optional whole-job subprocess timeout.

        Raises:
            ValueError: Model or numeric settings are invalid, or extra agent
                settings attempt to override fixed harness keys.
        """
        if not student_model.strip():
            raise ValueError("student_model must not be empty")
        if n_concurrent < 1:
            raise ValueError(f"n_concurrent must be at least 1; got {n_concurrent}")
        if process_timeout_sec is not None and process_timeout_sec <= 0:
            raise ValueError("process_timeout_sec must be positive when provided")
        extra_kwargs = dict(student_agent_kwargs or {})
        fixed_keys = {
            "disable_skills",
            "max_episodes",
            "max_turns",
            "mcp_servers",
            "parser_name",
            "prompt_template_path",
            "record_terminal_session",
            "skills_dir",
            "store_all_messages",
            "tmux_pane_height",
            "tmux_pane_width",
            "trajectory_config",
        }
        overridden = fixed_keys.intersection(extra_kwargs)
        if overridden:
            raise ValueError(f"student_agent_kwargs cannot override fixed harness keys: {sorted(overridden)}")

        self.student_model = student_model
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.agent_python_path = Path(agent_python_path).expanduser().resolve()
        self.n_concurrent = n_concurrent
        self.harbor_executable = harbor_executable
        self.docker_executable = docker_executable
        self.student_api_base = student_api_base
        self.student_agent_kwargs = extra_kwargs
        self.process_timeout_sec = process_timeout_sec

    @staticmethod
    def _resolve_executable(executable: str, label: str) -> str:
        """Resolve one required CLI to an executable path.

        Args:
            executable: CLI name or explicit path.
            label: Human-readable dependency name for the error.

        Returns:
            The executable path.

        Raises:
            HarborRequirementError: The CLI cannot be found.
        """
        resolved = shutil.which(executable)
        if resolved is None:
            raise HarborRequirementError(f"{label} executable {executable!r} was not found on PATH")
        return resolved

    def check_requirements(self) -> tuple[str, str]:
        """Require Harbor 0.22.0 and a reachable Docker daemon.

        Returns:
            Resolved Harbor and Docker executable paths.

        Raises:
            HarborRequirementError: A CLI is missing, Harbor is the wrong
                version, or Docker cannot reach its daemon.
        """
        harbor = self._resolve_executable(self.harbor_executable, "Harbor")
        docker = self._resolve_executable(self.docker_executable, "Docker")
        harbor_version = subprocess.run(
            [harbor, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if harbor_version.returncode != 0:
            raise HarborRequirementError(
                f"failed to query Harbor version: {harbor_version.stderr.strip() or harbor_version.stdout.strip()}"
            )
        actual_version = harbor_version.stdout.strip()
        if actual_version != PINNED_HARBOR_VERSION:
            raise HarborRequirementError(
                f"Harbor {PINNED_HARBOR_VERSION} is required; found {actual_version!r}. "
                f"Install it with `uv tool install --force harbor=={PINNED_HARBOR_VERSION}`."
            )

        docker_info = subprocess.run(
            [docker, "info", "--format", "{{json .ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if docker_info.returncode != 0:
            detail = docker_info.stderr.strip() or docker_info.stdout.strip()
            raise HarborRequirementError(f"Docker is installed but its daemon is unavailable: {detail}")
        return harbor, docker

    def build_job_config(
        self,
        task_ids: Sequence[str],
        *,
        prompt_path: Path,
        jobs_dir: Path,
        job_name: str,
    ) -> dict[str, Any]:
        """Build the exact Harbor job for one candidate/batch evaluation.

        Args:
            task_ids: Fully qualified pinned task IDs.
            prompt_path: Candidate-specific rendered Terminus template.
            jobs_dir: Candidate-specific Harbor jobs directory.
            job_name: Unique job name inside ``jobs_dir``.

        Returns:
            JSON-serializable Harbor v0.22.0 job configuration.
        """
        agent_kwargs: dict[str, Any] = {
            "disable_skills": True,
            "prompt_template_path": str(prompt_path),
            "record_terminal_session": True,
            "store_all_messages": True,
            "trajectory_config": {"linear_history": False},
            **self.student_agent_kwargs,
        }
        if self.student_api_base is not None:
            agent_kwargs["api_base"] = self.student_api_base
        return {
            "job_name": job_name,
            "jobs_dir": str(jobs_dir),
            "n_attempts": 1,
            "timeout_multiplier": 1.0,
            "n_concurrent_trials": self.n_concurrent,
            "quiet": True,
            "environment": {"type": "docker", "force_build": False, "delete": True},
            "agents": [
                {
                    "import_path": PROMPTED_TERMINUS_IMPORT_PATH,
                    "model_name": self.student_model,
                    "skills": [],
                    "kwargs": agent_kwargs,
                }
            ],
            "datasets": [
                {
                    "name": PINNED_DATASET_IDENTIFIER,
                    "ref": PINNED_DATASET_CONTENT_HASH,
                    "task_names": list(task_ids),
                }
            ],
        }

    def run(self, task_ids: Sequence[str], candidate_prompt: str) -> HarborEvaluation:
        """Run one isolated Harbor job and parse every task by exact ID.

        Args:
            task_ids: Unique fully qualified task IDs in desired output order.
            candidate_prompt: Current GEPA instruction prompt.

        Returns:
            Evaluation metadata and a task-ID keyed trial map.

        Raises:
            HarborExecutionError: Harbor times out or fails, the job summary is
                missing or invalid, or parsed trial evidence violates the
                benchmark contract.
            HarborRequirementError: Harbor or Docker is unavailable.
            ValueError: Task IDs are empty or duplicated.
            json.JSONDecodeError: A per-trial result is not valid JSON.
            OSError: A per-trial result cannot be read.
            UnicodeDecodeError: A per-trial result is not valid UTF-8 text.
        """
        if not task_ids:
            raise ValueError("task_ids must not be empty")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be unique within one Harbor job")
        harbor, _docker = self.check_requirements()

        candidate_digest = hashlib.sha256(candidate_prompt.encode()).hexdigest()
        evaluation_id = f"{candidate_digest[:12]}-{uuid.uuid4().hex}"
        evaluation_dir = self.work_dir / "evaluations" / evaluation_id
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        prompt_path = evaluation_dir / "terminus-prompt.txt"
        prompt_path.write_text(render_terminus_prompt(candidate_prompt))
        jobs_dir = evaluation_dir / "jobs"
        job_name = f"candidate-{candidate_digest[:12]}"
        config = self.build_job_config(
            task_ids,
            prompt_path=prompt_path,
            jobs_dir=jobs_dir,
            job_name=job_name,
        )
        config_path = evaluation_dir / "harbor-job.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_parts = [str(self.agent_python_path)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        command = [harbor, "run", "--config", str(config_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=self.agent_python_path,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.process_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarborExecutionError(
                f"Harbor evaluation {evaluation_id} exceeded the configured whole-job timeout"
            ) from exc

        stdout_path = evaluation_dir / "harbor.stdout.log"
        stderr_path = evaluation_dir / "harbor.stderr.log"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        job_dir = jobs_dir / job_name
        job_result_path = job_dir / "result.json"
        if not job_result_path.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise HarborExecutionError(
                f"Harbor evaluation {evaluation_id} returned {completed.returncode} without a job result: {detail}"
            )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise HarborExecutionError(
                f"Harbor evaluation {evaluation_id} exited with status {completed.returncode}: {detail}"
            )
        try:
            raw_job_result = json.loads(job_result_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HarborExecutionError(f"Harbor job result {job_result_path} is unreadable") from exc
        _validate_job_result(raw_job_result, len(task_ids), job_result_path)

        trials: dict[str, HarborTrialResult] = {}
        for result_path in sorted(job_dir.glob("*/result.json")):
            raw_result = json.loads(result_path.read_text())
            task_id = raw_result.get("task_name")
            if not isinstance(task_id, str):
                raise HarborExecutionError(f"Harbor trial result {result_path} has no string task_name")
            if task_id in trials:
                raise HarborExecutionError(f"Harbor produced duplicate results for task {task_id!r}")

            verifier_result = raw_result.get("verifier_result")
            raw_rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
            if not isinstance(raw_rewards, dict) or "reward" not in raw_rewards:
                raise HarborExecutionError(
                    f"Harbor trial {task_id!r} did not return the canonical verifier reward in {result_path}"
                )
            try:
                rewards = {name: float(value) for name, value in raw_rewards.items()}
            except (TypeError, ValueError) as exc:
                raise HarborExecutionError(f"Harbor trial {task_id!r} returned non-numeric rewards") from exc
            errors: list[str] = []
            exception_info = raw_result.get("exception_info")
            if isinstance(exception_info, dict):
                exception_type = exception_info.get("exception_type", "Exception")
                exception_message = exception_info.get("exception_message", "")
                errors.append(f"{exception_type}: {exception_message}".rstrip())
            step_results = raw_result.get("step_results")
            if isinstance(step_results, list):
                for step in step_results:
                    if not isinstance(step, dict) or not isinstance(step.get("exception_info"), dict):
                        continue
                    step_exception = step["exception_info"]
                    step_name = step.get("step_name", "unknown step")
                    exception_type = step_exception.get("exception_type", "Exception")
                    exception_message = step_exception.get("exception_message", "")
                    errors.append(f"{step_name}: {exception_type}: {exception_message}".rstrip())
            if errors:
                raise HarborExecutionError(f"Harbor trial {task_id!r} reported execution errors: {'; '.join(errors)}")
            agent_dir = result_path.parent / "agent"
            atif_trajectories: list[dict[str, Any]] = []
            for trajectory_path in sorted(agent_dir.glob("trajectory*.json")):
                atif_trajectories.append(_load_atif_trajectory(trajectory_path))
            if not atif_trajectories:
                raise HarborExecutionError(f"Harbor trial {task_id!r} did not emit an ATIF trajectory")

            trials[task_id] = HarborTrialResult(
                task_id=task_id,
                reward=rewards["reward"],
                rewards=rewards,
                errors=errors,
                atif_trajectories=atif_trajectories,
                raw_result=raw_result,
                trial_dir=result_path.parent,
            )

        missing = [task_id for task_id in task_ids if task_id not in trials]
        unexpected = sorted(set(trials).difference(task_ids))
        if missing or unexpected:
            raise HarborExecutionError(
                f"Harbor result/task mismatch for evaluation {evaluation_id}: missing={missing}, unexpected={unexpected}"
            )
        return HarborEvaluation(
            evaluation_id=evaluation_id,
            candidate_digest=candidate_digest,
            config_path=config_path,
            job_dir=job_dir,
            returncode=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            trials=trials,
        )


class TerminalBenchAdapter(GEPAAdapter[TerminalBenchTask, TerminalBenchTrajectory, TerminalBenchOutput]):
    """Evaluate one prompt component with Terminus and official Harbor rewards.

    Args:
        manifest: Checked-in, validated v3 manifest.
        harbor: Pinned Harbor subprocess runner configured with the student
            model. The proposer model is supplied separately to ``gepa.optimize``.
    """

    def __init__(self, manifest: TerminalBenchManifest, harbor: HarborCLI) -> None:
        """Bind the validated manifest to its Harbor runner.

        Args:
            manifest: Checked-in, validated Terminal-Bench v3 manifest.
            harbor: Pinned runner configured with the student model.
        """
        self.manifest = manifest
        self.harbor = harbor

    def evaluate(
        self,
        batch: list[TerminalBenchTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TerminalBenchTrajectory, TerminalBenchOutput]:
        """Run the candidate on a batch and preserve its incoming task order.

        Args:
            batch: Pinned task records.
            candidate: Exactly one ``instruction_prompt`` component.
            capture_traces: Whether to return full ATIF/result evidence to GEPA.

        Returns:
            Harbor rewards, outputs, and optional complete trajectories.

        Raises:
            ValueError: Candidate components or task IDs violate the harness
                contract.
        """
        if set(candidate) != {INSTRUCTION_COMPONENT}:
            raise ValueError(f"TerminalBenchAdapter optimizes only {INSTRUCTION_COMPONENT!r}; got {sorted(candidate)}")
        task_ids = [task.task_id for task in batch]
        unknown = sorted(set(task_ids).difference(self.manifest.task_refs))
        if unknown:
            raise ValueError(f"tasks are not in pinned {PINNED_DATASET_REFERENCE}: {unknown}")
        evaluation = self.harbor.run(task_ids, candidate[INSTRUCTION_COMPONENT])

        outputs: list[TerminalBenchOutput] = []
        scores: list[float] = []
        trajectories: list[TerminalBenchTrajectory] | None = [] if capture_traces else None
        for task_id in task_ids:
            trial = evaluation.trials[task_id]
            errors = list(trial.errors)
            output: TerminalBenchOutput = {
                "task_id": task_id,
                "reward": trial.reward,
                "rewards": trial.rewards,
                "errors": errors,
                "evaluation_id": evaluation.evaluation_id,
                "harbor_returncode": evaluation.returncode,
                "harbor_stdout_path": str(evaluation.stdout_path),
                "harbor_stderr_path": str(evaluation.stderr_path),
                "trial_dir": str(trial.trial_dir),
            }
            outputs.append(output)
            scores.append(trial.reward)
            if trajectories is not None:
                trajectories.append(
                    {
                        "task_id": task_id,
                        "candidate_prompt": candidate[INSTRUCTION_COMPONENT],
                        "reward": trial.reward,
                        "rewards": trial.rewards,
                        "errors": errors,
                        "atif_trajectories": trial.atif_trajectories,
                        "trial_result": trial.raw_result,
                        "evaluation_id": evaluation.evaluation_id,
                        "harbor_returncode": evaluation.returncode,
                        "harbor_stdout_path": str(evaluation.stdout_path),
                        "harbor_stderr_path": str(evaluation.stderr_path),
                        "trial_dir": str(trial.trial_dir),
                    }
                )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TerminalBenchTrajectory, TerminalBenchOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Expose complete ATIF trajectories, rewards, and errors for reflection.

        Args:
            candidate: Candidate used for the captured evaluation.
            eval_batch: Evaluation batch returned with ``capture_traces=True``.
            components_to_update: Components requested by GEPA.

        Returns:
            Reflection rows for the single optimized instruction component.

        Raises:
            ValueError: A different component is requested or the candidate
                does not contain exactly the optimized instruction component.
            RuntimeError: The evaluation omitted trajectories.
        """
        if components_to_update != [INSTRUCTION_COMPONENT]:
            raise ValueError(
                f"TerminalBenchAdapter can update only [{INSTRUCTION_COMPONENT!r}]; got {components_to_update!r}"
            )
        if set(candidate) != {INSTRUCTION_COMPONENT}:
            raise ValueError(f"TerminalBenchAdapter optimizes only {INSTRUCTION_COMPONENT!r}; got {sorted(candidate)}")
        if eval_batch.trajectories is None:
            raise RuntimeError("Terminal-Bench reflection requires capture_traces=True")

        rows: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories:
            rows.append(
                {
                    "Inputs": {
                        "dataset": PINNED_DATASET_REFERENCE,
                        "task_id": trajectory["task_id"],
                    },
                    "Generated Outputs": {
                        "atif_trajectories": trajectory["atif_trajectories"],
                        "trial_result": trajectory["trial_result"],
                        "harbor_process": {
                            "returncode": trajectory["harbor_returncode"],
                            "stdout_path": trajectory["harbor_stdout_path"],
                            "stderr_path": trajectory["harbor_stderr_path"],
                        },
                    },
                    "Feedback": json.dumps(
                        {
                            "reward": trajectory["reward"],
                            "rewards": trajectory["rewards"],
                            "errors": trajectory["errors"],
                        },
                        sort_keys=True,
                    ),
                }
            )
        return {INSTRUCTION_COMPONENT: rows}


TerminusAdapter = TerminalBenchAdapter
