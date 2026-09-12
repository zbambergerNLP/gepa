"""Port GEPA's TerminusAdapter to Terminal-Bench 2.1 and the pinned Harbor CLI.

Harbor runs in a separate Python environment through a subprocess, so GEPA
retains Python 3.10+ support. Harbor supplies the official Docker verifier and
ATIF trajectories. This maintained port replaces the upstream legacy ``tb run``
transport and single-prompt feedback; see ``TERMINUS_ADAPTER_CONTRACT`` for its
upstream source and the README for the compatibility differences.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, TypedDict

from gepa.adapters.terminal_bench_adapter.documents import (
    COMPONENT_KINDS,
    render_instruction,
    validate_documents,
    write_document_bundle,
)
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.strategies.text_limits import TextLimits, clip_text, resolve_text_limits, validate_char_limit

PINNED_HARBOR_VERSION = "0.22.0"
TASK_CONTEXT_SETTINGS = {
    "enable_summarize": True,
    "proactive_summarization_threshold": 8_000,
}
EXPERIMENT_DATASETS = {
    "tb2.1": {
        "identifier": "terminal-bench/terminal-bench-2-1",
        "version": "2.1",
        "reference": "terminal-bench/terminal-bench-2-1",
        "registry_content_hash": "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a",
        "source_repository": "https://github.com/harbor-framework/terminal-bench-2-1",
        "source_tag": None,
        "source_commit": None,
        "task_count": 89,
        "task_refs_digest": "c3ff7071ba153cac0c12523235ddee0e6ce942777748c260aebc9cbaf6c0c1ed",
        "registry_version_id": "f92eea12-ff70-4d30-ace0-003abf294998",
        "harbor_version": PINNED_HARBOR_VERSION,
    },
}
EXPERIMENT_SPLIT_COUNTS = {
    "tb2.1": {"train": 30, "val": 19, "test": 40},
}
PROMPTED_TERMINUS_IMPORT_PATH = "examples.terminalbench.terminus_agent:PromptedTerminus"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_WEIGHTS = {"train": 0.40, "val": 0.30, "test": 0.30}
SUPPORTED_ATIF_SCHEMA_VERSIONS = {f"ATIF-v1.{minor}" for minor in range(8)}
VERIFIER_LOG_FILENAMES = ("test-stdout.txt", "test-stderr.txt")
REFLECTION_FEEDBACK_CONTRACT = {
    "version": 5,
    "score": "official_verifier_reward",
    "reflection_split": "train",
    "trajectory_directories": ["agent", "steps/*/agent"],
    "trajectory_projection": "complete_atif_without_deduplication",
    "raw_trial_and_process_metadata": "included_in_reflection",
    "verifier_log_filenames": list(VERIFIER_LOG_FILENAMES),
    "verifier_log_directories": ["verifier", "steps/*/verifier"],
    "max_chars_per_verifier_log": None,
    "log_truncation": "optional_equal_head_and_tail_with_omitted_character_marker",
    "log_decoding": "utf-8-replace",
    "missing_verifier_logs": "explicitly_unavailable",
}
FAILURE_POLICY_CONTRACT = {
    "version": 1,
    "accepted_trial_exceptions": ["AgentTimeoutError"],
    "agent_timeout_score": "official_verifier_reward",
    "timed_out_step_requires_verifier_rewards": True,
    "harbor_max_retries": 0,
    "infrastructure_or_evidence_failure": "raise_without_score",
    "recovery": "explicit_resume_after_repair",
    "failed_job_usage": "preserved_in_harbor_artifacts_separate_from_scored_evaluations",
}


class TerminalBenchOutput(TypedDict):
    """Opaque per-task output retained by GEPA."""

    task_id: str
    reward: float
    rewards: dict[str, float]
    errors: list[str]
    evaluation_id: str
    candidate_digest: str
    job_dir: str
    config_path: str
    harbor_returncode: int
    harbor_stdout_path: str
    harbor_stderr_path: str
    trial_dir: str


class TerminalBenchTrajectory(TypedDict):
    """Complete Harbor evidence used to construct reflection records."""

    task_id: str
    candidate_documents: dict[str, str]
    reward: float
    rewards: dict[str, float]
    errors: list[str]
    atif_trajectories: list[dict[str, Any]]
    verifier_logs: dict[str, str]
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
        task_id: Qualified Hub task name such as ``terminal-bench/bn-fit-modify``.
    """

    task_id: str


@dataclass(frozen=True)
class TerminalBenchManifest:
    """Validated task refs and deterministic train/validation/test splits."""

    path: Path
    experiment: str
    dataset: dict[str, Any]
    split_policy: dict[str, Any]
    task_refs: dict[str, str]
    splits: dict[str, list[str]]

    @property
    def component_kinds(self) -> dict[str, str]:
        """Return the full text and skill surface shared by all methods."""
        return dict(COMPONENT_KINDS)

    def validate_candidate(self, candidate: Mapping[str, str]) -> None:
        """Reject candidates from a different optimization target.

        Args:
            candidate: Documents proposed for this experiment.

        Raises:
            ValueError: The component set or value types differ from the target.
        """
        validate_documents(candidate)

    def candidate_digest(self, candidate: Mapping[str, str]) -> str:
        """Hash the experiment identity together with its candidate text.

        Args:
            candidate: Complete candidate for this experiment.

        Returns:
            Stable SHA-256 digest independent of component insertion order.
        """
        self.validate_candidate(candidate)
        payload = {"experiment": self.experiment, "documents": dict(candidate)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

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
    verifier_logs: dict[str, str] = dataclass_field(default_factory=dict)


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


def _read_verifier_logs(trial_dir: Path, max_chars: int | None = None) -> dict[str, str]:
    """Read complete verifier logs unless a character cutoff is configured.

    Args:
        trial_dir: One completed Harbor trial's artifact directory.
        max_chars: Source-character allowance per log, or ``None`` for unlimited.

    Returns:
        Relative log paths mapped to UTF-8 text. Oversized files retain their
        beginning and end with an explicit omitted-character count between them.

    Raises:
        HarborExecutionError: A present log is unreadable or resolves outside
            this trial's directory.
    """
    validate_char_limit("verifier_log_chars", max_chars)
    logs = {}
    directories = [trial_dir / "verifier", *sorted((trial_dir / "steps").glob("*/verifier"))]
    for directory in directories:
        for filename in VERIFIER_LOG_FILENAMES:
            path = directory / filename
            if not path.resolve().is_relative_to(trial_dir.resolve()):
                raise HarborExecutionError(f"Verifier log {path} resolves outside its trial directory")
            try:
                text = clip_text(path.read_bytes().decode("utf-8", errors="replace"), max_chars, head_and_tail=True)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise HarborExecutionError(f"Verifier log {path} is unreadable") from exc
            logs[path.relative_to(trial_dir).as_posix()] = text
    return logs


def _validate_job_result(raw_result: Any, expected_trials: int, result_path: Path, *, verified_timeouts: int) -> None:
    """Reconcile Harbor's completed-job counters with verified timeout trials.

    Args:
        raw_result: Decoded Harbor job result.
        expected_trials: Exact requested trial count.
        result_path: Result path included in boundary errors.
        verified_timeouts: Trial-level timeouts already validated against
            official verifier results. Harbor counts these as both completed
            and errored; step-level exceptions do not increment that counter.

    Raises:
        HarborExecutionError: The result shape, completion marker, or trial
            counters do not describe a complete, single-attempt evaluation.
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
            "n_retries",
        )
    }
    if (
        raw_result.get("finished_at") is None
        or raw_result.get("n_total_trials") != expected_trials
        or counts["n_completed_trials"] != expected_trials
        or counts["n_errored_trials"] != verified_timeouts
        or any(counts[name] != 0 for name in counts if name not in {"n_completed_trials", "n_errored_trials"})
    ):
        raise HarborExecutionError(
            f"Harbor job result {result_path} is not a complete single-attempt job: "
            f"finished_at={raw_result.get('finished_at')!r}, "
            f"n_total_trials={raw_result.get('n_total_trials')!r}, stats={counts!r}"
        )


def _read_verifier_rewards(
    raw_result: Mapping[str, Any], task_id: str, *, require_canonical: bool = True
) -> dict[str, float]:
    """Require finite official rewards before accepting a trial or timed-out step.

    Args:
        raw_result: Trial or step result containing verifier evidence.
        task_id: Task identity, including step name when applicable.
        require_canonical: Require the overall trial's canonical score key.

    Returns:
        Official verifier rewards converted to finite floats.

    Raises:
        HarborExecutionError: Verification is missing or its rewards are invalid.
    """
    verifier_result = raw_result.get("verifier_result")
    raw_rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    if not isinstance(raw_rewards, dict) or not raw_rewards or (require_canonical and "reward" not in raw_rewards):
        required = "canonical verifier reward" if require_canonical else "verifier rewards"
        raise HarborExecutionError(f"Harbor trial {task_id!r} did not return the required {required}")
    try:
        rewards = {name: float(value) for name, value in raw_rewards.items()}
    except (TypeError, ValueError, OverflowError) as exc:
        raise HarborExecutionError(f"Harbor trial {task_id!r} returned non-numeric rewards") from exc
    if not all(math.isfinite(value) for value in rewards.values()):
        raise HarborExecutionError(f"Harbor trial {task_id!r} returned non-finite verifier rewards")
    return rewards


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


def derive_terminalbench_splits(
    task_ids: Sequence[str], seed: str, counts: Mapping[str, int] | None = None
) -> dict[str, list[str]]:
    """Derive stable splits with hash ordering and explicit or 40/30/30 counts.

    Args:
        task_ids: Unique task IDs; hash names without the terminal-bench namespace
            to preserve the approved assignments when moving to Hub packages.
        seed: Versioned text seed recorded in the manifest.
        counts: Explicit split sizes; omitted uses Hamilton 40/30/30 allocation.

    Returns:
        ``train``, ``val``, and ``test`` lists in deterministic hash order.

    Raises:
        ValueError: Task IDs are empty or duplicated, or counts are invalid.
    """
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")

    ordered = [
        task_id
        for _, task_id in sorted(
            (hashlib.sha256(f"{seed}\0{task_id.removeprefix('terminal-bench/')}".encode()).hexdigest(), task_id)
            for task_id in task_ids
        )
    ]
    if counts is None:
        quotas = {name: len(ordered) * SPLIT_WEIGHTS[name] for name in SPLIT_NAMES}
        counts = {name: int(quotas[name]) for name in SPLIT_NAMES}
        unassigned = len(ordered) - sum(counts.values())
        remainder_order = [name for _, name in sorted((-(quotas[name] - counts[name]), name) for name in SPLIT_NAMES)]
        for name in remainder_order[:unassigned]:
            counts[name] += 1
    elif (
        set(counts) != set(SPLIT_NAMES)
        or any(type(count) is not int or count < 0 for count in counts.values())
        or sum(counts.values()) != len(ordered)
    ):
        raise ValueError("split counts must be non-negative integers covering every task exactly once")

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def load_terminalbench_manifest(path: str | Path) -> TerminalBenchManifest:
    """Load and verify the pinned TB2.1 experiment before benchmark work begins.

    Args:
        path: JSON manifest generated from the official Harbor registry.

    Returns:
        A validated manifest.

    Raises:
        ValueError: Pin metadata, task refs, counts, or splits are inconsistent.
    """
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError("Terminal-Bench manifest schema_version must be 2")
    experiment = payload.get("experiment")
    if experiment not in EXPERIMENT_DATASETS:
        raise ValueError(f"Unknown Terminal-Bench experiment: {experiment!r}")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Terminal-Bench manifest must contain a dataset object")
    expected_pins = EXPERIMENT_DATASETS[experiment]
    for field, expected in expected_pins.items():
        if dataset.get(field) != expected:
            raise ValueError(f"manifest dataset.{field} must be {expected!r}; got {dataset.get(field)!r}")
    task_refs = payload.get("task_refs")
    if not isinstance(task_refs, dict) or not task_refs:
        raise ValueError("Terminal-Bench manifest task_refs must be a non-empty object")
    normalized_refs: dict[str, str] = {}
    for task_id, ref in task_refs.items():
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"invalid Terminal-Bench task ID: {task_id!r}")
        if not isinstance(ref, str):
            raise ValueError(f"task {task_id!r} must have an immutable source ref")
        normalized_refs[task_id] = ref
    if dataset.get("task_count") != len(normalized_refs):
        raise ValueError(
            f"manifest task_count is {dataset.get('task_count')!r}, but task_refs contains {len(normalized_refs)} tasks"
        )
    refs_digest = hashlib.sha256(json.dumps(normalized_refs, sort_keys=True).encode()).hexdigest()
    if refs_digest != dataset["task_refs_digest"]:
        raise ValueError("manifest task refs differ from the pinned official task set")

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
    expected_splits = derive_terminalbench_splits(
        list(normalized_refs), split_policy["seed"], EXPERIMENT_SPLIT_COUNTS[experiment]
    )
    if normalized_splits != expected_splits:
        raise ValueError("Terminal-Bench manifest splits do not match the recorded deterministic split policy")
    expected_counts = {name: len(expected_splits[name]) for name in SPLIT_NAMES}
    if split_policy.get("counts") != expected_counts:
        raise ValueError(f"manifest split counts must be {expected_counts}; got {split_policy.get('counts')!r}")

    return TerminalBenchManifest(
        path=manifest_path,
        experiment=experiment,
        dataset=dataset,
        split_policy=split_policy,
        task_refs=normalized_refs,
        splits=normalized_splits,
    )


def render_terminus_prompt(candidate: Mapping[str, str]) -> str:
    """Render main-agent guidance and command instructions above runtime inputs.

    Args:
        candidate: Complete candidate bundle.

    Returns:
        Terminus template with task and terminal-state placeholders.
    """
    return render_instruction(candidate)


class HarborCLI:
    """Run pinned experiment candidates in isolated Harbor jobs.

    Args:
        manifest: Validated experiment whose dataset and target this runner uses.
        student_model: Model used by Terminus to solve benchmark tasks.
        work_dir: Root for immutable document-bundle/config/job artifacts.
        agent_python_path: Directory added to ``PYTHONPATH`` so Harbor can load
            the checked-in ``PromptedTerminus`` wrapper.
        n_concurrent: Maximum trials Harbor may run concurrently.
        harbor_executable: Harbor CLI name or path.
        docker_executable: Docker CLI name or path used for readiness checks.
        student_api_base: Optional LiteLLM API base for the student model.
        student_agent_kwargs: Extra Terminus kwargs that do not alter the fixed
            documents, tmux tool, skill loading, context management, or
            unbounded-turn default.
        process_timeout_sec: Optional whole-job subprocess timeout. ``None``
            leaves long-horizon completion governed by each pinned task's
            Harbor agent/verifier timeouts.
    """

    def __init__(
        self,
        *,
        manifest: TerminalBenchManifest,
        student_model: str,
        work_dir: str | Path,
        agent_python_path: str | Path,
        n_concurrent: int = 1,
        harbor_executable: str = "harbor",
        docker_executable: str = "docker",
        student_api_base: str | None = None,
        student_agent_kwargs: Mapping[str, Any] | None = None,
        process_timeout_sec: float | None = None,
        text_limits: TextLimits | None = None,
    ) -> None:
        """Configure the pinned Harbor subprocess boundary.

        Args:
            manifest: Validated experiment whose dataset and target this runner uses.
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
            text_limits: Optional verifier-log character allowance.

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
            *TASK_CONTEXT_SETTINGS,
            "disable_skills",
            "max_episodes",
            "max_turns",
            "mcp_servers",
            "parser_name",
            "prompt_template_path",
            "document_bundle_path",
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

        self.manifest = manifest
        self.student_model = student_model
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.agent_python_path = Path(agent_python_path).expanduser().resolve()
        self.n_concurrent = n_concurrent
        self.harbor_executable = harbor_executable
        self.docker_executable = docker_executable
        self.student_api_base = student_api_base
        self.student_agent_kwargs = extra_kwargs
        self.process_timeout_sec = process_timeout_sec
        self.text_limits = resolve_text_limits(text_limits)

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
        bundle_path: Path | None,
        jobs_dir: Path,
        job_name: str,
    ) -> dict[str, Any]:
        """Build the exact Harbor job for one candidate/batch evaluation.

        Args:
            task_ids: Pinned task IDs using this dataset's native naming.
            prompt_path: Candidate-specific rendered Terminus template.
            bundle_path: Complete prompts and skills for the full-text experiment.
            jobs_dir: Candidate-specific Harbor jobs directory.
            job_name: Unique job name inside ``jobs_dir``.

        Returns:
            JSON-serializable Harbor v0.22.0 job configuration.
        """
        agent_kwargs: dict[str, Any] = {
            "prompt_template_path": str(prompt_path),
            "record_terminal_session": True,
            "store_all_messages": True,
            "trajectory_config": {"linear_history": False},
            **TASK_CONTEXT_SETTINGS,
            **self.student_agent_kwargs,
        }
        if self.student_api_base is not None:
            agent_kwargs["api_base"] = self.student_api_base
        if bundle_path is None:
            raise ValueError(f"{self.manifest.experiment} requires a document bundle")
        agent_kwargs["document_bundle_path"] = str(bundle_path)
        unknown = sorted(set(task_ids).difference(self.manifest.task_refs))
        if unknown:
            raise ValueError(f"tasks are not in pinned {self.manifest.dataset['reference']}: {unknown}")
        config: dict[str, Any] = {
            "job_name": job_name,
            "jobs_dir": str(jobs_dir),
            "n_attempts": 1,
            "retry": {"max_retries": FAILURE_POLICY_CONTRACT["harbor_max_retries"]},
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
        }
        config["datasets"] = [
            {
                "name": self.manifest.dataset["identifier"],
                "ref": self.manifest.dataset["registry_content_hash"],
                "task_names": list(task_ids),
            }
        ]
        return config

    def run(self, task_ids: Sequence[str], candidate: Mapping[str, str]) -> HarborEvaluation:
        """Run one isolated Harbor job and parse every task by exact ID.

        Args:
            task_ids: Unique pinned task IDs in desired output order.
            candidate: Complete set of editable documents for this experiment.

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
        self.manifest.validate_candidate(candidate)
        harbor, _docker = self.check_requirements()

        candidate_digest = self.manifest.candidate_digest(candidate)
        evaluation_id = f"{candidate_digest[:12]}-{uuid.uuid4().hex}"
        evaluation_dir = self.work_dir / "evaluations" / evaluation_id
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        prompt_path = evaluation_dir / "terminus-prompt.txt"
        bundle_path = write_document_bundle(evaluation_dir, candidate)
        (evaluation_dir / "candidate.json").write_text(
            json.dumps(
                {"experiment": self.manifest.experiment, "digest": candidate_digest, "documents": dict(candidate)},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        jobs_dir = evaluation_dir / "jobs"
        job_name = f"candidate-{candidate_digest[:12]}"
        config = self.build_job_config(
            task_ids,
            prompt_path=prompt_path,
            bundle_path=bundle_path,
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

        trials: dict[str, HarborTrialResult] = {}
        for result_path in sorted(job_dir.glob("*/result.json")):
            raw_result = json.loads(result_path.read_text())
            task_id = raw_result.get("task_name")
            if not isinstance(task_id, str):
                raise HarborExecutionError(f"Harbor trial result {result_path} has no string task_name")
            if task_id in trials:
                raise HarborExecutionError(f"Harbor produced duplicate results for task {task_id!r}")

            rewards = _read_verifier_rewards(raw_result, task_id)
            errors: list[str] = []
            step_results = raw_result.get("step_results")
            if step_results is not None and not isinstance(step_results, list):
                raise HarborExecutionError(f"Harbor trial {task_id!r} returned invalid step results")
            for record in [raw_result, *(step_results or [])]:
                if not isinstance(record, dict):
                    raise HarborExecutionError(f"Harbor trial {task_id!r} returned an invalid step result")
                exception_info = record.get("exception_info")
                if exception_info is None:
                    continue
                if not isinstance(exception_info, dict):
                    raise HarborExecutionError(f"Harbor trial {task_id!r} returned invalid exception evidence")
                exception_type = exception_info.get("exception_type", "Exception")
                exception_message = exception_info.get("exception_message", "")
                scope = f"{record.get('step_name', 'unknown step')}: " if record is not raw_result else ""
                error = f"{scope}{exception_type}: {exception_message}".rstrip()
                if exception_type not in FAILURE_POLICY_CONTRACT["accepted_trial_exceptions"]:
                    raise HarborExecutionError(f"Harbor trial {task_id!r} reported execution errors: {error}")
                if record is not raw_result:
                    _read_verifier_rewards(record, f"{task_id}/{record.get('step_name')}", require_canonical=False)
                errors.append(error)
            atif_trajectories: list[dict[str, Any]] = []
            trajectory_paths = sorted(result_path.parent.glob("agent/trajectory*.json"))
            trajectory_paths.extend(sorted(result_path.parent.glob("steps/*/agent/trajectory*.json")))
            for trajectory_path in trajectory_paths:
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
                verifier_logs=_read_verifier_logs(result_path.parent, self.text_limits.verifier_log_chars),
            )

        missing = [task_id for task_id in task_ids if task_id not in trials]
        unexpected = sorted(set(trials).difference(task_ids))
        if missing or unexpected:
            raise HarborExecutionError(
                f"Harbor result/task mismatch for evaluation {evaluation_id}: missing={missing}, unexpected={unexpected}"
            )
        _validate_job_result(
            raw_job_result,
            len(task_ids),
            job_result_path,
            verified_timeouts=sum(trial.raw_result.get("exception_info") is not None for trial in trials.values()),
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


class TerminusAdapter(GEPAAdapter[TerminalBenchTask, TerminalBenchTrajectory, TerminalBenchOutput]):
    """Evaluate prompts and skills through GEPA's Terminus adapter port for Harbor.

    The upstream adapter name and GEPA evaluation/reflection interface are retained.
    Its legacy runner, result parsing, and single-prompt feedback are replaced for
    TB2.1; this is not the unmodified upstream implementation or constructor API.
    Candidates expose one unified initial prompt with fixed auxiliary text by
    default, or all prompts and skills when the all-text scope is selected.

    Args:
        manifest: Checked-in, validated experiment manifest.
        harbor: Pinned Harbor subprocess runner configured with the student
            model. The proposer model is supplied separately to ``gepa.optimize``.
        text_scope: Editable candidate boundary and provider seed family.
    """

    def __init__(
        self, manifest: TerminalBenchManifest, harbor: HarborCLI, *, text_scope: TerminalBenchTextScope | None = None
    ) -> None:
        """Bind the validated manifest to its Harbor runner.

        Args:
            manifest: Checked-in, validated experiment manifest.
            harbor: Pinned runner configured with the student model.
            text_scope: Editable candidate boundary; omitted means the unified initial prompt.

        Raises:
            ValueError: The adapter and runner use different manifests.
        """
        if manifest != harbor.manifest:
            raise ValueError("Adapter and Harbor runner must use the same Terminal-Bench manifest")
        self.manifest = manifest
        self.harbor = harbor
        self.text_scope = text_scope or TerminalBenchTextScope()

    def evaluate(
        self,
        batch: list[TerminalBenchTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TerminalBenchTrajectory, TerminalBenchOutput]:
        """Run the candidate on a batch and preserve its incoming task order.

        Args:
            batch: Pinned task records.
            candidate: The full prompt-and-skill bundle.
            capture_traces: Whether to return full ATIF/result evidence to GEPA.

        Returns:
            Harbor rewards, outputs, and optional complete trajectories.

        Raises:
            ValueError: Candidate components or task IDs violate the harness
                contract.
        """
        documents = self.text_scope.materialize(candidate)
        task_ids = [task.task_id for task in batch]
        unknown = sorted(set(task_ids).difference(self.manifest.task_refs))
        if unknown:
            raise ValueError(f"tasks are not in pinned {self.manifest.dataset['reference']}: {unknown}")
        evaluation = self.harbor.run(task_ids, documents)

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
                "candidate_digest": evaluation.candidate_digest,
                "job_dir": str(evaluation.job_dir),
                "config_path": str(evaluation.config_path),
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
                        "candidate_documents": documents,
                        "reward": trial.reward,
                        "rewards": trial.rewards,
                        "errors": errors,
                        "atif_trajectories": trial.atif_trajectories,
                        "verifier_logs": dict(trial.verifier_logs),
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
        """Expose training trajectories, verifier diagnostics, and rewards for reflection.

        Args:
            candidate: Candidate used for the captured evaluation.
            eval_batch: Evaluation batch returned with ``capture_traces=True``.
            components_to_update: Components requested by GEPA.

        Returns:
            Execution evidence for each selected document component.

        Raises:
            ValueError: A component is unknown, the candidate is incomplete,
                or feedback contains a task outside the training split.
            RuntimeError: The evaluation omitted trajectories.
        """
        if not components_to_update or not set(components_to_update).issubset(self.text_scope.component_kinds):
            raise ValueError(f"Unknown Terminal Bench document selection: {components_to_update}")
        self.text_scope.materialize(candidate)
        if eval_batch.trajectories is None:
            raise RuntimeError("Terminal-Bench reflection requires capture_traces=True")
        if any(trajectory["task_id"] not in self.manifest.splits["train"] for trajectory in eval_batch.trajectories):
            raise ValueError("Terminal-Bench reflection feedback is restricted to training tasks")

        rows: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories:
            rows.append(
                {
                    "Inputs": {
                        "dataset": self.manifest.dataset["reference"],
                        "experiment": self.manifest.experiment,
                        "task_id": trajectory["task_id"],
                    },
                    "Generated Outputs": {
                        "atif_trajectories": deepcopy(trajectory["atif_trajectories"]),
                        "trial_result": deepcopy(trajectory["trial_result"]),
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
                            "verifier_log_status": "available" if trajectory["verifier_logs"] else "unavailable",
                            "verifier_logs": trajectory["verifier_logs"],
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                }
            )
        return {
            component: [
                {
                    **row,
                    "Document": {
                        "name": component,
                        "kind": self.text_scope.component_kinds[component],
                        "text": candidate[component],
                    },
                }
                for row in rows
            ]
            for component in components_to_update
        }


TerminalBenchAdapter = TerminusAdapter

TERMINUS_ADAPTER_CONTRACT = {
    "version": 1,
    "entry_point": f"{TerminusAdapter.__module__}.{TerminusAdapter.__qualname__}",
    "implementation": "harbor_port",
    "upstream_repository": "https://github.com/gepa-ai/gepa",
    "upstream_commit": "4f1613773d0c13c8f1551543a801b299bd8acf73",
    "upstream_path": "src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py",
    "upstream_blob": "1786e06b8e4bdee4129d85521bca4a26fcab7c7e",
    "upstream_class": "TerminusAdapter",
    "runtime": "harbor_cli",
    "harbor_version": PINNED_HARBOR_VERSION,
}
