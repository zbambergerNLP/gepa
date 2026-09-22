"""Preserve complete Terminal-Bench traces, including repeated and copied context."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

from gepa.adapters.terminal_bench_adapter import HarborCLI, TerminusAdapter, load_terminalbench_manifest
from gepa.core.adapter import EvaluationBatch


def _reflect(trajectories: list[dict]) -> list[dict]:
    """Build reflection evidence through the real adapter without running Harbor."""
    manifest = load_terminalbench_manifest(
        Path(__file__).resolve().parents[1] / "examples/terminalbench/terminalbench-v2.1-manifest.json"
    )
    adapter = TerminusAdapter(manifest, Mock(spec=HarborCLI, manifest=manifest))
    candidate = adapter.text_scope.seed_candidate()
    evaluated = EvaluationBatch(
        outputs=[{}],
        scores=[0.0],
        trajectories=[
            {
                "task_id": manifest.splits["train"][0],
                "atif_trajectories": trajectories,
                "trial_result": {},
                "harbor_returncode": 0,
                "harbor_stdout_path": "stdout.txt",
                "harbor_stderr_path": "stderr.txt",
                "reward": 0.0,
                "rewards": {"reward": 0.0},
                "errors": [],
                "verifier_logs": {},
            }
        ],
    )
    rows = adapter.make_reflective_dataset(candidate, evaluated, ["instruction_prompt"])
    return rows["instruction_prompt"][0]["Generated Outputs"]["atif_trajectories"]


def _trajectory(steps: list[dict]) -> dict:
    """Build an ATIF trace containing bulky operational metadata."""
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "terminus", "version": "2", "extra": {"configuration": "CONFIGURATION" * 1000}},
        "final_metrics": {"logprobs": [0.1] * 1000},
        "steps": steps,
    }


def test_reflection_preserves_complete_original_and_copied_context() -> None:
    """Keep every original field and copied step without altering stored traces."""
    original = {
        "step_id": 1,
        "source": "agent",
        "message": "Run the failing test",
        "reasoning_content": "Need to verify the output",
        "tool_calls": [{"tool_call_id": "call-1", "function_name": "terminal", "arguments": {"command": "pytest"}}],
        "observation": {"results": [{"content": "FAILED: important error", "source_call_id": "call-1"}]},
        "metrics": {"logprobs": [0.0] * 1000},
    }
    copied = {**deepcopy(original), "is_copied_context": True, "metrics": None}
    traces = [
        _trajectory([copied, {"step_id": 2, "source": "agent", "message": "New summary reasoning"}]),
        _trajectory([original]),
    ]
    untouched = deepcopy(traces)
    reflected = _reflect(traces)
    assert traces == untouched
    assert reflected == traces
    assert str(reflected).count("Run the failing test") == 2
    reflected[0]["steps"][0]["message"] = "Changed by reflection consumer"
    assert traces == untouched


def test_copied_content_without_a_visible_original_is_kept() -> None:
    """Avoid dangling references or lost evidence when the original trace is absent."""
    text = "\n".join(f"Unique evidence {index}" for index in range(1000))
    traces = [_trajectory([{"step_id": 1, "source": "user", "message": text, "is_copied_context": True}])]
    assert _reflect(traces)[0]["steps"][0]["message"] == text


def test_real_repeated_actions_remain_separate_steps() -> None:
    """Keep repeated attempts and their outcomes when they are not copied history."""
    traces = [_trajectory([{"step_id": index, "source": "agent", "message": "retry"} for index in range(1, 4)])]
    steps = _reflect(traces)[0]["steps"]
    assert [step["step_id"] for step in steps] == [1, 2, 3]
    assert all(step["message"] == "retry" for step in steps)


def test_embedded_subagent_evidence_keeps_its_full_copied_history() -> None:
    """Retain nested children and copied parent context at their original positions."""
    parent_step = {"step_id": 1, "source": "user", "message": "Task instruction"}
    parent = _trajectory([parent_step])
    parent["subagent_trajectories"] = [
        {**_trajectory([{**parent_step, "is_copied_context": True}]), "trajectory_id": "child-1"}
    ]
    reflected = _reflect([parent])
    assert reflected == [parent]
    assert str(reflected).count("Task instruction") == 2
