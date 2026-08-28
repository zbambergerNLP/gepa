"""Configure a pinned Terminal-Bench v3 GEPA optimization run.

The held-out test split is not evaluated automatically.

* ``vanilla`` uses stock free-form GEPA reflection.
* ``react_v2`` uses the Controller -> Manifestor -> ReAct V2 workflow.

Within each model arm, all conditions use the same official Harbor rewards,
manifest, student/proposer model, task splits, and metric-call budget.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    QWEN3_8_27B_MODEL_INFO,
    experiment_decoding,
    validate_experiment_model_pair,
)
from examples.common.react_v2 import resolve_template_family, structured_prompt
from gepa import optimize
from gepa.adapters.terminal_bench_adapter import (
    HarborCLI,
    TerminalBenchAdapter,
    TerminalBenchManifest,
    TerminalBenchTask,
    load_terminalbench_manifest,
)
from gepa.strategies.intervention import CONTROLLER_POLICY_CONTRACT, SEMANTIC_ACTION_CATALOGS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("terminalbench-v3-manifest.json")

SEED_INSTRUCTION = """Complete the assigned command-line task in the provided Linux tmux session.

Each turn includes the task description and current terminal screen. Use only the fixed tmux interface. Inspect the environment before editing, run targeted commands, check their results, and handle errors or incomplete output before continuing.

Stop when the verifier's required artifact or behavior is present. Format replies with the appended Terminus JSON command contract."""
RUN_CONTRACT_FILENAME = "terminalbench-run-contract.json"


def seed_candidate(student_model: str, template_family: str) -> tuple[dict[str, str], str]:
    """Build the Terminus user-message target with the selected provider template.

    Args:
        student_model: Task model used for automatic provider inference.
        template_family: Explicit provider family or ``"auto"``.

    Returns:
        Single instruction component and its resolved template family.
    """
    resolved_family = resolve_template_family(template_family, student_model)
    return {"instruction_prompt": structured_prompt(SEED_INSTRUCTION, resolved_family, "user_prompt")}, resolved_family


def ensure_run_contract(run_dir: Path, contract: dict[str, Any]) -> Path:
    """Write the run contract or reject an incompatible resumable directory.

    Args:
        run_dir: Experiment directory that owns the resumable state.
        contract: Complete material configuration for the requested run.

    Returns:
        Path to the existing or newly written contract file.

    Raises:
        ValueError: Existing state has a different contract, or legacy GEPA
            state has no contract to validate.
    """
    path = run_dir / RUN_CONTRACT_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise ValueError(f"Run directory {run_dir} contains a different Terminal-Bench configuration.")
        return path
    if (run_dir / "gepa_state.bin").exists():
        raise ValueError(
            f"Run directory {run_dir} has GEPA state but no {RUN_CONTRACT_FILENAME}; choose a clean directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment CLI without launching any evaluation.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="GEPA on pinned Terminal-Bench v3 through Harbor")
    parser.add_argument(
        "--condition",
        choices=("vanilla", "react_v2", "action"),
        required=True,
        help="Optimization condition to run",
    )
    parser.add_argument(
        "--student-model",
        default=QWEN3_8_27B_MODEL,
        help="Terminus model; use the same supported model as --proposer-model",
    )
    parser.add_argument(
        "--proposer-model",
        default=QWEN3_8_27B_MODEL,
        help="GEPA proposer; use the same supported model as --student-model",
    )
    parser.add_argument("--student-api-base", default=None)
    parser.add_argument("--proposer-api-base", default=None)
    parser.add_argument("--max-metric-calls", type=int, required=True)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edit-tool-set",
        choices=("minimal", "broad"),
        default="broad",
        help="Edit tools used by ReAct V2",
    )
    parser.add_argument(
        "--reflection-level",
        type=int,
        choices=(1, 2),
        default=2,
        help="Reflection level: region only, or region plus an applied semantic action",
    )
    parser.add_argument(
        "--template-family",
        choices=("auto", "generic", "openai", "anthropic", "google", "alibaba"),
        default="auto",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--harbor-work-dir", type=Path, required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument(
        "--harbor-process-timeout-sec",
        type=float,
        default=None,
        help="Optional whole-job timeout; default leaves long-horizon runs to task-level Harbor timeouts",
    )
    return parser


def build_run_contract(
    args: argparse.Namespace,
    manifest: TerminalBenchManifest,
    trainset: list[TerminalBenchTask],
    valset: list[TerminalBenchTask],
    condition: str,
    resolved_family: str,
) -> dict[str, Any]:
    """Record every material axis needed for safe resume and comparison.

    Args:
        args: Parsed Terminal-Bench CLI arguments.
        manifest: Validated pinned benchmark manifest.
        trainset: Selected training tasks in manifest order.
        valset: Selected validation tasks in manifest order.
        condition: Canonical optimization condition.
        resolved_family: Provider template family used by the student prompt.

    Returns:
        JSON-serializable run contract including exact task identities.
    """
    validate_experiment_model_pair(args.student_model, args.proposer_model)
    operated = condition == "react_v2"
    reflection_level = args.reflection_level if operated else 0
    return {
        "schema_version": 4,
        "condition": condition,
        "component_kinds": {"instruction_prompt": "user_prompt"},
        "dataset": manifest.dataset,
        "edit_tool_set": args.edit_tool_set,
        "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
        "manifest": str(args.manifest.resolve()),
        "max_metric_calls": args.max_metric_calls,
        "n_concurrent": args.n_concurrent,
        "proposer_api_base": args.proposer_api_base,
        "proposer_backend": "react_v2" if operated else "stateless",
        "proposer_decoding": experiment_decoding(args.proposer_model),
        "proposer_model": args.proposer_model,
        "proposer_num_retries": EXPERIMENT_NUM_RETRIES,
        "reflection_level": reflection_level,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "max_proposer_model_calls": 8 if operated else None,
        "semantic_action_space": deepcopy(SEMANTIC_ACTION_CATALOGS["prompt"]) if reflection_level == 2 else None,
        "semantic_controller_policy": deepcopy(CONTROLLER_POLICY_CONTRACT) if reflection_level == 2 else None,
        "seed": args.seed,
        "student_api_base": args.student_api_base,
        "student_decoding": experiment_decoding(args.student_model),
        "student_model": args.student_model,
        "student_model_info": dict(QWEN3_8_27B_MODEL_INFO) if args.student_model == QWEN3_8_27B_MODEL else None,
        "student_num_retries": EXPERIMENT_NUM_RETRIES,
        "template_family": resolved_family,
        "train_task_ids": [task.task_id for task in trainset],
        "val_task_ids": [task.task_id for task in valset],
    }


def main() -> None:
    """Validate the pinned harness and start the requested GEPA condition.

    Raises:
        ValueError: Training or validation selection is empty.
    """
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_experiment_model_pair(args.student_model, args.proposer_model)
    except ValueError as exc:
        parser.error(str(exc))
    manifest = load_terminalbench_manifest(args.manifest)
    trainset = manifest.tasks("train", args.train_limit)
    valset = manifest.tasks("val", args.val_limit)
    if not trainset or not valset:
        raise ValueError("train and validation selections must both be non-empty")

    candidate, resolved_family = seed_candidate(args.student_model, args.template_family)
    condition = "react_v2" if args.condition == "action" else args.condition
    contract = build_run_contract(args, manifest, trainset, valset, condition, resolved_family)
    ensure_run_contract(args.run_dir, contract)

    student_agent_kwargs: dict[str, Any] = {
        "llm_kwargs": {"num_retries": EXPERIMENT_NUM_RETRIES, **experiment_decoding(args.student_model)}
    }
    if args.student_model == QWEN3_8_27B_MODEL:
        student_agent_kwargs["model_info"] = dict(QWEN3_8_27B_MODEL_INFO)

    harbor = HarborCLI(
        student_model=args.student_model,
        student_api_base=args.student_api_base,
        work_dir=args.harbor_work_dir,
        agent_python_path=REPO_ROOT,
        n_concurrent=args.n_concurrent,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        student_agent_kwargs=student_agent_kwargs,
        process_timeout_sec=args.harbor_process_timeout_sec,
    )
    harbor.check_requirements()
    adapter = TerminalBenchAdapter(manifest, harbor)

    reflection_lm_kwargs: dict[str, Any] = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(args.proposer_model),
    }
    if args.proposer_api_base is not None:
        reflection_lm_kwargs["api_base"] = args.proposer_api_base

    reflection_level = 0 if condition == "vanilla" else args.reflection_level
    optimize(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.proposer_model,
        reflection_lm_kwargs=reflection_lm_kwargs,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        run_dir=str(args.run_dir),
        seed=args.seed,
        reflection_level=reflection_level,
        edit_tool_set=args.edit_tool_set,
        component_kinds={"instruction_prompt": "user_prompt"},
        template_family=resolved_family,
        template_model=args.student_model,
    )


if __name__ == "__main__":
    main()
