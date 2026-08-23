"""Configure a pinned Terminal-Bench v3 GEPA optimization run.

The harness offers three comparable proposer conditions and does not evaluate the
held-out test split automatically:

* ``vanilla`` uses stock free-form GEPA reflection.
* ``react_v2`` uses the Controller -> Manifestor -> ReAct V2 workflow.
* ``rlm`` is the trusted-model, in-process RLM proposer ablation.

All conditions use the same official Harbor rewards, manifest, student model,
task splits, and metric-call budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.common.react_v2 import resolve_template_family, structured_prompt
from gepa import optimize
from gepa.adapters.terminal_bench_adapter import (
    HarborCLI,
    TerminalBenchAdapter,
    TerminalBenchManifest,
    TerminalBenchTask,
    load_terminalbench_manifest,
)
from gepa.lm import LM
from gepa.proposer.reflective_mutation.rlm_environment import RLMBudget
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.intervention import controller_policy_contract, semantic_action_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("terminalbench-v3-manifest.json")

SEED_INSTRUCTION = """You are an expert autonomous terminal agent operating in a Linux tmux session.

Solve the assigned command-line task completely and leave the environment in the state required by its verifier. You receive the task description and current terminal screen on every turn.

Work only through the fixed tmux terminal interface. Inspect the environment before making assumptions. Prefer reproducible, targeted changes, verify the final state, and recover explicitly from command errors or incomplete output. Continue until the requested artifact or behavior is present. Follow the fixed Terminus JSON command contract appended to this instruction prompt."""
RUN_CONTRACT_FILENAME = "terminalbench-run-contract.json"


def _rlm_budget() -> RLMBudget:
    """Return the RLM budget matched to ReAct V2's eight proposer turns."""
    return RLMBudget(
        max_root_iterations=4,
        max_child_iterations=2,
        max_repl_calls=6,
        max_llm_queries=2,
        max_rlm_queries=1,
        max_recursion_depth=1,
        max_exec_seconds=5,
        max_output_chars=4000,
    )


def _rlm_max_model_calls(budget: RLMBudget) -> int:
    """Derive the root, child, and leaf model-call cap."""
    return budget.max_model_calls


def seed_candidate(student_model: str, template_family: str) -> tuple[dict[str, str], str]:
    """Build the system-prompt target in the student's canonical template."""
    resolved_family = resolve_template_family(template_family, student_model)
    return {"instruction_prompt": structured_prompt(SEED_INSTRUCTION, resolved_family)}, resolved_family


def ensure_run_contract(run_dir: Path, contract: dict[str, Any]) -> Path:
    """Write the run contract or reject an incompatible resumable directory."""
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
        choices=("vanilla", "react_v2", "rlm", "action"),
        required=True,
        help="rlm is an explicit trusted-model in-process ablation, not a security sandbox",
    )
    parser.add_argument("--student-model", required=True, help="Terminus task-solving model")
    parser.add_argument("--proposer-model", required=True, help="GEPA reflection/proposer model")
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
        help="Operator basis; the RLM condition requires broad",
    )
    parser.add_argument(
        "--reflection-level",
        type=int,
        choices=(1, 2),
        default=2,
        help="Operated-proposer rung: region only, or region plus manifested semantic action",
    )
    parser.add_argument(
        "--template-family",
        choices=("auto", "generic", "openai", "openai-gpt-5.6", "anthropic", "google", "alibaba"),
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
    """Record every material axis needed for safe resume and comparison."""
    operated = condition in ("react_v2", "rlm")
    reflection_level = args.reflection_level if operated else 0
    rlm_budget = _rlm_budget() if condition == "rlm" else None
    return {
        "condition": condition,
        "dataset": manifest.dataset,
        "edit_tool_set": args.edit_tool_set,
        "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
        "manifest": str(args.manifest.resolve()),
        "max_metric_calls": args.max_metric_calls,
        "n_concurrent": args.n_concurrent,
        "proposer_api_base": args.proposer_api_base,
        "proposer_backend": condition if operated else "stateless",
        "proposer_model": args.proposer_model,
        "reflection_level": reflection_level,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "rlm_budget": asdict(rlm_budget) if rlm_budget is not None else None,
        "max_proposer_model_calls": (
            _rlm_max_model_calls(rlm_budget) if rlm_budget is not None else 8 if condition == "react_v2" else None
        ),
        "semantic_action_space": semantic_action_catalog("prompt") if reflection_level == 2 else None,
        "semantic_controller_policy": controller_policy_contract() if reflection_level == 2 else None,
        "seed": args.seed,
        "student_api_base": args.student_api_base,
        "student_model": args.student_model,
        "template_family": resolved_family,
        "train_task_ids": [task.task_id for task in trainset],
        "val_task_ids": [task.task_id for task in valset],
    }


def main() -> None:
    """Validate the pinned harness and start the requested GEPA condition."""
    parser = build_parser()
    args = parser.parse_args()
    manifest = load_terminalbench_manifest(args.manifest)
    trainset = manifest.tasks("train", args.train_limit)
    valset = manifest.tasks("val", args.val_limit)
    if not trainset or not valset:
        raise ValueError("train and validation selections must both be non-empty")

    candidate, resolved_family = seed_candidate(args.student_model, args.template_family)
    condition = "react_v2" if args.condition == "action" else args.condition
    if condition == "rlm" and args.edit_tool_set != "broad":
        parser.error("--condition rlm requires --edit-tool-set broad")
    if condition == "rlm" and args.reflection_level != 2:
        parser.error("--condition rlm requires --reflection-level 2")
    contract = build_run_contract(args, manifest, trainset, valset, condition, resolved_family)
    ensure_run_contract(args.run_dir, contract)

    harbor = HarborCLI(
        student_model=args.student_model,
        student_api_base=args.student_api_base,
        work_dir=args.harbor_work_dir,
        agent_python_path=REPO_ROOT,
        n_concurrent=args.n_concurrent,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        process_timeout_sec=args.harbor_process_timeout_sec,
    )
    harbor.check_requirements()
    adapter = TerminalBenchAdapter(manifest, harbor)

    reflection_lm_kwargs: dict[str, Any] = {}
    if args.proposer_api_base is not None:
        reflection_lm_kwargs["api_base"] = args.proposer_api_base

    reflection_level = 0 if condition == "vanilla" else args.reflection_level
    reflection_strategy = None
    if condition == "rlm":
        manifestor_lm_kwargs = {**reflection_lm_kwargs, "temperature": 0}
        reflection_strategy = ThreeRoleReflectionLM(
            base_lm=LM(args.proposer_model, **reflection_lm_kwargs),
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            component_kinds={"instruction_prompt": "prompt"},
            template_family=resolved_family,
            manifestor_lm=LM(args.proposer_model, **manifestor_lm_kwargs),
            proposer_model=args.proposer_model,
            proposer_backend="rlm",
            rlm_budget=_rlm_budget(),
        )
    optimize(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.proposer_model,
        reflection_lm_kwargs=reflection_lm_kwargs,
        reflection_strategy=reflection_strategy,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        run_dir=str(args.run_dir),
        seed=args.seed,
        reflection_level=reflection_level,
        edit_tool_set=args.edit_tool_set,
        component_kinds={"instruction_prompt": "prompt"},
        template_family=resolved_family,
        template_model=args.student_model,
    )


if __name__ == "__main__":
    main()
