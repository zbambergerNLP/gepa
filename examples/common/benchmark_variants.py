"""Resolve supported experiment variants identically for every benchmark."""

from __future__ import annotations

import argparse
from typing import Any

from gepa.strategies.forest_constants import DEFAULT_REFLECTION_LEVEL
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)
from gepa.strategies.proposal_selection import AllImprovements, BestImprovement, TopKImprovements

CONDITIONS = ("vanilla", "random", "action", "react_v2_random", "react_v2")
FOREST_CONDITIONS = {"react_v2", "react_v2_random"}


def add_variant_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose existing scientific conditions and built-in search ablations."""
    parser.add_argument("--condition", choices=(*CONDITIONS, "forest", "both", "all"), default="both")
    parser.add_argument("--reflection-level", type=int, choices=(0, 1, 2), default=DEFAULT_REFLECTION_LEVEL)
    parser.add_argument("--controller-selection", choices=("verbalized", "uniform_random", "jev"), default="verbalized")
    parser.add_argument("--module-selector", choices=("round_robin", "all", "controller"), default="round_robin")
    parser.add_argument("--edit-tool-set", choices=("broad", "minimal"), default="broad")
    parser.add_argument(
        "--editor-mode",
        choices=("react", "single_call"),
        default=None,
        help="Default: single_call for level-2 real_edit, otherwise react",
    )
    parser.add_argument("--proposal-policy", choices=("real_edit", "independent"), default="real_edit")
    parser.add_argument("--react-max-iterations", type=int, default=None)
    parser.add_argument("--react-max-tool-calls", type=int, default=None)
    parser.add_argument(
        "--candidate-selection", choices=("pareto", "current_best", "epsilon_greedy", "top_k_pareto"), default="pareto"
    )
    parser.add_argument(
        "--acceptance", choices=("strict_improvement", "improvement_or_equal"), default="strict_improvement"
    )
    parser.add_argument(
        "--sampling-strategy", choices=("single", "same_parent", "independent", "pxn"), default="single"
    )
    parser.add_argument("--proposal-count", type=int, default=1, help="Mutations per parent, or independent proposals")
    parser.add_argument("--parent-count", type=int, default=1, help="Parents for pxn proposal sampling")
    parser.add_argument(
        "--proposal-selection", choices=("all_improvements", "best_improvement", "top_k"), default="all_improvements"
    )
    parser.add_argument("--proposal-top-k", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="Enable GEPA's five-invocation merge ablation")


def selected_conditions(args: argparse.Namespace) -> tuple[str, ...]:
    """Expand historical condition groups and the descriptive FOREST alias."""
    if args.condition == "all":
        return CONDITIONS
    if args.condition == "both":
        return ("vanilla", "react_v2")
    return ("react_v2" if args.condition == "forest" else args.condition,)


def variant_settings(args: argparse.Namespace, condition: str) -> dict[str, Any]:
    """Return effective settings and reject unsupported combinations before model work."""
    forest = condition in FOREST_CONDITIONS
    level = args.reflection_level if forest else 0
    controller = "uniform_random" if condition == "react_v2_random" else args.controller_selection
    policy = args.proposal_policy if forest and level == 2 else "independent"
    editor = args.editor_mode or ("single_call" if policy == "real_edit" else "react")
    if forest:
        if level == 0 and controller != "verbalized":
            raise ValueError("Reflection level 0 requires the verbalized Controller setting")
        if controller == "jev" and level != 2:
            raise ValueError("The Jev Controller requires --reflection-level 2")
        if policy == "real_edit" and editor != "single_call":
            raise ValueError("Level-2 real_edit requires single_call; use --proposal-policy independent for react")
        if editor == "single_call" and args.edit_tool_set != "broad":
            raise ValueError("Single-call editing requires --edit-tool-set broad")
    if args.module_selector == "controller" and (not forest or level != 2 or controller == "uniform_random"):
        raise ValueError(
            "Controller module selection requires one level-2 FOREST arm with a verbalized or Jev Controller"
        )
    if args.sampling_strategy == "single" and args.proposal_count != 1:
        raise ValueError("Single proposal sampling requires --proposal-count 1")
    if args.sampling_strategy != "pxn" and args.parent_count != 1:
        raise ValueError("--parent-count requires --sampling-strategy pxn")
    if args.proposal_selection != "top_k" and args.proposal_top_k != 1:
        raise ValueError("--proposal-top-k requires --proposal-selection top_k")
    return {
        "reflection_level": level,
        "controller_selection": controller if forest and level else None,
        "stateless_action_selection": {"random": "random", "action": "verbalized"}.get(condition),
        "module_selector": args.module_selector,
        "edit_tool_set": args.edit_tool_set if forest and level else None,
        "editor_mode": editor if forest and level else None,
        "proposal_policy": policy if forest and level else None,
        "react_max_iterations": args.react_max_iterations if forest and editor == "react" and level else None,
        "react_max_tool_calls": args.react_max_tool_calls if forest and editor == "react" and level else None,
        "candidate_selection": args.candidate_selection,
        "frontier_type": "instance",
        "acceptance": args.acceptance,
        "proposal_sampling": {
            "strategy": args.sampling_strategy,
            "proposal_count": args.proposal_count,
            "parent_count": args.parent_count,
        },
        "proposal_selection": {"strategy": args.proposal_selection, "top_k": args.proposal_top_k},
        "merge": args.merge,
        "max_merge_invocations": 5 if args.merge else 0,
    }


def proposal_strategies(args: argparse.Namespace) -> tuple[Any, Any]:
    """Build the tested core sampling and admission strategies without wrappers."""
    sampling = {
        "single": lambda: SingleMutationSampling(),
        "same_parent": lambda: SameParentSampling(args.proposal_count),
        "independent": lambda: IndependentSampling(args.proposal_count),
        "pxn": lambda: PxNSampling(args.parent_count, args.proposal_count),
    }[args.sampling_strategy]()
    selection = {
        "all_improvements": lambda: AllImprovements(),
        "best_improvement": lambda: BestImprovement(),
        "top_k": lambda: TopKImprovements(args.proposal_top_k),
    }[args.proposal_selection]()
    return sampling, selection


def optimizer_budget(args: argparse.Namespace, max_candidate_proposals: int | None) -> dict[str, int | None]:
    """Preserve exact opportunity caps when one iteration samples several proposals."""
    proposals_per_iteration = args.proposal_count * args.parent_count
    if args.mode == "optimizer-pilot":
        iterations = args.pilot_proposals
        max_candidate_proposals = iterations * proposals_per_iteration
    elif max_candidate_proposals is not None:
        if max_candidate_proposals % proposals_per_iteration:
            raise ValueError(
                f"Proposal group size {proposals_per_iteration} does not divide the pinned "
                f"{max_candidate_proposals}-opportunity budget; choose proposal counts that divide the budget"
            )
        iterations = max_candidate_proposals // proposals_per_iteration
    else:
        iterations = None
    return {
        "max_metric_calls": None if args.mode == "optimizer-pilot" else args.max_metric_calls,
        "max_candidate_proposals": max_candidate_proposals,
        "max_optimizer_iterations": iterations,
        "proposals_per_iteration": proposals_per_iteration,
    }
