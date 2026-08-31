"""Summarize HotPotQA performance, proposal diversity, and action behavior."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.common.react_v2 import WIKIPEDIA_RUN_CONTRACT_FILENAME

_BUDGET_LABELS = {6_871: "standard", 13_742: "expanded"}
_CONDITION_ORDER = {
    "vanilla": 0,
    "react_v2": 1,
    "react_v2_random": 2,
    "action": 3,
}
_MODEL_LABELS = {
    "hosted_vllm/Qwen/Qwen3.8-27B": "Qwen3.8-27B",
    "hosted_vllm/zai-org/GLM-5.3-Flash": "GLM-5.3-Flash",
}
_APPROVED_CELLS = {
    (6_871, "vanilla"),
    (6_871, "react_v2"),
    (6_871, "react_v2_random"),
    (6_871, "action"),
    (13_742, "vanilla"),
    (13_742, "react_v2"),
}


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk.

    Args:
        path: JSON file to read.

    Returns:
        Parsed string-keyed object.

    Raises:
        ValueError: The file contains a non-object JSON value.
        OSError: The file cannot be read.
        json.JSONDecodeError: The file is not valid JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def entropy_bits(probabilities: Iterable[float]) -> float:
    """Calculate Shannon entropy in bits.

    Args:
        probabilities: Iterable of non-negative probability masses. Zero
            entries are ignored.

    Returns:
        Shannon entropy in bits.

    Raises:
        ValueError: A probability is negative or the positive mass does not
            sum to one within numerical tolerance.
    """
    values = [float(probability) for probability in probabilities]
    if any(probability < 0 for probability in values):
        raise ValueError("Entropy probabilities must be non-negative.")
    positive_mass = sum(probability for probability in values if probability > 0)
    if values and not math.isclose(positive_mass, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Entropy probabilities must sum to one; found {positive_mass}.")
    entropy = -sum(probability * math.log2(probability) for probability in values if probability > 0)
    return entropy


def jaccard_diversity(texts: Sequence[str]) -> float | None:
    """Calculate mean pairwise token-set Jaccard distance.

    Args:
        texts: Proposal or candidate texts from one document component.

    Returns:
        Mean pairwise distance, or ``None`` when fewer than two non-empty texts
        are available.
    """
    token_sets = [set(text.lower().split()) for text in texts if text.strip()]
    distances = []
    for left, right in itertools.combinations(token_sets, 2):
        union = left | right
        if union:
            distances.append(1.0 - (len(left & right) / len(union)))
    if not distances:
        return None
    mean_distance = sum(distances) / len(distances)
    return mean_distance


def proposal_diversity(proposal_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Measure proposal diversity independently for every prompt component.

    Args:
        proposal_records: Callback records containing ``texts_by_component``.

    Returns:
        Component names mapped to mean pairwise Jaccard distance. Components
        with fewer than two completed proposals are omitted.
    """
    texts_by_component: dict[str, list[str]] = {}
    for record in proposal_records:
        raw_texts = record.get("texts_by_component", {})
        if not isinstance(raw_texts, Mapping):
            raise ValueError("Proposal texts must be stored as a component mapping.")
        for component, text in raw_texts.items():
            texts_by_component.setdefault(str(component), []).append(str(text))

    diversity: dict[str, float] = {}
    for component, texts in sorted(texts_by_component.items()):
        distance = jaccard_diversity(texts)
        if distance is not None:
            diversity[component] = distance
    return diversity


def candidate_diversity(candidates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Measure explored-candidate diversity from the candidate artifact.

    Args:
        candidates: Explored prompt-component mappings in optimizer order.

    Returns:
        Per-component mean Jaccard distance and unique-text count.

    Raises:
        ValueError: Candidates do not share the same component keys.
    """
    if not candidates:
        return {}
    components = tuple(str(component) for component in candidates[0])
    expected_components = set(components)
    result: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        if {str(component) for component in candidate} != expected_components:
            raise ValueError("Every candidate must contain the same prompt components.")
    for component in components:
        texts = [str(candidate[component]) for candidate in candidates]
        distance = jaccard_diversity(texts)
        result[component] = {
            "mean_pairwise_jaccard_distance": distance if distance is not None else 0.0,
            "num_unique_texts": float(len(set(texts))),
        }
    return result


def action_statistics(
    action_payload: Mapping[str, Any],
    semantic_action_count: int,
) -> dict[str, Any] | None:
    """Summarize semantic-action frequency and outcomes.

    Args:
        action_payload: Parsed ``action_summary.json`` object.
        semantic_action_count: Size of the complete semantic-action catalog.

    Returns:
        Action counts, rates, score deltas, and entropy, or ``None`` for an
        unconditioned run.

    Raises:
        ValueError: Stored summary fields have incompatible types.
    """
    summary = action_payload.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ValueError("Action summary must be a mapping.")
    raw_counts = summary.get("action_proposal_counts", {})
    raw_acceptance_counts = summary.get("action_acceptance_counts", {})
    raw_rejection_counts = summary.get("action_rejection_counts", {})
    raw_rates = summary.get("action_acceptance_rates", {})
    raw_deltas = summary.get("action_score_deltas", {})
    if any(
        not isinstance(value, Mapping)
        for value in (raw_counts, raw_acceptance_counts, raw_rejection_counts, raw_rates, raw_deltas)
    ):
        raise ValueError("Action counts, rates, and score deltas must be mappings.")

    counts = {str(action): int(count) for action, count in raw_counts.items()}
    acceptance_counts = {str(action): int(count) for action, count in raw_acceptance_counts.items()}
    rejection_counts = {str(action): int(count) for action, count in raw_rejection_counts.items()}
    if any(count < 0 for count in (*counts.values(), *acceptance_counts.values(), *rejection_counts.values())):
        raise ValueError("Action counts must be non-negative.")
    total = sum(counts.values())
    if int(summary.get("total_proposals", -1)) != total:
        raise ValueError("Action proposal total disagrees with per-action counts.")
    if int(summary.get("total_accepted", -1)) != sum(acceptance_counts.values()):
        raise ValueError("Action acceptance total disagrees with per-action counts.")
    acceptance_rates = {str(action): float(rate) for action, rate in raw_rates.items()}
    if (
        not set(acceptance_counts).issubset(counts)
        or not set(rejection_counts).issubset(counts)
        or set(acceptance_rates) != set(counts)
        or not {str(action) for action in raw_deltas}.issubset(counts)
    ):
        raise ValueError("Action outcomes contain unknown or missing action names.")
    if total == 0:
        if acceptance_counts or rejection_counts or acceptance_rates or raw_deltas:
            raise ValueError("An actionless summary cannot contain action outcomes.")
        return None
    for action, count in counts.items():
        accepted = acceptance_counts.get(action, 0)
        rejected = rejection_counts.get(action, 0)
        if accepted + rejected > count:
            raise ValueError(f"Action {action!r} has more outcomes than proposals.")
        expected_rate = accepted / count
        if not math.isclose(acceptance_rates.get(action, math.nan), expected_rate, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Action {action!r} has an inconsistent acceptance rate.")
    probabilities = [count / total for count in counts.values()]
    choice_entropy = entropy_bits(probabilities)
    uniform_entropy = math.log2(semantic_action_count) if semantic_action_count > 0 else None
    mean_score_deltas = {}
    for action, values in raw_deltas.items():
        numeric_values = [float(value) for value in values]
        mean_score_deltas[str(action)] = sum(numeric_values) / len(numeric_values) if numeric_values else None
    return {
        "total_proposals": total,
        "total_accepted": int(summary.get("total_accepted", 0)),
        "actions_used": len(counts),
        "proposal_counts": counts,
        "acceptance_rates": acceptance_rates,
        "mean_score_deltas": mean_score_deltas,
        "choice_entropy_bits": choice_entropy,
        "uniform_entropy_bits": uniform_entropy,
    }


def controller_statistics(
    action_payload: Mapping[str, Any],
    fallback_tau: float,
) -> dict[str, Any] | None:
    """Summarize verbalized or random Controller distributions.

    Args:
        action_payload: Parsed ``action_summary.json`` object containing either
            stateless selector history or ReAct V2 proposal records.
        fallback_tau: Tail threshold used only when an older record omitted its
            configured threshold.

    Returns:
        Distribution entropy, fallback, policy, sampling, and tail statistics,
        or ``None`` when the condition has no Controller history.

    Raises:
        ValueError: A stored history or probability distribution is malformed.
    """
    histories: list[Mapping[str, Any]] = []
    raw_selector_history = action_payload.get("verbalized_history", [])
    if not isinstance(raw_selector_history, list):
        raise ValueError("Verbalized selector history must be a list.")
    for history in raw_selector_history:
        if not isinstance(history, Mapping):
            raise ValueError("Every verbalized selector record must be a mapping.")
        histories.append(history)

    raw_proposal_records = action_payload.get("proposal_records", [])
    if not isinstance(raw_proposal_records, list):
        raise ValueError("Proposal records must be a list.")
    for record in raw_proposal_records:
        if not isinstance(record, Mapping):
            raise ValueError("Every proposal record must be a mapping.")
        controller_sampling = record.get("controller_sampling")
        if controller_sampling is not None:
            if not isinstance(controller_sampling, Mapping):
                raise ValueError("Controller sampling provenance must be a mapping.")
            histories.append(controller_sampling)

    if not histories:
        return None

    entropies = []
    uniform_entropies = []
    fallback_count = 0
    policy_counts: Counter[str] = Counter()
    sampled_counts: Counter[str] = Counter()
    tail_hits = 0
    tail_total = 0
    for history in histories:
        raw_probabilities = history.get("probs", {})
        sampled = history.get("sampled", [])
        if not isinstance(raw_probabilities, Mapping) or not isinstance(sampled, list):
            raise ValueError("Controller probabilities must be a mapping and sampled choices must be a list.")
        probabilities = {str(name): float(probability) for name, probability in raw_probabilities.items()}
        if probabilities:
            recorded_entropy = history.get("entropy_bits")
            entropy = float(recorded_entropy) if recorded_entropy is not None else entropy_bits(probabilities.values())
            entropies.append(entropy)
            uniform_entropies.append(math.log2(len(probabilities)))
        fallback_count += int(bool(history.get("fallback", False)))
        policy_counts[str(history.get("sampling_policy", "unknown"))] += 1
        sampled_counts.update(str(name) for name in sampled)

        if history.get("sampling_policy") == "tail":
            threshold = float(history.get("tau", fallback_tau))
            for name in sampled:
                tail_total += 1
                if probabilities.get(str(name), 1.0) < threshold:
                    tail_hits += 1

    return {
        "distribution_count": len(histories),
        "mean_distribution_entropy_bits": sum(entropies) / len(entropies) if entropies else None,
        "mean_uniform_entropy_bits": (sum(uniform_entropies) / len(uniform_entropies) if uniform_entropies else None),
        "fallback_rate": fallback_count / len(histories),
        "tail_sampling_rate": tail_hits / tail_total if tail_total else None,
        "policy_counts": dict(policy_counts),
        "sampled_choice_counts": dict(sampled_counts),
    }


def analyze_run(run_dir: Path, fallback_tau: float) -> dict[str, Any]:
    """Build a Till-style performance and mechanism report for one run.

    Args:
        run_dir: Completed HotPotQA run directory.
        fallback_tau: Tail threshold for legacy selector records.

    Returns:
        JSON-serializable run analysis.

    Raises:
        FileNotFoundError: A completion artifact is absent.
        ValueError: Contracts and result artifacts disagree or are malformed.
    """
    contract = load_json_object(run_dir / WIKIPEDIA_RUN_CONTRACT_FILENAME)
    candidates = load_json_object(run_dir / "candidates.json")
    final_metrics = load_json_object(run_dir / "final_metrics.json")
    action_payload = load_json_object(run_dir / "action_summary.json")

    if contract.get("benchmark") != "hotpotqa-fullwiki-wiki17":
        raise ValueError(f"Run {run_dir} is not a scientific HotPotQA run.")
    if contract.get("scientific_contract_enforced") is not True:
        raise ValueError(f"Run {run_dir} did not enforce the scientific contract.")
    embedded_contract = candidates.get("run_contract")
    if embedded_contract != contract:
        raise ValueError(f"Run contract mismatch in {run_dir}.")
    if action_payload.get("run_contract") != contract:
        raise ValueError(f"Action-summary contract mismatch in {run_dir}.")
    candidate_artifact_sha256 = hashlib.sha256((run_dir / "candidates.json").read_bytes()).hexdigest()
    if action_payload.get("candidate_artifact_sha256") != candidate_artifact_sha256:
        raise ValueError(f"Action summary does not match the candidate artifact in {run_dir}.")
    action_summary = action_payload.get("summary")
    required_action_summary_fields = {
        "action_proposal_counts",
        "action_acceptance_counts",
        "action_rejection_counts",
        "action_acceptance_rates",
        "action_score_deltas",
        "textual_diversity_per_iteration",
        "total_proposals",
        "total_accepted",
    }
    if (
        action_payload.get("schema_version") != 1
        or not isinstance(action_summary, Mapping)
        or not required_action_summary_fields.issubset(action_summary)
    ):
        raise ValueError(f"Run {run_dir} has an incomplete action summary.")
    condition = str(contract.get("condition", ""))
    if final_metrics.get("condition") != condition:
        raise ValueError(f"Condition mismatch in {run_dir}.")

    optimizer = contract.get("optimizer", {})
    models = contract.get("models", {})
    runtime = contract.get("execution_runtime", {})
    if not isinstance(optimizer, Mapping) or not isinstance(models, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError(f"Run {run_dir} has malformed optimizer, model, or runtime metadata.")
    max_metric_calls = int(optimizer.get("max_metric_calls", 0))
    if (max_metric_calls, condition) not in _APPROVED_CELLS:
        raise ValueError(f"Run {run_dir} is not one of the six approved campaign cells.")
    solver_model = str(models.get("solver", ""))
    reflection_model = str(models.get("reflection", ""))
    if solver_model not in _MODEL_LABELS or reflection_model != solver_model:
        raise ValueError(f"Run {run_dir} does not use an approved homogeneous model pair.")
    campaign_id = runtime.get("campaign_id")
    source_commit = runtime.get("source_commit")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError(f"Run {run_dir} lacks a campaign identity.")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError(f"Run {run_dir} lacks an exact source identity.")
    total_metric_calls = int(candidates.get("total_metric_calls", 0))
    if total_metric_calls < max_metric_calls:
        raise ValueError(f"Run {run_dir} stopped at {total_metric_calls} of {max_metric_calls} metric calls.")
    raw_candidates = candidates.get("candidates", [])
    validation_scores = candidates.get("val_aggregate_scores", [])
    if not isinstance(raw_candidates, list) or not isinstance(validation_scores, list):
        raise ValueError(f"Run {run_dir} has malformed candidates or validation scores.")
    if any(not isinstance(candidate, Mapping) for candidate in raw_candidates):
        raise ValueError(f"Run {run_dir} has a non-mapping prompt candidate.")
    best_idx = int(candidates.get("best_idx", -1))
    if best_idx < 0 or best_idx >= len(raw_candidates) or best_idx >= len(validation_scores):
        raise ValueError(f"Run {run_dir} has an invalid best-candidate index.")
    candidate_count = len(raw_candidates)
    if int(final_metrics.get("candidates_explored", -1)) != candidate_count:
        raise ValueError(f"Candidate count mismatch in {run_dir}.")
    best_candidate_sha256 = hashlib.sha256(
        json.dumps(
            raw_candidates[best_idx],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if final_metrics.get("candidate_sha256") != best_candidate_sha256:
        raise ValueError(f"Best-candidate identity mismatch in {run_dir}.")
    best_validation_score = float(validation_scores[best_idx])
    if not math.isclose(
        float(final_metrics.get("best_validation_exact_match", math.nan)),
        best_validation_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Best-validation score mismatch in {run_dir}.")

    heldout_summary = load_json_object(run_dir / "heldout" / best_candidate_sha256 / "summary.json")
    if heldout_summary.get("candidate_sha256") != best_candidate_sha256:
        raise ValueError(f"Held-out candidate identity mismatch in {run_dir}.")
    heldout_comparisons = (
        ("example_count", "test_example_count", int),
        ("exact_match", "test_exact_match", float),
        ("f1", "test_f1", float),
    )
    for heldout_field, final_field, converter in heldout_comparisons:
        heldout_value = converter(heldout_summary.get(heldout_field, math.nan))
        final_value = converter(final_metrics.get(final_field, math.nan))
        if heldout_value != final_value:
            raise ValueError(f"Held-out {heldout_field} mismatch in {run_dir}.")

    semantic_space = optimizer.get("semantic_action_space")
    semantic_action_count = 0
    if isinstance(semantic_space, Mapping):
        actions = semantic_space.get("actions", [])
        if isinstance(actions, list):
            semantic_action_count = len(actions)
    if condition != "vanilla" and semantic_action_count != 10:
        raise ValueError(f"Run {run_dir} does not contain the locked ten-action catalog.")

    proposal_records = action_payload.get("proposal_records", [])
    if not isinstance(proposal_records, list) or any(not isinstance(record, Mapping) for record in proposal_records):
        raise ValueError(f"Run {run_dir} has malformed proposal records.")
    if not proposal_records:
        raise ValueError(f"Run {run_dir} has no recorded proposal evidence.")
    completed_component_proposals = 0
    for record in proposal_records:
        texts_by_component = record.get("texts_by_component", {})
        if not isinstance(texts_by_component, Mapping):
            raise ValueError(f"Run {run_dir} has malformed component-scoped proposal text.")
        completed_component_proposals += len(texts_by_component)
    action_stats = action_statistics(action_payload, semantic_action_count)
    controller_stats = controller_statistics(action_payload, fallback_tau)
    if condition == "vanilla" and (action_stats is not None or controller_stats is not None):
        raise ValueError(f"Run {run_dir} attributes semantic actions to vanilla GEPA.")
    if condition != "vanilla" and action_stats is None:
        raise ValueError(f"Run {run_dir} has no semantic-action evidence.")
    if condition in {"react_v2", "react_v2_random", "action"} and controller_stats is None:
        raise ValueError(f"Run {run_dir} has no Controller evidence.")
    recorded_action_counts = Counter(str(record["action"]) for record in proposal_records if record.get("action"))
    if action_stats is not None and dict(recorded_action_counts) != action_stats["proposal_counts"]:
        raise ValueError(f"Run {run_dir} has inconsistent proposal action counts.")
    return {
        "run": run_dir.name,
        "path": str(run_dir.resolve()),
        "campaign_id": runtime.get("campaign_id"),
        "source_commit": runtime.get("source_commit"),
        "model": solver_model,
        "model_label": _MODEL_LABELS[solver_model],
        "condition": condition,
        "budget_profile": _BUDGET_LABELS.get(max_metric_calls, str(max_metric_calls)),
        "max_metric_calls": max_metric_calls,
        "total_metric_calls": total_metric_calls,
        "candidates_explored": candidate_count,
        "proposal_attempts_recorded": len(proposal_records),
        "completed_component_proposals": completed_component_proposals,
        "best_validation_exact_match": float(final_metrics["best_validation_exact_match"]),
        "test_exact_match": float(final_metrics["test_exact_match"]),
        "test_f1": float(final_metrics["test_f1"]),
        "test_example_count": int(final_metrics["test_example_count"]),
        "candidate_diversity": candidate_diversity(raw_candidates),
        "proposal_diversity": proposal_diversity(proposal_records),
        "action_stats": action_stats,
        "controller_stats": controller_stats,
    }


def discover_completed_runs(
    root: Path,
    fallback_tau: float,
    *,
    campaign_id: str | None = None,
    source_commit: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find and analyze completed HotPotQA runs below one artifact root.

    Args:
        root: Directory containing fetched run directories.
        fallback_tau: Tail threshold for legacy selector records.
        campaign_id: Optional campaign identity used to exclude other runs
            created from the same source tree.
        source_commit: Optional exact source identity used to exclude results
            from other revisions.

    Returns:
        Completed run reports and human-readable incomplete-run notices.
    """
    reports = []
    incomplete = []
    for contract_path in sorted(root.rglob(WIKIPEDIA_RUN_CONTRACT_FILENAME)):
        run_dir = contract_path.parent
        contract = load_json_object(contract_path)
        runtime = contract.get("execution_runtime", {})
        if not isinstance(runtime, Mapping):
            raise ValueError(f"Run {run_dir} has malformed execution metadata.")
        if campaign_id is not None and runtime.get("campaign_id") != campaign_id:
            continue
        if source_commit is not None and runtime.get("source_commit") != source_commit:
            continue
        missing = [
            name
            for name in ("candidates.json", "final_metrics.json", "action_summary.json")
            if not (run_dir / name).is_file()
        ]
        if missing:
            incomplete.append(f"{run_dir}: missing {', '.join(missing)}")
            continue
        reports.append(analyze_run(run_dir, fallback_tau))

    def report_order(report: Mapping[str, Any]) -> tuple[int, str, int, int, str]:
        """Sort Qwen before GLM, then standard before expanded and launcher order.

        Args:
            report: Run analysis containing model, budget, and condition labels.

        Returns:
            Stable tuple used to order printed and serialized runs.
        """
        model_label = str(report["model_label"])
        model_rank = {"Qwen3.8-27B": 0, "GLM-5.3-Flash": 1}.get(model_label, 2)
        budget_rank = {"standard": 0, "expanded": 1}.get(str(report["budget_profile"]), 2)
        condition_rank = _CONDITION_ORDER.get(str(report["condition"]), len(_CONDITION_ORDER))
        return model_rank, model_label, budget_rank, condition_rank, str(report["run"])

    reports.sort(key=report_order)
    seen_cells: set[tuple[str, int, str]] = set()
    for report in reports:
        cell = (str(report["model"]), int(report["max_metric_calls"]), str(report["condition"]))
        if cell in seen_cells:
            raise ValueError(f"Duplicate HotPotQA campaign cell: {cell!r}.")
        seen_cells.add(cell)
    return reports, incomplete


def render_markdown(reports: Sequence[Mapping[str, Any]]) -> str:
    """Render performance and mechanism tables for terminal review.

    Args:
        reports: Ordered completed-run analyses.

    Returns:
        Two GitHub-flavored Markdown tables.
    """
    lines = [
        "| model | tree | condition | calls | candidates | best val EM | test EM | test F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['model_label']} | {report['budget_profile']} | `{report['condition']}` "
            f"| {report['total_metric_calls']:,} | {report['candidates_explored']} "
            f"| {report['best_validation_exact_match']:.2%} | {report['test_exact_match']:.2%} "
            f"| {report['test_f1']:.2%} |"
        )

    lines.extend(
        [
            "",
            "| model | tree | condition | proposals | actions accepted/proposed | candidate Jaccard | proposal Jaccard | action H / uniform | controller H / uniform | fallback | tail rate |",
            "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for report in reports:
        action_stats = report.get("action_stats")
        controller_stats = report.get("controller_stats")
        if isinstance(action_stats, Mapping):
            action_total = f"{action_stats['total_accepted']}/{action_stats['total_proposals']}"
            choice_entropy = action_stats.get("choice_entropy_bits")
            uniform_entropy = action_stats.get("uniform_entropy_bits")
            action_entropy = (
                f"{float(choice_entropy):.2f}/{float(uniform_entropy):.2f}"
                if choice_entropy is not None and uniform_entropy is not None
                else "-"
            )
        else:
            action_total = "-"
            action_entropy = "-"
        if isinstance(controller_stats, Mapping):
            distribution_entropy = controller_stats.get("mean_distribution_entropy_bits")
            uniform_entropy = controller_stats.get("mean_uniform_entropy_bits")
            controller_entropy = (
                f"{float(distribution_entropy):.2f}/{float(uniform_entropy):.2f}"
                if distribution_entropy is not None and uniform_entropy is not None
                else "-"
            )
            fallback = f"{controller_stats['fallback_rate']:.1%}"
            tail_rate_value = controller_stats.get("tail_sampling_rate")
            tail_rate = f"{tail_rate_value:.1%}" if tail_rate_value is not None else "-"
        else:
            controller_entropy = "-"
            fallback = "-"
            tail_rate = "-"

        candidate_parts = []
        raw_candidate_diversity = report.get("candidate_diversity", {})
        if isinstance(raw_candidate_diversity, Mapping):
            for component, statistics in sorted(raw_candidate_diversity.items()):
                if isinstance(statistics, Mapping) and "mean_pairwise_jaccard_distance" in statistics:
                    candidate_parts.append(f"{component}={float(statistics['mean_pairwise_jaccard_distance']):.3f}")
        proposal_parts = [
            f"{component}={float(distance):.3f}"
            for component, distance in sorted(report.get("proposal_diversity", {}).items())
        ]
        lines.append(
            f"| {report['model_label']} | {report['budget_profile']} | `{report['condition']}` "
            f"| {report['completed_component_proposals']} | {action_total} "
            f"| {'; '.join(candidate_parts) or '-'} | {'; '.join(proposal_parts) or '-'} "
            f"| {action_entropy} | {controller_entropy} | {fallback} | {tail_rate} |"
        )
    return "\n".join(lines)


def write_campaign_analysis(
    reports: Sequence[Mapping[str, Any]],
    output_path: Path,
    analysis_source_commit: str,
) -> None:
    """Atomically write the combined machine-readable campaign analysis.

    Args:
        reports: Ordered completed-run analyses.
        output_path: Destination JSON file.
        analysis_source_commit: Exact Git commit containing the analyzer.

    Raises:
        ValueError: Reports contain more than one campaign or source revision,
            or the analyzer commit is not exact.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(analysis_source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in analysis_source_commit
    ):
        raise ValueError("Analysis source commit must be a full lowercase Git commit.")
    campaign_ids = {str(report["campaign_id"]) for report in reports}
    source_commits = {str(report["source_commit"]) for report in reports}
    if len(campaign_ids) > 1 or len(source_commits) > 1:
        raise ValueError("Campaign analysis cannot mix campaign IDs or source revisions.")
    payload = {
        "schema_version": 1,
        "analysis_source_commit": analysis_source_commit,
        "campaign_ids": sorted(campaign_ids),
        "source_commits": sorted(source_commits),
        "runs": list(reports),
    }
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.{os.getpid()}.part")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def main() -> None:
    """Analyze completed HotPotQA runs selected from command-line arguments.

    The command prints human-readable performance and mechanism tables and
    atomically writes the same evidence as a machine-readable JSON object.

    Raises:
        FileNotFoundError: A discovered completed run lacks a required artifact.
        ValueError: A result artifact is malformed or disagrees with its locked
            run contract.
    """
    parser = argparse.ArgumentParser(description="Analyze fetched HotPotQA campaign artifacts")
    parser.add_argument("root", type=Path, help="Root containing fetched HotPotQA run directories")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.10,
        help="Fallback tail threshold for legacy selector records that omitted tau",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Combined JSON output path (default: ROOT/hotpotqa_analysis.json)",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Analyze only runs with this locked campaign identity",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="Analyze only runs from this exact source commit",
    )
    parser.add_argument(
        "--analysis-source-commit",
        required=True,
        help="Git commit containing the analyzer used for this report",
    )
    args = parser.parse_args()
    output_path = args.output if args.output is not None else args.root / "hotpotqa_analysis.json"
    reports, incomplete = discover_completed_runs(
        args.root,
        args.tau,
        campaign_id=args.campaign_id,
        source_commit=args.source_commit,
    )
    if not reports:
        raise ValueError("No completed HotPotQA runs matched the requested campaign.")
    for notice in incomplete:
        print(f"Incomplete run: {notice}", file=sys.stderr)
    write_campaign_analysis(reports, output_path, analysis_source_commit=args.analysis_source_commit)
    print(render_markdown(reports))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
