"""Tests for the fetched HotPotQA campaign analyzer."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.hotpotqa.analyze_results import (
    analyze_run,
    discover_completed_runs,
    entropy_bits,
    jaccard_diversity,
    render_markdown,
    write_campaign_analysis,
)


def write_json(path: Path, payload: dict) -> None:
    """Write one deterministic JSON fixture.

    Args:
        path: Fixture file to create.
        payload: JSON object written to the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_completed_run(
    root: Path,
    *,
    condition: str = "react_v2",
    model: str = "hosted_vllm/Qwen/Qwen3.8-27B",
    budget: int = 6_871,
) -> Path:
    """Create one internally consistent completed-run fixture.

    Args:
        root: Parent directory for the run.
        condition: Scientific condition stored in every artifact.
        model: Homogeneous solver and proposer model identifier.
        budget: Requested metric-call budget.

    Returns:
        Created run directory.
    """
    run_dir = root / f"{model.rsplit('/', 1)[-1]}-{budget}-{condition}"
    semantic_actions = [{"name": f"action-{index}"} for index in range(10)]
    contract = {
        "benchmark": "hotpotqa-fullwiki-wiki17",
        "scientific_contract_enforced": True,
        "condition": condition,
        "models": {"solver": model, "reflection": model},
        "optimizer": {
            "max_metric_calls": budget,
            "semantic_action_space": {"actions": semantic_actions},
        },
        "execution_runtime": {
            "campaign_id": "hotpotqa-final-v1",
            "source_commit": "a" * 40,
        },
    }
    candidate_values = [
        {"summarize1": "seed text"},
        {"summarize1": "alpha beta"},
        {"summarize1": "alpha gamma"},
    ]
    best_candidate_sha256 = hashlib.sha256(
        json.dumps(
            candidate_values[2],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    candidates = {
        "run_contract": contract,
        "best_idx": 2,
        "total_metric_calls": budget + 2,
        "num_full_val_evals": 3,
        "candidates": candidate_values,
        "parents": [None, 0, 1],
        "val_aggregate_scores": [0.2, 0.4, 0.5],
        "discovery_eval_counts": [0, 3, 3],
    }
    final_metrics = {
        "schema_version": 1,
        "condition": condition,
        "candidate_sha256": best_candidate_sha256,
        "candidates_explored": 3,
        "best_validation_exact_match": 0.5,
        "test_exact_match": 0.4,
        "test_f1": 0.55,
        "test_example_count": 300,
        "diversity": {
            "summarize1": {
                "mean_pairwise_jaccard_distance": 0.6,
                "num_unique_texts": 3.0,
            }
        },
    }
    action_summary = {
        "summary": {
            "action_proposal_counts": {"reexpress": 2},
            "action_acceptance_counts": {"reexpress": 1},
            "action_rejection_counts": {"reexpress": 1},
            "action_acceptance_rates": {"reexpress": 0.5},
            "action_score_deltas": {"reexpress": [0.2, -0.1]},
            "textual_diversity_per_iteration": {},
            "total_proposals": 2,
            "total_accepted": 1,
        },
        "proposal_records": [
            {
                "iteration": 1,
                "action": "reexpress",
                "texts_by_component": {"summarize1": "alpha beta"},
                "controller_sampling": {
                    "probs": {"choice-a": 0.75, "choice-b": 0.25},
                    "sampled": ["choice-a"],
                    "fallback": False,
                    "tau": 0.2,
                    "sampling_policy": "positive_support_uniform_mixture",
                    "entropy_bits": entropy_bits([0.75, 0.25]),
                },
            },
            {
                "iteration": 2,
                "action": "reexpress",
                "texts_by_component": {"summarize1": "alpha gamma"},
                "controller_sampling": {
                    "probs": {"choice-a": 0.5, "choice-b": 0.5},
                    "sampled": ["choice-b"],
                    "fallback": False,
                    "tau": 0.2,
                    "sampling_policy": "positive_support_uniform_mixture",
                    "entropy_bits": 1.0,
                },
            },
        ],
    }
    if condition == "vanilla":
        action_summary["summary"] = {
            "action_proposal_counts": {},
            "action_acceptance_counts": {},
            "action_rejection_counts": {},
            "action_acceptance_rates": {},
            "action_score_deltas": {},
            "textual_diversity_per_iteration": {},
            "total_proposals": 0,
            "total_accepted": 0,
        }
        action_summary["proposal_records"] = [
            {
                "iteration": 1,
                "action": None,
                "texts_by_component": {"summarize1": "alpha beta"},
            },
            {
                "iteration": 2,
                "action": None,
                "texts_by_component": {"summarize1": "alpha gamma"},
            },
        ]
    write_json(run_dir / "wikipedia-run-contract.json", contract)
    candidate_artifact_path = run_dir / "candidates.json"
    write_json(candidate_artifact_path, candidates)
    write_json(run_dir / "final_metrics.json", final_metrics)
    action_summary["schema_version"] = 1
    action_summary["run_contract"] = contract
    action_summary["candidate_artifact_sha256"] = hashlib.sha256(candidate_artifact_path.read_bytes()).hexdigest()
    write_json(run_dir / "action_summary.json", action_summary)
    write_json(
        run_dir / "heldout" / final_metrics["candidate_sha256"] / "summary.json",
        {
            "schema_version": 1,
            "candidate_sha256": final_metrics["candidate_sha256"],
            "example_count": final_metrics["test_example_count"],
            "exact_match": final_metrics["test_exact_match"],
            "f1": final_metrics["test_f1"],
        },
    )
    return run_dir


def test_entropy_and_jaccard_match_till_metrics() -> None:
    """Preserve Till's Shannon entropy and token-set Jaccard definitions."""
    assert entropy_bits([0.5, 0.5]) == pytest.approx(1.0)
    assert jaccard_diversity(["alpha beta", "alpha gamma"]) == pytest.approx(2 / 3)
    assert jaccard_diversity(["only one text"]) is None


def test_analyze_run_combines_performance_and_react_controller_evidence(tmp_path: Path) -> None:
    """Join final scores with proposal, action, and Controller diagnostics.

    Args:
        tmp_path: Isolated artifact root.
    """
    run_dir = create_completed_run(tmp_path)

    report = analyze_run(run_dir, fallback_tau=0.1)

    assert report["budget_profile"] == "standard"
    assert report["total_metric_calls"] == 6_873
    assert report["test_exact_match"] == pytest.approx(0.4)
    assert report["proposal_attempts_recorded"] == 2
    assert report["completed_component_proposals"] == 2
    assert report["candidate_diversity"]["summarize1"]["mean_pairwise_jaccard_distance"] == pytest.approx(8 / 9)
    assert report["proposal_diversity"]["summarize1"] == pytest.approx(2 / 3)
    assert report["action_stats"]["total_accepted"] == 1
    assert report["action_stats"]["actions_used"] == 1
    assert report["action_stats"]["uniform_entropy_bits"] == pytest.approx(math.log2(10))
    assert report["controller_stats"]["distribution_count"] == 2
    assert report["controller_stats"]["fallback_rate"] == pytest.approx(0.0)


def test_discovery_reports_incomplete_runs_and_orders_the_campaign(tmp_path: Path) -> None:
    """List incomplete directories while keeping launcher-compatible ordering.

    Args:
        tmp_path: Isolated artifact root.
    """
    create_completed_run(tmp_path, condition="react_v2", model="hosted_vllm/zai-org/GLM-5.3-Flash")
    create_completed_run(tmp_path, condition="vanilla")
    incomplete_dir = tmp_path / "incomplete"
    write_json(
        incomplete_dir / "wikipedia-run-contract.json",
        {"benchmark": "hotpotqa-fullwiki-wiki17"},
    )

    reports, incomplete = discover_completed_runs(tmp_path, fallback_tau=0.1)

    assert [(report["model_label"], report["condition"]) for report in reports] == [
        ("Qwen3.8-27B", "vanilla"),
        ("GLM-5.3-Flash", "react_v2"),
    ]
    assert incomplete == [f"{incomplete_dir}: missing candidates.json, final_metrics.json, action_summary.json"]


def test_discovery_filters_other_campaigns_from_a_reused_source_tree(tmp_path: Path) -> None:
    """Select only the requested campaign when one source tree holds several.

    Args:
        tmp_path: Isolated artifact root.
    """
    selected_run = create_completed_run(tmp_path / "selected")
    other_run = create_completed_run(tmp_path / "other", condition="action")
    other_contract_path = other_run / "wikipedia-run-contract.json"
    other_contract = json.loads(other_contract_path.read_text(encoding="utf-8"))
    other_contract["execution_runtime"]["campaign_id"] = "earlier-smoke"
    write_json(other_contract_path, other_contract)
    candidates_path = other_run / "candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates["run_contract"] = other_contract
    write_json(candidates_path, candidates)

    reports, incomplete = discover_completed_runs(
        tmp_path,
        fallback_tau=0.1,
        campaign_id="hotpotqa-final-v1",
        source_commit="a" * 40,
    )

    assert [report["path"] for report in reports] == [str(selected_run.resolve())]
    assert incomplete == []


def test_markdown_and_json_present_the_same_completed_runs(tmp_path: Path) -> None:
    """Produce a readable table and one atomic machine-readable report.

    Args:
        tmp_path: Isolated artifact and output root.
    """
    run_dir = create_completed_run(tmp_path)
    report = analyze_run(run_dir, fallback_tau=0.1)
    markdown = render_markdown([report])
    output_path = tmp_path / "hotpotqa_analysis.json"

    write_campaign_analysis([report], output_path, analysis_source_commit="c" * 40)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "| Qwen3.8-27B | standard | `react_v2`" in markdown
    assert "| Qwen3.8-27B | standard | `react_v2` | 2 | 1/2 |" in markdown
    assert payload["runs"][0]["test_f1"] == pytest.approx(0.55)
    assert payload["analysis_source_commit"] == "c" * 40
    assert not list(tmp_path.glob("*.part"))


def test_analyzer_rejects_cross_artifact_condition_mismatch(tmp_path: Path) -> None:
    """Fail rather than compare mislabeled result artifacts.

    Args:
        tmp_path: Isolated artifact root.
    """
    run_dir = create_completed_run(tmp_path)
    final_metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    final_metrics["condition"] = "vanilla"
    write_json(run_dir / "final_metrics.json", final_metrics)

    with pytest.raises(ValueError, match="Condition mismatch"):
        analyze_run(run_dir, fallback_tau=0.1)


def test_analyzer_rejects_stale_winner_metrics(tmp_path: Path) -> None:
    """Reject final metrics belonging to a different winning prompt.

    Args:
        tmp_path: Isolated artifact root.
    """
    run_dir = create_completed_run(tmp_path)
    final_metrics_path = run_dir / "final_metrics.json"
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    final_metrics["candidate_sha256"] = "b" * 64
    write_json(final_metrics_path, final_metrics)

    with pytest.raises(ValueError, match="Best-candidate identity mismatch"):
        analyze_run(run_dir, fallback_tau=0.1)


def test_analyzer_rejects_empty_mechanism_evidence(tmp_path: Path) -> None:
    """Reject a completed run whose action artifact contains no proposals.

    Args:
        tmp_path: Isolated artifact root.
    """
    run_dir = create_completed_run(tmp_path)
    action_path = run_dir / "action_summary.json"
    action_payload = json.loads(action_path.read_text(encoding="utf-8"))
    action_payload["proposal_records"] = []
    write_json(action_path, action_payload)

    with pytest.raises(ValueError, match="no recorded proposal evidence"):
        analyze_run(run_dir, fallback_tau=0.1)


def test_analyzer_rejects_incomplete_action_summary(tmp_path: Path) -> None:
    """Reject an action artifact missing its aggregate mechanism fields.

    Args:
        tmp_path: Isolated artifact root.
    """
    run_dir = create_completed_run(tmp_path)
    action_path = run_dir / "action_summary.json"
    action_payload = json.loads(action_path.read_text(encoding="utf-8"))
    action_payload["summary"] = {}
    write_json(action_path, action_payload)

    with pytest.raises(ValueError, match="incomplete action summary"):
        analyze_run(run_dir, fallback_tau=0.1)


def test_discovery_rejects_duplicate_campaign_cells(tmp_path: Path) -> None:
    """Reject two result directories claiming the same scientific cell.

    Args:
        tmp_path: Isolated artifact root.
    """
    create_completed_run(tmp_path / "first")
    create_completed_run(tmp_path / "second")

    with pytest.raises(ValueError, match="Duplicate HotPotQA campaign cell"):
        discover_completed_runs(tmp_path, fallback_tau=0.1)


def test_campaign_writer_rejects_mixed_campaigns(tmp_path: Path) -> None:
    """Refuse to publish one report that pools independent campaigns.

    Args:
        tmp_path: Isolated artifact root.
    """
    first = analyze_run(create_completed_run(tmp_path / "first"), fallback_tau=0.1)
    second = dict(first)
    second["campaign_id"] = "earlier-smoke"

    with pytest.raises(ValueError, match="cannot mix"):
        write_campaign_analysis([first, second], tmp_path / "analysis.json", "c" * 40)
