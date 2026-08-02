"""Analyze action-choice diversity and proposal diversity across IFBench runs.

Runs locally on fetched run artifacts (candidates.json, action_summary.json,
run_log.txt per run dir). Reports, per run:

- choice entropy over sampled actions (bits) vs uniform entropy
- mean verbalized distribution entropy and fallback rate (verbalized runs)
- tail-sampling rate (fraction of sampled actions with p < tau in their distribution)
- per-action proposal counts and acceptance rates
- all-proposal Jaccard diversity (from run_log.txt)

Usage:
    uv run python examples/ifbench/analyze_actions.py outputs/della_rev2 [--tau 0.10]

Writes action_analysis.json next to each run dir's artifacts and prints a
markdown summary.
"""

import argparse
import itertools
import json
import math
import os
import re
from collections import Counter


def entropy_bits(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def parse_proposals(run_log_path: str) -> list[dict]:
    with open(run_log_path) as f:
        text = f.read()
    pattern = re.compile(r"Iteration (\d+): Proposed new text for (\w+): (.*?)(?=\nIteration \d+:|\Z)", re.DOTALL)
    return [
        {"iter": int(m.group(1)), "component": m.group(2), "text": m.group(3).strip()} for m in pattern.finditer(text)
    ]


def jaccard_diversity(texts: list[str]) -> float:
    sets = [set(t.lower().split()) for t in texts]
    dists = [1 - len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    return sum(dists) / len(dists) if dists else 0.0


def analyze_run(run_dir: str, tau: float) -> dict | None:
    candidates_path = os.path.join(run_dir, "candidates.json")
    if not os.path.exists(candidates_path):
        return None
    with open(candidates_path) as f:
        cands = json.load(f)

    report: dict = {
        "run": os.path.basename(run_dir),
        "num_candidates": len(cands["candidates"]),
        "best_val": max(cands["val_aggregate_scores"]),
        "total_metric_calls": cands.get("total_metric_calls"),
    }

    # Accepted-candidate diversity per component.
    report["candidate_diversity"] = {
        comp: round(jaccard_diversity([c[comp] for c in cands["candidates"]]), 4) for comp in cands["candidates"][0]
    }

    # All-proposal diversity from the run log.
    run_log = os.path.join(run_dir, "run_log.txt")
    if os.path.exists(run_log):
        proposals = parse_proposals(run_log)
        report["num_proposals"] = len(proposals)
        report["proposal_diversity"] = {
            comp: round(jaccard_diversity([p["text"] for p in proposals if p["component"] == comp]), 4)
            for comp in sorted({p["component"] for p in proposals})
        }

    # Action stats (random / verbalized runs only).
    summary_path = os.path.join(run_dir, "action_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            action_data = json.load(f)
        summary = action_data["summary"]
        counts = summary.get("action_proposal_counts", {})
        total = sum(counts.values())
        if total:
            n_actions = len(counts)
            choice_probs = [c / total for c in counts.values()]
            report["action_stats"] = {
                "total_proposals": summary.get("total_proposals"),
                "total_accepted": summary.get("total_accepted"),
                "actions_used": n_actions,
                "choice_entropy_bits": round(entropy_bits(choice_probs), 3),
                "acceptance_rates": summary.get("action_acceptance_rates", {}),
                "proposal_counts": counts,
            }

        history = action_data.get("verbalized_history") or []
        if history:
            dist_entropies = [entropy_bits(list(h["probs"].values())) for h in history if h["probs"]]
            n_menu = max((len(h["probs"]) for h in history), default=0)
            tail_hits = 0
            tail_total = 0
            for h in history:
                for name in h["sampled"]:
                    tail_total += 1
                    if h["probs"].get(name, 1.0) < tau:
                        tail_hits += 1
            sampled_all = Counter(name for h in history for name in h["sampled"])
            report["verbalized_stats"] = {
                "num_distributions": len(history),
                "fallback_rate": round(sum(1 for h in history if h["fallback"]) / len(history), 3),
                "mean_distribution_entropy_bits": round(sum(dist_entropies) / len(dist_entropies), 3)
                if dist_entropies
                else None,
                "uniform_entropy_bits": round(math.log2(n_menu), 3) if n_menu else None,
                "tail_sampling_rate": round(tail_hits / tail_total, 3) if tail_total else None,
                "sampled_action_counts": dict(sampled_all),
            }

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze IFBench action/proposal diversity")
    parser.add_argument("root", help="Directory containing run dirs (each with candidates.json)")
    parser.add_argument("--tau", type=float, default=0.10, help="Tail threshold used by the verbalized selector")
    args = parser.parse_args()

    reports = []
    for name in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, name)
        if not os.path.isdir(run_dir):
            continue
        report = analyze_run(run_dir, args.tau)
        if report is None:
            continue
        reports.append(report)
        out_path = os.path.join(run_dir, "action_analysis.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    # Markdown summary.
    print(
        "| run | cands | best val | proposals | proposal jaccard | choice H (bits) | dist H / uniform | tail rate | fallback |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        pj = ", ".join(f"{k}={v}" for k, v in r.get("proposal_diversity", {}).items())
        astats = r.get("action_stats", {})
        vstats = r.get("verbalized_stats", {})
        dist_h = (
            f"{vstats.get('mean_distribution_entropy_bits')} / {vstats.get('uniform_entropy_bits')}" if vstats else "-"
        )
        print(
            f"| {r['run']} | {r['num_candidates']} | {r['best_val']:.3f} | {r.get('num_proposals', '-')} "
            f"| {pj} | {astats.get('choice_entropy_bits', '-')} | {dist_h} "
            f"| {vstats.get('tail_sampling_rate', '-')} | {vstats.get('fallback_rate', '-')} |"
        )


if __name__ == "__main__":
    main()
