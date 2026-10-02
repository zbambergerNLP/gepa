"""Build a local, offline study of candidate context and validation breakthroughs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from examples.hotpotqa.action_signal_data import import_manifest, mean
from examples.hotpotqa.action_signal_models import evaluate_signal

CHART_WIDTH = 880
CHART_HEIGHT = 280
CHART_MARGIN = 45
CHART_COLORS = ("#6d28d9", "#087e8b", "#c35c16", "#5f6f52")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a table with escaped text in every cell.

    Args:
        headers: Column labels.
        rows: Display values.

    Returns:
        HTML table containing no unescaped archive content.
    """
    heading = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-scroll'><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"


def validation_chart(lineages: dict[str, Any]) -> str:
    """Plot observed incumbent validation scores against logical optimization evaluations.

    Args:
        lineages: Descriptive evidence, including vanilla comparisons.

    Returns:
        Accessible inline SVG and a legend.
    """
    runs = [(name, run) for name, run in lineages.items() if run.get("trajectory")]
    if not runs:
        return "<p>No search trajectories were available.</p>"
    maximum = max(run["followup"]["end_evaluations"] for _, run in runs)
    width = CHART_WIDTH - 2 * CHART_MARGIN
    height = CHART_HEIGHT - 2 * CHART_MARGIN
    paths = []
    legend = []
    for index, (_name, run) in enumerate(runs):
        color = CHART_COLORS[index % len(CHART_COLORS)]
        best = run["trajectory"][0]["score"]
        points = []
        for point in run["trajectory"]:
            x = CHART_MARGIN + width * point["evaluations"] / maximum
            points.append(f"{x:.2f},{CHART_HEIGHT - CHART_MARGIN - height * best:.2f}")
            best = max(best, point["score"])
            points.append(f"{x:.2f},{CHART_HEIGHT - CHART_MARGIN - height * best:.2f}")
        x = CHART_MARGIN + width * run["followup"]["end_evaluations"] / maximum
        points.append(f"{x:.2f},{CHART_HEIGHT - CHART_MARGIN - height * best:.2f}")
        paths.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        legend.append(f"<span style='color:{color}'>● {html.escape(run['cohort'])} ({best:.2%})</span>")
    axes = f"<path d='M{CHART_MARGIN} {CHART_MARGIN}V{CHART_HEIGHT - CHART_MARGIN}H{CHART_WIDTH - CHART_MARGIN}' fill='none' stroke='#8a94a3'/>"
    labels = (
        f"<text x='5' y='{CHART_MARGIN}'>100%</text><text x='18' y='{CHART_HEIGHT - CHART_MARGIN}'>0%</text>"
        f"<text x='{CHART_MARGIN}' y='{CHART_HEIGHT - 20}'>0</text>"
        f"<text text-anchor='end' x='{CHART_WIDTH - CHART_MARGIN}' y='{CHART_HEIGHT - 20}'>{maximum:,}</text>"
        f"<text text-anchor='middle' x='{CHART_WIDTH // 2}' y='{CHART_HEIGHT - 4}'>Logical optimization evaluations</text>"
    )
    return (
        f"<svg viewBox='0 0 {CHART_WIDTH} {CHART_HEIGHT}' role='img' aria-label='Observed best validation score by optimization evaluations'>"
        f"{axes}{''.join(paths)}{labels}</svg><div class='legend'>{''.join(legend)}</div>"
    )


def render_report(dataset: dict[str, Any], results: dict[str, Any]) -> str:
    """Render the evidence and its limitations without publishing source prompts.

    Args:
        dataset: Imported decisions and provenance audit.
        results: Prespecified held-out predictive evaluations.

    Returns:
        Standalone HTML report with no network assets or scripts.
    """
    primary = [row for row in dataset["decisions"] if row["kind"] == "search"]
    events = sum(row["outcome"]["record_break"] or 0 for row in primary)
    coverage = []
    for run in dataset["lineages"].values():
        counts = run["counts"]
        coverage.append(
            [
                run["cohort"],
                run["kind"],
                counts["decisions"],
                counts.get("validated_children", "—"),
                counts.get("record_improvements", "—"),
            ]
        )
    metrics = []
    for result in results["folds"]:
        score = result["metrics"]
        metrics.append(
            [
                result["fold"],
                result["model"],
                f"{score['events']}/{score['count']}",
                f"{score['log_loss']:.4f}",
                f"{score['brier_score']:.4f}",
                f"{score['average_precision']:.4f}"
                if score["average_precision"] is not None
                else "Unavailable: one class",
            ]
        )
    differences = [
        [
            row["fold"],
            row["events"],
            f"{row['stage_log_loss_minus_action_rates']:+.4f}",
            f"{row['action_effects_log_loss_minus_stage']:+.4f}",
            f"{row['interaction_log_loss_minus_action_effects']:+.4f}",
            f"{row['interaction_log_loss_minus_stage']:+.4f}",
        ]
        for row in results["comparisons"]
    ]
    pilots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset["decisions"]:
        if row["kind"] == "pilot":
            pilots[row["cohort"]].append(row)
    pilot_rows = []
    for cohort, rows in pilots.items():
        transfer = mean([row["outcome"]["transfer"]["delta"] for row in rows])
        pilot_rows.append(
            [
                cohort,
                len(rows),
                rows[0]["controller_backend"],
                rows[0]["novelty_backend"] or "None",
                f"{transfer:+.3f}" if transfer is not None else "Missing",
            ]
        )
    ancestry = []
    for lineage, run in dataset["lineages"].items():
        for event in run.get("delayed_records", []):
            ancestors = ", ".join(
                f"{a['candidate']} ({a['evaluations_until_record']:,} evals)" for a in event["ancestors"]
            )
            ancestry.append([lineage, event["record_candidate"], event["evaluations"], ancestors])
    missing = table(
        ["Unavailable evidence", "Decisions"], [[key, value] for key, value in dataset["audit"]["missingness"].items()]
    )
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in results["limitations"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FOREST plateau escape: candidate-context signal study</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;color:#203047;background:#f6f8fc;margin:0}}main{{max-width:1100px;margin:auto;padding:44px 24px}}
h1{{font-size:34px;line-height:1.2;letter-spacing:-.8px}}h2{{margin-top:40px;font-size:23px}}p{{max-width:900px}}.eyebrow{{font-size:13px;color:#63588b}}
.finding{{background:#eee9fa;border-left:4px solid #6d28d9;padding:16px 20px;margin:24px 0}}.table-scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #dde3ee;text-align:left;vertical-align:top}}th{{background:#edf0f6}}
svg{{width:100%;background:white;border-radius:12px}}svg text{{font:12px system-ui,sans-serif;fill:#425168}}.legend{{display:flex;gap:24px;flex-wrap:wrap;font-size:14px;margin:12px 0}}
details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}a{{color:#5631a7}}code{{overflow-wrap:anywhere}}footer{{margin-top:40px;color:#667085;font-size:13px}}
</style></head><body><main>
<div class="eyebrow">Offline study · observed historical decisions · no inference</div>
<h1>Can candidate context help FOREST escape a plateau?</h1>
<p>The objective is a higher best validation score at a fixed optimization budget, followed by a frozen winner's test evaluation. Diversity, training improvement, and token savings receive no independent reward.</p>
<div class="finding"><strong>{len(primary):,} unique search decisions · {events} validation records</strong><br>{html.escape(results["assessment"])}</div>
<h2>Evidence across runs</h2>
{table(["Cohort", "Use", "Decisions", "Validated children", "New validation records"], coverage)}
<p>Continuation histories count once. The importer removed {dataset["audit"]["duplicate_decisions_removed"]} repeated decision records. Search lineages and matched pilot opportunities are the relevant groups, not individual questions or copied allocations.</p>
<h2>Observed plateaus</h2>{validation_chart(dataset["lineages"])}
<p>These curves describe searches that actually ran. They are not simulated outcomes of a learned controller. Early validation records and post-standard-budget results must be interpreted separately.</p>
<h2>Does candidate-action context add predictive signal?</h2>
{table(["Held-out fold", "Breakthroughs", "Candidate/stage minus action averages", "Action effects minus candidate/stage", "Interactions minus action effects", "Interactions minus candidate/stage"], differences)}
<p>Each difference is a change in log loss; negative values favor the more detailed model named first. A lower loss on a fold with no breakthroughs can reflect better prediction of failure; it cannot demonstrate plateau escape. Models use fixed regularization and training-only preprocessing. Complete-lineage tests measure transfer separately from forward prediction within an existing search.</p>
<details><summary>All model metrics</summary>{table(["Fold", "Model", "Events / decisions", "Log loss ↓", "Brier score ↓", "Average precision ↑"], metrics)}</details>
<h2>Supporting pilot transfer evidence</h2>
{table(["Pilot arm", "Decisions", "Action controller", "Novelty verifier", "Mean transfer delta"], pilot_rows)}
<p>These matched training diagnostics have no full-validation breakthrough label and never enter the primary predictor. No-op reuse and protocol differences remain in the normalized records. Arms sharing an opportunity are paired evidence, not independent replicates.</p>
<details><summary>Later records and their ancestry</summary>{table(["Lineage", "Record candidate", "Evaluations at result", "Ancestors and elapsed evaluations"], ancestry)}
<p>Ancestry describes observed descendants, not causal credit. Follow-up ends at each archive's last evaluation; unexplored branches are not evidence that an action cannot work. Each breakthrough is represented once.</p></details>
<details><summary>Missingness and interpretation limits</summary>{missing}<ul>{limitations}</ul></details>
<h2>Reproduce and inspect</h2>
<p><a href="audit.json">Input audit</a> · <a href="decisions.jsonl">Normalized decisions</a> · <a href="results.json">Metrics, calibration, and predictions</a> · <a href="lineages.json">Trajectories and ancestry</a></p>
<p>Method references: <a href="https://proceedings.mlr.press/v28/agrawal13.html">contextual Thompson sampling</a>, <a href="https://arxiv.org/abs/2606.23933">flow-corrected Thompson sampling</a>, and <a href="https://proceedings.mlr.press/v26/li12a.html">offline bandit evaluation</a>. This study fits predictors; it implements neither a bandit policy nor reward transport.</p>
<footer>Manifest SHA-256: <code>{dataset["manifest_sha256"]}</code><br>Test responses were not read. No prompts, providers, controllers, or jobs were changed.</footer>
</main></body></html>"""


def run_study(manifest: Path, output: Path) -> dict[str, Any]:
    """Validate inputs, fit the fixed study, and write only to a separate output directory.

    Args:
        manifest: Explicit archive manifest.
        output: Destination outside every archive root.

    Returns:
        Compact coverage and interpretation summary.

    Raises:
        ValueError: The output would modify an input archive.
    """
    specification = json.loads(manifest.read_text())
    for entry in specification["archives"]:
        root = (manifest.parent / entry["root"]).resolve()
        if output.resolve().is_relative_to(root):
            raise ValueError("Analysis output must be outside every input archive")
    dataset = import_manifest(manifest)
    results = evaluate_signal(dataset["decisions"])
    results["implementation"] = {
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("analyze_action_signal.py", "action_signal_data.py", "action_signal_models.py")
        },
        "dependencies": {name: version(name) for name in ("numpy", "scikit-learn")},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("audit.json", {"manifest_sha256": dataset["manifest_sha256"], **dataset["audit"]}),
        ("results.json", results),
        ("lineages.json", dataset["lineages"]),
    ):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in dataset["decisions"])
    )
    (output / "report.html").write_text(render_report(dataset, results))
    return {
        "decisions_by_kind": dict(Counter(row["kind"] for row in dataset["decisions"])),
        "duplicates_removed": dataset["audit"]["duplicate_decisions_removed"],
        "model_evaluations": len(results["folds"]),
        "assessment": results["assessment"],
        "report": str(output / "report.html"),
    }


def main() -> None:
    """Run the offline study with explicit manifest and output arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_study(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
