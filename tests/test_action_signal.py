"""Exercise archive reconciliation, temporal leakage barriers, and offline signal evaluation."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import pickle
import socket
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.hotpotqa.action_signal_data import (
    PARENT_JOURNAL,
    Archive,
    DataOnlyUnpickler,
    contract_identity,
    import_manifest,
    load_search,
)
from examples.hotpotqa.action_signal_models import (
    MODEL_NAMES,
    evaluate_signal,
    make_folds,
    predict_fold,
    score_predictions,
)
from examples.hotpotqa.analyze_action_signal import run_study
from gepa.core.adapter import EvaluationBatch


def write_json(path: Path, value: object) -> None:
    """Write a deterministic synthetic archive artifact.

    Args:
        path: Destination file.
        value: JSON-compatible synthetic evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def refresh_hashes(entry: dict) -> None:
    """Refresh a fixture manifest after intentionally changing its evidence.

    Args:
        entry: Mutable synthetic manifest entry.
    """
    root = Path(entry["root"])
    entry["files"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in entry["files"]}


def search_fixture(root: Path, *, lineage: str = "first", length: int = 6) -> dict:
    """Create a serial search with records, a weak bridge, ties, losses, and a discard.

    Args:
        root: Synthetic archive directory.
        lineage: Independent search identity.
        length: Number of chronological decisions to retain.

    Returns:
        Checksum-bound manifest entry.
    """
    root.mkdir(parents=True, exist_ok=True)
    contract = {
        "optimizer": {
            "component_selector": "round_robin",
            "acceptance_criterion": "strict_improvement",
            "validation_evaluation": "full_eval",
            "merge": None,
            "reflection_minibatch_size": 2,
            "proposal_sampling_strategy": {"parents_per_iteration": 1, "mutations_per_parent": 1},
            "template_family": "fixture",
            "semantic_action_space": {"version": 1, "actions": ["x", "y"]},
        },
        "data": {
            "splits": {
                split: {"count": count, "sha256": split} for split, count in (("train", 10), ("val", 4), ("test", 8))
            }
        },
        "models": {"solver": "fixture-solver", "reflection": "fixture-editor"},
        "retrieval": {"corpus": "fixture"},
        "execution_runtime": {"source_commit": "fixture-source"},
    }
    programs = [{"a": "## Objective\nSeed a", "b": "## Objective\nSeed b"}]
    vectors = [{0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}]
    parents = [[None]]
    anchors = ["seed"]
    discovery = [0]
    cursors = {0: 0}
    trace = []
    records = []
    clock = 4
    schedule = [
        (0, [0, 0], [1, 0], [1, 1, 1, 0]),
        (0, [1, 0], [1, 0], None),
        (1, [0, 0], [1, 0], [1, 0, 0, 1]),
        (1, [1, 0], None, None),
        (2, [0, 0], [1, 1], [1, 1, 1, 1]),
        (0, [1, 1], [0, 1], None),
    ]
    journal_path = root / PARENT_JOURNAL
    journal_path.parent.mkdir()
    connection = sqlite3.connect(journal_path)
    connection.execute(
        "CREATE TABLE responses (scope TEXT, namespace TEXT, ordinal INTEGER, request_sha256 TEXT, response_json TEXT, response_sha256 TEXT)"
    )
    for index, (parent, before, after, validation) in enumerate(schedule[:length]):
        clock += 2
        module = ("a", "b")[cursors[parent]]
        cursors[parent] = (cursors[parent] + 1) % 2
        task = {"parent_idx": parent, "subsample_ids": [index, index + 1]}
        row = {"i": index, "iteration_id": f"it-{index}", "selected_program_candidate": parent, "tasks": [task]}
        if after is not None:
            task.update(subsample_scores=before, new_subsample_scores=after)
            clock += 2
        if validation is not None:
            child = len(programs)
            programs.append({**programs[parent], module: f"## Objective\nChild {child}"})
            vectors.append(dict(enumerate(map(float, validation))))
            parents.append([parent])
            anchors.append(row["iteration_id"])
            discovery.append(clock)
            clock += 4
            row["new_program_idx"] = child
            cursors[child] = cursors[parent]
        trace.append(row)
        action = "x" if index % 2 else "y"
        choice = f"{action}@Objective/REPLACE_TEXT"
        records.append(
            {
                "iteration": index + 1,
                "action": action,
                "semantic_action": action,
                "action_choice": choice,
                "action_target_section": "Objective",
                "action_operator": "REPLACE_TEXT",
                "texts_by_component": {module: "proposal text"} if after is not None else {},
                "controller_sampling": {
                    "sampling_probs": {"x@Objective/REPLACE_TEXT": 0.5, "y@Objective/REPLACE_TEXT": 0.5},
                    "policy": "fixture-policy",
                },
            }
        )
        batches = [EvaluationBatch(outputs=[{}, {}], scores=list(map(float, before)), trajectories=None)]
        payload = {
            "kind": "evaluation_batch",
            "schema_version": 1,
            "data": base64.b64encode(pickle.dumps((batches, None))).decode(),
        }
        raw = json.dumps(payload)
        connection.execute(
            "INSERT INTO responses VALUES (?, 'parents', 0, 'fixture', ?, ?)",
            (f"optimizer-iteration-{index}", raw, hashlib.sha256(raw.encode()).hexdigest()),
        )
    # A selected SQL query must never deserialize this held-out response.
    connection.execute("INSERT INTO responses VALUES ('heldout-test', 'parents', 0, 'test', 'FORBIDDEN', 'invalid')")
    connection.commit()
    connection.close()
    state = {
        "program_candidates": programs,
        "prog_candidate_val_subscores": vectors,
        "parent_program_for_candidate": parents,
        "iteration_ids_by_candidate_idx": anchors,
        "list_of_named_predictors": ["a", "b"],
        "num_metric_calls_by_discovery": discovery,
        "total_num_evals": clock,
        "evaluation_cache": None,
    }
    (root / "gepa_state.bin").write_bytes(pickle.dumps(state))
    write_json(root / "run_log.json", trace)
    write_json(
        root / "candidates.json",
        {
            "run_contract": contract,
            "candidates": programs,
            "total_metric_calls": clock,
            "val_aggregate_scores": [sum(v.values()) / 4 for v in vectors],
        },
    )
    write_json(root / "action_summary.json", {"run_contract": contract, "proposal_records": records})
    write_json(root / "heldout" / "records.json", {"forbidden": "test answers"})
    entry = {
        "id": root.name,
        "root": str(root),
        "kind": "search",
        "lineage": lineage,
        "cohort": lineage,
        "completed_iterations": length,
        "standard_end_iteration": 4,
        "files": dict.fromkeys(
            ["candidates.json", "run_log.json", "action_summary.json", "gepa_state.bin", PARENT_JOURNAL]
        ),
        "identity": contract_identity(contract),
    }
    refresh_hashes(entry)
    return entry


def manifest_fixture(path: Path, entries: list[dict]) -> Path:
    """Write an input manifest containing only the supplied fixture archives.

    Args:
        path: Manifest destination.
        entries: Archive specifications.

    Returns:
        Saved manifest path.
    """
    write_json(path, {"schema_version": 1, "archives": entries})
    return path


def test_reconstruct_incumbent_and_weak_bridge(tmp_path: Path) -> None:
    """Credit records against the incumbent and retain a weaker useful ancestor."""
    entry = search_fixture(tmp_path / "run")
    result = load_search(Archive(entry, tmp_path))
    rows = result["decisions"]
    assert [r["outcome"]["record_break"] for r in rows] == [1, 0, 0, 0, 1, 0]
    assert rows[2]["outcome"]["validation_score"] == 0.5
    assert rows[2]["features"]["incumbent_score"] == 0.75
    assert rows[3]["features"]["training_score"] == 0.5
    assert rows[3]["outcome"]["validation_score"] is None
    assert rows[1]["outcome"]["status"] == "training_tie"
    assert rows[-1]["outcome"]["status"] == "training_loss"
    assert len(result["delayed_records"]) == 2
    assert [a["candidate"] for a in result["delayed_records"][-1]["ancestors"]] == [2, 1, 0]
    assert result["delayed_records"][-1]["ancestors"][0]["evaluations_until_record"] > 0
    assert result["delayed_records"][-1]["ancestors"][0]["parent_selections_before_record"] == 1
    assert result["delayed_records"][-1]["ancestors"][0]["parent_selections_after_record"] == 0


def test_prefixes_deduplicate_but_distinct_lineages_do_not(tmp_path: Path) -> None:
    """Keep distinct stochastic decisions while counting imported history once."""
    full = search_fixture(tmp_path / "full")
    prefix = search_fixture(tmp_path / "prefix", length=3)
    other = search_fixture(tmp_path / "other", lineage="other")
    data = import_manifest(manifest_fixture(tmp_path / "manifest.json", [full, prefix, other]))
    assert len(data["decisions"]) == 12
    assert data["audit"]["duplicate_decisions_removed"] == 3
    assert len(data["decisions"][0]["archives"]) == 2


def test_conflicting_prefix_fails_closed(tmp_path: Path) -> None:
    """Reject shared decision IDs whose archived actions disagree."""
    full = search_fixture(tmp_path / "full")
    prefix = search_fixture(tmp_path / "prefix", length=3)
    path = Path(prefix["root"]) / "action_summary.json"
    content = json.loads(path.read_text())
    content["proposal_records"][0]["semantic_action"] = "different"
    write_json(path, content)
    refresh_hashes(prefix)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        import_manifest(manifest_fixture(tmp_path / "manifest.json", [full, prefix]))


def test_unknown_discard_stays_unknown(tmp_path: Path) -> None:
    """Do not turn an interrupted final proposal into a negative example."""
    entry = search_fixture(tmp_path / "run", length=4)
    entry["completed_iterations"] = 3
    result = load_search(Archive(entry, tmp_path))
    assert result["decisions"][-1]["outcome"]["record_break"] is None
    assert result["decisions"][-1]["outcome"]["incumbent_gain"] is None


def test_future_outcomes_do_not_change_prior_features(tmp_path: Path) -> None:
    """Keep pre-action features independent of child scores and later descendants."""
    entry = search_fixture(tmp_path / "run")
    original = load_search(Archive(entry, tmp_path))["decisions"]
    root = Path(entry["root"])
    state = pickle.loads((root / "gepa_state.bin").read_bytes())
    state["prog_candidate_val_subscores"][-1] = dict.fromkeys(range(4), 0.0)
    state["program_candidates"][-1]["a"] = "Future outcome text that must not become a feature"
    (root / "gepa_state.bin").write_bytes(pickle.dumps(state))
    candidates = json.loads((root / "candidates.json").read_text())
    candidates["val_aggregate_scores"][-1] = 0.0
    candidates["candidates"] = state["program_candidates"]
    write_json(root / "candidates.json", candidates)
    refresh_hashes(entry)
    changed = load_search(Archive(entry, tmp_path))["decisions"]
    for before, after in zip(original[:5], changed[:5], strict=True):
        assert before["features"] == after["features"]
        assert before["parent_failure_pattern"] == after["parent_failure_pattern"]


def test_missing_journal_never_uses_post_selection_score_availability(tmp_path: Path) -> None:
    """Leave training scores missing when no pre-action evidence is available."""
    entry = search_fixture(tmp_path / "run")
    del entry["files"][PARENT_JOURNAL]
    result = load_search(Archive(entry, tmp_path))
    assert all(row["features"]["training_score"] is None for row in result["decisions"])


def test_future_journal_missingness_does_not_change_prior_features(tmp_path: Path) -> None:
    """Keep earlier feature values independent of later archive coverage."""
    entry = search_fixture(tmp_path / "run")
    before = load_search(Archive(entry, tmp_path))["decisions"]
    connection = sqlite3.connect(Path(entry["root"]) / PARENT_JOURNAL)
    connection.execute("DELETE FROM responses WHERE scope = 'optimizer-iteration-5'")
    connection.commit()
    connection.close()
    refresh_hashes(entry)
    after = load_search(Archive(entry, tmp_path))["decisions"]
    assert [row["features"] for row in before[:-1]] == [row["features"] for row in after[:-1]]
    assert after[-1]["features"]["training_score"] is None


def test_no_test_files_or_payloads_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete the study while test records and network access are forbidden."""
    entry = search_fixture(tmp_path / "run")
    original = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        """Reject test-file reads while preserving ordinary fixture I/O."""
        if "heldout" in path.parts:
            pytest.fail("Opened held-out test evidence")
        return original(path, *args, **kwargs)

    def forbidden_network(*args: object, **kwargs: object) -> None:
        """Reject any attempted network connection during analysis."""
        pytest.fail("Attempted network access during offline analysis")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    summary = run_study(manifest_fixture(tmp_path / "manifest.json", [entry]), tmp_path / "study")
    assert summary["decisions_by_kind"] == {"search": 6}
    assert (tmp_path / "study" / "report.html").exists()
    entry["files"]["heldout/records.json"] = "forbidden"
    with pytest.raises(ValueError, match="non-test allowlist"):
        Archive(entry, tmp_path)


def test_checksum_and_candidate_join_failures(tmp_path: Path) -> None:
    """Reject both modified bytes and internally inconsistent candidate joins."""
    entry = search_fixture(tmp_path / "run")
    path = Path(entry["root"]) / "run_log.json"
    rows = json.loads(path.read_text())
    rows[0]["new_program_idx"] = 2
    write_json(path, rows)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_search(Archive(entry, tmp_path))
    refresh_hashes(entry)
    with pytest.raises(ValueError, match="iteration join"):
        load_search(Archive(entry, tmp_path))


def test_checkpoint_rejects_executable_globals() -> None:
    """Reject checkpoint payloads that could invoke arbitrary functions."""
    with pytest.raises(ValueError, match="Executable checkpoint"):
        DataOnlyUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_pilot_checksum_uses_its_original_protocol(tmp_path: Path) -> None:
    """Verify pilot spaced-JSON hashing rather than the Jev mailbox digest."""
    payload = {"unicode": "שלום", "score": 0.5}
    saved = {
        "record": payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False, default=str).encode()
        ).hexdigest(),
    }
    write_json(tmp_path / "summary.json", saved)
    entry = {"id": "pilot", "kind": "pilot", "root": str(tmp_path), "files": {"summary.json": ""}}
    refresh_hashes(entry)
    assert Archive(entry, tmp_path).json("summary.json", sealed_pilot=True) == payload
    saved["record"]["score"] = 1.0
    write_json(tmp_path / "summary.json", saved)
    refresh_hashes(entry)
    with pytest.raises(ValueError, match="envelope checksum"):
        Archive(entry, tmp_path).json("summary.json", sealed_pilot=True)


def test_pilot_cohorts_preserve_matched_opportunities_and_jev_role(tmp_path: Path) -> None:
    """Keep a Jev novelty verifier separate from the action controller and search labels."""
    protocol = {
        "identity": "diversity-quality-paired-training-pilot-v1",
        "arms": {"generative": ["quality", "generative"], "jev": ["quality", "jev"]},
    }
    contract = {"protocol": protocol, "source": "fixture", "training_examples": ["opportunity"]}
    comparisons = [
        {
            "opportunity": 0,
            "arm": arm,
            "component": "a",
            "changed": arm == "generative",
            "training": {"delta": -0.5},
            "transfer": {"delta": 0.25},
            "proposal_metadata": {
                "semantic_action": "restrict_meaning",
                "action_target_section": "Objective",
                "action_choice": "restrict_meaning@Objective/REPLACE_TEXT",
                "controller_sampling": {
                    "distribution": [{"pair": "restrict_meaning@Objective/REPLACE_TEXT", "probability": 1.0}],
                    "sampled": ["restrict_meaning@Objective/REPLACE_TEXT"],
                    "sampled_probabilities": [1.0],
                },
            },
        }
        for arm in protocol["arms"]
    ]
    payload = {"protocol": protocol, "comparisons": comparisons}
    write_json(tmp_path / "pilot-contract.json", contract)
    write_json(
        tmp_path / "summary.json",
        {"record": payload, "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()},
    )
    entry = {
        "id": "pilot",
        "root": str(tmp_path),
        "kind": "pilot",
        "lineage": "pilot",
        "cohort": "pilot",
        "files": dict.fromkeys(["pilot-contract.json", "summary.json"]),
    }
    refresh_hashes(entry)
    data = import_manifest(manifest_fixture(tmp_path / "manifest.json", [entry]))
    rows = data["decisions"]
    assert len({row["matched_group"] for row in rows}) == 1
    assert {row["controller_backend"] for row in rows} == {"verbalized"}
    assert {row["novelty_backend"] for row in rows} == {"generative", "jev"}
    assert all(row["outcome"]["record_break"] is None for row in rows)
    assert all(row["selection"]["selected_probability"] == 1.0 for row in rows)
    results = evaluate_signal(rows)
    assert results["folds"] == []
    assert len(results["pilot_matched_opportunities"]) == 1


def test_protocol_changes_cannot_share_a_lineage(tmp_path: Path) -> None:
    """Reject a continuation whose editor protocol silently changes."""
    first = search_fixture(tmp_path / "first")
    second = search_fixture(tmp_path / "second")
    for name in ("candidates.json", "action_summary.json"):
        path = Path(second["root"]) / name
        content = json.loads(path.read_text())
        content["run_contract"]["optimizer"]["react_execution"] = {"completion": "different-editor"}
        write_json(path, content)
    second["identity"] = contract_identity(content["run_contract"])
    refresh_hashes(second)
    with pytest.raises(ValueError, match="Incompatible protocols"):
        import_manifest(manifest_fixture(tmp_path / "manifest.json", [first, second]))


def test_candidate_only_model_does_not_use_selected_action(tmp_path: Path) -> None:
    """Keep selected section length and action labels out of the candidate baseline."""
    rows = load_search(Archive(search_fixture(tmp_path / "run"), tmp_path))["decisions"]
    before, _ = predict_fold(rows[:4], rows[4:], "candidate_stage")
    changed = copy.deepcopy(rows)
    for row in changed:
        row["selection"]["action"] = "different-action"
        row["features"]["section_chars"] = 99999
    after, _ = predict_fold(changed[:4], changed[4:], "candidate_stage")
    np.testing.assert_array_equal(before, after)


def test_vanilla_null_role_contract_has_no_action_training_rows(tmp_path: Path) -> None:
    """Admit vanilla's null controller settings only as descriptive search evidence."""
    entry = search_fixture(tmp_path / "run")
    entry["kind"] = "vanilla"
    path = Path(entry["root"]) / "candidates.json"
    content = json.loads(path.read_text())
    content["run_contract"]["models"]["reflection_role_decoding"] = None
    content["run_contract"]["optimizer"]["generalization"] = None
    write_json(path, content)
    refresh_hashes(entry)
    result = load_search(Archive(entry, tmp_path))
    assert result["decisions"] == []
    assert result["counts"]["record_improvements"] == 2


def test_folds_preserve_time_lineages_and_budget_transition(tmp_path: Path) -> None:
    """Avoid shuffled rows and hold out complete imported lineages."""
    entries = [search_fixture(tmp_path / name, lineage=name) for name in ("first", "second")]
    data = import_manifest(manifest_fixture(tmp_path / "manifest.json", entries))
    folds = make_folds(data["decisions"])
    assert {fold["scheme"] for fold in folds} == {"forward", "lineage", "extension"}
    for fold in folds:
        assert not {r["decision_id"] for r in fold["train"]} & {r["decision_id"] for r in fold["test"]}
        if fold["scheme"] == "lineage":
            assert not {r["lineage"] for r in fold["train"]} & {r["lineage"] for r in fold["test"]}
        else:
            assert max(r["iteration"] for r in fold["train"]) < min(r["iteration"] for r in fold["test"])


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_heldout_labels_do_not_change_predictions(tmp_path: Path, model: str) -> None:
    """Prevent test labels from entering preprocessing or model fitting."""
    entry = search_fixture(tmp_path / "run")
    rows = load_search(Archive(entry, tmp_path))["decisions"]
    test = copy.deepcopy(rows[4:])
    before, _ = predict_fold(rows[:4], test, model)
    for row in test:
        row["outcome"]["record_break"] = 1 - row["outcome"]["record_break"]
        row["outcome"]["validation_score"] = 999
    after, _ = predict_fold(rows[:4], test, model)
    np.testing.assert_array_equal(before, after)


def test_preprocessing_does_not_fit_on_test_rows(tmp_path: Path) -> None:
    """Keep one prediction unchanged when other test features become extreme."""
    rows = load_search(Archive(search_fixture(tmp_path / "run"), tmp_path))["decisions"]
    first, _ = predict_fold(rows[:4], rows[4:], "candidate_action_interactions")
    changed = copy.deepcopy(rows[4:])
    changed[1]["features"]["prompt_chars"] = 10**9
    changed[1]["parent_failure_pattern"] = [1.0, 0.0, 1.0, 0.0]
    second, _ = predict_fold(rows[:4], changed, "candidate_action_interactions")
    assert first[0] == second[0]


def test_no_events_does_not_claim_discrimination(tmp_path: Path) -> None:
    """Retain proper scoring rules while marking PR discrimination unavailable."""
    rows = load_search(Archive(search_fixture(tmp_path / "run"), tmp_path))["decisions"]
    result = score_predictions([rows[1], rows[2]], np.asarray([0.1, 0.2]))
    assert result["average_precision"] is None
    assert result["precision_recall"] is None
    assert result["log_loss"] > 0


def test_incompatible_cohorts_do_not_enter_lineage_transfer(tmp_path: Path) -> None:
    """Avoid pooling different validation identities in a learned comparison."""
    entries = [search_fixture(tmp_path / name, lineage=name) for name in ("first", "second")]
    data = import_manifest(manifest_fixture(tmp_path / "manifest.json", entries))
    for row in data["decisions"]:
        if row["lineage"] == "second":
            row["comparison_id"] = "different-validation"
    result = evaluate_signal(data["decisions"])
    assert all(fold["scheme"] != "lineage" for fold in result["folds"])


def test_output_cannot_mutate_archive(tmp_path: Path) -> None:
    """Reject an output directory nested inside a frozen run archive."""
    entry = search_fixture(tmp_path / "run")
    with pytest.raises(ValueError, match="outside every input"):
        run_study(manifest_fixture(tmp_path / "manifest.json", [entry]), tmp_path / "run" / "analysis")
