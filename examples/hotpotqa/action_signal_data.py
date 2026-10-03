"""Read audited search archives without evaluating prompts or opening test records."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import pickle
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PARENT_JOURNAL = ".evaluation-journal/responses.sqlite3"
RUN_FILES = {"action_summary.json", "candidates.json", "run_log.json", "gepa_state.bin", PARENT_JOURNAL}
PILOT_FILES = {"summary.json", "pilot-contract.json"}
PILOT_PROPOSAL = re.compile(r"opportunity-\d+/(verbalized|jev)/proposal\.json")
PILOT_PROTOCOLS = {"jev-paired-proposal-quality-v1", "diversity-quality-paired-training-pilot-v1"}
COMPARISON_EXCLUDED_OPTIMIZER_FIELDS = {"max_metric_calls", "semantic_controller_policy", "generalization"}
PROBABILITY_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-12


def digest(value: Any) -> str:
    """Hash a normalized JSON value independently of formatting.

    Args:
        value: JSON-compatible value.

    Returns:
        Canonical SHA-256 digest.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class DataOnlyUnpickler(pickle.Unpickler):
    """Read legacy dictionary checkpoints without allowing executable globals."""

    def find_class(self, module: str, name: str) -> Any:
        """Reject object construction through pickle globals.

        Args:
            module: Requested module.
            name: Requested global.

        Raises:
            ValueError: Always; these archives must contain only built-in data.
        """
        raise ValueError(f"Executable checkpoint global is forbidden: {module}.{name}")


class ArchivedEvaluation:
    """Hold inert evaluation attributes without importing an executable adapter."""

    scores: list[float]


class EvaluationOnlyUnpickler(DataOnlyUnpickler):
    """Substitute inert storage for the journal's one permitted result class."""

    def find_class(self, module: str, name: str) -> Any:
        """Allow only the archived evaluation container.

        Args:
            module: Serialized class module.
            name: Serialized class name.

        Returns:
            Inert storage class for evaluation records.

        Raises:
            ValueError: Any other executable global is requested.
        """
        if (module, name) == ("gepa.core.adapter", "EvaluationBatch"):
            return ArchivedEvaluation
        return super().find_class(module, name)


class Archive:
    """Enforce an explicit, checksum-bound allowlist of non-test input files."""

    def __init__(self, entry: dict[str, Any], manifest_dir: Path) -> None:
        """Resolve one manifest entry without discovering files recursively.

        Args:
            entry: Archive identity, protocol, and per-file SHA-256 values.
            manifest_dir: Base for relative archive roots.
        """
        self.entry = entry
        self.root = (manifest_dir / entry["root"]).resolve()
        self.hashes = entry["files"]
        self.reads: dict[str, str] = {}
        self.cache: dict[str, bytes] = {}
        allowed = RUN_FILES if entry["kind"] in {"search", "vanilla"} else PILOT_FILES
        for name in self.hashes:
            if name not in allowed and not (entry["kind"] == "pilot" and PILOT_PROPOSAL.fullmatch(name)):
                raise ValueError(f"Input outside the non-test allowlist: {name}")

    def read(self, name: str) -> bytes:
        """Read and verify an explicitly listed artifact.

        Args:
            name: Allowed relative artifact path.

        Returns:
            Hash-verified bytes, cached for this import.

        Raises:
            ValueError: The path escapes the archive or its hash disagrees.
        """
        if name not in self.hashes:
            raise ValueError(f"Artifact absent from manifest: {name}")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Artifact escapes archive: {name}")
        if name not in self.cache:
            data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != self.hashes[name]:
                raise ValueError(f"Archive checksum mismatch: {self.entry['id']}/{name}")
            self.cache[name] = data
            self.reads[name] = actual
        return self.cache[name]

    def json(self, name: str, *, sealed_pilot: bool = False) -> Any:
        """Decode JSON and verify the historical pilot envelope when requested.

        Args:
            name: Manifest-listed JSON artifact.
            sealed_pilot: Use the pilot's original spaced-JSON checksum protocol.

        Returns:
            Decoded payload, without a verified envelope.

        Raises:
            ValueError: A pilot envelope fails its own checksum protocol.
        """
        value = json.loads(self.read(name))
        if sealed_pilot:
            payload = value["record"]
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True, allow_nan=False, default=str).encode()
            ).hexdigest()
            if value["sha256"] != expected:
                raise ValueError(f"Pilot envelope checksum mismatch: {name}")
            return payload
        return value


def mean(values: list[float]) -> float | None:
    """Return the mean of observed scores, leaving missing observations null.

    Args:
        values: Observed scores.

    Returns:
        Arithmetic mean or null for an empty collection.
    """
    return sum(values) / len(values) if values else None


def parent_scores(archive: Archive) -> dict[int, list[float]]:
    """Read only pre-action parent batches from an immutable evaluation journal.

    Args:
        archive: Archive optionally containing a checksum-bound journal.

    Returns:
        Zero-based iteration numbers mapped to parent scores.

    Raises:
        ValueError: A journal response is corrupt or represents an unsupported batch.
    """
    if PARENT_JOURNAL not in archive.hashes:
        return {}
    archive.read(PARENT_JOURNAL)
    connection = sqlite3.connect((archive.root / PARENT_JOURNAL).as_uri() + "?mode=ro&immutable=1", uri=True)
    scores = {}
    try:
        rows = connection.execute(
            "SELECT scope, ordinal, response_json, response_sha256 FROM responses "
            "WHERE namespace = 'parents' AND scope GLOB 'optimizer-iteration-[0-9]*'"
        )
        for scope, ordinal, raw, checksum in rows:
            match = re.fullmatch(r"optimizer-iteration-(\d+)", scope)
            if match is None or ordinal != 0 or hashlib.sha256(raw.encode()).hexdigest() != checksum:
                raise ValueError("Invalid parent evaluation journal record")
            payload = json.loads(raw)
            if payload.get("kind") != "evaluation_batch":
                raise ValueError("Unsupported parent journal payload")
            batches, _ = EvaluationOnlyUnpickler(io.BytesIO(base64.b64decode(payload["data"], validate=True))).load()
            if len(batches) != 1 or not isinstance(batches[0], ArchivedEvaluation):
                raise ValueError("Ambiguous parent evaluation batch")
            index = int(match[1])
            if index in scores or any(not math.isfinite(v) or not 0 <= v <= 1 for v in batches[0].scores):
                raise ValueError("Duplicate or invalid parent evaluation scores")
            scores[index] = batches[0].scores
    finally:
        connection.close()
    return scores


def contract_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """Extract scientific identities while excluding test data and runtime addresses.

    Args:
        contract: Archived Wikipedia run contract.

    Returns:
        Provenance and an explicit comparison identity across controller backends.
    """
    optimizer = contract["optimizer"]
    splits = contract["data"]["splits"]
    role_decoding = contract["models"].get("reflection_role_decoding") or {}
    generalization = optimizer.get("generalization") or {}
    catalog = optimizer.get("semantic_action_space")
    model_fields = (
        "solver",
        "solver_version",
        "solver_decoding",
        "reflection",
        "reflection_version",
        "reflection_decoding",
    )
    comparison = {
        "data": {split: {k: splits[split][k] for k in ("count", "sha256")} for split in ("train", "val")},
        "data_source": contract["data"].get("source"),
        "models": {k: contract["models"].get(k) for k in model_fields},
        "editor_role_decoding": {role: decoding for role, decoding in role_decoding.items() if role != "controller"},
        "program": {k: v for k, v in contract.get("program", {}).items() if k != "reported_supplemental_metric"},
        "retrieval": contract["retrieval"],
        "template_family": optimizer.get("template_family"),
        "action_catalog": {"version": catalog["version"], "sha256": digest(catalog)} if catalog else None,
        "module_selector": optimizer["component_selector"],
        "minibatch_size": optimizer["reflection_minibatch_size"],
        "sampling": optimizer["proposal_sampling_strategy"],
        "admission": optimizer["acceptance_criterion"],
        "optimizer_contract_sha256": digest(
            {k: v for k, v in optimizer.items() if k not in COMPARISON_EXCLUDED_OPTIMIZER_FIELDS}
        ),
        "generalization_sha256": digest({k: v for k, v in generalization.items() if k != "controller_direction"}),
    }
    return {
        "archive_source": contract["execution_runtime"]["source_commit"],
        "source_scope": "archive contract; imported decisions retain their logged controller policy",
        "controller": optimizer.get("semantic_controller_policy"),
        "controller_direction": generalization.get("controller_direction"),
        "controller_decoding": role_decoding.get("controller"),
        "comparison": comparison,
        "comparison_id": digest(comparison),
    }


def section_length(text: str, section: str) -> int:
    """Count a selected Markdown section's body without consulting the child.

    Args:
        text: Parent component text.
        section: Exact catalog section name.

    Returns:
        Body character count; absent sections have an empty body.
    """
    match = re.search(rf"(?m)^## {re.escape(section)}\s*\n", text)
    if match is None:
        return 0
    rest = text[match.end() :]
    end = re.search(r"(?m)^## ", rest)
    return len((rest[: end.start()] if end else rest).strip())


def sampling_record(metadata: dict[str, Any]) -> dict[str, Any]:
    """Preserve actual action probabilities without treating them as rewards.

    Args:
        metadata: Archived action metadata.

    Returns:
        Selected choice, logged support, probability, and controller policy.

    Raises:
        ValueError: A recorded distribution is invalid or omits the chosen action.
    """
    sampling = metadata.get("controller_sampling") or {}
    choice = metadata.get("action_choice")
    probabilities = sampling.get("sampling_probs") or {}
    if not probabilities and sampling.get("distribution"):
        distribution = sampling["distribution"]
        probabilities = {option["pair"]: option["probability"] for option in distribution}
        if len(probabilities) != len(distribution):
            raise ValueError("Ambiguous historical pilot sampling distribution")
    probability = probabilities.get(choice)
    if "sampled" in sampling:
        sampled = dict(zip(sampling["sampled"], sampling["sampled_probabilities"], strict=True))
        if choice not in sampled or probability is None or not math.isclose(sampled[choice], probability):
            raise ValueError("Historical pilot selected probability disagrees with its distribution")
    if probabilities:
        if any(not math.isfinite(v) or v < 0 for v in probabilities.values()):
            raise ValueError("Invalid logged sampling probabilities")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=PROBABILITY_TOLERANCE) or not probability:
            raise ValueError("Selected action is outside the recorded sampling support")
    return {
        "choice": choice,
        "action": metadata.get("semantic_action") or metadata.get("action"),
        "section": metadata.get("action_target_section"),
        "operator": metadata.get("action_operator"),
        "logged_menu": sorted(set(probabilities) | set(sampling.get("probs") or {})),
        "excluded_choices": sampling.get("excluded_choices", []),
        "sampling_probabilities": probabilities,
        "selected_probability": probability,
        "policy": sampling.get("policy"),
        "request_id": sampling.get("request_id"),
    }


def load_search(archive: Archive) -> dict[str, Any]:
    """Reconstruct completed search decisions in chronological order.

    Args:
        archive: Explicitly authorized search artifacts.

    Returns:
        Decisions, candidate trajectory, delayed ancestry, and coverage audit.

    Raises:
        ValueError: Identities, budgets, candidate joins, or source assumptions disagree.
    """
    entry = archive.entry
    candidates = archive.json("candidates.json")
    state = DataOnlyUnpickler(io.BytesIO(archive.read("gepa_state.bin"))).load()
    trace = archive.json("run_log.json")
    identity = contract_identity(candidates["run_contract"])
    contract = candidates["run_contract"]
    optimizer = contract["optimizer"]
    sampling = optimizer["proposal_sampling_strategy"]
    supported = (
        optimizer["component_selector"] == "round_robin"
        and optimizer["acceptance_criterion"] == "strict_improvement"
        and optimizer.get("validation_evaluation") == "full_eval"
        and not optimizer.get("merge")
        and state.get("evaluation_cache") is None
        and sampling.get("parents_per_iteration") == 1
        and sampling.get("mutations_per_parent") == 1
    )
    if not supported:
        raise ValueError("Unsupported search protocol: requires serial round-robin, full validation, no merge/cache")
    programs = state["program_candidates"]
    if programs != candidates["candidates"] or state["total_num_evals"] != candidates["total_metric_calls"]:
        raise ValueError("Checkpoint and candidate archive disagree")
    vectors = state["prog_candidate_val_subscores"]
    val_count = identity["comparison"]["data"]["val"]["count"]
    scores = []
    for vector in vectors:
        if set(map(int, vector)) != set(range(val_count)):
            raise ValueError("Partial or incompatible validation score vector")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in vector.values()):
            raise ValueError("Invalid validation score")
        scores.append(sum(vector.values()) / val_count)
    if len(scores) != len(programs) or any(
        not math.isclose(a, b, abs_tol=SCORE_TOLERANCE)
        for a, b in zip(scores, candidates["val_aggregate_scores"], strict=True)
    ):
        raise ValueError("Candidate validation aggregates disagree")
    anchors = state["iteration_ids_by_candidate_idx"]
    if len(set(anchors)) != len(anchors):
        raise ValueError("Duplicate candidate iteration anchors")
    child_by_anchor = {anchor: idx for idx, anchor in enumerate(anchors) if idx}
    parents = state["parent_program_for_candidate"]
    if len(parents) != len(programs) or any(
        len(p) != 1 or p[0] is None or p[0] >= i for i, p in enumerate(parents) if i
    ):
        raise ValueError("Unsupported or cyclic candidate lineage")
    action_records = []
    if entry["kind"] == "search":
        action_summary = archive.json("action_summary.json")
        if action_summary["run_contract"] != contract:
            raise ValueError("Action and candidate contracts disagree")
        action_records = action_summary["proposal_records"]
    by_iteration: dict[int, dict[str, Any]] = {}
    for record in action_records:
        index = record["iteration"] - 1
        if index in by_iteration or not 0 <= index < len(trace):
            raise ValueError("Ambiguous or out-of-range action iteration")
        by_iteration[index] = record
    journal_scores = parent_scores(archive) if action_records else {}
    if not 0 <= entry["completed_iterations"] <= len(trace):
        raise ValueError("Invalid completed-iteration boundary")
    modules = state["list_of_named_predictors"]
    cursors = {0: 0}
    depths = {0: 0}
    attempts: Counter[int] = Counter()
    wins: Counter[int] = Counter()
    observed = {0}
    clock = val_count
    last_record_cost = clock
    incumbent = scores[0]
    decisions = []
    trajectory = [{"candidate": 0, "score": incumbent, "evaluations": clock, "record_break": False}]
    created_by = {}
    selection_costs: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(trace):
        if row["i"] != index or len(row.get("tasks", [])) != 1:
            raise ValueError("Non-sequential or ambiguous iteration trace")
        task = row["tasks"][0]
        parent = task["parent_idx"]
        if parent not in observed or parent != row["selected_program_candidate"]:
            raise ValueError("Parent is not available before the decision")
        batch = task["subsample_ids"]
        clock += len(batch)
        decision_cost = clock
        selection_costs[parent].append(decision_cost)
        record = by_iteration.get(index)
        child = child_by_anchor.get(row["iteration_id"])
        if child != row.get("new_program_idx"):
            raise ValueError("Candidate iteration join disagrees with trace")
        if child is not None and entry["kind"] == "search" and record is None:
            raise ValueError("Validated FOREST child has no action record")
        before = task.get("subsample_scores")
        after = task.get("new_subsample_scores")
        if after is not None and (before is None or len(before) != len(batch) or len(after) != len(batch)):
            raise ValueError("Unpaired minibatch outcomes")
        pre_action_scores = journal_scores.get(index)
        if pre_action_scores is not None and (
            len(pre_action_scores) != len(batch) or (before is not None and before != pre_action_scores)
        ):
            raise ValueError("Parent journal and iteration scores disagree")
        if record is not None:
            module = modules[cursors[parent]]
            cursors[parent] = (cursors[parent] + 1) % len(modules)
            texts = record.get("texts_by_component") or {}
            if texts and set(texts) != {module}:
                raise ValueError("Logged component disagrees with round-robin reconstruction")
            selection = sampling_record(record)
            known = child is not None or after is not None or index < entry["completed_iterations"]
            label = int(child is not None and scores[child] > incumbent) if known else None
            status = "unresolved"
            if child is not None:
                status = "validated_child"
            elif after is not None:
                status = "training_tie" if sum(after) == sum(before) else "training_loss"
                if sum(after) > sum(before):
                    raise ValueError("Training improvement has no validated child")
            elif known:
                status = "discarded_without_evaluation"
            features = {
                "parent_score": scores[parent],
                "incumbent_score": incumbent,
                "incumbent_gap": incumbent - scores[parent],
                # Trace score availability depends on child evaluation, so it must not become a predictive signal.
                "training_score": mean(pre_action_scores or []),
                "prompt_chars": sum(len(text) for text in programs[parent].values()),
                "module_prompt_chars": len(programs[parent][module]),
                "section_chars": section_length(programs[parent][module], selection["section"] or ""),
                "depth": depths[parent],
                "prior_parent_decisions": attempts[parent],
                "prior_parent_breakthroughs": wins[parent],
                "evaluations_before_action": decision_cost,
                "evaluations_since_record": decision_cost - last_record_cost,
            }
            decision_id = f"{entry['lineage']}:{row['iteration_id']}"
            decision = {
                "decision_id": decision_id,
                "lineage": entry["lineage"],
                "cohort": entry["cohort"],
                "comparison_id": identity["comparison_id"],
                "kind": "search",
                "iteration": index + 1,
                "phase": "expanded" if index >= entry["standard_end_iteration"] else "standard",
                "parent": parent,
                "parent_sha256": digest(programs[parent]),
                "module": module,
                "selection": selection,
                "batch_ids": batch,
                "features": features,
                "parent_failure_pattern": [1.0 - vectors[parent][k] for k in sorted(vectors[parent], key=int)],
                "outcome": {
                    "status": status,
                    "record_break": label,
                    "child": child,
                    "validation_score": scores[child] if child is not None else None,
                    "incumbent_gain": max(0.0, scores[child] - incumbent)
                    if child is not None
                    else (0.0 if known else None),
                    "training_delta": (sum(after) - sum(before)) / len(batch) if after is not None else None,
                },
                "missingness": [name for name, value in features.items() if value is None]
                + (["selected_sampling_probability_not_recorded"] if selection["selected_probability"] is None else [])
                + (["discard_reason_not_recorded"] if status == "discarded_without_evaluation" else [])
                + (["child_validation_not_measured"] if child is None else []),
                "archives": [entry["id"]],
            }
            decisions.append(decision)
            attempts[parent] += 1
            wins[parent] += label or 0
            if child is not None:
                created_by[child] = decision_id
        if after is not None:
            clock += len(batch)
        if child is not None:
            if parents[child] != [parent] or state["num_metric_calls_by_discovery"][child] != clock:
                raise ValueError("Candidate parent or logical evaluation accounting disagrees")
            clock += val_count
            improvement = scores[child] > incumbent
            trajectory.append(
                {"candidate": child, "score": scores[child], "evaluations": clock, "record_break": improvement}
            )
            if improvement:
                incumbent = scores[child]
                last_record_cost = clock
            observed.add(child)
            depths[child] = depths[parent] + 1
            cursors[child] = cursors[parent]
    if observed != set(range(len(programs))) or clock != state["total_num_evals"]:
        raise ValueError("Incomplete candidate joins or logical evaluation accounting")
    delayed = []
    costs = {point["candidate"]: point["evaluations"] for point in trajectory}
    for point in trajectory:
        if not point["record_break"]:
            continue
        ancestors = []
        parent = parents[point["candidate"]][0]
        while parent is not None:
            ancestors.append(
                {
                    "candidate": parent,
                    "decision_id": created_by.get(parent),
                    "evaluations_until_record": point["evaluations"] - costs[parent],
                    "parent_selections_before_record": sum(
                        cost < point["evaluations"] for cost in selection_costs[parent]
                    ),
                    "parent_selections_after_record": sum(
                        cost > point["evaluations"] for cost in selection_costs[parent]
                    ),
                }
            )
            parent = parents[parent][0]
        delayed.append(
            {"record_candidate": point["candidate"], "evaluations": point["evaluations"], "ancestors": ancestors}
        )
    return {
        "decisions": decisions,
        "identity": identity,
        "trajectory": trajectory,
        "delayed_records": delayed,
        "followup": {
            "end_evaluations": clock,
            "descendant_followup": "right-censored at archive end; ancestry is not causal credit",
            "parent_selections": dict(Counter(row["selected_program_candidate"] for row in trace)),
            "candidate_followup_evaluations": {candidate: clock - cost for candidate, cost in costs.items()},
        },
        "counts": {
            "decisions": len(decisions),
            "validated_children": len(programs) - 1,
            "record_improvements": len(delayed),
        },
        "parent_training_feature": {
            "source": "pre-action journal only; missing rows stay null",
            "missing_decisions": sum(index not in journal_scores for index in by_iteration),
        },
    }


def load_pilot(archive: Archive) -> dict[str, Any]:
    """Import training-transfer diagnostics without constructing validation labels.

    Args:
        archive: One completed matched pilot's explicit files.

    Returns:
        Supplementary rows grouped by matched opportunity and source protocol.

    Raises:
        ValueError: The protocol is unrecognized or the contract disagrees.
    """
    contract = archive.json("pilot-contract.json")
    protocol = contract["protocol"]
    name = protocol["identity"]
    if name not in PILOT_PROTOCOLS:
        raise ValueError(f"Unsupported pilot protocol: {name}")
    sealed = name == "diversity-quality-paired-training-pilot-v1"
    summary = archive.json("summary.json", sealed_pilot=sealed)
    if summary["protocol"] != protocol:
        raise ValueError("Pilot summary and contract disagree")
    decisions = []
    for row in summary["comparisons"]:
        if "training" not in row:
            continue
        metadata = row.get("proposal_metadata")
        if metadata is None:
            metadata = archive.json(f"opportunity-{row['opportunity']}/{row['arm']}/proposal.json")["metadata"]
        controller = "verbalized" if sealed else row["arm"]
        selection = sampling_record(metadata)
        decisions.append(
            {
                "decision_id": f"{archive.entry['lineage']}:{row['opportunity']}:{row['arm']}",
                "kind": "pilot",
                "lineage": archive.entry["lineage"],
                "cohort": f"{archive.entry['cohort']}:{row['arm']}",
                "comparison_id": digest(protocol),
                "matched_group": f"{archive.entry['lineage']}:{row['opportunity']}",
                "iteration": row["opportunity"] + 1,
                "module": row["component"],
                "selection": selection,
                "controller_backend": controller,
                "novelty_backend": protocol.get("arms", {}).get(row["arm"], [None, None])[1] if sealed else None,
                "features": {},
                "parent_failure_pattern": None,
                "outcome": {
                    "record_break": None,
                    "validation_score": None,
                    "status": row.get("generation_outcome", "changed" if row["changed"] else "noop"),
                    "training": row["training"],
                    "transfer": row["transfer"],
                    "changed": row["changed"],
                },
                "missingness": ["pilot_has_no_full_validation_or_search_lineage"]
                + (["selected_sampling_probability_not_recorded"] if selection["selected_probability"] is None else []),
                "archives": [archive.entry["id"]],
            }
        )
    return {
        "decisions": decisions,
        "identity": {
            "protocol": protocol,
            "source": contract["source"],
            "training_identity": digest(contract["training_examples"]),
        },
        "counts": {"decisions": len(decisions), "matched_groups": len({r["matched_group"] for r in decisions})},
    }


def import_manifest(path: Path) -> dict[str, Any]:
    """Import explicitly selected archives and reconcile duplicate histories.

    Args:
        path: Input manifest with archive identities and per-file SHA-256 values.

    Returns:
        Unique decisions, archive audits, and per-lineage descriptive evidence.

    Raises:
        ValueError: Manifest, provenance, expected counts, or shared histories conflict.
    """
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported action-signal manifest version")
    ids = [entry["id"] for entry in manifest["archives"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate archive IDs")
    decisions: dict[str, dict[str, Any]] = {}
    audits = []
    lineages = {}
    lineage_protocols = {}
    duplicate_count = 0
    for entry in manifest["archives"]:
        if entry["kind"] not in {"search", "vanilla", "pilot"}:
            raise ValueError(f"Unknown archive kind: {entry['kind']}")
        archive = Archive(entry, path.parent)
        result = load_pilot(archive) if entry["kind"] == "pilot" else load_search(archive)
        if "identity" in entry and entry["identity"] != result["identity"]:
            raise ValueError(f"Scientific identity mismatch: {entry['id']}")
        for key, expected in entry.get("expected_counts", {}).items():
            if result["counts"].get(key) != expected:
                raise ValueError(f"Expected count mismatch: {entry['id']}/{key}")
        protocol = (
            entry["kind"],
            entry["cohort"],
            result["identity"].get("comparison_id", digest(result["identity"].get("protocol"))),
            entry.get("standard_end_iteration"),
        )
        previous_protocol = lineage_protocols.setdefault(entry["lineage"], protocol)
        if previous_protocol != protocol:
            raise ValueError(f"Incompatible protocols assigned to one lineage: {entry['lineage']}")
        for row in result.pop("decisions"):
            previous = decisions.get(row["decision_id"])
            if previous is not None:
                old = {k: v for k, v in previous.items() if k != "archives"}
                new = {k: v for k, v in row.items() if k != "archives"}
                if old != new:
                    raise ValueError(f"Conflicting duplicate decision: {row['decision_id']}")
                previous["archives"].extend(row["archives"])
                duplicate_count += 1
            else:
                decisions[row["decision_id"]] = row
        audits.append(
            {
                "archive": entry["id"],
                "lineage": entry["lineage"],
                "files_read": archive.reads,
                **result["counts"],
                "identity": result["identity"],
            }
        )
        previous_lineage = lineages.get(entry["lineage"])
        if previous_lineage is None or result.get("followup", {}).get("end_evaluations", 0) > previous_lineage.get(
            "followup", {}
        ).get("end_evaluations", 0):
            lineages[entry["lineage"]] = {"kind": entry["kind"], "cohort": entry["cohort"], **result}
    rows = sorted(decisions.values(), key=lambda row: (row["lineage"], row["iteration"], row["decision_id"]))
    missing = Counter(reason for row in rows for reason in row["missingness"])
    cohorts: dict[str, list[str]] = defaultdict(list)
    for lineage, evidence in lineages.items():
        comparison = evidence["identity"].get("comparison_id")
        if comparison:
            cohorts[comparison].append(lineage)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decisions": rows,
        "audit": {
            "archives": audits,
            "duplicate_decisions_removed": duplicate_count,
            "unique_decisions": len(rows),
            "missingness": dict(missing),
            "comparison_groups": dict(cohorts),
        },
        "lineages": lineages,
    }
