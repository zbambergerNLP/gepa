"""Evaluate observed-action breakthrough predictions with temporal and lineage holdouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "parent_score",
    "incumbent_score",
    "incumbent_gap",
    "training_score",
    "prompt_chars",
    "module_prompt_chars",
    "depth",
    "prior_parent_decisions",
    "prior_parent_breakthroughs",
    "evaluations_before_action",
    "evaluations_since_record",
)
MODEL_NAMES = ("action_rates", "candidate_stage", "action_effects", "candidate_action_interactions")
REGULARIZATION_C = 0.1
MAX_COMPONENTS = 5
MAX_FIT_ITERATIONS = 2000
RANDOM_SEED = 0
ACTION_PRIOR_STRENGTH = 10.0
FORWARD_FRACTIONS = (0.5, 0.75)
CALIBRATION_BINS = 5
PROBABILITY_EPSILON = 1e-9


def numeric_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    """Select only declared, pre-decision numerical features.

    Args:
        rows: Normalized decisions.

    Returns:
        Dense feature matrix with missing values represented by NaN.
    """
    return np.asarray([[row["features"].get(name) for name in NUMERIC_FEATURES] for row in rows], dtype=float)


def categories(rows: list[dict[str, Any]], *, actions: bool) -> list[list[str]]:
    """Extract module and optional action/section categories.

    Args:
        rows: Normalized decisions.
        actions: Include semantic actions and sections.

    Returns:
        Categorical values without raw candidate or lineage IDs.
    """
    return [
        [row["module"], *([row["selection"]["action"], row["selection"]["section"]] if actions else [])] for row in rows
    ]


def predict_fold(
    train: list[dict[str, Any]], test: list[dict[str, Any]], model: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit preprocessing and one fixed model using training observations only.

    Args:
        train: Earlier decisions or complete training lineages.
        test: Held-out decisions with outcomes ignored during prediction.
        model: One of the predefined model names.

    Returns:
        Predicted breakthrough probabilities and fit diagnostics.
    """
    y = np.asarray([row["outcome"]["record_break"] for row in train], dtype=int)
    prior = float((y.sum() + 1) / (len(y) + 2))
    diagnostics: dict[str, Any] = {"train_count": len(train), "train_events": int(y.sum()), "pca_components": 0}
    if model == "action_rates":
        totals = Counter(row["selection"]["action"] for row in train)
        wins = Counter(row["selection"]["action"] for row in train if row["outcome"]["record_break"])
        predictions = [
            (wins[row["selection"]["action"]] + ACTION_PRIOR_STRENGTH * prior)
            / (totals[row["selection"]["action"]] + ACTION_PRIOR_STRENGTH)
            for row in test
        ]
        return np.asarray(predictions), diagnostics
    if len(set(y)) < 2:
        diagnostics["fallback"] = "training fold has one outcome class; smoothed prevalence only"
        return np.full(len(test), prior), diagnostics
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    x_train = np.asarray(scaler.fit_transform(imputer.fit_transform(numeric_matrix(train))), dtype=float)
    x_test = np.asarray(scaler.transform(imputer.transform(numeric_matrix(test))), dtype=float)
    unique_parents = {
        (row["parent_sha256"], tuple(row["parent_failure_pattern"])): row["parent_failure_pattern"] for row in train
    }
    failures = np.asarray(list(unique_parents.values()), dtype=float)
    n_components = min(MAX_COMPONENTS, len(failures) - 1, failures.shape[1])
    if n_components > 0 and np.any(np.var(failures, axis=0) > 0):
        pca = PCA(n_components=n_components, svd_solver="full", random_state=RANDOM_SEED).fit(failures)
        behavior_scaler = StandardScaler()
        train_behavior = behavior_scaler.fit_transform(pca.transform([row["parent_failure_pattern"] for row in train]))
        test_behavior = behavior_scaler.transform(pca.transform([row["parent_failure_pattern"] for row in test]))
        x_train = np.column_stack((x_train, train_behavior))
        x_test = np.column_stack((x_test, test_behavior))
        diagnostics["pca_components"] = n_components
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    include_actions = model != "candidate_stage"
    if include_actions:
        # The chosen section is known only after selection, so it cannot enter the candidate-only baseline.
        section_scaler = StandardScaler()
        x_train = np.column_stack(
            (x_train, section_scaler.fit_transform([[row["features"]["section_chars"]] for row in train]))
        )
        x_test = np.column_stack(
            (x_test, section_scaler.transform([[row["features"]["section_chars"]] for row in test]))
        )
    train_categories = np.asarray(encoder.fit_transform(categories(train, actions=include_actions)), dtype=float)
    test_categories = np.asarray(encoder.transform(categories(test, actions=include_actions)), dtype=float)
    if model == "candidate_action_interactions":
        action_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_actions = np.asarray(
            action_encoder.fit_transform([[row["selection"]["choice"]] for row in train]), dtype=float
        )
        test_actions = np.asarray(action_encoder.transform([[row["selection"]["choice"]] for row in test]), dtype=float)
        train_categories = np.column_stack(
            (train_categories, (x_train[:, :, None] * train_actions[:, None, :]).reshape(len(train), -1))
        )
        test_categories = np.column_stack(
            (test_categories, (x_test[:, :, None] * test_actions[:, None, :]).reshape(len(test), -1))
        )
    x_train = np.column_stack((x_train, train_categories))
    x_test = np.column_stack((x_test, test_categories))
    estimator = LogisticRegression(C=REGULARIZATION_C, max_iter=MAX_FIT_ITERATIONS, random_state=RANDOM_SEED)
    estimator.fit(x_train, y)
    if estimator.n_iter_[0] >= MAX_FIT_ITERATIONS:
        raise ValueError("Signal model did not converge; refusing to report an unchecked fit")
    diagnostics["feature_count"] = x_train.shape[1]
    return estimator.predict_proba(x_test)[:, 1], diagnostics


def score_predictions(rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    """Report proper scoring rules and rare-event discrimination without accuracy inflation.

    Args:
        rows: Held-out decisions with observed outcomes.
        predictions: Corresponding predicted probabilities.

    Returns:
        Metrics, calibration bins, and an explicitly unavailable PR curve for one-class folds.
    """
    labels = np.asarray([row["outcome"]["record_break"] for row in rows], dtype=int)
    probabilities = np.clip(predictions, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    calibration = []
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    for low, high in pairwise(edges):
        mask = (probabilities >= low) & (probabilities < high)
        if mask.any():
            calibration.append(
                {
                    "lower": float(low),
                    "upper": float(high),
                    "count": int(mask.sum()),
                    "mean_prediction": float(probabilities[mask].mean()),
                    "observed_rate": float(labels[mask].mean()),
                }
            )
    discrimination = len(set(labels)) == 2
    curve = None
    if discrimination:
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        curve = {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}
    return {
        "count": len(rows),
        "events": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)) if discrimination else None,
        "discrimination_available": discrimination,
        "limitation": None
        if discrimination
        else "One outcome class; breakthrough discrimination cannot be established",
        "calibration": calibration,
        "precision_recall": curve,
    }


def make_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Construct disjoint temporal windows, lineage holdouts, and budget extensions.

    Args:
        rows: Unique primary decisions within one scientific comparison group.

    Returns:
        Explicit train/test row lists; different evaluation schemes remain separate.
    """
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lineage[row["lineage"]].append(row)
    folds = []
    for lineage, members in sorted(by_lineage.items()):
        members.sort(key=lambda row: row["iteration"])
        boundaries = [int(len(members) * fraction) for fraction in FORWARD_FRACTIONS] + [len(members)]
        for start, end in pairwise(boundaries):
            if start and end > start:
                folds.append(
                    {
                        "id": f"forward:{lineage}:{start}-{end}",
                        "scheme": "forward",
                        "train": members[:start],
                        "test": members[start:end],
                    }
                )
        other = [row for row in rows if row["lineage"] != lineage]
        if other:
            folds.append({"id": f"lineage:{lineage}", "scheme": "lineage", "train": other, "test": members})
        standard = [row for row in members if row["phase"] == "standard"]
        expanded = [row for row in members if row["phase"] == "expanded"]
        if standard and expanded:
            folds.append({"id": f"extension:{lineage}", "scheme": "extension", "train": standard, "test": expanded})
    return folds


def evaluate_signal(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the prespecified models without pooling incompatible protocols or pilots.

    Args:
        decisions: Deduplicated primary and supplementary observations.

    Returns:
        Fold metrics and predictions with a conservative evidence assessment.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row["kind"] == "search" and row["outcome"]["record_break"] is not None:
            groups[row["comparison_id"]].append(row)
    results = []
    predictions = []
    for comparison, rows in sorted(groups.items()):
        for fold in make_folds(rows):
            for model in MODEL_NAMES:
                probabilities, fit = predict_fold(fold["train"], fold["test"], model)
                metrics = score_predictions(fold["test"], probabilities)
                late_indices = [i for i, row in enumerate(fold["test"]) if row["phase"] == "expanded"]
                late = (
                    score_predictions([fold["test"][i] for i in late_indices], probabilities[late_indices])
                    if late_indices
                    else None
                )
                results.append(
                    {
                        "fold": fold["id"],
                        "scheme": fold["scheme"],
                        "comparison_id": comparison,
                        "model": model,
                        "fit": fit,
                        "metrics": metrics,
                        "expanded_metrics": late,
                        "train_decisions": [row["decision_id"] for row in fold["train"]],
                        "test_decisions": [row["decision_id"] for row in fold["test"]],
                    }
                )
                predictions.extend(
                    {
                        "fold": fold["id"],
                        "model": model,
                        "decision_id": row["decision_id"],
                        "probability": float(probability),
                        "observed_record_break": row["outcome"]["record_break"],
                    }
                    for row, probability in zip(fold["test"], probabilities, strict=True)
                )
    by_fold: dict[str, dict[str, Any]] = defaultdict(dict)
    for result in results:
        by_fold[result["fold"]][result["model"]] = result["metrics"]
    comparisons = []
    for fold_id, models in by_fold.items():
        candidate = models["candidate_action_interactions"]
        comparisons.append(
            {
                "fold": fold_id,
                "events": candidate["events"],
                "stage_log_loss_minus_action_rates": models["candidate_stage"]["log_loss"]
                - models["action_rates"]["log_loss"],
                "action_effects_log_loss_minus_stage": models["action_effects"]["log_loss"]
                - models["candidate_stage"]["log_loss"],
                "interaction_log_loss_minus_action_effects": candidate["log_loss"]
                - models["action_effects"]["log_loss"],
                "interaction_log_loss_minus_stage": candidate["log_loss"] - models["candidate_stage"]["log_loss"],
            }
        )
    eventful = [row for row in comparisons if row["events"] > 0]
    improved = [
        row
        for row in eventful
        if row["interaction_log_loss_minus_action_effects"] < 0 and row["interaction_log_loss_minus_stage"] < 0
    ]
    held_lineages = [row for row in eventful if row["fold"].startswith("lineage:")]
    held_improved = [row for row in improved if row["fold"].startswith("lineage:")]
    stage_improved = sum(row["stage_log_loss_minus_action_rates"] < 0 for row in comparisons)
    action_improved = sum(row["action_effects_log_loss_minus_stage"] < 0 for row in comparisons)
    pilot_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row["kind"] == "pilot":
            pilot_groups[row["matched_group"]].append({"cohort": row["cohort"], "outcome": row["outcome"]})
    assessment = (
        f"Candidate/stage features lower log loss than action averages in {stage_improved} of {len(comparisons)} folds. "
        f"Adding action/section effects improves on candidate/stage in {action_improved} of {len(comparisons)} folds. "
        f"Candidate-action interactions improve log loss against both simpler context models in {len(improved)} of "
        f"{len(eventful)} folds containing breakthroughs, including {len(held_improved)} of {len(held_lineages)} complete-lineage holdouts. "
        "These overlapping observational tests do not establish that a new policy improves plateau escape."
    )
    return {
        "settings": {
            "regularization_c": REGULARIZATION_C,
            "max_pca_components": MAX_COMPONENTS,
            "random_seed": RANDOM_SEED,
            "forward_train_fractions": FORWARD_FRACTIONS,
            "model_names": MODEL_NAMES,
            "class_weight": None,
            "hyperparameter_search": False,
        },
        "folds": results,
        "predictions": predictions,
        "comparisons": comparisons,
        "assessment": assessment,
        "pilot_matched_opportunities": dict(pilot_groups),
        "limitations": [
            "Forward folds can share parents with their training prefix; complete-lineage holdouts test transfer separately.",
            "The same decision can be tested in different schemes; do not pool their metrics as independent samples.",
            "Pilots measure training transfer and are excluded from validation-breakthrough model fitting.",
            "Held-out test outcomes are neither read nor used for model fitting, tuning, or selection.",
            "Zero-support actions have no observed counterfactual reward; no new search tree is simulated.",
        ],
    }
