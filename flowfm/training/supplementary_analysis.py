"""Post-evaluation analyses declared in the supplementary protocol.

Every function works on validation or sealed-test probabilities that were
archived after the single sealed-test open. Policies are always chosen from
validation records; test labels only score policies that were already chosen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import numpy as np

from flowfm.training.a0_mr_protocol import FIXED_A0_ALPHA_XGBOOST, FORMAL_CONDITIONS
from flowfm.training.learned_ensemble import (
    CONDITIONS as STACKING_CONDITIONS,
    fit_logistic_stacker,
    make_group_folds,
    predict_locked_stacker,
)
from flowfm.training.probability_fusion import pool_probabilities
from flowfm.training.vlmf_strengthening import (
    MAIN_CONDITIONS,
    rank_condition_candidates,
    rank_global_candidates,
    validate_condition_searches,
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion; None when trials is 0."""

    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return low, high


def macro_f1(probabilities, targets, class_count: int) -> float:
    """Macro-F1 of the argmax prediction; matches sklearn with zero_division=0."""

    predicted = np.asarray(probabilities).argmax(axis=1)
    actual = np.asarray(targets, dtype=np.int64)
    matrix = np.bincount(
        actual * class_count + predicted, minlength=class_count * class_count
    ).reshape(class_count, class_count)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = 2 * true_positive + (matrix.sum(axis=0) - true_positive) + (
        matrix.sum(axis=1) - true_positive
    )
    scores = np.divide(
        2 * true_positive, denominator, out=np.zeros(class_count), where=denominator > 0
    )
    return float(scores.mean())


def dominant_group_labels(group_ids, targets):
    """Return group codes, one label per group (its most frequent target), sizes."""

    codes, groups = np.unique(np.asarray(group_ids), return_inverse=True)
    labels = np.asarray(targets, dtype=np.int64)
    class_count = int(labels.max()) + 1
    counts = np.zeros((codes.size, class_count), dtype=np.int64)
    np.add.at(counts, (groups, labels), 1)
    return groups, counts.argmax(axis=1), counts.sum(axis=1)


def stratified_group_subsample(group_ids, targets, rows: int, rng) -> np.ndarray:
    """Row indices of whole groups, about ``rows`` in total, class-proportional by group label.

    Every label that has at least one group keeps at least one group, so small
    subsamples still contain every class.
    """

    groups, group_label, group_size = dominant_group_labels(group_ids, targets)
    total = int(group_size.sum())
    chosen = np.zeros(group_size.size, dtype=bool)
    for label in np.unique(group_label):
        members = np.flatnonzero(group_label == label)
        quota = rows * group_size[members].sum() / total
        order = rng.permutation(members)
        take = max(1, int(np.searchsorted(np.cumsum(group_size[order]), quota) + 1))
        chosen[order[: min(take, order.size)]] = True
    return np.flatnonzero(chosen[groups])


def stratified_group_halves(group_ids, targets, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into two group-disjoint halves, balanced within each group label."""

    groups, group_label, group_size = dominant_group_labels(group_ids, targets)
    side = np.zeros(group_size.size, dtype=bool)
    for label in np.unique(group_label):
        order = rng.permutation(np.flatnonzero(group_label == label))
        half = group_size[order].sum() / 2
        side[order[np.cumsum(group_size[order]) <= half]] = True
    in_a = side[groups]
    return np.flatnonzero(in_a), np.flatnonzero(~in_a)


def endpoint_outputs(arrays: Mapping[str, np.ndarray], conditions, index=None):
    """``{condition: {xgboost, schemaflow_pretrained, targets}}`` from an archive."""

    take = slice(None) if index is None else index
    targets = np.asarray(arrays["targets"])[take]
    return {
        condition: {
            "xgboost": np.asarray(arrays[f"{condition}__xgboost"])[take],
            "schemaflow_pretrained": np.asarray(arrays[f"{condition}__schemaflow_pretrained"])[take],
            "targets": targets,
        }
        for condition in conditions
    }


def policy_macro_f1(outputs, policies, class_count: int) -> dict[str, float]:
    """Macro-F1 of each condition's pooled probabilities under ``policies``."""

    return {
        condition: macro_f1(
            pool_probabilities(
                values["xgboost"],
                values["schemaflow_pretrained"],
                alpha=policies[condition]["alpha"],
                method=policies[condition]["method"],
            ),
            values["targets"],
            class_count,
        )
        for condition, values in outputs.items()
    }


def fixed_policies(conditions) -> dict[str, dict[str, object]]:
    return {name: {"method": "linear", "alpha": FIXED_A0_ALPHA_XGBOOST} for name in conditions}


def validation_controls(selection: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Global validation fusion and always-conditioned linear/log from a stored selection.

    Both reuse the frozen 42-candidate validation records and tie order of the
    strengthening suite; neither uses a family gate or fallback.
    """

    searches = {
        name: selection["random_condition_selection"]["linear_or_log"][name]
        for name in MAIN_CONDITIONS
    }
    validated = validate_condition_searches(searches)
    best = rank_global_candidates(validated)[0]
    always = {}
    for name in MAIN_CONDITIONS:
        top = rank_condition_candidates(validated[name])[0]
        always[name] = {"method": top["method"], "alpha": top["alpha"]}
    return {
        "global_validation_fusion": {
            name: {"method": best["method"], "alpha": best["alpha"]} for name in MAIN_CONDITIONS
        },
        "always_conditioned_linear_or_log": always,
    }


def stacking_scores(validation_arrays, test_arrays, class_count: int) -> dict[str, object]:
    """Refit the frozen logistic stacker on validation and score it on test."""

    validation = {
        name: {
            "xgboost": np.asarray(validation_arrays[f"{name}__xgboost"]),
            "schemaflow": np.asarray(validation_arrays[f"{name}__schemaflow_pretrained"]),
        }
        for name in STACKING_CONDITIONS
    }
    targets = np.asarray(validation_arrays["targets"])
    folds = make_group_folds(targets, np.asarray(validation_arrays["group_ids"]))
    fitted = fit_logistic_stacker(validation, targets, folds)
    test_targets = np.asarray(test_arrays["targets"])
    scores = {}
    for name in STACKING_CONDITIONS:
        probabilities = predict_locked_stacker(
            fitted["model"],
            np.asarray(test_arrays[f"{name}__xgboost"]),
            np.asarray(test_arrays[f"{name}__schemaflow_pretrained"]),
        )
        scores[name] = macro_f1(probabilities, test_targets, class_count)
    return {"selected_C": fitted["selection"]["selected_C"], "test_macro_f1": scores}


def _selection_metrics(probabilities, targets, label_names):
    """The three metrics validation selection reads, with the same sklearn calls as metric_payload."""

    from sklearn.metrics import f1_score, matthews_corrcoef, recall_score

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    label_ids = np.arange(len(label_names), dtype=np.int64)
    recalls = recall_score(targets, predictions, labels=label_ids, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(targets, predictions, labels=label_ids, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "balanced_accuracy": float(recalls.mean()),
    }


@contextmanager
def selection_metrics_only():
    """Run frozen validation selection without the PR-AUC/ECE fields it never reads."""

    from flowfm.training import a0_mr_protocol, probability_fusion

    saved = probability_fusion.metric_payload, a0_mr_protocol.metric_payload
    probability_fusion.metric_payload = _selection_metrics
    a0_mr_protocol.metric_payload = _selection_metrics
    try:
        yield
    finally:
        probability_fusion.metric_payload, a0_mr_protocol.metric_payload = saved


def policy_agreement(first, second, conditions=FORMAL_CONDITIONS) -> float:
    return float(
        np.mean(
            [
                (first[name]["method"], round(float(first[name]["alpha"]), 12))
                == (second[name]["method"], round(float(second[name]["alpha"]), 12))
                for name in conditions
            ]
        )
    )


__all__ = [
    "dominant_group_labels",
    "endpoint_outputs",
    "fixed_policies",
    "macro_f1",
    "policy_agreement",
    "policy_macro_f1",
    "selection_metrics_only",
    "stacking_scores",
    "stratified_group_halves",
    "stratified_group_subsample",
    "validation_controls",
    "wilson_interval",
]
