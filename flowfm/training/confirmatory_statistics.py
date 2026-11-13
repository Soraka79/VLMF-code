"""Predeclared statistics and gate for independent A0-MR confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from flowfm.training.baseline_evaluation import metric_payload


RARE_NATIVE_LABELS = ("backdoor", "mitm", "ransomware")
NF_CSE_RARE_COARSE_LABELS = ("brute_force", "exploit", "botnet")
MIN_MEAN_XGBOOST_GAIN = 0.01
MAX_RARE_MEAN_RECALL_DROP = 0.02


def confirmatory_metric_payload(
    probabilities,
    targets,
    label_names: tuple[str, ...],
) -> dict[str, object]:
    """Extend the stable baseline metrics only for confirmation reports."""

    values = np.asarray(probabilities, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.int64)
    payload = metric_payload(values, target_values, label_names)
    label_ids = np.arange(len(label_names), dtype=np.int64)
    predictions = values.argmax(axis=1)
    precision, _, per_class_f1, support = precision_recall_fscore_support(
        target_values,
        predictions,
        labels=label_ids,
        zero_division=0,
    )
    return {
        **payload,
        "nll": float(log_loss(target_values, values, labels=label_ids)),
        "per_class_precision": {
            label: float(precision[index])
            for index, label in enumerate(label_names)
        },
        "per_class_f1": {
            label: float(per_class_f1[index])
            for index, label in enumerate(label_names)
        },
        "per_class_support": {
            label: int(support[index])
            for index, label in enumerate(label_names)
        },
    }


def _targets_and_predictions(targets, probabilities, class_count: int):
    target_values = np.asarray(targets, dtype=np.int64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if target_values.ndim != 1 or target_values.size == 0:
        raise ValueError("bootstrap targets must be a non-empty vector")
    if probability_values.shape != (target_values.size, class_count):
        raise ValueError("bootstrap probabilities do not align with targets")
    if np.any(target_values < 0) or np.any(target_values >= class_count):
        raise ValueError("bootstrap targets fall outside the label space")
    return target_values, probability_values.argmax(axis=1)


def _macro_f1_gradient(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
) -> tuple[float, np.ndarray]:
    labels = np.arange(class_count, dtype=np.int64)
    matrix = confusion_matrix(targets, predictions, labels=labels).astype(
        np.float64
    )
    observed = float(
        f1_score(
            targets,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    gradient = np.zeros_like(matrix)
    for class_id in range(class_count):
        true_positive = matrix[class_id, class_id]
        false_positive = matrix[:, class_id].sum() - true_positive
        false_negative = matrix[class_id, :].sum() - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        if denominator <= 0.0:
            continue
        negative = -2.0 * true_positive / (denominator * denominator)
        gradient[:, class_id] += negative / class_count
        gradient[class_id, :] += negative / class_count
        gradient[class_id, class_id] = (
            2.0 * (denominator - 2.0 * true_positive)
            / (denominator * denominator)
            / class_count
        )
    return observed, gradient


def _group_influence(
    targets,
    probabilities,
    group_codes: np.ndarray,
    group_count: int,
    class_count: int,
) -> tuple[float, np.ndarray]:
    target_values, predictions = _targets_and_predictions(
        targets, probabilities, class_count
    )
    observed, gradient = _macro_f1_gradient(
        target_values, predictions, class_count
    )
    row_influence = gradient[target_values, predictions]
    grouped = np.bincount(
        group_codes,
        weights=row_influence,
        minlength=group_count,
    ).astype(np.float64, copy=False)
    grouped -= grouped.mean()
    return observed, grouped


def paired_cluster_bootstrap_macro_f1(
    outputs: Mapping[str, Mapping[str, object]],
    first_probabilities: Mapping[str, np.ndarray],
    second_probabilities: Mapping[str, np.ndarray],
    group_ids: Sequence[str],
    *,
    class_count: int,
    resamples: int = 2000,
    seed: int = 20260724,
    batch_size: int = 32,
) -> dict[str, object]:
    """Paired Poisson cluster bootstrap using macro-F1 influence values.

    One Poisson weight is drawn per leakage group and reused for both models and
    every missingness condition. The macro-F1 functional is linearized at the
    observed paired confusion matrices, which keeps a 500k-row group bootstrap
    tractable while preserving group-level dependence.
    """

    if type(resamples) is not int or resamples < 200:
        raise ValueError("bootstrap resamples must be an integer of at least 200")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("bootstrap batch_size must be positive")
    conditions = tuple(outputs)
    if not conditions or set(first_probabilities) != set(conditions) or set(
        second_probabilities
    ) != set(conditions):
        raise ValueError("bootstrap probability mappings must match outputs")
    if len(group_ids) != len(next(iter(outputs.values()))["targets"]):
        raise ValueError("group IDs must align with output rows")
    unique_groups, group_codes = np.unique(
        np.asarray(group_ids, dtype="U64"), return_inverse=True
    )
    group_count = len(unique_groups)
    if group_count < 2:
        raise ValueError("cluster bootstrap requires at least two groups")

    observed_deltas = []
    influence_columns = []
    for condition in conditions:
        targets = outputs[condition]["targets"]
        if len(targets) != len(group_codes):
            raise ValueError("all bootstrap conditions must share sample order")
        first_observed, first_influence = _group_influence(
            targets,
            first_probabilities[condition],
            group_codes,
            group_count,
            class_count,
        )
        second_observed, second_influence = _group_influence(
            targets,
            second_probabilities[condition],
            group_codes,
            group_count,
            class_count,
        )
        observed_deltas.append(first_observed - second_observed)
        influence_columns.append(first_influence - second_influence)
    observed = np.asarray(observed_deltas, dtype=np.float64)
    influence = np.column_stack(influence_columns)
    mean_influence = influence.mean(axis=1, keepdims=True)
    all_influence = np.column_stack((influence, mean_influence))
    all_observed = np.concatenate((observed, (float(observed.mean()),)))

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((resamples, len(conditions) + 1), dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        weights = rng.poisson(1.0, size=(stop - start, group_count))
        bootstrap[start:stop] = all_observed + weights @ all_influence
    lower = np.quantile(bootstrap, 0.025, axis=0)
    upper = np.quantile(bootstrap, 0.975, axis=0)
    summaries = {
        condition: {
            "delta_macro_f1": float(observed[index]),
            "ci95_lower": float(lower[index]),
            "ci95_upper": float(upper[index]),
        }
        for index, condition in enumerate(conditions)
    }
    summaries["mean_across_conditions"] = {
        "delta_macro_f1": float(all_observed[-1]),
        "ci95_lower": float(lower[-1]),
        "ci95_upper": float(upper[-1]),
    }
    return {
        "format_version": 1,
        "method": "paired_poisson_cluster_bootstrap_linearized_macro_f1",
        "pairing": "same_leakage_group_weights_for_models_and_conditions",
        "resamples": resamples,
        "seed": seed,
        "cluster_count": group_count,
        "confidence_level": 0.95,
        "summaries": summaries,
    }


def evaluate_confirmatory_gate(
    *,
    xgboost_metrics: Mapping[str, Mapping[str, object]],
    fixed_a0_metrics: Mapping[str, Mapping[str, object]],
    locked_metrics: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, object],
    rare_labels: Sequence[str] = RARE_NATIVE_LABELS,
    rare_label_definition: str = (
        "official full-source native prevalence at or below 0.1%"
    ),
    maximum_rare_mean_recall_drop: float = MAX_RARE_MEAN_RECALL_DROP,
) -> dict[str, object]:
    """Apply the predeclared confirmatory gate without post-hoc choices."""

    conditions = tuple(locked_metrics)
    if conditions != ("clean", "missing_30pct", "missing_50pct"):
        raise ValueError("confirmatory gate requires clean/30%/50% conditions")
    if set(xgboost_metrics) != set(conditions) or set(fixed_a0_metrics) != set(
        conditions
    ):
        raise ValueError("confirmatory metric conditions do not align")
    if (
        isinstance(rare_labels, (str, bytes))
        or not isinstance(rare_labels, Sequence)
        or not rare_labels
    ):
        raise ValueError("rare_labels must be a non-empty sequence")
    normalized_rare_labels = tuple(rare_labels)
    if (
        len(set(normalized_rare_labels)) != len(normalized_rare_labels)
        or not all(isinstance(label, str) and label for label in normalized_rare_labels)
    ):
        raise ValueError("rare_labels must contain unique non-empty strings")
    if not isinstance(rare_label_definition, str) or not rare_label_definition:
        raise ValueError("rare_label_definition must be a non-empty string")
    if (
        isinstance(maximum_rare_mean_recall_drop, bool)
        or not np.isfinite(maximum_rare_mean_recall_drop)
        or maximum_rare_mean_recall_drop < 0
    ):
        raise ValueError("maximum_rare_mean_recall_drop must be non-negative")

    xgb_deltas = {
        condition: float(
            locked_metrics[condition]["macro_f1"]
            - xgboost_metrics[condition]["macro_f1"]
        )
        for condition in conditions
    }
    fixed_deltas = {
        condition: float(
            locked_metrics[condition]["macro_f1"]
            - fixed_a0_metrics[condition]["macro_f1"]
        )
        for condition in conditions
    }
    mean_xgb_gain = float(np.mean(tuple(xgb_deltas.values())))
    rare_differences = []
    rare_zero_out = []
    for condition in conditions:
        locked_recall = locked_metrics[condition]["per_class_recall"]
        xgb_recall = xgboost_metrics[condition]["per_class_recall"]
        missing_labels = set(normalized_rare_labels) - set(locked_recall)
        if missing_labels or not set(normalized_rare_labels).issubset(xgb_recall):
            raise ValueError(
                "rare labels are absent from confirmatory per-class recall: "
                f"{sorted(missing_labels | (set(normalized_rare_labels) - set(xgb_recall)))}"
            )
        for label in normalized_rare_labels:
            rare_differences.append(
                float(locked_recall[label] - xgb_recall[label])
            )
            if xgb_recall[label] >= 0.10 and locked_recall[label] <= 0.0:
                rare_zero_out.append({"condition": condition, "label": label})
    rare_mean_delta = float(np.mean(rare_differences))
    bootstrap_mean = bootstrap["summaries"]["mean_across_conditions"]
    checks = {
        "noninferior_to_xgboost_all_conditions": all(
            delta >= -1e-12 for delta in xgb_deltas.values()
        ),
        "mean_xgboost_gain_at_least_0p01": (
            mean_xgb_gain >= MIN_MEAN_XGBOOST_GAIN
        ),
        "missing_conditions_better_than_fixed_a0": (
            fixed_deltas["missing_30pct"] > 0.0
            and fixed_deltas["missing_50pct"] > 0.0
        ),
        "bootstrap_mean_ci_lower_above_zero": (
            bootstrap_mean["ci95_lower"] > 0.0
        ),
        "no_rare_class_overall_recall_collapse": (
            rare_mean_delta >= -maximum_rare_mean_recall_drop
            and not rare_zero_out
        ),
    }
    return {
        "format_version": 1,
        "predeclared": True,
        "thresholds": {
            "minimum_mean_xgboost_macro_f1_gain": MIN_MEAN_XGBOOST_GAIN,
            "maximum_rare_mean_recall_drop": maximum_rare_mean_recall_drop,
            "rare_labels": list(normalized_rare_labels),
            "rare_label_definition": rare_label_definition,
        },
        "xgboost_macro_f1_deltas": xgb_deltas,
        "fixed_a0_macro_f1_deltas": fixed_deltas,
        "mean_xgboost_macro_f1_gain": mean_xgb_gain,
        "rare_mean_recall_delta_vs_xgboost": rare_mean_delta,
        "rare_zero_out_failures": rare_zero_out,
        "checks": checks,
        "passed": all(checks.values()),
        "full_matrix_allowed": all(checks.values()),
    }


__all__ = [
    "MAX_RARE_MEAN_RECALL_DROP",
    "MIN_MEAN_XGBOOST_GAIN",
    "NF_CSE_RARE_COARSE_LABELS",
    "RARE_NATIVE_LABELS",
    "confirmatory_metric_payload",
    "evaluate_confirmatory_gate",
    "paired_cluster_bootstrap_macro_f1",
]
