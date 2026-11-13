"""Shared task and metric contract for fair NF-UNSW baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)

from flowfm.data.labels import (
    NF_UNSW_LABEL_SPACE,
    DatasetLabelSpace,
    encode_supervised_targets,
)


TASKS = ("binary", "coarse", "native")


def task_names(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return TASKS
    if requested not in TASKS:
        raise ValueError(f"unsupported task: {requested}")
    return (requested,)


def task_labels(
    task: str,
    label_space: DatasetLabelSpace = NF_UNSW_LABEL_SPACE,
) -> tuple[str, ...]:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    if not isinstance(label_space, DatasetLabelSpace):
        raise TypeError("label_space must be a DatasetLabelSpace")
    return tuple(getattr(label_space, f"{task}_labels"))


def target_arrays(
    samples: Sequence,
    label_space: DatasetLabelSpace = NF_UNSW_LABEL_SPACE,
) -> dict[str, np.ndarray]:
    encoded = encode_supervised_targets(tuple(samples), label_space)
    return {
        task: getattr(encoded, task).cpu().numpy().astype(np.int64, copy=False)
        for task in TASKS
    }


def _ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if selected.any():
            accuracy = float((predictions[selected] == targets[selected]).mean())
            total += float(selected.mean()) * abs(
                accuracy - float(confidence[selected].mean())
            )
    return float(total)


def metric_payload(
    probabilities: np.ndarray,
    targets: np.ndarray,
    label_names: tuple[str, ...],
) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("metric targets must be a non-empty rank-1 array")
    if not isinstance(label_names, tuple) or not label_names:
        raise ValueError("label_names must be a non-empty tuple")
    class_count = len(label_names)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (
        targets.size,
        class_count,
    ):
        raise ValueError(
            "probability columns must exactly match the versioned label space"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("model probabilities must be finite")
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("model probability rows must sum to one")
    if np.any(targets < 0) or np.any(targets >= class_count):
        raise ValueError("metric targets fall outside the versioned label space")

    label_ids = np.arange(class_count, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    recalls = recall_score(
        targets,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    matrix_values = confusion_matrix(
        targets,
        predictions,
        labels=label_ids,
    )
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    per_class_pr_auc = {}
    for class_id, label in enumerate(label_names):
        positives = int(one_hot[:, class_id].sum())
        if positives == 0:
            score = 0.0
        elif positives == targets.size:
            score = 1.0
        else:
            score = float(
                average_precision_score(
                    one_hot[:, class_id],
                    probabilities[:, class_id],
                )
            )
        per_class_pr_auc[label] = score

    class_counts = np.bincount(targets, minlength=class_count)
    prediction_counts = np.bincount(predictions, minlength=class_count)
    return {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "balanced_accuracy": float(recalls.mean()),
        "macro_pr_auc": float(np.mean(tuple(per_class_pr_auc.values()))),
        "per_class_pr_auc": per_class_pr_auc,
        "ece": _ece(probabilities, targets),
        "brier": float(np.mean((probabilities - one_hot) ** 2)),
        "brier_definition": (
            "mean_squared_probability_error_over_samples_and_classes"
        ),
        "per_class_recall": {
            label: float(recalls[index])
            for index, label in enumerate(label_names)
        },
        "confusion_matrix": matrix_values.tolist(),
        "class_counts": {
            label: int(class_counts[index])
            for index, label in enumerate(label_names)
        },
        "predicted_class_counts": {
            label: int(prediction_counts[index])
            for index, label in enumerate(label_names)
        },
        "sample_count": int(targets.size),
        "label_names": list(label_names),
    }


__all__ = [
    "TASKS",
    "metric_payload",
    "target_arrays",
    "task_labels",
    "task_names",
]
