"""Validation-selected probability fusion for tabular and neural classifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from flowfm.training.baseline_evaluation import metric_payload


POOLING_METHODS = ("linear", "log")


def default_alpha_grid(step: float = 0.05) -> tuple[float, ...]:
    """Return a stable grid that includes both single-model endpoints."""

    if not isinstance(step, (int, float)) or isinstance(step, bool):
        raise TypeError("alpha step must be numeric")
    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("alpha step must be in (0, 1]")
    count = int(round(1.0 / step))
    if not np.isclose(count * step, 1.0, atol=1e-12, rtol=0):
        raise ValueError("alpha step must divide one exactly")
    return tuple(round(index * step, 12) for index in range(count + 1))


def _probability_matrix(
    values,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty rank-2 probability matrix")
    if expected_shape is not None and matrix.shape != expected_shape:
        raise ValueError(f"{name} shape does not match the paired probabilities")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{name} probability rows must sum to one")
    return matrix


def probability_sha256(values) -> str:
    """Hash shape, dtype, and canonical float64 probability bytes."""

    matrix = _probability_matrix(values, name="probabilities")
    canonical = np.ascontiguousarray(matrix.astype("<f8", copy=False))
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def pool_probabilities(
    xgboost_probabilities,
    schemaflow_probabilities,
    *,
    alpha: float,
    method: str = "linear",
) -> np.ndarray:
    """Pool probabilities, where alpha is the XGBoost contribution."""

    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError("alpha must be numeric")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if method not in POOLING_METHODS:
        raise ValueError(f"unsupported pooling method: {method}")
    tree = _probability_matrix(xgboost_probabilities, name="XGBoost")
    neural = _probability_matrix(
        schemaflow_probabilities,
        name="SchemaFlow",
        expected_shape=tree.shape,
    )

    if method == "linear":
        pooled = alpha * tree + (1.0 - alpha) * neural
    else:
        epsilon = np.finfo(np.float64).tiny
        log_values = (
            alpha * np.log(np.clip(tree, epsilon, 1.0))
            + (1.0 - alpha) * np.log(np.clip(neural, epsilon, 1.0))
        )
        log_values -= log_values.max(axis=1, keepdims=True)
        pooled = np.exp(log_values)
    row_sums = pooled.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0.0):
        raise ValueError("pooled probabilities cannot be normalized")
    return pooled / row_sums


def _alpha_grid(values: Iterable[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("alpha grid must be non-empty")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("alpha grid values must be finite and in [0, 1]")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("alpha grid must be strictly increasing and unique")
    return normalized


def select_fusion(
    xgboost_probabilities,
    schemaflow_probabilities,
    targets,
    label_names: tuple[str, ...],
    *,
    alpha_grid: Sequence[float] | None = None,
    methods: Sequence[str] = ("linear",),
) -> dict[str, object]:
    """Select a low-dimensional fusion policy using validation targets only."""

    tree = _probability_matrix(xgboost_probabilities, name="XGBoost")
    neural = _probability_matrix(
        schemaflow_probabilities,
        name="SchemaFlow",
        expected_shape=tree.shape,
    )
    target_values = np.asarray(targets, dtype=np.int64)
    if target_values.shape != (tree.shape[0],):
        raise ValueError("targets must align with probability rows")
    if not isinstance(label_names, tuple) or len(label_names) != tree.shape[1]:
        raise ValueError("label names must match probability columns")
    grid = _alpha_grid(default_alpha_grid() if alpha_grid is None else alpha_grid)
    normalized_methods = tuple(methods)
    if not normalized_methods or len(set(normalized_methods)) != len(normalized_methods):
        raise ValueError("pooling methods must be non-empty and unique")
    if any(method not in POOLING_METHODS for method in normalized_methods):
        raise ValueError("pooling methods contain an unsupported value")

    candidates = []
    for method in normalized_methods:
        for alpha in grid:
            probabilities = pool_probabilities(
                tree,
                neural,
                alpha=alpha,
                method=method,
            )
            metrics = metric_payload(probabilities, target_values, label_names)
            candidates.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "macro_f1": metrics["macro_f1"],
                    "mcc": metrics["mcc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )

    method_preference = {"log": 0, "linear": 1}
    selected_summary = max(
        candidates,
        key=lambda candidate: (
            candidate["macro_f1"],
            candidate["mcc"],
            candidate["balanced_accuracy"],
            method_preference[candidate["method"]],
            candidate["alpha"],
        ),
    )
    selected_probabilities = pool_probabilities(
        tree,
        neural,
        alpha=selected_summary["alpha"],
        method=selected_summary["method"],
    )
    return {
        "format_version": 1,
        "selection_scope": "validation_only",
        "objective": "native_macro_f1",
        "tie_break": [
            "mcc",
            "balanced_accuracy",
            "linear_pooling",
            "higher_xgboost_alpha",
        ],
        "alpha_semantics": "xgboost_weight",
        "alpha_grid": list(grid),
        "methods": list(normalized_methods),
        "selected": {
            "method": selected_summary["method"],
            "alpha": selected_summary["alpha"],
            "metrics": metric_payload(
                selected_probabilities,
                target_values,
                label_names,
            ),
        },
        "candidates": candidates,
    }


def evaluate_policy(
    xgboost_probabilities,
    schemaflow_probabilities,
    targets,
    label_names: tuple[str, ...],
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate a previously locked method/alpha pair without reselection."""

    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a mapping")
    method = policy.get("method")
    alpha = policy.get("alpha")
    probabilities = pool_probabilities(
        xgboost_probabilities,
        schemaflow_probabilities,
        alpha=alpha,
        method=method,
    )
    return metric_payload(
        probabilities,
        np.asarray(targets, dtype=np.int64),
        label_names,
    )


__all__ = [
    "POOLING_METHODS",
    "default_alpha_grid",
    "evaluate_policy",
    "pool_probabilities",
    "probability_sha256",
    "select_fusion",
]
