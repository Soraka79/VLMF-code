"""Validation-only fitting helpers for the frozen learned-ensemble protocol."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
C_GRID = (0.01, 0.1, 1.0, 10.0)
META_SEED = 20260731
TARGET_TOLERANCE = 1e-12
PROBABILITY_CLIP = 1e-12
GATE_CONTINUOUS_INDICES = tuple(range(3, 10))


def _probability_matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(matrix < -1e-12) or not np.allclose(
        matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError(f"{name} is not a probability matrix")
    return matrix


def classification_metrics(
    probabilities, targets, class_count: int
) -> dict[str, float]:
    matrix = _probability_matrix(probabilities, name="probabilities")
    labels = np.asarray(targets, dtype=np.int64)
    if matrix.shape != (labels.size, class_count):
        raise ValueError("probabilities and targets do not align")
    expected = np.arange(class_count, dtype=np.int64)
    predictions = matrix.argmax(axis=1)
    recalls = recall_score(
        labels,
        predictions,
        labels=expected,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=expected,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(recalls.mean()),
    }


def stacking_feature_names(class_count: int) -> tuple[str, ...]:
    if class_count < 2:
        raise ValueError("stacking requires at least two classes")
    return tuple(
        [f"xgboost_log_probability_class_{index}" for index in range(class_count)]
        + [
            f"schemaflow_log_probability_class_{index}"
            for index in range(class_count)
        ]
    )


def stacking_features(xgboost, schemaflow) -> np.ndarray:
    tree = _probability_matrix(xgboost, name="xgboost")
    neural = _probability_matrix(schemaflow, name="schemaflow")
    if tree.shape != neural.shape:
        raise ValueError("branch probability matrices do not align")
    return np.concatenate(
        (
            np.log(np.clip(tree, PROBABILITY_CLIP, 1.0)),
            np.log(np.clip(neural, PROBABILITY_CLIP, 1.0)),
        ),
        axis=1,
    )


def gate_feature_names(
    conditions: Sequence[str] = CONDITIONS,
) -> tuple[str, ...]:
    if tuple(conditions) != CONDITIONS:
        raise ValueError("gate conditions must match the frozen protocol")
    return (
        *(f"condition_{name}" for name in conditions),
        "xgboost_max_probability",
        "schemaflow_max_probability",
        "xgboost_top1_minus_top2_margin",
        "schemaflow_top1_minus_top2_margin",
        "xgboost_normalized_entropy",
        "schemaflow_normalized_entropy",
        "branch_jensen_shannon_divergence",
        "branch_top1_agreement",
    )


def _top_margin(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_CLIP, 1.0)
    return -np.sum(probabilities * np.log(clipped), axis=1) / math.log(
        probabilities.shape[1]
    )


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    midpoint = np.clip(midpoint, PROBABILITY_CLIP, 1.0)
    left_log = np.log(np.clip(left, PROBABILITY_CLIP, 1.0) / midpoint)
    right_log = np.log(np.clip(right, PROBABILITY_CLIP, 1.0) / midpoint)
    return 0.5 * (
        np.sum(left * left_log, axis=1) + np.sum(right * right_log, axis=1)
    )


def gate_features(
    xgboost,
    schemaflow,
    *,
    condition_index: int,
    conditions: Sequence[str] = CONDITIONS,
) -> np.ndarray:
    tree = _probability_matrix(xgboost, name="xgboost")
    neural = _probability_matrix(schemaflow, name="schemaflow")
    if tree.shape != neural.shape:
        raise ValueError("branch probability matrices do not align")
    if tuple(conditions) != CONDITIONS or condition_index not in range(len(conditions)):
        raise ValueError("invalid declared-condition index")
    one_hot = np.zeros((tree.shape[0], len(conditions)), dtype=np.float64)
    one_hot[:, condition_index] = 1.0
    continuous = np.column_stack(
        (
            tree.max(axis=1),
            neural.max(axis=1),
            _top_margin(tree),
            _top_margin(neural),
            _normalized_entropy(tree),
            _normalized_entropy(neural),
            _jensen_shannon(tree, neural),
        )
    )
    agreement = (tree.argmax(axis=1) == neural.argmax(axis=1)).astype(
        np.float64
    )[:, None]
    return np.concatenate((one_hot, continuous, agreement), axis=1)


def gate_targets(
    xgboost,
    schemaflow,
    targets,
    *,
    tolerance: float = TARGET_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    tree = _probability_matrix(xgboost, name="xgboost")
    neural = _probability_matrix(schemaflow, name="schemaflow")
    labels = np.asarray(targets, dtype=np.int64)
    if tree.shape != neural.shape or tree.shape[0] != labels.size:
        raise ValueError("gate target inputs do not align")
    if np.any(labels < 0) or np.any(labels >= tree.shape[1]):
        raise ValueError("gate targets are outside the class range")
    rows = np.arange(labels.size)
    delta = tree[rows, labels] - neural[rows, labels]
    eligible = np.abs(delta) > tolerance
    binary = (delta > tolerance).astype(np.int64)
    return binary, eligible


def make_group_folds(
    clean_targets,
    group_ids,
    *,
    folds: int = 5,
    seed: int = META_SEED,
) -> np.ndarray:
    labels = np.asarray(clean_targets, dtype=np.int64)
    groups = np.asarray(group_ids)
    if labels.ndim != 1 or groups.ndim != 1 or labels.shape != groups.shape:
        raise ValueError("clean targets and group ids must be aligned vectors")
    assignments = np.full(labels.size, -1, dtype=np.int32)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            splits = list(splitter.split(np.zeros(labels.size), labels, groups))
    except (ValueError, UserWarning) as error:
        raise RuntimeError("frozen group-aware fold construction failed") from error
    for fold_index, (train_indices, holdout_indices) in enumerate(splits):
        if np.any(assignments[holdout_indices] != -1):
            raise RuntimeError("validation rows were assigned to multiple folds")
        train_groups = set(np.asarray(groups[train_indices]).tolist())
        holdout_groups = set(np.asarray(groups[holdout_indices]).tolist())
        if train_groups & holdout_groups:
            raise RuntimeError("group leakage detected in meta-validation folds")
        assignments[holdout_indices] = fold_index
    if np.any(assignments < 0) or set(assignments.tolist()) != set(range(folds)):
        raise RuntimeError("meta-validation fold assignment is incomplete")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    for group_index in range(unique_groups.size):
        if np.unique(assignments[inverse == group_index]).size != 1:
            raise RuntimeError("one group was split across meta-validation folds")
    return assignments


def fold_assignment_sha256(assignments) -> str:
    values = np.asarray(assignments, dtype="<i4")
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("fold assignments must be a nonnegative vector")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def fold_summary(assignments, group_ids) -> list[dict[str, int]]:
    values = np.asarray(assignments, dtype=np.int32)
    groups = np.asarray(group_ids)
    if values.shape != groups.shape:
        raise ValueError("fold assignments and groups do not align")
    rows = []
    for fold_index in sorted(set(values.tolist())):
        mask = values == fold_index
        rows.append(
            {
                "fold": int(fold_index),
                "sample_count": int(mask.sum()),
                "group_count": int(np.unique(groups[mask]).size),
            }
        )
    return rows


def _fit_scaler(features: np.ndarray, indices: Sequence[int]) -> tuple[np.ndarray, dict]:
    selected = np.asarray(tuple(indices), dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("at least one standardized feature is required")
    scaler = StandardScaler()
    transformed = np.asarray(features, dtype=np.float64).copy()
    transformed[:, selected] = scaler.fit_transform(transformed[:, selected])
    payload = {
        "standardized_feature_indices": selected.tolist(),
        "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "var": np.asarray(scaler.var_, dtype=np.float64).tolist(),
    }
    return transformed, payload


def _apply_scaler(features: np.ndarray, payload: Mapping[str, object]) -> np.ndarray:
    transformed = np.asarray(features, dtype=np.float64).copy()
    indices = np.asarray(payload["standardized_feature_indices"], dtype=np.int64)
    means = np.asarray(payload["mean"], dtype=np.float64)
    scales = np.asarray(payload["scale"], dtype=np.float64)
    if indices.size != means.size or means.shape != scales.shape:
        raise ValueError("locked scaler parameters are inconsistent")
    transformed[:, indices] = (transformed[:, indices] - means) / scales
    return transformed


def _fit_estimator(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    C: float,
    multinomial: bool,
) -> LogisticRegression:
    kwargs = {
        "C": float(C),
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 2000,
        "tol": 1e-8,
        "random_state": META_SEED,
    }
    if multinomial:
        kwargs["multi_class"] = "multinomial"
    estimator = LogisticRegression(**kwargs)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*'multi_class'.*deprecated.*",
                category=FutureWarning,
            )
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(features, targets)
    except (ValueError, ConvergenceWarning) as error:
        raise RuntimeError("frozen logistic fit failed or did not converge") from error
    return estimator


def _candidate_summary(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    sample_count: int,
    C: float,
    fold_iterations: Sequence[Sequence[int]],
) -> dict[str, object]:
    by_condition = {}
    for index, condition in enumerate(CONDITIONS):
        start = index * sample_count
        stop = start + sample_count
        by_condition[condition] = classification_metrics(
            probabilities[start:stop], targets, probabilities.shape[1]
        )
    return {
        "C": float(C),
        "condition_metrics": by_condition,
        "mean_macro_f1": float(
            np.mean([row["macro_f1"] for row in by_condition.values()])
        ),
        "mean_mcc": float(np.mean([row["mcc"] for row in by_condition.values()])),
        "mean_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in by_condition.values()])
        ),
        "fold_n_iter": [list(map(int, row)) for row in fold_iterations],
    }


def candidate_ranking_key(candidate: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(candidate["mean_macro_f1"]),
        float(candidate["mean_mcc"]),
        float(candidate["mean_balanced_accuracy"]),
        -float(candidate["C"]),
    )


def _selection_trace(ranked: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(ranked) != len(C_GRID):
        raise ValueError("selection requires the complete frozen C grid")
    first, second = ranked[:2]
    criteria = (
        ("mean_macro_f1", "equal_mean_condition_macro_f1"),
        ("mean_mcc", "equal_mean_condition_mcc"),
        ("mean_balanced_accuracy", "equal_mean_condition_balanced_accuracy"),
    )
    deciding = "smaller_C_stronger_regularization"
    for field, label in criteria:
        if float(first[field]) != float(second[field]):
            deciding = label
            break
    return {
        "selected_C": float(first["C"]),
        "deciding_criterion_against_runner_up": deciding,
        "runner_up_C": float(second["C"]),
        "ranking": [
            {
                "rank": index,
                "C": float(row["C"]),
                "mean_macro_f1": float(row["mean_macro_f1"]),
                "mean_mcc": float(row["mean_mcc"]),
                "mean_balanced_accuracy": float(row["mean_balanced_accuracy"]),
            }
            for index, row in enumerate(ranked, start=1)
        ],
    }


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    result = np.empty_like(scores, dtype=np.float64)
    positive = scores >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exponentials = np.exp(scores[~positive])
    result[~positive] = exponentials / (1.0 + exponentials)
    return result


def _estimator_payload(estimator: LogisticRegression) -> dict[str, object]:
    return {
        "classes": np.asarray(estimator.classes_, dtype=np.int64).tolist(),
        "coefficients": np.asarray(estimator.coef_, dtype=np.float64).tolist(),
        "intercepts": np.asarray(estimator.intercept_, dtype=np.float64).tolist(),
        "n_iter": np.asarray(estimator.n_iter_, dtype=np.int64).tolist(),
        "solver": "lbfgs",
        "penalty": "l2",
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 2000,
        "tolerance": 1e-8,
        "random_state": META_SEED,
    }


def predict_locked_stacker(
    model: Mapping[str, object], xgboost, schemaflow
) -> np.ndarray:
    features = stacking_features(xgboost, schemaflow)
    transformed = _apply_scaler(features, model["preprocessing"])
    estimator = model["estimator"]
    coefficients = np.asarray(estimator["coefficients"], dtype=np.float64)
    intercepts = np.asarray(estimator["intercepts"], dtype=np.float64)
    classes = np.asarray(estimator["classes"], dtype=np.int64)
    if coefficients.shape[0] != classes.size or not np.array_equal(
        classes, np.arange(classes.size)
    ):
        raise ValueError("locked stacker classes or coefficients are invalid")
    return _softmax(transformed @ coefficients.T + intercepts)


def predict_locked_gate_alpha(
    model: Mapping[str, object], xgboost, schemaflow, *, condition_index: int
) -> np.ndarray:
    features = gate_features(
        xgboost, schemaflow, condition_index=condition_index
    )
    transformed = _apply_scaler(features, model["preprocessing"])
    estimator = model["estimator"]
    coefficients = np.asarray(estimator["coefficients"], dtype=np.float64)
    intercepts = np.asarray(estimator["intercepts"], dtype=np.float64)
    classes = np.asarray(estimator["classes"], dtype=np.int64)
    if coefficients.shape != (1, transformed.shape[1]) or not np.array_equal(
        classes, np.asarray((0, 1), dtype=np.int64)
    ):
        raise ValueError("locked gate classes or coefficients are invalid")
    return _sigmoid(transformed @ coefficients[0] + intercepts[0])


def predict_locked_gate(
    model: Mapping[str, object], xgboost, schemaflow, *, condition_index: int
) -> np.ndarray:
    tree = _probability_matrix(xgboost, name="xgboost")
    neural = _probability_matrix(schemaflow, name="schemaflow")
    alpha = predict_locked_gate_alpha(
        model, tree, neural, condition_index=condition_index
    )
    return alpha[:, None] * tree + (1.0 - alpha[:, None]) * neural


def fit_logistic_stacker(
    probabilities_by_condition: Mapping[str, Mapping[str, np.ndarray]],
    targets,
    base_folds,
    *,
    C_grid: Sequence[float] = C_GRID,
) -> dict[str, object]:
    if tuple(probabilities_by_condition) != CONDITIONS or tuple(C_grid) != C_GRID:
        raise ValueError("stacker inputs changed from the frozen protocol")
    labels = np.asarray(targets, dtype=np.int64)
    folds = np.asarray(base_folds, dtype=np.int32)
    if labels.shape != folds.shape:
        raise ValueError("targets and folds do not align")
    class_count = None
    condition_features = []
    for condition in CONDITIONS:
        branches = probabilities_by_condition[condition]
        features = stacking_features(branches["xgboost"], branches["schemaflow"])
        if features.shape[0] != labels.size:
            raise ValueError("condition features and targets do not align")
        observed_classes = features.shape[1] // 2
        class_count = class_count or observed_classes
        if observed_classes != class_count:
            raise ValueError("class count changed across conditions")
        condition_features.append(features)
    all_features = np.concatenate(condition_features, axis=0)
    all_targets = np.tile(labels, len(CONDITIONS))
    all_folds = np.tile(folds, len(CONDITIONS))
    expected_classes = np.arange(class_count, dtype=np.int64)
    candidates = []
    for C in C_grid:
        oof = np.empty((all_targets.size, class_count), dtype=np.float64)
        fold_iterations = []
        for fold_index in range(5):
            train = all_folds != fold_index
            holdout = ~train
            transformed_train, scaler = _fit_scaler(
                all_features[train], range(all_features.shape[1])
            )
            transformed_holdout = _apply_scaler(all_features[holdout], scaler)
            estimator = _fit_estimator(
                transformed_train,
                all_targets[train],
                C=float(C),
                multinomial=True,
            )
            if not np.array_equal(estimator.classes_, expected_classes):
                raise RuntimeError("a stacker fold is missing a declared class")
            oof[holdout] = estimator.predict_proba(transformed_holdout)
            fold_iterations.append(estimator.n_iter_.tolist())
        candidates.append(
            _candidate_summary(
                oof,
                labels,
                sample_count=labels.size,
                C=float(C),
                fold_iterations=fold_iterations,
            )
        )
    ranked = sorted(candidates, key=candidate_ranking_key, reverse=True)
    selection = _selection_trace(ranked)
    transformed, scaler = _fit_scaler(all_features, range(all_features.shape[1]))
    estimator = _fit_estimator(
        transformed,
        all_targets,
        C=selection["selected_C"],
        multinomial=True,
    )
    if not np.array_equal(estimator.classes_, expected_classes):
        raise RuntimeError("refit stacker is missing a declared class")
    model = {
        "format_version": 1,
        "method": "logistic_stacking",
        "selected_C": selection["selected_C"],
        "class_count": int(class_count),
        "feature_names": list(stacking_feature_names(class_count)),
        "preprocessing": scaler,
        "estimator": {
            **_estimator_payload(estimator),
            "multi_class": "multinomial",
        },
    }
    locked = predict_locked_stacker(
        model,
        np.concatenate(
            [probabilities_by_condition[name]["xgboost"] for name in CONDITIONS]
        ),
        np.concatenate(
            [probabilities_by_condition[name]["schemaflow"] for name in CONDITIONS]
        ),
    )
    reference = estimator.predict_proba(transformed)
    return {
        "selection": selection,
        "oof_candidates": candidates,
        "model": model,
        "locked_inference_max_absolute_error": float(
            np.max(np.abs(locked - reference))
        ),
    }


def fit_learned_gate(
    probabilities_by_condition: Mapping[str, Mapping[str, np.ndarray]],
    targets,
    base_folds,
    *,
    C_grid: Sequence[float] = C_GRID,
    target_tolerance: float = TARGET_TOLERANCE,
) -> dict[str, object]:
    if tuple(probabilities_by_condition) != CONDITIONS or tuple(C_grid) != C_GRID:
        raise ValueError("gate inputs changed from the frozen protocol")
    labels = np.asarray(targets, dtype=np.int64)
    folds = np.asarray(base_folds, dtype=np.int32)
    if labels.shape != folds.shape:
        raise ValueError("targets and folds do not align")
    features = []
    tree_rows = []
    neural_rows = []
    binary_rows = []
    eligible_rows = []
    for index, condition in enumerate(CONDITIONS):
        branches = probabilities_by_condition[condition]
        tree = _probability_matrix(branches["xgboost"], name="xgboost")
        neural = _probability_matrix(branches["schemaflow"], name="schemaflow")
        if tree.shape != neural.shape or tree.shape[0] != labels.size:
            raise ValueError("gate condition inputs do not align")
        binary, eligible = gate_targets(
            tree, neural, labels, tolerance=target_tolerance
        )
        features.append(
            gate_features(tree, neural, condition_index=index)
        )
        tree_rows.append(tree)
        neural_rows.append(neural)
        binary_rows.append(binary)
        eligible_rows.append(eligible)
    all_features = np.concatenate(features, axis=0)
    all_tree = np.concatenate(tree_rows, axis=0)
    all_neural = np.concatenate(neural_rows, axis=0)
    all_binary = np.concatenate(binary_rows, axis=0)
    all_eligible = np.concatenate(eligible_rows, axis=0)
    all_folds = np.tile(folds, len(CONDITIONS))
    candidates = []
    for C in C_grid:
        oof = np.empty_like(all_tree)
        fold_iterations = []
        for fold_index in range(5):
            train = (all_folds != fold_index) & all_eligible
            holdout = all_folds == fold_index
            if not np.array_equal(np.unique(all_binary[train]), np.asarray((0, 1))):
                raise RuntimeError("a gate fold is missing one binary target class")
            transformed_train, scaler = _fit_scaler(
                all_features[train], GATE_CONTINUOUS_INDICES
            )
            transformed_holdout = _apply_scaler(all_features[holdout], scaler)
            estimator = _fit_estimator(
                transformed_train,
                all_binary[train],
                C=float(C),
                multinomial=False,
            )
            if not np.array_equal(estimator.classes_, np.asarray((0, 1))):
                raise RuntimeError("a gate fold is missing one binary target class")
            class_one = int(np.flatnonzero(estimator.classes_ == 1)[0])
            alpha = estimator.predict_proba(transformed_holdout)[:, class_one]
            oof[holdout] = (
                alpha[:, None] * all_tree[holdout]
                + (1.0 - alpha[:, None]) * all_neural[holdout]
            )
            fold_iterations.append(estimator.n_iter_.tolist())
        candidates.append(
            _candidate_summary(
                oof,
                labels,
                sample_count=labels.size,
                C=float(C),
                fold_iterations=fold_iterations,
            )
        )
    ranked = sorted(candidates, key=candidate_ranking_key, reverse=True)
    selection = _selection_trace(ranked)
    if not np.array_equal(
        np.unique(all_binary[all_eligible]), np.asarray((0, 1))
    ):
        raise RuntimeError("gate refit data is missing one binary target class")
    transformed, scaler = _fit_scaler(
        all_features[all_eligible], GATE_CONTINUOUS_INDICES
    )
    estimator = _fit_estimator(
        transformed,
        all_binary[all_eligible],
        C=selection["selected_C"],
        multinomial=False,
    )
    model = {
        "format_version": 1,
        "method": "learned_gate",
        "selected_C": selection["selected_C"],
        "class_count": int(all_tree.shape[1]),
        "feature_names": list(gate_feature_names()),
        "continuous_feature_indices": list(GATE_CONTINUOUS_INDICES),
        "condition_order": list(CONDITIONS),
        "target_tolerance": float(target_tolerance),
        "target_ties_excluded_from_fitting": True,
        "preprocessing": scaler,
        "estimator": _estimator_payload(estimator),
    }
    locked_alpha = []
    for index, condition in enumerate(CONDITIONS):
        branches = probabilities_by_condition[condition]
        locked_alpha.append(
            predict_locked_gate_alpha(
                model,
                branches["xgboost"],
                branches["schemaflow"],
                condition_index=index,
            )
        )
    locked_alpha = np.concatenate(locked_alpha)[all_eligible]
    reference_alpha = estimator.predict_proba(transformed)[:, 1]
    return {
        "selection": selection,
        "oof_candidates": candidates,
        "target_counts": {
            "eligible": int(all_eligible.sum()),
            "ties_excluded": int((~all_eligible).sum()),
            "xgboost_preferred": int(
                np.sum(all_binary[all_eligible] == 1)
            ),
            "schemaflow_preferred": int(
                np.sum(all_binary[all_eligible] == 0)
            ),
        },
        "model": model,
        "locked_inference_max_absolute_error": float(
            np.max(np.abs(locked_alpha - reference_alpha))
        ),
    }


def dataset_stratified_paired_setting_bootstrap(
    deltas_by_dataset: Mapping[str, Sequence[float]],
    *,
    datasets: Sequence[str],
    resamples: int = 20_000,
    seed: int = 20260730,
) -> dict[str, object]:
    if type(resamples) is not int or resamples < 2_000:
        raise ValueError("bootstrap requires at least 2,000 resamples")
    if set(deltas_by_dataset) != set(datasets):
        raise ValueError("bootstrap dataset strata are incomplete")
    arrays = {}
    for dataset in datasets:
        values = np.asarray(deltas_by_dataset[dataset], dtype=np.float64)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("each dataset requires nine finite paired deltas")
        arrays[dataset] = values
    observed_by_dataset = {
        name: float(values.mean()) for name, values in arrays.items()
    }
    observed = float(np.mean(list(observed_by_dataset.values())))
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        samples[index] = float(
            np.mean(
                [
                    values[rng.integers(0, 9, size=9)].mean()
                    for values in arrays.values()
                ]
            )
        )
    return {
        "method": "dataset_stratified_paired_setting_bootstrap",
        "within_dataset_units": 9,
        "within_dataset_sampling": "with_replacement",
        "dataset_weighting": "equal",
        "resamples": resamples,
        "seed": seed,
        "observed_by_dataset": observed_by_dataset,
        "observed_equal_dataset_mean": observed,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "p_value_inferred": False,
    }
