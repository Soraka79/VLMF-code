"""Pure selection and diagnostic helpers for VLMF strengthening analyses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


MAIN_CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
POOL_METHODS = ("linear", "log")
ALPHA_GRID = tuple(round(index * 0.05, 12) for index in range(21))
RANKING = (
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "linear_before_log",
    "higher_xgboost_alpha",
)
GLOBAL_RANKING = (
    "mean_macro_f1",
    "mean_mcc",
    "mean_balanced_accuracy",
    "linear_before_log",
    "higher_xgboost_alpha",
)
CANONICAL_THRESHOLDS = {
    "candidate_min_mean_gain_vs_fixed_a0": 0.005,
    "log_pool_min_mean_gain_vs_linear_only": 0.002,
    "clean_min_delta_vs_fixed_a0": 0.0,
    "maximum_single_condition_drop_vs_fixed_a0": 0.002,
    "fallback": "fixed_a0",
    "force_candidate": None,
}


def threshold_variants() -> tuple[dict[str, object], ...]:
    """Return the predeclared one-factor and simplified sensitivity variants."""

    canonical = dict(CANONICAL_THRESHOLDS)

    def changed(name: str, **updates) -> dict[str, object]:
        payload = {"name": name, **canonical}
        payload.update(updates)
        return payload

    return (
        {"name": "canonical", **canonical},
        changed(
            "candidate_min_gain_0",
            candidate_min_mean_gain_vs_fixed_a0=0.0,
        ),
        changed(
            "candidate_min_gain_0p0025",
            candidate_min_mean_gain_vs_fixed_a0=0.0025,
        ),
        changed(
            "candidate_min_gain_0p010",
            candidate_min_mean_gain_vs_fixed_a0=0.010,
        ),
        changed(
            "pool_min_gain_0",
            log_pool_min_mean_gain_vs_linear_only=0.0,
        ),
        changed(
            "pool_min_gain_0p001",
            log_pool_min_mean_gain_vs_linear_only=0.001,
        ),
        changed(
            "pool_min_gain_0p005",
            log_pool_min_mean_gain_vs_linear_only=0.005,
        ),
        changed(
            "maximum_drop_0",
            maximum_single_condition_drop_vs_fixed_a0=0.0,
        ),
        changed(
            "maximum_drop_0p001",
            maximum_single_condition_drop_vs_fixed_a0=0.001,
        ),
        changed(
            "maximum_drop_0p005",
            maximum_single_condition_drop_vs_fixed_a0=0.005,
        ),
        changed("fallback_xgboost", fallback="xgboost"),
        changed(
            "fallback_global_validation_fusion",
            fallback="global_validation_fusion",
        ),
        changed(
            "always_conditioned_linear",
            force_candidate="linear",
        ),
        changed(
            "always_conditioned_linear_or_log",
            force_candidate="linear_or_log",
        ),
    )


def candidate_key(method: object, alpha: object) -> tuple[str, float]:
    method_value = str(method)
    alpha_value = round(float(alpha), 12)
    if method_value not in POOL_METHODS or alpha_value not in ALPHA_GRID:
        raise ValueError(f"candidate outside frozen grid: {method_value}/{alpha_value}")
    return method_value, alpha_value


def _finite_metric(row: Mapping[str, object], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"candidate {name} must be finite")
    return value


def validate_condition_searches(
    searches: Mapping[str, object],
    *,
    conditions: Sequence[str] = MAIN_CONDITIONS,
) -> dict[str, dict[tuple[str, float], dict[str, object]]]:
    """Validate the complete 42-candidate validation records."""

    if tuple(searches) != tuple(conditions):
        raise ValueError("condition searches are incomplete or out of order")
    expected = {
        (method, alpha) for method in POOL_METHODS for alpha in ALPHA_GRID
    }
    validated = {}
    for condition in conditions:
        search = searches[condition]
        if tuple(search.get("methods", ())) != POOL_METHODS:
            raise ValueError(f"{condition} pooling methods changed")
        if tuple(float(value) for value in search.get("alpha_grid", ())) != ALPHA_GRID:
            raise ValueError(f"{condition} alpha grid changed")
        rows = {}
        for source in search.get("candidates", ()):
            key = candidate_key(source.get("method"), source.get("alpha"))
            if key in rows:
                raise ValueError(f"duplicate candidate for {condition}: {key}")
            rows[key] = {
                "method": key[0],
                "alpha": key[1],
                "macro_f1": _finite_metric(source, "macro_f1"),
                "mcc": _finite_metric(source, "mcc"),
                "balanced_accuracy": _finite_metric(
                    source, "balanced_accuracy"
                ),
            }
        if set(rows) != expected:
            raise ValueError(f"candidate pool mismatch for {condition}")
        validated[condition] = rows
    return validated


def candidate_ranking_key(row: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(row["macro_f1"]),
        float(row["mcc"]),
        float(row["balanced_accuracy"]),
        1.0 if row["method"] == "linear" else 0.0,
        float(row["alpha"]),
    )


def global_ranking_key(row: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(row["mean_macro_f1"]),
        float(row["mean_mcc"]),
        float(row["mean_balanced_accuracy"]),
        1.0 if row["method"] == "linear" else 0.0,
        float(row["alpha"]),
    )


def rank_condition_candidates(
    candidates: Mapping[tuple[str, float], Mapping[str, object]],
    *,
    methods: Sequence[str] = POOL_METHODS,
) -> list[dict[str, object]]:
    allowed = set(methods)
    rows = [dict(row) for row in candidates.values() if row["method"] in allowed]
    if not rows:
        raise ValueError("candidate ranking cannot be empty")
    return sorted(rows, key=candidate_ranking_key, reverse=True)


def rank_global_candidates(
    condition_candidates: Mapping[
        str, Mapping[tuple[str, float], Mapping[str, object]]
    ],
    *,
    conditions: Sequence[str] = MAIN_CONDITIONS,
) -> list[dict[str, object]]:
    rows = []
    for method in POOL_METHODS:
        for alpha in ALPHA_GRID:
            values = [condition_candidates[name][(method, alpha)] for name in conditions]
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "mean_macro_f1": float(
                        np.mean([row["macro_f1"] for row in values])
                    ),
                    "mean_mcc": float(np.mean([row["mcc"] for row in values])),
                    "mean_balanced_accuracy": float(
                        np.mean([row["balanced_accuracy"] for row in values])
                    ),
                }
            )
    return sorted(rows, key=global_ranking_key, reverse=True)


def select_branch_policies(
    searches: Mapping[str, object],
    *,
    conditions: Sequence[str] = MAIN_CONDITIONS,
) -> dict[str, object]:
    """Select global and condition-indexed XGBoost/SchemaFlow endpoints."""

    validated = validate_condition_searches(searches, conditions=conditions)
    endpoint_keys = (("linear", 0.0), ("linear", 1.0))
    conditioned = {}
    for condition in conditions:
        ranked = sorted(
            (dict(validated[condition][key]) for key in endpoint_keys),
            key=candidate_ranking_key,
            reverse=True,
        )
        selected = ranked[0]
        conditioned[condition] = {
            "branch": "xgboost" if selected["alpha"] == 1.0 else "schemaflow_pretrained",
            "method": "linear",
            "alpha": selected["alpha"],
            "validation_metrics": {
                key: selected[key]
                for key in ("macro_f1", "mcc", "balanced_accuracy")
            },
            "runner_up": ranked[1],
        }

    global_rows = []
    for method, alpha in endpoint_keys:
        values = [validated[name][(method, alpha)] for name in conditions]
        global_rows.append(
            {
                "method": method,
                "alpha": alpha,
                "branch": "xgboost" if alpha == 1.0 else "schemaflow_pretrained",
                "mean_macro_f1": float(
                    np.mean([row["macro_f1"] for row in values])
                ),
                "mean_mcc": float(np.mean([row["mcc"] for row in values])),
                "mean_balanced_accuracy": float(
                    np.mean([row["balanced_accuracy"] for row in values])
                ),
            }
        )
    global_rows.sort(key=global_ranking_key, reverse=True)
    return {
        "tie_break": [
            "macro_f1",
            "mcc",
            "balanced_accuracy",
            "xgboost_on_exact_tie",
        ],
        "global": {
            "branch": global_rows[0]["branch"],
            "method": "linear",
            "alpha": global_rows[0]["alpha"],
            "validation_objective": {
                key: global_rows[0][key]
                for key in (
                    "mean_macro_f1",
                    "mean_mcc",
                    "mean_balanced_accuracy",
                )
            },
            "runner_up": global_rows[1],
        },
        "conditioned": conditioned,
    }


def _mean_metrics(
    policies: Mapping[str, Mapping[str, object]],
    validated: Mapping[str, Mapping[tuple[str, float], Mapping[str, object]]],
    conditions: Sequence[str],
) -> dict[str, float]:
    rows = [
        validated[name][candidate_key(policies[name]["method"], policies[name]["alpha"])]
        for name in conditions
    ]
    return {
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "mean_mcc": float(np.mean([row["mcc"] for row in rows])),
        "mean_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in rows])
        ),
    }


def _fallback_policies(
    fallback: str,
    global_policy: Mapping[str, object],
    conditions: Sequence[str],
) -> dict[str, dict[str, object]]:
    if fallback == "fixed_a0":
        method, alpha = "linear", 0.55
    elif fallback == "xgboost":
        method, alpha = "linear", 1.0
    elif fallback == "global_validation_fusion":
        method = str(global_policy["method"])
        alpha = float(global_policy["alpha"])
    else:
        raise ValueError(f"unsupported fallback: {fallback}")
    return {
        name: {
            "method": method,
            "alpha": alpha,
            "selected_on": f"fallback_{fallback}",
        }
        for name in conditions
    }


def select_threshold_variant(
    searches: Mapping[str, object],
    fixed_metrics: Mapping[str, Mapping[str, object]],
    global_policy: Mapping[str, object],
    variant: Mapping[str, object],
    *,
    conditions: Sequence[str] = MAIN_CONDITIONS,
) -> dict[str, object]:
    """Apply one fully specified admission/fallback variant on validation."""

    validated = validate_condition_searches(searches, conditions=conditions)
    linear_policies = {}
    pooled_policies = {}
    for condition in conditions:
        linear = rank_condition_candidates(
            validated[condition], methods=("linear",)
        )[0]
        pooled = rank_condition_candidates(validated[condition])[0]
        linear_policies[condition] = {
            "method": linear["method"],
            "alpha": linear["alpha"],
            "selected_on": condition,
        }
        pooled_policies[condition] = {
            "method": pooled["method"],
            "alpha": pooled["alpha"],
            "selected_on": condition,
        }
    linear_objective = _mean_metrics(linear_policies, validated, conditions)
    pooled_objective = _mean_metrics(pooled_policies, validated, conditions)

    force = variant.get("force_candidate")
    if force == "linear":
        candidate_name = "missing_rate_conditioned_linear"
        candidate_policies = linear_policies
        candidate_objective = linear_objective
        gate_passed = True
    elif force == "linear_or_log":
        candidate_name = "missing_rate_conditioned_linear_or_log"
        candidate_policies = pooled_policies
        candidate_objective = pooled_objective
        gate_passed = True
    elif force is None:
        use_pool = (
            pooled_objective["mean_macro_f1"]
            - linear_objective["mean_macro_f1"]
            >= float(variant["log_pool_min_mean_gain_vs_linear_only"])
        )
        if use_pool:
            candidate_name = "missing_rate_conditioned_linear_or_log"
            candidate_policies = pooled_policies
            candidate_objective = pooled_objective
        else:
            candidate_name = "missing_rate_conditioned_linear"
            candidate_policies = linear_policies
            candidate_objective = linear_objective
        fixed_mean = float(
            np.mean([float(fixed_metrics[name]["macro_f1"]) for name in conditions])
        )
        condition_deltas = {
            name: float(
                validated[name][
                    candidate_key(
                        candidate_policies[name]["method"],
                        candidate_policies[name]["alpha"],
                    )
                ]["macro_f1"]
                - float(fixed_metrics[name]["macro_f1"])
            )
            for name in conditions
        }
        gate_passed = (
            candidate_objective["mean_macro_f1"] - fixed_mean
            >= float(variant["candidate_min_mean_gain_vs_fixed_a0"])
            and condition_deltas["clean"]
            >= float(variant["clean_min_delta_vs_fixed_a0"]) - 1e-12
            and min(condition_deltas.values())
            >= -float(variant["maximum_single_condition_drop_vs_fixed_a0"])
        )
    else:
        raise ValueError(f"unsupported forced candidate: {force}")

    fixed_mean = float(
        np.mean([float(fixed_metrics[name]["macro_f1"]) for name in conditions])
    )
    condition_deltas = {
        name: float(
            validated[name][
                candidate_key(
                    candidate_policies[name]["method"],
                    candidate_policies[name]["alpha"],
                )
            ]["macro_f1"]
            - float(fixed_metrics[name]["macro_f1"])
        )
        for name in conditions
    }
    fallback_policies = _fallback_policies(
        str(variant["fallback"]), global_policy, conditions
    )
    locked = candidate_policies if gate_passed else fallback_policies
    return {
        "variant": dict(variant),
        "candidate_name": candidate_name,
        "candidate_objective": candidate_objective,
        "fixed_a0_mean_macro_f1": fixed_mean,
        "candidate_mean_gain": candidate_objective["mean_macro_f1"] - fixed_mean,
        "condition_deltas_vs_fixed_a0": condition_deltas,
        "gate_passed": bool(gate_passed),
        "locked_policies": locked,
    }


def complementarity_payload(
    xgboost_probabilities,
    schemaflow_probabilities,
    targets,
) -> dict[str, object]:
    """Partition paired branch outcomes into exhaustive sample-level cells."""

    tree = np.asarray(xgboost_probabilities, dtype=np.float64)
    neural = np.asarray(schemaflow_probabilities, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int64)
    if (
        tree.ndim != 2
        or tree.shape != neural.shape
        or target.shape != (tree.shape[0],)
        or tree.shape[0] == 0
    ):
        raise ValueError("paired probabilities and targets must align")
    tree_prediction = np.argmax(tree, axis=1)
    neural_prediction = np.argmax(neural, axis=1)
    tree_correct = tree_prediction == target
    neural_correct = neural_prediction == target
    disagreement = tree_prediction != neural_prediction
    both_correct = tree_correct & neural_correct
    tree_only = tree_correct & ~neural_correct
    neural_only = ~tree_correct & neural_correct
    both_wrong = ~tree_correct & ~neural_correct
    both_wrong_same = both_wrong & ~disagreement
    both_wrong_different = both_wrong & disagreement
    tree_error = ~tree_correct
    neural_error = ~neural_correct
    error_intersection = tree_error & neural_error
    error_union = tree_error | neural_error

    counts = {
        "sample_count": int(target.size),
        "prediction_disagreement": int(np.count_nonzero(disagreement)),
        "both_correct": int(np.count_nonzero(both_correct)),
        "xgboost_only_correct": int(np.count_nonzero(tree_only)),
        "schemaflow_only_correct": int(np.count_nonzero(neural_only)),
        "both_wrong_same_prediction": int(np.count_nonzero(both_wrong_same)),
        "both_wrong_different_prediction": int(
            np.count_nonzero(both_wrong_different)
        ),
        "xgboost_errors": int(np.count_nonzero(tree_error)),
        "schemaflow_errors": int(np.count_nonzero(neural_error)),
        "error_intersection": int(np.count_nonzero(error_intersection)),
        "error_union": int(np.count_nonzero(error_union)),
    }
    partition_total = sum(
        counts[name]
        for name in (
            "both_correct",
            "xgboost_only_correct",
            "schemaflow_only_correct",
            "both_wrong_same_prediction",
            "both_wrong_different_prediction",
        )
    )
    if partition_total != counts["sample_count"]:
        raise RuntimeError("complementarity partition is not exhaustive")
    sample_count = counts["sample_count"]
    rates = {
        name + "_rate": counts[name] / sample_count
        for name in (
            "prediction_disagreement",
            "both_correct",
            "xgboost_only_correct",
            "schemaflow_only_correct",
            "both_wrong_same_prediction",
            "both_wrong_different_prediction",
        )
    }
    rates["oracle_branch_accuracy"] = 1.0 - (
        counts["error_intersection"] / sample_count
    )
    rates["error_jaccard"] = (
        counts["error_intersection"] / counts["error_union"]
        if counts["error_union"]
        else None
    )
    smaller_error_set = min(counts["xgboost_errors"], counts["schemaflow_errors"])
    rates["error_overlap_coefficient"] = (
        counts["error_intersection"] / smaller_error_set
        if smaller_error_set
        else None
    )
    return {"counts": counts, "rates": rates}


def route_stability_record(
    validation_policy: Mapping[str, object],
    test_ranking: Sequence[Mapping[str, object]],
    *,
    deployed_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare a validation route with a post-lock test-oracle ranking."""

    if not test_ranking:
        raise ValueError("test candidate ranking cannot be empty")
    validation_key = candidate_key(
        validation_policy["method"], validation_policy["alpha"]
    )
    deployed = validation_policy if deployed_policy is None else deployed_policy
    deployed_key = candidate_key(deployed["method"], deployed["alpha"])
    indexed = {
        candidate_key(row["method"], row["alpha"]): (index, row)
        for index, row in enumerate(test_ranking, start=1)
    }
    if validation_key not in indexed or deployed_key not in indexed:
        raise ValueError("locked route is absent from the test candidate grid")
    oracle = dict(test_ranking[0])
    oracle_key = candidate_key(oracle["method"], oracle["alpha"])
    validation_rank, validation_row = indexed[validation_key]
    deployed_rank, deployed_row = indexed[deployed_key]
    return {
        "test_oracle": oracle,
        "validation_policy": {
            "method": validation_key[0],
            "alpha": validation_key[1],
        },
        "deployed_policy": {
            "method": deployed_key[0],
            "alpha": deployed_key[1],
        },
        "validation_exact_match": validation_key == oracle_key,
        "validation_pool_match": validation_key[0] == oracle_key[0],
        "validation_alpha_absolute_difference": abs(
            validation_key[1] - oracle_key[1]
        ),
        "validation_policy_test_rank": validation_rank,
        "validation_policy_test_macro_f1_regret": float(
            oracle["macro_f1"] - validation_row["macro_f1"]
        ),
        "deployed_exact_match": deployed_key == oracle_key,
        "deployed_pool_match": deployed_key[0] == oracle_key[0],
        "deployed_alpha_absolute_difference": abs(
            deployed_key[1] - oracle_key[1]
        ),
        "deployed_policy_test_rank": deployed_rank,
        "deployed_policy_test_macro_f1_regret": float(
            oracle["macro_f1"] - deployed_row["macro_f1"]
        ),
    }


__all__ = [
    "ALPHA_GRID",
    "CANONICAL_THRESHOLDS",
    "GLOBAL_RANKING",
    "MAIN_CONDITIONS",
    "POOL_METHODS",
    "RANKING",
    "candidate_key",
    "candidate_ranking_key",
    "complementarity_payload",
    "global_ranking_key",
    "rank_condition_candidates",
    "rank_global_candidates",
    "route_stability_record",
    "select_branch_policies",
    "select_threshold_variant",
    "threshold_variants",
    "validate_condition_searches",
]
