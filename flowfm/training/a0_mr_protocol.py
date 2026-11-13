"""Frozen, dataset-neutral A0-MR validation selection protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from flowfm.data.identity import stable_hash
from flowfm.data.missingness import (
    MISSINGNESS_ALGORITHM,
    STRUCTURED_MISSINGNESS_ALGORITHM,
    STRUCTURED_MISSINGNESS_GROUPS,
)
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.probability_fusion import (
    default_alpha_grid,
    pool_probabilities,
    select_fusion,
)


A0_MR_PROTOCOL_VERSION = "a0-mr-selection-v1"
A0_MR_PROTOCOL_HASH_NAMESPACE = "flowfm-a0-mr-selection-protocol-v1"
RANDOM_CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
STRUCTURED_CONDITIONS = tuple(
    f"structured__{name}" for name in STRUCTURED_MISSINGNESS_GROUPS
)
FORMAL_CONDITIONS = RANDOM_CONDITIONS + STRUCTURED_CONDITIONS
ALPHA_GRID = default_alpha_grid(0.05)
FIXED_A0_ALPHA_XGBOOST = 0.55
CANDIDATE_MIN_GAIN = 0.005
POOL_MIN_GAIN = 0.002
MAX_CONDITION_DROP_VS_FIXED = 0.002
TIE_BREAK = (
    "mcc",
    "balanced_accuracy",
    "linear_pooling",
    "higher_xgboost_alpha",
)


def frozen_protocol_payload() -> dict[str, object]:
    """Return the complete pre-test protocol surface used by formal matrices."""

    return {
        "format_version": 1,
        "protocol_version": A0_MR_PROTOCOL_VERSION,
        "alpha_semantics": "xgboost_weight",
        "alpha_grid": list(ALPHA_GRID),
        "pooling_methods": ["linear", "log"],
        "tie_break": list(TIE_BREAK),
        "fixed_a0": {
            "method": "linear",
            "alpha_xgboost": FIXED_A0_ALPHA_XGBOOST,
        },
        "random_conditions": list(RANDOM_CONDITIONS),
        "structured_conditions": list(STRUCTURED_CONDITIONS),
        "random_missingness_algorithm": MISSINGNESS_ALGORITHM,
        "structured_missingness_algorithm": STRUCTURED_MISSINGNESS_ALGORITHM,
        "structured_missingness_groups": {
            name: list(fields)
            for name, fields in STRUCTURED_MISSINGNESS_GROUPS.items()
        },
        "candidate_selection": {
            "candidate_min_mean_gain_vs_fixed_a0": CANDIDATE_MIN_GAIN,
            "log_pool_min_mean_gain_vs_linear_only": POOL_MIN_GAIN,
            "clean_min_delta_vs_fixed_a0": 0.0,
            "maximum_single_condition_drop_vs_fixed_a0": (
                MAX_CONDITION_DROP_VS_FIXED
            ),
            "fallback": "fixed_a0",
        },
        "structured_extension": {
            "enabled_only_when_random_condition_gate_passes": True,
            "allowed_pooling_methods_follow_random_candidate": True,
            "selection_scope": "same_setting_validation_only",
            "fallback": "fixed_a0",
        },
        "calibration": {
            "method": "scalar_temperature",
            "fit_scope": "validation_only_after_policy_lock",
            "objective": "nll",
            "participates_in_macro_f1_selection": False,
        },
    }


def frozen_protocol_hash() -> str:
    return stable_hash(frozen_protocol_payload(), A0_MR_PROTOCOL_HASH_NAMESPACE)


def fixed_a0_policies(
    conditions: Sequence[str] = FORMAL_CONDITIONS,
) -> dict[str, dict[str, object]]:
    normalized = tuple(conditions)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("conditions must be non-empty and unique")
    return {
        condition: {
            "method": "linear",
            "alpha": FIXED_A0_ALPHA_XGBOOST,
            "selected_on": "frozen",
        }
        for condition in normalized
    }


def policy_probabilities(outputs, policies):
    if set(outputs) != set(policies):
        raise ValueError("fusion policies must match output conditions")
    return {
        condition: pool_probabilities(
            values["xgboost"],
            values["schemaflow_pretrained"],
            alpha=policies[condition]["alpha"],
            method=policies[condition]["method"],
        )
        for condition, values in outputs.items()
    }


def policy_metrics(outputs, policies, label_names: tuple[str, ...]):
    probabilities = policy_probabilities(outputs, policies)
    return {
        condition: metric_payload(
            probabilities[condition], values["targets"], label_names
        )
        for condition, values in outputs.items()
    }


def _mean_macro_f1(metrics: Mapping[str, Mapping[str, object]]) -> float:
    return float(np.mean([values["macro_f1"] for values in metrics.values()]))


def select_a0_mr_validation(
    outputs: Mapping[str, Mapping[str, object]],
    label_names: tuple[str, ...],
) -> dict[str, object]:
    """Lock A0-MR using validation outputs and the already validated procedure."""

    if tuple(outputs) != FORMAL_CONDITIONS:
        raise ValueError("A0-MR outputs must follow the frozen condition order")
    if not isinstance(label_names, tuple) or len(label_names) < 2:
        raise ValueError("label_names must contain at least two classes")

    random_outputs = {name: outputs[name] for name in RANDOM_CONDITIONS}
    linear = {}
    linear_or_log = {}
    for condition, values in random_outputs.items():
        arguments = (
            values["xgboost"],
            values["schemaflow_pretrained"],
            values["targets"],
            label_names,
        )
        linear[condition] = select_fusion(
            *arguments, alpha_grid=ALPHA_GRID, methods=("linear",)
        )
        linear_or_log[condition] = select_fusion(
            *arguments, alpha_grid=ALPHA_GRID, methods=("linear", "log")
        )

    def policies(results):
        return {
            condition: {
                "method": result["selected"]["method"],
                "alpha": result["selected"]["alpha"],
                "selected_on": condition,
            }
            for condition, result in results.items()
        }

    random_fixed = fixed_a0_policies(RANDOM_CONDITIONS)
    linear_policies = policies(linear)
    pool_policies = policies(linear_or_log)
    fixed_metrics = policy_metrics(random_outputs, random_fixed, label_names)
    linear_metrics = policy_metrics(
        random_outputs, linear_policies, label_names
    )
    pool_metrics = policy_metrics(random_outputs, pool_policies, label_names)
    fixed_mean = _mean_macro_f1(fixed_metrics)
    linear_mean = _mean_macro_f1(linear_metrics)
    pool_mean = _mean_macro_f1(pool_metrics)
    if pool_mean - linear_mean >= POOL_MIN_GAIN:
        candidate_name = "missing_rate_conditioned_linear_or_log"
        candidate_search = linear_or_log
        candidate_policies = pool_policies
        candidate_metrics = pool_metrics
        candidate_mean = pool_mean
        allowed_methods = ("linear", "log")
    else:
        candidate_name = "missing_rate_conditioned_linear"
        candidate_search = linear
        candidate_policies = linear_policies
        candidate_metrics = linear_metrics
        candidate_mean = linear_mean
        allowed_methods = ("linear",)
    random_deltas = {
        condition: candidate_metrics[condition]["macro_f1"]
        - fixed_metrics[condition]["macro_f1"]
        for condition in RANDOM_CONDITIONS
    }
    random_gate_passed = (
        candidate_mean - fixed_mean >= CANDIDATE_MIN_GAIN
        and random_deltas["clean"] >= -1e-12
        and min(random_deltas.values()) >= -MAX_CONDITION_DROP_VS_FIXED
    )
    locked_random = candidate_policies if random_gate_passed else random_fixed

    structured_search = {}
    structured_policies = {}
    for condition in STRUCTURED_CONDITIONS:
        if random_gate_passed:
            values = outputs[condition]
            result = select_fusion(
                values["xgboost"],
                values["schemaflow_pretrained"],
                values["targets"],
                label_names,
                alpha_grid=ALPHA_GRID,
                methods=allowed_methods,
            )
            structured_search[condition] = result
            structured_policies[condition] = {
                "method": result["selected"]["method"],
                "alpha": result["selected"]["alpha"],
                "selected_on": condition,
            }
        else:
            structured_search[condition] = {
                "selection_scope": "validation_only",
                "search_skipped_by_frozen_random_condition_gate": True,
            }
            structured_policies[condition] = {
                "method": "linear",
                "alpha": FIXED_A0_ALPHA_XGBOOST,
                "selected_on": "frozen_fallback",
            }

    locked_policies = {**locked_random, **structured_policies}
    fixed_policies = fixed_a0_policies()
    return {
        "format_version": 1,
        "protocol_version": A0_MR_PROTOCOL_VERSION,
        "protocol_hash": frozen_protocol_hash(),
        "selection_scope": "validation_only",
        "test_samples_materialized": False,
        "test_metrics_used": False,
        "random_condition_selection": {
            "linear_only": linear,
            "linear_or_log": linear_or_log,
            "selected_candidate_search": candidate_search,
        },
        "routing_gate": {
            "passed": random_gate_passed,
            "candidate_name": candidate_name,
            "candidate_mean_macro_f1": candidate_mean,
            "fixed_a0_mean_macro_f1": fixed_mean,
            "candidate_mean_gain": candidate_mean - fixed_mean,
            "condition_deltas_vs_fixed_a0": random_deltas,
            "thresholds": frozen_protocol_payload()["candidate_selection"],
        },
        "structured_condition_search": structured_search,
        "fixed_a0": {
            "name": "a0_fixed_linear_alpha_xgb_0p55",
            "policies": fixed_policies,
            "validation_metrics": policy_metrics(
                outputs, fixed_policies, label_names
            ),
        },
        "locked_policy": {
            "name": (
                candidate_name
                if random_gate_passed
                else "a0_fixed_linear_alpha_xgb_0p55"
            ),
            "policies": locked_policies,
            "validation_metrics": policy_metrics(
                outputs, locked_policies, label_names
            ),
        },
    }


__all__ = [
    "A0_MR_PROTOCOL_VERSION",
    "ALPHA_GRID",
    "FORMAL_CONDITIONS",
    "RANDOM_CONDITIONS",
    "STRUCTURED_CONDITIONS",
    "fixed_a0_policies",
    "frozen_protocol_hash",
    "frozen_protocol_payload",
    "policy_metrics",
    "policy_probabilities",
    "select_a0_mr_validation",
]
