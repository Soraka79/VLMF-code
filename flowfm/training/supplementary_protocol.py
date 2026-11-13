"""Frozen supplementary protocol for the revision resampling matrix.

The protocol extends the frozen A0-MR selection (``a0_mr_protocol``) with
exporter-profile conditions and per-record policy dispatch. Everything that can
change a sealed-test result is declared here and hashed before any sealed test
file is opened; the original eight conditions and their selection are reused
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from flowfm.data.exporter_missingness import (
    BASE_RANDOM_FRACTIONS,
    COMPOUND_PROFILES,
    EXPORTER_MISSINGNESS_ALGORITHM,
    EXPORTER_PROFILES,
    INTERMITTENT_MISSINGNESS_ALGORITHM,
    INTERMITTENT_PROFILES,
    MODEL_FIELDS,
)
from flowfm.data.identity import stable_hash
from flowfm.training.a0_mr_protocol import (
    ALPHA_GRID,
    FIXED_A0_ALPHA_XGBOOST,
    FORMAL_CONDITIONS,
    fixed_a0_policies,
    frozen_protocol_hash,
)
from flowfm.training.probability_fusion import pool_probabilities, select_fusion
from flowfm.training.safe_fusion import apply_scalar_temperature


SUPPLEMENTARY_PROTOCOL_VERSION = "revision-supplementary-a0-mr-v1"
SUPPLEMENTARY_HASH_NAMESPACE = "flowfm-revision-supplementary-protocol-v1"

PROFILE_CONDITIONS = tuple(f"profile__{name}" for name in EXPORTER_PROFILES)
INTERMITTENT_CONDITIONS = tuple(
    f"intermittent__{name}" for name in INTERMITTENT_PROFILES
)
COMPOUND_CONDITIONS = tuple(
    f"compound__{name.removeprefix('compound_')}" for name in COMPOUND_PROFILES
)
VALIDATION_CONDITIONS = FORMAL_CONDITIONS + PROFILE_CONDITIONS
TEST_CONDITIONS = VALIDATION_CONDITIONS + INTERMITTENT_CONDITIONS + COMPOUND_CONDITIONS

ENDPOINTS = ("xgboost", "lightgbm", "schemaflow_pretrained")
# a0_mr uses the registry of the eight paper conditions; a0_mr_registered also
# registers the three exporter profiles. Both resolve each test record's mask.
POLICIES = ("fixed_a0", "a0_mr", "a0_mr_registered")
FALLBACK_POLICY = {
    "method": "linear",
    "alpha": FIXED_A0_ALPHA_XGBOOST,
    "temperature": 1.0,
    "selected_on": "unresolved_mask_fallback",
}

POST_EVALUATION_ANALYSES = {
    "policy_controls": {
        "global_validation_fusion": (
            "one pool and weight per setting from the 42 candidates, chosen by the "
            "equal validation mean over clean, 30% and 50% random loss"
        ),
        "always_conditioned_linear_or_log": (
            "per-condition 42-candidate validation maximiser with no family gate "
            "and no fallback"
        ),
        "logistic_stacking": (
            "learned_ensemble_protocol_v1 definition, refit on these validation "
            "archives"
        ),
    },
    "validation_size_curve": {
        "validation_group_subsample_rows": [1000, 5000, 20000, "all"],
        "repeats": 20,
        "seed": 20261018,
        "unit": "whole leakage groups, class-stratified",
    },
    "split_half_validation": {
        "repeats": 50,
        "seed": 20261019,
        "unit": "whole leakage groups, class-stratified halves",
        "reported": ["policy agreement", "held-out-half optimism", "test regret"],
    },
    "test_group_cluster_bootstrap": {
        "resamples": 2000,
        "seed": 20261020,
        "unit": "sealed-test leakage group",
    },
    "rare_class_recall": {"interval": "wilson_95", "with_support": True},
    "resolver_latency": {
        "scope": "mask resolution, policy lookup, pooling and temperature",
        "endpoint_inference_reported_separately": True,
    },
}


def frozen_supplementary_payload() -> dict[str, object]:
    """Return every pre-test decision of the supplementary matrix."""

    return {
        "format_version": 1,
        "protocol_version": SUPPLEMENTARY_PROTOCOL_VERSION,
        "base_a0_mr_protocol_hash": frozen_protocol_hash(),
        "endpoints": list(ENDPOINTS),
        "policies": list(POLICIES),
        "validation_conditions": list(VALIDATION_CONDITIONS),
        "test_conditions": list(TEST_CONDITIONS),
        "model_fields": list(MODEL_FIELDS),
        "exporter_profiles": {
            name: list(fields) for name, fields in EXPORTER_PROFILES.items()
        },
        "compound_profiles": {
            name: list(fields) for name, fields in COMPOUND_PROFILES.items()
        },
        "intermittent_profiles": {
            name: dict(spec) for name, spec in INTERMITTENT_PROFILES.items()
        },
        "missingness_algorithms": {
            "exporter_and_compound": EXPORTER_MISSINGNESS_ALGORITHM,
            "intermittent": INTERMITTENT_MISSINGNESS_ALGORITHM,
            "intermittent_seed": "cohort split_seed",
        },
        "resolver": {
            "order": ["exact missing-field set", "random-loss missing-field count"],
            "random_fractions": dict(BASE_RANDOM_FRACTIONS),
            "a0_mr_registry": list(FORMAL_CONDITIONS),
            "a0_mr_registered_registry": list(VALIDATION_CONDITIONS),
            "known_count_collision": (
                "compound__tcp_state_packet_length has 20 missing fields and "
                "resolves to missing_50pct"
            ),
        },
        "fallback_policy": dict(FALLBACK_POLICY),
        "profile_selection": {
            "rule": (
                "same as the frozen structured extension: selected on same-setting "
                "validation from the admitted pool family when the random gate "
                "passes, otherwise fixed_a0"
            ),
            "alpha_grid": list(ALPHA_GRID),
        },
        "calibration": {
            "method": "scalar_temperature",
            "fit_scope": "validation_only_after_policy_lock",
            "applied": "per record by resolved condition; T=1 for the fallback",
            "primary_metric_uses_calibration": False,
        },
        "primary_metric": "macro_f1",
        "sealed_test": {
            "open_budget_per_cohort": 1,
            "claim_receipt": True,
            "validation_dry_run_before_claim": True,
        },
        "post_evaluation_analyses": POST_EVALUATION_ANALYSES,
    }


def frozen_supplementary_hash() -> str:
    return stable_hash(frozen_supplementary_payload(), SUPPLEMENTARY_HASH_NAMESPACE)


def _admitted_methods(selection: Mapping[str, object]) -> tuple[str, ...]:
    name = selection["routing_gate"]["candidate_name"]
    if name == "missing_rate_conditioned_linear_or_log":
        return ("linear", "log")
    if name == "missing_rate_conditioned_linear":
        return ("linear",)
    raise ValueError(f"unknown A0-MR candidate family: {name}")


def select_registered_profiles(
    outputs: Mapping[str, Mapping[str, object]],
    selection: Mapping[str, object],
    label_names: tuple[str, ...],
) -> dict[str, object]:
    """Extend a locked A0-MR selection to the registered exporter profiles."""

    if selection.get("protocol_hash") != frozen_protocol_hash():
        raise ValueError("A0-MR selection protocol changed")
    missing = [name for name in PROFILE_CONDITIONS if name not in outputs]
    if missing:
        raise ValueError("profile validation outputs are missing: " + ", ".join(missing))
    gate_passed = bool(selection["routing_gate"]["passed"])
    methods = _admitted_methods(selection)
    search = {}
    policies = {}
    for condition in PROFILE_CONDITIONS:
        if gate_passed:
            values = outputs[condition]
            result = select_fusion(
                values["xgboost"],
                values["schemaflow_pretrained"],
                values["targets"],
                label_names,
                alpha_grid=ALPHA_GRID,
                methods=methods,
            )
            search[condition] = result
            policies[condition] = {
                "method": result["selected"]["method"],
                "alpha": result["selected"]["alpha"],
                "selected_on": condition,
            }
        else:
            search[condition] = {
                "selection_scope": "validation_only",
                "search_skipped_by_frozen_random_condition_gate": True,
            }
            policies[condition] = dict(fixed_a0_policies((condition,))[condition])
            policies[condition]["selected_on"] = "frozen_fallback"
    return {
        "format_version": 1,
        "protocol_version": SUPPLEMENTARY_PROTOCOL_VERSION,
        "protocol_hash": frozen_supplementary_hash(),
        "base_a0_mr_protocol_hash": frozen_protocol_hash(),
        "random_gate_passed": gate_passed,
        "admitted_methods": list(methods),
        "profile_condition_search": search,
        "registered_policies": {
            **selection["locked_policy"]["policies"],
            **policies,
        },
    }


def missing_mask_matrix(samples) -> tuple[np.ndarray, tuple[str, ...]]:
    """Boolean ``(records, fields)`` missing mask with the samples' field order."""

    if not samples:
        raise ValueError("missing-mask matrix requires samples")
    field_order = tuple(samples[0].feature_ids)
    if any(tuple(sample.feature_ids) != field_order for sample in samples):
        raise ValueError("samples do not share one field order")
    return np.asarray([sample.missing_mask for sample in samples], dtype=bool), field_order


def dispatch_probabilities(
    tree_probabilities,
    neural_probabilities,
    resolved_names,
    policies: Mapping[str, Mapping[str, object]],
    *,
    temperatures: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Pool each record with its resolved condition's locked policy.

    ``resolved_names`` holds one registered condition per record, or ``""``
    for a mask the resolver could not place; those records use the fixed-fusion
    fallback at temperature 1.
    """

    tree = np.asarray(tree_probabilities, dtype=np.float64)
    neural = np.asarray(neural_probabilities, dtype=np.float64)
    names = np.asarray(resolved_names, dtype=object)
    if tree.shape != neural.shape or names.shape != (tree.shape[0],):
        raise ValueError("dispatch inputs do not align")
    result = np.empty_like(tree)
    counts = {}
    for name in sorted(set(names.tolist())):
        index = np.flatnonzero(names == name)
        if name == "":
            policy, temperature = FALLBACK_POLICY, FALLBACK_POLICY["temperature"]
        else:
            if name not in policies:
                raise ValueError(f"resolved condition has no locked policy: {name}")
            policy = policies[name]
            temperature = 1.0 if temperatures is None else float(temperatures[name])
        pooled = pool_probabilities(
            tree[index],
            neural[index],
            alpha=policy["alpha"],
            method=policy["method"],
        )
        if temperature != 1.0:
            pooled = apply_scalar_temperature(pooled, temperature)
        result[index] = pooled
        counts[name or "fallback"] = int(index.size)
    return result, {
        "record_count": int(tree.shape[0]),
        "resolved_counts": counts,
        "fallback_fraction": counts.get("fallback", 0) / tree.shape[0],
    }


__all__ = [
    "COMPOUND_CONDITIONS",
    "ENDPOINTS",
    "FALLBACK_POLICY",
    "INTERMITTENT_CONDITIONS",
    "POLICIES",
    "POST_EVALUATION_ANALYSES",
    "PROFILE_CONDITIONS",
    "SUPPLEMENTARY_PROTOCOL_VERSION",
    "TEST_CONDITIONS",
    "VALIDATION_CONDITIONS",
    "dispatch_probabilities",
    "frozen_supplementary_hash",
    "frozen_supplementary_payload",
    "missing_mask_matrix",
    "select_registered_profiles",
]
