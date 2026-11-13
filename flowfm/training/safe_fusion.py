"""Low-capacity robustness components for A0-MR-Safe development."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from flowfm.data.contracts import FlowSample
from flowfm.data.missingness import (
    MISSINGNESS_ALGORITHM,
    STRUCTURED_MISSINGNESS_ALGORITHM,
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_deterministic_field_missingness,
    apply_structured_field_group_missingness,
)
from flowfm.training.confirmatory_statistics import (
    confirmatory_metric_payload,
    paired_cluster_bootstrap_macro_f1,
)


SAFE_PROTOCOL_VERSION = "a0-mr-safe-v1"
CONSISTENCY_ALGORITHM = "deterministic-one-view-per-train-record-v1"
CONSISTENCY_CONDITIONS = (
    "missing_30pct",
    "missing_50pct",
) + tuple(f"structured__{name}" for name in STRUCTURED_MISSINGNESS_GROUPS)
MIN_RARE_VALIDATION_SUPPORT = 100
MAX_RARE_RECALL_DROP = 0.02


def balanced_softmax_cross_entropy(logits, targets, class_counts):
    """Balanced Softmax using train-only class counts."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise TypeError("balanced softmax logits must be a rank-2 tensor")
    if not isinstance(targets, torch.Tensor) or targets.shape != (logits.shape[0],):
        raise TypeError("balanced softmax targets must align with logits")
    counts = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device)
    if counts.shape != (logits.shape[1],) or torch.any(counts < 0):
        raise ValueError("balanced softmax class counts must match logits")
    if torch.any(counts[targets] <= 0):
        raise ValueError("balanced softmax target has zero train support")
    epsilon = torch.finfo(logits.dtype).tiny
    adjusted = logits + torch.log(counts.clamp_min(epsilon)).unsqueeze(0)
    return F.cross_entropy(adjusted, targets)


def jensen_shannon_consistency(clean_logits, missing_logits):
    """Mean Jensen-Shannon divergence between paired coarse predictions."""

    if (
        not isinstance(clean_logits, torch.Tensor)
        or not isinstance(missing_logits, torch.Tensor)
        or clean_logits.shape != missing_logits.shape
        or clean_logits.ndim != 2
    ):
        raise TypeError("paired consistency logits must be aligned rank-2 tensors")
    clean_log = F.log_softmax(clean_logits, dim=1)
    missing_log = F.log_softmax(missing_logits, dim=1)
    clean = clean_log.exp()
    missing = missing_log.exp()
    midpoint = 0.5 * (clean + missing)
    midpoint_log = torch.log(midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny))
    divergence = 0.5 * (
        torch.sum(clean * (clean_log - midpoint_log), dim=1)
        + torch.sum(missing * (missing_log - midpoint_log), dim=1)
    )
    return divergence.mean()


def _consistency_condition(seed: int, sample_id: str) -> str:
    value = "\0".join((CONSISTENCY_ALGORITHM, str(seed), sample_id))
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return CONSISTENCY_CONDITIONS[
        int.from_bytes(digest[:8], byteorder="big") % len(CONSISTENCY_CONDITIONS)
    ]


def deterministic_paired_missing_view(
    samples: Sequence[FlowSample],
    *,
    seed: int,
) -> tuple[tuple[FlowSample, ...], dict[str, object]]:
    """Assign exactly one deterministic missing view to each train record."""

    if type(seed) is not int or seed < 0:
        raise ValueError("consistency missing seed must be a non-negative integer")
    values = tuple(samples)
    if not values or any(not isinstance(sample, FlowSample) for sample in values):
        raise ValueError("consistency training samples must be non-empty FlowSamples")
    assignments = tuple(
        _consistency_condition(seed, sample.sample_id) for sample in values
    )
    by_condition = {
        condition: tuple(
            index for index, assigned in enumerate(assignments) if assigned == condition
        )
        for condition in CONSISTENCY_CONDITIONS
    }
    if any(not indices for indices in by_condition.values()):
        raise ValueError("training data is too small to cover every missing condition")

    perturbed: list[FlowSample | None] = [None] * len(values)
    condition_reports = {}
    for condition, indices in by_condition.items():
        members = tuple(values[index] for index in indices)
        if condition == "missing_30pct":
            conditioned, report = apply_deterministic_field_missingness(
                members, fraction=0.3, seed=seed
            )
        elif condition == "missing_50pct":
            conditioned, report = apply_deterministic_field_missingness(
                members, fraction=0.5, seed=seed
            )
        else:
            conditioned, report = apply_structured_field_group_missingness(
                members, group_name=condition.removeprefix("structured__")
            )
        for index, sample in zip(indices, conditioned, strict=True):
            perturbed[index] = sample
        condition_reports[condition] = {
            **report,
            "training_data_perturbed": True,
            "training_use": "coarse_prediction_consistency_only",
        }
    if any(sample is None for sample in perturbed):
        raise RuntimeError("paired missing view did not preserve train order")

    assignment_digest = hashlib.sha256()
    for sample, condition in sorted(
        zip(values, assignments, strict=True), key=lambda item: item[0].sample_id
    ):
        assignment_digest.update(
            json.dumps(
                [sample.sample_id, condition], separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        )
        assignment_digest.update(b"\n")
    return tuple(perturbed), {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "algorithm": CONSISTENCY_ALGORITHM,
        "random_missingness_algorithm": MISSINGNESS_ALGORITHM,
        "structured_missingness_algorithm": STRUCTURED_MISSINGNESS_ALGORITHM,
        "scope": "target_train_only",
        "seed": seed,
        "sample_count": len(values),
        "conditions": list(CONSISTENCY_CONDITIONS),
        "condition_counts": dict(Counter(assignments)),
        "assignment_sha256": assignment_digest.hexdigest(),
        "condition_reports": condition_reports,
        "validation_or_test_records_used": False,
    }


def _probabilities(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] < 2
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a finite probability matrix")
    return matrix


def apply_scalar_temperature(probabilities, temperature: float) -> np.ndarray:
    matrix = _probabilities(probabilities, name="temperature probabilities")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("temperature must be finite and positive")
    log_values = np.log(np.clip(matrix, np.finfo(np.float64).tiny, 1.0))
    log_values /= float(temperature)
    log_values -= log_values.max(axis=1, keepdims=True)
    calibrated = np.exp(log_values)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _nll(probabilities: np.ndarray, targets: np.ndarray) -> float:
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean())


def fit_scalar_temperature(
    probabilities,
    targets,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    iterations: int = 80,
) -> dict[str, object]:
    """Fit one validation-only temperature with deterministic golden search."""

    matrix = _probabilities(probabilities, name="temperature fit probabilities")
    target_values = np.asarray(targets, dtype=np.int64)
    if target_values.shape != (matrix.shape[0],) or np.any(target_values < 0) or np.any(
        target_values >= matrix.shape[1]
    ):
        raise ValueError("temperature targets do not align with probabilities")
    if not 0.0 < minimum < 1.0 < maximum or type(iterations) is not int or iterations < 20:
        raise ValueError("invalid frozen scalar-temperature search")

    lower, upper = math.log(minimum), math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)

    def objective(log_temperature):
        return _nll(
            apply_scalar_temperature(matrix, math.exp(log_temperature)), target_values
        )

    left_value, right_value = objective(left), objective(right)
    for _ in range(iterations):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    candidates = ((0.0, objective(0.0)), (left, left_value), (right, right_value))
    selected_log, selected_nll = min(candidates, key=lambda item: (item[1], abs(item[0])))
    before = objective(0.0)
    if selected_nll >= before - 1e-12:
        selected_log, selected_nll = 0.0, before
    temperature = math.exp(selected_log)
    return {
        "format_version": 1,
        "selection_scope": "validation_only",
        "objective": "nll",
        "method": "bounded_log_temperature_golden_section",
        "bounds": [minimum, maximum],
        "iterations": iterations,
        "temperature": temperature,
        "nll_before": before,
        "nll_after": selected_nll,
        "improved_nll": selected_nll < before - 1e-12,
    }


def safe_route_and_calibrate(
    outputs: Mapping[str, Mapping[str, object]],
    candidate_probabilities: Mapping[str, np.ndarray],
    xgboost_probabilities: Mapping[str, np.ndarray],
    group_ids: Sequence[str],
    label_names: tuple[str, ...],
    *,
    rare_labels: Sequence[str],
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
    minimum_rare_support: int = MIN_RARE_VALIDATION_SUPPORT,
    maximum_rare_recall_drop: float = MAX_RARE_RECALL_DROP,
):
    """Apply the predeclared condition-level safety fallback and calibration."""

    conditions = tuple(outputs)
    if set(candidate_probabilities) != set(conditions) or set(
        xgboost_probabilities
    ) != set(conditions):
        raise ValueError("safe routing probability conditions do not align")
    bootstrap = paired_cluster_bootstrap_macro_f1(
        outputs,
        candidate_probabilities,
        xgboost_probabilities,
        group_ids,
        class_count=len(label_names),
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    candidate_metrics = {
        condition: confirmatory_metric_payload(
            candidate_probabilities[condition], outputs[condition]["targets"], label_names
        )
        for condition in conditions
    }
    xgboost_metrics = {
        condition: confirmatory_metric_payload(
            xgboost_probabilities[condition], outputs[condition]["targets"], label_names
        )
        for condition in conditions
    }
    decisions = {}
    routed = {}
    for condition in conditions:
        candidate = candidate_metrics[condition]
        xgboost = xgboost_metrics[condition]
        eligible_rare = tuple(
            label
            for label in rare_labels
            if candidate["per_class_support"].get(label, 0) >= minimum_rare_support
        )
        rare_drops = {
            label: float(
                candidate["per_class_recall"][label]
                - xgboost["per_class_recall"][label]
            )
            for label in eligible_rare
        }
        zero_out = tuple(
            label
            for label in eligible_rare
            if xgboost["per_class_recall"][label] >= 0.10
            and candidate["per_class_recall"][label] <= 0.0
        )
        ci = bootstrap["summaries"][condition]
        checks = {
            "bootstrap_ci_lower_above_zero": ci["ci95_lower"] > 0.0,
            "rare_recall_guard": all(
                delta >= -maximum_rare_recall_drop for delta in rare_drops.values()
            ),
            "no_rare_zero_out": not zero_out,
        }
        use_fusion = all(checks.values())
        routed[condition] = np.asarray(
            candidate_probabilities[condition]
            if use_fusion
            else xgboost_probabilities[condition],
            dtype=np.float64,
        )
        decisions[condition] = {
            "route": "a0_mr_candidate" if use_fusion else "xgboost_fallback",
            "checks": checks,
            "candidate_minus_xgboost": ci,
            "eligible_rare_labels": list(eligible_rare),
            "rare_recall_deltas": rare_drops,
            "rare_zero_out": list(zero_out),
        }

    routed_metrics = {
        condition: confirmatory_metric_payload(
            routed[condition], outputs[condition]["targets"], label_names
        )
        for condition in conditions
    }
    temperatures = {}
    calibrated = {}
    calibrated_metrics = {}
    for condition in conditions:
        targets = np.asarray(outputs[condition]["targets"], dtype=np.int64)
        temperatures[condition] = fit_scalar_temperature(routed[condition], targets)
        calibrated[condition] = apply_scalar_temperature(
            routed[condition], temperatures[condition]["temperature"]
        )
        calibrated_metrics[condition] = confirmatory_metric_payload(
            calibrated[condition], targets, label_names
        )
    payload = {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "selection_scope": "validation_only",
        "fallback": "xgboost",
        "minimum_rare_support": minimum_rare_support,
        "maximum_rare_recall_drop": maximum_rare_recall_drop,
        "bootstrap": bootstrap,
        "decisions": decisions,
        "candidate_metrics": candidate_metrics,
        "xgboost_metrics": xgboost_metrics,
        "routed_metrics": routed_metrics,
        "scalar_temperature": temperatures,
        "calibrated_metrics": calibrated_metrics,
    }
    return payload, routed, calibrated


def evaluate_seed42_setting_gate(
    *,
    current_a0_mr_metrics: Mapping[str, Mapping[str, object]],
    safe_metrics: Mapping[str, Mapping[str, object]],
    calibrated_metrics: Mapping[str, Mapping[str, object]],
    xgboost_metrics: Mapping[str, Mapping[str, object]],
    rare_labels: Sequence[str],
) -> dict[str, object]:
    """Apply the frozen per-setting seed-42 continuation gate."""

    random_conditions = ("clean", "missing_30pct", "missing_50pct")
    tcp_condition = "structured__tcp_state_retransmission"
    conditions = tuple(safe_metrics)
    if not set(random_conditions + (tcp_condition,)).issubset(conditions):
        raise ValueError("safe gate is missing required validation conditions")
    if any(set(values) != set(conditions) for values in (current_a0_mr_metrics, calibrated_metrics, xgboost_metrics)):
        raise ValueError("safe gate metrics do not align")

    random_deltas = {
        condition: float(
            safe_metrics[condition]["macro_f1"]
            - current_a0_mr_metrics[condition]["macro_f1"]
        )
        for condition in random_conditions
    }
    current_retention = {
        condition: float(
            current_a0_mr_metrics[condition]["macro_f1"]
            / current_a0_mr_metrics["clean"]["macro_f1"]
        )
        for condition in random_conditions[1:]
    }
    safe_retention = {
        condition: float(
            safe_metrics[condition]["macro_f1"] / safe_metrics["clean"]["macro_f1"]
        )
        for condition in random_conditions[1:]
    }
    rare_failures = []
    zero_out = []
    for condition in conditions:
        for label in rare_labels:
            if safe_metrics[condition]["per_class_support"].get(label, 0) < MIN_RARE_VALIDATION_SUPPORT:
                continue
            delta = float(
                safe_metrics[condition]["per_class_recall"][label]
                - xgboost_metrics[condition]["per_class_recall"][label]
            )
            if delta < -MAX_RARE_RECALL_DROP:
                rare_failures.append(
                    {"condition": condition, "label": label, "recall_delta": delta}
                )
            if (
                xgboost_metrics[condition]["per_class_recall"][label] >= 0.10
                and safe_metrics[condition]["per_class_recall"][label] <= 0.0
            ):
                zero_out.append({"condition": condition, "label": label})

    mean_nll_before = float(np.mean([safe_metrics[c]["nll"] for c in conditions]))
    mean_nll_after = float(
        np.mean([calibrated_metrics[c]["nll"] for c in conditions])
    )
    mean_ece_before = float(np.mean([safe_metrics[c]["ece"] for c in conditions]))
    mean_ece_after = float(
        np.mean([calibrated_metrics[c]["ece"] for c in conditions])
    )
    checks = {
        "mean_random_macro_f1_gain_at_least_0p005": (
            float(np.mean(tuple(random_deltas.values()))) >= 0.005
        ),
        "no_random_condition_drop_below_minus_0p002": min(random_deltas.values()) >= -0.002,
        "tcp_not_below_xgboost": (
            safe_metrics[tcp_condition]["macro_f1"]
            >= xgboost_metrics[tcp_condition]["macro_f1"] - 1e-12
        ),
        "rare_recall_guard": not rare_failures,
        "missing_30_retention_improved_or_at_least_0p90": (
            safe_retention["missing_30pct"] >= 0.90
            or safe_retention["missing_30pct"]
            - current_retention["missing_30pct"]
            >= 0.02
        ),
        "missing_50_retention_improved_at_least_0p02": (
            safe_retention["missing_50pct"]
            - current_retention["missing_50pct"]
            >= 0.02
        ),
        "calibration_mean_nll_improved": mean_nll_after < mean_nll_before - 1e-12,
        "calibration_mean_ece_not_worse": mean_ece_after <= mean_ece_before + 1e-12,
        "no_rare_class_collapse": not zero_out,
    }
    return {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "predeclared": True,
        "random_macro_f1_deltas_vs_current_a0_mr": random_deltas,
        "mean_random_macro_f1_gain": float(np.mean(tuple(random_deltas.values()))),
        "current_retention": current_retention,
        "safe_retention": safe_retention,
        "rare_recall_failures": rare_failures,
        "rare_zero_out_failures": zero_out,
        "calibration": {
            "mean_nll_before": mean_nll_before,
            "mean_nll_after": mean_nll_after,
            "mean_ece_before": mean_ece_before,
            "mean_ece_after": mean_ece_after,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "CONSISTENCY_ALGORITHM",
    "CONSISTENCY_CONDITIONS",
    "MAX_RARE_RECALL_DROP",
    "MIN_RARE_VALIDATION_SUPPORT",
    "SAFE_PROTOCOL_VERSION",
    "apply_scalar_temperature",
    "balanced_softmax_cross_entropy",
    "deterministic_paired_missing_view",
    "evaluate_seed42_setting_gate",
    "fit_scalar_temperature",
    "jensen_shannon_consistency",
    "safe_route_and_calibrate",
]
