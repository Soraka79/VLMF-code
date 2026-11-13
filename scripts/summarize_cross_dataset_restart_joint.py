#!/usr/bin/env python3
"""Joint audit of independent formal matrices and the disclosed UNSW restart."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.training.a0_mr_protocol import FORMAL_CONDITIONS, RANDOM_CONDITIONS
from scripts.run_nf_cse_a0_mr_confirmation import _sha256_file
from scripts.summarize_cross_dataset_a0_mr import (
    _comparison,
    _macro_f1_means,
    _new_dataset_evidence,
    _seed_stability,
    dataset_stratified_paired_bootstrap,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_MATRIX_ROOTS = {
    "NF-BoT-IoT-v2": _RESULTS_ROOT / "cross-dataset-formal-matrix-v1/nf_bot",
    "NF-CSE-CIC-IDS2018-v2": (
        _RESULTS_ROOT / "cross-dataset-formal-matrix-v1/nf_cse"
    ),
    "NF-ToN-IoT-v2": _RESULTS_ROOT / "cross-dataset-formal-matrix-v3/nf_ton",
}
_RESTART_ROOT = (
    _RESULTS_ROOT / "cross-dataset-formal-restart-matrix-v1/nf_unsw_restart"
)
_RESTART_AUDIT_ROOT = (
    _RESULTS_ROOT / "cross-dataset-formal-restart-audits-v1/nf_unsw_restart"
)
_DEPLOYMENT_ROOT = (
    _RESULTS_ROOT / "nf-unsw-restart-template-tree-deployment-v2"
)
_DEFAULT_OUTPUT = _RESULTS_ROOT / "cross-dataset-a0-mr-joint-audit-v2"
_INDEPENDENT_DATASETS = tuple(_MATRIX_ROOTS)
_RESTART_DATASET = "NF-UNSW-NB15-v2"
_RESTART_KEY = "nf_unsw_restart"
_RESTART_CLAIM_SCOPE = (
    "full_source_leakage_safe_restart_not_historically_independent"
)
_RESTART_HISTORY_POLICY = (
    "reuse_retired_historical_exposure_with_disclosure"
)
_RECOVERY_PROTOCOL_VERSION = "sealed-test-pre-materialization-recovery-v1"
_MODEL_SEEDS = (17, 42, 20260725)
_TRAIN_PERCENTS = (1, 5, 10)
_RESTART_RARE_LABELS = ("dos", "scan", "backdoor", "malware")
_UPGRADE_THRESHOLDS = {
    "minimum_nonnegative_datasets": 3,
    "minimum_cross_dataset_mean_macro_f1_gain": 0.005,
    "dataset_stratified_bootstrap_ci95_lower_above_zero": True,
    "minimum_datasets_missing_30_better_than_fixed_a0": 3,
    "minimum_datasets_missing_50_better_than_fixed_a0": 3,
    "no_rare_class_recall_collapse": True,
    "calibrated_mean_nll_and_brier_improve": True,
    "calibrated_mean_ece_not_worse": True,
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(path):
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_artifact(record, description):
    path = Path(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or _sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError(f"locked {description} artifact changed")
    return path


def _assert_finite_tree(value, path="root"):
    if isinstance(value, dict):
        for key, current in value.items():
            _assert_finite_tree(current, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, current in enumerate(value):
            _assert_finite_tree(current, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite formal result at {path}")


def _verify_formal_matrix(root, dataset_key):
    root = Path(root)
    if (root / ".incomplete").exists():
        raise RuntimeError(f"{dataset_key} formal matrix is incomplete")
    summary_path = root / "formal_matrix_summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("status") != "formal_matrix_completed"
        or summary.get("protocol_version")
        != "cross-dataset-a0-mr-formal-matrix-v1"
        or summary.get("dataset") != dataset_key
        or summary.get("setting_count") != 9
        or tuple(summary.get("conditions", ())) != FORMAL_CONDITIONS
    ):
        raise RuntimeError(f"{dataset_key} formal matrix contract is invalid")
    lock = _read_json(root / "all_validation_selections.lock.json")
    if (
        lock.get("status")
        != "all_nine_validation_selections_locked_before_test"
        or lock.get("setting_count") != 9
    ):
        raise RuntimeError(f"{dataset_key} global validation lock is invalid")
    receipt_path = _verify_artifact(
        summary["sealed_test_access_receipt"],
        f"{dataset_key} sealed-test receipt",
    )
    receipt = _read_json(receipt_path)
    if (
        receipt.get("status") != "single_test_open_completed"
        or receipt.get("sealed_test_open_count") != 1
        or receipt.get("setting_count") != 9
    ):
        raise RuntimeError(f"{dataset_key} sealed-test access is invalid")
    evidence = _new_dataset_evidence(summary)
    _assert_finite_tree(summary, dataset_key)
    return summary_path, summary, evidence


def verify_restart_disclosure(plan, feasibility, overlap):
    """Verify that the UNSW result cannot be treated as independent evidence."""

    if (
        plan.get("dataset") != _RESTART_KEY
        or plan.get("dataset_id") != _RESTART_DATASET
        or plan.get("claim_scope") != _RESTART_CLAIM_SCOPE
        or plan.get("historical_exposure_policy") != _RESTART_HISTORY_POLICY
        or plan.get("historically_independent") is not False
    ):
        raise RuntimeError("NF-UNSW restart plan lacks the frozen disclosure")
    if (
        feasibility.get("status")
        != "eligible_for_disclosed_full_source_restart"
        or feasibility.get("claim_scope") != _RESTART_CLAIM_SCOPE
        or feasibility.get("historically_independent_confirmatory_target")
        is not False
        or feasibility.get("historical_overlap_expected") is not True
    ):
        raise RuntimeError("NF-UNSW restart feasibility disclosure is invalid")
    if (
        overlap.get("verified_zero_internal_overlap") is not True
        or overlap.get("verified_zero_historical_overlap") is not False
    ):
        raise RuntimeError("NF-UNSW restart overlap classification is invalid")
    internal = overlap.get("internal_split_overlap_counts", {})
    if not internal or any(
        count != 0
        for comparisons in internal.values()
        for count in comparisons.values()
    ):
        raise RuntimeError("NF-UNSW restart has internal split overlap")
    history = overlap.get("history_identity_overlap_counts", {})
    if not history or not any(count > 0 for count in history.values()):
        raise RuntimeError("NF-UNSW restart historical reuse is not evidenced")
    return {
        "claim_scope": _RESTART_CLAIM_SCOPE,
        "historically_independent": False,
        "eligible_for_primary_cross_dataset_inference": False,
        "historical_exposure_policy": _RESTART_HISTORY_POLICY,
        "verified_zero_internal_overlap": True,
        "verified_zero_historical_overlap": False,
        "history_identity_overlap_counts": history,
    }


def verify_restart_test_access_recovery(receipt):
    """Require the documented NF-UNSW pre-materialization recovery counts."""

    expected = {
        "sealed_test_claim_count": 1,
        "sealed_artifact_hash_read_attempt_count": 3,
        "sealed_artifact_hash_read_counting_unit": (
            "one_complete_records_manifest_features_sha256_pass"
        ),
        "pre_materialization_failure_count": 1,
        "diagnostic_hash_verification_count": 1,
        "successful_test_materialization_count": 1,
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise RuntimeError("NF-UNSW sealed-test recovery accounting is invalid")
    recovery = receipt.get("pre_materialization_recovery")
    if not isinstance(recovery, dict) or set(recovery) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError("NF-UNSW sealed-test recovery event is missing")
    return {**expected, "recovery_event": recovery}


def _verify_restart(root, audit_root):
    summary_path, summary, evidence = _verify_formal_matrix(root, _RESTART_KEY)
    receipt_path = _verify_artifact(
        summary["sealed_test_access_receipt"],
        "NF-UNSW restart sealed-test receipt",
    )
    receipt = _read_json(receipt_path)
    recovery = verify_restart_test_access_recovery(receipt)
    recovery_event_path = _verify_artifact(
        recovery["recovery_event"], "NF-UNSW restart recovery event"
    )
    recovery_event = _read_json(recovery_event_path)
    if (
        recovery_event.get("protocol_version") != _RECOVERY_PROTOCOL_VERSION
        or recovery_event.get("dataset") != _RESTART_KEY
        or recovery_event.get("status")
        != "explicit_pre_materialization_recovery_claimed"
        or recovery_event.get("automatic_retry_allowed") is not False
        or recovery_event.get("sealed_results_present_before_recovery") is not False
    ):
        raise RuntimeError("NF-UNSW restart recovery event is invalid")
    original_claim_path = _verify_artifact(
        recovery_event["original_claim"], "NF-UNSW restart original claim"
    )
    global_lock_path = _verify_artifact(
        recovery_event["global_selection_lock"],
        "NF-UNSW restart recovery global selection lock",
    )
    if (
        original_claim_path != Path(root) / "sealed_test_access.claim.json"
        or global_lock_path
        != Path(root) / "all_validation_selections.lock.json"
    ):
        raise RuntimeError("NF-UNSW restart recovery references changed")
    incident_path = _verify_artifact(
        recovery_event["incident"], "NF-UNSW restart recovery incident"
    )
    incident = _read_json(incident_path)
    if (
        incident.get("protocol_version") != _RECOVERY_PROTOCOL_VERSION
        or incident.get("status") != "authorized_for_single_explicit_recovery"
        or incident.get("dataset") != _RESTART_KEY
        or incident.get("historically_independent") is not False
        or incident.get("eligible_for_primary_cross_dataset_inference") is not False
        or incident.get("access_accounting_before_recovery")
        != recovery_event.get("access_accounting_before_recovery")
        or incident.get("authorized_recovery")
        != recovery_event.get("authorized_recovery")
    ):
        raise RuntimeError("NF-UNSW restart recovery incident is invalid")
    sealed_hashes = receipt.get("sealed_artifact_hashes", {})
    if any(
        incident.get("sealed_artifacts", {}).get(name, {}).get("sha256")
        != sealed_hashes.get(name)
        for name in ("records", "manifest", "features")
    ):
        raise RuntimeError("NF-UNSW restart recovery sealed hashes changed")
    recovery_disclosure = {
        **{name: recovery[name] for name in recovery if name != "recovery_event"},
        "failure_stage": incident["failure"]["stage"],
        "manifest_parsed_during_failed_attempt": incident["failure"][
            "manifest_parsed"
        ],
        "records_materialized_during_failed_attempt": incident["failure"][
            "records_materialized"
        ],
        "predictions_generated_during_failed_attempt": incident["failure"][
            "predictions_generated"
        ],
        "automatic_retry_allowed": False,
        "recovery_event_path": recovery_event_path,
        "incident_path": incident_path,
    }
    plan_path = Path(root) / "matrix_plan.json"
    feasibility_path = Path(audit_root) / "feasibility.json"
    plan = _read_json(plan_path)
    feasibility = _read_json(feasibility_path)
    overlap_record = plan["artifacts"]["cohort_overlap_audit"]
    overlap_path = _verify_artifact(
        overlap_record, "NF-UNSW restart cohort overlap audit"
    )
    locked_feasibility_path = _verify_artifact(
        plan["artifacts"]["feasibility"], "NF-UNSW restart feasibility"
    )
    if locked_feasibility_path != feasibility_path:
        raise RuntimeError("NF-UNSW restart feasibility artifact changed")
    source_checkpoint = _verify_artifact(
        plan["artifacts"]["source_checkpoint"],
        "NF-UNSW restart source checkpoint",
    )
    source_lock_path = _verify_artifact(
        plan["artifacts"]["source_checkpoint_lock"],
        "NF-UNSW restart source checkpoint lock",
    )
    source_lock = _read_json(source_lock_path)
    if (
        source_lock.get("status")
        != "target_excluded_source_checkpoint_locked"
        or source_lock.get("source_dataset_id")
        != "NF-CSE-CIC-IDS2018-v2"
        or source_lock.get("target_dataset_id") != _RESTART_DATASET
        or source_lock.get("target_excluded") is not True
        or source_lock.get("source_test_files_read") is not False
        or source_lock.get("checkpoint", {}).get("path")
        != str(source_checkpoint)
        or source_lock.get("checkpoint", {}).get("sha256")
        != _sha256_file(source_checkpoint)
    ):
        raise RuntimeError("NF-UNSW restart source checkpoint lock is invalid")
    disclosure = verify_restart_disclosure(
        plan, feasibility, _read_json(overlap_path)
    )
    return {
        "summary_path": summary_path,
        "summary": summary,
        "evidence": evidence,
        "plan_path": plan_path,
        "feasibility_path": feasibility_path,
        "overlap_path": overlap_path,
        "source_checkpoint": source_checkpoint,
        "source_lock_path": source_lock_path,
        "disclosure": disclosure,
        "test_access_recovery": recovery_disclosure,
    }


def _distribution(values):
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise RuntimeError("deployment efficiency values must be finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def deployment_rare_zero_failures(metrics, setting_id):
    failures = []
    for condition in RANDOM_CONDITIONS:
        values = metrics["conditions"][condition]["template_tree"]
        for label in _RESTART_RARE_LABELS:
            recall = float(values["per_class_recall"][label])
            support = int(values["class_counts"][label])
            if support > 0 and recall <= 0.0:
                failures.append(
                    {
                        "setting_id": setting_id,
                        "condition": condition,
                        "component": "template_tree",
                        "label": label,
                        "recall": recall,
                        "support": support,
                        "type": "validation_rare_class_zero_recall",
                    }
                )
    return failures


def _verify_deployment_baseline(root):
    root = Path(root)
    if (root / ".incomplete").exists():
        raise RuntimeError("NF-UNSW deployment baseline is incomplete")
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("status")
        != "all_nine_validation_only_cpu_settings_completed"
        or summary.get("protocol_version")
        != "formal-template-tree-deployment-v2"
        or summary.get("dataset_key") != _RESTART_KEY
        or summary.get("setting_count") != 9
        or tuple(summary.get("conditions", ())) != FORMAL_CONDITIONS
        or tuple(summary.get("components", ()))
        != ("template_only", "shallow_tree", "template_tree")
        or summary.get("sealed_test_files_read") is not False
        or summary.get("test_samples_materialized") is not False
        or summary.get("historically_independent") is not False
    ):
        raise RuntimeError("NF-UNSW deployment baseline contract is invalid")
    selection = summary.get("development_selection", {})
    if (
        selection.get("status")
        != "frozen_after_bounded_validation_diagnostic"
        or selection.get("selection_scope")
        != "train_01pct_seed17_clean_validation_only"
        or selection.get("selected_profile") != "all_d8_l128"
        or selection.get("candidate_count") != 4
        or selection.get("sealed_test_files_read") is not False
        or selection.get("test_samples_materialized") is not False
    ):
        raise RuntimeError("deployment development selection is invalid")
    expected = {
        (percent, seed)
        for percent in _TRAIN_PERCENTS
        for seed in _MODEL_SEEDS
    }
    observed = set()
    per_class_values = {}
    rare_zero_failures = []
    for setting in summary["settings"]:
        current_setting = (setting["train_percent"], setting["model_seed"])
        observed.add(current_setting)
        for name in ("model", "metrics", "efficiency"):
            _verify_artifact(setting[name], f"deployment {name}")
        metrics = _read_json(setting["metrics"]["path"])
        if (
            metrics.get("evaluation_scope") != "validation_only"
            or metrics.get("sealed_test_files_read") is not False
            or tuple(metrics.get("components", ()))
            != ("template_only", "shallow_tree", "template_tree")
            or set(metrics.get("conditions", {})) != set(FORMAL_CONDITIONS)
        ):
            raise RuntimeError("deployment validation metrics are invalid")
        setting_id = (
            f"pct-{current_setting[0]:02d}-seed-{current_setting[1]}"
        )
        rare_zero_failures.extend(
            deployment_rare_zero_failures(metrics, setting_id)
        )
        for condition, components in metrics["conditions"].items():
            for component, values in components.items():
                for label, recall in values["per_class_recall"].items():
                    per_class_values.setdefault(component, {}).setdefault(
                        condition, {}
                    ).setdefault(label, []).append(float(recall))
    if observed != expected:
        raise RuntimeError("deployment settings are incomplete")
    _assert_finite_tree(summary, "deployment")
    return {
        "status": "validation_only_cpu_baseline_completed",
        "eligible_for_sealed_test_claim": False,
        "components": summary["components"],
        "aggregates": summary["aggregates"],
        "fit_seconds": _distribution(
            setting["fit_seconds"] for setting in summary["settings"]
        ),
        "model_artifact_bytes": _distribution(
            setting["model"]["bytes"] for setting in summary["settings"]
        ),
        "per_class_recall_aggregates": {
            component: {
                condition: {
                    label: _distribution(values)
                    for label, values in by_label.items()
                }
                for condition, by_label in by_condition.items()
            }
            for component, by_condition in per_class_values.items()
        },
        "random_condition_rare_zero_recall_failures": rare_zero_failures,
        "rare_class_safety_claim_allowed": not rare_zero_failures,
        "source_artifact": _artifact(summary_path),
    }


def _mean(values):
    return float(np.mean(tuple(values)))


def _calibration_checks(calibration):
    required = (
        "mean_nll_improved",
        "mean_brier_improved",
        "mean_ece_not_worse",
    )
    if any(key not in values for values in calibration.values() for key in required):
        raise RuntimeError("a formal calibration chain is incomplete")
    return {
        "complete_across_all_independent_datasets": True,
        "mean_nll_improved": all(
            values["mean_nll_improved"] for values in calibration.values()
        ),
        "mean_brier_improved": all(
            values["mean_brier_improved"] for values in calibration.values()
        ),
        "mean_ece_not_worse": all(
            values["mean_ece_not_worse"] for values in calibration.values()
        ),
    }


def summarize(output_path, *, bootstrap_resamples=20_000):
    output = validate_output_path(
        output_path, enforce_autodl_data_disk=True, data_root=_DATA_ROOT
    )
    if output.exists():
        raise ValueError("joint audit output already exists")

    summaries = {}
    evidence = {}
    summary_paths = {}
    dataset_keys = {
        "NF-BoT-IoT-v2": "nf_bot",
        "NF-CSE-CIC-IDS2018-v2": "nf_cse",
        "NF-ToN-IoT-v2": "nf_ton",
    }
    for dataset_id, root in _MATRIX_ROOTS.items():
        path, summary, current = _verify_formal_matrix(
            root, dataset_keys[dataset_id]
        )
        summary_paths[dataset_id] = path
        summaries[dataset_id] = summary
        evidence[dataset_id] = current

    restart = _verify_restart(_RESTART_ROOT, _RESTART_AUDIT_ROOT)
    deployment = _verify_deployment_baseline(_DEPLOYMENT_ROOT)
    independent_gains = {
        dataset: tuple(current["setting_gains"].values())
        for dataset, current in evidence.items()
    }
    bootstrap = dataset_stratified_paired_bootstrap(
        independent_gains, resamples=bootstrap_resamples
    )
    direction = {
        dataset: _mean(gains) >= -1e-12
        for dataset, gains in independent_gains.items()
    }
    missing_means = {
        dataset: {
            condition: current["condition_deltas"][condition][
                "a0_mr_minus_fixed_a0"
            ]
            for condition in RANDOM_CONDITIONS[1:]
        }
        for dataset, current in evidence.items()
    }
    missing_wins = {
        condition: sum(
            current[condition] > 0.0 for current in missing_means.values()
        )
        for condition in RANDOM_CONDITIONS[1:]
    }
    rare_failures = {
        dataset: current["rare_collapse_failures"]
        for dataset, current in evidence.items()
    }
    calibration = {
        dataset: summaries[dataset]["a0_mr_calibration"]
        for dataset in _INDEPENDENT_DATASETS
    }
    calibration_checks = _calibration_checks(calibration)
    comparisons = {
        dataset: _comparison(
            evidence[dataset],
            _macro_f1_means(summaries[dataset]),
            lightgbm_available=True,
        )
        for dataset in _INDEPENDENT_DATASETS
    }
    seed_stability = {
        dataset: _seed_stability(evidence[dataset]["setting_gains"])
        for dataset in _INDEPENDENT_DATASETS
    }
    checks = {
        "at_least_three_datasets_nonnegative_vs_xgboost": (
            sum(direction.values()) >= 3
        ),
        "cross_dataset_mean_gain_at_least_0p005": (
            bootstrap["observed_mean_gain"] >= 0.005
        ),
        "dataset_stratified_bootstrap_ci95_lower_above_zero": (
            bootstrap["ci95_lower"] > 0.0
        ),
        "missing_30_better_than_fixed_a0_on_at_least_three": (
            missing_wins["missing_30pct"] >= 3
        ),
        "missing_50_better_than_fixed_a0_on_at_least_three": (
            missing_wins["missing_50pct"] >= 3
        ),
        "no_rare_class_recall_collapse": not any(rare_failures.values()),
        "calibrated_mean_nll_and_brier_improve_ece_not_worse": all(
            calibration_checks.values()
        ),
    }
    restart_evidence = restart["evidence"]
    restart_summary = restart["summary"]
    restart_gains = tuple(restart_evidence["setting_gains"].values())
    restart_result = {
        **restart["disclosure"],
        "mean_random_condition_gain_vs_xgboost": _mean(restart_gains),
        "setting_gains": restart_evidence["setting_gains"],
        "comparison": _comparison(
            restart_evidence,
            _macro_f1_means(restart_summary),
            lightgbm_available=True,
        ),
        "model_seed_stability": _seed_stability(
            restart_evidence["setting_gains"]
        ),
        "rare_failures": restart_evidence["rare_collapse_failures"],
        "calibration": restart_summary["a0_mr_calibration"],
        "test_access_recovery": {
            name: value
            for name, value in restart["test_access_recovery"].items()
            if name not in {"recovery_event_path", "incident_path"}
        },
        "deployment_baseline": deployment,
    }
    payload = {
        "format_version": 2,
        "protocol_version": "cross-dataset-a0-mr-joint-audit-v2",
        "status": "three_independent_datasets_plus_disclosed_restart_completed",
        "dataset_roles": {
            **{
                dataset: "historically_independent_formal_matrix"
                for dataset in _INDEPENDENT_DATASETS
            },
            _RESTART_DATASET: (
                "full_source_restart_not_historically_independent"
            ),
        },
        "independent_evaluable_dataset_count": 3,
        "disclosed_restart_replication_count": 1,
        "primary_inference_excludes_restart": True,
        "figure_3_display_policy": {
            "display_all_four_results": True,
            "mark_nf_unsw_as_disclosed_restart": True,
            "cross_dataset_aggregate_uses_only_independent_datasets": list(
                _INDEPENDENT_DATASETS
            ),
        },
        "thresholds": _UPGRADE_THRESHOLDS,
        "mean_random_gain_by_independent_dataset": bootstrap[
            "observed_by_dataset"
        ],
        "nonnegative_direction_by_independent_dataset": direction,
        "comparison_by_independent_dataset": comparisons,
        "model_seed_stability_by_independent_dataset": seed_stability,
        "missing_deltas_vs_fixed_a0_by_independent_dataset": missing_means,
        "missing_win_independent_dataset_counts": missing_wins,
        "dataset_stratified_paired_bootstrap": bootstrap,
        "rare_failures_by_independent_dataset": rare_failures,
        "calibration_by_independent_dataset": calibration,
        "calibration_checks": calibration_checks,
        "primary_upgrade_checks": checks,
        "a0_mr_cross_dataset_upgrade_passed": all(checks.values()),
        "upgrade_gate_blockers": [
            name for name, passed in checks.items() if not passed
        ],
        "universal_missing_robust_framework_claim_allowed": all(
            checks.values()
        ),
        "nf_unsw_restart_replication": restart_result,
        "source_artifacts": {
            "independent_formal_summaries": {
                dataset: _artifact(path)
                for dataset, path in summary_paths.items()
            },
            "nf_unsw_restart_summary": _artifact(restart["summary_path"]),
            "nf_unsw_restart_plan": _artifact(restart["plan_path"]),
            "nf_unsw_restart_feasibility": _artifact(
                restart["feasibility_path"]
            ),
            "nf_unsw_restart_overlap": _artifact(restart["overlap_path"]),
            "nf_unsw_restart_source_checkpoint": _artifact(
                restart["source_checkpoint"]
            ),
            "nf_unsw_restart_source_checkpoint_lock": _artifact(
                restart["source_lock_path"]
            ),
            "nf_unsw_restart_test_access_recovery_event": _artifact(
                restart["test_access_recovery"]["recovery_event_path"]
            ),
            "nf_unsw_restart_test_access_incident": _artifact(
                restart["test_access_recovery"]["incident_path"]
            ),
        },
    }
    _assert_finite_tree(payload)
    output.mkdir(parents=True)
    marker = output / ".incomplete"
    marker.write_text("joint audit incomplete\n", encoding="ascii")
    write_json_snapshot(output / "joint_audit.json", payload)
    marker.unlink()
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit independent formal matrices and disclosed UNSW restart"
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    result = summarize(
        arguments.output, bootstrap_resamples=arguments.bootstrap_resamples
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
