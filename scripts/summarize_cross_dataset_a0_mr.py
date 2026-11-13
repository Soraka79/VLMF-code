#!/usr/bin/env python3
"""Apply the predeclared cross-dataset A0-MR upgrade gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flowfm.data.formal_cohort import get_formal_dataset_spec
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from scripts.run_nf_cse_a0_mr_confirmation import _sha256_file


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_CSE_SUMMARY = _RESULTS_ROOT / "nf-cse-formal-matrix-v2/formal_matrix_summary.json"
_BOT_ROOT = _RESULTS_ROOT / "cross-dataset-formal-matrix-v1/nf_bot"
_TON_ROOT = _RESULTS_ROOT / "cross-dataset-formal-matrix-v3/nf_ton"
_AUDIT_ROOT = _RESULTS_ROOT / "cross-dataset-formal-audits-v1"
_DEFAULT_OUTPUT = _RESULTS_ROOT / "cross-dataset-a0-mr-summary-v1"
_RANDOM_CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
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


def _mean(values):
    return float(np.mean(tuple(values)))


def _predeclared_rare_labels(summary):
    return tuple(
        get_formal_dataset_spec(summary["dataset"]).rare_coarse_labels
    )


def _verified_cse_evidence(summary):
    values = {}
    condition_deltas = {}
    retention = {}
    rare_recall = {}
    rare_collapse_failures = []
    for artifact in summary["result_artifacts"]:
        path = Path(artifact["metrics_path"])
        if _sha256_file(path) != artifact["metrics_sha256"]:
            raise RuntimeError("locked NF-CSE metrics artifact changed")
        metrics = _read_json(path)
        random_deltas = tuple(
            metrics["macro_f1_deltas"][condition]["a0_mr_minus_xgboost"]
            for condition in _RANDOM_CONDITIONS
        )
        values[artifact["setting_id"]] = _mean(random_deltas)
        for condition, deltas in metrics["macro_f1_deltas"].items():
            target = condition_deltas.setdefault(condition, {})
            for name, value in deltas.items():
                target.setdefault(name, []).append(float(value))
        for model, by_condition in metrics["retention"].items():
            for condition, value in by_condition.items():
                retention.setdefault(model, {}).setdefault(condition, []).append(
                    float(value)
                )
        for condition in metrics["a0_mr"]:
            for model, values_by_condition in (
                ("a0_mr", metrics["a0_mr"]),
                ("xgboost", metrics["base_models"]),
            ):
                current = (
                    values_by_condition[condition]
                    if model == "a0_mr"
                    else values_by_condition[condition]["xgboost"]
                )
                for label in metrics["rare_class_analysis"]["rare_labels"]:
                    rare_recall.setdefault(condition, {}).setdefault(
                        model, {}
                    ).setdefault(label, []).append(
                        float(current["per_class_recall"][label])
                    )
        random_rare_deltas = [
            row["mean_recall_delta_vs_xgboost"]
            for row in metrics["rare_class_analysis"]["conditions"]
            if row["condition"] in _RANDOM_CONDITIONS
        ]
        random_rare_mean = _mean(random_rare_deltas)
        if random_rare_mean < -0.02:
            rare_collapse_failures.append(
                {
                    "setting_id": artifact["setting_id"],
                    "type": "rare_random_mean_recall_drop",
                    "mean_delta_vs_xgboost": random_rare_mean,
                    "maximum_allowed_drop": 0.02,
                }
            )
        rare_collapse_failures.extend(
            {
                "setting_id": artifact["setting_id"],
                "type": "rare_recall_zero_out",
                **failure,
            }
            for failure in metrics["rare_class_analysis"]["zero_out_failures"]
        )
    if len(values) != 9:
        raise RuntimeError("locked NF-CSE formal matrix must contain nine settings")
    return {
        "setting_gains": values,
        "condition_deltas": {
            condition: {
                name: _mean(current)
                for name, current in by_name.items()
            }
            for condition, by_name in condition_deltas.items()
        },
        "retention": {
            model: {
                condition: _mean(current)
                for condition, current in by_condition.items()
            }
            for model, by_condition in retention.items()
        },
        "rare_recall": {
            condition: {
                model: {
                    label: _mean(current)
                    for label, current in by_label.items()
                }
                for model, by_label in by_model.items()
            }
            for condition, by_model in rare_recall.items()
        },
        "rare_collapse_failures": rare_collapse_failures,
    }


def _new_dataset_evidence(summary):
    for bundle in summary["result_artifacts"]:
        for name in ("metrics", "bootstrap", "efficiency", "prediction_lock"):
            artifact = bundle[name]
            path = Path(artifact["path"])
            if (
                not path.is_file()
                or path.stat().st_size != artifact["bytes"]
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise RuntimeError(
                    f"locked {summary['dataset']} {name} artifact changed"
                )
    by_setting = summary["random_condition_mean_a0_mr_minus_xgboost"][
        "by_setting"
    ]
    values = {
        setting: float(value) for setting, value in sorted(by_setting.items())
    }
    if len(values) != 9:
        raise RuntimeError("new formal matrix must contain nine setting gains")
    rare_labels = _predeclared_rare_labels(summary)
    return {
        "setting_gains": values,
        "condition_deltas": {
            condition: {
                name: distribution["mean"]
                for name, distribution in by_name.items()
            }
            for condition, by_name in summary["macro_f1_deltas"].items()
        },
        "retention": {
            model: {
                condition: distribution["mean"]
                for condition, distribution in by_condition.items()
            }
            for model, by_condition in summary["retention_aggregates"].items()
        },
        "rare_recall": {
            condition: {
                model: {
                    label: distribution["mean"]
                    for label, distribution in by_label.items()
                    if label in rare_labels
                }
                for model, by_label in by_model.items()
                if model in {"a0_mr", "xgboost"}
            }
            for condition, by_model in summary[
                "per_class_recall_aggregates"
            ].items()
        },
        "rare_collapse_failures": summary["rare_collapse_failures"],
    }


def _macro_f1_means(summary, *, cse=False):
    aggregates = (
        summary["aggregates"]
        if cse
        else summary["aggregates"]["uncalibrated"]
    )
    return {
        condition: {
            model: values["macro_f1"]["mean"]
            for model, values in by_model.items()
        }
        for condition, by_model in aggregates.items()
    }


def _comparison(evidence, macro_f1, *, lightgbm_available):
    baselines = ("xgboost", "fixed_a0") + (
        ("lightgbm",) if lightgbm_available else ()
    )
    by_condition = {}
    for condition, scores in macro_f1.items():
        by_condition[condition] = {
            "macro_f1": scores,
            "a0_mr_deltas": {
                baseline: float(scores["a0_mr"] - scores[baseline])
                for baseline in baselines
            },
        }
        if not lightgbm_available:
            by_condition[condition]["a0_mr_deltas"]["lightgbm"] = None
    random_means = {
        model: _mean(macro_f1[condition][model] for condition in _RANDOM_CONDITIONS)
        for model in macro_f1["clean"]
    }
    random_deltas = {
        baseline: float(random_means["a0_mr"] - random_means[baseline])
        for baseline in baselines
    }
    if not lightgbm_available:
        random_deltas["lightgbm"] = None
    return {
        "by_condition": by_condition,
        "random_condition_mean_macro_f1": random_means,
        "random_condition_mean_a0_mr_deltas": random_deltas,
        "random_conditions_noninferior": {
            baseline: all(
                by_condition[condition]["a0_mr_deltas"][baseline] >= -1e-12
                for condition in _RANDOM_CONDITIONS
            )
            for baseline in baselines
        }
        | ({"lightgbm": None} if not lightgbm_available else {}),
        "structured_conditions_noninferior": {
            baseline: all(
                values["a0_mr_deltas"][baseline] >= -1e-12
                for condition, values in by_condition.items()
                if condition not in _RANDOM_CONDITIONS
            )
            for baseline in baselines
        }
        | ({"lightgbm": None} if not lightgbm_available else {}),
        "retention": evidence["retention"],
        "rare_recall": evidence["rare_recall"],
    }


def _seed_stability(setting_gains, *, cse=False):
    grouped = {}
    for setting_id, gain in setting_gains.items():
        parts = setting_id.split("-")
        if cse:
            seed, percent = int(parts[1]), int(parts[3])
        else:
            percent, seed = int(parts[1]), int(parts[3])
        grouped.setdefault(str(percent), {})[str(seed)] = float(gain)
    result = {}
    for percent, gains in sorted(grouped.items(), key=lambda item: int(item[0])):
        if set(gains) != {"17", "42", "20260725"}:
            raise RuntimeError("formal setting seeds are incomplete")
        nonnegative = sum(value >= -1e-12 for value in gains.values())
        result[percent] = {
            "gains": gains,
            "nonnegative_count": nonnegative,
            "at_least_two_of_three_direction_consistent": nonnegative >= 2,
        }
    return result


def dataset_stratified_paired_bootstrap(
    gains_by_dataset,
    *,
    resamples=20_000,
    seed=20260730,
):
    """Resample paired setting gains within each equally weighted dataset."""

    if type(resamples) is not int or resamples < 2000:
        raise ValueError("cross-dataset bootstrap requires at least 2000 resamples")
    names = tuple(sorted(gains_by_dataset))
    if len(names) < 2:
        raise ValueError("cross-dataset bootstrap requires at least two datasets")
    arrays = {}
    for name in names:
        values = np.asarray(gains_by_dataset[name], dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("each dataset requires finite paired setting gains")
        arrays[name] = values
    observed_by_dataset = {
        name: float(values.mean()) for name, values in arrays.items()
    }
    observed = float(np.mean(tuple(observed_by_dataset.values())))
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        strata = [
            float(values[rng.integers(0, len(values), size=len(values))].mean())
            for values in arrays.values()
        ]
        samples[index] = float(np.mean(strata))
    return {
        "format_version": 1,
        "method": "dataset_stratified_paired_setting_bootstrap",
        "pairing_unit": "same_setting_a0_mr_minus_xgboost_macro_f1",
        "dataset_weighting": "equal",
        "within_dataset_resampling": "settings_with_replacement",
        "resamples": resamples,
        "seed": seed,
        "dataset_count": len(names),
        "observed_mean_gain": observed,
        "observed_by_dataset": observed_by_dataset,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def summarize(output_path, *, bootstrap_resamples=20_000):
    output = validate_output_path(
        output_path, enforce_autodl_data_disk=True, data_root=_DATA_ROOT
    )
    if output.exists():
        raise ValueError("cross-dataset summary output already exists")
    cse = _read_json(_CSE_SUMMARY)
    bot_path = _BOT_ROOT / "formal_matrix_summary.json"
    ton_path = _TON_ROOT / "formal_matrix_summary.json"
    bot = _read_json(bot_path)
    ton = _read_json(ton_path)
    unsw_path = _AUDIT_ROOT / "nf_unsw/feasibility.json"
    unsw = _read_json(unsw_path)
    if (
        cse.get("status") != "formal_matrix_completed"
        or bot.get("status") != "formal_matrix_completed"
        or ton.get("status") != "formal_matrix_completed"
        or unsw.get("status") != "not_evaluable_required_class_exhausted"
    ):
        raise RuntimeError("all four dataset closures are not complete")
    evidence = {
        "NF-CSE-CIC-IDS2018-v2": _verified_cse_evidence(cse),
        "NF-BoT-IoT-v2": _new_dataset_evidence(bot),
        "NF-ToN-IoT-v2": _new_dataset_evidence(ton),
    }
    gains = {
        name: tuple(values["setting_gains"].values())
        for name, values in evidence.items()
    }
    bootstrap = dataset_stratified_paired_bootstrap(
        gains, resamples=bootstrap_resamples
    )
    direction = {
        name: float(np.mean(values)) >= -1e-12
        for name, values in gains.items()
    }
    missing_means = {
        name: {
            condition: values["condition_deltas"][condition][
                "a0_mr_minus_fixed_a0"
            ]
            for condition in _RANDOM_CONDITIONS[1:]
        }
        for name, values in evidence.items()
    }
    missing_wins = {
        condition: sum(
            values[condition] > 0.0 for values in missing_means.values()
        )
        for condition in _RANDOM_CONDITIONS[1:]
    }
    rare_failures = {
        name: values["rare_collapse_failures"]
        for name, values in evidence.items()
    }
    calibration = {
        "NF-CSE-CIC-IDS2018-v2": {
            "status": "not_evaluable_locked_formal_matrix_has_no_temperature_test_metrics",
            "sealed_test_reopen_forbidden": True,
        },
        "NF-BoT-IoT-v2": bot["a0_mr_calibration"],
        "NF-ToN-IoT-v2": ton["a0_mr_calibration"],
    }
    calibration_complete = all(
        "mean_nll_improved" in values for values in calibration.values()
    )
    calibration_checks = {
        "complete_across_all_evaluable_datasets": calibration_complete,
        "mean_nll_improved": calibration_complete
        and all(values["mean_nll_improved"] for values in calibration.values()),
        "mean_brier_improved": calibration_complete
        and all(values["mean_brier_improved"] for values in calibration.values()),
        "mean_ece_not_worse": calibration_complete
        and all(values["mean_ece_not_worse"] for values in calibration.values()),
    }
    comparisons = {
        "NF-CSE-CIC-IDS2018-v2": _comparison(
            evidence["NF-CSE-CIC-IDS2018-v2"],
            _macro_f1_means(cse, cse=True),
            lightgbm_available=False,
        ),
        "NF-BoT-IoT-v2": _comparison(
            evidence["NF-BoT-IoT-v2"],
            _macro_f1_means(bot),
            lightgbm_available=True,
        ),
        "NF-ToN-IoT-v2": _comparison(
            evidence["NF-ToN-IoT-v2"],
            _macro_f1_means(ton),
            lightgbm_available=True,
        ),
    }
    seed_stability = {
        name: _seed_stability(
            values["setting_gains"],
            cse=(name == "NF-CSE-CIC-IDS2018-v2"),
        )
        for name, values in evidence.items()
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
    payload = {
        "format_version": 1,
        "protocol_version": "cross-four-dataset-a0-mr-upgrade-gate-v1",
        "status": "four_dataset_closure_completed",
        "dataset_closure": {
            "NF-CSE-CIC-IDS2018-v2": "formal_completed_reused_without_test_reopen",
            "NF-BoT-IoT-v2": "formal_completed",
            "NF-UNSW-NB15-v2": "not_evaluable_record_independent_attacks_exhausted",
            "NF-ToN-IoT-v2": "formal_completed_shared_coarse_five_class",
        },
        "evaluable_dataset_count": 3,
        "not_evaluable_dataset_count": 1,
        "thresholds": _UPGRADE_THRESHOLDS,
        "mean_random_gain_by_dataset": bootstrap["observed_by_dataset"],
        "nonnegative_direction_by_dataset": direction,
        "comparison_by_dataset": comparisons,
        "model_seed_stability_by_dataset": seed_stability,
        "missing_deltas_vs_fixed_a0_by_dataset": missing_means,
        "missing_win_dataset_counts": missing_wins,
        "dataset_stratified_paired_bootstrap": bootstrap,
        "rare_failures": rare_failures,
        "calibration": calibration,
        "calibration_checks": calibration_checks,
        "checks": checks,
        "a0_mr_cross_dataset_upgrade_passed": all(checks.values()),
        "upgrade_gate_blockers": [
            name for name, passed in checks.items() if not passed
        ],
        "paper_draft_dataset_closure_prerequisite_met": True,
        "q1_cross_dataset_main_claim_experimental_basis": all(checks.values()),
        "universal_missing_robust_framework_claim_allowed": all(checks.values()),
        "efficiency": {
            "NF-CSE-CIC-IDS2018-v2": {
                "status": "locked_existing_summary_only",
                "total_recorded_job_wall_seconds": cse.get(
                    "total_recorded_job_wall_seconds"
                ),
                "prediction_archive_total_bytes": cse.get(
                    "prediction_archive_total_bytes"
                ),
            },
            "NF-BoT-IoT-v2": bot["efficiency"],
            "NF-ToN-IoT-v2": ton["efficiency"],
        },
        "source_artifacts": {
            "nf_cse_summary": {
                "path": str(_CSE_SUMMARY),
                "sha256": _sha256_file(_CSE_SUMMARY),
            },
            "nf_bot_summary": {
                "path": str(bot_path),
                "sha256": _sha256_file(bot_path),
            },
            "nf_unsw_feasibility": {
                "path": str(unsw_path),
                "sha256": _sha256_file(unsw_path),
            },
            "nf_ton_summary": {
                "path": str(ton_path),
                "sha256": _sha256_file(ton_path),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    write_json_snapshot(output / "cross_dataset_summary.json", payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply the frozen cross-dataset A0-MR upgrade gate"
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
