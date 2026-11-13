#!/usr/bin/env python3
"""Freeze, but do not execute, the learned-ensemble comparison protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from flowfm.experiment.metadata import write_json_snapshot


EXPERIMENT_NAME = "learned_ensemble_protocol_v1"
PROTOCOL_VERSION = "learned-ensemble-comparators-v1"
DATASETS = (
    "NF-BoT-IoT-v2",
    "NF-CSE-CIC-IDS2018-v2",
    "NF-ToN-IoT-v2-compatible-shared-coarse",
)
CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
TRAIN_PERCENTS = (1, 5, 10)
MODEL_SEEDS = (17, 42, 20260725)
META_SEED = 20260731
C_GRID = (0.01, 0.1, 1.0, 10.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def protocol_payload() -> dict[str, object]:
    """Return the complete pre-test protocol for the deferred third group."""

    return {
        "format_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": "protocol_locked_execution_not_started",
        "purpose": (
            "deferred fair comparison with logistic stacking and a low-capacity "
            "sample-wise learned gate"
        ),
        "scope": {
            "datasets": list(DATASETS),
            "train_percents": list(TRAIN_PERCENTS),
            "model_seeds": list(MODEL_SEEDS),
            "setting_count": 27,
            "conditions": list(CONDITIONS),
            "base_predictors": ["xgboost", "schemaflow_pretrained"],
            "base_models_retrained": False,
            "probability_calibration": "uncalibrated",
            "structured_missingness_included": False,
        },
        "authority_boundary": {
            "protocol_locked_before_validation_model_fitting": True,
            "all_hyperparameter_authority": "validation_only",
            "formal_test_used_for_model_fitting_or_selection": False,
            "formal_test_used_once_after_model_parameter_lock": True,
            "test_results_may_change_protocol_or_candidate_grid": False,
            "required_input_verification": (
                "existing validation registry and sealed prediction SHA-256 locks"
            ),
        },
        "meta_validation": {
            "unit": "validation group_id",
            "splitter": "StratifiedGroupKFold",
            "folds": 5,
            "shuffle": True,
            "seed": META_SEED,
            "stratification_target": "clean validation label",
            "group_exclusivity": (
                "all rows for one group_id and all three conditions remain in "
                "the same fold"
            ),
            "condition_rows": (
                "clean, missing_30pct, and missing_50pct rows are pooled with "
                "equal multiplicity inside each fold"
            ),
            "preprocessing_fit_scope": "meta-training folds only",
            "candidate_scoring": {
                "primary": "equal_mean_condition_macro_f1_on_oof_predictions",
                "tie_break": [
                    "equal_mean_condition_mcc",
                    "equal_mean_condition_balanced_accuracy",
                    "smaller_C_stronger_regularization",
                ],
                "float_comparison": "exact_binary64_lexicographic",
            },
            "post_selection_refit": (
                "refit the selected candidate on all validation groups, then "
                "serialize parameters and hashes before any test read"
            ),
            "fold_failure_policy": (
                "stop the setting; do not alter folds, reduce fold count, or "
                "substitute a test-informed fallback"
            ),
        },
        "comparators": {
            "logistic_stacking": {
                "selection_scope": "one global stacker per setting",
                "input_features": (
                    "concatenated per-class log probabilities from XGBoost and "
                    "pretrained SchemaFlow, clipped to [1e-12, 1]"
                ),
                "condition_indicator_used": False,
                "preprocessing": "StandardScaler fitted within meta-training data",
                "estimator": "multinomial L2 LogisticRegression",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "fit_intercept": True,
                "max_iter": 2000,
                "tolerance": 1e-8,
                "C_grid": list(C_GRID),
                "random_state": META_SEED,
                "output": "multiclass predict_proba",
            },
            "learned_gate": {
                "selection_scope": "one dynamic gate per setting",
                "tree_weight": "sigmoid(w^T z + b)",
                "pool": "linear",
                "features": [
                    "three-way declared-condition one-hot",
                    "xgboost maximum probability",
                    "schemaflow maximum probability",
                    "xgboost top1-minus-top2 margin",
                    "schemaflow top1-minus-top2 margin",
                    "xgboost normalized entropy",
                    "schemaflow normalized entropy",
                    "Jensen-Shannon divergence between branch probabilities",
                    "branch top1-agreement indicator",
                ],
                "raw_telemetry_features_used": False,
                "target": (
                    "1 when XGBoost assigns higher probability to the true class; "
                    "0 when SchemaFlow does"
                ),
                "target_tolerance": 1e-12,
                "exact_target_ties": "excluded from gate fitting only",
                "preprocessing": (
                    "continuous features standardized within meta-training data; "
                    "binary and one-hot features unchanged"
                ),
                "estimator": "binary L2 LogisticRegression",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "fit_intercept": True,
                "max_iter": 2000,
                "tolerance": 1e-8,
                "C_grid": list(C_GRID),
                "random_state": META_SEED,
                "inference": (
                    "alpha*xgboost_probability + (1-alpha)*schemaflow_probability"
                ),
            },
        },
        "model_lock": {
            "required_per_setting": [
                "fold assignment hash",
                "OOF metrics for every C candidate",
                "selected C and tie-break trace",
                "feature schema and preprocessing parameters",
                "refit coefficients and intercepts",
                "serialized model SHA-256",
            ],
            "model_count": {
                "logistic_stacking": 27,
                "learned_gate": 27,
            },
            "test_access_enabled_only_after_all_54_model_locks": True,
        },
        "future_test_evaluation": {
            "conditions": list(CONDITIONS),
            "metrics": ["macro_f1", "mcc", "balanced_accuracy"],
            "primary_differences": [
                "VLMF-minus-logistic-stacking macro_f1",
                "VLMF-minus-learned-gate macro_f1",
            ],
            "aggregation": (
                "equal conditions within setting, nine settings within dataset, "
                "then equal weight over three datasets"
            ),
            "bootstrap": {
                "method": "dataset_stratified_paired_setting_bootstrap",
                "resamples": 20000,
                "seed": 20260730,
                "interval": "percentile_95pct",
                "p_value_inferred": False,
            },
            "report_all_results_regardless_of_direction": True,
        },
        "explicit_exclusions": [
            "base-model retraining",
            "test-time hyperparameter selection",
            "raw-feature gate",
            "neural MLP gate",
            "calibration tuning",
            "structured-missingness evaluation",
            "threshold tuning",
            "candidate expansion after test access",
        ],
    }


def _readme(protocol_sha256: str) -> str:
    return f"""# Learned Ensemble Protocol v1

This directory freezes the deferred logistic-stacking and learned-gate
comparison protocol. It intentionally contains no fitted meta-model, validation
selection result, sealed-test read, or test metric.

Protocol SHA-256: `{protocol_sha256}`

Recreate this lock from `/root/autodl-tmp/Res-AE-Attn/repository`:

```bash
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.lock_learned_ensemble_protocol \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/{EXPERIMENT_NAME}
```

The command refuses an existing output directory. Running either learned
comparator requires a separate future authorization and a model-lock/evaluate
implementation that conforms byte-for-byte to `protocol.json`.
"""


def run_lock(output_dir: Path, repo_root: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite protocol lock: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = write_json_snapshot(output_dir / "protocol.json", protocol_payload())
    protocol_sha256 = _sha256(protocol_path)
    (output_dir / "protocol.sha256").write_text(
        f"{protocol_sha256}  protocol.json\n", encoding="ascii"
    )
    git = _git_state(repo_root)
    audit = {
        "format_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "phase": "protocol_locked_only",
        "timestamp_utc": _utc_now(),
        "protocol_sha256": protocol_sha256,
        "git": git,
        "assertions": {
            "validation_prediction_files_read": 0,
            "sealed_test_files_read": 0,
            "meta_models_fitted": 0,
            "test_metrics_computed": 0,
            "execution_started": False,
        },
    }
    write_json_snapshot(output_dir / "audit.json", audit)
    (output_dir / "README.md").write_text(
        _readme(protocol_sha256), encoding="utf-8"
    )
    names = ("README.md", "audit.json", "protocol.json", "protocol.sha256")
    hashes = [f"{_sha256(output_dir / name)}  {name}" for name in names]
    (output_dir / "output_hashes.sha256").write_text(
        "\n".join(hashes) + "\n", encoding="ascii"
    )
    return {
        "status": "learned_ensemble_protocol_locked_execution_not_started",
        "output_dir": str(output_dir),
        "protocol_sha256": protocol_sha256,
        "validation_files_read": 0,
        "test_files_read": 0,
        "git": git,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/Res-AE-Attn/results") / EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/root/autodl-tmp/Res-AE-Attn/repository"),
    )
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    print(
        json.dumps(
            run_lock(arguments.output_dir, arguments.repo_root),
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
