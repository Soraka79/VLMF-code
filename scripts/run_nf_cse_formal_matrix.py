#!/usr/bin/env python3
"""Run the predeclared NF-CSE A0-MR formal matrix with sealed-test barriers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from xgboost import XGBClassifier

from flowfm.config import load_config
from flowfm.data.labels import NF_CSE_SHARED_COARSE_LABEL_SPACE
from flowfm.data.missingness import (
    STRUCTURED_MISSINGNESS_ALGORITHM,
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_structured_field_group_missingness,
)
from flowfm.data.nf_cse_dataset import load_nf_cse_development_samples
from flowfm.data.nf_cse_few_shot import (
    SELECTION_MANIFEST_FILENAME,
    SELECTION_RECORDS_FILENAME,
    SELECTION_SUMMARY_FILENAME,
    publish_nf_cse_few_shot_split,
)
from flowfm.data.nf_cse_preprocessing import (
    SELECTION_FEATURES_FILENAME,
    publish_nf_cse_train_only_preprocessor,
)
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import (
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import write_json_snapshot
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.checkpoints import load_checkpoint
from flowfm.training.confirmatory_statistics import (
    NF_CSE_RARE_COARSE_LABELS,
    confirmatory_metric_payload,
    paired_cluster_bootstrap_macro_f1,
)
from flowfm.training.neural_baselines import build_neural_model
from flowfm.training.probability_fusion import (
    default_alpha_grid,
    pool_probabilities,
    probability_sha256,
    select_fusion,
)
from scripts.run_nf_cse_a0_mr_confirmation import (
    _CONDITIONS,
    _checkpoint_provenance,
    _coarse_targets,
    _collect_outputs,
    _conditions,
    _encoder_from_lock,
    _environment,
    _fit_xgboost,
    _git_state,
    _load_selection_inputs,
    _model_visible_samples,
    _policy_metrics,
    _policy_probabilities,
    _restore_model,
    _run_dir,
    _select_a0_mr,
    _sha256_file,
    _validate_initialization_provenance,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_DEFAULT_MATRIX_ROOT = _RESULTS_ROOT / "nf-cse-formal-matrix-v2"
_BASE_SPLIT = (
    _PROJECT_ROOT
    / "data/splits/NF-CSE-CIC-IDS2018-v2/"
    "confirmatory-unused-1m-seed20260725-shared-coarse-v1"
)
_SOURCE_CHECKPOINT = (
    _RESULTS_ROOT
    / "schemaflow-nf-unsw-pretrain-smoke-20260722T054417924340Z-"
    "9965a490-7de8cdfb4f42/checkpoints/epoch-0001.pt"
)
_CONFIRMATION_SELECTION = (
    _RESULTS_ROOT
    / "nf-cse-a0-mr-selection-20260725T052957672054Z-"
    "055ea090e76e-93bf5ed9"
)
_CONFIRMATION_RESULT = (
    _RESULTS_ROOT
    / "nf-cse-a0-mr-confirmation-20260725T054554366812Z-"
    "055ea090e76e-48876719"
)
_CONFIRMATION_SCRATCH = (
    _RESULTS_ROOT
    / "schemaflow-nf-cse-confirmatory-1pct-scratch-"
    "20260725T051826917352Z-9c9f49e8-860433b829e9/"
    "checkpoints/epoch-0020.pt"
)
_CONFIRMATION_PRETRAINED = (
    _RESULTS_ROOT
    / "schemaflow-nf-cse-confirmatory-1pct-nf-unsw-pretrained-"
    "20260725T052223377646Z-adf02ec7-2657dd1aaba6/"
    "checkpoints/epoch-0050.pt"
)
_CONFIRMATION_SPLIT = (
    _PROJECT_ROOT
    / "data/splits/NF-CSE-CIC-IDS2018-v2/"
    "few-shot-train-1pct-seed17-from-confirmatory-1m-v1"
)
_CONFIRMATION_PREPROCESSOR = (
    _PROJECT_ROOT
    / "data/preprocessing/NF-CSE-CIC-IDS2018-v2/"
    "train-only-confirmatory-1pct-seed17-v1"
)
_DATA_MANIFEST = (
    _PROJECT_ROOT
    / "repository/manifests/datasets/NF-CSE-CIC-IDS2018-v2.source.json"
)
_SEEDS = (17, 42, 20260725)
_TRAIN_PERCENTS = (1, 5, 10)
_STRUCTURED_CONDITIONS = tuple(
    f"structured__{name}" for name in STRUCTURED_MISSINGNESS_GROUPS
)
_FORMAL_CONDITIONS = _CONDITIONS + _STRUCTURED_CONDITIONS
_BASE_MODELS = (
    "xgboost",
    "ft_transformer",
    "schemaflow_scratch",
    "schemaflow_pretrained",
)
_FT_MODEL_CONFIG = {
    "d_token": 64,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}
_BUDGET_VERSION = "nf-cse-formal-matrix-budget-v2"
_PROTOCOL_VERSION = "nf-cse-a0-mr-formal-matrix-v2"


def _settings():
    return tuple(
        {
            "setting_id": f"seed-{seed}-pct-{percent:02d}",
            "seed": seed,
            "train_percent": percent,
        }
        for seed in _SEEDS
        for percent in _TRAIN_PERCENTS
    )


def _setting(seed: int, percent: int):
    matches = [
        value
        for value in _settings()
        if value["seed"] == seed and value["train_percent"] == percent
    ]
    if len(matches) != 1:
        raise ValueError("seed/train-percent is outside the formal matrix")
    return matches[0]


def _matrix_root(path) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_relative_to(_DATA_ROOT) or value == _DATA_ROOT:
        raise ValueError("matrix-root must be nested below /root/autodl-tmp")
    return value


def _state_path(root: Path, setting) -> Path:
    return root / "states" / f"{setting['setting_id']}.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    write_json_snapshot(path, payload)


def _artifact(path) -> dict[str, object]:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise RuntimeError(f"required artifact is missing: {value}")
    return {
        "path": str(value),
        "sha256": _sha256_file(value),
        "bytes": value.stat().st_size,
    }


def _budget_payload():
    return {
        "format_version": 1,
        "budget_version": _BUDGET_VERSION,
        "setting_count": 9,
        "seeds": list(_SEEDS),
        "train_percents": list(_TRAIN_PERCENTS),
        "random_missing_conditions": list(_CONDITIONS),
        "structured_missing_conditions": list(_STRUCTURED_CONDITIONS),
        "test_conditions_per_setting": len(_FORMAL_CONDITIONS),
        "nominal_jobs": {
            "schemaflow_scratch_training": 9,
            "schemaflow_pretrained_training": 9,
            "ft_transformer_training": 9,
            "xgboost_fit_and_validation_selection": 9,
            "locked_test_evaluation": 9,
            "total": 45,
        },
        "reused_completed_jobs": {
            "schemaflow_scratch_seed17_1pct": 1,
            "schemaflow_pretrained_seed17_1pct": 1,
            "total": 2,
        },
        "new_jobs": 43,
        "estimated_wall_hours": {"lower": 8.0, "upper": 14.0},
        "estimated_peak_gpu_gib": 2.0,
        "estimated_peak_system_memory_gib": 16.0,
        "estimated_incremental_storage_gib": 16.0,
        "estimate_basis": {
            "confirmation_selection_wall_seconds": 645.57,
            "confirmation_test_wall_seconds": 823.0,
            "scratch_1pct_train_seconds": 92.66,
            "pretrained_1pct_train_seconds": 218.97,
            "few_shot_scaling": "approximately linear in selected train rows",
            "formal_base_prediction_archive_estimate_gib": 2.3,
        },
        "gpu_over_two_hours_is_not_a_stop_condition": True,
        "stop_conditions": [
            "data_integrity_failure",
            "insufficient_disk",
            "fairness_contract_failure",
        ],
    }


def _split_paths(setting):
    if setting["seed"] == 17 and setting["train_percent"] == 1:
        return _CONFIRMATION_SPLIT, _CONFIRMATION_PREPROCESSOR
    split = (
        _PROJECT_ROOT
        / "data/splits/NF-CSE-CIC-IDS2018-v2"
        / (
            f"few-shot-train-{setting['train_percent']}pct-"
            f"seed{setting['seed']}-formal-v1"
        )
    )
    preprocessor = (
        _PROJECT_ROOT
        / "data/preprocessing/NF-CSE-CIC-IDS2018-v2"
        / (
            f"train-only-formal-{setting['train_percent']}pct-"
            f"seed{setting['seed']}-v1"
        )
    )
    return split, preprocessor


def _config_paths(root: Path, setting):
    prefix = root / "configs" / setting["setting_id"]
    return Path(f"{prefix}-scratch.yaml"), Path(f"{prefix}-pretrained.yaml")


def _training_config(setting, split_dir, preprocessor_dir, *, pretrained):
    name = (
        "schemaflow-nf-cse-formal-"
        f"{setting['train_percent']}pct-seed{setting['seed']}-"
        + ("nf-unsw-pretrained" if pretrained else "scratch")
    )
    return {
        "workflow": "finetune",
        "experiment": {
            "name": name,
            "output_root": str(_RESULTS_ROOT),
            "seed": setting["seed"],
            "enforce_autodl_data_disk": True,
        },
        "schema": {"version": "nf-cse-cic-ids2018-v2-model-41-v1"},
        "runtime": {"device": "cuda", "execution": "train"},
        "data": {
            "dataset_id": "NF-CSE-CIC-IDS2018-v2",
            "split_dir": str(split_dir),
            "preprocessor_dir": str(preprocessor_dir),
            "data_manifest": str(_DATA_MANIFEST),
        },
        "model": {
            "name": "schemaflow_fm_supervised",
            "d_model": 128,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "adapter_dim": 32,
        },
        "training": {
            "epochs": 50 if pretrained else 20,
            "batch_size": 64,
            "learning_rate": 0.001 if pretrained else 0.0003,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "scheduler_gamma": 0.98 if pretrained else 0.95,
            "shuffle": True,
            "class_balance": "inverse_frequency",
            "evaluation_scope": "validation_only",
            "loss_weights": {"binary": 1.0, "coarse": 1.0, "native": 1.0},
        },
    }


def _write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"formal config changed after publication: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _plan(root: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    plan_path = root / "matrix_plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        if (
            existing.get("protocol_version") != _PROTOCOL_VERSION
            or existing.get("matrix_root") != str(root)
            or existing.get("git") != git
            or existing.get("settings") != list(_settings())
            or existing.get("budget_sha256")
            != hashlib.sha256(
                json.dumps(_budget_payload(), sort_keys=True).encode("utf-8")
            ).hexdigest()
        ):
            raise RuntimeError("existing formal matrix plan is incompatible")
        for name, artifact in existing.get("artifacts", {}).items():
            path = Path(artifact["path"])
            if (
                not path.is_file()
                or path.stat().st_size != artifact["bytes"]
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise RuntimeError(f"planned formal artifact changed: {name}")
        if _read_json(
            Path(existing["artifacts"]["confirmation_gate"]["path"])
        ).get("full_matrix_allowed") is not True:
            raise RuntimeError("confirmation gate no longer permits the formal matrix")
        _write_json(root / "budget.json", _budget_payload())
        return existing
    required = {
        "base_split_summary": _BASE_SPLIT / "split_summary.json",
        "base_overlap_audit": _BASE_SPLIT / "overlap_audit.json",
        "base_sealed_test": _BASE_SPLIT / "sealed_test.json",
        "confirmation_gate": _CONFIRMATION_RESULT / "gate.json",
        "confirmation_prediction_lock": (
            _CONFIRMATION_RESULT / "prediction_lock.json"
        ),
        "source_checkpoint": _SOURCE_CHECKPOINT,
        "data_manifest": _DATA_MANIFEST,
    }
    artifacts = {name: _artifact(path) for name, path in required.items()}
    gate = _read_json(required["confirmation_gate"])
    if gate.get("passed") is not True or gate.get("full_matrix_allowed") is not True:
        raise RuntimeError("predeclared confirmation gate does not allow formal matrix")
    overlap = _read_json(required["base_overlap_audit"])
    if overlap.get("verified_zero_overlap") is not True:
        raise RuntimeError("base cohort no longer proves zero historical overlap")
    disk = shutil.disk_usage(_DATA_ROOT)
    budget = _budget_payload()
    if disk.free < int(budget["estimated_incremental_storage_gib"] * 1024**3 * 1.5):
        raise RuntimeError("insufficient disk for the formal matrix safety margin")
    payload = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "status": "planned_before_formal_training",
        "git": git,
        "matrix_root": str(root),
        "base_split_dir": str(_BASE_SPLIT),
        "confirmation_selection_dir": str(_CONFIRMATION_SELECTION),
        "confirmation_result_dir": str(_CONFIRMATION_RESULT),
        "settings": list(_settings()),
        "labels": list(NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels),
        "conditions": list(_FORMAL_CONDITIONS),
        "base_models": list(_BASE_MODELS),
        "fusion_models": ["fixed_a0", "a0_mr"],
        "alpha_grid": list(default_alpha_grid(0.05)),
        "tie_break": [
            "mcc",
            "balanced_accuracy",
            "linear_pooling",
            "higher_xgboost_alpha",
        ],
        "fixed_a0": {"method": "linear", "alpha_xgboost": 0.55},
        "structured_missingness_algorithm": STRUCTURED_MISSINGNESS_ALGORITHM,
        "all_validation_selections_must_lock_before_any_test": True,
        "test_evaluations_per_setting": 1,
        "prior_confirmatory_test_access": True,
        "artifacts": artifacts,
        "disk_at_plan": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "budget_sha256": hashlib.sha256(
            json.dumps(budget, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".incomplete"
    if not marker.exists():
        marker.write_text("formal matrix incomplete\n", encoding="ascii")
    _write_json(plan_path, payload)
    _write_json(root / "budget.json", budget)
    return payload


def _verify_existing_few_shot(split_dir: Path, setting):
    required = (
        "config.snapshot.json",
        "split_summary.json",
        SELECTION_SUMMARY_FILENAME,
        SELECTION_RECORDS_FILENAME,
        SELECTION_MANIFEST_FILENAME,
    )
    if any(not (split_dir / name).is_file() for name in required):
        raise RuntimeError(f"few-shot split is incomplete: {split_dir}")
    config = _read_json(split_dir / "config.snapshot.json")
    if (
        config.get("seed") != setting["seed"]
        or config.get("train_percent") != setting["train_percent"]
        or Path(config.get("source_split_dir", "")).resolve() != _BASE_SPLIT
        or config.get("sealed_test") is not True
    ):
        raise RuntimeError("existing formal few-shot split has different provenance")


def _verify_existing_preprocessor(preprocessor_dir: Path, split_dir: Path):
    required = (
        "preprocessor.json",
        "preprocessor_summary.json",
        "config.snapshot.json",
        SELECTION_SUMMARY_FILENAME,
        SELECTION_FEATURES_FILENAME,
    )
    if any(not (preprocessor_dir / name).is_file() for name in required):
        raise RuntimeError(f"preprocessor is incomplete: {preprocessor_dir}")
    config = _read_json(preprocessor_dir / "config.snapshot.json")
    if Path(config.get("source_split_dir", "")).resolve() != split_dir:
        raise RuntimeError("existing formal preprocessor belongs to another split")


def _prepare(root: Path):
    plan = _plan(root)
    rows = []
    repo_dir = Path(__file__).resolve().parents[1]
    for setting in _settings():
        split_dir, preprocessor_dir = _split_paths(setting)
        if split_dir.exists():
            _verify_existing_few_shot(split_dir, setting)
        else:
            publish_nf_cse_few_shot_split(
                _BASE_SPLIT,
                split_dir,
                train_percent=setting["train_percent"],
                seed=setting["seed"],
                repo_dir=repo_dir,
            )
        if preprocessor_dir.exists():
            _verify_existing_preprocessor(preprocessor_dir, split_dir)
        else:
            publish_nf_cse_train_only_preprocessor(
                split_dir,
                preprocessor_dir,
                repo_dir=repo_dir,
            )
        scratch_config, pretrained_config = _config_paths(root, setting)
        _write_yaml(
            scratch_config,
            _training_config(
                setting, split_dir, preprocessor_dir, pretrained=False
            ),
        )
        _write_yaml(
            pretrained_config,
            _training_config(
                setting, split_dir, preprocessor_dir, pretrained=True
            ),
        )
        selection_summary = _read_json(split_dir / SELECTION_SUMMARY_FILENAME)
        preprocessing_summary = _read_json(
            preprocessor_dir / "preprocessor_summary.json"
        )
        rows.append(
            {
                **setting,
                "split_dir": str(split_dir),
                "preprocessor_dir": str(preprocessor_dir),
                "scratch_config": _artifact(scratch_config),
                "pretrained_config": _artifact(pretrained_config),
                "selection_records_sha256": selection_summary[
                    "selected_records_sha256"
                ],
                "selection_manifest_sha256": selection_summary[
                    "split_manifest_sha256"
                ],
                "preprocessor_file_sha256": preprocessing_summary[
                    "preprocessor_file_sha256"
                ],
                "train_rows": preprocessing_summary["training_rows"],
                "validation_test_natural_distribution_preserved": True,
                "test_labels_or_statistics_in_selection_view": False,
            }
        )
    payload = {
        "format_version": 1,
        "protocol_version": plan["protocol_version"],
        "status": "all_formal_data_views_prepared",
        "settings": rows,
    }
    _write_json(root / "prepare.completed.json", payload)
    return payload


def _load_state(root: Path, setting):
    path = _state_path(root, setting)
    if not path.exists():
        return {
            "format_version": 1,
            "protocol_version": _PROTOCOL_VERSION,
            **setting,
            "steps": {},
        }
    payload = _read_json(path)
    if any(payload.get(key) != value for key, value in setting.items()):
        raise RuntimeError("formal state belongs to another setting")
    return payload


def _save_state(root: Path, setting, state):
    path = _state_path(root, setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, state)


def _completed_step(state, name):
    step = state["steps"].get(name)
    if not isinstance(step, dict) or step.get("status") != "completed":
        return None
    artifacts = step.get("artifacts", {})
    for value in artifacts.values():
        path = Path(value["path"])
        if not path.is_file() or _sha256_file(path) != value["sha256"]:
            raise RuntimeError(f"completed {name} artifact changed: {path}")
    return step


def _run_logged(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        tuple(str(value) for value in command),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"formal subprocess failed ({completed.returncode}); see {log_path}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"formal subprocess produced no result JSON: {log_path}")
    try:
        result = json.loads(lines[-1])
    except ValueError as exc:
        raise RuntimeError(f"formal subprocess result is not JSON: {log_path}") from exc
    return result, time.perf_counter() - started


def _record_training_step(root, setting, state, name, result, wall, *, reused):
    checkpoint = Path(result["checkpoint"]).resolve()
    run_dir = Path(result["run_dir"]).resolve()
    config_name = (
        "config.json" if name == "ft_transformer" else "config.snapshot.json"
    )
    state["steps"][name] = {
        "status": "completed",
        "reused_from_confirmation": reused,
        "wall_seconds": wall,
        "run_dir": str(run_dir),
        "artifacts": {
            "checkpoint": _artifact(checkpoint),
            "metrics": _artifact(run_dir / "metrics.json"),
            "config": _artifact(run_dir / config_name),
        },
    }
    _save_state(root, setting, state)


def _train_all(root: Path):
    prepared = _prepare(root)
    prepared_by_id = {row["setting_id"]: row for row in prepared["settings"]}
    for setting in _settings():
        row = prepared_by_id[setting["setting_id"]]
        state = _load_state(root, setting)
        scratch_config = Path(row["scratch_config"]["path"])
        pretrained_config = Path(row["pretrained_config"]["path"])
        if _completed_step(state, "schemaflow_scratch") is None:
            if setting["seed"] == 17 and setting["train_percent"] == 1:
                result = {
                    "checkpoint": str(_CONFIRMATION_SCRATCH),
                    "run_dir": str(_CONFIRMATION_SCRATCH.parents[1]),
                }
                _record_training_step(
                    root, setting, state, "schemaflow_scratch", result, 0.0,
                    reused=True,
                )
            else:
                result, wall = _run_logged(
                    (
                        sys.executable,
                        "-m",
                        "flowfm",
                        "finetune",
                        "--config",
                        scratch_config,
                    ),
                    root / "logs" / f"{setting['setting_id']}-scratch.log",
                )
                _record_training_step(
                    root, setting, state, "schemaflow_scratch", result, wall,
                    reused=False,
                )
        state = _load_state(root, setting)
        if _completed_step(state, "schemaflow_pretrained") is None:
            if setting["seed"] == 17 and setting["train_percent"] == 1:
                result = {
                    "checkpoint": str(_CONFIRMATION_PRETRAINED),
                    "run_dir": str(_CONFIRMATION_PRETRAINED.parents[1]),
                }
                _record_training_step(
                    root, setting, state, "schemaflow_pretrained", result, 0.0,
                    reused=True,
                )
            else:
                result, wall = _run_logged(
                    (
                        sys.executable,
                        "-m",
                        "flowfm",
                        "finetune",
                        "--config",
                        pretrained_config,
                        "--initialize-from",
                        _SOURCE_CHECKPOINT,
                        "--allow-cross-dataset-init",
                    ),
                    root / "logs" / f"{setting['setting_id']}-pretrained.log",
                )
                _record_training_step(
                    root, setting, state, "schemaflow_pretrained", result, wall,
                    reused=False,
                )
        state = _load_state(root, setting)
        if _completed_step(state, "ft_transformer") is None:
            result, wall = _run_logged(
                (
                    sys.executable,
                    "-m",
                    "scripts.run_nf_cse_ft_transformer",
                    "--data-config",
                    scratch_config,
                    "--seed",
                    setting["seed"],
                    "--epochs",
                    20,
                    "--batch-size",
                    256,
                    "--learning-rate",
                    0.001,
                    "--weight-decay",
                    0.01,
                    "--gradient-clip",
                    1.0,
                    "--device",
                    "cuda",
                ),
                root / "logs" / f"{setting['setting_id']}-ft-transformer.log",
            )
            _record_training_step(
                root, setting, state, "ft_transformer", result, wall,
                reused=False,
            )
    return {"status": "all_formal_neural_training_completed", "settings": 9}


def _normalize_probabilities(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("formal probabilities must be finite and non-negative")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("formal probability row has no mass")
    return matrix / row_sums


def _formal_conditions(samples, seed):
    values, random_reports = _conditions(samples, seed)
    reports = dict(random_reports)
    for group_name in STRUCTURED_MISSINGNESS_GROUPS:
        condition = f"structured__{group_name}"
        conditioned, report = apply_structured_field_group_missingness(
            samples,
            group_name=group_name,
        )
        values[condition] = conditioned
        reports[condition] = report
    if tuple(values) != _FORMAL_CONDITIONS:
        raise RuntimeError("formal missingness condition order changed")
    return values, reports


def _restore_ft_model(run_dir, checkpoint_path, tokenizer, device):
    run_dir = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    required = (
        "config.json",
        "metrics.json",
        "formal_provenance.json",
        "tokenizer_metadata.json",
        "label_space.json",
        "checkpoints/index.json",
    )
    if any(not (run_dir / name).is_file() for name in required):
        raise RuntimeError("FT-Transformer provenance is incomplete")
    if (run_dir / ".incomplete").exists():
        raise RuntimeError("FT-Transformer training run is incomplete")
    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    provenance = _read_json(run_dir / "formal_provenance.json")
    saved_tokenizer = TokenizerMetadata.from_dict(
        _read_json(run_dir / "tokenizer_metadata.json")
    )
    if saved_tokenizer != tokenizer:
        raise RuntimeError("FT-Transformer tokenizer differs from formal selection")
    if _read_json(run_dir / "label_space.json") != (
        NF_CSE_SHARED_COARSE_LABEL_SPACE.to_dict()
    ):
        raise RuntimeError("FT-Transformer label task is not the frozen five-class view")
    if (
        config.get("architecture") != "ft_transformer"
        or config.get("task") != "coarse"
        or config.get("task_labels")
        != list(NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels)
        or config.get("model_config") != _FT_MODEL_CONFIG
        or config.get("evaluation_scope") != "validation_only"
        or "test" in metrics
        or provenance.get("test_samples_materialized") is not False
        or provenance.get("sealed_test_file_content_read") is not False
    ):
        raise RuntimeError("FT-Transformer checkpoint violates sealed formal protocol")
    index = _read_json(run_dir / "checkpoints/index.json")
    if (
        Path(index.get("latest", "")).name != checkpoint_path.name
        or index.get("sha256") != _sha256_file(checkpoint_path)
        or index.get("verified") is not True
    ):
        raise RuntimeError("FT-Transformer checkpoint index changed")
    try:
        raw_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        raw_payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_neural_model(
        "ft_transformer",
        "coarse",
        tokenizer_metadata=tokenizer,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        label_space=NF_CSE_SHARED_COARSE_LABEL_SPACE,
        model_config=_FT_MODEL_CONFIG,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    payload = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        map_location=device,
        restore_rng=False,
        expected_model_data_metadata=raw_payload["model_data_metadata"],
    )
    if (
        payload["config"] != config
        or payload["extra_state"].get("metrics") != metrics
        or "test" in payload["extra_state"].get("metrics", {})
    ):
        raise RuntimeError("FT-Transformer checkpoint payload changed")
    model.eval()
    return model, payload, provenance


def _ft_probabilities(model, samples, tokenizer, device, batch_size):
    blocks = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = collate_schemaflow_samples(
                samples[start : start + batch_size],
                tokenizer,
                registry=SCHEMAFLOW_MAIN_REGISTRY,
            ).to(device)
            blocks.append(torch.softmax(model(batch).logits, dim=1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    probabilities = _normalize_probabilities(
        torch.cat(blocks).numpy().astype(np.float64, copy=False)
    )
    return probabilities, time.perf_counter() - started


def _collect_formal_outputs(
    samples_by_condition,
    *,
    encoder,
    xgboost_model,
    scratch_model,
    pretrained_model,
    ft_model,
    tokenizer,
    device,
    schemaflow_batch_size,
    ft_batch_size,
):
    outputs, timings = _collect_outputs(
        samples_by_condition,
        encoder=encoder,
        xgboost_model=xgboost_model,
        scratch_model=scratch_model,
        pretrained_model=pretrained_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=schemaflow_batch_size,
    )
    timings["ft_transformer"] = 0.0
    for condition, samples in samples_by_condition.items():
        for model_name in (
            "xgboost",
            "schemaflow_scratch",
            "schemaflow_pretrained",
        ):
            outputs[condition][model_name] = _normalize_probabilities(
                outputs[condition][model_name]
            )
        ft, seconds = _ft_probabilities(
            ft_model,
            samples,
            tokenizer,
            device,
            ft_batch_size,
        )
        outputs[condition]["ft_transformer"] = ft
        timings["ft_transformer"] += seconds
    return outputs, timings


def _formal_base_metrics(outputs, *, confirmatory=False):
    metric = confirmatory_metric_payload if confirmatory else metric_payload
    return {
        condition: {
            model: metric(
                values[model],
                values["targets"],
                NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels,
            )
            for model in _BASE_MODELS
        }
        for condition, values in outputs.items()
    }


def _fixed_policies():
    return {
        condition: {
            "method": "linear",
            "alpha": 0.55,
            "selected_on": "frozen",
        }
        for condition in _FORMAL_CONDITIONS
    }


def _select_formal_policies(outputs):
    if tuple(outputs) != _FORMAL_CONDITIONS:
        raise ValueError("formal outputs do not match predeclared conditions")
    random_outputs = {name: outputs[name] for name in _CONDITIONS}
    random_selection = _select_a0_mr(
        random_outputs,
        default_alpha_grid(0.05),
        0.005,
        0.002,
    )
    routing_passed = random_selection["routing_gate"]["passed"]
    candidate_name = random_selection["routing_gate"]["candidate_name"]
    if candidate_name == "missing_rate_conditioned_linear_or_log":
        methods = ("linear", "log")
    elif candidate_name == "missing_rate_conditioned_linear":
        methods = ("linear",)
    else:
        raise RuntimeError("frozen A0-MR candidate name changed")
    structured = {}
    structured_policies = {}
    for condition in _STRUCTURED_CONDITIONS:
        if routing_passed:
            result = select_fusion(
                outputs[condition]["xgboost"],
                outputs[condition]["schemaflow_pretrained"],
                outputs[condition]["targets"],
                NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels,
                alpha_grid=default_alpha_grid(0.05),
                methods=methods,
            )
            policy = {
                "method": result["selected"]["method"],
                "alpha": result["selected"]["alpha"],
                "selected_on": condition,
            }
            structured[condition] = result
        else:
            policy = {
                "method": "linear",
                "alpha": 0.55,
                "selected_on": "frozen_fallback",
            }
            structured[condition] = {
                "selection_scope": "validation_only",
                "search_skipped_by_frozen_random_condition_routing_gate": True,
            }
        structured_policies[condition] = policy
    policies = {
        **random_selection["locked_policy"]["policies"],
        **structured_policies,
    }
    locked_metrics = _policy_metrics(outputs, policies)
    fixed = _fixed_policies()
    return {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "selection_scope": "validation_only",
        "test_samples_materialized": False,
        "test_metrics_used": False,
        "random_condition_selection": random_selection,
        "structured_extension_rule": {
            "frozen_before_formal_test": True,
            "routing_source": "random_clean_30pct_50pct_frozen_a0_mr_gate",
            "routing_passed": routing_passed,
            "allowed_methods": list(methods) if routing_passed else ["linear"],
            "alpha_grid": list(default_alpha_grid(0.05)),
            "tie_break": [
                "mcc",
                "balanced_accuracy",
                "linear_pooling",
                "higher_xgboost_alpha",
            ],
            "fallback": "fixed_a0_linear_alpha_xgb_0p55",
        },
        "structured_condition_search": structured,
        "fixed_a0": {
            "name": "a0_fixed_linear_alpha_xgb_0p55",
            "policies": fixed,
            "validation_metrics": _policy_metrics(outputs, fixed),
        },
        "locked_policy": {
            "name": (
                candidate_name if routing_passed else "a0_fixed_linear_alpha_xgb_0p55"
            ),
            "policies": policies,
            "validation_metrics": locked_metrics,
            "mean_random_condition_macro_f1": float(
                np.mean(
                    [locked_metrics[name]["macro_f1"] for name in _CONDITIONS]
                )
            ),
        },
    }


def _formal_output_hashes(outputs):
    return {
        condition: {
            **{
                f"{model}_probabilities_sha256": probability_sha256(values[model])
                for model in _BASE_MODELS
            },
            "targets_sha256": hashlib.sha256(
                np.ascontiguousarray(values["targets"], dtype="<i8").tobytes()
            ).hexdigest(),
            "sample_order_sha256": values["sample_order_sha256"],
            "sample_count": int(len(values["targets"])),
        }
        for condition, values in outputs.items()
    }


def _selection_dependencies(inputs, scratch_config, pretrained_config, checkpoints, ft_step):
    values = {
        "scratch_config": scratch_config,
        "pretrained_config": pretrained_config,
        "scratch_checkpoint": checkpoints[0],
        "pretrained_checkpoint": checkpoints[1],
        "ft_checkpoint": ft_step["artifacts"]["checkpoint"]["path"],
        "ft_config": ft_step["artifacts"]["config"]["path"],
        "ft_metrics": ft_step["artifacts"]["metrics"]["path"],
        "ft_formal_provenance": Path(ft_step["run_dir"])
        / "formal_provenance.json",
        "ft_tokenizer_metadata": Path(ft_step["run_dir"])
        / "tokenizer_metadata.json",
        "ft_label_space": Path(ft_step["run_dir"]) / "label_space.json",
        "source_pretrain_checkpoint": _SOURCE_CHECKPOINT,
        "data_manifest": inputs["data_manifest_path"],
        "selection_records": inputs["split_selection_records_path"],
        "selection_manifest": inputs["split_manifest_path"],
        "split_selection_summary": inputs["split_selection_summary_path"],
        "preprocessor_selection_summary": inputs[
            "preprocessor_selection_summary_path"
        ],
        "selection_features": inputs["selection_features_path"],
        "preprocessor": inputs["preprocessor_dir"] / "preprocessor.json",
        "sealed_selected_records": inputs["split_dir"] / "selected_records.jsonl",
        "sealed_split_manifest": inputs["split_dir"] / "split_manifest.json",
        "base_selected_records": inputs["source_split_dir"] / "selected_records.jsonl",
        "base_split_manifest": inputs["source_split_dir"] / "split_manifest.json",
        "base_split_summary": inputs["source_split_dir"] / "split_summary.json",
        "base_overlap_audit": inputs["source_split_dir"] / "overlap_audit.json",
        "base_sealed_test": inputs["source_split_dir"] / "sealed_test.json",
    }
    return {name: _artifact(path) for name, path in values.items()}


def _select_setting(root: Path, setting):
    state = _load_state(root, setting)
    for name in ("schemaflow_scratch", "schemaflow_pretrained", "ft_transformer"):
        if _completed_step(state, name) is None:
            raise RuntimeError(f"cannot select before {name} completes")
    prepared = _read_json(root / "prepare.completed.json")
    row = next(
        item for item in prepared["settings"] if item["setting_id"] == setting["setting_id"]
    )
    scratch_config = Path(row["scratch_config"]["path"])
    pretrained_config = Path(row["pretrained_config"]["path"])
    checkpoints = (
        Path(state["steps"]["schemaflow_scratch"]["artifacts"]["checkpoint"]["path"]),
        Path(state["steps"]["schemaflow_pretrained"]["artifacts"]["checkpoint"]["path"]),
    )
    inputs = _load_selection_inputs(scratch_config, pretrained_config)
    if (
        inputs["configs"][0].experiment["seed"] != setting["seed"]
        or inputs["split_config"].get("seed") != setting["seed"]
        or inputs["split_config"].get("train_percent") != setting["train_percent"]
    ):
        raise RuntimeError("formal selection setting differs from data provenance")
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train = _model_visible_samples(inputs["dataset"].splits["train"])
    targets = _coarse_targets(train)
    from flowfm.training.baselines import TabularFeatureEncoder

    encoder = TabularFeatureEncoder.fit(
        train,
        category_encoders=inputs["tokenizer"].categorical_vocabularies,
    )
    xgb, xgb_parameters, xgb_fit_seconds = _fit_xgboost(
        encoder.transform(train), targets, setting["seed"]
    )
    scratch_model, scratch_payload, scratch_adaptation = _restore_model(
        inputs["configs"][0], checkpoints[0], inputs["tokenizer"], device
    )
    pretrained_model, pretrained_payload, pretrained_adaptation = _restore_model(
        inputs["configs"][1], checkpoints[1], inputs["tokenizer"], device
    )
    initialization = _validate_initialization_provenance(
        scratch_payload, pretrained_payload, _SOURCE_CHECKPOINT
    )
    ft_step = state["steps"]["ft_transformer"]
    ft_run_dir = Path(ft_step["run_dir"])
    ft_checkpoint = Path(ft_step["artifacts"]["checkpoint"]["path"])
    ft_model, ft_payload, ft_provenance = _restore_ft_model(
        ft_run_dir, ft_checkpoint, inputs["tokenizer"], device
    )
    validation_conditions, missingness = _formal_conditions(
        inputs["dataset"].splits["val"], setting["seed"]
    )
    outputs, timings = _collect_formal_outputs(
        validation_conditions,
        encoder=encoder,
        xgboost_model=xgb,
        scratch_model=scratch_model,
        pretrained_model=pretrained_model,
        ft_model=ft_model,
        tokenizer=inputs["tokenizer"],
        device=device,
        schemaflow_batch_size=inputs["configs"][0].training["batch_size"],
        ft_batch_size=256,
    )
    selection = _select_formal_policies(outputs)
    run_dir = _run_dir(
        _RESULTS_ROOT,
        f"nf-cse-formal-selection-{setting['setting_id']}",
        git["commit"],
    )
    xgb_path = run_dir / "xgboost_model.json"
    xgb.save_model(str(xgb_path))
    dependencies = _selection_dependencies(
        inputs,
        scratch_config,
        pretrained_config,
        checkpoints,
        ft_step,
    )
    config = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "workflow": "nf_cse_a0_mr_formal_selection",
        **setting,
        "selection_scope": ["train", "validation"],
        "materialized_splits": ["train", "val"],
        "test_manifest_members_read": False,
        "sealed_test_file_content_read": False,
        "test_samples_materialized": False,
        "test_metrics_used": False,
        "split_dir": str(inputs["split_dir"]),
        "preprocessor_dir": str(inputs["preprocessor_dir"]),
        "base_split_dir": str(inputs["source_split_dir"]),
        "scratch_config_path": str(scratch_config),
        "pretrained_config_path": str(pretrained_config),
        "scratch_checkpoint_path": str(checkpoints[0]),
        "pretrained_checkpoint_path": str(checkpoints[1]),
        "ft_run_dir": str(ft_run_dir),
        "ft_checkpoint_path": str(ft_checkpoint),
        "source_pretrain_checkpoint_path": str(_SOURCE_CHECKPOINT),
        "seed": setting["seed"],
        "missing_seed": setting["seed"],
        "conditions": list(_FORMAL_CONDITIONS),
        "device": "cuda",
        "git": git,
        "dependencies": dependencies,
        "xgboost": {
            "parameters": xgb_parameters,
            "fit_scope": "target_train_only",
            "class_balance": "inverse_frequency_sample_weight",
            "training_sample_count": len(train),
        },
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "selection.json", selection)
    _write_json(run_dir / "validation_metrics.json", _formal_base_metrics(outputs))
    _write_json(run_dir / "validation_probability_hashes.json", _formal_output_hashes(outputs))
    _write_json(run_dir / "missingness.json", {"validation": missingness})
    _write_json(run_dir / "encoder.json", encoder.to_dict())
    _write_json(run_dir / "tokenizer_metadata.json", inputs["tokenizer"].to_dict())
    _write_json(run_dir / "feature_registry.json", SCHEMAFLOW_MAIN_REGISTRY.to_dict())
    _write_json(
        run_dir / "checkpoint_provenance.json",
        {
            "scratch": _checkpoint_provenance(
                checkpoints[0], scratch_payload, scratch_adaptation
            ),
            "pretrained": _checkpoint_provenance(
                checkpoints[1], pretrained_payload, pretrained_adaptation
            ),
            "pretrained_initialization": initialization,
            "ft_transformer": {
                "path": str(ft_checkpoint),
                "sha256": _sha256_file(ft_checkpoint),
                "epoch": ft_payload["epoch"],
                "global_step": ft_payload["global_step"],
                "git_commit": ft_payload["git_commit"],
                "evaluation_scope": ft_payload["config"]["evaluation_scope"],
                "test_metrics_present": "test" in ft_payload["extra_state"]["metrics"],
                "formal_provenance_sha256": _sha256_file(
                    ft_run_dir / "formal_provenance.json"
                ),
                "formal_provenance": ft_provenance,
            },
        },
    )
    _write_json(run_dir / "environment.json", _environment())
    _write_json(
        run_dir / "efficiency.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "xgboost_fit_seconds": xgb_fit_seconds,
            "validation_inference_seconds": timings,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
    )
    (run_dir / "xgboost_model.sha256").write_text(
        _sha256_file(xgb_path) + "  xgboost_model.json\n", encoding="ascii"
    )
    shutil.copyfile(inputs["overlap_path"], run_dir / "overlap_audit.json")
    _write_json(
        run_dir / "sealed_test.reference.json",
        {
            "status": "sealed_reference_not_opened_during_formal_selection",
            "content_read_during_selection": False,
            "path": str(inputs["seal_path"]),
            "expected_sha256": _sha256_file(inputs["seal_path"]),
        },
    )
    (run_dir / ".incomplete").unlink()
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "selection": str(run_dir / "selection.json"),
    }


def _record_selection_step(root, setting, result, wall):
    state = _load_state(root, setting)
    run_dir = Path(result["run_dir"]).resolve()
    state["steps"]["validation_selection"] = {
        "status": "completed",
        "wall_seconds": wall,
        "run_dir": str(run_dir),
        "artifacts": {
            "selection": _artifact(run_dir / "selection.json"),
            "config": _artifact(run_dir / "config.json"),
            "xgboost_model": _artifact(run_dir / "xgboost_model.json"),
            "validation_metrics": _artifact(run_dir / "validation_metrics.json"),
        },
    }
    _save_state(root, setting, state)


def _select_all(root: Path):
    _train_all(root)
    for setting in _settings():
        state = _load_state(root, setting)
        if _completed_step(state, "validation_selection") is not None:
            continue
        result, wall = _run_logged(
            (
                sys.executable,
                "-m",
                "scripts.run_nf_cse_formal_matrix",
                "select",
                "--matrix-root",
                root,
                "--seed",
                setting["seed"],
                "--train-percent",
                setting["train_percent"],
            ),
            root / "logs" / f"{setting['setting_id']}-selection.log",
        )
        _record_selection_step(root, setting, result, wall)
    return _lock_all_selections(root)


def _lock_all_selections(root: Path):
    path = root / "all_validation_selections.lock.json"
    if path.exists():
        existing = _read_json(path)
        if (
            existing.get("status")
            != "all_nine_validation_selections_locked_before_any_formal_test"
            or existing.get("setting_count") != 9
        ):
            raise RuntimeError("existing global validation selection lock is invalid")
        for record in existing["settings"]:
            selection_dir = Path(record["selection_dir"])
            if (
                _sha256_file(selection_dir / "selection.json")
                != record["selection_sha256"]
                or _sha256_file(selection_dir / "config.json")
                != record["selection_config_sha256"]
            ):
                raise RuntimeError("a globally locked validation selection changed")
        return existing
    records = []
    for setting in _settings():
        state = _load_state(root, setting)
        step = _completed_step(state, "validation_selection")
        if step is None:
            raise RuntimeError("all validation selections must finish before test")
        selection = step["artifacts"]["selection"]
        records.append(
            {
                **setting,
                "selection_dir": step["run_dir"],
                "selection_sha256": selection["sha256"],
                "selection_config_sha256": step["artifacts"]["config"]["sha256"],
            }
        )
    if any(
        (Path(record["selection_dir"]) / "formal_test_access.claim.json").exists()
        for record in records
    ):
        raise RuntimeError("a formal test was accessed before the global selection lock")
    payload = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "status": "all_nine_validation_selections_locked_before_any_formal_test",
        "setting_count": len(records),
        "settings": records,
    }
    _write_json(path, payload)
    (root / "all_validation_selections.lock.sha256").write_text(
        _sha256_file(path) + "  all_validation_selections.lock.json\n",
        encoding="ascii",
    )
    return payload


def _verify_selection_for_test(root: Path, setting):
    lock_path = root / "all_validation_selections.lock.json"
    if not lock_path.is_file():
        raise RuntimeError("formal test blocked until all validation selections lock")
    lock = _read_json(lock_path)
    if (
        lock.get("status")
        != "all_nine_validation_selections_locked_before_any_formal_test"
        or lock.get("setting_count") != 9
    ):
        raise RuntimeError("global validation selection lock is invalid")
    state = _load_state(root, setting)
    step = _completed_step(state, "validation_selection")
    if step is None:
        raise RuntimeError("setting validation selection is incomplete")
    record = next(
        value
        for value in lock["settings"]
        if value["setting_id"] == setting["setting_id"]
    )
    selection_dir = Path(step["run_dir"]).resolve()
    if (
        Path(record["selection_dir"]).resolve() != selection_dir
        or record["selection_sha256"]
        != _sha256_file(selection_dir / "selection.json")
        or record["selection_config_sha256"]
        != _sha256_file(selection_dir / "config.json")
    ):
        raise RuntimeError("setting selection differs from the global lock")
    config = _read_json(selection_dir / "config.json")
    selection = _read_json(selection_dir / "selection.json")
    if (
        config.get("protocol_version") != _PROTOCOL_VERSION
        or config.get("setting_id") != setting["setting_id"]
        or config.get("test_samples_materialized") is not False
        or selection.get("test_metrics_used") is not False
        or tuple(config.get("conditions", ())) != _FORMAL_CONDITIONS
        or set(selection["locked_policy"]["policies"]) != set(_FORMAL_CONDITIONS)
    ):
        raise RuntimeError("formal validation selection lock is malformed")
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    if config["git"]["commit"] != git["commit"]:
        raise RuntimeError("selection and formal test must use the same commit")
    for name, artifact in config["dependencies"].items():
        path = Path(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or _sha256_file(path) != artifact["sha256"]
        ):
            raise RuntimeError(f"formal selection dependency changed: {name}")
    xgb_hash = (selection_dir / "xgboost_model.sha256").read_text().split()[0]
    if _sha256_file(selection_dir / "xgboost_model.json") != xgb_hash:
        raise RuntimeError("formal XGBoost model changed after validation lock")
    overlap = _read_json(Path(config["base_split_dir"]) / "overlap_audit.json")
    if overlap.get("verified_zero_overlap") is not True:
        raise RuntimeError("formal base cohort no longer proves independence")
    seal = _read_json(Path(config["base_split_dir"]) / "sealed_test.json")
    if (
        seal.get("status") != "sealed_before_model_selection"
        or seal.get("selection_access_allowed") is not False
        or seal.get("opened_for_evaluation") is not False
    ):
        raise RuntimeError("formal base sealed-test lock is invalid")
    return git, selection_dir, config, selection, lock_path


def _exclusive_json(path: Path, payload):
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _write_formal_prediction_lock(run_dir, outputs, group_ids):
    if tuple(outputs) != _FORMAL_CONDITIONS:
        raise RuntimeError("formal test condition order changed")
    targets = np.asarray(outputs["clean"]["targets"], dtype=np.int64)
    if any(
        not np.array_equal(targets, outputs[name]["targets"])
        for name in _FORMAL_CONDITIONS[1:]
    ):
        raise RuntimeError("formal test targets differ across missing conditions")
    unique_groups, group_codes = np.unique(
        np.asarray(group_ids, dtype="U64"), return_inverse=True
    )
    if len(group_codes) != len(targets):
        raise RuntimeError("formal test groups do not align with predictions")
    arrays = {
        "targets": targets,
        "group_codes": group_codes.astype(np.int32, copy=False),
    }
    for condition, values in outputs.items():
        for model in _BASE_MODELS:
            arrays[f"{model}__{condition}"] = values[model]
    predictions_path = run_dir / "predictions.npz"
    np.savez(predictions_path, **arrays)
    groups_path = run_dir / "group_hashes.txt"
    groups_path.write_text(
        "".join(f"{value}\n" for value in unique_groups), encoding="ascii"
    )
    lock = {
        "format_version": 1,
        "status": "locked_before_metric_computation",
        "conditions": _formal_output_hashes(outputs),
        "base_models": list(_BASE_MODELS),
        "predictions_path": predictions_path.name,
        "predictions_sha256": _sha256_file(predictions_path),
        "prediction_archive_bytes": predictions_path.stat().st_size,
        "group_hashes_path": groups_path.name,
        "group_hashes_sha256": _sha256_file(groups_path),
        "group_count": int(len(unique_groups)),
        "targets_sha256": hashlib.sha256(
            np.ascontiguousarray(targets, dtype="<i8").tobytes()
        ).hexdigest(),
        "group_codes_sha256": hashlib.sha256(
            np.ascontiguousarray(group_codes, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    _write_json(run_dir / "prediction_lock.json", lock)
    (run_dir / "prediction_lock.sha256").write_text(
        _sha256_file(run_dir / "prediction_lock.json")
        + "  prediction_lock.json\n",
        encoding="ascii",
    )
    reloaded = {}
    with np.load(predictions_path, allow_pickle=False) as persisted:
        if hashlib.sha256(
            np.ascontiguousarray(persisted["targets"], dtype="<i8").tobytes()
        ).hexdigest() != lock["targets_sha256"]:
            raise RuntimeError("formal locked targets changed on reload")
        for condition in _FORMAL_CONDITIONS:
            reloaded[condition] = {
                model: persisted[f"{model}__{condition}"].copy()
                for model in _BASE_MODELS
            }
            reloaded[condition].update(
                {
                    "targets": persisted["targets"].copy(),
                    "sample_order_sha256": outputs[condition][
                        "sample_order_sha256"
                    ],
                }
            )
    if _formal_output_hashes(reloaded) != lock["conditions"]:
        raise RuntimeError("formal locked probabilities changed on reload")
    return reloaded, tuple(group_ids), lock


def _all_retention(metrics):
    clean = metrics["clean"]["macro_f1"]
    return {
        condition: float(values["macro_f1"] / clean)
        for condition, values in metrics.items()
        if condition != "clean"
    }


def _rare_class_analysis(base, locked):
    rows = []
    zero_out = []
    for condition in _FORMAL_CONDITIONS:
        xgb_recall = base[condition]["xgboost"]["per_class_recall"]
        locked_recall = locked[condition]["per_class_recall"]
        deltas = {
            label: float(locked_recall[label] - xgb_recall[label])
            for label in NF_CSE_RARE_COARSE_LABELS
        }
        for label in NF_CSE_RARE_COARSE_LABELS:
            if xgb_recall[label] >= 0.10 and locked_recall[label] <= 0.0:
                zero_out.append({"condition": condition, "label": label})
        rows.append(
            {
                "condition": condition,
                "mean_recall_delta_vs_xgboost": float(np.mean(tuple(deltas.values()))),
                "per_class_recall_delta_vs_xgboost": deltas,
            }
        )
    return {
        "rare_labels": list(NF_CSE_RARE_COARSE_LABELS),
        "definition": "official full-source coarse prevalence below 1%",
        "conditions": rows,
        "zero_out_failures": zero_out,
        "descriptive_only_no_formal_reselection": True,
    }


def _evaluate_setting(root: Path, setting, *, bootstrap_resamples=2000):
    git, selection_dir, config, selection, global_lock_path = (
        _verify_selection_for_test(root, setting)
    )
    claim_path = selection_dir / "formal_test_access.claim.json"
    completed_path = selection_dir / "formal_test_access.completed.json"
    if claim_path.exists() or completed_path.exists():
        raise RuntimeError("formal test access was already claimed for this setting")
    tokenizer = TokenizerMetadata.from_dict(
        _read_json(selection_dir / "tokenizer_metadata.json")
    )
    encoder = _encoder_from_lock(_read_json(selection_dir / "encoder.json"), tokenizer)
    scratch_config = load_config(
        config["scratch_config_path"], expected_workflow="finetune"
    )
    pretrained_config = load_config(
        config["pretrained_config_path"], expected_workflow="finetune"
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    scratch_model, scratch_payload, scratch_adaptation = _restore_model(
        scratch_config, config["scratch_checkpoint_path"], tokenizer, device
    )
    pretrained_model, pretrained_payload, pretrained_adaptation = _restore_model(
        pretrained_config,
        config["pretrained_checkpoint_path"],
        tokenizer,
        device,
    )
    initialization = _validate_initialization_provenance(
        scratch_payload,
        pretrained_payload,
        config["source_pretrain_checkpoint_path"],
    )
    ft_model, ft_payload, ft_provenance = _restore_ft_model(
        config["ft_run_dir"], config["ft_checkpoint_path"], tokenizer, device
    )
    xgb = XGBClassifier()
    xgb.load_model(str(selection_dir / "xgboost_model.json"))
    run_dir = _run_dir(
        _RESULTS_ROOT,
        f"nf-cse-formal-test-{setting['setting_id']}",
        git["commit"],
    )
    claim = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        **setting,
        "status": "claimed_before_test_materialization",
        "access_count_for_setting": 1,
        "all_validation_selections_lock_sha256": _sha256_file(global_lock_path),
        "selection_sha256": _sha256_file(selection_dir / "selection.json"),
        "model_and_artifact_preflight_completed": True,
        "post_access_model_or_policy_selection_allowed": False,
        "run_dir": str(run_dir),
    }
    _exclusive_json(claim_path, claim)

    started = time.perf_counter()
    dataset = load_nf_cse_development_samples(
        config["split_dir"],
        config["preprocessor_dir"],
        data_root=_DATA_ROOT,
        split_names=("test",),
    )
    if dataset.preprocessor_hash != tokenizer.preprocessor_hash:
        raise RuntimeError("formal test preprocessing differs from validation lock")
    manifest = SplitManifest.from_json(
        (Path(config["split_dir"]) / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    test_samples = dataset.splits["test"]
    group_ids = tuple(
        manifest.sample_groups[sample.sample_id] for sample in test_samples
    )
    test_conditions, missingness = _formal_conditions(
        test_samples, config["missing_seed"]
    )
    outputs, timings = _collect_formal_outputs(
        test_conditions,
        encoder=encoder,
        xgboost_model=xgb,
        scratch_model=scratch_model,
        pretrained_model=pretrained_model,
        ft_model=ft_model,
        tokenizer=tokenizer,
        device=device,
        schemaflow_batch_size=scratch_config.training["batch_size"],
        ft_batch_size=256,
    )
    locked_outputs, locked_groups, prediction_lock = (
        _write_formal_prediction_lock(run_dir, outputs, group_ids)
    )

    base = _formal_base_metrics(locked_outputs, confirmatory=True)
    fixed_policies = _fixed_policies()
    fixed = _policy_metrics(
        locked_outputs, fixed_policies, confirmatory=True
    )
    locked_policies = selection["locked_policy"]["policies"]
    locked = _policy_metrics(
        locked_outputs, locked_policies, confirmatory=True
    )
    locked_probabilities = _policy_probabilities(
        locked_outputs, locked_policies
    )
    xgb_probabilities = {
        condition: values["xgboost"]
        for condition, values in locked_outputs.items()
    }
    bootstrap = paired_cluster_bootstrap_macro_f1(
        locked_outputs,
        locked_probabilities,
        xgb_probabilities,
        locked_groups,
        class_count=len(NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels),
        resamples=bootstrap_resamples,
        seed=setting["seed"],
    )
    metrics = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "evaluation_scope": "formal_sealed_test_once_per_setting",
        "task": "shared_coarse_nf_cse_supported_five_class",
        "label_names": list(NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels),
        "base_models": base,
        "fixed_a0": fixed,
        "locked_policy_name": selection["locked_policy"]["name"],
        "a0_mr": locked,
        "retention": {
            **{
                model: _all_retention(
                    {
                        condition: values[model]
                        for condition, values in base.items()
                    }
                )
                for model in _BASE_MODELS
            },
            "fixed_a0": _all_retention(fixed),
            "a0_mr": _all_retention(locked),
        },
        "macro_f1_deltas": {
            condition: {
                "a0_mr_minus_xgboost": locked[condition]["macro_f1"]
                - base[condition]["xgboost"]["macro_f1"],
                "a0_mr_minus_fixed_a0": locked[condition]["macro_f1"]
                - fixed[condition]["macro_f1"],
            }
            for condition in _FORMAL_CONDITIONS
        },
        "rare_class_analysis": _rare_class_analysis(base, locked),
    }
    result_config = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "workflow": "nf_cse_a0_mr_formal_test",
        **setting,
        "test_access": "single_locked_evaluation_for_this_setting",
        "prior_confirmation_access_on_same_base_test": True,
        "all_validation_selections_locked_before_access": True,
        "prediction_lock_before_metrics": True,
        "selection_dir": str(selection_dir),
        "selection_sha256": _sha256_file(selection_dir / "selection.json"),
        "global_selection_lock_sha256": _sha256_file(global_lock_path),
        "test_rows": len(test_samples),
        "conditions": list(_FORMAL_CONDITIONS),
        "locked_policies": locked_policies,
        "fixed_a0_policies": fixed_policies,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": setting["seed"],
        "device": "cuda",
        "git": git,
    }
    _write_json(run_dir / "config.json", result_config)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "bootstrap.json", bootstrap)
    _write_json(run_dir / "missingness.json", {"test": missingness})
    _write_json(run_dir / "test_access.claim.json", claim)
    _write_json(
        run_dir / "checkpoint_provenance.json",
        {
            "scratch": _checkpoint_provenance(
                config["scratch_checkpoint_path"],
                scratch_payload,
                scratch_adaptation,
            ),
            "pretrained": _checkpoint_provenance(
                config["pretrained_checkpoint_path"],
                pretrained_payload,
                pretrained_adaptation,
            ),
            "pretrained_initialization": initialization,
            "ft_transformer": {
                "path": config["ft_checkpoint_path"],
                "sha256": _sha256_file(Path(config["ft_checkpoint_path"])),
                "epoch": ft_payload["epoch"],
                "global_step": ft_payload["global_step"],
                "git_commit": ft_payload["git_commit"],
                "evaluation_scope": ft_payload["config"]["evaluation_scope"],
                "formal_provenance": ft_provenance,
            },
        },
    )
    _write_json(run_dir / "environment.json", _environment())
    _write_json(
        run_dir / "efficiency.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "test_inference_seconds": timings,
            "prediction_archive_bytes": prediction_lock[
                "prediction_archive_bytes"
            ],
            "prediction_archive_sha256": prediction_lock[
                "predictions_sha256"
            ],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
    )
    shutil.copyfile(selection_dir / "selection.json", run_dir / "selection.lock.json")
    shutil.copyfile(selection_dir / "overlap_audit.json", run_dir / "overlap_audit.json")
    completion = {
        **claim,
        "status": "completed_prediction_locked_and_metrics_published",
        "prediction_lock_sha256": _sha256_file(run_dir / "prediction_lock.json"),
        "metrics_sha256": _sha256_file(run_dir / "metrics.json"),
    }
    _write_json(run_dir / "test_access.completed.json", completion)
    (run_dir / ".incomplete").unlink()
    _exclusive_json(completed_path, completion)
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "metrics": str(run_dir / "metrics.json"),
    }


def _record_test_step(root, setting, result, wall):
    state = _load_state(root, setting)
    run_dir = Path(result["run_dir"]).resolve()
    state["steps"]["formal_test"] = {
        "status": "completed",
        "wall_seconds": wall,
        "run_dir": str(run_dir),
        "artifacts": {
            "metrics": _artifact(run_dir / "metrics.json"),
            "config": _artifact(run_dir / "config.json"),
            "bootstrap": _artifact(run_dir / "bootstrap.json"),
            "prediction_lock": _artifact(run_dir / "prediction_lock.json"),
            "test_access_completed": _artifact(
                run_dir / "test_access.completed.json"
            ),
        },
    }
    _save_state(root, setting, state)


def _recover_completed_test(selection_dir: Path):
    claim_path = selection_dir / "formal_test_access.claim.json"
    completed_path = selection_dir / "formal_test_access.completed.json"
    if not claim_path.exists():
        return None
    claim = _read_json(claim_path)
    run_dir = Path(claim.get("run_dir", "")).resolve()
    required = (
        "metrics.json",
        "config.json",
        "bootstrap.json",
        "prediction_lock.json",
        "test_access.completed.json",
    )
    if not (run_dir / ".incomplete").exists() and all(
        (run_dir / name).is_file() for name in required
    ):
        completion = _read_json(run_dir / "test_access.completed.json")
        if (
            completion.get("prediction_lock_sha256")
            != _sha256_file(run_dir / "prediction_lock.json")
            or completion.get("metrics_sha256")
            != _sha256_file(run_dir / "metrics.json")
        ):
            raise RuntimeError("completed formal test recovery hashes changed")
        if completed_path.exists():
            if _read_json(completed_path) != completion:
                raise RuntimeError("selection-side formal completion receipt changed")
        else:
            _exclusive_json(completed_path, completion)
        return {"status": "completed", "run_dir": str(run_dir)}
    raise RuntimeError(
        "formal test access was claimed but did not complete; sealed test cannot be rerun"
    )


def _evaluate_all(root: Path, *, bootstrap_resamples=2000):
    _select_all(root)
    _lock_all_selections(root)
    for setting in _settings():
        state = _load_state(root, setting)
        if _completed_step(state, "formal_test") is not None:
            continue
        selection_step = _completed_step(state, "validation_selection")
        recovered = _recover_completed_test(Path(selection_step["run_dir"]))
        if recovered is not None:
            _record_test_step(root, setting, recovered, 0.0)
            continue
        result, wall = _run_logged(
            (
                sys.executable,
                "-m",
                "scripts.run_nf_cse_formal_matrix",
                "evaluate",
                "--matrix-root",
                root,
                "--seed",
                setting["seed"],
                "--train-percent",
                setting["train_percent"],
                "--bootstrap-resamples",
                bootstrap_resamples,
            ),
            root / "logs" / f"{setting['setting_id']}-formal-test.log",
        )
        _record_test_step(root, setting, result, wall)
    return {"status": "all_nine_formal_tests_completed", "settings": 9}


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _summarize(root: Path):
    reports = []
    total_state_wall = 0.0
    for setting in _settings():
        state = _load_state(root, setting)
        test = _completed_step(state, "formal_test")
        if test is None:
            raise RuntimeError("formal matrix summary requires all nine tests")
        metrics_path = Path(test["artifacts"]["metrics"]["path"])
        metrics = _read_json(metrics_path)
        selection_step = _completed_step(state, "validation_selection")
        selection = _read_json(
            Path(selection_step["artifacts"]["selection"]["path"])
        )
        reports.append(
            {
                **setting,
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256_file(metrics_path),
                "metrics": metrics,
                "selection": selection,
            }
        )
        total_state_wall += sum(
            float(step.get("wall_seconds", 0.0))
            for step in state["steps"].values()
        )
    model_names = _BASE_MODELS + ("fixed_a0", "a0_mr")

    def metric_for(report, condition, model):
        if model in _BASE_MODELS:
            return report["metrics"]["base_models"][condition][model]
        return report["metrics"][model][condition]

    aggregates = {
        condition: {
            model: {
                metric: _distribution(
                    [
                        metric_for(report, condition, model)[metric]
                        for report in reports
                    ]
                )
                for metric in ("macro_f1", "mcc", "nll", "brier", "ece")
            }
            for model in model_names
        }
        for condition in _FORMAL_CONDITIONS
    }
    deltas = {
        condition: {
            name: _distribution(
                [
                    report["metrics"]["macro_f1_deltas"][condition][name]
                    for report in reports
                ]
            )
            for name in ("a0_mr_minus_xgboost", "a0_mr_minus_fixed_a0")
        }
        for condition in _FORMAL_CONDITIONS
    }
    random_mean_deltas = [
        float(
            np.mean(
                [
                    report["metrics"]["macro_f1_deltas"][condition][
                        "a0_mr_minus_xgboost"
                    ]
                    for condition in _CONDITIONS
                ]
            )
        )
        for report in reports
    ]
    policy_counts = {
        condition: dict(
            Counter(
                f"{report['selection']['locked_policy']['policies'][condition]['method']}@"
                f"{report['selection']['locked_policy']['policies'][condition]['alpha']:.2f}"
                for report in reports
            )
        )
        for condition in _FORMAL_CONDITIONS
    }
    zero_out = [
        {
            "setting_id": report["setting_id"],
            **failure,
        }
        for report in reports
        for failure in report["metrics"]["rare_class_analysis"][
            "zero_out_failures"
        ]
    ]
    prediction_bytes = sum(
        Path(report["metrics_path"]).parent.joinpath("predictions.npz").stat().st_size
        for report in reports
    )
    payload = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "status": "formal_matrix_completed",
        "setting_count": 9,
        "seeds": list(_SEEDS),
        "train_percents": list(_TRAIN_PERCENTS),
        "conditions": list(_FORMAL_CONDITIONS),
        "models": list(model_names),
        "aggregates": aggregates,
        "macro_f1_deltas": deltas,
        "random_condition_mean_a0_mr_minus_xgboost": _distribution(
            random_mean_deltas
        ),
        "settings_with_positive_random_condition_mean_gain": int(
            sum(value > 0 for value in random_mean_deltas)
        ),
        "locked_policy_frequency": policy_counts,
        "rare_zero_out_failures": zero_out,
        "calibration_metrics_are_descriptive_not_selection_targets": True,
        "formal_results_do_not_trigger_post_test_reselection": True,
        "total_recorded_job_wall_seconds": total_state_wall,
        "prediction_archive_total_bytes": prediction_bytes,
        "result_artifacts": [
            {
                "setting_id": report["setting_id"],
                "metrics_path": report["metrics_path"],
                "metrics_sha256": report["metrics_sha256"],
            }
            for report in reports
        ],
    }
    _write_json(root / "formal_matrix_summary.json", payload)
    (root / "formal_matrix_summary.sha256").write_text(
        _sha256_file(root / "formal_matrix_summary.json")
        + "  formal_matrix_summary.json\n",
        encoding="ascii",
    )
    marker = root / ".incomplete"
    if marker.exists():
        marker.unlink()
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the sealed NF-CSE formal A0-MR matrix"
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in (
        "plan",
        "prepare",
        "train",
        "select-all",
        "lock-selections",
        "evaluate-all",
        "summarize",
        "run",
    ):
        child = subparsers.add_parser(phase)
        child.add_argument(
            "--matrix-root", type=Path, default=_DEFAULT_MATRIX_ROOT
        )
        if phase in {"evaluate-all", "run"}:
            child.add_argument("--bootstrap-resamples", type=int, default=2000)
    for phase in ("select", "evaluate"):
        child = subparsers.add_parser(phase)
        child.add_argument(
            "--matrix-root", type=Path, default=_DEFAULT_MATRIX_ROOT
        )
        child.add_argument("--seed", type=int, required=True)
        child.add_argument("--train-percent", type=int, required=True)
        if phase == "evaluate":
            child.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser


def run(arguments):
    root = _matrix_root(arguments.matrix_root)
    if arguments.phase == "plan":
        return _plan(root)
    if arguments.phase == "prepare":
        return _prepare(root)
    if arguments.phase == "train":
        return _train_all(root)
    if arguments.phase == "select-all":
        return _select_all(root)
    if arguments.phase == "lock-selections":
        return _lock_all_selections(root)
    if arguments.phase == "select":
        return _select_setting(
            root, _setting(arguments.seed, arguments.train_percent)
        )
    if arguments.phase == "evaluate":
        return _evaluate_setting(
            root,
            _setting(arguments.seed, arguments.train_percent),
            bootstrap_resamples=arguments.bootstrap_resamples,
        )
    if arguments.phase == "evaluate-all":
        return _evaluate_all(
            root, bootstrap_resamples=arguments.bootstrap_resamples
        )
    if arguments.phase == "summarize":
        return _summarize(root)
    if arguments.phase == "run":
        _evaluate_all(root, bootstrap_resamples=arguments.bootstrap_resamples)
        return _summarize(root)
    raise ValueError(f"unsupported formal phase: {arguments.phase}")


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
