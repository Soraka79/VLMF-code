#!/usr/bin/env python3
"""Run the frozen A0-MR matrix under the declared target-exposure design."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from xgboost import XGBClassifier

from flowfm.config import load_config
from flowfm.data.formal_cohort import (
    get_formal_dataset_spec,
    validate_formal_overlap_contract,
)
from flowfm.data.formal_dataset import (
    TRAIN_PERCENTS,
    load_formal_preprocessor,
    load_formal_selection_samples,
    materialize_formal_test_samples,
    open_formal_sealed_test_rows,
    publish_formal_training_view,
)
from flowfm.data.identity import stable_hash
from flowfm.data.missingness import (
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_deterministic_field_missingness,
    apply_structured_field_group_missingness,
)
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import (
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.training.a0_mr_protocol import (
    FORMAL_CONDITIONS,
    RANDOM_CONDITIONS,
    STRUCTURED_CONDITIONS,
    fixed_a0_policies,
    frozen_protocol_hash,
    frozen_protocol_payload,
    policy_probabilities,
    select_a0_mr_validation,
)
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.baselines import (
    TabularFeatureEncoder,
    inverse_frequency_sample_weights,
)
from flowfm.training.checkpoints import load_checkpoint
from flowfm.training.confirmatory_statistics import (
    MAX_RARE_MEAN_RECALL_DROP,
    confirmatory_metric_payload,
    paired_cluster_bootstrap_macro_f1,
)
from flowfm.training.neural_baselines import build_neural_model
from flowfm.training.probability_fusion import probability_sha256
from flowfm.training.safe_fusion import (
    apply_scalar_temperature,
    fit_scalar_temperature,
)
from flowfm.training.supervised import (
    _build_model,
    _configure_adaptation,
    _model_data_metadata,
)
from scripts.run_nf_cse_a0_mr_confirmation import (
    _encoder_from_lock,
    _model_visible_samples,
    _sha256_file,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_REPO_ROOT = _PROJECT_ROOT / "repository"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_AUDIT_ROOT = _RESULTS_ROOT / "cross-dataset-formal-audits-v1"
_RESTART_AUDIT_ROOT = _RESULTS_ROOT / "cross-dataset-formal-restart-audits-v1"
_MATRIX_BASE = _RESULTS_ROOT / "cross-dataset-formal-matrix-v1"
_RESTART_MATRIX_BASE = _RESULTS_ROOT / "cross-dataset-formal-restart-matrix-v1"
_VIEW_BASES = MappingProxyType(
    {
        "nf_bot": _PROJECT_ROOT / "data" / "formal_views_v1" / "nf_bot",
        "nf_cse": _PROJECT_ROOT / "data" / "formal_views_v1" / "nf_cse",
        "nf_unsw_restart": (
            _PROJECT_ROOT
            / "data"
            / "formal_views_restart_v1"
            / "nf_unsw_restart"
        ),
        "nf_ton": _PROJECT_ROOT / "data" / "formal_views_v2" / "nf_ton",
    }
)
_DEFAULT_SOURCE_CHECKPOINT = (
    _RESULTS_ROOT
    / "schemaflow-nf-unsw-pretrain-smoke-20260722T054417924340Z-"
    "9965a490-7de8cdfb4f42/checkpoints/epoch-0001.pt"
)
_RESTART_SOURCE_LOCK = (
    _RESULTS_ROOT
    / "cross-dataset-formal-source-pretrain-v1"
    / "nf_cse_for_nf_unsw_restart"
    / "source_checkpoint_lock.json"
)
_MODEL_SEEDS = (17, 42, 20260725)
_NEURAL_MODELS = (
    "schemaflow_scratch",
    "schemaflow_pretrained",
    "ft_transformer",
)
_BASE_MODELS = (
    "xgboost",
    "lightgbm",
    "ft_transformer",
    "schemaflow_scratch",
    "schemaflow_pretrained",
)
_ALL_MODELS = _BASE_MODELS + ("fixed_a0", "a0_mr")
_FT_MODEL_CONFIG = {
    "d_token": 64,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}
_PROTOCOL_VERSION = "cross-dataset-a0-mr-formal-matrix-v1"
_RECOVERY_PROTOCOL_VERSION = "sealed-test-pre-materialization-recovery-v1"
_RECOVERY_DATASET = "nf_unsw_restart"


def _settings():
    return tuple(
        {
            "setting_id": f"pct-{percent:02d}-seed-{seed}",
            "train_percent": percent,
            "model_seed": seed,
        }
        for percent in TRAIN_PERCENTS
        for seed in _MODEL_SEEDS
    )


def _setting(percent, seed):
    for value in _settings():
        if value["train_percent"] == percent and value["model_seed"] == seed:
            return value
    raise ValueError("train-percent/model-seed is outside the formal matrix")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    return write_json_snapshot(path, payload)


def _artifact(path):
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise RuntimeError(f"required formal artifact is missing: {value}")
    return {
        "path": str(value),
        "sha256": _sha256_file(value),
        "bytes": value.stat().st_size,
    }


def _git_state(repo_root=None):
    repo_root = _REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("formal matrix requires a clean AutoDL worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": False}


def _matrix_root(path):
    return validate_output_path(
        path, enforce_autodl_data_disk=True, data_root=_DATA_ROOT
    )


def _cohort_path(spec):
    return (
        _PROJECT_ROOT / "data" / "splits" / spec.dataset_id / spec.cohort_name
    )


def _audit_path(dataset):
    base = _RESTART_AUDIT_ROOT if dataset == "nf_unsw_restart" else _AUDIT_ROOT
    return base / dataset


def _default_matrix_root(dataset):
    base = _RESTART_MATRIX_BASE if dataset == "nf_unsw_restart" else _MATRIX_BASE
    return base / dataset


def _source_inputs(dataset):
    if dataset != "nf_unsw_restart":
        return _DEFAULT_SOURCE_CHECKPOINT, None
    if not _RESTART_SOURCE_LOCK.is_file():
        raise RuntimeError(
            "NF-UNSW restart requires the locked CSE-only source checkpoint"
        )
    lock = _read_json(_RESTART_SOURCE_LOCK)
    try:
        checkpoint_artifact = lock["checkpoint"]
        checkpoint = Path(checkpoint_artifact["path"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("restart source checkpoint lock is malformed") from exc
    if (
        (_RESTART_SOURCE_LOCK.parent / ".incomplete").exists()
        or lock.get("source_dataset_key") != "nf_cse"
        or lock.get("source_dataset_id") != "NF-CSE-CIC-IDS2018-v2"
        or lock.get("target_dataset_key") != "nf_unsw_restart"
        or lock.get("target_dataset_id") != "NF-UNSW-NB15-v2"
        or lock.get("target_excluded") is not True
        or lock.get("source_test_files_read") is not False
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != checkpoint_artifact.get("bytes")
        or _sha256_file(checkpoint) != checkpoint_artifact.get("sha256")
    ):
        raise RuntimeError("restart source checkpoint lock is invalid")
    return checkpoint, _RESTART_SOURCE_LOCK


def _view_path(dataset, percent):
    try:
        base = _VIEW_BASES[dataset]
    except KeyError as error:
        raise ValueError(
            f"formal training views are not configured for {dataset}"
        ) from error
    return base / f"train-{percent:02d}pct"


def _state_path(root, setting):
    return root / "states" / f"{setting['setting_id']}.json"


def _load_state(root, setting):
    path = _state_path(root, setting)
    if not path.exists():
        return {
            "format_version": 1,
            "protocol_version": _PROTOCOL_VERSION,
            **setting,
            "steps": {},
        }
    payload = _read_json(path)
    if any(payload.get(name) != value for name, value in setting.items()):
        raise RuntimeError("formal state belongs to another setting")
    return payload


def _save_state(root, setting, state):
    path = _state_path(root, setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, state)


def _completed_step(state, name):
    step = state["steps"].get(name)
    if not isinstance(step, dict) or step.get("status") != "completed":
        return None
    for artifact in step.get("artifacts", {}).values():
        path = Path(artifact["path"])
        if not path.is_file() or _sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"completed artifact changed: {path}")
    return step


def _budget(dataset):
    return {
        "format_version": 1,
        "dataset": dataset,
        "settings": 9,
        "model_seeds": list(_MODEL_SEEDS),
        "train_percents": list(TRAIN_PERCENTS),
        "conditions": list(FORMAL_CONDITIONS),
        "nominal_jobs": {
            "schemaflow_scratch": 9,
            "schemaflow_target_excluded_pretrained": 9,
            "ft_transformer": 9,
            "xgboost_lightgbm_validation_selection": 9,
            "single_batched_sealed_test": 1,
            "total": 37,
        },
        "estimated_wall_hours": {"lower": 12.0, "upper": 24.0},
        "estimated_peak_gpu_gib": 4.0,
        "estimated_peak_system_memory_gib": 64.0,
        "estimated_incremental_storage_gib": 24.0,
        "lightgbm_requirement": "lightgbm==4.6.0",
        "gpu_over_two_hours_is_not_a_stop_condition": True,
        "stop_conditions": [
            "data_integrity_failure",
            "leakage_or_fairness_contract_failure",
            "insufficient_disk",
            "provenance_failure",
        ],
    }


def _plan(dataset, root):
    spec = get_formal_dataset_spec(dataset)
    cohort = _cohort_path(spec)
    audit_dir = _audit_path(dataset)
    source_checkpoint, source_lock = _source_inputs(dataset)
    required = {
        "cohort_config": cohort / "config.snapshot.json",
        "cohort_summary": cohort / "cohort_summary.json",
        "cohort_overlap_audit": cohort / "overlap_audit.json",
        "cohort_seal": cohort / "sealed_test.json",
        "selection_records": cohort / "selection_records.jsonl",
        "selection_manifest": cohort / "selection_manifest.json",
        "selection_features": cohort / "selection_features.parquet",
        "history_exposure": audit_dir / "historical_exposure.json",
        "feasibility": audit_dir / "feasibility.json",
        "data_manifest": cohort / "data_manifest.json",
        "formal_requirement": _REPO_ROOT / "requirements-formal.txt",
        "source_checkpoint": source_checkpoint,
    }
    if source_lock is not None:
        required["source_checkpoint_lock"] = source_lock
    if (cohort / ".incomplete").exists():
        raise RuntimeError("formal cohort publication is still incomplete")
    artifacts = {name: _artifact(path) for name, path in required.items()}
    feasibility = _read_json(required["feasibility"])
    if feasibility.get("status") != spec.feasibility_status:
        raise RuntimeError("dataset feasibility does not permit a formal matrix")
    overlap = _read_json(required["cohort_overlap_audit"])
    try:
        validate_formal_overlap_contract(spec, overlap)
    except ValueError as exc:
        raise RuntimeError("formal cohort overlap contract is invalid") from exc
    seal = _read_json(required["cohort_seal"])
    seal_historical_independence = seal.get(
        "historically_independent",
        True if spec.requires_zero_historical_overlap else None,
    )
    if (
        seal.get("status") != spec.seal_status
        or seal.get("opened_for_evaluation") is not False
        or seal.get("created_without_model_predictions") is not True
        or seal_historical_independence
        is not spec.requires_zero_historical_overlap
    ):
        raise RuntimeError("formal cohort test seal is invalid before planning")
    path = root / "matrix_plan.json"
    if path.exists():
        existing = _read_json(path)
        if (
            existing.get("protocol_version") != _PROTOCOL_VERSION
            or existing.get("dataset") != dataset
            or existing.get("matrix_root") != str(root)
            or existing.get("a0_mr_protocol_hash") != frozen_protocol_hash()
            or existing.get("git") != _git_state()
        ):
            raise RuntimeError("existing formal matrix plan changed")
        for name, artifact in existing.get("artifacts", {}).items():
            current = Path(artifact["path"])
            if (
                not current.is_file()
                or current.stat().st_size != artifact["bytes"]
                or _sha256_file(current) != artifact["sha256"]
            ):
                raise RuntimeError(f"planned formal artifact changed: {name}")
        _write_json(root / "budget.json", existing["budget"])
        return existing
    disk = shutil.disk_usage(_DATA_ROOT)
    budget = _budget(dataset)
    if disk.free < int(budget["estimated_incremental_storage_gib"] * 1.5 * 1024**3):
        raise RuntimeError("insufficient disk for the formal matrix safety margin")
    payload = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "status": "planned_before_formal_training",
        "dataset": dataset,
        "dataset_id": spec.dataset_id,
        "matrix_root": str(root),
        "cohort_dir": str(cohort),
        "history_audit_dir": str(audit_dir),
        "claim_scope": feasibility.get("claim_scope"),
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent": spec.requires_zero_historical_overlap,
        "git": _git_state(),
        "settings": list(_settings()),
        "task_label_space": spec.task_label_space.to_dict(),
        "rare_coarse_labels": list(spec.rare_coarse_labels),
        "rare_definition": "official full-source task-coarse prevalence below 1%",
        "conditions": list(FORMAL_CONDITIONS),
        "models": list(_ALL_MODELS),
        "rare_coarse_labels": list(
            get_formal_dataset_spec(dataset).rare_coarse_labels
        ),
        "a0_mr_protocol": frozen_protocol_payload(),
        "a0_mr_protocol_hash": frozen_protocol_hash(),
        "missingness_seed": spec.split_seed,
        "same_missing_assignments_for_all_models_and_model_seeds": True,
        "all_nine_validation_selections_lock_before_single_test_open": True,
        "sealed_test_open_count": 1,
        "artifacts": artifacts,
        "lightgbm_installed_at_plan": importlib.util.find_spec("lightgbm") is not None,
        "disk_at_plan": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "budget": budget,
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".incomplete"
    if not marker.exists():
        marker.write_text("cross-dataset formal matrix incomplete\n", encoding="ascii")
    _write_json(path, payload)
    _write_json(root / "budget.json", budget)
    return payload


def _prepare(dataset, root):
    plan = _plan(dataset, root)
    rows = []
    for percent in TRAIN_PERCENTS:
        path = _view_path(dataset, percent)
        if not path.exists():
            publish_formal_training_view(
                dataset,
                plan["cohort_dir"],
                path,
                train_percent=percent,
                data_root=_DATA_ROOT,
            )
        summary = _read_json(path / "training_view_summary.json")
        if (
            summary.get("train_percent") != percent
            or summary.get("sealed_test_files_read") is not False
            or summary.get("test_labels_or_statistics_present") is not False
        ):
            raise RuntimeError("formal training view violates sealed scope")
        rows.append(
            {
                "train_percent": percent,
                "view_dir": str(path),
                "summary": _artifact(path / "training_view_summary.json"),
                "manifest": _artifact(path / "selection_manifest.json"),
                "records": _artifact(path / "selection_records.jsonl"),
                "preprocessor": _artifact(path / "preprocessor.json"),
            }
        )
    result = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "status": "all_train_validation_views_prepared",
        "views": rows,
    }
    _write_json(root / "prepare.completed.json", result)
    return result


def _run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        tuple(str(value) for value in command),
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"formal subprocess failed; see {log_path}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("formal subprocess produced no result JSON")
    return json.loads(lines[-1]), time.perf_counter() - started


def _record_neural(root, setting, model, result, wall):
    state = _load_state(root, setting)
    run_dir = Path(result["run_dir"])
    checkpoint = Path(result["checkpoint"])
    config_name = "config.json" if model == "ft_transformer" else "config.snapshot.json"
    state["steps"][model] = {
        "status": "completed",
        "wall_seconds": wall,
        "run_dir": str(run_dir),
        "artifacts": {
            "checkpoint": _artifact(checkpoint),
            "config": _artifact(run_dir / config_name),
            "metrics": _artifact(run_dir / "metrics.json"),
            "formal_provenance": _artifact(run_dir / "formal_provenance.json"),
        },
    }
    _save_state(root, setting, state)


def _train(dataset, root):
    plan = _plan(dataset, root)
    _prepare(dataset, root)
    for setting in _settings():
        for model in _NEURAL_MODELS:
            state = _load_state(root, setting)
            if _completed_step(state, model) is not None:
                continue
            result, wall = _run_logged(
                (
                    sys.executable,
                    "-m",
                    "scripts.run_formal_neural_model",
                    "--dataset",
                    dataset,
                    "--cohort-dir",
                    plan["cohort_dir"],
                    "--view-dir",
                    _view_path(dataset, setting["train_percent"]),
                    "--matrix-root",
                    root,
                    "--source-checkpoint",
                    plan["artifacts"]["source_checkpoint"]["path"],
                    "--model-seed",
                    setting["model_seed"],
                    "--train-percent",
                    setting["train_percent"],
                    "--model",
                    model,
                ),
                root / "logs" / f"{setting['setting_id']}-{model}.log",
            )
            _record_neural(root, setting, model, result, wall)
    return {"status": "all_27_neural_jobs_completed"}


def _normalize_probabilities(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("formal probabilities must be finite and non-negative")
    sums = matrix.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("formal probability row has no mass")
    return matrix / sums


def _coarse_targets(samples, label_space):
    to_id = {label: index for index, label in enumerate(label_space.coarse_labels)}
    return np.asarray([to_id[sample.coarse_label] for sample in samples], dtype=np.int64)


def _conditions(samples, seed):
    eligible = tuple(
        feature.name for feature in SCHEMAFLOW_MAIN_REGISTRY.model_features
    )
    result = {}
    reports = {}
    for condition, fraction in (
        ("clean", 0.0),
        ("missing_30pct", 0.3),
        ("missing_50pct", 0.5),
    ):
        values, report = apply_deterministic_field_missingness(
            samples,
            fraction=fraction,
            seed=seed,
            eligible_feature_ids=eligible,
        )
        result[condition] = values
        reports[condition] = report
    for group_name in STRUCTURED_MISSINGNESS_GROUPS:
        condition = f"structured__{group_name}"
        values, report = apply_structured_field_group_missingness(
            samples, group_name=group_name
        )
        result[condition] = values
        reports[condition] = report
    if tuple(result) != FORMAL_CONDITIONS:
        raise RuntimeError("formal condition order changed")
    return result, reports


def _fit_xgboost(matrix, targets, class_count, seed):
    expected = np.arange(class_count, dtype=np.int64)
    if not np.array_equal(np.unique(targets), expected):
        raise ValueError("formal training does not cover all coarse labels")
    parameters = {
        "n_estimators": 100,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": min(os.cpu_count() or 1, 15),
        "random_state": seed,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": class_count,
    }
    model = XGBClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        matrix,
        targets,
        sample_weight=inverse_frequency_sample_weights(targets),
        verbose=False,
    )
    return model, parameters, time.perf_counter() - started


def _fit_lightgbm(matrix, targets, class_count, seed):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required by requirements-formal.txt and was not installed"
        ) from exc
    parameters = {
        "n_estimators": 300,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "multiclass",
        "num_class": class_count,
        "random_state": seed,
        "n_jobs": min(os.cpu_count() or 1, 15),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    model = LGBMClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        matrix,
        targets,
        sample_weight=inverse_frequency_sample_weights(targets),
    )
    return model, parameters, time.perf_counter() - started


def _restore_schemaflow(step, samples, device):
    run_dir = Path(step["run_dir"])
    config = load_config(run_dir / "config.yaml", expected_workflow="finetune")
    model = _build_model(
        config,
        samples.tokenizer_metadata,
        samples.label_space,
        SCHEMAFLOW_MAIN_REGISTRY,
    ).to(device)
    adaptation = _configure_adaptation(
        model, config.training.get("adaptation_mode", "full")
    )
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training["learning_rate"],
        weight_decay=config.training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.training["scheduler_gamma"]
    )
    checkpoint = Path(step["artifacts"]["checkpoint"]["path"])
    payload = load_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
        restore_rng=False,
        expected_model_data_metadata=_model_data_metadata(
            SCHEMAFLOW_MAIN_REGISTRY,
            samples.tokenizer_metadata,
            samples.label_space,
        ),
    )
    metrics = payload["extra_state"].get("metrics", {})
    if metrics.get("evaluation_scope") != "validation_only" or "test" in metrics:
        raise RuntimeError("SchemaFlow checkpoint has premature test metrics")
    model.eval()
    return model, config, payload


def _restore_ft(step, samples, device):
    run_dir = Path(step["run_dir"])
    config = _read_json(run_dir / "config.json")
    provenance = _read_json(run_dir / "formal_provenance.json")
    if (
        config.get("evaluation_scope") != "validation_only"
        or provenance.get("sealed_test_files_read") is not False
        or provenance.get("test_samples_materialized") is not False
        or _read_json(run_dir / "label_space.json") != samples.label_space.to_dict()
        or _read_json(run_dir / "tokenizer_metadata.json")
        != samples.tokenizer_metadata.to_dict()
    ):
        raise RuntimeError("FT-Transformer checkpoint violates formal provenance")
    model = build_neural_model(
        "ft_transformer",
        "coarse",
        tokenizer_metadata=samples.tokenizer_metadata,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        label_space=samples.label_space,
        model_config=_FT_MODEL_CONFIG,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    checkpoint = Path(step["artifacts"]["checkpoint"]["path"])
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload = load_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        map_location=device,
        restore_rng=False,
        expected_model_data_metadata=raw["model_data_metadata"],
    )
    if "test" in payload["extra_state"].get("metrics", {}):
        raise RuntimeError("FT-Transformer checkpoint contains test metrics")
    model.eval()
    return model, config, payload


def _schemaflow_probabilities(model, samples, tokenizer, device, batch_size):
    blocks = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = collate_schemaflow_samples(
                samples[start : start + batch_size],
                tokenizer,
                registry=SCHEMAFLOW_MAIN_REGISTRY,
            ).to(device)
            blocks.append(torch.softmax(model(batch).coarse_logits, dim=1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return _normalize_probabilities(torch.cat(blocks).numpy()), time.perf_counter() - started


def _ft_probabilities(model, samples, tokenizer, device, batch_size=256):
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
    return _normalize_probabilities(torch.cat(blocks).numpy()), time.perf_counter() - started


def _tree_probabilities(model, matrix, class_count):
    values = _normalize_probabilities(model.predict_proba(matrix))
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(class_count)):
        raise RuntimeError("tree probability columns changed")
    return values


def _collect_outputs(
    conditions,
    *,
    encoder,
    xgboost,
    lightgbm,
    scratch,
    pretrained,
    ft,
    tokenizer,
    label_space,
    device,
    schemaflow_batch_size,
):
    outputs = {}
    timings = dict.fromkeys(_BASE_MODELS, 0.0)
    for condition, samples in conditions.items():
        projected = _model_visible_samples(samples)
        started = time.perf_counter()
        matrix = encoder.transform(projected)
        xgb = _tree_probabilities(xgboost, matrix, len(label_space.coarse_labels))
        timings["xgboost"] += time.perf_counter() - started
        started = time.perf_counter()
        lgb = _tree_probabilities(lightgbm, matrix, len(label_space.coarse_labels))
        timings["lightgbm"] += time.perf_counter() - started
        scratch_values, elapsed = _schemaflow_probabilities(
            scratch, samples, tokenizer, device, schemaflow_batch_size
        )
        timings["schemaflow_scratch"] += elapsed
        pretrained_values, elapsed = _schemaflow_probabilities(
            pretrained, samples, tokenizer, device, schemaflow_batch_size
        )
        timings["schemaflow_pretrained"] += elapsed
        ft_values, elapsed = _ft_probabilities(ft, samples, tokenizer, device)
        timings["ft_transformer"] += elapsed
        outputs[condition] = {
            "xgboost": xgb,
            "lightgbm": lgb,
            "schemaflow_scratch": scratch_values,
            "schemaflow_pretrained": pretrained_values,
            "ft_transformer": ft_values,
            "targets": _coarse_targets(samples, label_space),
            "sample_order_sha256": stable_hash(
                [sample.sample_id for sample in samples],
                "flowfm-formal-sample-order-v1",
            ),
        }
    return outputs, timings


def _probability_sets(outputs, selection):
    fixed = policy_probabilities(outputs, fixed_a0_policies())
    a0_mr = policy_probabilities(
        outputs, selection["locked_policy"]["policies"]
    )
    return {
        model: {
            condition: values[model] for condition, values in outputs.items()
        }
        for model in _BASE_MODELS
    } | {"fixed_a0": fixed, "a0_mr": a0_mr}


def _metrics(probabilities, outputs, label_names, *, confirmatory=False):
    metric = confirmatory_metric_payload if confirmatory else metric_payload
    return {
        condition: metric(values, outputs[condition]["targets"], label_names)
        for condition, values in probabilities.items()
    }


def _fit_temperatures(probabilities, outputs, label_names):
    locks = {}
    calibrated = {}
    metrics_before = {}
    metrics_after = {}
    for model, by_condition in probabilities.items():
        locks[model] = {}
        calibrated[model] = {}
        metrics_before[model] = _metrics(by_condition, outputs, label_names)
        metrics_after[model] = {}
        for condition, values in by_condition.items():
            target = outputs[condition]["targets"]
            lock = fit_scalar_temperature(values, target)
            current = apply_scalar_temperature(values, lock["temperature"])
            if not np.array_equal(current.argmax(axis=1), values.argmax(axis=1)):
                raise RuntimeError("scalar temperature changed argmax predictions")
            locks[model][condition] = lock
            calibrated[model][condition] = current
            metrics_after[model][condition] = metric_payload(
                current, target, label_names
            )
    return locks, calibrated, metrics_before, metrics_after


def _write_prediction_archive(path, outputs, groups):
    arrays = {"group_ids": np.asarray(groups, dtype="U64")}
    for condition, values in outputs.items():
        arrays[f"{condition}__targets"] = np.asarray(values["targets"], dtype="<i8")
        for model in _BASE_MODELS:
            arrays[f"{condition}__{model}"] = np.asarray(
                values[model], dtype="<f8"
            )
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "locked_before_metrics": True,
    }


def _select_setting(dataset, root, setting):
    plan = _plan(dataset, root)
    state = _load_state(root, setting)
    steps = {name: _completed_step(state, name) for name in _NEURAL_MODELS}
    if any(value is None for value in steps.values()):
        raise RuntimeError("all neural models must finish before validation selection")
    samples = load_formal_selection_samples(
        dataset,
        plan["cohort_dir"],
        _view_path(dataset, setting["train_percent"]),
        data_root=_DATA_ROOT,
    )
    train = _model_visible_samples(samples.splits["train"])
    targets = _coarse_targets(train, samples.label_space)
    encoder = TabularFeatureEncoder.fit(
        train,
        category_encoders=samples.tokenizer_metadata.categorical_vocabularies,
    )
    train_matrix = encoder.transform(train)
    xgboost, xgb_parameters, xgb_seconds = _fit_xgboost(
        train_matrix,
        targets,
        len(samples.label_space.coarse_labels),
        setting["model_seed"],
    )
    lightgbm, lgb_parameters, lgb_seconds = _fit_lightgbm(
        train_matrix,
        targets,
        len(samples.label_space.coarse_labels),
        setting["model_seed"],
    )
    del train_matrix
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    scratch, scratch_config, scratch_payload = _restore_schemaflow(
        steps["schemaflow_scratch"], samples, device
    )
    pretrained, pretrained_config, pretrained_payload = _restore_schemaflow(
        steps["schemaflow_pretrained"], samples, device
    )
    ft, ft_config, ft_payload = _restore_ft(
        steps["ft_transformer"], samples, device
    )
    conditions, missingness = _conditions(
        samples.splits["val"],
        get_formal_dataset_spec(dataset).split_seed,
    )
    outputs, timings = _collect_outputs(
        conditions,
        encoder=encoder,
        xgboost=xgboost,
        lightgbm=lightgbm,
        scratch=scratch,
        pretrained=pretrained,
        ft=ft,
        tokenizer=samples.tokenizer_metadata,
        label_space=samples.label_space,
        device=device,
        schemaflow_batch_size=scratch_config.training["batch_size"],
    )
    selection = select_a0_mr_validation(
        outputs, samples.label_space.coarse_labels
    )
    probabilities = _probability_sets(outputs, selection)
    temperatures, _, metrics_before, metrics_after = _fit_temperatures(
        probabilities, outputs, samples.label_space.coarse_labels
    )
    run_dir = root / "selections" / setting["setting_id"]
    if run_dir.exists():
        raise RuntimeError("formal validation selection output already exists")
    run_dir.mkdir(parents=True)
    marker = run_dir / ".incomplete"
    marker.write_text("formal validation selection incomplete\n", encoding="ascii")
    xgb_path = run_dir / "xgboost_model.json"
    xgboost.save_model(str(xgb_path))
    lgb_path = run_dir / "lightgbm_model.txt"
    lightgbm.booster_.save_model(str(lgb_path))
    _write_json(run_dir / "encoder.json", encoder.to_dict())
    _write_json(
        run_dir / "tokenizer_metadata.json", samples.tokenizer_metadata.to_dict()
    )
    _write_json(run_dir / "label_space.json", samples.label_space.to_dict())
    prediction_lock = _write_prediction_archive(
        run_dir / "validation_predictions.npz",
        outputs,
        samples.groups["val"],
    )
    _write_json(run_dir / "selection.json", selection)
    _write_json(run_dir / "temperature_lock.json", temperatures)
    _write_json(
        run_dir / "validation_metrics.json",
        {
            "format_version": 1,
            "uncalibrated": metrics_before,
            "temperature_calibrated": metrics_after,
        },
    )
    _write_json(run_dir / "missingness.json", missingness)
    _write_json(
        run_dir / "config.json",
        {
            "format_version": 1,
            "protocol_version": _PROTOCOL_VERSION,
            "dataset": dataset,
            **setting,
            "selection_scope": ["train", "validation"],
            "sealed_test_files_read": False,
            "test_samples_materialized": False,
            "a0_mr_protocol_hash": frozen_protocol_hash(),
            "missingness_seed": get_formal_dataset_spec(dataset).split_seed,
            "xgboost_parameters": xgb_parameters,
            "lightgbm_parameters": lgb_parameters,
            "temperature_fit_scope": "validation_only_after_policy_lock",
            "temperature_participates_in_macro_f1_selection": False,
            "git": _git_state(),
        },
    )
    _write_json(
        run_dir / "checkpoint_provenance.json",
        {
            "schemaflow_scratch": steps["schemaflow_scratch"]["artifacts"]["checkpoint"],
            "schemaflow_pretrained": steps["schemaflow_pretrained"]["artifacts"]["checkpoint"],
            "ft_transformer": steps["ft_transformer"]["artifacts"]["checkpoint"],
            "scratch_initialization": scratch_payload["extra_state"].get("initialization"),
            "pretrained_initialization": pretrained_payload["extra_state"].get("initialization"),
            "ft_epoch": ft_payload["epoch"],
        },
    )
    _write_json(
        run_dir / "efficiency.json",
        {
            "xgboost_fit_seconds": xgb_seconds,
            "lightgbm_fit_seconds": lgb_seconds,
            "validation_inference_seconds": timings,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "validation_prediction_archive": prediction_lock,
        },
    )
    marker.unlink()
    return {"status": "completed", "run_dir": str(run_dir)}


def _record_selection(root, setting, result, wall):
    state = _load_state(root, setting)
    run_dir = Path(result["run_dir"])
    state["steps"]["validation_selection"] = {
        "status": "completed",
        "wall_seconds": wall,
        "run_dir": str(run_dir),
        "artifacts": {
            name: _artifact(run_dir / filename)
            for name, filename in {
                "selection": "selection.json",
                "config": "config.json",
                "temperature_lock": "temperature_lock.json",
                "xgboost_model": "xgboost_model.json",
                "lightgbm_model": "lightgbm_model.txt",
                "encoder": "encoder.json",
                "tokenizer": "tokenizer_metadata.json",
                "label_space": "label_space.json",
                "validation_metrics": "validation_metrics.json",
                "validation_predictions": "validation_predictions.npz",
            }.items()
        },
    }
    _save_state(root, setting, state)


def _lock_selections(dataset, root):
    path = root / "all_validation_selections.lock.json"
    if path.exists():
        payload = _read_json(path)
        for row in payload["settings"]:
            if _sha256_file(Path(row["selection_path"])) != row["selection_sha256"]:
                raise RuntimeError("globally locked selection changed")
        return payload
    rows = []
    for setting in _settings():
        state = _load_state(root, setting)
        steps = {
            name: _completed_step(state, name)
            for name in (*_NEURAL_MODELS, "validation_selection")
        }
        if any(step is None for step in steps.values()):
            raise RuntimeError("all validation selections must finish before lock")
        step = steps["validation_selection"]
        selection_path = Path(step["artifacts"]["selection"]["path"])
        rows.append(
            {
                **setting,
                "selection_dir": step["run_dir"],
                "selection_path": str(selection_path),
                "selection_sha256": _sha256_file(selection_path),
                "temperature_lock_sha256": step["artifacts"]["temperature_lock"]["sha256"],
                "locked_steps": {
                    name: {
                        "run_dir": current["run_dir"],
                        "artifacts": current["artifacts"],
                    }
                    for name, current in steps.items()
                },
            }
        )
    payload = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "dataset": dataset,
        "status": "all_nine_validation_selections_locked_before_test",
        "setting_count": 9,
        "a0_mr_protocol_hash": frozen_protocol_hash(),
        "settings": rows,
    }
    _write_json(path, payload)
    return payload


def _select_all(dataset, root):
    _train(dataset, root)
    for setting in _settings():
        state = _load_state(root, setting)
        if _completed_step(state, "validation_selection") is not None:
            continue
        result, wall = _run_logged(
            (
                sys.executable,
                "-m",
                "scripts.run_cross_dataset_formal_matrix",
                "select",
                "--dataset",
                dataset,
                "--matrix-root",
                root,
                "--train-percent",
                setting["train_percent"],
                "--model-seed",
                setting["model_seed"],
            ),
            root / "logs" / f"{setting['setting_id']}-selection.log",
        )
        _record_selection(root, setting, result, wall)
    return _lock_selections(dataset, root)


class _LightGBMBoosterAdapter:
    def __init__(self, model_path, class_count):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("LightGBM is required for sealed evaluation") from exc
        self.booster = lgb.Booster(model_file=str(model_path))
        self.classes_ = np.arange(class_count, dtype=np.int64)

    def predict_proba(self, matrix):
        return self.booster.predict(matrix)


def _open_claim(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        raise


def _verify_declared_artifact(record, path, description):
    if not isinstance(record, dict) or record != _artifact(path):
        raise RuntimeError(f"recovery {description} artifact changed")


def _expected_recovery_sealed_artifacts(plan, seal):
    cohort = Path(plan["cohort_dir"]).resolve()
    return {
        "records": {
            "path": str(cohort / "sealed_test_records.jsonl"),
            "sha256": seal["sealed_test_records_sha256"],
        },
        "manifest": {
            "path": str(cohort / "sealed_test_manifest.json"),
            "sha256": seal["sealed_test_manifest_sha256"],
        },
        "features": {
            "path": str(cohort / "sealed_test_features.parquet"),
            "sha256": seal["sealed_test_features_sha256"],
        },
    }


def _validate_claimed_test_recovery(
    dataset,
    root,
    *,
    global_lock_path,
    global_lock,
    claim_path,
    incident_path,
    bootstrap_resamples,
):
    """Validate the one documented recovery without rereading sealed data."""

    root = Path(root).resolve()
    claim_path = Path(claim_path).resolve()
    incident_path = Path(incident_path).resolve()
    if dataset != _RECOVERY_DATASET:
        raise RuntimeError("claimed-test recovery is restricted to NF-UNSW restart")
    if (root / "sealed_test").exists():
        raise RuntimeError("claimed-test recovery found existing sealed-test results")
    if (root / "sealed_test_access.recovery.json").exists():
        raise RuntimeError("claimed-test recovery was already claimed")

    plan_path = root / "matrix_plan.json"
    if not plan_path.is_file():
        raise RuntimeError("claimed-test recovery matrix plan is missing")
    plan = _read_json(plan_path)
    claim = _read_json(claim_path)
    expected_claim = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "dataset": dataset,
        "status": "single_test_open_claimed",
        "global_selection_lock_sha256": _sha256_file(global_lock_path),
        "setting_count": global_lock["setting_count"],
        "sealed_test_open_budget": 1,
        "bootstrap_resamples": bootstrap_resamples,
        "git": plan.get("git"),
    }
    if (
        plan.get("protocol_version") != _PROTOCOL_VERSION
        or plan.get("dataset") != dataset
        or claim != expected_claim
    ):
        raise RuntimeError("original sealed-test claim does not match the matrix plan")

    incident = _read_json(incident_path)
    spec = get_formal_dataset_spec(dataset)
    expected_access = {
        "diagnostic_hash_verification_count": 1,
        "pre_materialization_failure_count": 1,
        "sealed_artifact_hash_read_attempt_count": 2,
        "sealed_artifact_hash_read_counting_unit": (
            "one_complete_records_manifest_features_sha256_pass"
        ),
        "sealed_test_claim_count": 1,
        "successful_test_materialization_count": 0,
    }
    expected_authorization = {
        "additional_sealed_artifact_hash_read_attempt_count": 1,
        "automatic_retry_allowed": False,
        "maximum_successful_test_materialization_count": 1,
        "recovery_scope": (
            "resume_original_claim_after_pre_materialization_contract_failure"
        ),
    }
    expected_failure = {
        "exception": "ValueError: sealed formal test lock is invalid",
        "expected_dataset_seal_status": spec.seal_status,
        "failed_evaluator_git": plan["git"],
        "labels_or_statistics_observed": False,
        "manifest_parsed": False,
        "observed_dataset_seal_status": spec.seal_status,
        "predictions_generated": False,
        "reader_hard_coded_status": "sealed_before_any_model_selection",
        "records_materialized": False,
        "stage": "sealed_artifact_hash_verification_before_manifest_parse",
    }
    if (
        incident.get("format_version") != 1
        or incident.get("protocol_version") != _RECOVERY_PROTOCOL_VERSION
        or incident.get("status") != "authorized_for_single_explicit_recovery"
        or incident.get("dataset") != dataset
        or incident.get("dataset_id") != spec.dataset_id
        or incident.get("historically_independent") is not False
        or incident.get("eligible_for_primary_cross_dataset_inference") is not False
        or incident.get("access_accounting_before_recovery") != expected_access
        or incident.get("authorized_recovery") != expected_authorization
        or incident.get("failure") != expected_failure
    ):
        raise RuntimeError("claimed-test recovery incident contract is invalid")

    artifacts = incident.get("artifacts", {})
    _verify_declared_artifact(
        artifacts.get("original_claim"), claim_path, "original claim"
    )
    _verify_declared_artifact(
        artifacts.get("global_selection_lock"),
        global_lock_path,
        "global selection lock",
    )
    seal_path = Path(plan["artifacts"]["cohort_seal"]["path"])
    _verify_declared_artifact(
        artifacts.get("cohort_seal"), seal_path, "cohort seal"
    )
    pipeline_record = artifacts.get("pipeline_log")
    if not isinstance(pipeline_record, dict):
        raise RuntimeError("recovery pipeline log artifact is missing")
    _verify_declared_artifact(
        pipeline_record, pipeline_record.get("path", ""), "pipeline log"
    )
    seal = _read_json(seal_path)
    if (
        seal.get("status") != spec.seal_status
        or incident.get("sealed_artifacts")
        != _expected_recovery_sealed_artifacts(plan, seal)
    ):
        raise RuntimeError("recovery sealed-artifact declarations changed")
    return claim, incident


def _recovery_runtime_git_state(incident_path):
    repo_root = Path(__file__).resolve().parents[1]
    incident_path = Path(incident_path).resolve()
    try:
        relative = incident_path.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError("recovery incident must be inside the runtime worktree") from exc
    state = _git_state(repo_root)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode:
        raise RuntimeError("recovery incident must be committed before execution")
    return state


def _begin_claimed_test_recovery(
    dataset,
    root,
    *,
    global_lock_path,
    global_lock,
    claim_path,
    incident_path,
    bootstrap_resamples,
):
    claim, incident = _validate_claimed_test_recovery(
        dataset,
        root,
        global_lock_path=global_lock_path,
        global_lock=global_lock,
        claim_path=claim_path,
        incident_path=incident_path,
        bootstrap_resamples=bootstrap_resamples,
    )
    recovery_path = Path(root) / "sealed_test_access.recovery.json"
    event = {
        "format_version": 1,
        "protocol_version": _RECOVERY_PROTOCOL_VERSION,
        "dataset": dataset,
        "status": "explicit_pre_materialization_recovery_claimed",
        "automatic_retry_allowed": False,
        "original_claim": _artifact(claim_path),
        "global_selection_lock": _artifact(global_lock_path),
        "incident": _artifact(incident_path),
        "access_accounting_before_recovery": incident[
            "access_accounting_before_recovery"
        ],
        "authorized_recovery": incident["authorized_recovery"],
        "planned_evaluator_git": claim["git"],
        "recovery_code_git": _recovery_runtime_git_state(incident_path),
        "sealed_results_present_before_recovery": False,
    }
    _open_claim(recovery_path, event)
    return claim, recovery_path


def _verify_global_lock(dataset, root):
    path = root / "all_validation_selections.lock.json"
    if not path.is_file():
        raise RuntimeError("sealed test is blocked until all selections lock")
    payload = _read_json(path)
    if (
        payload.get("dataset") != dataset
        or payload.get("status")
        != "all_nine_validation_selections_locked_before_test"
        or payload.get("setting_count") != 9
        or payload.get("a0_mr_protocol_hash") != frozen_protocol_hash()
    ):
        raise RuntimeError("global validation lock is invalid")
    expected_settings = {
        row["setting_id"]: row for row in _settings()
    }
    locked_settings = {
        row.get("setting_id"): row for row in payload["settings"]
    }
    if locked_settings.keys() != expected_settings.keys():
        raise RuntimeError("global validation lock settings changed")
    for setting_id, row in locked_settings.items():
        expected = expected_settings[setting_id]
        if any(row.get(name) != expected[name] for name in expected):
            raise RuntimeError("global validation lock setting metadata changed")
        if _sha256_file(Path(row["selection_path"])) != row["selection_sha256"]:
            raise RuntimeError("a globally locked selection changed")
        temperature_path = Path(row["selection_dir"]) / "temperature_lock.json"
        if _sha256_file(temperature_path) != row["temperature_lock_sha256"]:
            raise RuntimeError("a globally locked temperature changed")
        state = _load_state(root, expected)
        locked_steps = row.get("locked_steps", {})
        expected_step_names = (*_NEURAL_MODELS, "validation_selection")
        if set(locked_steps) != set(expected_step_names):
            raise RuntimeError("global lock does not cover every model artifact")
        for name in expected_step_names:
            current = _completed_step(state, name)
            if current is None:
                raise RuntimeError("a globally locked model is no longer complete")
            locked = locked_steps[name]
            if (
                locked.get("run_dir") != current.get("run_dir")
                or locked.get("artifacts") != current.get("artifacts")
            ):
                raise RuntimeError("a globally locked model artifact changed")
    if importlib.util.find_spec("lightgbm") is None:
        raise RuntimeError("LightGBM must be installed before sealed test access")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available before sealed test access")
    return path, payload


def _apply_temperature_locks(probabilities, locks):
    calibrated = {}
    for model, by_condition in probabilities.items():
        if set(locks.get(model, {})) != set(by_condition):
            raise RuntimeError("temperature lock does not match conditions")
        calibrated[model] = {}
        for condition, values in by_condition.items():
            current = apply_scalar_temperature(
                values, locks[model][condition]["temperature"]
            )
            if not np.array_equal(current.argmax(axis=1), values.argmax(axis=1)):
                raise RuntimeError("locked scalar temperature changed argmax")
            calibrated[model][condition] = current
    return calibrated


def _write_test_prediction_lock(
    run_dir,
    outputs,
    uncalibrated,
    calibrated,
    group_ids,
):
    targets = np.asarray(outputs["clean"]["targets"], dtype="<i8")
    arrays = {
        "group_ids": np.asarray(group_ids, dtype="U64"),
        "targets": targets,
    }
    hashes = {}
    for calibration_name, models in (
        ("uncalibrated", uncalibrated),
        ("temperature_calibrated", calibrated),
    ):
        for model, by_condition in models.items():
            for condition, values in by_condition.items():
                key = f"{calibration_name}__{model}__{condition}"
                matrix = np.asarray(values, dtype="<f8")
                arrays[key] = matrix
                hashes[key] = probability_sha256(matrix)
    path = run_dir / "predictions.npz"
    np.savez_compressed(path, **arrays)
    archive_hash = _sha256_file(path)
    with np.load(path, allow_pickle=False) as locked:
        if set(locked.files) != set(arrays):
            raise RuntimeError("test prediction archive keys changed")
        if not np.array_equal(locked["targets"], targets):
            raise RuntimeError("test targets changed during prediction lock")
        for key, expected in hashes.items():
            if probability_sha256(locked[key]) != expected:
                raise RuntimeError("test probabilities changed during lock")
    lock = {
        "format_version": 1,
        "status": "all_setting_predictions_locked_before_metrics",
        "prediction_archive_sha256": archive_hash,
        "prediction_archive_bytes": path.stat().st_size,
        "sample_count": len(targets),
        "condition_order": list(FORMAL_CONDITIONS),
        "model_order": list(_ALL_MODELS),
        "probability_hashes": hashes,
    }
    _write_json(run_dir / "prediction_lock.json", lock)
    return lock


def _confirmatory_metrics(probabilities, outputs, label_names):
    return {
        model: _metrics(
            by_condition,
            outputs,
            label_names,
            confirmatory=True,
        )
        for model, by_condition in probabilities.items()
    }


def _retention(metrics):
    clean = metrics["clean"]["macro_f1"]
    return {
        condition: float(values["macro_f1"] / clean)
        for condition, values in metrics.items()
        if condition != "clean"
    }


def _rare_analysis(spec, metrics):
    failures = []
    rows = []
    random_deltas = []
    for condition in FORMAL_CONDITIONS:
        xgb = metrics["xgboost"][condition]["per_class_recall"]
        a0_mr = metrics["a0_mr"][condition]["per_class_recall"]
        deltas = {
            label: float(a0_mr[label] - xgb[label])
            for label in spec.rare_coarse_labels
        }
        for label in spec.rare_coarse_labels:
            if condition in RANDOM_CONDITIONS:
                random_deltas.append(deltas[label])
            if xgb[label] >= 0.10 and a0_mr[label] <= 0.0:
                failures.append(
                    {
                        "type": "rare_recall_zero_out",
                        "condition": condition,
                        "label": label,
                    }
                )
        rows.append(
            {
                "condition": condition,
                "per_class_recall_delta_vs_xgboost": deltas,
                "mean_recall_delta_vs_xgboost": (
                    float(np.mean(tuple(deltas.values()))) if deltas else None
                ),
            }
        )
    random_mean = (
        float(np.mean(random_deltas)) if random_deltas else None
    )
    mean_drop_failure = (
        random_mean is not None
        and random_mean < -MAX_RARE_MEAN_RECALL_DROP
    )
    collapse_failures = list(failures)
    if mean_drop_failure:
        collapse_failures.append(
            {
                "type": "rare_random_mean_recall_drop",
                "mean_delta_vs_xgboost": random_mean,
                "maximum_allowed_drop": MAX_RARE_MEAN_RECALL_DROP,
            }
        )
    return {
        "rare_labels": list(spec.rare_coarse_labels),
        "definition": "official full-source task-coarse prevalence below 1%",
        "no_eligible_rare_coarse_labels": not spec.rare_coarse_labels,
        "conditions": rows,
        "random_mean_recall_delta_vs_xgboost": random_mean,
        "maximum_random_mean_recall_drop": MAX_RARE_MEAN_RECALL_DROP,
        "random_mean_recall_drop_failure": mean_drop_failure,
        "zero_out_failures": failures,
        "collapse_rule": "xgboost_recall_at_least_0p10_and_a0_mr_zero",
        "collapse_failures": collapse_failures,
        "no_rare_class_recall_collapse": not collapse_failures,
    }


def _bootstrap_pairs(outputs, probabilities, groups, class_count, seed, resamples):
    result = {}
    for baseline in ("xgboost", "lightgbm", "fixed_a0"):
        result[f"a0_mr_vs_{baseline}"] = paired_cluster_bootstrap_macro_f1(
            outputs,
            probabilities["a0_mr"],
            probabilities[baseline],
            groups,
            class_count=class_count,
            resamples=resamples,
            seed=seed,
        )
    return result


def _test_setting(
    dataset,
    root,
    setting,
    sealed_rows,
    *,
    bootstrap_resamples,
):
    spec = get_formal_dataset_spec(dataset)
    state = _load_state(root, setting)
    neural = {name: _completed_step(state, name) for name in _NEURAL_MODELS}
    selection_step = _completed_step(state, "validation_selection")
    if any(value is None for value in neural.values()) or selection_step is None:
        raise RuntimeError("setting is not globally ready for sealed test")
    selection_dir = Path(selection_step["run_dir"])
    selection = _read_json(selection_dir / "selection.json")
    if selection.get("protocol_hash") != frozen_protocol_hash():
        raise RuntimeError("setting A0-MR selection protocol changed")
    selection_samples = load_formal_selection_samples(
        dataset,
        _cohort_path(spec),
        _view_path(dataset, setting["train_percent"]),
        data_root=_DATA_ROOT,
    )
    test_samples = materialize_formal_test_samples(
        sealed_rows, selection_samples.preprocessor
    )
    tokenizer = TokenizerMetadata.from_dict(
        _read_json(selection_dir / "tokenizer_metadata.json")
    )
    if tokenizer != selection_samples.tokenizer_metadata:
        raise RuntimeError("locked tokenizer changed before sealed test")
    encoder = _encoder_from_lock(
        _read_json(selection_dir / "encoder.json"), tokenizer
    )
    xgboost = XGBClassifier()
    xgboost.load_model(str(selection_dir / "xgboost_model.json"))
    lightgbm = _LightGBMBoosterAdapter(
        selection_dir / "lightgbm_model.txt",
        len(selection_samples.label_space.coarse_labels),
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    scratch, scratch_config, _ = _restore_schemaflow(
        neural["schemaflow_scratch"], selection_samples, device
    )
    pretrained, _, _ = _restore_schemaflow(
        neural["schemaflow_pretrained"], selection_samples, device
    )
    ft, _, _ = _restore_ft(neural["ft_transformer"], selection_samples, device)
    conditions, missingness = _conditions(test_samples, spec.split_seed)
    outputs, timings = _collect_outputs(
        conditions,
        encoder=encoder,
        xgboost=xgboost,
        lightgbm=lightgbm,
        scratch=scratch,
        pretrained=pretrained,
        ft=ft,
        tokenizer=tokenizer,
        label_space=selection_samples.label_space,
        device=device,
        schemaflow_batch_size=scratch_config.training["batch_size"],
    )
    uncalibrated = _probability_sets(outputs, selection)
    temperature_locks = _read_json(selection_dir / "temperature_lock.json")
    calibrated = _apply_temperature_locks(uncalibrated, temperature_locks)
    run_dir = root / "sealed_test" / setting["setting_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    marker = run_dir / ".incomplete"
    marker.write_text("sealed setting output incomplete\n", encoding="ascii")
    prediction_lock = _write_test_prediction_lock(
        run_dir, outputs, uncalibrated, calibrated, sealed_rows.group_ids
    )
    uncalibrated_metrics = _confirmatory_metrics(
        uncalibrated, outputs, selection_samples.label_space.coarse_labels
    )
    calibrated_metrics = _confirmatory_metrics(
        calibrated, outputs, selection_samples.label_space.coarse_labels
    )
    bootstraps = _bootstrap_pairs(
        outputs,
        uncalibrated,
        sealed_rows.group_ids,
        len(selection_samples.label_space.coarse_labels),
        setting["model_seed"],
        bootstrap_resamples,
    )
    deltas = {
        condition: {
            "a0_mr_minus_xgboost": (
                uncalibrated_metrics["a0_mr"][condition]["macro_f1"]
                - uncalibrated_metrics["xgboost"][condition]["macro_f1"]
            ),
            "a0_mr_minus_lightgbm": (
                uncalibrated_metrics["a0_mr"][condition]["macro_f1"]
                - uncalibrated_metrics["lightgbm"][condition]["macro_f1"]
            ),
            "a0_mr_minus_fixed_a0": (
                uncalibrated_metrics["a0_mr"][condition]["macro_f1"]
                - uncalibrated_metrics["fixed_a0"][condition]["macro_f1"]
            ),
        }
        for condition in FORMAL_CONDITIONS
    }
    metrics = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "dataset": dataset,
        **setting,
        "label_names": list(selection_samples.label_space.coarse_labels),
        "uncalibrated": uncalibrated_metrics,
        "temperature_calibrated": calibrated_metrics,
        "retention": {
            model: _retention(values)
            for model, values in uncalibrated_metrics.items()
        },
        "macro_f1_deltas": deltas,
        "rare_class_analysis": _rare_analysis(spec, uncalibrated_metrics),
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "bootstrap.json", bootstraps)
    _write_json(run_dir / "missingness.json", missingness)
    _write_json(
        run_dir / "config.json",
        {
            "format_version": 1,
            "protocol_version": _PROTOCOL_VERSION,
            "dataset": dataset,
            **setting,
            "evaluation_scope": "single_batched_sealed_test_process",
            "selection_dir": str(selection_dir),
            "selection_sha256": _sha256_file(selection_dir / "selection.json"),
            "temperature_lock_sha256": _sha256_file(
                selection_dir / "temperature_lock.json"
            ),
            "prediction_locked_before_metrics": True,
            "bootstrap_resamples": bootstrap_resamples,
            "git": _git_state(),
        },
    )
    _write_json(
        run_dir / "efficiency.json",
        {
            "test_inference_seconds": timings,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "prediction_archive_bytes": prediction_lock[
                "prediction_archive_bytes"
            ],
            "prediction_archive_sha256": prediction_lock[
                "prediction_archive_sha256"
            ],
        },
    )
    marker.unlink()
    return {
        "setting": setting,
        "run_dir": str(run_dir),
        "metrics": _artifact(run_dir / "metrics.json"),
        "bootstrap": _artifact(run_dir / "bootstrap.json"),
        "prediction_lock": _artifact(run_dir / "prediction_lock.json"),
        "efficiency": _artifact(run_dir / "efficiency.json"),
    }


def _evaluate(
    dataset,
    root,
    *,
    bootstrap_resamples=2000,
    resume_claimed_test=False,
    recovery_incident=None,
):
    global_lock_path, global_lock = _verify_global_lock(dataset, root)
    completed_path = root / "sealed_test_access.completed.json"
    if completed_path.exists():
        completed = _read_json(completed_path)
        if completed.get("status") != "single_test_open_completed":
            raise RuntimeError("sealed test completion receipt is invalid")
        return completed
    claim_path = root / "sealed_test_access.claim.json"
    recovery_path = None
    if claim_path.exists():
        if not resume_claimed_test or recovery_incident is None:
            raise RuntimeError(
                "sealed test access was previously claimed but did not complete; "
                "automatic rerun is forbidden"
            )
        claim, recovery_path = _begin_claimed_test_recovery(
            dataset,
            root,
            global_lock_path=global_lock_path,
            global_lock=global_lock,
            claim_path=claim_path,
            incident_path=recovery_incident,
            bootstrap_resamples=bootstrap_resamples,
        )
    else:
        if resume_claimed_test or recovery_incident is not None:
            raise RuntimeError("claimed-test recovery requires an existing claim")
        claim = {
            "format_version": 1,
            "protocol_version": _PROTOCOL_VERSION,
            "dataset": dataset,
            "status": "single_test_open_claimed",
            "global_selection_lock_sha256": _sha256_file(global_lock_path),
            "setting_count": global_lock["setting_count"],
            "sealed_test_open_budget": 1,
            "bootstrap_resamples": bootstrap_resamples,
            "git": _git_state(),
        }
        _open_claim(claim_path, claim)
    spec = get_formal_dataset_spec(dataset)
    started = time.perf_counter()
    sealed_rows = open_formal_sealed_test_rows(
        dataset, _cohort_path(spec), data_root=_DATA_ROOT
    )
    results = []
    for setting in _settings():
        results.append(
            _test_setting(
                dataset,
                root,
                setting,
                sealed_rows,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
    receipt = {
        **claim,
        "status": "single_test_open_completed",
        "sealed_test_open_count": 1,
        "sealed_test_claim_count": 1,
        "sealed_artifact_hash_read_attempt_count": (
            3 if recovery_path is not None else 1
        ),
        "sealed_artifact_hash_read_counting_unit": (
            "one_complete_records_manifest_features_sha256_pass"
        ),
        "pre_materialization_failure_count": (
            1 if recovery_path is not None else 0
        ),
        "diagnostic_hash_verification_count": (
            1 if recovery_path is not None else 0
        ),
        "successful_test_materialization_count": 1,
        "sealed_artifact_hashes": {
            "seal": sealed_rows.seal_sha256,
            "records": sealed_rows.records_sha256,
            "manifest": sealed_rows.manifest_sha256,
            "features": sealed_rows.features_sha256,
        },
        "sample_count": len(sealed_rows.sample_order),
        "setting_results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    if recovery_path is not None:
        receipt["pre_materialization_recovery"] = _artifact(recovery_path)
    _write_json(completed_path, receipt)
    return receipt


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _calibration_summary(
    nll_before,
    nll_after,
    brier_before,
    brier_after,
    ece_before,
    ece_after,
):
    return {
        "mean_nll_before": float(nll_before),
        "mean_nll_after": float(nll_after),
        "mean_nll_improved": bool(nll_after < nll_before),
        "mean_brier_before": float(brier_before),
        "mean_brier_after": float(brier_after),
        "mean_brier_improved": bool(brier_after < brier_before),
        "mean_ece_before": float(ece_before),
        "mean_ece_after": float(ece_after),
        "mean_ece_not_worse": bool(ece_after <= ece_before + 1e-12),
    }


def _summarize(dataset, root, *, bootstrap_resamples=2000):
    receipt = _evaluate(
        dataset, root, bootstrap_resamples=bootstrap_resamples
    )
    reports = []
    for row in receipt["setting_results"]:
        reports.append(
            {
                **row["setting"],
                "metrics_path": row["metrics"]["path"],
                "metrics": _read_json(row["metrics"]["path"]),
                "bootstrap": _read_json(row["bootstrap"]["path"]),
                "efficiency": _read_json(row["efficiency"]["path"]),
            }
        )

    def value(report, model, condition, metric, calibrated=False):
        section = (
            "temperature_calibrated" if calibrated else "uncalibrated"
        )
        return report["metrics"][section][model][condition][metric]

    aggregates = {
        calibration: {
            condition: {
                model: {
                    metric: _distribution(
                        [
                            value(
                                report,
                                model,
                                condition,
                                metric,
                                calibrated=(calibration == "temperature_calibrated"),
                            )
                            for report in reports
                        ]
                    )
                    for metric in ("macro_f1", "mcc", "nll", "brier", "ece")
                }
                for model in _ALL_MODELS
            }
            for condition in FORMAL_CONDITIONS
        }
        for calibration in ("uncalibrated", "temperature_calibrated")
    }
    delta_names = (
        "a0_mr_minus_xgboost",
        "a0_mr_minus_lightgbm",
        "a0_mr_minus_fixed_a0",
    )
    deltas = {
        condition: {
            name: _distribution(
                [
                    report["metrics"]["macro_f1_deltas"][condition][name]
                    for report in reports
                ]
            )
            for name in delta_names
        }
        for condition in FORMAL_CONDITIONS
    }
    retention = {
        model: {
            condition: _distribution(
                [
                    report["metrics"]["retention"][model][condition]
                    for report in reports
                ]
            )
            for condition in FORMAL_CONDITIONS
            if condition != "clean"
        }
        for model in _ALL_MODELS
    }
    label_names = tuple(reports[0]["metrics"]["label_names"])
    per_class_recall = {
        condition: {
            model: {
                label: _distribution(
                    [
                        report["metrics"]["uncalibrated"][model][condition][
                            "per_class_recall"
                        ][label]
                        for report in reports
                    ]
                )
                for label in label_names
            }
            for model in _ALL_MODELS
        }
        for condition in FORMAL_CONDITIONS
    }
    setting_random_gains = {
        report["setting_id"]: float(
            np.mean(
                [
                    report["metrics"]["macro_f1_deltas"][condition][
                        "a0_mr_minus_xgboost"
                    ]
                    for condition in RANDOM_CONDITIONS
                ]
            )
        )
        for report in reports
    }
    seed_stability = {}
    for percent in TRAIN_PERCENTS:
        values = [
            setting_random_gains[f"pct-{percent:02d}-seed-{seed}"]
            for seed in _MODEL_SEEDS
        ]
        positive = sum(current >= -1e-12 for current in values)
        seed_stability[str(percent)] = {
            "gains": dict(zip(map(str, _MODEL_SEEDS), values)),
            "nonnegative_count": positive,
            "at_least_two_of_three_direction_consistent": positive >= 2,
        }
    rare_zero_out_failures = [
        {"setting_id": report["setting_id"], **failure}
        for report in reports
        for failure in report["metrics"]["rare_class_analysis"][
            "zero_out_failures"
        ]
    ]
    rare_collapse_failures = [
        {"setting_id": report["setting_id"], **failure}
        for report in reports
        for failure in report["metrics"]["rare_class_analysis"][
            "collapse_failures"
        ]
    ]
    random_mean_gain = float(np.mean(tuple(setting_random_gains.values())))
    missing_vs_fixed = {
        condition: deltas[condition]["a0_mr_minus_fixed_a0"]["mean"]
        for condition in ("missing_30pct", "missing_50pct")
    }
    calibration_before = np.mean(
        [
            value(report, "a0_mr", condition, metric)
            for report in reports
            for condition in FORMAL_CONDITIONS
            for metric in ("nll",)
        ]
    )
    calibration_after = np.mean(
        [
            value(report, "a0_mr", condition, "nll", calibrated=True)
            for report in reports
            for condition in FORMAL_CONDITIONS
        ]
    )
    brier_before = np.mean(
        [
            value(report, "a0_mr", condition, "brier")
            for report in reports
            for condition in FORMAL_CONDITIONS
        ]
    )
    brier_after = np.mean(
        [
            value(report, "a0_mr", condition, "brier", calibrated=True)
            for report in reports
            for condition in FORMAL_CONDITIONS
        ]
    )
    ece_before = np.mean(
        [
            value(report, "a0_mr", condition, "ece")
            for report in reports
            for condition in FORMAL_CONDITIONS
        ]
    )
    ece_after = np.mean(
        [
            value(report, "a0_mr", condition, "ece", calibrated=True)
            for report in reports
            for condition in FORMAL_CONDITIONS
        ]
    )
    sample_condition_count = [
        report["metrics"]["uncalibrated"]["a0_mr"]["clean"][
            "sample_count"
        ]
        * len(FORMAL_CONDITIONS)
        for report in reports
    ]
    inference_efficiency = {
        model: {
            "seconds": _distribution(
                [
                    report["efficiency"]["test_inference_seconds"][model]
                    for report in reports
                ]
            ),
            "records_conditions_per_second": _distribution(
                [
                    count
                    / report["efficiency"]["test_inference_seconds"][model]
                    for count, report in zip(sample_condition_count, reports)
                ]
            ),
        }
        for model in _BASE_MODELS
    }
    training_wall = {name: [] for name in (*_NEURAL_MODELS, "validation_selection")}
    validation_fit_seconds = {"xgboost": [], "lightgbm": []}
    for setting in _settings():
        state = _load_state(root, setting)
        for name in training_wall:
            step = _completed_step(state, name)
            if step is None:
                raise RuntimeError("completed summary is missing a formal job")
            training_wall[name].append(step["wall_seconds"])
        selection_efficiency = _read_json(
            Path(state["steps"]["validation_selection"]["run_dir"])
            / "efficiency.json"
        )
        validation_fit_seconds["xgboost"].append(
            selection_efficiency["xgboost_fit_seconds"]
        )
        validation_fit_seconds["lightgbm"].append(
            selection_efficiency["lightgbm_fit_seconds"]
        )
    efficiency = {
        "training_and_selection_wall_seconds": {
            name: _distribution(values)
            for name, values in training_wall.items()
        },
        "validation_tree_fit_seconds": {
            name: _distribution(values)
            for name, values in validation_fit_seconds.items()
        },
        "sealed_test_inference": inference_efficiency,
        "sealed_test_peak_gpu_memory_bytes": _distribution(
            [report["efficiency"]["peak_gpu_memory_bytes"] for report in reports]
        ),
        "sealed_test_peak_system_memory_bytes": _distribution(
            [
                report["efficiency"]["peak_system_memory_bytes"]
                for report in reports
            ]
        ),
        "prediction_archive_bytes": _distribution(
            [report["efficiency"]["prediction_archive_bytes"] for report in reports]
        ),
        "total_recorded_training_and_selection_wall_seconds": float(
            sum(sum(values) for values in training_wall.values())
        ),
    }
    payload = {
        "format_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "status": "formal_matrix_completed",
        "dataset": dataset,
        "setting_count": 9,
        "model_seeds": list(_MODEL_SEEDS),
        "train_percents": list(TRAIN_PERCENTS),
        "conditions": list(FORMAL_CONDITIONS),
        "models": list(_ALL_MODELS),
        "aggregates": aggregates,
        "macro_f1_deltas": deltas,
        "retention_aggregates": retention,
        "per_class_recall_aggregates": per_class_recall,
        "random_condition_mean_a0_mr_minus_xgboost": {
            **_distribution(tuple(setting_random_gains.values())),
            "by_setting": setting_random_gains,
        },
        "random_direction_nonnegative": random_mean_gain >= -1e-12,
        "missing_conditions_mean_better_than_fixed_a0": {
            condition: value > 0.0 for condition, value in missing_vs_fixed.items()
        },
        "model_seed_stability": seed_stability,
        "at_least_two_of_three_seeds_consistent_for_all_train_percents": all(
            value["at_least_two_of_three_direction_consistent"]
            for value in seed_stability.values()
        ),
        "rare_zero_out_failures": rare_zero_out_failures,
        "rare_collapse_failures": rare_collapse_failures,
        "no_rare_class_recall_collapse": not rare_collapse_failures,
        "a0_mr_calibration": _calibration_summary(
            calibration_before,
            calibration_after,
            brier_before,
            brier_after,
            ece_before,
            ece_after,
        ),
        "efficiency": efficiency,
        "sealed_test_access_receipt": _artifact(
            root / "sealed_test_access.completed.json"
        ),
        "result_artifacts": [
            {
                "setting_id": report["setting_id"],
                "metrics": _artifact(report["metrics_path"]),
                "bootstrap": _artifact(
                    Path(report["metrics_path"]).parent / "bootstrap.json"
                ),
                "efficiency": _artifact(
                    Path(report["metrics_path"]).parent / "efficiency.json"
                ),
                "prediction_lock": _artifact(
                    Path(report["metrics_path"]).parent / "prediction_lock.json"
                ),
            }
            for report in reports
        ],
    }
    _write_json(root / "formal_matrix_summary.json", payload)
    marker = root / ".incomplete"
    if marker.exists():
        marker.unlink()
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one cross-dataset A0-MR formal matrix"
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in (
        "plan",
        "prepare",
        "train",
        "select-all",
        "lock-selections",
        "evaluate",
        "summarize",
        "run",
    ):
        child = subparsers.add_parser(phase)
        child.add_argument(
            "--dataset",
            choices=("nf_bot", "nf_cse", "nf_unsw_restart", "nf_ton"),
            required=True,
        )
        child.add_argument("--matrix-root", type=Path)
        if phase in {"evaluate", "summarize", "run"}:
            child.add_argument("--bootstrap-resamples", type=int, default=2000)
        if phase == "evaluate":
            child.add_argument("--resume-claimed-test", action="store_true")
            child.add_argument("--recovery-incident", type=Path)
    child = subparsers.add_parser("select")
    child.add_argument(
        "--dataset",
        choices=("nf_bot", "nf_cse", "nf_unsw_restart", "nf_ton"),
        required=True,
    )
    child.add_argument("--matrix-root", type=Path)
    child.add_argument("--train-percent", type=int, choices=TRAIN_PERCENTS, required=True)
    child.add_argument("--model-seed", type=int, choices=_MODEL_SEEDS, required=True)
    return parser


def run(arguments):
    root = _matrix_root(
        arguments.matrix_root or _default_matrix_root(arguments.dataset)
    )
    if arguments.phase == "plan":
        return _plan(arguments.dataset, root)
    if arguments.phase == "prepare":
        return _prepare(arguments.dataset, root)
    if arguments.phase == "train":
        return _train(arguments.dataset, root)
    if arguments.phase == "select":
        return _select_setting(
            arguments.dataset,
            root,
            _setting(arguments.train_percent, arguments.model_seed),
        )
    if arguments.phase == "select-all":
        return _select_all(arguments.dataset, root)
    if arguments.phase == "lock-selections":
        return _lock_selections(arguments.dataset, root)
    if arguments.phase == "evaluate":
        return _evaluate(
            arguments.dataset,
            root,
            bootstrap_resamples=arguments.bootstrap_resamples,
            resume_claimed_test=arguments.resume_claimed_test,
            recovery_incident=arguments.recovery_incident,
        )
    if arguments.phase == "summarize":
        return _summarize(
            arguments.dataset,
            root,
            bootstrap_resamples=arguments.bootstrap_resamples,
        )
    _select_all(arguments.dataset, root)
    _evaluate(
        arguments.dataset,
        root,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    return _summarize(
        arguments.dataset,
        root,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
