#!/usr/bin/env python3
"""Validation-lock and sealed-test confirmation of A0-MR on NF-CSE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
import sklearn
import torch
import xgboost
from xgboost import XGBClassifier

from flowfm.config import load_config
from flowfm.data.labels import NF_CSE_LABEL_SPACE
from flowfm.data.missingness import apply_deterministic_field_missingness
from flowfm.data.nf_cse_confirmatory import SUPPORTED_COARSE_LABELS
from flowfm.data.nf_cse_dataset import load_nf_cse_development_samples
from flowfm.data.nf_cse_few_shot import (
    SELECTION_MANIFEST_FILENAME,
    SELECTION_RECORDS_FILENAME,
    SELECTION_SUMMARY_FILENAME,
)
from flowfm.data.nf_cse_preprocessing import (
    SELECTION_FEATURES_FILENAME,
    load_nf_cse_preprocessor,
)
from flowfm.data.nf_cse_tokenization import build_nf_cse_tokenizer_metadata
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import (
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.data.splits import SplitManifest
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.baselines import (
    TabularFeatureEncoder,
    inverse_frequency_sample_weights,
)
from flowfm.training.checkpoints import load_checkpoint
from flowfm.training.confirmatory_statistics import (
    MAX_RARE_MEAN_RECALL_DROP,
    MIN_MEAN_XGBOOST_GAIN,
    NF_CSE_RARE_COARSE_LABELS,
    confirmatory_metric_payload,
    evaluate_confirmatory_gate,
    paired_cluster_bootstrap_macro_f1,
)
from flowfm.training.probability_fusion import (
    default_alpha_grid,
    pool_probabilities,
    probability_sha256,
    select_fusion,
)
from flowfm.training.supervised import (
    _build_model,
    _configure_adaptation,
    _model_data_metadata,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_DEFAULT_OUTPUT_ROOT = _DATA_ROOT / "Res-AE-Attn" / "results"
_CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
_FRACTIONS = (0.0, 0.3, 0.5)
_FIXED_A0 = {
    condition: {"method": "linear", "alpha": 0.55, "selected_on": "frozen"}
    for condition in _CONDITIONS
}
_NF_TON_REFERENCE = {
    "clean": {"method": "linear", "alpha": 0.55},
    "missing_30pct": {"method": "log", "alpha": 0.85},
    "missing_50pct": {"method": "log", "alpha": 0.70},
}
_RARE_DEFINITION = "official full-source coarse prevalence below 1%"


def _args():
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    select = phases.add_parser("select")
    select.add_argument("--scratch-config", required=True, type=Path)
    select.add_argument("--scratch-checkpoint", required=True, type=Path)
    select.add_argument("--pretrained-config", required=True, type=Path)
    select.add_argument("--pretrained-checkpoint", required=True, type=Path)
    select.add_argument("--source-pretrain-checkpoint", required=True, type=Path)
    select.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT, type=Path)
    select.add_argument("--seed", default=17, type=int)
    select.add_argument("--missing-seed", default=17, type=int)
    select.add_argument("--alpha-step", default=0.05, type=float)
    select.add_argument("--candidate-min-mean-gain", default=0.005, type=float)
    select.add_argument("--pool-min-mean-gain", default=0.002, type=float)
    select.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    confirm = phases.add_parser("confirm")
    confirm.add_argument("--selection-dir", required=True, type=Path)
    confirm.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT, type=Path)
    confirm.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    confirm.add_argument("--bootstrap-resamples", default=2000, type=int)
    confirm.add_argument("--bootstrap-seed", default=20260725, type=int)
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values, dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sample_order_sha256(samples) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_state(repo_dir: Path):
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise RuntimeError("confirmation experiments require a clean Git worktree")
    return {"commit": commit, "dirty": False, "status": []}


def _output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(_DATA_ROOT):
        raise SystemExit("output-root must be below /root/autodl-tmp")
    return resolved


def _run_dir(root: Path, prefix: str, commit: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{prefix}-{stamp}-{commit[:12]}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / ".incomplete").write_text("experiment incomplete\n", encoding="utf-8")
    return path


def _environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _condition_name(fraction: float) -> str:
    if fraction == 0.0:
        return "clean"
    return f"missing_{int(round(fraction * 100))}pct"


def _model_visible_samples(samples):
    visible = {
        spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.features if spec.model_visible
    }
    projected = []
    for sample in samples:
        positions = {name: index for index, name in enumerate(sample.feature_ids)}
        names = tuple(name for name in sample.feature_ids if name in visible)
        indices = [positions[name] for name in names]
        projected.append(
            replace(
                sample,
                features=tuple(sample.features[index] for index in indices),
                feature_ids=names,
                categorical_mask=tuple(
                    sample.categorical_mask[index] for index in indices
                ),
                missing_mask=tuple(sample.missing_mask[index] for index in indices),
                category_tokens=tuple(
                    sample.category_tokens[index] for index in indices
                ),
            )
        )
    return tuple(projected)


def _coarse_targets(samples):
    to_id = {label: index for index, label in enumerate(SUPPORTED_COARSE_LABELS)}
    try:
        return np.asarray(
            [to_id[sample.coarse_label] for sample in samples], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(f"unsupported NF-CSE coarse label: {exc.args[0]}") from exc


def _load_selection_inputs(scratch_config_path, pretrained_config_path):
    paths = tuple(
        Path(value).expanduser().resolve()
        for value in (scratch_config_path, pretrained_config_path)
    )
    configs = tuple(load_config(path, expected_workflow="finetune") for path in paths)
    for config in configs:
        if config.data["dataset_id"] != NF_CSE_LABEL_SPACE.dataset_id:
            raise ValueError("confirmation finetune config must use NF-CSE")
        if config.training.get("evaluation_scope") != "validation_only":
            raise ValueError("confirmation training must use validation_only scope")
    data_keys = ("split_dir", "preprocessor_dir", "data_manifest")
    if any(
        configs[0].data[key] != configs[1].data[key] for key in data_keys
    ):
        raise ValueError("scratch and pretrained configs must use identical data")
    split_dir = Path(configs[0].data["split_dir"]).expanduser().resolve()
    preprocessor_dir = Path(
        configs[0].data["preprocessor_dir"]
    ).expanduser().resolve()
    data_manifest_path = Path(
        configs[0].data["data_manifest"]
    ).expanduser().resolve()
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    dataset = load_nf_cse_development_samples(
        split_dir,
        preprocessor_dir,
        data_root=_DATA_ROOT,
        split_names=("train", "val"),
    )
    preprocessor = load_nf_cse_preprocessor(preprocessor_dir / "preprocessor.json")
    tokenizer = build_nf_cse_tokenizer_metadata(
        dataset, preprocessor, data_manifest=data_manifest
    )
    split_manifest_path = split_dir / SELECTION_MANIFEST_FILENAME
    split_selection_records_path = split_dir / SELECTION_RECORDS_FILENAME
    split_selection_summary_path = split_dir / SELECTION_SUMMARY_FILENAME
    preprocessor_selection_summary_path = (
        preprocessor_dir / SELECTION_SUMMARY_FILENAME
    )
    selection_features_path = preprocessor_dir / SELECTION_FEATURES_FILENAME
    split_config = json.loads(
        (split_dir / "config.snapshot.json").read_text(encoding="utf-8")
    )
    try:
        source_split_dir = Path(split_config["source_split_dir"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError("confirmation training split must be a few-shot derivation") from exc
    overlap_path = source_split_dir / "overlap_audit.json"
    seal_path = source_split_dir / "sealed_test.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if overlap.get("verified_zero_overlap") is not True:
        raise ValueError("base confirmatory split does not prove zero overlap")
    if split_config.get("source_zero_overlap_verified") is not True:
        raise ValueError("few-shot preparation did not lock zero historical overlap")
    if split_config.get("source_sealed_test_status") != (
        "sealed_before_model_selection"
    ):
        raise ValueError("few-shot preparation did not lock sealed-test status")
    if split_config.get("source_selected_rows", 0) < 200_000:
        raise ValueError("base confirmatory split is too small")
    split_selection_summary = json.loads(
        split_selection_summary_path.read_text(encoding="utf-8")
    )
    preprocessor_selection_summary = json.loads(
        preprocessor_selection_summary_path.read_text(encoding="utf-8")
    )
    safe_artifact_hashes = {
        "selected_records_sha256": _sha256_file(
            split_selection_records_path
        ),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "source_manifest_sha256": _sha256_file(data_manifest_path),
    }
    for field, value in safe_artifact_hashes.items():
        if split_selection_summary.get(field) != value:
            raise ValueError(f"split selection summary changed: {field}")
        if preprocessor_selection_summary.get(field) != value:
            raise ValueError(f"preprocessor selection summary changed: {field}")
    if preprocessor_selection_summary.get(
        "split_selection_summary_sha256"
    ) != _sha256_file(split_selection_summary_path):
        raise ValueError("preprocessor does not lock the split selection summary")
    if preprocessor_selection_summary.get(
        "selection_features_sha256"
    ) != _sha256_file(selection_features_path):
        raise ValueError("selection feature cache hash changed")
    for summary_name, summary in (
        ("split", split_selection_summary),
        ("preprocessor", preprocessor_selection_summary),
    ):
        if (
            summary.get("selection_scope") != "train_validation_only"
            or summary.get("materialized_splits") != ["train", "val"]
            or summary.get("test_labels_present") is not False
            or summary.get("test_statistics_present") is not False
        ):
            raise ValueError(
                f"{summary_name} selection view violates sealed-test scope"
            )
    opaque_base_fields = (
        "source_selected_records_sha256",
        "source_split_manifest_sha256",
        "source_split_summary_sha256",
        "source_overlap_audit_sha256",
        "source_sealed_test_sha256",
    )
    for field in opaque_base_fields:
        value = split_config.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            or split_selection_summary.get(field) != value
        ):
            raise ValueError(f"invalid preparation-time provenance: {field}")
    for field in (
        "sealed_selected_records_sha256",
        "sealed_split_manifest_sha256",
    ):
        value = split_selection_summary.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"invalid sealed artifact reference: {field}")
    if split_config["source_overlap_audit_sha256"] != _sha256_file(overlap_path):
        raise ValueError("base overlap audit changed after few-shot preparation")
    if split_selection_summary.get(
        "sealed_split_manifest_sha256"
    ) != preprocessor_selection_summary.get("sealed_split_manifest_sha256"):
        raise ValueError("selection views disagree on the sealed manifest hash")
    return {
        "config_paths": paths,
        "configs": configs,
        "split_dir": split_dir,
        "source_split_dir": source_split_dir,
        "preprocessor_dir": preprocessor_dir,
        "data_manifest_path": data_manifest_path,
        "split_manifest_path": split_manifest_path,
        "split_selection_records_path": split_selection_records_path,
        "split_selection_summary_path": split_selection_summary_path,
        "preprocessor_selection_summary_path": (
            preprocessor_selection_summary_path
        ),
        "selection_features_path": selection_features_path,
        "overlap_path": overlap_path,
        "seal_path": seal_path,
        "split_config": split_config,
        "split_selection_summary": split_selection_summary,
        "preprocessor_selection_summary": preprocessor_selection_summary,
        "data_manifest": data_manifest,
        "dataset": dataset,
        "tokenizer": tokenizer,
    }


def _restore_model(config, checkpoint_path, tokenizer, device):
    path = Path(checkpoint_path).expanduser().resolve()
    model = _build_model(
        config, tokenizer, NF_CSE_LABEL_SPACE, SCHEMAFLOW_MAIN_REGISTRY
    ).to(device)
    adaptation = _configure_adaptation(
        model, config.training.get("adaptation_mode", "full")
    )
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training["learning_rate"],
        weight_decay=config.training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.training["scheduler_gamma"]
    )
    payload = load_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
        restore_rng=False,
        expected_model_data_metadata=_model_data_metadata(
            SCHEMAFLOW_MAIN_REGISTRY, tokenizer, NF_CSE_LABEL_SPACE
        ),
    )
    checkpoint_adaptation = payload["extra_state"].get("adaptation")
    if checkpoint_adaptation is not None and checkpoint_adaptation != adaptation:
        raise RuntimeError("checkpoint adaptation does not match config")
    metrics = payload["extra_state"].get("metrics", {})
    if metrics.get("evaluation_scope") != "validation_only" or "test" in metrics:
        raise RuntimeError("checkpoint provenance shows premature test evaluation")
    model.eval()
    return model, payload, adaptation


def _validate_initialization_provenance(
    scratch_payload, pretrained_payload, source_checkpoint
):
    if scratch_payload["extra_state"].get("initialization") is not None:
        raise RuntimeError("scratch checkpoint unexpectedly has pretrained initialization")
    initialization = pretrained_payload["extra_state"].get("initialization")
    if not isinstance(initialization, dict):
        raise RuntimeError("pretrained checkpoint lacks initialization provenance")
    source_path = Path(source_checkpoint).expanduser().resolve()
    if not source_path.is_file():
        raise RuntimeError("NF-UNSW source pretraining checkpoint is missing")
    source_sha256 = _sha256_file(source_path)
    if (
        Path(initialization.get("source_checkpoint", "")).resolve() != source_path
        or initialization.get("source_checkpoint_sha256") != source_sha256
        or initialization.get("source_schema_version")
        != "nf-unsw-nb15-v2-model-41-v1"
    ):
        raise RuntimeError("pretrained finetune does not trace to the frozen source")
    source_payload = torch.load(source_path, map_location="cpu", weights_only=True)
    source_config = source_payload.get("config", {})
    source_data = source_config.get("data", {})
    if (
        source_config.get("workflow") != "pretrain"
        or source_data.get("dataset_id") != "NF-UNSW-NB15-v2"
        or "sources" in source_data
    ):
        raise RuntimeError("source checkpoint is not NF-UNSW-only pretraining")
    return {
        **initialization,
        "source_checkpoint_current_sha256": source_sha256,
        "source_dataset_id": source_data["dataset_id"],
        "source_count": 1,
        "nf_cse_exposure": False,
        "verified_nf_unsw_only": True,
    }


def _fit_xgboost(matrix, targets, seed):
    expected = np.arange(len(SUPPORTED_COARSE_LABELS), dtype=np.int64)
    if not np.array_equal(np.unique(targets), expected):
        raise ValueError("training does not cover all supported coarse labels")
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
        "num_class": len(SUPPORTED_COARSE_LABELS),
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


def _xgboost_probabilities(model, matrix):
    expected = np.arange(len(SUPPORTED_COARSE_LABELS), dtype=np.int64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    values = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if values.shape[1] != len(expected) or not np.array_equal(classes, expected):
        raise RuntimeError("XGBoost probability columns changed")
    return values


def _schemaflow_probabilities(model, samples, tokenizer, device, batch_size):
    global_indices = tuple(
        NF_CSE_LABEL_SPACE.coarse_labels.index(label)
        for label in SUPPORTED_COARSE_LABELS
    )
    batches = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            members = samples[start : start + batch_size]
            batch = collate_schemaflow_samples(
                members, tokenizer, registry=SCHEMAFLOW_MAIN_REGISTRY
            ).to(device)
            logits = model(batch).coarse_logits[:, global_indices]
            batches.append(torch.softmax(logits, dim=1).detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        torch.cat(batches).numpy().astype(np.float64, copy=False),
        time.perf_counter() - started,
    )


def _conditions(samples, missing_seed):
    eligible = tuple(
        spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.features if spec.model_visible
    )
    values = {}
    reports = {}
    for fraction in _FRACTIONS:
        condition = _condition_name(fraction)
        conditioned, report = apply_deterministic_field_missingness(
            samples,
            fraction=fraction,
            seed=missing_seed,
            eligible_feature_ids=eligible,
        )
        values[condition] = conditioned
        reports[condition] = report
    return values, reports


def _collect_outputs(
    samples_by_condition,
    *,
    encoder,
    xgboost_model,
    scratch_model,
    pretrained_model,
    tokenizer,
    device,
    batch_size,
):
    outputs = {}
    timings = {"xgboost": 0.0, "schemaflow_scratch": 0.0, "schemaflow_pretrained": 0.0}
    for condition, samples in samples_by_condition.items():
        projected = _model_visible_samples(samples)
        started = time.perf_counter()
        xgb = _xgboost_probabilities(
            xgboost_model, encoder.transform(projected)
        )
        timings["xgboost"] += time.perf_counter() - started
        scratch, scratch_seconds = _schemaflow_probabilities(
            scratch_model, samples, tokenizer, device, batch_size
        )
        pretrained, pretrained_seconds = _schemaflow_probabilities(
            pretrained_model, samples, tokenizer, device, batch_size
        )
        timings["schemaflow_scratch"] += scratch_seconds
        timings["schemaflow_pretrained"] += pretrained_seconds
        targets = _coarse_targets(samples)
        outputs[condition] = {
            "xgboost": xgb,
            "schemaflow_scratch": scratch,
            "schemaflow_pretrained": pretrained,
            "targets": targets,
            "sample_order_sha256": _sample_order_sha256(samples),
        }
    return outputs, timings


def _output_hashes(outputs):
    return {
        condition: {
            **{
                f"{model}_probabilities_sha256": probability_sha256(values[model])
                for model in (
                    "xgboost",
                    "schemaflow_scratch",
                    "schemaflow_pretrained",
                )
            },
            "targets_sha256": _array_sha256(values["targets"], "<i8"),
            "sample_order_sha256": values["sample_order_sha256"],
            "sample_count": int(len(values["targets"])),
        }
        for condition, values in outputs.items()
    }


def _policy_probabilities(outputs, policies):
    return {
        condition: pool_probabilities(
            values["xgboost"],
            values["schemaflow_pretrained"],
            alpha=policies[condition]["alpha"],
            method=policies[condition]["method"],
        )
        for condition, values in outputs.items()
    }


def _policy_metrics(outputs, policies, *, confirmatory=False):
    metric = confirmatory_metric_payload if confirmatory else metric_payload
    probabilities = _policy_probabilities(outputs, policies)
    return {
        condition: metric(
            probabilities[condition], values["targets"], SUPPORTED_COARSE_LABELS
        )
        for condition, values in outputs.items()
    }


def _mean_macro_f1(metrics):
    return float(np.mean([values["macro_f1"] for values in metrics.values()]))


def _select_a0_mr(outputs, alpha_grid, candidate_min_gain, pool_min_gain):
    linear = {}
    linear_or_log = {}
    for condition, values in outputs.items():
        arguments = (
            values["xgboost"],
            values["schemaflow_pretrained"],
            values["targets"],
            SUPPORTED_COARSE_LABELS,
        )
        linear[condition] = select_fusion(
            *arguments, alpha_grid=alpha_grid, methods=("linear",)
        )
        linear_or_log[condition] = select_fusion(
            *arguments, alpha_grid=alpha_grid, methods=("linear", "log")
        )
    linear_policies = {
        condition: {
            "method": result["selected"]["method"],
            "alpha": result["selected"]["alpha"],
            "selected_on": condition,
        }
        for condition, result in linear.items()
    }
    pool_policies = {
        condition: {
            "method": result["selected"]["method"],
            "alpha": result["selected"]["alpha"],
            "selected_on": condition,
        }
        for condition, result in linear_or_log.items()
    }
    fixed_metrics = _policy_metrics(outputs, _FIXED_A0)
    linear_metrics = _policy_metrics(outputs, linear_policies)
    pool_metrics = _policy_metrics(outputs, pool_policies)
    fixed_mean = _mean_macro_f1(fixed_metrics)
    linear_mean = _mean_macro_f1(linear_metrics)
    pool_mean = _mean_macro_f1(pool_metrics)
    if pool_mean - linear_mean >= pool_min_gain:
        candidate_name = "missing_rate_conditioned_linear_or_log"
        candidate_policies = pool_policies
        candidate_metrics = pool_metrics
        candidate_mean = pool_mean
    else:
        candidate_name = "missing_rate_conditioned_linear"
        candidate_policies = linear_policies
        candidate_metrics = linear_metrics
        candidate_mean = linear_mean
    deltas = {
        condition: candidate_metrics[condition]["macro_f1"]
        - fixed_metrics[condition]["macro_f1"]
        for condition in _CONDITIONS
    }
    passed = (
        candidate_mean - fixed_mean >= candidate_min_gain
        and deltas["clean"] >= -1e-12
        and min(deltas.values()) >= -0.002
    )
    locked_policies = candidate_policies if passed else _FIXED_A0
    locked_metrics = candidate_metrics if passed else fixed_metrics
    reference_match = {
        condition: {
            "method": locked_policies[condition]["method"],
            "alpha": locked_policies[condition]["alpha"],
        }
        == _NF_TON_REFERENCE[condition]
        for condition in _CONDITIONS
    }
    return {
        "format_version": 1,
        "selection_scope": "validation_only",
        "test_samples_materialized": False,
        "test_probabilities_computed": False,
        "test_metrics_used": False,
        "task": "shared_coarse_nf_cse_supported_five_class",
        "label_names": list(SUPPORTED_COARSE_LABELS),
        "fixed_a0": {
            "name": "a0_fixed_linear_alpha_xgb_0p55",
            "policies": _FIXED_A0,
            "validation_metrics": fixed_metrics,
            "mean_shared_coarse_macro_f1": fixed_mean,
        },
        "conditioned_linear": {
            "policies": linear_policies,
            "validation_metrics": linear_metrics,
            "mean_shared_coarse_macro_f1": linear_mean,
        },
        "conditioned_linear_or_log": {
            "policies": pool_policies,
            "validation_metrics": pool_metrics,
            "mean_shared_coarse_macro_f1": pool_mean,
        },
        "routing_gate": {
            "candidate_name": candidate_name,
            "candidate_min_mean_gain": candidate_min_gain,
            "pool_min_mean_gain": pool_min_gain,
            "candidate_minus_fixed_mean_macro_f1": candidate_mean - fixed_mean,
            "candidate_minus_fixed_by_condition": deltas,
            "passed": passed,
        },
        "locked_policy": {
            "name": candidate_name if passed else "a0_fixed_linear_alpha_xgb_0p55",
            "policies": locked_policies,
            "validation_metrics": locked_metrics,
            "mean_shared_coarse_macro_f1": _mean_macro_f1(locked_metrics),
        },
        "nf_ton_policy_reproduction": {
            "reference": _NF_TON_REFERENCE,
            "matches_by_condition": reference_match,
            "matches_all": all(reference_match.values()),
            "same_weights_required": False,
        },
        "search_details": {"linear": linear, "linear_or_log": linear_or_log},
    }


def _base_metrics(outputs, *, confirmatory=False):
    metric = confirmatory_metric_payload if confirmatory else metric_payload
    return {
        condition: {
            model: metric(values[model], values["targets"], SUPPORTED_COARSE_LABELS)
            for model in (
                "xgboost",
                "schemaflow_scratch",
                "schemaflow_pretrained",
            )
        }
        for condition, values in outputs.items()
    }


def _checkpoint_provenance(path, payload, adaptation):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "git_commit": payload["git_commit"],
        "schema_version": payload["schema_version"],
        "adaptation": adaptation,
        "evaluation_scope": payload["extra_state"]["metrics"]["evaluation_scope"],
        "test_metrics_present": "test" in payload["extra_state"]["metrics"],
        "restore_verified": True,
    }


def _select(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.seed < 0 or args.missing_seed < 0:
        raise SystemExit("seeds must be non-negative")
    if args.candidate_min_mean_gain < 0 or args.pool_min_mean_gain < 0:
        raise SystemExit("selection thresholds must be non-negative")
    frozen_arguments = {
        "seed": (args.seed, 17),
        "missing_seed": (args.missing_seed, 17),
        "alpha_step": (args.alpha_step, 0.05),
        "candidate_min_mean_gain": (args.candidate_min_mean_gain, 0.005),
        "pool_min_mean_gain": (args.pool_min_mean_gain, 0.002),
    }
    changed = [
        name
        for name, (actual, expected) in frozen_arguments.items()
        if not np.isclose(actual, expected, atol=0, rtol=0)
    ]
    if changed:
        raise SystemExit(
            "frozen A0-MR selection arguments changed: " + ", ".join(changed)
        )
    alpha_grid = default_alpha_grid(args.alpha_step)
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    inputs = _load_selection_inputs(args.scratch_config, args.pretrained_config)
    checkpoints = tuple(
        Path(value).expanduser().resolve()
        for value in (args.scratch_checkpoint, args.pretrained_checkpoint)
    )
    run_dir = _run_dir(
        _output_root(args.output_root), "nf-cse-a0-mr-selection", git["commit"]
    )
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train = _model_visible_samples(inputs["dataset"].splits["train"])
    train_targets = _coarse_targets(train)
    encoder = TabularFeatureEncoder.fit(
        train,
        category_encoders=inputs["tokenizer"].categorical_vocabularies,
    )
    xgb, xgb_parameters, xgb_fit_seconds = _fit_xgboost(
        encoder.transform(train), train_targets, args.seed
    )
    xgb_path = run_dir / "xgboost_model.json"
    xgb.save_model(str(xgb_path))
    scratch_model, scratch_payload, scratch_adaptation = _restore_model(
        inputs["configs"][0], checkpoints[0], inputs["tokenizer"], device
    )
    pretrained_model, pretrained_payload, pretrained_adaptation = _restore_model(
        inputs["configs"][1], checkpoints[1], inputs["tokenizer"], device
    )
    initialization = _validate_initialization_provenance(
        scratch_payload, pretrained_payload, args.source_pretrain_checkpoint
    )
    validation_conditions, missingness = _conditions(
        inputs["dataset"].splits["val"], args.missing_seed
    )
    outputs, timings = _collect_outputs(
        validation_conditions,
        encoder=encoder,
        xgboost_model=xgb,
        scratch_model=scratch_model,
        pretrained_model=pretrained_model,
        tokenizer=inputs["tokenizer"],
        device=device,
        batch_size=inputs["configs"][0].training["batch_size"],
    )
    selection = _select_a0_mr(
        outputs,
        alpha_grid,
        args.candidate_min_mean_gain,
        args.pool_min_mean_gain,
    )
    config = {
        "format_version": 1,
        "workflow": "nf_cse_a0_mr_shared_coarse_selection",
        "phase": "selection",
        "selection_scope": ["train", "validation"],
        "materialized_splits": ["train", "val"],
        "test_manifest_members_read": False,
        "sealed_test_file_content_read": False,
        "test_samples_materialized": False,
        "test_probabilities_computed": False,
        "test_metrics_used": False,
        "dataset_id": NF_CSE_LABEL_SPACE.dataset_id,
        "task": "shared_coarse_nf_cse_supported_five_class",
        "label_names": list(SUPPORTED_COARSE_LABELS),
        "label_order_rule": "global_shared_coarse_order_filtered_to_natural_support",
        "scratch_config_path": str(inputs["config_paths"][0]),
        "pretrained_config_path": str(inputs["config_paths"][1]),
        "scratch_config_sha256": _sha256_file(inputs["config_paths"][0]),
        "pretrained_config_sha256": _sha256_file(inputs["config_paths"][1]),
        "scratch_checkpoint_path": str(checkpoints[0]),
        "pretrained_checkpoint_path": str(checkpoints[1]),
        "scratch_checkpoint_sha256": _sha256_file(checkpoints[0]),
        "pretrained_checkpoint_sha256": _sha256_file(checkpoints[1]),
        "source_pretrain_checkpoint_path": str(
            Path(args.source_pretrain_checkpoint).resolve()
        ),
        "source_pretrain_checkpoint_sha256": _sha256_file(
            Path(args.source_pretrain_checkpoint).resolve()
        ),
        "split_dir": str(inputs["split_dir"]),
        "base_confirmatory_split_dir": str(inputs["source_split_dir"]),
        "preprocessor_dir": str(inputs["preprocessor_dir"]),
        "data_manifest_path": str(inputs["data_manifest_path"]),
        "data_manifest_sha256": _sha256_file(inputs["data_manifest_path"]),
        "selection_records_sha256": _sha256_file(
            inputs["split_selection_records_path"]
        ),
        "selection_manifest_sha256": _sha256_file(
            inputs["split_manifest_path"]
        ),
        "split_selection_summary_sha256": _sha256_file(
            inputs["split_selection_summary_path"]
        ),
        "preprocessor_selection_summary_sha256": _sha256_file(
            inputs["preprocessor_selection_summary_path"]
        ),
        "selection_features_sha256": _sha256_file(
            inputs["selection_features_path"]
        ),
        "preprocessor_file_sha256": inputs[
            "preprocessor_selection_summary"
        ]["preprocessor_file_sha256"],
        "sealed_selected_records_sha256": inputs[
            "split_selection_summary"
        ]["sealed_selected_records_sha256"],
        "sealed_split_manifest_sha256": inputs[
            "split_selection_summary"
        ]["sealed_split_manifest_sha256"],
        "base_selected_records_sha256": inputs["split_config"][
            "source_selected_records_sha256"
        ],
        "base_split_manifest_sha256": inputs["split_config"][
            "source_split_manifest_sha256"
        ],
        "base_split_summary_sha256": inputs["split_config"][
            "source_split_summary_sha256"
        ],
        "base_overlap_audit_sha256": _sha256_file(inputs["overlap_path"]),
        "base_sealed_test_sha256": inputs["split_config"][
            "source_sealed_test_sha256"
        ],
        "tokenizer_metadata_hash": inputs["tokenizer"].metadata_hash,
        "label_space_hash": NF_CSE_LABEL_SPACE.label_space_hash,
        "feature_registry_hash": SCHEMAFLOW_MAIN_REGISTRY.registry_hash,
        "seed": args.seed,
        "missing_fractions": list(_FRACTIONS),
        "missing_seed": args.missing_seed,
        "missingness_algorithm": "sha256-sample-feature-ranking-v1",
        "alpha_grid": list(alpha_grid),
        "alpha_step": args.alpha_step,
        "tie_break": [
            "mcc",
            "balanced_accuracy",
            "linear_pooling",
            "higher_xgboost_alpha",
        ],
        "candidate_min_mean_gain": args.candidate_min_mean_gain,
        "pool_min_mean_gain": args.pool_min_mean_gain,
        "fixed_a0": {"method": "linear", "alpha_xgboost": 0.55},
        "device": args.device,
        "git": git,
        "xgboost": {
            "parameters": xgb_parameters,
            "fit_scope": "target_train_only",
            "class_balance": "inverse_frequency_sample_weight",
            "training_sample_count": len(train),
        },
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "selection.json", selection)
    _write_json(run_dir / "validation_metrics.json", _base_metrics(outputs))
    _write_json(run_dir / "validation_probability_hashes.json", _output_hashes(outputs))
    _write_json(run_dir / "missingness.json", {"validation": missingness})
    _write_json(run_dir / "encoder.json", encoder.to_dict())
    _write_json(run_dir / "tokenizer_metadata.json", inputs["tokenizer"].to_dict())
    _write_json(run_dir / "feature_registry.json", SCHEMAFLOW_MAIN_REGISTRY.to_dict())
    _write_json(
        run_dir / "task_label_space.json",
        {
            "dataset_label_space": NF_CSE_LABEL_SPACE.to_dict(),
            "task": config["task"],
            "label_names": list(SUPPORTED_COARSE_LABELS),
        },
    )
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
        },
    )
    _write_json(run_dir / "environment.json", _environment())
    _write_json(
        run_dir / "efficiency.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "xgboost_fit_seconds": xgb_fit_seconds,
            "validation_inference_seconds": timings,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
    )
    (run_dir / "xgboost_model.sha256").write_text(
        _sha256_file(xgb_path) + "  xgboost_model.json\n", encoding="ascii"
    )
    shutil.copyfile(inputs["data_manifest_path"], run_dir / "data_manifest.json")
    shutil.copyfile(
        inputs["split_manifest_path"], run_dir / SELECTION_MANIFEST_FILENAME
    )
    shutil.copyfile(inputs["overlap_path"], run_dir / "overlap_audit.json")
    _write_json(
        run_dir / "sealed_test.reference.json",
        {
            "status": "sealed_reference_not_opened_during_selection",
            "selection_access_allowed": False,
            "content_read_during_selection": False,
            "path": str(inputs["seal_path"]),
            "expected_sha256": config["base_sealed_test_sha256"],
        },
    )
    (run_dir / "git_commit.txt").write_text(git["commit"] + "\n", encoding="ascii")
    (run_dir / ".incomplete").unlink()
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), "selection": selection["locked_policy"]}, sort_keys=True))
    return 0


def _verify_selection(selection_dir, git):
    required = (
        "config.json",
        "selection.json",
        "validation_probability_hashes.json",
        "encoder.json",
        "tokenizer_metadata.json",
        "checkpoint_provenance.json",
        "xgboost_model.json",
        "xgboost_model.sha256",
        SELECTION_MANIFEST_FILENAME,
        "overlap_audit.json",
        "sealed_test.reference.json",
    )
    if any(not (selection_dir / name).is_file() for name in required):
        raise RuntimeError("selection artifacts are incomplete")
    if (selection_dir / ".incomplete").exists():
        raise RuntimeError("selection run is incomplete")
    config = json.loads((selection_dir / "config.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (selection_dir / "selection.json").read_text(encoding="utf-8")
    )
    if config["git"]["commit"] != git["commit"]:
        raise RuntimeError("selection and confirmation must use the same commit")
    if config.get("test_samples_materialized") is not False:
        raise RuntimeError("selection did not preserve sealed-test scope")
    if (
        config.get("test_manifest_members_read") is not False
        or config.get("sealed_test_file_content_read") is not False
    ):
        raise RuntimeError("selection opened sealed-test artifacts")
    if tuple(config.get("missing_fractions", ())) != _FRACTIONS:
        raise RuntimeError("selection missingness conditions changed")
    if config.get("missing_seed") != 17:
        raise RuntimeError("missingness seed is not frozen at 17")
    if config.get("seed") != 17:
        raise RuntimeError("XGBoost/confirmation seed is not frozen at 17")
    if config.get("alpha_grid") != list(default_alpha_grid(0.05)):
        raise RuntimeError("A0-MR alpha grid changed")
    if config.get("tie_break") != [
        "mcc",
        "balanced_accuracy",
        "linear_pooling",
        "higher_xgboost_alpha",
    ]:
        raise RuntimeError("A0-MR tie-break changed")
    if not np.isclose(
        config.get("candidate_min_mean_gain"), 0.005, atol=0, rtol=0
    ) or not np.isclose(
        config.get("pool_min_mean_gain"), 0.002, atol=0, rtol=0
    ):
        raise RuntimeError("A0-MR routing thresholds changed")
    if selection.get("test_metrics_used") is not False:
        raise RuntimeError("selection used sealed test metrics")
    reference = json.loads(
        (selection_dir / "sealed_test.reference.json").read_text(
            encoding="utf-8"
        )
    )
    if reference != {
        "status": "sealed_reference_not_opened_during_selection",
        "selection_access_allowed": False,
        "content_read_during_selection": False,
        "path": str(
            Path(config["base_confirmatory_split_dir"]) / "sealed_test.json"
        ),
        "expected_sha256": config["base_sealed_test_sha256"],
    }:
        raise RuntimeError("sealed-test reference is invalid")
    if _sha256_file(
        selection_dir / SELECTION_MANIFEST_FILENAME
    ) != config.get("selection_manifest_sha256"):
        raise RuntimeError("copied selection manifest hash changed")
    if _sha256_file(
        selection_dir / "overlap_audit.json"
    ) != config.get("base_overlap_audit_sha256"):
        raise RuntimeError("copied overlap audit hash changed")
    if selection["fixed_a0"]["policies"] != {
        condition: dict(policy) for condition, policy in _FIXED_A0.items()
    }:
        raise RuntimeError("fixed A0 changed")
    recorded = (selection_dir / "xgboost_model.sha256").read_text().split()[0]
    if _sha256_file(selection_dir / "xgboost_model.json") != recorded:
        raise RuntimeError("XGBoost model hash changed")
    return config, selection


def _encoder_from_lock(payload, tokenizer):
    expected_fields = tuple(payload["feature_names"])
    categorical = tuple(payload["categorical_feature_names"])
    values = MappingProxyType(
        {
            name: tuple(ids)
            for name, ids in sorted(payload["categorical_values"].items())
        }
    )
    encoder = TabularFeatureEncoder(
        feature_names=expected_fields,
        categorical_feature_names=categorical,
        categorical_values=values,
        unknown_category_id=payload["unknown_category_id"],
        category_encoders=tokenizer.categorical_vocabularies,
    )
    if encoder.to_dict() != payload:
        raise RuntimeError("locked tabular encoder cannot be reproduced")
    return encoder


def _write_prediction_lock(run_dir, outputs, group_ids):
    targets = np.asarray(outputs["clean"]["targets"], dtype=np.int64)
    if tuple(outputs) != _CONDITIONS or any(
        not np.array_equal(targets, outputs[name]["targets"])
        for name in _CONDITIONS[1:]
    ):
        raise RuntimeError("test targets or condition order changed")
    unique_groups, group_codes = np.unique(
        np.asarray(group_ids, dtype="U64"), return_inverse=True
    )
    if len(group_codes) != len(targets):
        raise RuntimeError("test group IDs do not align with predictions")
    arrays = {
        "targets": targets,
        "group_codes": group_codes.astype(np.int32, copy=False),
    }
    for condition, values in outputs.items():
        for model in (
            "xgboost",
            "schemaflow_scratch",
            "schemaflow_pretrained",
        ):
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
        "predictions_path": predictions_path.name,
        "predictions_sha256": _sha256_file(predictions_path),
        "group_hashes_path": groups_path.name,
        "group_hashes_sha256": _sha256_file(groups_path),
        "group_count": int(len(unique_groups)),
        "group_codes_sha256": _array_sha256(group_codes, "<i8"),
        "targets_sha256": _array_sha256(targets, "<i8"),
        "conditions": _output_hashes(outputs),
    }
    _write_json(run_dir / "prediction_lock.json", lock)
    (run_dir / "prediction_lock.sha256").write_text(
        _sha256_file(run_dir / "prediction_lock.json")
        + "  prediction_lock.json\n",
        encoding="ascii",
    )
    reloaded = {}
    with np.load(predictions_path, allow_pickle=False) as persisted:
        if _array_sha256(persisted["targets"], "<i8") != lock["targets_sha256"]:
            raise RuntimeError("locked targets changed on reload")
        for condition in _CONDITIONS:
            reloaded[condition] = {
                model: persisted[f"{model}__{condition}"].copy()
                for model in (
                    "xgboost",
                    "schemaflow_scratch",
                    "schemaflow_pretrained",
                )
            }
            reloaded[condition].update(
                {
                    "targets": persisted["targets"].copy(),
                    "sample_order_sha256": outputs[condition][
                        "sample_order_sha256"
                    ],
                }
            )
    if _output_hashes(reloaded) != lock["conditions"]:
        raise RuntimeError("locked probabilities changed on reload")
    return reloaded, tuple(group_ids), lock


def _retention(metrics):
    clean = metrics["clean"]["macro_f1"]
    return {
        condition: float(metrics[condition]["macro_f1"] / clean)
        for condition in ("missing_30pct", "missing_50pct")
    }


def _confirm(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.bootstrap_resamples < 200 or args.bootstrap_seed < 0:
        raise SystemExit("invalid bootstrap configuration")
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    selection_dir = args.selection_dir.expanduser().resolve()
    config, selection = _verify_selection(selection_dir, git)
    if args.device != config["device"]:
        raise RuntimeError("confirmation device must match selection")
    paths_to_verify = {
        "scratch_config_sha256": Path(config["scratch_config_path"]),
        "pretrained_config_sha256": Path(config["pretrained_config_path"]),
        "scratch_checkpoint_sha256": Path(config["scratch_checkpoint_path"]),
        "pretrained_checkpoint_sha256": Path(config["pretrained_checkpoint_path"]),
        "source_pretrain_checkpoint_sha256": Path(
            config["source_pretrain_checkpoint_path"]
        ),
        "data_manifest_sha256": Path(config["data_manifest_path"]),
        "selection_records_sha256": Path(config["split_dir"])
        / SELECTION_RECORDS_FILENAME,
        "selection_manifest_sha256": Path(config["split_dir"])
        / SELECTION_MANIFEST_FILENAME,
        "split_selection_summary_sha256": Path(config["split_dir"])
        / SELECTION_SUMMARY_FILENAME,
        "preprocessor_selection_summary_sha256": Path(
            config["preprocessor_dir"]
        )
        / SELECTION_SUMMARY_FILENAME,
        "selection_features_sha256": Path(config["preprocessor_dir"])
        / SELECTION_FEATURES_FILENAME,
        "preprocessor_file_sha256": Path(config["preprocessor_dir"])
        / "preprocessor.json",
        "sealed_selected_records_sha256": Path(config["split_dir"])
        / "selected_records.jsonl",
        "sealed_split_manifest_sha256": Path(config["split_dir"])
        / "split_manifest.json",
        "base_selected_records_sha256": Path(
            config["base_confirmatory_split_dir"]
        )
        / "selected_records.jsonl",
        "base_split_manifest_sha256": Path(
            config["base_confirmatory_split_dir"]
        )
        / "split_manifest.json",
        "base_split_summary_sha256": Path(
            config["base_confirmatory_split_dir"]
        )
        / "split_summary.json",
        "base_overlap_audit_sha256": Path(config["base_confirmatory_split_dir"])
        / "overlap_audit.json",
        "base_sealed_test_sha256": Path(config["base_confirmatory_split_dir"])
        / "sealed_test.json",
    }
    for field, path in paths_to_verify.items():
        if _sha256_file(path) != config[field]:
            raise RuntimeError(f"selection provenance changed: {field}")
    overlap = json.loads(
        paths_to_verify["base_overlap_audit_sha256"].read_text(encoding="utf-8")
    )
    if overlap.get("verified_zero_overlap") is not True:
        raise RuntimeError("base split no longer proves zero overlap")
    sealed_test = json.loads(
        paths_to_verify["base_sealed_test_sha256"].read_text(encoding="utf-8")
    )
    if (
        sealed_test.get("status") != "sealed_before_model_selection"
        or sealed_test.get("selection_access_allowed") is not False
        or sealed_test.get("opened_for_evaluation") is not False
    ):
        raise RuntimeError("base sealed-test lock is invalid at confirmation open")

    scratch_config = load_config(
        config["scratch_config_path"], expected_workflow="finetune"
    )
    pretrained_config = load_config(
        config["pretrained_config_path"], expected_workflow="finetune"
    )
    tokenizer = TokenizerMetadata.from_dict(
        json.loads(
            (selection_dir / "tokenizer_metadata.json").read_text(encoding="utf-8")
        )
    )
    encoder_payload = json.loads(
        (selection_dir / "encoder.json").read_text(encoding="utf-8")
    )
    encoder = _encoder_from_lock(encoder_payload, tokenizer)
    xgb = XGBClassifier()
    xgb.load_model(str(selection_dir / "xgboost_model.json"))
    device = torch.device(args.device)
    if device.type == "cuda":
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
    dataset = load_nf_cse_development_samples(
        config["split_dir"],
        config["preprocessor_dir"],
        data_root=_DATA_ROOT,
        split_names=("test",),
    )
    if dataset.preprocessor_hash != tokenizer.preprocessor_hash:
        raise RuntimeError("test preprocessing differs from validation lock")
    manifest = SplitManifest.from_json(
        (Path(config["split_dir"]) / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    test_samples = dataset.splits["test"]
    group_ids = tuple(
        manifest.sample_groups[sample.sample_id] for sample in test_samples
    )
    started = time.perf_counter()
    run_dir = _run_dir(
        _output_root(args.output_root), "nf-cse-a0-mr-confirmation", git["commit"]
    )
    test_conditions, missingness = _conditions(test_samples, config["missing_seed"])
    outputs, timings = _collect_outputs(
        test_conditions,
        encoder=encoder,
        xgboost_model=xgb,
        scratch_model=scratch_model,
        pretrained_model=pretrained_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=scratch_config.training["batch_size"],
    )
    locked_outputs, locked_groups, prediction_lock = _write_prediction_lock(
        run_dir, outputs, group_ids
    )

    base = _base_metrics(locked_outputs, confirmatory=True)
    fixed = _policy_metrics(locked_outputs, _FIXED_A0, confirmatory=True)
    locked_policies = selection["locked_policy"]["policies"]
    locked = _policy_metrics(locked_outputs, locked_policies, confirmatory=True)
    locked_probabilities = _policy_probabilities(locked_outputs, locked_policies)
    xgb_probabilities = {
        condition: values["xgboost"] for condition, values in locked_outputs.items()
    }
    bootstrap = paired_cluster_bootstrap_macro_f1(
        locked_outputs,
        locked_probabilities,
        xgb_probabilities,
        locked_groups,
        class_count=len(SUPPORTED_COARSE_LABELS),
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    xgb_metrics = {
        condition: values["xgboost"] for condition, values in base.items()
    }
    gate = evaluate_confirmatory_gate(
        xgboost_metrics=xgb_metrics,
        fixed_a0_metrics=fixed,
        locked_metrics=locked,
        bootstrap=bootstrap,
        rare_labels=NF_CSE_RARE_COARSE_LABELS,
        rare_label_definition=_RARE_DEFINITION,
        maximum_rare_mean_recall_drop=MAX_RARE_MEAN_RECALL_DROP,
    )
    metrics = {
        "format_version": 1,
        "evaluation_scope": "record_level_independent_sealed_test",
        "task": "shared_coarse_nf_cse_supported_five_class",
        "label_names": list(SUPPORTED_COARSE_LABELS),
        "base_models": base,
        "fixed_a0": fixed,
        "locked_policy_name": selection["locked_policy"]["name"],
        "locked_policy": locked,
        "retention": {
            "xgboost": _retention(xgb_metrics),
            "schemaflow_scratch": _retention(
                {
                    condition: values["schemaflow_scratch"]
                    for condition, values in base.items()
                }
            ),
            "schemaflow_pretrained": _retention(
                {
                    condition: values["schemaflow_pretrained"]
                    for condition, values in base.items()
                }
            ),
            "fixed_a0": _retention(fixed),
            "a0_mr": _retention(locked),
        },
        "macro_f1_deltas": {
            condition: {
                "a0_mr_minus_xgboost": locked[condition]["macro_f1"]
                - xgb_metrics[condition]["macro_f1"],
                "a0_mr_minus_fixed_a0": locked[condition]["macro_f1"]
                - fixed[condition]["macro_f1"],
            }
            for condition in _CONDITIONS
        },
    }
    confirmation_config = {
        "format_version": 1,
        "workflow": "nf_cse_a0_mr_shared_coarse_confirmation",
        "phase": "confirmation",
        "test_access": "single_locked_evaluation",
        "test_opened_after_model_and_artifact_preflight": True,
        "prediction_lock_before_metrics": True,
        "selection_dir": str(selection_dir),
        "selection_sha256": _sha256_file(selection_dir / "selection.json"),
        "base_confirmatory_split_dir": config["base_confirmatory_split_dir"],
        "overlap_audit_sha256": config["base_overlap_audit_sha256"],
        "sealed_test_sha256": config["base_sealed_test_sha256"],
        "test_rows": len(test_samples),
        "locked_policies": locked_policies,
        "fixed_a0_policies": _FIXED_A0,
        "missing_seed": config["missing_seed"],
        "bootstrap": {
            "method": "paired_poisson_cluster_bootstrap_linearized_macro_f1",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
        },
        "gate": {
            "minimum_mean_xgboost_macro_f1_gain": MIN_MEAN_XGBOOST_GAIN,
            "bootstrap_mean_ci95_lower_must_exceed": 0.0,
            "missing_conditions_must_exceed_fixed_a0": True,
            "all_conditions_must_be_noninferior_to_xgboost": True,
            "rare_labels": list(NF_CSE_RARE_COARSE_LABELS),
            "rare_label_definition": _RARE_DEFINITION,
            "maximum_rare_mean_recall_drop": MAX_RARE_MEAN_RECALL_DROP,
        },
        "device": args.device,
        "git": git,
    }
    _write_json(run_dir / "config.json", confirmation_config)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "bootstrap.json", bootstrap)
    _write_json(run_dir / "gate.json", gate)
    _write_json(run_dir / "missingness.json", {"test": missingness})
    _write_json(
        run_dir / "test_access_receipt.json",
        {
            "access_count": 1,
            "purpose": "locked_confirmatory_evaluation_only",
            "selection_completed_before_access": True,
            "model_and_artifact_preflight_completed_before_access": True,
            "sealed_test_original_sha256": config["base_sealed_test_sha256"],
            "prediction_lock_sha256": _sha256_file(
                run_dir / "prediction_lock.json"
            ),
            "post_access_model_or_policy_selection": False,
        },
    )
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
        },
    )
    _write_json(run_dir / "environment.json", _environment())
    _write_json(
        run_dir / "efficiency.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "test_inference_seconds": timings,
            "prediction_archive_bytes": (run_dir / "predictions.npz").stat().st_size,
            "prediction_archive_sha256": prediction_lock["predictions_sha256"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
    )
    shutil.copyfile(selection_dir / "selection.json", run_dir / "selection.lock.json")
    shutil.copyfile(selection_dir / "overlap_audit.json", run_dir / "overlap_audit.json")
    shutil.copyfile(
        selection_dir / "sealed_test.reference.json",
        run_dir / "sealed_test.reference.json",
    )
    shutil.copyfile(
        paths_to_verify["base_sealed_test_sha256"],
        run_dir / "sealed_test.original_lock.json",
    )
    shutil.copyfile(selection_dir / "tokenizer_metadata.json", run_dir / "tokenizer_metadata.json")
    (run_dir / "git_commit.txt").write_text(git["commit"] + "\n", encoding="ascii")
    (run_dir / ".incomplete").unlink()
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), "gate": gate, "full_matrix_allowed": gate["full_matrix_allowed"]}, sort_keys=True))
    return 0


def main():
    args = _args()
    return _select(args) if args.phase == "select" else _confirm(args)


if __name__ == "__main__":
    raise SystemExit(main())
