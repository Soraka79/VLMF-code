#!/usr/bin/env python3
"""Select and evaluate an auditable SchemaFlow/XGBoost probability fusion."""

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

import numpy as np
import sklearn
import torch
import xgboost
from xgboost import XGBClassifier

from flowfm.config import load_config
from flowfm.data.labels import NF_TON_LABEL_SPACE, encode_supervised_targets
from flowfm.data.missingness import apply_deterministic_field_missingness
from flowfm.data.nf_ton_confirmatory import load_nf_ton_confirmatory_samples
from flowfm.data.nf_ton_dataset import load_nf_ton_development_samples
from flowfm.data.nf_ton_preprocessing import load_nf_ton_preprocessor
from flowfm.data.nf_ton_tokenization import build_nf_ton_tokenizer_metadata
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.baselines import (
    TabularFeatureEncoder,
    inverse_frequency_sample_weights,
)
from flowfm.training.checkpoints import load_checkpoint
from flowfm.training.probability_fusion import (
    default_alpha_grid,
    evaluate_policy,
    pool_probabilities,
    probability_sha256,
    select_fusion,
)
from flowfm.training.confirmatory_statistics import (
    MAX_RARE_MEAN_RECALL_DROP,
    MIN_MEAN_XGBOOST_GAIN,
    RARE_NATIVE_LABELS,
    confirmatory_metric_payload,
    evaluate_confirmatory_gate,
    paired_cluster_bootstrap_macro_f1,
)
from flowfm.training.supervised import (
    _build_model,
    _configure_adaptation,
    _model_data_metadata,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_DEFAULT_OUTPUT_ROOT = _DATA_ROOT / "Res-AE-Attn" / "results"
_CONFIRMATORY_CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
_CONFIRMATORY_POLICIES = {
    "clean": {"method": "linear", "alpha": 0.55},
    "missing_30pct": {"method": "log", "alpha": 0.85},
    "missing_50pct": {"method": "log", "alpha": 0.70},
}
_FIXED_A0_POLICIES = {
    condition: {"method": "linear", "alpha": 0.55}
    for condition in _CONFIRMATORY_CONDITIONS
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--config", required=True, type=Path)
    select.add_argument("--checkpoint", required=True, type=Path)
    select.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT, type=Path)
    select.add_argument("--seed", default=17, type=int)
    select.add_argument("--missing-fractions", default=(0.3, 0.5), nargs="+", type=float)
    select.add_argument("--missing-seed", default=17, type=int)
    select.add_argument("--alpha-step", default=0.05, type=float)
    select.add_argument("--candidate-min-mean-gain", default=0.005, type=float)
    select.add_argument("--pool-min-mean-gain", default=0.002, type=float)
    select.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--selection-dir", required=True, type=Path)
    evaluate.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT, type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--selection-dir", required=True, type=Path)
    confirm.add_argument("--cohort-dir", required=True, type=Path)
    confirm.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT, type=Path)
    confirm.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    confirm.add_argument("--bootstrap-resamples", default=2000, type=int)
    confirm.add_argument("--bootstrap-seed", default=20260724, type=int)
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _integer_array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
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


def _git_state(repo_dir: Path) -> dict[str, object]:
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
        raise RuntimeError("fusion experiments require a clean Git worktree")
    return {"commit": commit, "dirty": False, "status": []}


def _validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(_DATA_ROOT):
        raise SystemExit("output-root must be located below /root/autodl-tmp")
    return resolved


def _validate_missing_fractions(values) -> tuple[float, ...]:
    fractions = tuple(float(value) for value in values)
    if not fractions:
        raise ValueError("at least one missing fraction is required")
    if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("missing fractions must be finite and in (0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("missing fractions must be strictly increasing and unique")
    return (0.0, *fractions)


def _condition_name(fraction: float) -> str:
    if fraction == 0.0:
        return "clean"
    percentage = fraction * 100.0
    if np.isclose(percentage, round(percentage), atol=1e-10, rtol=0):
        token = str(int(round(percentage)))
    else:
        token = f"{percentage:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"missing_{token}pct"


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


def _load_experiment_inputs(config_path: Path, checkpoint_path: Path):
    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit("checkpoint does not exist")
    config = load_config(config_path, expected_workflow="finetune")
    if config.data["dataset_id"] != NF_TON_LABEL_SPACE.dataset_id:
        raise SystemExit("fusion runner requires an NF-ToN finetune config")
    split_dir = Path(config.data["split_dir"]).expanduser().resolve()
    preprocessor_dir = Path(config.data["preprocessor_dir"]).expanduser().resolve()
    manifest_path = Path(config.data["data_manifest"]).expanduser().resolve()
    split_manifest_path = split_dir / "split_manifest.json"
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = load_nf_ton_development_samples(
        split_dir,
        preprocessor_dir,
        data_root=_DATA_ROOT,
    )
    preprocessor = load_nf_ton_preprocessor(preprocessor_dir / "preprocessor.json")
    tokenizer = build_nf_ton_tokenizer_metadata(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )
    return {
        "config": config,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "split_dir": split_dir,
        "preprocessor_dir": preprocessor_dir,
        "manifest_path": manifest_path,
        "split_manifest_path": split_manifest_path,
        "data_manifest": data_manifest,
        "dataset": dataset,
        "tokenizer": tokenizer,
    }


def _restore_schemaflow(inputs, device: torch.device):
    config = inputs["config"]
    tokenizer = inputs["tokenizer"]
    model = _build_model(
        config,
        tokenizer,
        NF_TON_LABEL_SPACE,
        SCHEMAFLOW_MAIN_REGISTRY,
    ).to(device)
    adaptation = _configure_adaptation(
        model,
        config.training.get("adaptation_mode", "full"),
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
        optimizer,
        gamma=config.training["scheduler_gamma"],
    )
    payload = load_checkpoint(
        inputs["checkpoint_path"],
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
        restore_rng=False,
        expected_model_data_metadata=_model_data_metadata(
            SCHEMAFLOW_MAIN_REGISTRY,
            tokenizer,
            NF_TON_LABEL_SPACE,
        ),
    )
    checkpoint_adaptation = payload["extra_state"].get("adaptation")
    if checkpoint_adaptation is not None and checkpoint_adaptation != adaptation:
        raise RuntimeError("checkpoint adaptation policy does not match config")
    model.eval()
    return model, payload, adaptation


def _fit_xgboost(matrix, targets, seed: int):
    expected = np.arange(len(NF_TON_LABEL_SPACE.native_labels), dtype=np.int64)
    if not np.array_equal(np.unique(targets), expected):
        raise ValueError("training labels do not cover the native label space")
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
        "num_class": len(NF_TON_LABEL_SPACE.native_labels),
    }
    weights = inverse_frequency_sample_weights(targets)
    model = XGBClassifier(**parameters)
    started = time.perf_counter()
    model.fit(matrix, targets, sample_weight=weights, verbose=False)
    return model, parameters, time.perf_counter() - started


def _xgboost_probabilities(model, matrix) -> np.ndarray:
    expected = np.arange(len(NF_TON_LABEL_SPACE.native_labels), dtype=np.int64)
    classes = np.asarray(getattr(model, "classes_", ()), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if probabilities.shape[1] != len(expected) or not np.array_equal(classes, expected):
        raise ValueError("XGBoost probability columns do not match native labels")
    return probabilities


def _schemaflow_probabilities(
    model,
    samples,
    tokenizer,
    device: torch.device,
    batch_size: int,
):
    probability_batches = []
    target_batches = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            members = samples[start : start + batch_size]
            batch = collate_schemaflow_samples(
                members,
                tokenizer,
                registry=SCHEMAFLOW_MAIN_REGISTRY,
            ).to(device)
            output = model(batch)
            probability_batches.append(
                torch.softmax(output.native_logits, dim=1).detach().cpu()
            )
            target_batches.append(
                encode_supervised_targets(
                    members,
                    NF_TON_LABEL_SPACE,
                ).native
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        torch.cat(probability_batches).numpy().astype(np.float64, copy=False),
        torch.cat(target_batches).numpy().astype(np.int64, copy=False),
        time.perf_counter() - started,
    )


def _conditioned_samples(samples, fractions, missing_seed: int):
    eligible = tuple(
        spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.features if spec.model_visible
    )
    conditioned = {}
    reports = {}
    for fraction in fractions:
        name = _condition_name(fraction)
        values, report = apply_deterministic_field_missingness(
            samples,
            fraction=fraction,
            seed=missing_seed,
            eligible_feature_ids=eligible,
        )
        conditioned[name] = values
        reports[name] = report
    return conditioned, reports


def _collect_outputs(
    samples_by_condition,
    *,
    encoder,
    xgboost_model,
    schemaflow_model,
    tokenizer,
    device,
    batch_size,
):
    outputs = {}
    total_schemaflow_seconds = 0.0
    for condition, samples in samples_by_condition.items():
        projected = _model_visible_samples(samples)
        tree = _xgboost_probabilities(
            xgboost_model,
            encoder.transform(projected),
        )
        neural, targets, elapsed = _schemaflow_probabilities(
            schemaflow_model,
            samples,
            tokenizer,
            device,
            batch_size,
        )
        expected_targets = encode_supervised_targets(
            samples,
            NF_TON_LABEL_SPACE,
        ).native.numpy()
        if not np.array_equal(targets, expected_targets):
            raise RuntimeError("SchemaFlow and tabular targets are misaligned")
        outputs[condition] = {
            "xgboost": tree,
            "schemaflow": neural,
            "targets": targets,
            "sample_order_sha256": _sample_order_sha256(samples),
        }
        total_schemaflow_seconds += elapsed
    return outputs, total_schemaflow_seconds


def _output_hashes(outputs) -> dict[str, object]:
    return {
        condition: {
            "xgboost_probabilities_sha256": probability_sha256(values["xgboost"]),
            "schemaflow_probabilities_sha256": probability_sha256(
                values["schemaflow"]
            ),
            "targets_sha256": _target_sha256(values["targets"]),
            "sample_order_sha256": values["sample_order_sha256"],
            "sample_count": int(len(values["targets"])),
        }
        for condition, values in outputs.items()
    }


def _base_metrics(outputs) -> dict[str, object]:
    labels = NF_TON_LABEL_SPACE.native_labels
    return {
        condition: {
            "xgboost": metric_payload(
                values["xgboost"], values["targets"], labels
            ),
            "schemaflow": metric_payload(
                values["schemaflow"], values["targets"], labels
            ),
        }
        for condition, values in outputs.items()
    }


def _policy_metrics(outputs, policies) -> dict[str, object]:
    labels = NF_TON_LABEL_SPACE.native_labels
    return {
        condition: evaluate_policy(
            values["xgboost"],
            values["schemaflow"],
            values["targets"],
            labels,
            policies[condition],
        )
        for condition, values in outputs.items()
    }


def _confirmatory_base_metrics(outputs) -> dict[str, object]:
    labels = NF_TON_LABEL_SPACE.native_labels
    return {
        condition: {
            "xgboost": confirmatory_metric_payload(
                values["xgboost"], values["targets"], labels
            ),
            "schemaflow": confirmatory_metric_payload(
                values["schemaflow"], values["targets"], labels
            ),
        }
        for condition, values in outputs.items()
    }


def _confirmatory_policy_metrics(outputs, policies) -> dict[str, object]:
    labels = NF_TON_LABEL_SPACE.native_labels
    return {
        condition: confirmatory_metric_payload(
            pool_probabilities(
                values["xgboost"],
                values["schemaflow"],
                alpha=policies[condition]["alpha"],
                method=policies[condition]["method"],
            ),
            values["targets"],
            labels,
        )
        for condition, values in outputs.items()
    }


def _mean_macro_f1(metrics) -> float:
    return float(np.mean([values["macro_f1"] for values in metrics.values()]))


def _select_policy(
    outputs,
    *,
    alpha_grid,
    candidate_min_mean_gain: float,
    pool_min_mean_gain: float,
):
    labels = NF_TON_LABEL_SPACE.native_labels
    linear_selections = {}
    pool_selections = {}
    for condition, values in outputs.items():
        linear_selections[condition] = select_fusion(
            values["xgboost"],
            values["schemaflow"],
            values["targets"],
            labels,
            alpha_grid=alpha_grid,
            methods=("linear",),
        )
        pool_selections[condition] = select_fusion(
            values["xgboost"],
            values["schemaflow"],
            values["targets"],
            labels,
            alpha_grid=alpha_grid,
            methods=("linear", "log"),
        )

    fixed_choice = linear_selections["clean"]["selected"]
    fixed_policies = {
        condition: {
            "method": fixed_choice["method"],
            "alpha": fixed_choice["alpha"],
            "selected_on": "clean",
        }
        for condition in outputs
    }
    conditioned_linear_policies = {
        condition: {
            "method": selection["selected"]["method"],
            "alpha": selection["selected"]["alpha"],
            "selected_on": condition,
        }
        for condition, selection in linear_selections.items()
    }
    conditioned_pool_policies = {
        condition: {
            "method": selection["selected"]["method"],
            "alpha": selection["selected"]["alpha"],
            "selected_on": condition,
        }
        for condition, selection in pool_selections.items()
    }
    fixed_metrics = _policy_metrics(outputs, fixed_policies)
    conditioned_linear_metrics = _policy_metrics(
        outputs,
        conditioned_linear_policies,
    )
    conditioned_pool_metrics = _policy_metrics(outputs, conditioned_pool_policies)
    fixed_mean = _mean_macro_f1(fixed_metrics)
    linear_mean = _mean_macro_f1(conditioned_linear_metrics)
    pool_mean = _mean_macro_f1(conditioned_pool_metrics)

    if pool_mean - linear_mean >= pool_min_mean_gain:
        candidate_name = "missing_rate_conditioned_linear_or_log"
        candidate_policies = conditioned_pool_policies
        candidate_metrics = conditioned_pool_metrics
        candidate_mean = pool_mean
    else:
        candidate_name = "missing_rate_conditioned_linear"
        candidate_policies = conditioned_linear_policies
        candidate_metrics = conditioned_linear_metrics
        candidate_mean = linear_mean
    deltas = {
        condition: (
            candidate_metrics[condition]["macro_f1"]
            - fixed_metrics[condition]["macro_f1"]
        )
        for condition in outputs
    }
    candidate_passed = (
        candidate_mean - fixed_mean >= candidate_min_mean_gain
        and deltas["clean"] >= -1e-12
        and min(deltas.values()) >= -0.002
    )
    if candidate_passed:
        locked_name = candidate_name
        locked_policies = candidate_policies
        locked_metrics = candidate_metrics
    else:
        locked_name = "a0_fixed_clean_linear"
        locked_policies = fixed_policies
        locked_metrics = fixed_metrics
    return {
        "format_version": 1,
        "selection_scope": "validation_only",
        "test_metrics_used": False,
        "conditions": list(outputs),
        "fixed_a0": {
            "name": "a0_fixed_clean_linear",
            "policies": fixed_policies,
            "validation_metrics": fixed_metrics,
            "mean_native_macro_f1": fixed_mean,
        },
        "conditioned_linear": {
            "policies": conditioned_linear_policies,
            "validation_metrics": conditioned_linear_metrics,
            "mean_native_macro_f1": linear_mean,
        },
        "conditioned_linear_or_log": {
            "policies": conditioned_pool_policies,
            "validation_metrics": conditioned_pool_metrics,
            "mean_native_macro_f1": pool_mean,
        },
        "gate": {
            "candidate_name": candidate_name,
            "candidate_min_mean_gain": candidate_min_mean_gain,
            "pool_min_mean_gain": pool_min_mean_gain,
            "candidate_minus_fixed_mean_macro_f1": candidate_mean - fixed_mean,
            "candidate_minus_fixed_by_condition": deltas,
            "passed": candidate_passed,
        },
        "locked_policy": {
            "name": locked_name,
            "policies": locked_policies,
            "validation_metrics": locked_metrics,
            "mean_native_macro_f1": _mean_macro_f1(locked_metrics),
        },
        "search_details": {
            "linear": linear_selections,
            "linear_or_log": pool_selections,
        },
    }


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def _run_dir(output_root: Path, prefix: str, git_commit: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_root / (
        f"{prefix}-{timestamp}-{git_commit[:12]}-{uuid.uuid4().hex[:8]}"
    )
    path.mkdir(parents=True, exist_ok=False)
    (path / ".incomplete").write_text("fusion experiment incomplete\n", encoding="utf-8")
    return path


def _selection_config(args, inputs, git, fractions) -> dict[str, object]:
    return {
        "format_version": 1,
        "workflow": "schemaflow_xgboost_probability_fusion_selection",
        "phase": "selection",
        "selection_scope": ["train", "validation"],
        "test_probabilities_computed": False,
        "test_metrics_used": False,
        "dataset_id": NF_TON_LABEL_SPACE.dataset_id,
        "training_config_path": str(inputs["config_path"]),
        "training_config_sha256": _sha256_file(inputs["config_path"]),
        "checkpoint_path": str(inputs["checkpoint_path"]),
        "checkpoint_sha256": _sha256_file(inputs["checkpoint_path"]),
        "split_dir": str(inputs["split_dir"]),
        "preprocessor_dir": str(inputs["preprocessor_dir"]),
        "data_manifest_path": str(inputs["manifest_path"]),
        "data_manifest_sha256": _sha256_file(inputs["manifest_path"]),
        "split_manifest_sha256": _sha256_file(inputs["split_manifest_path"]),
        "tokenizer_metadata_hash": inputs["tokenizer"].metadata_hash,
        "label_space_hash": NF_TON_LABEL_SPACE.label_space_hash,
        "feature_registry_hash": SCHEMAFLOW_MAIN_REGISTRY.registry_hash,
        "seed": args.seed,
        "missing_fractions": list(fractions),
        "missing_seed": args.missing_seed,
        "alpha_step": args.alpha_step,
        "candidate_min_mean_gain": args.candidate_min_mean_gain,
        "pool_min_mean_gain": args.pool_min_mean_gain,
        "device": args.device,
        "git": git,
    }


def _select(args) -> int:
    if args.seed < 0 or args.missing_seed < 0:
        raise SystemExit("seed values must be non-negative")
    if args.candidate_min_mean_gain < 0 or args.pool_min_mean_gain < 0:
        raise SystemExit("selection gain thresholds must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    fractions = _validate_missing_fractions(args.missing_fractions)
    alpha_grid = default_alpha_grid(args.alpha_step)
    output_root = _validate_output_root(args.output_root)
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    started = time.perf_counter()
    inputs = _load_experiment_inputs(args.config, args.checkpoint)
    run_dir = _run_dir(
        output_root,
        "schemaflow-xgb-fusion-selection",
        git["commit"],
    )
    try:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train = _model_visible_samples(inputs["dataset"].splits["train"])
        train_targets = encode_supervised_targets(
            train,
            NF_TON_LABEL_SPACE,
        ).native.numpy()
        encoder = TabularFeatureEncoder.fit(
            train,
            category_encoders=inputs["tokenizer"].categorical_vocabularies,
        )
        xgboost_model, xgboost_parameters, xgboost_fit_seconds = _fit_xgboost(
            encoder.transform(train),
            train_targets,
            args.seed,
        )
        xgboost_path = run_dir / "xgboost_model.json"
        xgboost_model.save_model(str(xgboost_path))
        schemaflow_model, checkpoint_payload, adaptation = _restore_schemaflow(
            inputs,
            device,
        )
        validation_conditions, missingness = _conditioned_samples(
            inputs["dataset"].splits["val"],
            fractions,
            args.missing_seed,
        )
        outputs, schemaflow_seconds = _collect_outputs(
            validation_conditions,
            encoder=encoder,
            xgboost_model=xgboost_model,
            schemaflow_model=schemaflow_model,
            tokenizer=inputs["tokenizer"],
            device=device,
            batch_size=inputs["config"].training["batch_size"],
        )
        selection = _select_policy(
            outputs,
            alpha_grid=alpha_grid,
            candidate_min_mean_gain=args.candidate_min_mean_gain,
            pool_min_mean_gain=args.pool_min_mean_gain,
        )
        config = _selection_config(args, inputs, git, fractions)
        config["xgboost"] = {
            "parameters": xgboost_parameters,
            "fit_scope": "train_only",
            "class_balance": "inverse_frequency_sample_weight",
            "model_path": xgboost_path.name,
            "training_sample_count": len(train),
        }
        checkpoint_provenance = {
            "path": str(inputs["checkpoint_path"]),
            "sha256": config["checkpoint_sha256"],
            "epoch": checkpoint_payload["epoch"],
            "global_step": checkpoint_payload["global_step"],
            "git_commit": checkpoint_payload["git_commit"],
            "schema_version": checkpoint_payload["schema_version"],
            "adaptation": adaptation,
            "restore_verified": True,
        }
        _write_json(run_dir / "config.json", config)
        _write_json(run_dir / "selection.json", selection)
        _write_json(run_dir / "validation_metrics.json", _base_metrics(outputs))
        _write_json(run_dir / "validation_probability_hashes.json", _output_hashes(outputs))
        _write_json(run_dir / "missingness.json", {"validation": missingness})
        _write_json(run_dir / "encoder.json", encoder.to_dict())
        _write_json(
            run_dir / "tokenizer_metadata.json",
            inputs["tokenizer"].to_dict(),
        )
        _write_json(
            run_dir / "feature_registry.json",
            SCHEMAFLOW_MAIN_REGISTRY.to_dict(),
        )
        _write_json(run_dir / "label_space.json", NF_TON_LABEL_SPACE.to_dict())
        _write_json(run_dir / "checkpoint_provenance.json", checkpoint_provenance)
        _write_json(run_dir / "environment.json", _environment())
        shutil.copyfile(inputs["manifest_path"], run_dir / "data_manifest.json")
        shutil.copyfile(
            inputs["split_manifest_path"],
            run_dir / "split_manifest.json",
        )
        (run_dir / "git_commit.txt").write_text(
            git["commit"] + "\n",
            encoding="utf-8",
        )
        _write_json(
            run_dir / "efficiency.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "xgboost_fit_seconds": xgboost_fit_seconds,
                "schemaflow_validation_seconds": schemaflow_seconds,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else 0
                ),
                "peak_system_memory_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
            },
        )
        model_sha = _sha256_file(xgboost_path)
        (run_dir / "xgboost_model.sha256").write_text(
            model_sha + "  " + xgboost_path.name + "\n",
            encoding="utf-8",
        )
        selection_sha = _sha256_file(run_dir / "selection.json")
        (run_dir / "selection.sha256").write_text(
            selection_sha + "  selection.json\n",
            encoding="utf-8",
        )
        (run_dir / ".incomplete").unlink()
    except Exception:
        raise

    print(
        json.dumps(
            {
                "status": "selected",
                "run_dir": str(run_dir),
                "locked_policy": selection["locked_policy"],
                "gate": selection["gate"],
                "test_metrics_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_selection(
    selection_dir: Path,
    git,
    *,
    require_current_git: bool = True,
) -> tuple[dict, dict]:
    selection_dir = selection_dir.expanduser().resolve()
    if not selection_dir.is_dir() or (selection_dir / ".incomplete").exists():
        raise SystemExit("selection directory is missing or incomplete")
    config_path = selection_dir / "config.json"
    selection_path = selection_dir / "selection.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if config.get("phase") != "selection" or selection.get("test_metrics_used") is not False:
        raise RuntimeError("selection artifact does not enforce validation-only selection")
    if require_current_git and config["git"]["commit"] != git["commit"]:
        raise RuntimeError("evaluation Git commit differs from the selection commit")
    recorded_selection_sha = (selection_dir / "selection.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if _sha256_file(selection_path) != recorded_selection_sha:
        raise RuntimeError("selection lock hash does not match selection.json")
    recorded_model_sha = (selection_dir / "xgboost_model.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if _sha256_file(selection_dir / "xgboost_model.json") != recorded_model_sha:
        raise RuntimeError("persisted XGBoost model hash does not match")
    return config, selection


def _policy_core(policy) -> dict[str, object]:
    return {"method": policy.get("method"), "alpha": float(policy.get("alpha"))}


def _verify_confirmatory_policy(selection_config, selection) -> None:
    if tuple(float(value) for value in selection_config["missing_fractions"]) != (
        0.0,
        0.3,
        0.5,
    ):
        raise RuntimeError("selection missingness conditions are not the frozen set")
    if selection_config.get("missing_seed") != 17:
        raise RuntimeError("selection missingness seed is not the frozen seed 17")
    if not np.isclose(selection_config.get("alpha_step"), 0.05, atol=0, rtol=0):
        raise RuntimeError("selection alpha grid step is not frozen at 0.05")
    locked = selection["locked_policy"]["policies"]
    fixed = selection["fixed_a0"]["policies"]
    if (
        tuple(locked) != _CONFIRMATORY_CONDITIONS
        or tuple(fixed) != _CONFIRMATORY_CONDITIONS
    ):
        raise RuntimeError("selection policy conditions are not in frozen order")
    for condition in _CONFIRMATORY_CONDITIONS:
        if _policy_core(locked[condition]) != _CONFIRMATORY_POLICIES[condition]:
            raise RuntimeError(f"locked policy changed for {condition}")
        if _policy_core(fixed[condition]) != _FIXED_A0_POLICIES[condition]:
            raise RuntimeError(f"fixed A0 policy changed for {condition}")
        details = selection["search_details"]["linear_or_log"][condition]
        if details.get("alpha_grid") != list(default_alpha_grid(0.05)):
            raise RuntimeError(f"selection alpha grid changed for {condition}")
        if details.get("tie_break") != [
            "mcc",
            "balanced_accuracy",
            "linear_pooling",
            "higher_xgboost_alpha",
        ]:
            raise RuntimeError(f"selection tie break changed for {condition}")


def _locked_probabilities(outputs, policies) -> dict[str, np.ndarray]:
    return {
        condition: pool_probabilities(
            values["xgboost"],
            values["schemaflow"],
            alpha=policies[condition]["alpha"],
            method=policies[condition]["method"],
        )
        for condition, values in outputs.items()
    }


def _write_prediction_lock(run_dir: Path, outputs, group_ids):
    conditions = tuple(outputs)
    if conditions != _CONFIRMATORY_CONDITIONS:
        raise RuntimeError("prediction lock conditions are not predeclared")
    targets = np.asarray(outputs["clean"]["targets"], dtype=np.int64)
    if any(
        not np.array_equal(targets, outputs[condition]["targets"])
        for condition in conditions[1:]
    ):
        raise RuntimeError("confirmatory targets changed across conditions")
    unique_groups, group_codes = np.unique(
        np.asarray(group_ids, dtype="U64"), return_inverse=True
    )
    if len(group_codes) != len(targets):
        raise RuntimeError("confirmatory group IDs do not align with predictions")
    if not np.array_equal(
        unique_groups[group_codes], np.asarray(group_ids, dtype="U64")
    ):
        raise RuntimeError("confirmatory group coding changed sample order")
    predictions_path = run_dir / "predictions.npz"
    arrays = {
        "targets": targets.astype(np.int64, copy=False),
        "group_codes": group_codes.astype(np.int32, copy=False),
    }
    for condition, values in outputs.items():
        arrays[f"xgboost__{condition}"] = values["xgboost"]
        arrays[f"schemaflow__{condition}"] = values["schemaflow"]
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
        "group_codes_sha256": _integer_array_sha256(group_codes),
        "targets_sha256": _target_sha256(targets),
        "conditions": _output_hashes(outputs),
    }
    lock_path = run_dir / "prediction_lock.json"
    _write_json(lock_path, lock)
    (run_dir / "prediction_lock.sha256").write_text(
        _sha256_file(lock_path) + "  prediction_lock.json\n",
        encoding="ascii",
    )

    reloaded = {}
    with np.load(predictions_path, allow_pickle=False) as persisted:
        if _integer_array_sha256(persisted["group_codes"]) != lock[
            "group_codes_sha256"
        ]:
            raise RuntimeError("persisted group codes do not match prediction lock")
        if _target_sha256(persisted["targets"]) != lock["targets_sha256"]:
            raise RuntimeError("persisted targets do not match prediction lock")
        for condition in conditions:
            reloaded[condition] = {
                "xgboost": persisted[f"xgboost__{condition}"].copy(),
                "schemaflow": persisted[f"schemaflow__{condition}"].copy(),
                "targets": persisted["targets"].copy(),
                "sample_order_sha256": outputs[condition][
                    "sample_order_sha256"
                ],
            }
    if _output_hashes(reloaded) != lock["conditions"]:
        raise RuntimeError("reloaded predictions do not match the prediction lock")
    return reloaded, tuple(group_ids), lock


def _retention(metrics) -> dict[str, float]:
    clean = float(metrics["clean"]["macro_f1"])
    return {
        condition: float(metrics[condition]["macro_f1"] / clean)
        for condition in ("missing_30pct", "missing_50pct")
    }


def _evaluate(args) -> int:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    output_root = _validate_output_root(args.output_root)
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    selection_dir = args.selection_dir.expanduser().resolve()
    selection_config, selection = _verify_selection(selection_dir, git)
    inputs = _load_experiment_inputs(
        Path(selection_config["training_config_path"]),
        Path(selection_config["checkpoint_path"]),
    )
    for key, path in (
        ("training_config_sha256", inputs["config_path"]),
        ("checkpoint_sha256", inputs["checkpoint_path"]),
        ("data_manifest_sha256", inputs["manifest_path"]),
        ("split_manifest_sha256", inputs["split_manifest_path"]),
    ):
        if _sha256_file(path) != selection_config[key]:
            raise RuntimeError(f"selection provenance changed: {key}")
    fractions = tuple(float(value) for value in selection_config["missing_fractions"])
    expected_conditions = tuple(_condition_name(value) for value in fractions)
    locked_policies = selection["locked_policy"]["policies"]
    if set(locked_policies) != set(expected_conditions):
        raise RuntimeError("locked policy conditions do not match selection config")
    if args.device != selection_config["device"]:
        raise RuntimeError("evaluation device must match the selection device")

    started = time.perf_counter()
    run_dir = _run_dir(
        output_root,
        "schemaflow-xgb-fusion-evaluation",
        git["commit"],
    )
    try:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train = _model_visible_samples(inputs["dataset"].splits["train"])
        encoder = TabularFeatureEncoder.fit(
            train,
            category_encoders=inputs["tokenizer"].categorical_vocabularies,
        )
        recorded_encoder = json.loads(
            (selection_dir / "encoder.json").read_text(encoding="utf-8")
        )
        if encoder.to_dict() != recorded_encoder:
            raise RuntimeError("reconstructed tabular encoder differs from selection")
        xgboost_model = XGBClassifier()
        xgboost_model.load_model(str(selection_dir / "xgboost_model.json"))
        schemaflow_model, checkpoint_payload, adaptation = _restore_schemaflow(
            inputs,
            device,
        )

        validation_conditions, validation_missingness = _conditioned_samples(
            inputs["dataset"].splits["val"],
            fractions,
            selection_config["missing_seed"],
        )
        validation_outputs, validation_seconds = _collect_outputs(
            validation_conditions,
            encoder=encoder,
            xgboost_model=xgboost_model,
            schemaflow_model=schemaflow_model,
            tokenizer=inputs["tokenizer"],
            device=device,
            batch_size=inputs["config"].training["batch_size"],
        )
        recorded_hashes = json.loads(
            (selection_dir / "validation_probability_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        if _output_hashes(validation_outputs) != recorded_hashes:
            raise RuntimeError("validation probabilities do not reproduce selection")

        test_conditions, test_missingness = _conditioned_samples(
            inputs["dataset"].splits["test"],
            fractions,
            selection_config["missing_seed"],
        )
        test_outputs, test_seconds = _collect_outputs(
            test_conditions,
            encoder=encoder,
            xgboost_model=xgboost_model,
            schemaflow_model=schemaflow_model,
            tokenizer=inputs["tokenizer"],
            device=device,
            batch_size=inputs["config"].training["batch_size"],
        )
        fixed_policies = selection["fixed_a0"]["policies"]
        metrics = {
            "format_version": 1,
            "test_access": "single_locked_evaluation",
            "base_models": _base_metrics(test_outputs),
            "fixed_a0": _policy_metrics(test_outputs, fixed_policies),
            "locked_policy_name": selection["locked_policy"]["name"],
            "locked_policy": _policy_metrics(test_outputs, locked_policies),
        }
        selection_sha = _sha256_file(selection_dir / "selection.json")
        evaluation_config = {
            "format_version": 1,
            "workflow": "schemaflow_xgboost_probability_fusion_evaluation",
            "phase": "evaluation",
            "test_access": "single_locked_evaluation",
            "selection_dir": str(selection_dir),
            "selection_sha256": selection_sha,
            "locked_policy_name": selection["locked_policy"]["name"],
            "locked_policies": locked_policies,
            "device": args.device,
            "git": git,
        }
        _write_json(run_dir / "config.json", evaluation_config)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "test_probability_hashes.json", _output_hashes(test_outputs))
        _write_json(
            run_dir / "missingness.json",
            {
                "validation_reproduction": validation_missingness,
                "test": test_missingness,
            },
        )
        _write_json(run_dir / "label_space.json", NF_TON_LABEL_SPACE.to_dict())
        shutil.copyfile(
            selection_dir / "tokenizer_metadata.json",
            run_dir / "tokenizer_metadata.json",
        )
        shutil.copyfile(
            selection_dir / "feature_registry.json",
            run_dir / "feature_registry.json",
        )
        _write_json(run_dir / "environment.json", _environment())
        _write_json(
            run_dir / "checkpoint_provenance.json",
            {
                "path": str(inputs["checkpoint_path"]),
                "sha256": selection_config["checkpoint_sha256"],
                "epoch": checkpoint_payload["epoch"],
                "global_step": checkpoint_payload["global_step"],
                "git_commit": checkpoint_payload["git_commit"],
                "schema_version": checkpoint_payload["schema_version"],
                "adaptation": adaptation,
                "restore_verified": True,
            },
        )
        shutil.copyfile(inputs["manifest_path"], run_dir / "data_manifest.json")
        shutil.copyfile(
            inputs["split_manifest_path"],
            run_dir / "split_manifest.json",
        )
        shutil.copyfile(selection_dir / "selection.json", run_dir / "selection.lock.json")
        _write_json(
            run_dir / "efficiency.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "schemaflow_validation_reproduction_seconds": validation_seconds,
                "schemaflow_test_seconds": test_seconds,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else 0
                ),
                "peak_system_memory_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
            },
        )
        (run_dir / "git_commit.txt").write_text(
            git["commit"] + "\n",
            encoding="utf-8",
        )
        (run_dir / ".incomplete").unlink()
    except Exception:
        raise

    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "locked_policy_name": selection["locked_policy"]["name"],
                "test_metrics": metrics,
            },
            sort_keys=True,
        )
    )
    return 0


def _confirm(args) -> int:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.bootstrap_resamples < 200:
        raise SystemExit("bootstrap-resamples must be at least 200")
    if args.bootstrap_seed < 0:
        raise SystemExit("bootstrap-seed must be non-negative")
    output_root = _validate_output_root(args.output_root)
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    selection_dir = args.selection_dir.expanduser().resolve()
    selection_config, selection = _verify_selection(
        selection_dir,
        git,
        require_current_git=False,
    )
    _verify_confirmatory_policy(selection_config, selection)
    inputs = _load_experiment_inputs(
        Path(selection_config["training_config_path"]),
        Path(selection_config["checkpoint_path"]),
    )
    for key, path in (
        ("training_config_sha256", inputs["config_path"]),
        ("checkpoint_sha256", inputs["checkpoint_path"]),
        ("data_manifest_sha256", inputs["manifest_path"]),
        ("split_manifest_sha256", inputs["split_manifest_path"]),
    ):
        if _sha256_file(path) != selection_config[key]:
            raise RuntimeError(f"selection provenance changed: {key}")
    if args.device != selection_config["device"]:
        raise RuntimeError("confirmation device must match the selection device")

    cohort_dir = args.cohort_dir.expanduser().resolve()
    cohort_manifest_path = cohort_dir / "cohort_manifest.json"
    overlap_audit_path = cohort_dir / "overlap_audit.json"
    if not cohort_manifest_path.is_file() or not overlap_audit_path.is_file():
        raise RuntimeError("confirmatory cohort artifacts are incomplete")
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    overlap_audit = json.loads(overlap_audit_path.read_text(encoding="utf-8"))
    if cohort_manifest.get("selected_rows", 0) < 200_000:
        raise RuntimeError("confirmatory cohort must contain at least 200000 rows")
    if overlap_audit.get("verified_zero_overlap") is not True:
        raise RuntimeError("confirmatory cohort does not prove zero overlap")

    started = time.perf_counter()
    run_dir = _run_dir(
        output_root,
        "schemaflow-xgb-fusion-confirmation",
        git["commit"],
    )
    try:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train = _model_visible_samples(inputs["dataset"].splits["train"])
        encoder = TabularFeatureEncoder.fit(
            train,
            category_encoders=inputs["tokenizer"].categorical_vocabularies,
        )
        recorded_encoder = json.loads(
            (selection_dir / "encoder.json").read_text(encoding="utf-8")
        )
        if encoder.to_dict() != recorded_encoder:
            raise RuntimeError("reconstructed tabular encoder differs from selection")
        xgboost_model = XGBClassifier()
        xgboost_model.load_model(str(selection_dir / "xgboost_model.json"))
        schemaflow_model, checkpoint_payload, adaptation = _restore_schemaflow(
            inputs,
            device,
        )

        validation_conditions, validation_missingness = _conditioned_samples(
            inputs["dataset"].splits["val"],
            (0.0, 0.3, 0.5),
            selection_config["missing_seed"],
        )
        validation_outputs, validation_seconds = _collect_outputs(
            validation_conditions,
            encoder=encoder,
            xgboost_model=xgboost_model,
            schemaflow_model=schemaflow_model,
            tokenizer=inputs["tokenizer"],
            device=device,
            batch_size=inputs["config"].training["batch_size"],
        )
        recorded_hashes = json.loads(
            (selection_dir / "validation_probability_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        if _output_hashes(validation_outputs) != recorded_hashes:
            raise RuntimeError("validation probabilities do not reproduce selection")

        cohort = load_nf_ton_confirmatory_samples(
            cohort_dir,
            inputs["preprocessor_dir"],
            data_root=_DATA_ROOT,
        )
        if cohort.preprocessor_hash != inputs["dataset"].preprocessor_hash:
            raise RuntimeError("cohort did not use the frozen train-only preprocessor")
        cohort_conditions, cohort_missingness = _conditioned_samples(
            cohort.samples,
            (0.0, 0.3, 0.5),
            selection_config["missing_seed"],
        )
        cohort_outputs, cohort_seconds = _collect_outputs(
            cohort_conditions,
            encoder=encoder,
            xgboost_model=xgboost_model,
            schemaflow_model=schemaflow_model,
            tokenizer=inputs["tokenizer"],
            device=device,
            batch_size=inputs["config"].training["batch_size"],
        )

        confirmation_config = {
            "format_version": 1,
            "workflow": "schemaflow_xgboost_probability_fusion_confirmation",
            "phase": "confirmation",
            "evaluation_scope": "independent_group_disjoint_cohort",
            "selection_dir": str(selection_dir),
            "selection_sha256": _sha256_file(selection_dir / "selection.json"),
            "selection_git_commit": selection_config["git"]["commit"],
            "confirmation_git_commit": git["commit"],
            "cohort_dir": str(cohort_dir),
            "cohort_manifest_sha256": _sha256_file(cohort_manifest_path),
            "overlap_audit_sha256": _sha256_file(overlap_audit_path),
            "locked_policies": _CONFIRMATORY_POLICIES,
            "fixed_a0_policies": _FIXED_A0_POLICIES,
            "missing_seed": selection_config["missing_seed"],
            "bootstrap": {
                "method": (
                    "paired_poisson_cluster_bootstrap_linearized_macro_f1"
                ),
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
            },
            "gate": {
                "minimum_mean_xgboost_macro_f1_gain": MIN_MEAN_XGBOOST_GAIN,
                "bootstrap_mean_ci95_lower_must_exceed": 0.0,
                "missing_conditions_must_exceed_fixed_a0": True,
                "all_conditions_must_be_noninferior_to_xgboost": True,
                "rare_labels": list(RARE_NATIVE_LABELS),
                "maximum_rare_mean_recall_drop": MAX_RARE_MEAN_RECALL_DROP,
            },
            "prediction_lock_before_metrics": True,
            "device": args.device,
            "git": git,
        }
        _write_json(run_dir / "config.json", confirmation_config)
        locked_outputs, locked_group_ids, prediction_lock = _write_prediction_lock(
            run_dir,
            cohort_outputs,
            cohort.group_ids,
        )

        fixed_policies = selection["fixed_a0"]["policies"]
        locked_policies = selection["locked_policy"]["policies"]
        base_metrics = _confirmatory_base_metrics(locked_outputs)
        fixed_metrics = _confirmatory_policy_metrics(
            locked_outputs, fixed_policies
        )
        locked_metrics = _confirmatory_policy_metrics(
            locked_outputs, locked_policies
        )
        locked_probabilities = _locked_probabilities(
            locked_outputs, locked_policies
        )
        xgboost_probabilities = {
            condition: values["xgboost"]
            for condition, values in locked_outputs.items()
        }
        bootstrap = paired_cluster_bootstrap_macro_f1(
            locked_outputs,
            locked_probabilities,
            xgboost_probabilities,
            locked_group_ids,
            class_count=len(NF_TON_LABEL_SPACE.native_labels),
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        xgboost_metrics = {
            condition: values["xgboost"]
            for condition, values in base_metrics.items()
        }
        gate = evaluate_confirmatory_gate(
            xgboost_metrics=xgboost_metrics,
            fixed_a0_metrics=fixed_metrics,
            locked_metrics=locked_metrics,
            bootstrap=bootstrap,
        )
        metrics = {
            "format_version": 1,
            "evaluation_scope": "independent_group_disjoint_confirmation",
            "base_models": base_metrics,
            "fixed_a0": fixed_metrics,
            "locked_policy_name": selection["locked_policy"]["name"],
            "locked_policy": locked_metrics,
            "retention": {
                "xgboost": _retention(xgboost_metrics),
                "schemaflow": _retention(
                    {
                        condition: values["schemaflow"]
                        for condition, values in base_metrics.items()
                    }
                ),
                "fixed_a0": _retention(fixed_metrics),
                "locked_policy": _retention(locked_metrics),
            },
        }
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "bootstrap.json", bootstrap)
        _write_json(run_dir / "gate.json", gate)
        _write_json(
            run_dir / "missingness.json",
            {
                "validation_reproduction": validation_missingness,
                "confirmation": cohort_missingness,
            },
        )
        _write_json(run_dir / "label_space.json", NF_TON_LABEL_SPACE.to_dict())
        _write_json(run_dir / "environment.json", _environment())
        _write_json(
            run_dir / "checkpoint_provenance.json",
            {
                "path": str(inputs["checkpoint_path"]),
                "sha256": selection_config["checkpoint_sha256"],
                "epoch": checkpoint_payload["epoch"],
                "global_step": checkpoint_payload["global_step"],
                "git_commit": checkpoint_payload["git_commit"],
                "schema_version": checkpoint_payload["schema_version"],
                "adaptation": adaptation,
                "restore_verified": True,
            },
        )
        shutil.copyfile(
            cohort_manifest_path, run_dir / "cohort_manifest.json"
        )
        shutil.copyfile(overlap_audit_path, run_dir / "overlap_audit.json")
        shutil.copyfile(
            selection_dir / "selection.json", run_dir / "selection.lock.json"
        )
        shutil.copyfile(
            selection_dir / "tokenizer_metadata.json",
            run_dir / "tokenizer_metadata.json",
        )
        shutil.copyfile(
            selection_dir / "feature_registry.json",
            run_dir / "feature_registry.json",
        )
        _write_json(
            run_dir / "efficiency.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "schemaflow_validation_reproduction_seconds": validation_seconds,
                "schemaflow_confirmation_seconds": cohort_seconds,
                "prediction_archive_bytes": (run_dir / "predictions.npz").stat().st_size,
                "prediction_archive_sha256": prediction_lock[
                    "predictions_sha256"
                ],
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else 0
                ),
                "peak_system_memory_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
            },
        )
        (run_dir / "git_commit.txt").write_text(
            git["commit"] + "\n", encoding="ascii"
        )
        (run_dir / ".incomplete").unlink()
    except Exception:
        raise

    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "gate": gate,
                "full_matrix_allowed": gate["full_matrix_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _args()
    if args.phase == "select":
        return _select(args)
    if args.phase == "evaluate":
        return _evaluate(args)
    return _confirm(args)


if __name__ == "__main__":
    raise SystemExit(main())
