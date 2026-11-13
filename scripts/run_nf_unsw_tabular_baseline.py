#!/usr/bin/env python3
"""Run reproducible, train-only tabular baselines on NF-UNSW development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from flowfm.data.labels import (
    NF_TON_LABEL_SPACE,
    NF_UNSW_LABEL_SPACE,
    encode_supervised_targets,
)
from flowfm.data.missingness import apply_deterministic_field_missingness
from flowfm.data.nf_ton_dataset import load_nf_ton_development_samples
from flowfm.data.nf_ton_preprocessing import load_nf_ton_preprocessor
from flowfm.data.nf_ton_tokenization import build_nf_ton_tokenizer_metadata
from flowfm.data.nf_unsw_dataset import load_nf_unsw_development_samples
from flowfm.data.nf_unsw_preprocessing import load_nf_unsw_preprocessor
from flowfm.data.nf_unsw_tokenization import build_nf_unsw_tokenizer_metadata
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.training.baselines import (
    TabularFeatureEncoder,
    inverse_frequency_sample_weights,
)


_TASKS = ("binary", "coarse", "native")


def _dataset_components(name: str):
    if name == "nf_unsw":
        return (
            NF_UNSW_LABEL_SPACE,
            load_nf_unsw_development_samples,
            load_nf_unsw_preprocessor,
            build_nf_unsw_tokenizer_metadata,
        )
    if name == "nf_ton":
        return (
            NF_TON_LABEL_SPACE,
            load_nf_ton_development_samples,
            load_nf_ton_preprocessor,
            build_nf_ton_tokenizer_metadata,
        )
    raise ValueError(f"unsupported dataset: {name}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("nf_unsw", "nf_ton"),
        default="nf_unsw",
    )
    parser.add_argument(
        "--split-dir",
        default="/root/autodl-tmp/Res-AE-Attn/data/splits/NF-UNSW-NB15-v2/development-conflict-free-core-100k-seed42-91bec69",
    )
    parser.add_argument(
        "--preprocessor-dir",
        default="/root/autodl-tmp/Res-AE-Attn/data/processed/NF-UNSW-NB15-v2/development-train-only-preprocessor-d3e0749",
    )
    parser.add_argument(
        "--data-manifest",
        default="/root/autodl-tmp/Res-AE-Attn/data/manifests/NF-UNSW-NB15-v2.source.json",
    )
    parser.add_argument("--model", choices=("xgboost", "mlp"), required=True)
    parser.add_argument("--task", choices=("all", *_TASKS), default="all")
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/Res-AE-Attn/results",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--missing-fraction", type=float, default=0.0)
    parser.add_argument("--missing-seed", type=int, default=17)
    return parser.parse_args()


def _task_names(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return _TASKS
    if requested not in _TASKS:
        raise ValueError(f"unsupported task: {requested}")
    return (requested,)


def _task_labels(task: str, label_space) -> tuple[str, ...]:
    if task not in _TASKS:
        raise ValueError(f"unsupported task: {task}")
    return tuple(getattr(label_space, f"{task}_labels"))


def _model_visible_samples(samples):
    visible = {
        spec.name
        for spec in SCHEMAFLOW_MAIN_REGISTRY.features
        if spec.model_visible
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
                missing_mask=tuple(
                    sample.missing_mask[index] for index in indices
                ),
                category_tokens=tuple(
                    sample.category_tokens[index] for index in indices
                ),
            )
        )
    return tuple(projected)


def _target_arrays(samples, label_space) -> dict[str, np.ndarray]:
    encoded = encode_supervised_targets(
        tuple(samples),
        label_space,
    )
    return {
        task: getattr(encoded, task).cpu().numpy().astype(np.int64, copy=False)
        for task in _TASKS
    }


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if selected.any():
            accuracy = float((predictions[selected] == targets[selected]).mean())
            total += float(selected.mean()) * abs(
                accuracy - float(confidence[selected].mean())
            )
    return float(total)


def _metric_payload(
    model,
    matrix,
    targets: np.ndarray,
    label_names: tuple[str, ...],
) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("metric targets must be a non-empty rank-1 array")
    class_count = len(label_names)
    expected_classes = np.arange(class_count, dtype=np.int64)
    classes = np.asarray(getattr(model, "classes_", ()), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (targets.size, class_count)
        or not np.array_equal(classes, expected_classes)
    ):
        raise ValueError(
            "model probability columns must exactly match the versioned label space"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("model probabilities must be finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("model probability rows must sum to one")
    if np.any(targets < 0) or np.any(targets >= class_count):
        raise ValueError("metric targets fall outside the versioned label space")

    predictions = probabilities.argmax(axis=1)
    label_ids = expected_classes
    recalls = recall_score(
        targets,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    matrix_values = confusion_matrix(
        targets,
        predictions,
        labels=label_ids,
    )
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    per_class_pr_auc = {}
    for class_id, label in enumerate(label_names):
        positives = int(one_hot[:, class_id].sum())
        if positives == 0:
            score = 0.0
        elif positives == targets.size:
            score = 1.0
        else:
            score = float(
                average_precision_score(
                    one_hot[:, class_id],
                    probabilities[:, class_id],
                )
            )
        per_class_pr_auc[label] = score

    class_counts = np.bincount(targets, minlength=class_count)
    prediction_counts = np.bincount(predictions, minlength=class_count)
    return {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "balanced_accuracy": float(recalls.mean()),
        "macro_pr_auc": float(np.mean(tuple(per_class_pr_auc.values()))),
        "per_class_pr_auc": per_class_pr_auc,
        "ece": _ece(probabilities, targets),
        "brier": float(np.mean((probabilities - one_hot) ** 2)),
        "brier_definition": "mean_squared_probability_error_over_samples_and_classes",
        "per_class_recall": {
            label: float(recalls[index])
            for index, label in enumerate(label_names)
        },
        "confusion_matrix": matrix_values.tolist(),
        "class_counts": {
            label: int(class_counts[index])
            for index, label in enumerate(label_names)
        },
        "predicted_class_counts": {
            label: int(prediction_counts[index])
            for index, label in enumerate(label_names)
        },
        "sample_count": int(targets.size),
        "label_names": list(label_names),
    }


def _run_model(
    name,
    task,
    x_train,
    y_train,
    seed,
    max_iter,
    label_space,
):
    label_names = _task_labels(task, label_space)
    expected = np.arange(len(label_names), dtype=np.int64)
    observed = np.unique(y_train)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"{task} training labels do not cover the complete versioned label space"
        )
    weights = inverse_frequency_sample_weights(y_train)
    diagnostics = {
        "class_counts": {
            label: int(np.sum(y_train == index))
            for index, label in enumerate(label_names)
        },
        "sample_weight_min": float(weights.min()),
        "sample_weight_max": float(weights.max()),
        "sample_weight_mean": float(weights.mean()),
    }

    if name == "xgboost":
        parameters = {
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": min(os.cpu_count() or 1, 15),
            "random_state": seed,
        }
        if len(label_names) == 2:
            parameters.update(
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            parameters.update(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=len(label_names),
            )
        model = XGBClassifier(**parameters)
        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
            verbose=False,
        )
        training_policy = "train_only_inverse_frequency_sample_weight"
    else:
        probabilities = weights / weights.sum()
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            len(y_train),
            size=len(y_train),
            replace=True,
            p=probabilities,
        )
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        model.fit(x_train[indices], y_train[indices])
        diagnostics["bootstrap_unique_samples"] = int(np.unique(indices).size)
        training_policy = "deterministic_train_only_inverse_frequency_bootstrap"
    return model, training_policy, diagnostics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_manifest(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> int:
    args = _args()
    if (
        args.seed < 0
        or args.max_iter <= 0
        or args.missing_seed < 0
        or not 0.0 <= args.missing_fraction <= 1.0
    ):
        raise SystemExit("invalid seed, max-iter, or missing-fraction")
    tasks = _task_names(args.task)
    label_space, load_dataset, load_preprocessor, build_tokenizer = (
        _dataset_components(args.dataset)
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root).expanduser().resolve()
    data_root = Path("/root/autodl-tmp").resolve()
    if not output_root.is_relative_to(data_root):
        raise SystemExit("output-root must be located below /root/autodl-tmp")

    started = time.perf_counter()
    split_dir = Path(args.split_dir).expanduser().resolve()
    preprocessor_dir = Path(args.preprocessor_dir).expanduser().resolve()
    source_manifest = Path(args.data_manifest).expanduser().resolve()
    split_manifest_path = split_dir / "split_manifest.json"
    data_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    dataset = load_dataset(
        split_dir,
        preprocessor_dir,
        data_root=data_root,
    )
    train = _model_visible_samples(dataset.splits["train"])
    validation = _model_visible_samples(dataset.splits["val"])
    test = _model_visible_samples(dataset.splits["test"])
    eligible_feature_ids = tuple(train[0].feature_ids)
    missing_validation, validation_missingness = (
        apply_deterministic_field_missingness(
            validation,
            fraction=args.missing_fraction,
            seed=args.missing_seed,
            eligible_feature_ids=eligible_feature_ids,
        )
    )
    missing_test, test_missingness = apply_deterministic_field_missingness(
        test,
        fraction=args.missing_fraction,
        seed=args.missing_seed,
        eligible_feature_ids=eligible_feature_ids,
    )
    missingness = {
        "training_data_perturbed": False,
        "validation": validation_missingness,
        "test": test_missingness,
    }
    preprocessor = load_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer = build_tokenizer(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )
    encoder = TabularFeatureEncoder.fit(
        train,
        category_encoders=tokenizer.categorical_vocabularies,
    )
    matrices = {
        "train": encoder.transform(train),
        "validation": encoder.transform(validation),
        "test": encoder.transform(test),
        "missing_validation": encoder.transform(missing_validation),
        "missing_test": encoder.transform(missing_test),
    }
    targets = {
        "train": _target_arrays(train, label_space),
        "validation": _target_arrays(validation, label_space),
        "test": _target_arrays(test, label_space),
    }

    metrics = {}
    training = {}
    task_wall_seconds = {}
    for task in tasks:
        task_started = time.perf_counter()
        model, training_policy, diagnostics = _run_model(
            args.model,
            task,
            matrices["train"],
            targets["train"][task],
            args.seed,
            args.max_iter,
            label_space,
        )
        task_wall_seconds[task] = time.perf_counter() - task_started
        label_names = _task_labels(task, label_space)
        metrics[task] = {
            split: _metric_payload(
                model,
                matrices[split],
                targets[split][task],
                label_names,
            )
            for split in ("validation", "test")
        }
        metrics[task]["missing"] = {
            split: _metric_payload(
                model,
                matrices[f"missing_{split}"],
                targets[split][task],
                label_names,
            )
            for split in ("validation", "test")
        }
        training[task] = {
            "policy": training_policy,
            "diagnostics": diagnostics,
        }

    repo_dir = Path(__file__).resolve().parents[1]
    git_state = _git_state(repo_dir)
    config = {
        "format_version": 2,
        "dataset_id": label_space.dataset_id,
        "model": args.model,
        "task_request": args.task,
        "tasks": list(tasks),
        "seed": args.seed,
        "max_iter": args.max_iter,
        "fit_scope": "train_only",
        "training": training,
        "missingness": {
            "fraction": args.missing_fraction,
            "seed": args.missing_seed,
            "scope": ["validation", "test"],
            "training_data_perturbed": False,
        },
        "split_dir": str(split_dir),
        "preprocessor_dir": str(preprocessor_dir),
        "data_manifest_path": str(source_manifest),
        "data_manifest_sha256": _sha256_file(source_manifest),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "feature_count": int(matrices["train"].shape[1]),
        "output_features": encoder.to_dict()["output_feature_names"],
        "tokenizer_metadata_hash": tokenizer.metadata_hash,
        "label_space_hash": label_space.label_space_hash,
        "label_mapping_version": label_space.mapping_version,
        "categorical_table_sizes": {
            name: vocabulary.table_size
            for name, vocabulary in tokenizer.categorical_vocabularies.items()
        },
        "git": git_state,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / (
        f"baseline-{args.dataset.replace('_', '-')}-{args.model}-{args.task}-{timestamp}-"
        f"{config_hash}-{uuid.uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "encoder.json", encoder.to_dict())
    _write_json(run_dir / "missingness.json", missingness)
    _write_json(
        run_dir / "label_space.json",
        label_space.to_dict(),
    )
    _copy_manifest(source_manifest, run_dir / "data_manifest.json")
    _copy_manifest(split_manifest_path, run_dir / "split_manifest.json")
    _write_json(
        run_dir / "metrics.json",
        {
            "format_version": 2,
            "tasks": metrics,
        },
    )
    _write_json(
        run_dir / "predictions_summary.json",
        {
            task: {
                split: {
                    "true_class_counts": metrics[task][split]["class_counts"],
                    "predicted_class_counts": metrics[task][split][
                        "predicted_class_counts"
                    ],
                    "confusion_matrix": metrics[task][split][
                        "confusion_matrix"
                    ],
                }
                for split in ("validation", "test")
            }
            for task in tasks
        },
    )
    _write_json(
        run_dir / "missing_predictions_summary.json",
        {
            task: {
                split: {
                    "true_class_counts": metrics[task]["missing"][split][
                        "class_counts"
                    ],
                    "predicted_class_counts": metrics[task]["missing"][split][
                        "predicted_class_counts"
                    ],
                    "confusion_matrix": metrics[task]["missing"][split][
                        "confusion_matrix"
                    ],
                }
                for split in ("validation", "test")
            }
            for task in tasks
        },
    )


    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    _write_json(run_dir / "environment.json", environment)
    (run_dir / "git_commit.txt").write_text(
        git_state["commit"] + "\n",
        encoding="utf-8",
    )
    (run_dir / "schema_version.txt").write_text(
        label_space.schema_version + "\n",
        encoding="utf-8",
    )
    efficiency = {
        "wall_seconds": time.perf_counter() - started,
        "task_training_wall_seconds": task_wall_seconds,
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "matrix_columns": int(matrices["train"].shape[1]),
        "gpu_used": False,
    }
    _write_json(run_dir / "efficiency.json", efficiency)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model": args.model,
                "tasks": list(tasks),
                "metrics": metrics,
                "efficiency": efficiency,
                "status": "completed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
