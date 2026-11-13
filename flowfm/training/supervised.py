"""Deterministic supervised finetuning for SchemaFlow-FM."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)

from flowfm.config import FlowConfig
from flowfm.data.contracts import FlowSample
from flowfm.data.identity import stable_hash
from flowfm.data.labels import (
    DatasetLabelSpace,
    NF_CSE_LABEL_SPACE,
    NF_TON_LABEL_SPACE,
    NF_UNSW_LABEL_SPACE,
    encode_supervised_targets,
)
from flowfm.data.nf_cse_dataset import load_nf_cse_development_samples
from flowfm.data.nf_cse_few_shot import SELECTION_MANIFEST_FILENAME
from flowfm.data.nf_cse_preprocessing import load_nf_cse_preprocessor
from flowfm.data.nf_cse_tokenization import build_nf_cse_tokenizer_metadata
from flowfm.data.nf_unsw_dataset import load_nf_unsw_development_samples
from flowfm.data.nf_unsw_preprocessing import load_nf_unsw_preprocessor
from flowfm.data.nf_unsw_tokenization import build_nf_unsw_tokenizer_metadata
from flowfm.data.nf_ton_dataset import load_nf_ton_development_samples
from flowfm.data.nf_ton_preprocessing import load_nf_ton_preprocessor
from flowfm.data.nf_ton_tokenization import build_nf_ton_tokenizer_metadata
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.experiment.logging import JsonlLogger
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.experiment.runs import create_run_context
from flowfm.models.supervised import SchemaFlowSupervisedModel
from flowfm.training.checkpoints import (
    load_checkpoint,
    load_pretrained_encoder,
    save_checkpoint,
)
from flowfm.training.safe_fusion import (
    balanced_softmax_cross_entropy,
    deterministic_paired_missing_view,
    jensen_shannon_consistency,
)


_DATA_MANIFEST_HASH_NAMESPACE = "flowfm-training-data-manifest-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def validate_supervised_splits(splits, label_space: DatasetLabelSpace):
    if not isinstance(splits, Mapping) or not {"train", "val"}.issubset(splits):
        raise ValueError("splits must contain train and val")
    normalized = {}
    seen_sample_ids = set()
    feature_keys_by_split = {}
    split_names = ("train", "val") + (("test",) if "test" in splits else ())
    for split_name in split_names:
        values = splits[split_name]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{split_name} split must be non-empty")
        samples = tuple(values)
        split_feature_keys = set()
        for sample in samples:
            if not isinstance(sample, FlowSample):
                raise TypeError("training splits must contain FlowSample values")
            if sample.dataset_id != label_space.dataset_id:
                raise ValueError("training sample dataset does not match label space")
            if sample.sample_id in seen_sample_ids:
                raise ValueError("sample_id crosses supervised splits")
            seen_sample_ids.add(sample.sample_id)
            split_feature_keys.add(sample.feature_key)
        encode_supervised_targets(samples, label_space)
        normalized[split_name] = samples
        feature_keys_by_split[split_name] = split_feature_keys
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if feature_keys_by_split[first] & feature_keys_by_split[second]:
                raise ValueError("feature_key crosses supervised splits")
    return normalized


def _model_data_metadata(registry, tokenizer_metadata, label_space):
    return {
        "format_version": 2,
        "feature_registry": registry.to_dict(),
        "tokenizer_metadata": tokenizer_metadata.to_dict(),
        "categorical_table_sizes": {
            name: vocabulary.table_size
            for name, vocabulary in tokenizer_metadata.categorical_vocabularies.items()
        },
        "label_spaces": {label_space.dataset_id: label_space.to_dict()},
    }


def _build_model(config: FlowConfig, tokenizer_metadata, label_space, registry):
    values = config.model
    return SchemaFlowSupervisedModel(
        tokenizer_metadata,
        {label_space.dataset_id: label_space},
        registry=registry,
        d_model=values["d_model"],
        nhead=values["nhead"],
        num_layers=values["num_layers"],
        dim_feedforward=values["dim_feedforward"],
        dropout=values["dropout"],
        adapter_dim=values["adapter_dim"],
    )


def _configure_adaptation(model, mode):
    if mode not in {"full", "adapter_category_heads"}:
        raise ValueError(f"unsupported adaptation mode: {mode}")
    if mode == "adapter_category_heads":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.encoder.category_embeddings.parameters():
            parameter.requires_grad_(True)

    trainable_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if not trainable_names:
        raise RuntimeError("adaptation policy left no trainable parameters")
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_category_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.category_embeddings.") and parameter.requires_grad
    )
    return {
        "mode": mode,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_category_parameter_count": trainable_category_parameter_count,
        "trainable_parameter_names": list(trainable_names),
    }


def _class_weights(samples, label_space, device, strategy):
    targets = encode_supervised_targets(samples, label_space)
    result = {}
    counts_payload = {}
    count_tensors = {}
    for name, values, labels in (
        ("binary", targets.binary, label_space.binary_labels),
        ("coarse", targets.coarse, label_space.coarse_labels),
        ("native", targets.native, label_space.native_labels),
    ):
        counts = torch.bincount(values, minlength=len(labels)).to(torch.float64)
        counts_payload[name] = {
            label: int(counts[index].item()) for index, label in enumerate(labels)
        }
        count_tensors[name] = counts.to(dtype=torch.float32, device=device)
        if strategy == "none":
            weights = torch.ones(len(labels), dtype=torch.float32)
        else:
            weights = torch.zeros(len(labels), dtype=torch.float64)
            present = counts > 0
            weights[present] = counts[present].sum() / (
                present.sum() * counts[present]
            )
            weights = weights.to(torch.float32)
        result[name] = weights.to(device)
    return result, count_tensors, counts_payload


def _weighted_loss(
    output,
    targets,
    class_weights,
    class_counts,
    loss_weights,
    classification_loss,
):
    if classification_loss == "balanced_softmax":
        loss = balanced_softmax_cross_entropy
        binary = loss(output.binary_logits, targets.binary, class_counts["binary"])
        coarse = loss(output.coarse_logits, targets.coarse, class_counts["coarse"])
        native = loss(output.native_logits, targets.native, class_counts["native"])
    elif classification_loss == "weighted_cross_entropy":
        binary = torch.nn.functional.cross_entropy(
            output.binary_logits, targets.binary, weight=class_weights["binary"]
        )
        coarse = torch.nn.functional.cross_entropy(
            output.coarse_logits, targets.coarse, weight=class_weights["coarse"]
        )
        native = torch.nn.functional.cross_entropy(
            output.native_logits, targets.native, weight=class_weights["native"]
        )
    else:
        raise ValueError(f"unsupported classification loss: {classification_loss}")
    total = (
        loss_weights["binary"] * binary
        + loss_weights["coarse"] * coarse
        + loss_weights["native"] * native
    )
    return total, {"binary": binary, "coarse": coarse, "native": native}


def _batch_order(size, *, shuffle, seed, epoch):
    if not shuffle:
        return list(range(size))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).tolist()


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0
        if selected.any():
            accuracy = (predictions[selected] == targets[selected]).mean()
            value += selected.mean() * abs(accuracy - confidence[selected].mean())
    return float(value)


def _task_metrics(logits, targets, labels, *, binary=False):
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    label_ids = np.arange(len(labels))
    recalls = recall_score(
        targets,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    result = {
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
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "per_class_recall": {
            label: float(recalls[index]) for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=label_ids
        ).tolist(),
        "sample_count": int(targets.shape[0]),
    }
    if binary:
        positive_id = labels.index("attack")
        positive_targets = (targets == positive_id).astype(np.int64)
        result.update(
            {
                "pr_auc": float(
                    average_precision_score(
                        positive_targets, probabilities[:, positive_id]
                    )
                ),
                "ece": _ece(probabilities, targets),
                "brier": float(
                    np.mean(
                        (probabilities[:, positive_id] - positive_targets) ** 2
                    )
                ),
            }
        )
    return result


def _evaluate(model, samples, metadata, label_space, registry, device, batch_size):
    model.eval()
    logits = {"binary": [], "coarse": [], "native": []}
    targets = {"binary": [], "coarse": [], "native": []}
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            members = samples[start : start + batch_size]
            batch = collate_schemaflow_samples(
                members, metadata, registry=registry
            ).to(device)
            encoded = encode_supervised_targets(
                members, label_space, device=device
            )
            output = model(batch)
            for name, values in (
                ("binary", output.binary_logits),
                ("coarse", output.coarse_logits),
                ("native", output.native_logits),
            ):
                logits[name].append(values.detach().cpu())
                targets[name].append(getattr(encoded, name).detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    arrays = {
        name: torch.cat(values).numpy() for name, values in logits.items()
    }
    target_arrays = {
        name: torch.cat(values).numpy() for name, values in targets.items()
    }
    metrics = {
        "binary": _task_metrics(
            arrays["binary"],
            target_arrays["binary"],
            label_space.binary_labels,
            binary=True,
        ),
        "coarse": _task_metrics(
            arrays["coarse"],
            target_arrays["coarse"],
            label_space.coarse_labels,
        ),
        "native": _task_metrics(
            arrays["native"],
            target_arrays["native"],
            label_space.native_labels,
        ),
    }
    return metrics, elapsed


def _write_text(path: Path, value: str) -> None:
    path.write_text(value if value.endswith("\n") else value + "\n", encoding="utf-8")


def run_supervised_training(
    config: FlowConfig,
    *,
    splits,
    tokenizer_metadata: TokenizerMetadata,
    label_space: DatasetLabelSpace,
    registry: CanonicalFeatureRegistry,
    data_manifest: Mapping,
    split_manifest: Mapping,
    repo_dir,
    data_root=Path("/root/autodl-tmp"),
    git_commit: str,
    config_path=None,
    resume_from=None,
    initialize_from=None,
    allow_cross_dataset_init=False,
):
    if not isinstance(config, FlowConfig) or config.workflow != "finetune":
        raise TypeError("config must be a validated finetune FlowConfig")
    if config.runtime.get("execution", "validate_only") != "train":
        raise ValueError("supervised training requires runtime execution train")
    if config.schema["version"] != tokenizer_metadata.schema_version:
        raise ValueError("config Schema does not match tokenizer metadata")
    if label_space.schema_version != tokenizer_metadata.schema_version:
        raise ValueError("label space Schema does not match tokenizer metadata")
    normalized_splits = validate_supervised_splits(splits, label_space)
    if not isinstance(data_manifest, Mapping) or not data_manifest:
        raise ValueError("data_manifest must be a non-empty mapping")
    if not isinstance(split_manifest, Mapping) or not split_manifest:
        raise ValueError("split_manifest must be a non-empty mapping")
    if config.data["dataset_id"] != label_space.dataset_id:
        raise ValueError("config dataset_id does not match label space")
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    adaptation_mode = config.training.get("adaptation_mode", "full")
    if (
        adaptation_mode != "full"
        and resume_from is None
        and initialize_from is None
    ):
        raise ValueError("parameter-efficient adaptation requires a checkpoint")

    device = _resolve_device(config.runtime["device"])
    seed = config.experiment["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    dataset_hash = stable_hash(
        dict(data_manifest), _DATA_MANIFEST_HASH_NAMESPACE
    )
    context = create_run_context(
        config,
        repo_dir,
        dataset_hash=dataset_hash,
        split_hash=tokenizer_metadata.split_manifest_sha256,
        data_root=data_root,
    )
    marker = context.run_dir / ".incomplete"
    marker.write_text("supervised training incomplete\n", encoding="utf-8")
    logger = JsonlLogger(context.run_dir / "train.log")
    model_data_metadata = _model_data_metadata(
        registry, tokenizer_metadata, label_space
    )

    try:
        config_yaml = yaml.safe_dump(
            config.to_dict(), sort_keys=True, allow_unicode=True
        )
        _write_text(context.run_dir / "config.yaml", config_yaml)
        _write_text(context.run_dir / "git_commit.txt", git_commit)
        environment = json.loads(
            context.environment_snapshot.read_text(encoding="utf-8")
        )
        _write_text(
            context.run_dir / "environment.txt",
            "\n".join(
                f"{key}={environment.get(key)}"
                for key in (
                    "git_commit",
                    "git_dirty",
                    "python_version",
                    "platform",
                    "torch_version",
                    "cuda_runtime",
                    "cuda_available",
                )
            ),
        )
        write_json_snapshot(context.run_dir / "data_manifest.json", data_manifest)
        write_json_snapshot(context.run_dir / "split_manifest.json", split_manifest)
        _write_text(
            context.run_dir / "schema_version.txt",
            tokenizer_metadata.schema_version,
        )

        model = _build_model(
            config, tokenizer_metadata, label_space, registry
        ).to(device)
        initialization = None
        if initialize_from is not None:
            initialization = load_pretrained_encoder(
                initialize_from,
                model.encoder,
                map_location=device,
                expected_model_data_metadata=model_data_metadata,
                expected_model_config=config.model,
                allow_cross_dataset_init=allow_cross_dataset_init,
            )
            write_json_snapshot(
                context.run_dir / "pretrained_initialization.json",
                initialization,
            )
            logger.log(
                {"event": "pretrained_encoder_loaded", **initialization}
            )
        adaptation = _configure_adaptation(model, adaptation_mode)
        write_json_snapshot(
            context.run_dir / "adaptation.json",
            adaptation,
        )
        logger.log({"event": "adaptation_configured", **adaptation})
        trainable_parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.training["learning_rate"],
            weight_decay=config.training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.training["scheduler_gamma"]
        )
        class_weights, class_count_tensors, class_counts = _class_weights(
            normalized_splits["train"],
            label_space,
            device,
            config.training["class_balance"],
        )
        classification_loss = config.training.get(
            "classification_loss", "weighted_cross_entropy"
        )
        consistency_weight = float(config.training.get("consistency_weight", 0.0))
        paired_train = None
        consistency_report = None
        if consistency_weight > 0.0:
            paired_train, consistency_report = deterministic_paired_missing_view(
                normalized_splits["train"],
                seed=config.training.get("consistency_missing_seed", 0),
            )
            write_json_snapshot(
                context.run_dir / "paired_consistency.json", consistency_report
            )
        early_stopping_patience = config.training.get(
            "early_stopping_patience", 0
        )
        early_stopping_interval = config.training.get(
            "early_stopping_interval", 1
        )
        early_stopping_min_delta = float(
            config.training.get("early_stopping_min_delta", 0.0)
        )
        training_objective = {
            "format_version": 1,
            "classification_loss": classification_loss,
            "class_balance": config.training["class_balance"],
            "class_counts_scope": "target_train_only",
            "paired_consistency": {
                "enabled": paired_train is not None,
                "task": "coarse",
                "divergence": "jensen_shannon",
                "weight": consistency_weight,
                "scope": "target_train_only",
                "report_sha256": (
                    _sha256_file(context.run_dir / "paired_consistency.json")
                    if consistency_report is not None
                    else None
                ),
            },
            "early_stopping": {
                "enabled": early_stopping_patience > 0,
                "selection_scope": "validation_only",
                "task": "coarse",
                "metric": "macro_f1",
                "patience": early_stopping_patience,
                "interval_epochs": early_stopping_interval,
                "minimum_delta": early_stopping_min_delta,
            },
        }
        write_json_snapshot(
            context.run_dir / "training_objective.json", training_objective
        )
        start_epoch = 0
        global_step = 0
        if resume_from is not None and early_stopping_patience > 0:
            raise ValueError("resuming a validation-early-stopped run is unsupported")
        if resume_from is not None:
            payload = load_checkpoint(
                resume_from,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                expected_model_data_metadata=model_data_metadata,
            )
            checkpoint_adaptation = payload["extra_state"].get("adaptation")
            if checkpoint_adaptation is not None and checkpoint_adaptation != adaptation:
                raise ValueError("checkpoint adaptation policy does not match config")
            if (
                adaptation_mode != "full" and checkpoint_adaptation is None
            ):
                raise ValueError("adapter checkpoint is missing adaptation provenance")
            start_epoch = payload["epoch"]
            global_step = payload["global_step"]
        epochs_to_run = config.training["epochs"]
        end_epoch = start_epoch + epochs_to_run

        training_started = time.perf_counter()
        epoch_loss = None
        epoch_classification_loss = None
        epoch_consistency_loss = None
        early_validation_seconds = 0.0
        early_stopping_history = []
        best_model_state = None
        best_optimizer_state = None
        best_scheduler_state = None
        best_validation_score = None
        best_epoch = None
        best_global_step = None
        best_train_losses = None
        checks_without_improvement = 0
        stopped_early = False
        completed_epoch = start_epoch
        for epoch in range(start_epoch + 1, end_epoch + 1):
            model.train()
            ordered = _batch_order(
                len(normalized_splits["train"]),
                shuffle=config.training["shuffle"],
                seed=seed,
                epoch=epoch,
            )
            total_loss = 0.0
            total_classification_loss = 0.0
            total_consistency_loss = 0.0
            total_samples = 0
            for offset in range(0, len(ordered), config.training["batch_size"]):
                indices = ordered[offset : offset + config.training["batch_size"]]
                members = tuple(normalized_splits["train"][index] for index in indices)
                batch = collate_schemaflow_samples(
                    members, tokenizer_metadata, registry=registry
                ).to(device)
                targets = encode_supervised_targets(
                    members, label_space, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                classification, task_losses = _weighted_loss(
                    output,
                    targets,
                    class_weights,
                    class_count_tensors,
                    config.training["loss_weights"],
                    classification_loss,
                )
                consistency = torch.zeros((), dtype=classification.dtype, device=device)
                if paired_train is not None:
                    missing_members = tuple(paired_train[index] for index in indices)
                    missing_batch = collate_schemaflow_samples(
                        missing_members, tokenizer_metadata, registry=registry
                    ).to(device)
                    missing_output = model(missing_batch)
                    consistency = jensen_shannon_consistency(
                        output.coarse_logits, missing_output.coarse_logits
                    )
                loss = classification + consistency_weight * consistency
                if not torch.isfinite(loss).item():
                    raise RuntimeError("training loss became non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.training["gradient_clip"]
                )
                optimizer.step()
                global_step += 1
                total_loss += float(loss.detach().cpu().item()) * len(members)
                total_classification_loss += (
                    float(classification.detach().cpu().item()) * len(members)
                )
                total_consistency_loss += (
                    float(consistency.detach().cpu().item()) * len(members)
                )
                total_samples += len(members)
                logger.log(
                    {
                        "event": "train_batch",
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_samples": len(members),
                        "loss": float(loss.detach().cpu().item()),
                        "classification_loss": float(
                            classification.detach().cpu().item()
                        ),
                        "consistency_loss": float(
                            consistency.detach().cpu().item()
                        ),
                        "consistency_weight": consistency_weight,
                        "binary_loss": float(task_losses["binary"].detach().cpu().item()),
                        "coarse_loss": float(task_losses["coarse"].detach().cpu().item()),
                        "native_loss": float(task_losses["native"].detach().cpu().item()),
                    }
                )
            scheduler.step()
            epoch_loss = total_loss / total_samples
            epoch_classification_loss = total_classification_loss / total_samples
            epoch_consistency_loss = total_consistency_loss / total_samples
            completed_epoch = epoch
            logger.log(
                {
                    "event": "train_epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": epoch_loss,
                    "classification_loss": epoch_classification_loss,
                    "consistency_loss": epoch_consistency_loss,
                }
            )
            should_check = early_stopping_patience > 0 and (
                (epoch - start_epoch) % early_stopping_interval == 0
                or epoch == end_epoch
            )
            if should_check:
                early_metrics, elapsed = _evaluate(
                    model,
                    normalized_splits["val"],
                    tokenizer_metadata,
                    label_space,
                    registry,
                    device,
                    config.training["batch_size"],
                )
                early_validation_seconds += elapsed
                score = float(early_metrics["coarse"]["macro_f1"])
                improved = (
                    best_validation_score is None
                    or score > best_validation_score + early_stopping_min_delta
                )
                if improved:
                    best_validation_score = score
                    best_epoch = epoch
                    best_model_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_scheduler_state = copy.deepcopy(scheduler.state_dict())
                    best_global_step = global_step
                    best_train_losses = {
                        "total": epoch_loss,
                        "classification": epoch_classification_loss,
                        "consistency": epoch_consistency_loss,
                    }
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                early_stopping_history.append(
                    {
                        "epoch": epoch,
                        "coarse_macro_f1": score,
                        "coarse_mcc": float(early_metrics["coarse"]["mcc"]),
                        "improved": improved,
                        "checks_without_improvement": checks_without_improvement,
                    }
                )
                logger.log(
                    {
                        "event": "validation_early_stopping_check",
                        **early_stopping_history[-1],
                    }
                )
                if checks_without_improvement >= early_stopping_patience:
                    stopped_early = True
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = (
            time.perf_counter() - training_started - early_validation_seconds
        )
        last_trained_epoch = completed_epoch
        last_trained_global_step = global_step
        if early_stopping_patience > 0:
            if (
                best_model_state is None
                or best_optimizer_state is None
                or best_scheduler_state is None
                or best_epoch is None
                or best_global_step is None
                or best_train_losses is None
            ):
                raise RuntimeError("early stopping did not select a validation checkpoint")
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_optimizer_state)
            scheduler.load_state_dict(best_scheduler_state)
            completed_epoch = best_epoch
            global_step = best_global_step
            epoch_loss = best_train_losses["total"]
            epoch_classification_loss = best_train_losses["classification"]
            epoch_consistency_loss = best_train_losses["consistency"]
        early_stopping = {
            **training_objective["early_stopping"],
            "tie_break": "earlier_epoch",
            "maximum_epoch": end_epoch,
            "last_trained_epoch": last_trained_epoch,
            "last_trained_global_step": last_trained_global_step,
            "selected_epoch": best_epoch if best_epoch is not None else completed_epoch,
            "selected_coarse_macro_f1": best_validation_score,
            "stopped_early": stopped_early,
            "validation_seconds_during_selection": early_validation_seconds,
            "history": early_stopping_history,
        }

        validation_metrics, validation_seconds = _evaluate(
            model,
            normalized_splits["val"],
            tokenizer_metadata,
            label_space,
            registry,
            device,
            config.training["batch_size"],
        )
        evaluation_scope = config.training.get(
            "evaluation_scope", "validation_and_test"
        )
        test_metrics = None
        test_seconds = 0.0
        if evaluation_scope == "validation_and_test" and "test" in normalized_splits:
            test_metrics, test_seconds = _evaluate(
                model,
                normalized_splits["test"],
                tokenizer_metadata,
                label_space,
                registry,
                device,
                config.training["batch_size"],
            )
        metrics = {
            "epoch": completed_epoch,
            "global_step": global_step,
            "train_loss": epoch_loss,
            "train_classification_loss": epoch_classification_loss,
            "train_consistency_loss": epoch_consistency_loss,
            "validation": validation_metrics,
            "class_counts_train_only": class_counts,
            "evaluation_scope": evaluation_scope,
            "training_objective": training_objective,
            "early_stopping": early_stopping,
        }
        if test_metrics is not None:
            metrics["test"] = test_metrics
        write_json_snapshot(context.run_dir / "metrics.json", metrics)
        write_json_snapshot(
            context.predictions / "validation_summary.json",
            {
                "sample_count": len(normalized_splits["val"]),
                "metrics": validation_metrics,
            },
        )
        if test_metrics is not None:
            write_json_snapshot(
                context.predictions / "test_summary.json",
                {
                    "sample_count": len(normalized_splits["test"]),
                    "metrics": test_metrics,
                },
            )

        checkpoint_path = context.checkpoints / f"epoch-{completed_epoch:04d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=completed_epoch,
            global_step=global_step,
            git_commit=git_commit,
            schema_version=tokenizer_metadata.schema_version,
            seed=seed,
            config=config.to_dict(),
            model_data_metadata=model_data_metadata,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state={
                "metrics": metrics,
                "initialization": initialization,
                "adaptation": adaptation,
                "training_objective": training_objective,
                "early_stopping": early_stopping,
            },
        )

        verification_model = _build_model(
            config, tokenizer_metadata, label_space, registry
        ).to(device)
        verification_adaptation = _configure_adaptation(
            verification_model, adaptation_mode
        )
        if verification_adaptation != adaptation:
            raise RuntimeError("verification adaptation policy changed")
        verification_trainable_parameters = tuple(
            parameter
            for parameter in verification_model.parameters()
            if parameter.requires_grad
        )
        verification_optimizer = torch.optim.AdamW(
            verification_trainable_parameters,
            lr=config.training["learning_rate"],
            weight_decay=config.training["weight_decay"],
        )
        verification_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            verification_optimizer, gamma=config.training["scheduler_gamma"]
        )
        verified_payload = load_checkpoint(
            checkpoint_path,
            verification_model,
            optimizer=verification_optimizer,
            scheduler=verification_scheduler,
            map_location=device,
            restore_rng=False,
            expected_model_data_metadata=model_data_metadata,
        )
        checkpoint_verified = (
            verified_payload["epoch"] == completed_epoch
            and verified_payload["global_step"] == global_step
            and verified_payload["git_commit"] == git_commit
            and verified_payload["extra_state"].get("adaptation") == adaptation
            and verified_payload["extra_state"].get("training_objective")
            == training_objective
            and verified_payload["extra_state"].get("early_stopping")
            == early_stopping
        )
        if not checkpoint_verified:
            raise RuntimeError("saved checkpoint failed verification")

        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        efficiency = {
            "parameter_count": adaptation["total_parameter_count"],
            "trainable_parameter_count": adaptation["trainable_parameter_count"],
            "adaptation_mode": adaptation_mode,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "validation_selection_seconds": early_validation_seconds,
            "validation_selection_evaluations": len(early_stopping_history),
            "train_samples_per_second": len(normalized_splits["train"]) / train_seconds,
            "validation_samples_per_second": (
                len(normalized_splits["val"]) / validation_seconds
            ),
            "validation_seconds_per_sample": (
                validation_seconds / len(normalized_splits["val"])
            ),
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "peak_system_memory_bytes": peak_rss_bytes,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        }
        if test_metrics is not None:
            efficiency["test_seconds"] = test_seconds
            efficiency["test_samples_per_second"] = (
                len(normalized_splits["test"]) / test_seconds
            )
            efficiency["test_seconds_per_sample"] = (
                test_seconds / len(normalized_splits["test"])
            )
        write_json_snapshot(context.run_dir / "efficiency.json", efficiency)
        checkpoint_index = {
            "latest": checkpoint_path.name,
            "epoch": completed_epoch,
            "global_step": global_step,
            "sha256": _sha256_file(checkpoint_path),
            "verified": checkpoint_verified,
        }
        write_json_snapshot(context.checkpoints / "index.json", checkpoint_index)
        effective_config_path = (
            str(Path(config_path).expanduser().resolve())
            if config_path is not None
            else "<CONFIG_PATH>"
        )
        if early_stopping_patience > 0:
            resume_text = (
                "resume disabled by protocol: this checkpoint is the "
                "validation-selected early-stopping state"
            )
        else:
            resume_text = (
                "python -m flowfm finetune --config {}{} --resume-from {}".format(
                    effective_config_path,
                    " --allow-cross-dataset-init" if allow_cross_dataset_init else "",
                    checkpoint_path,
                )
            )
        _write_text(context.run_dir / "resume_command.txt", resume_text)
        logger.log(
            {
                "event": "training_completed",
                "epoch": completed_epoch,
                "global_step": global_step,
                "checkpoint": str(checkpoint_path),
                "checkpoint_verified": checkpoint_verified,
            }
        )
        marker.unlink()
        return {
            "status": "completed",
            "training_started": True,
            "run_dir": str(context.run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_verified": checkpoint_verified,
            "epochs_completed": completed_epoch,
            "last_trained_epoch": last_trained_epoch,
            "global_step": global_step,
            "metrics": metrics,
            "efficiency": efficiency,
            "adaptation": adaptation,
            "initialization": initialization,
            "training_objective": training_objective,
            "early_stopping": early_stopping,
        }
    except Exception:
        raise


def _clean_git_commit(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("refusing real training from a dirty Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("unable to resolve a full Git commit for training")
    return commit


def run_nf_unsw_finetune(
    config: FlowConfig,
    *,
    repo_dir,
    config_path,
    resume_from=None,
    initialize_from=None,
    allow_cross_dataset_init=False,
    data_root=Path("/root/autodl-tmp"),
):
    if config.data["dataset_id"] != NF_UNSW_LABEL_SPACE.dataset_id:
        raise ValueError("NF-UNSW finetune requires the NF-UNSW dataset_id")
    split_dir = validate_output_path(
        config.data["split_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    preprocessor_dir = validate_output_path(
        config.data["preprocessor_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    manifest_path = validate_output_path(
        config.data["data_manifest"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not manifest_path.is_file():
        raise ValueError("data Manifest path does not exist")
    try:
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_manifest = json.loads(
            (split_dir / "split_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError("invalid finetune Manifest input") from exc
    dataset = load_nf_unsw_development_samples(
        split_dir,
        preprocessor_dir,
        data_root=data_root,
    )
    preprocessor = load_nf_unsw_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer_metadata = build_nf_unsw_tokenizer_metadata(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )
    git_commit = _clean_git_commit(Path(repo_dir).expanduser().resolve())
    return run_supervised_training(
        config,
        splits=dataset.splits,
        tokenizer_metadata=tokenizer_metadata,
        label_space=NF_UNSW_LABEL_SPACE,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        repo_dir=repo_dir,
        data_root=data_root,
        git_commit=git_commit,
        config_path=config_path,
        resume_from=resume_from,
        initialize_from=initialize_from,
        allow_cross_dataset_init=allow_cross_dataset_init,
    )


def run_nf_ton_finetune(
    config: FlowConfig,
    *,
    repo_dir,
    config_path,
    resume_from=None,
    initialize_from=None,
    allow_cross_dataset_init=False,
    data_root=Path("/root/autodl-tmp"),
):
    if config.data["dataset_id"] != NF_TON_LABEL_SPACE.dataset_id:
        raise ValueError("NF-ToN finetune requires the NF-ToN dataset_id")
    split_dir = validate_output_path(
        config.data["split_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    preprocessor_dir = validate_output_path(
        config.data["preprocessor_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    manifest_path = validate_output_path(
        config.data["data_manifest"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not manifest_path.is_file():
        raise ValueError("data Manifest path does not exist")
    try:
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_manifest = json.loads(
            (split_dir / "split_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError("invalid finetune Manifest input") from exc
    dataset = load_nf_ton_development_samples(
        split_dir,
        preprocessor_dir,
        data_root=data_root,
    )
    preprocessor = load_nf_ton_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer_metadata = build_nf_ton_tokenizer_metadata(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )
    git_commit = _clean_git_commit(Path(repo_dir).expanduser().resolve())
    return run_supervised_training(
        config,
        splits=dataset.splits,
        tokenizer_metadata=tokenizer_metadata,
        label_space=NF_TON_LABEL_SPACE,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        repo_dir=repo_dir,
        data_root=data_root,
        git_commit=git_commit,
        config_path=config_path,
        resume_from=resume_from,
        initialize_from=initialize_from,
        allow_cross_dataset_init=allow_cross_dataset_init,
    )


def run_nf_cse_finetune(
    config: FlowConfig,
    *,
    repo_dir,
    config_path,
    resume_from=None,
    initialize_from=None,
    allow_cross_dataset_init=False,
    data_root=Path("/root/autodl-tmp"),
):
    if config.data["dataset_id"] != NF_CSE_LABEL_SPACE.dataset_id:
        raise ValueError("NF-CSE finetune requires the NF-CSE dataset_id")
    safe_development = os.environ.get("FLOWFM_NF_CSE_SAFE_DEVELOPMENT") == "1"
    if (
        safe_development
        and config.training.get("evaluation_scope") != "validation_only"
    ):
        raise RuntimeError(
            "A0-MR-Safe development blocks non-validation NF-CSE evaluation"
        )
    split_dir = validate_output_path(
        config.data["split_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    preprocessor_dir = validate_output_path(
        config.data["preprocessor_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    manifest_path = validate_output_path(
        config.data["data_manifest"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not manifest_path.is_file():
        raise ValueError("data Manifest path does not exist")
    requested_splits = (
        ("train", "val")
        if config.training.get("evaluation_scope", "validation_and_test")
        == "validation_only"
        else ("train", "val", "test")
    )
    split_config_path = split_dir / "config.snapshot.json"
    try:
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_config = json.loads(split_config_path.read_text(encoding="utf-8"))
        selection_view = (
            requested_splits == ("train", "val")
            and split_config.get("sealed_test") is True
        )
        if safe_development and not selection_view:
            raise RuntimeError(
                "A0-MR-Safe development requires the sealed selection view"
            )
        split_manifest_path = split_dir / (
            SELECTION_MANIFEST_FILENAME
            if selection_view
            else "split_manifest.json"
        )
        split_manifest = json.loads(
            split_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError("invalid finetune Manifest input") from exc
    dataset = load_nf_cse_development_samples(
        split_dir,
        preprocessor_dir,
        data_root=data_root,
        split_names=requested_splits,
    )
    preprocessor = load_nf_cse_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer_metadata = build_nf_cse_tokenizer_metadata(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )
    git_commit = _clean_git_commit(Path(repo_dir).expanduser().resolve())
    return run_supervised_training(
        config,
        splits=dataset.splits,
        tokenizer_metadata=tokenizer_metadata,
        label_space=NF_CSE_LABEL_SPACE,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        repo_dir=repo_dir,
        data_root=data_root,
        git_commit=git_commit,
        config_path=config_path,
        resume_from=resume_from,
        initialize_from=initialize_from,
        allow_cross_dataset_init=allow_cross_dataset_init,
    )


__all__ = [
    "run_nf_cse_finetune",
    "run_nf_ton_finetune",
    "run_nf_unsw_finetune",
    "run_supervised_training",
    "validate_supervised_splits",
]
