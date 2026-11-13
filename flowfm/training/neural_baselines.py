"""Fair single-task neural baseline training primitives."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import resource
import shutil
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from flowfm.data.labels import NF_UNSW_LABEL_SPACE, DatasetLabelSpace
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.models.registry import MODEL_REGISTRY
from flowfm.training.baseline_evaluation import (
    metric_payload,
    target_arrays,
    task_labels,
)
from flowfm.training.checkpoints import load_checkpoint, save_checkpoint


LEGACY_INPUT_DIM = 3690
NEURAL_ARCHITECTURES = (
    "rescae_bigru",
    "ft_transformer",
    "schema_bigru",
)


@dataclass(frozen=True)
class NeuralBatchResult:
    loss: float
    logits: torch.Tensor


def task_class_count(
    task: str,
    label_space: DatasetLabelSpace = NF_UNSW_LABEL_SPACE,
) -> int:
    return len(task_labels(task, label_space))


def legacy_input_tensor(
    matrix: np.ndarray,
    device: Any,
) -> torch.Tensor:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise TypeError("legacy input must be a rank-2 NumPy array")
    if matrix.shape[0] <= 0 or matrix.shape[1] != LEGACY_INPUT_DIM:
        raise ValueError("legacy input must contain exactly 3,690 columns")
    if matrix.dtype != np.float32:
        raise ValueError("legacy input matrix must use float32")
    if not np.isfinite(matrix).all():
        raise ValueError("legacy input matrix must contain only finite values")
    return torch.from_numpy(matrix).to(device).reshape(
        matrix.shape[0],
        1,
        1,
        LEGACY_INPUT_DIM,
    )


def build_legacy_model(
    task: str,
    label_space: DatasetLabelSpace = NF_UNSW_LABEL_SPACE,
):
    return MODEL_REGISTRY.create(
        "rescae_bigru_legacy",
        input_channels=1,
        num_features=LEGACY_INPUT_DIM,
        num_classes=task_class_count(task, label_space),
        use_attention_block=False,
    )


def build_neural_model(
    architecture: str,
    task: str,
    *,
    tokenizer_metadata: TokenizerMetadata | None = None,
    registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
    label_space: DatasetLabelSpace = NF_UNSW_LABEL_SPACE,
    model_config: Mapping[str, Any] | None = None,
):
    class_count = task_class_count(task, label_space)
    config = {} if model_config is None else dict(model_config)
    if architecture == "rescae_bigru":
        if config:
            raise ValueError(
                "legacy model definition is fixed and accepts no model_config"
            )
        return build_legacy_model(task, label_space)
    if architecture not in NEURAL_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    if not isinstance(tokenizer_metadata, TokenizerMetadata):
        raise TypeError(
            "tokenizer_metadata is required for schema-token neural baselines"
        )
    if not isinstance(registry, CanonicalFeatureRegistry):
        raise TypeError("registry must be a CanonicalFeatureRegistry")
    registry_name = {
        "ft_transformer": "ft_transformer_baseline",
        "schema_bigru": "schema_bigru",
    }[architecture]
    return MODEL_REGISTRY.create(
        registry_name,
        tokenizer_metadata,
        num_classes=class_count,
        registry=registry,
        **config,
    )


def categorical_training_audit(
    samples,
    *,
    categorical_names=None,
    registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
) -> tuple[dict[str, Any], ...]:
    values = tuple(samples)
    if not values:
        raise ValueError("training samples must be non-empty")
    if categorical_names is None:
        categorical_names = tuple(
            spec.name
            for spec in NF_UNSW_NB15_V2_SCHEMA.model_features
            if spec.categorical
        )
    else:
        categorical_names = tuple(categorical_names)
        if (
            not categorical_names
            or len(set(categorical_names)) != len(categorical_names)
            or any(
                not isinstance(name, str)
                or name not in registry.by_name
                or registry.by_name[name].feature_type != "categorical"
                for name in categorical_names
            )
        ):
            raise ValueError("categorical_names must identify unique registry fields")
    counters = {name: Counter() for name in categorical_names}
    missing_counts = {name: 0 for name in categorical_names}
    for sample in values:
        positions = {
            name: index for index, name in enumerate(sample.feature_ids)
        }
        if len(positions) != len(sample.feature_ids):
            raise ValueError("training sample feature IDs must be unique")
        for name in categorical_names:
            if name not in positions:
                raise ValueError(f"training sample is missing {name}")
            index = positions[name]
            if not sample.categorical_mask[index]:
                raise ValueError(f"training sample type mismatch for {name}")
            if sample.missing_mask[index]:
                missing_counts[name] += 1
                continue
            token = sample.category_tokens[index]
            if not isinstance(token, str) or not token:
                raise ValueError(
                    f"observed training category token is invalid for {name}"
                )
            counters[name][token] += 1

    rows = []
    for name in categorical_names:
        registry_spec = registry.by_name[name]
        counts = counters[name]
        frequencies = tuple(counts.values())
        top_values = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        rows.append(
            {
                "feature_id": registry_spec.feature_id,
                "feature_name": name,
                "model_visible": registry_spec.model_visible,
                "policy": registry_spec.value_policy,
                "training_rows": len(values),
                "observed_rows": int(sum(frequencies)),
                "missing_count": missing_counts[name],
                "vocabulary_size": len(counts),
                "singletons": sum(value == 1 for value in frequencies),
                "frequency_le_5": sum(value <= 5 for value in frequencies),
                "top_values": [
                    {"token": token, "count": count}
                    for token, count in top_values
                ],
                "fit_scope": "train_only",
            }
        )
    return tuple(rows)


def project_model_visible_samples(
    samples,
    *,
    registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
):
    visible = {
        spec.name for spec in registry.features if spec.model_visible
    }
    projected = []
    for sample in samples:
        positions = {
            name: index for index, name in enumerate(sample.feature_ids)
        }
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


def build_neural_model_data_metadata(
    *,
    architecture: str,
    task: str,
    model_config: Mapping[str, Any] | None,
    registry: CanonicalFeatureRegistry,
    tokenizer_metadata: TokenizerMetadata,
    legacy_encoder_metadata: Mapping[str, Any],
    class_counts,
    class_weights,
    categorical_audit,
    label_space: DatasetLabelSpace = NF_UNSW_LABEL_SPACE,
) -> dict[str, Any]:
    if architecture not in NEURAL_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    if not isinstance(registry, CanonicalFeatureRegistry):
        raise TypeError("registry must be a CanonicalFeatureRegistry")
    if not isinstance(tokenizer_metadata, TokenizerMetadata):
        raise TypeError("tokenizer_metadata must be TokenizerMetadata")
    if tokenizer_metadata.registry_version != registry.version or (
        tokenizer_metadata.registry_hash != registry.registry_hash
    ):
        raise ValueError("tokenizer metadata does not match feature registry")
    labels = task_labels(task, label_space)
    config = {} if model_config is None else dict(model_config)
    encoder = dict(legacy_encoder_metadata)
    if encoder.get("fit_scope") != "train_only":
        raise ValueError("legacy encoder must be fit on training data only")
    output_dimension = encoder.get("output_dimension")
    if type(output_dimension) is not int or output_dimension <= 0:
        raise ValueError("legacy encoder output dimension must be positive")
    if architecture == "rescae_bigru" and output_dimension != LEGACY_INPUT_DIM:
        raise ValueError("legacy encoder output dimension must remain 3,690")

    counts = tuple(class_counts)
    weights = tuple(class_weights)
    if len(counts) != len(labels) or any(
        type(value) is not int or value <= 0 for value in counts
    ):
        raise ValueError("class_counts must cover the complete task label space")
    if len(weights) != len(labels) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for value in weights
    ):
        raise ValueError("class_weights must be finite positive task weights")
    audit = [dict(row) for row in categorical_audit]

    return {
        "format_version": 2,
        "feature_registry": registry.to_dict(),
        "tokenizer_metadata": tokenizer_metadata.to_dict(),
        "categorical_table_sizes": {
            name: vocabulary.table_size
            for name, vocabulary in tokenizer_metadata.categorical_vocabularies.items()
        },
        "label_spaces": {label_space.dataset_id: label_space.to_dict()},
        "training_contract": {
            "format_version": 1,
            "architecture": architecture,
            "task": task,
            "task_labels": list(labels),
            "model_config": config,
            "class_balance": "inverse_frequency_train_only",
            "class_counts_train_only": list(counts),
            "class_weights_train_only": [float(value) for value in weights],
            "legacy_encoder": encoder,
            "categorical_training_audit": audit,
            "schema_version": tokenizer_metadata.schema_version,
            "schema_hash": tokenizer_metadata.schema_hash,
            "data_manifest": dict(tokenizer_metadata.data_manifest),
            "split_manifest_sha256": tokenizer_metadata.split_manifest_sha256,
            "preprocessor_hash": tokenizer_metadata.preprocessor_hash,
            "training_sample_ids_hash": (
                tokenizer_metadata.training_sample_ids_hash
            ),
            "tokenizer_metadata_hash": tokenizer_metadata.metadata_hash,
            "registry_hash": registry.registry_hash,
        },
    }


def inverse_frequency_class_weights(
    targets: np.ndarray,
    *,
    class_count: int,
    device: Any,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    values = np.asarray(targets)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty rank-1 array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("targets must contain integer class IDs")
    if type(class_count) is not int or class_count <= 1:
        raise ValueError("class_count must be an integer greater than one")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= class_count):
        raise ValueError("targets fall outside the selected label space")
    counts = np.bincount(values, minlength=class_count)
    observed = counts > 0
    weights = np.zeros(class_count, dtype=np.float64)
    weights[observed] = values.size / (
        int(observed.sum()) * counts[observed].astype(np.float64)
    )
    return (
        torch.as_tensor(weights, dtype=torch.float32, device=device),
        tuple(int(value) for value in counts),
    )


def _model_logits(model, architecture: str, inputs) -> torch.Tensor:
    output = model(inputs)
    logits = output if architecture == "rescae_bigru" else output.logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise RuntimeError("neural baseline must return rank-2 logits")
    return logits


def train_neural_batch(
    model,
    architecture: str,
    inputs,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    optimizer,
    *,
    gradient_clip: float,
) -> NeuralBatchResult:
    if architecture not in NEURAL_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dtype != torch.int64
        or targets.ndim != 1
    ):
        raise ValueError("targets must be a rank-1 int64 tensor")
    if (
        not isinstance(class_weights, torch.Tensor)
        or class_weights.ndim != 1
        or not torch.isfinite(class_weights).all()
    ):
        raise ValueError("class_weights must be a finite rank-1 tensor")
    if not isinstance(gradient_clip, (int, float)) or isinstance(
        gradient_clip,
        bool,
    ) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = _model_logits(model, architecture, inputs)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if logits.shape[1] != class_weights.numel():
        raise ValueError("class weights must align with logits")
    loss = F.cross_entropy(logits, targets, weight=class_weights)
    if not torch.isfinite(loss).item():
        raise RuntimeError("training loss became non-finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optimizer.step()
    return NeuralBatchResult(
        loss=float(loss.detach().cpu().item()),
        logits=logits.detach(),
    )


def neural_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] <= 0
        or logits.shape[1] <= 1
    ):
        raise ValueError("logits must be a non-empty rank-2 tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return torch.softmax(logits, dim=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_training_value(name: str, value, *, integer=False) -> None:
    valid_type = type(value) is int if integer else (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not valid_type or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_neural_splits(
    splits,
    label_space: DatasetLabelSpace,
    *,
    evaluation_scope: str,
):
    if evaluation_scope not in {"validation_only", "validation_and_test"}:
        raise ValueError("invalid neural baseline evaluation_scope")
    expected = {"train", "validation"}
    if evaluation_scope == "validation_and_test":
        expected.add("test")
    if not isinstance(splits, Mapping) or set(splits) != expected:
        raise ValueError(
            "splits must match the requested neural baseline evaluation_scope"
        )
    normalized = {}
    sample_ids = set()
    feature_keys = {}
    for name in ("train", "validation", "test"):
        if name not in expected:
            continue
        values = tuple(splits[name])
        if not values:
            raise ValueError(f"{name} split must be non-empty")
        current_keys = set()
        for sample in values:
            if sample.dataset_id != label_space.dataset_id:
                raise ValueError("sample dataset does not match label space")
            if sample.sample_id in sample_ids:
                raise ValueError("sample_id crosses neural baseline splits")
            sample_ids.add(sample.sample_id)
            current_keys.add(sample.feature_key)
        normalized[name] = values
        feature_keys[name] = current_keys
        target_arrays(values, label_space)
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if left not in expected or right not in expected:
            continue
        if feature_keys[left] & feature_keys[right]:
            raise ValueError("feature_key crosses neural baseline splits")
    return normalized


def _baseline_inputs(
    architecture,
    samples,
    indices,
    *,
    legacy_matrix,
    tokenizer_metadata,
    registry,
    device,
):
    if architecture == "rescae_bigru":
        return legacy_input_tensor(legacy_matrix[indices], device)
    members = tuple(samples[index] for index in indices)
    return collate_schemaflow_samples(
        members,
        tokenizer_metadata,
        registry=registry,
    ).to(device)


def _evaluate_neural_baseline(
    model,
    architecture,
    samples,
    targets,
    *,
    labels,
    legacy_matrix,
    tokenizer_metadata,
    registry,
    device,
    batch_size,
):
    model.eval()
    probability_blocks = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            indices = list(range(start, min(start + batch_size, len(samples))))
            inputs = _baseline_inputs(
                architecture,
                samples,
                indices,
                legacy_matrix=legacy_matrix,
                tokenizer_metadata=tokenizer_metadata,
                registry=registry,
                device=device,
            )
            logits = _model_logits(model, architecture, inputs)
            probability_blocks.append(
                neural_probabilities(logits).detach().cpu().numpy()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    probabilities = np.concatenate(probability_blocks, axis=0)
    return probabilities, metric_payload(probabilities, targets, labels)


def run_neural_baseline_training(
    *,
    architecture: str,
    task: str,
    splits,
    tokenizer_metadata: TokenizerMetadata,
    registry: CanonicalFeatureRegistry,
    label_space: DatasetLabelSpace,
    legacy_encoder_metadata: Mapping[str, Any],
    legacy_matrices,
    categorical_audit,
    data_manifest_path,
    split_manifest_path,
    output_root,
    repo_dir,
    git_commit: str,
    device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    model_config: Mapping[str, Any] | None = None,
    evaluation_scope: str = "validation_and_test",
    data_root=Path("/root/autodl-tmp"),
    resume_from=None,
) -> dict[str, Any]:
    values = _validate_neural_splits(
        splits,
        label_space,
        evaluation_scope=evaluation_scope,
    )
    if architecture not in NEURAL_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    labels = task_labels(task, label_space)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_training_value("epochs", epochs, integer=True)
    _validate_training_value("batch_size", batch_size, integer=True)
    _validate_training_value("learning_rate", learning_rate)
    if (
        not isinstance(weight_decay, (int, float))
        or isinstance(weight_decay, bool)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be non-negative")
    _validate_training_value("gradient_clip", gradient_clip)
    if not isinstance(git_commit, str) or len(git_commit) != 40:
        raise ValueError("git_commit must be a full 40-character SHA")

    root = Path(data_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    manifest_path = Path(data_manifest_path).expanduser().resolve()
    split_path = Path(split_manifest_path).expanduser().resolve()
    for name, path in (
        ("output_root", output),
        ("data_manifest_path", manifest_path),
        ("split_manifest_path", split_path),
    ):
        if not path.is_relative_to(root):
            raise ValueError(f"{name} must be located below the data disk")
    if not manifest_path.is_file() or not split_path.is_file():
        raise ValueError("Manifest inputs must be existing files")
    if _sha256_file(split_path) != tokenizer_metadata.split_manifest_sha256:
        raise ValueError("Split Manifest hash does not match tokenizer metadata")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Data Manifest must be valid JSON") from exc
    if manifest_payload != dict(tokenizer_metadata.data_manifest):
        raise ValueError("Data Manifest does not match tokenizer metadata")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(torch_device)

    targets = {
        name: target_arrays(samples, label_space)[task]
        for name, samples in values.items()
    }
    class_weights, class_counts = inverse_frequency_class_weights(
        targets["train"],
        class_count=len(labels),
        device=torch_device,
    )
    matrices = {name: None for name in values}
    if architecture == "rescae_bigru":
        if not isinstance(legacy_matrices, Mapping) or set(legacy_matrices) != set(
            matrices
        ):
            raise ValueError("legacy matrices must match all neural splits")
        matrices = {name: np.asarray(legacy_matrices[name]) for name in matrices}
        for name, matrix in matrices.items():
            if matrix.shape != (len(values[name]), LEGACY_INPUT_DIM):
                raise ValueError("legacy matrices must align and remain 3,690-wide")
            if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                raise ValueError("legacy matrices must be finite float32")

    weights_cpu = tuple(float(value) for value in class_weights.detach().cpu())
    config_values = {} if model_config is None else dict(model_config)
    model_data_metadata = build_neural_model_data_metadata(
        architecture=architecture,
        task=task,
        model_config=config_values,
        registry=registry,
        tokenizer_metadata=tokenizer_metadata,
        legacy_encoder_metadata=legacy_encoder_metadata,
        class_counts=class_counts,
        class_weights=weights_cpu,
        categorical_audit=categorical_audit,
        label_space=label_space,
    )
    config = {
        "format_version": 1,
        "dataset_id": label_space.dataset_id,
        "architecture": architecture,
        "task": task,
        "task_labels": list(labels),
        "seed": seed,
        "epochs_this_run": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "evaluation_scope": evaluation_scope,
        "class_balance": "inverse_frequency_train_only",
        "model_config": config_values,
        "data_manifest_sha256": _sha256_file(manifest_path),
        "split_manifest_sha256": _sha256_file(split_path),
        "schema_version": tokenizer_metadata.schema_version,
        "schema_hash": tokenizer_metadata.schema_hash,
        "tokenizer_metadata_hash": tokenizer_metadata.metadata_hash,
        "registry_hash": registry.registry_hash,
        "legacy_encoder": dict(legacy_encoder_metadata),
        "git_commit": git_commit,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=True)
    dataset_cli_name = {
        "NF-UNSW-NB15-v2": "nf_unsw",
        "NF-ToN-IoT-v2": "nf_ton",
        "NF-CSE-CIC-IDS2018-v2": "nf_cse",
    }.get(label_space.dataset_id, label_space.dataset_id)
    run_dir = output / (
        f"baseline-{dataset_cli_name.replace('_', '-')}-{architecture}-{task}-{timestamp}-"
        f"{config_hash}-{uuid.uuid4().hex[:8]}"
    )
    run_dir.mkdir(exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    checkpoints_dir.mkdir()
    predictions_dir.mkdir()
    marker = run_dir / ".incomplete"
    marker.write_text("neural baseline run incomplete\n", encoding="utf-8")

    started = time.perf_counter()
    try:
        _write_json(run_dir / "config.json", config)
        _write_json(run_dir / "label_space.json", label_space.to_dict())
        _write_json(run_dir / "feature_registry.json", registry.to_dict())
        _write_json(
            run_dir / "tokenizer_metadata.json", tokenizer_metadata.to_dict()
        )
        _write_json(
            run_dir / "categorical_training_audit.json",
            list(categorical_audit),
        )
        shutil.copyfile(manifest_path, run_dir / "data_manifest.json")
        shutil.copyfile(split_path, run_dir / "split_manifest.json")

        model = build_neural_model(
            architecture,
            task,
            tokenizer_metadata=tokenizer_metadata,
            registry=registry,
            label_space=label_space,
            model_config=config_values,
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        start_epoch = 0
        global_step = 0
        if resume_from is not None:
            payload = load_checkpoint(
                resume_from,
                model,
                optimizer=optimizer,
                map_location=torch_device,
                expected_model_data_metadata=model_data_metadata,
            )
            start_epoch = payload["epoch"]
            global_step = payload["global_step"]

        train_log = run_dir / "train.log"
        train_started = time.perf_counter()
        final_loss = None
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch)
            order = torch.randperm(len(values["train"]), generator=generator).tolist()
            weighted_loss = 0.0
            trained = 0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                inputs = _baseline_inputs(
                    architecture,
                    values["train"],
                    indices,
                    legacy_matrix=matrices["train"],
                    tokenizer_metadata=tokenizer_metadata,
                    registry=registry,
                    device=torch_device,
                )
                target_tensor = torch.as_tensor(
                    targets["train"][indices],
                    dtype=torch.int64,
                    device=torch_device,
                )
                result = train_neural_batch(
                    model,
                    architecture,
                    inputs,
                    target_tensor,
                    class_weights,
                    optimizer,
                    gradient_clip=gradient_clip,
                )
                global_step += 1
                weighted_loss += result.loss * len(indices)
                trained += len(indices)
                with train_log.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "event": "train_batch",
                                "epoch": epoch,
                                "global_step": global_step,
                                "samples": len(indices),
                                "loss": result.loss,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            final_loss = weighted_loss / trained
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        train_seconds = time.perf_counter() - train_started

        evaluation = {}
        prediction_payloads = {}
        evaluation_seconds = {}
        evaluation_splits = ("validation",) + (
            ("test",) if evaluation_scope == "validation_and_test" else ()
        )
        for split_name in evaluation_splits:
            split_started = time.perf_counter()
            probabilities, split_metrics = _evaluate_neural_baseline(
                model,
                architecture,
                values[split_name],
                targets[split_name],
                labels=labels,
                legacy_matrix=matrices[split_name],
                tokenizer_metadata=tokenizer_metadata,
                registry=registry,
                device=torch_device,
                batch_size=batch_size,
            )
            evaluation_seconds[split_name] = time.perf_counter() - split_started
            evaluation[split_name] = split_metrics
            prediction_payloads[split_name] = probabilities
            with (predictions_dir / f"{split_name}.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for sample, target, probability in zip(
                    values[split_name], targets[split_name], probabilities
                ):
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "target_id": int(target),
                                "target_label": labels[int(target)],
                                "probabilities": {
                                    label: float(probability[index])
                                    for index, label in enumerate(labels)
                                },
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

        completed_epoch = start_epoch + epochs
        metrics = {
            "epoch": completed_epoch,
            "global_step": global_step,
            "train_loss": final_loss,
            "class_counts_train_only": {
                label: class_counts[index] for index, label in enumerate(labels)
            },
            "class_weights_train_only": {
                label: weights_cpu[index] for index, label in enumerate(labels)
            },
            **evaluation,
        }
        _write_json(run_dir / "metrics.json", metrics)

        checkpoint_path = checkpoints_dir / f"epoch-{completed_epoch:04d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=completed_epoch,
            global_step=global_step,
            git_commit=git_commit,
            schema_version=tokenizer_metadata.schema_version,
            seed=seed,
            config=config,
            model_data_metadata=model_data_metadata,
            optimizer=optimizer,
            extra_state={"metrics": metrics},
        )
        verification_model = build_neural_model(
            architecture,
            task,
            tokenizer_metadata=tokenizer_metadata,
            registry=registry,
            label_space=label_space,
            model_config=config_values,
        ).to(torch_device)
        verification_optimizer = torch.optim.AdamW(
            verification_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        verified = load_checkpoint(
            checkpoint_path,
            verification_model,
            optimizer=verification_optimizer,
            map_location=torch_device,
            restore_rng=False,
            expected_model_data_metadata=model_data_metadata,
        )
        checkpoint_verified = (
            verified["epoch"] == completed_epoch
            and verified["global_step"] == global_step
            and verified["git_commit"] == git_commit
        )
        if not checkpoint_verified:
            raise RuntimeError("saved checkpoint failed verification")
        _write_json(
            checkpoints_dir / "index.json",
            {
                "latest": checkpoint_path.name,
                "epoch": completed_epoch,
                "global_step": global_step,
                "sha256": _sha256_file(checkpoint_path),
                "verified": checkpoint_verified,
            },
        )
        (run_dir / "resume_command.txt").write_text(
            "python -m scripts.run_nf_unsw_neural_baseline "
            f"--dataset {dataset_cli_name} --model {architecture} --task {task} "
            f"--resume-from {checkpoint_path}\n",
            encoding="utf-8",
        )
        peak_gpu = (
            int(torch.cuda.max_memory_allocated(torch_device))
            if torch_device.type == "cuda"
            else 0
        )
        efficiency = {
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "wall_seconds": time.perf_counter() - started,
            "train_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
            "train_samples_per_second": len(values["train"]) / train_seconds,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "device": str(torch_device),
            "device_name": (
                torch.cuda.get_device_name(torch_device)
                if torch_device.type == "cuda"
                else "CPU"
            ),
        }
        _write_json(run_dir / "efficiency.json", efficiency)
        _write_json(
            run_dir / "environment.json",
            {
                "git_commit": git_commit,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            },
        )
        marker.unlink()
        return {
            "status": "completed",
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_verified": checkpoint_verified,
            "epochs_completed": completed_epoch,
            "global_step": global_step,
            "metrics": metrics,
            "efficiency": efficiency,
        }
    except Exception:
        raise


__all__ = [
    "LEGACY_INPUT_DIM",
    "NEURAL_ARCHITECTURES",
    "NeuralBatchResult",
    "build_legacy_model",
    "build_neural_model",
    "build_neural_model_data_metadata",
    "categorical_training_audit",
    "inverse_frequency_class_weights",
    "legacy_input_tensor",
    "neural_probabilities",
    "project_model_visible_samples",
    "run_neural_baseline_training",
    "task_class_count",
    "train_neural_batch",
]
