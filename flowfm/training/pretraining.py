"""Self-supervised masked reconstruction and contrastive objectives."""

from __future__ import annotations

import json
import resource
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from flowfm.data.schemaflow_batch import (
    CATEGORICAL_FEATURE_TYPE_ID,
    NUMERIC_FEATURE_TYPE_ID,
    SchemaFlowBatch,
    collate_schemaflow_samples,
)
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.config import FlowConfig
from flowfm.data.labels import DatasetLabelSpace
from flowfm.data.identity import stable_hash
from flowfm.experiment.logging import JsonlLogger
from flowfm.experiment.metadata import write_json_snapshot
from flowfm.experiment.runs import create_run_context
from flowfm.models.schemaflow import SchemaFlowEncoder, SchemaFlowEncoderOutput
from flowfm.training.checkpoints import load_checkpoint, save_checkpoint
from flowfm.training.supervised import (
    _batch_order,
    _model_data_metadata,
    _resolve_device,
    _sha256_file,
    _write_text,
)


def _validate_fraction(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} fraction must be a finite number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} fraction must be between 0 and 1")
    return value


@dataclass(frozen=True)
class PretrainingView:
    batch: SchemaFlowBatch
    reconstruction_mask: torch.Tensor
    numeric_targets: torch.Tensor
    category_targets: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.batch, SchemaFlowBatch):
            raise TypeError("batch must be SchemaFlowBatch")
        tensors = (
            self.reconstruction_mask,
            self.numeric_targets,
            self.category_targets,
        )
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("pretraining targets must be tensors")
        if any(tuple(value.shape) != tuple(self.batch.feature_ids.shape) for value in tensors):
            raise ValueError("pretraining tensors must align with batch features")
        if self.reconstruction_mask.dtype is not torch.bool:
            raise ValueError("reconstruction_mask must be boolean")
        if self.numeric_targets.dtype != torch.float32:
            raise ValueError("numeric_targets must be float32")
        if self.category_targets.dtype != torch.int64:
            raise ValueError("category_targets must be int64")
        if not torch.isfinite(self.numeric_targets).all():
            raise ValueError("numeric_targets must be finite")

    def to(self, device) -> "PretrainingView":
        return PretrainingView(
            batch=self.batch.to(device),
            reconstruction_mask=self.reconstruction_mask.to(device),
            numeric_targets=self.numeric_targets.to(device),
            category_targets=self.category_targets.to(device),
        )


def make_pretraining_view(
    batch: SchemaFlowBatch,
    *,
    mask_fraction: float = 0.3,
    schema_corruption_fraction: float = 0.1,
    generator: torch.Generator | None = None,
) -> PretrainingView:
    if not isinstance(batch, SchemaFlowBatch):
        raise TypeError("batch must be SchemaFlowBatch")
    mask_fraction = _validate_fraction(mask_fraction, "mask")
    schema_corruption_fraction = _validate_fraction(
        schema_corruption_fraction, "schema corruption"
    )
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.seed()
    if generator.device.type != "cpu":
        raise ValueError("pretraining generator must be a CPU generator")

    numeric_targets = batch.numeric_values.detach().clone()
    category_targets = batch.category_ids.detach().clone()
    eligible = (
        batch.schema_present_mask
        & ~batch.missing_mask
    )
    random_values = torch.rand(
        batch.feature_ids.shape,
        generator=generator,
        device="cpu",
    ).to(batch.feature_ids.device)
    reconstruction_mask = eligible & (random_values < mask_fraction)

    schema_values = torch.rand(
        batch.feature_ids.shape,
        generator=generator,
        device="cpu",
    ).to(batch.feature_ids.device)
    schema_mask = (
        eligible
        & ~reconstruction_mask
        & (schema_values < schema_corruption_fraction)
    )
    reconstruction_mask = reconstruction_mask | schema_mask

    numeric_values = batch.numeric_values.clone()
    category_ids = batch.category_ids.clone()
    feature_missing = batch.missing_mask.clone()
    schema_present = batch.schema_present_mask.clone()
    numeric_mask = reconstruction_mask & (
        batch.feature_type_ids == NUMERIC_FEATURE_TYPE_ID
    )
    categorical_mask = reconstruction_mask & (
        batch.feature_type_ids == CATEGORICAL_FEATURE_TYPE_ID
    )
    numeric_values[numeric_mask] = 0.0
    category_ids[categorical_mask] = 0
    numeric_values[schema_mask] = 0.0
    category_ids[schema_mask] = 0
    feature_missing[reconstruction_mask] = True
    schema_present[schema_mask] = False

    masked_batch = SchemaFlowBatch(
        feature_ids=batch.feature_ids,
        numeric_values=numeric_values,
        category_ids=category_ids,
        feature_type_ids=batch.feature_type_ids,
        missing_mask=feature_missing,
        schema_present_mask=schema_present,
        sample_ids=batch.sample_ids,
        dataset_ids=batch.dataset_ids,
    )
    return PretrainingView(
        batch=masked_batch,
        reconstruction_mask=reconstruction_mask,
        numeric_targets=numeric_targets,
        category_targets=category_targets,
    )


@dataclass(frozen=True)
class SchemaFlowPretrainerOutput:
    encoder_output: SchemaFlowEncoderOutput
    projection: torch.Tensor
    numeric_predictions: torch.Tensor
    category_logits: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class PretrainingLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    contrastive: torch.Tensor
    numeric_reconstruction: torch.Tensor
    categorical_reconstruction: torch.Tensor


class SchemaFlowPretrainer(nn.Module):
    def __init__(
        self,
        tokenizer_metadata: TokenizerMetadata,
        *,
        registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if type(projection_dim) is not int or projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer")
        self.encoder = SchemaFlowEncoder(
            tokenizer_metadata,
            registry=registry,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.registry = registry
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
        )
        numeric_specs = {
            str(spec.feature_id): spec
            for spec in registry.model_features
            if spec.feature_type == "numeric"
        }
        categorical_specs = {
            str(spec.feature_id): spec
            for spec in registry.categorical_model_features
        }
        self.numeric_heads = nn.ModuleDict(
            {key: nn.Linear(d_model, 1) for key in numeric_specs}
        )
        self.category_heads = nn.ModuleDict(
            {
                key: nn.Linear(
                    d_model,
                    tokenizer_metadata.categorical_vocabularies[
                        spec.name
                    ].table_size,
                )
                for key, spec in categorical_specs.items()
            }
        )

    def forward(self, batch: SchemaFlowBatch) -> SchemaFlowPretrainerOutput:
        encoder_output = self.encoder(batch)
        feature_ids = batch.feature_ids[0]
        numeric_predictions = torch.zeros_like(
            batch.numeric_values,
            dtype=encoder_output.feature_embeddings.dtype,
        )
        category_logits = {}
        for column, feature_id in enumerate(feature_ids.tolist()):
            key = str(feature_id)
            embedding = encoder_output.feature_embeddings[:, column]
            if key in self.numeric_heads:
                numeric_predictions[:, column] = self.numeric_heads[key](
                    embedding
                ).squeeze(-1)
            if key in self.category_heads:
                category_logits[key] = self.category_heads[key](embedding)
        projection = F.normalize(
            self.projection_head(encoder_output.flow_embedding), dim=-1
        )
        return SchemaFlowPretrainerOutput(
            encoder_output=encoder_output,
            projection=projection,
            numeric_predictions=numeric_predictions,
            category_logits=category_logits,
        )


def _contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("contrastive projections must have identical rank-2 shapes")
    if first.shape[0] < 2:
        raise ValueError("contrastive loss requires at least two samples")
    logits = first @ second.transpose(0, 1) / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def compute_pretraining_loss(
    first_view: PretrainingView,
    first_output: SchemaFlowPretrainerOutput,
    second_view: PretrainingView,
    second_output: SchemaFlowPretrainerOutput,
    *,
    reconstruction_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    temperature: float = 0.1,
) -> PretrainingLoss:
    if not isinstance(first_view, PretrainingView) or not isinstance(
        second_view, PretrainingView
    ):
        raise TypeError("views must be PretrainingView values")
    if not isinstance(first_output, SchemaFlowPretrainerOutput) or not isinstance(
        second_output, SchemaFlowPretrainerOutput
    ):
        raise TypeError("outputs must be SchemaFlowPretrainerOutput values")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    if reconstruction_weight < 0 or contrastive_weight < 0:
        raise ValueError("pretraining loss weights must be non-negative")
    device = first_view.batch.feature_ids.device
    numeric_mask = first_view.reconstruction_mask & (
        first_view.batch.feature_type_ids == NUMERIC_FEATURE_TYPE_ID
    )
    categorical_mask = first_view.reconstruction_mask & (
        first_view.batch.feature_type_ids == CATEGORICAL_FEATURE_TYPE_ID
    )
    if numeric_mask.any():
        numeric_loss = F.mse_loss(
            first_output.numeric_predictions[numeric_mask],
            first_view.numeric_targets[numeric_mask],
        )
    else:
        numeric_loss = torch.zeros((), device=device)

    categorical_terms = []
    for column, feature_id in enumerate(first_view.batch.feature_ids[0].tolist()):
        key = str(feature_id)
        if key not in first_output.category_logits:
            continue
        selected = categorical_mask[:, column]
        if selected.any():
            categorical_terms.append(
                F.cross_entropy(
                    first_output.category_logits[key][selected],
                    first_view.category_targets[selected, column],
                )
            )
    categorical_loss = (
        torch.stack(categorical_terms).mean()
        if categorical_terms
        else torch.zeros((), device=device)
    )
    reconstruction = numeric_loss + categorical_loss
    contrastive = _contrastive_loss(
        first_output.projection,
        second_output.projection,
        float(temperature),
    )
    total = reconstruction_weight * reconstruction + contrastive_weight * contrastive
    return PretrainingLoss(
        total=total,
        reconstruction=reconstruction,
        contrastive=contrastive,
        numeric_reconstruction=numeric_loss,
        categorical_reconstruction=categorical_loss,
    )


def _normalize_label_spaces(label_space):
    if isinstance(label_space, DatasetLabelSpace):
        spaces = {label_space.dataset_id: label_space}
    elif isinstance(label_space, Mapping):
        spaces = dict(label_space)
    else:
        raise TypeError("label_space must be DatasetLabelSpace or a mapping")
    if not spaces:
        raise ValueError("pretraining label spaces must be non-empty")
    for dataset_id, value in spaces.items():
        if not isinstance(value, DatasetLabelSpace):
            raise TypeError("pretraining label spaces must contain DatasetLabelSpace")
        if dataset_id != value.dataset_id:
            raise ValueError("pretraining label space key does not match dataset_id")
    return spaces


def _validate_pretraining_samples(splits, label_spaces):
    if not isinstance(splits, Mapping) or "train" not in splits:
        raise ValueError("pretraining splits must contain train")
    samples = tuple(splits["train"])
    if not samples:
        raise ValueError("pretraining train split must be non-empty")
    identities = set()
    for sample in samples:
        if sample.dataset_id not in label_spaces:
            raise ValueError("pretraining sample dataset does not match label space")
        identity = (sample.dataset_id, sample.sample_id)
        if identity in identities:
            raise ValueError("dataset/sample identity is duplicated in train split")
        identities.add(identity)
    return samples


def _pretraining_model_data_metadata(registry, tokenizer_metadata, label_spaces):
    if len(label_spaces) == 1:
        return _model_data_metadata(
            registry, tokenizer_metadata, next(iter(label_spaces.values()))
        )
    return {
        "format_version": 2,
        "feature_registry": registry.to_dict(),
        "tokenizer_metadata": tokenizer_metadata.to_dict(),
        "categorical_table_sizes": {
            name: vocabulary.table_size
            for name, vocabulary in tokenizer_metadata.categorical_vocabularies.items()
        },
        "label_spaces": {
            dataset_id: space.to_dict()
            for dataset_id, space in sorted(label_spaces.items())
        },
    }


def run_pretraining(
    config: FlowConfig,
    *,
    splits,
    tokenizer_metadata: TokenizerMetadata,
    label_space,
    registry: CanonicalFeatureRegistry,
    data_manifest: Mapping,
    split_manifest: Mapping,
    repo_dir,
    data_root=Path("/root/autodl-tmp"),
    git_commit: str,
    config_path=None,
    resume_from=None,
):
    if not isinstance(config, FlowConfig) or config.workflow != "pretrain":
        raise TypeError("config must be a validated pretrain FlowConfig")
    if config.runtime.get("execution", "validate_only") != "train":
        raise ValueError("pretraining requires runtime execution train")
    if config.schema["version"] != tokenizer_metadata.schema_version:
        raise ValueError("config Schema does not match tokenizer metadata")
    label_spaces = _normalize_label_spaces(label_space)
    if (
        len(label_spaces) == 1
        and next(iter(label_spaces.values())).schema_version
        != tokenizer_metadata.schema_version
    ):
        raise ValueError("label space Schema does not match tokenizer metadata")
    samples = _validate_pretraining_samples(splits, label_spaces)
    if not isinstance(data_manifest, Mapping) or not data_manifest:
        raise ValueError("data_manifest must be a non-empty mapping")
    if not isinstance(split_manifest, Mapping) or not split_manifest:
        raise ValueError("split_manifest must be a non-empty mapping")

    device = _resolve_device(config.runtime["device"])
    seed = config.experiment["seed"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    dataset_hash = stable_hash(
        dict(data_manifest), "flowfm-training-data-manifest-v1"
    )
    context = create_run_context(
        config,
        repo_dir,
        dataset_hash=dataset_hash,
        split_hash=tokenizer_metadata.split_manifest_sha256,
        data_root=data_root,
    )
    marker = context.run_dir / ".incomplete"
    marker.write_text("pretraining incomplete\n", encoding="utf-8")
    logger = JsonlLogger(context.run_dir / "train.log")
    model_data_metadata = _pretraining_model_data_metadata(
        registry, tokenizer_metadata, label_spaces
    )
    try:
        import yaml

        _write_text(
            context.run_dir / "config.yaml",
            yaml.safe_dump(config.to_dict(), sort_keys=True, allow_unicode=True),
        )
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
        values = config.model
        model = SchemaFlowPretrainer(
            tokenizer_metadata,
            registry=registry,
            d_model=values["d_model"],
            nhead=values["nhead"],
            num_layers=values["num_layers"],
            dim_feedforward=values["dim_feedforward"],
            dropout=values["dropout"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training["learning_rate"],
            weight_decay=config.training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.training["scheduler_gamma"]
        )
        start_epoch = 0
        global_step = 0
        if resume_from is not None:
            payload = load_checkpoint(
                resume_from,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                expected_model_data_metadata=model_data_metadata,
            )
            start_epoch = payload["epoch"]
            global_step = payload["global_step"]
        end_epoch = start_epoch + config.training["epochs"]
        generator = torch.Generator(device="cpu")
        objective = {
            "mask_fraction": config.training.get("mask_fraction", 0.3),
            "schema_corruption_fraction": config.training.get(
                "schema_corruption_fraction", 0.1
            ),
            "reconstruction_weight": config.training.get("reconstruction_weight", 1.0),
            "contrastive_weight": config.training.get("contrastive_weight", 1.0),
            "contrastive_temperature": config.training.get("contrastive_temperature", 0.1),
        }
        total_started = time.perf_counter()
        loss_accumulator = {"total": 0.0, "reconstruction": 0.0, "contrastive": 0.0}
        total_samples = 0
        for epoch in range(start_epoch + 1, end_epoch + 1):
            model.train()
            order = _batch_order(
                len(samples),
                shuffle=config.training["shuffle"],
                seed=seed,
                epoch=epoch,
            )
            for offset in range(0, len(order), config.training["batch_size"]):
                members = tuple(
                    samples[index]
                    for index in order[offset : offset + config.training["batch_size"]]
                )
                batch = collate_schemaflow_samples(
                    members, tokenizer_metadata, registry=registry
                )
                generator.manual_seed(seed + epoch * 100000 + global_step)
                first_view = make_pretraining_view(
                    batch,
                    mask_fraction=objective["mask_fraction"],
                    schema_corruption_fraction=objective["schema_corruption_fraction"],
                    generator=generator,
                ).to(device)
                second_view = make_pretraining_view(
                    batch,
                    mask_fraction=objective["mask_fraction"],
                    schema_corruption_fraction=objective["schema_corruption_fraction"],
                    generator=generator,
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                first_output = model(first_view.batch)
                second_output = model(second_view.batch)
                losses = compute_pretraining_loss(
                    first_view,
                    first_output,
                    second_view,
                    second_output,
                    reconstruction_weight=objective["reconstruction_weight"],
                    contrastive_weight=objective["contrastive_weight"],
                    temperature=objective["contrastive_temperature"],
                )
                if not torch.isfinite(losses.total).item():
                    raise RuntimeError("pretraining loss became non-finite")
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training["gradient_clip"]
                )
                optimizer.step()
                global_step += 1
                total_samples += len(members)
                for name, value in (
                    ("total", losses.total),
                    ("reconstruction", losses.reconstruction),
                    ("contrastive", losses.contrastive),
                ):
                    loss_accumulator[name] += float(value.detach().cpu().item()) * len(
                        members
                    )
                logger.log(
                    {
                        "event": "pretrain_batch",
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_samples": len(members),
                        "loss": float(losses.total.detach().cpu().item()),
                        "reconstruction": float(
                            losses.reconstruction.detach().cpu().item()
                        ),
                        "contrastive": float(losses.contrastive.detach().cpu().item()),
                    }
                )
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - total_started
        pretraining_metrics = {
            name: value / total_samples
            for name, value in loss_accumulator.items()
        }
        metrics = {
            "epoch": end_epoch,
            "global_step": global_step,
            "pretraining": pretraining_metrics,
            "objective": objective,
            "train_samples": len(samples),
            "train_samples_by_dataset": {
                dataset_id: sum(sample.dataset_id == dataset_id for sample in samples)
                for dataset_id in sorted(label_spaces)
            },
            "uses_validation_or_test": False,
        }
        write_json_snapshot(context.run_dir / "metrics.json", metrics)
        write_json_snapshot(
            context.predictions / "pretraining_summary.json",
            {
                "sample_count": len(samples),
                "metrics": pretraining_metrics,
                "objective": objective,
            },
        )
        checkpoint_path = context.checkpoints / f"epoch-{end_epoch:04d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=end_epoch,
            global_step=global_step,
            git_commit=git_commit,
            schema_version=tokenizer_metadata.schema_version,
            seed=seed,
            config=config.to_dict(),
            model_data_metadata=model_data_metadata,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state={"metrics": metrics},
        )
        verification_model = SchemaFlowPretrainer(
            tokenizer_metadata,
            registry=registry,
            d_model=values["d_model"],
            nhead=values["nhead"],
            num_layers=values["num_layers"],
            dim_feedforward=values["dim_feedforward"],
            dropout=values["dropout"],
        ).to(device)
        verification_optimizer = torch.optim.AdamW(
            verification_model.parameters(),
            lr=config.training["learning_rate"],
            weight_decay=config.training["weight_decay"],
        )
        verification_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            verification_optimizer, gamma=config.training["scheduler_gamma"]
        )
        payload = load_checkpoint(
            checkpoint_path,
            verification_model,
            optimizer=verification_optimizer,
            scheduler=verification_scheduler,
            map_location=device,
            restore_rng=False,
            expected_model_data_metadata=model_data_metadata,
        )
        checkpoint_verified = (
            payload["epoch"] == end_epoch
            and payload["global_step"] == global_step
            and payload["git_commit"] == git_commit
        )
        if not checkpoint_verified:
            raise RuntimeError("saved pretraining checkpoint failed verification")
        peak_gpu = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        efficiency = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "train_seconds": elapsed,
            "train_samples_per_second": len(samples) / elapsed,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_system_memory_bytes": peak_rss,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        }
        write_json_snapshot(context.run_dir / "efficiency.json", efficiency)
        write_json_snapshot(
            context.checkpoints / "index.json",
            {
                "latest": checkpoint_path.name,
                "epoch": end_epoch,
                "global_step": global_step,
                "sha256": _sha256_file(checkpoint_path),
                "verified": checkpoint_verified,
            },
        )
        config_label = (
            str(Path(config_path).expanduser().resolve())
            if config_path is not None
            else "<CONFIG_PATH>"
        )
        _write_text(
            context.run_dir / "resume_command.txt",
            "python -m flowfm pretrain --config {} --resume-from {}".format(
                config_label, checkpoint_path
            ),
        )
        logger.log(
            {
                "event": "pretraining_completed",
                "epoch": end_epoch,
                "global_step": global_step,
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
            "epochs_completed": end_epoch,
            "global_step": global_step,
            "metrics": metrics,
            "efficiency": efficiency,
        }
    except Exception:
        raise


def run_nf_unsw_pretrain(
    config: FlowConfig,
    *,
    repo_dir,
    config_path,
    resume_from=None,
    data_root=Path("/root/autodl-tmp"),
):
    from flowfm.data.nf_unsw_dataset import load_nf_unsw_development_samples
    from flowfm.data.nf_unsw_preprocessing import load_nf_unsw_preprocessor
    from flowfm.data.nf_unsw_tokenization import build_nf_unsw_tokenizer_metadata
    from flowfm.data.labels import NF_UNSW_LABEL_SPACE
    from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
    from flowfm.experiment.metadata import validate_output_path
    from flowfm.training.supervised import _clean_git_commit

    if config.data["dataset_id"] != NF_UNSW_LABEL_SPACE.dataset_id:
        raise ValueError("NF-UNSW pretrain requires the NF-UNSW dataset_id")
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
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(
        (split_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    dataset = load_nf_unsw_development_samples(
        split_dir, preprocessor_dir, data_root=data_root
    )
    preprocessor = load_nf_unsw_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer_metadata = build_nf_unsw_tokenizer_metadata(
        dataset, preprocessor, data_manifest=data_manifest
    )
    git_commit = _clean_git_commit(Path(repo_dir).expanduser().resolve())
    return run_pretraining(
        config,
        splits={"train": dataset.splits["train"]},
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
    )


__all__ = [
    "PretrainingLoss",
    "PretrainingView",
    "SchemaFlowPretrainer",
    "SchemaFlowPretrainerOutput",
    "compute_pretraining_loss",
    "make_pretraining_view",
    "run_nf_unsw_pretrain",
    "run_pretraining",
]
