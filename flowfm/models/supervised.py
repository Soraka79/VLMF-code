"""Supervised multi-task heads and lightweight adapters for SchemaFlow-FM."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from flowfm.data.labels import DatasetLabelSpace, SupervisedTargets
from flowfm.data.schemaflow_batch import SchemaFlowBatch
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.models.schemaflow import SchemaFlowEncoder, SchemaFlowEncoderOutput


class ResidualAdapter(nn.Module):
    def __init__(self, d_model: int, adapter_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if type(adapter_dim) is not int or adapter_dim <= 0:
            raise ValueError("adapter_dim must be a positive integer")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(adapter_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        adapted = self.up(
            self.dropout(self.activation(self.down(self.norm(values))))
        )
        return values + adapted


@dataclass(frozen=True)
class SchemaFlowSupervisedOutput:
    dataset_id: str
    representation: torch.Tensor
    binary_logits: torch.Tensor
    coarse_logits: torch.Tensor
    native_logits: torch.Tensor
    encoder_output: SchemaFlowEncoderOutput


@dataclass(frozen=True)
class SchemaFlowSupervisedLoss:
    total: torch.Tensor
    binary: torch.Tensor
    coarse: torch.Tensor
    native: torch.Tensor


def _module_key(dataset_id: str) -> str:
    digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16]
    return f"dataset_{digest}"


class SchemaFlowSupervisedModel(nn.Module):
    def __init__(
        self,
        tokenizer_metadata: TokenizerMetadata,
        label_spaces: Mapping[str, DatasetLabelSpace],
        *,
        registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        adapter_dim: int = 32,
    ) -> None:
        super().__init__()
        if not isinstance(label_spaces, Mapping) or not label_spaces:
            raise ValueError("label_spaces must be a non-empty mapping")
        normalized = {}
        for dataset_id, label_space in sorted(label_spaces.items()):
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError("label space keys must be non-empty strings")
            if not isinstance(label_space, DatasetLabelSpace):
                raise TypeError("label_spaces must contain DatasetLabelSpace values")
            if dataset_id != label_space.dataset_id:
                raise ValueError("label space key does not match dataset_id")
            normalized[dataset_id] = label_space
        first = next(iter(normalized.values()))
        for label_space in normalized.values():
            if label_space.binary_labels != first.binary_labels:
                raise ValueError("all datasets must share binary_labels")
            if label_space.coarse_labels != first.coarse_labels:
                raise ValueError("all datasets must share coarse_labels")

        self.encoder = SchemaFlowEncoder(
            tokenizer_metadata,
            registry=registry,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.adapter = ResidualAdapter(d_model, adapter_dim, dropout)
        self.binary_head = nn.Linear(d_model, len(first.binary_labels))
        self.coarse_head = nn.Linear(d_model, len(first.coarse_labels))
        self._dataset_module_keys = {
            dataset_id: _module_key(dataset_id) for dataset_id in normalized
        }
        if len(set(self._dataset_module_keys.values())) != len(normalized):
            raise ValueError("dataset IDs produced colliding module keys")
        self.native_heads = nn.ModuleDict(
            {
                self._dataset_module_keys[dataset_id]: nn.Linear(
                    d_model, len(label_space.native_labels)
                )
                for dataset_id, label_space in normalized.items()
            }
        )
        self.label_spaces = normalized

    def forward(self, batch: SchemaFlowBatch) -> SchemaFlowSupervisedOutput:
        if not isinstance(batch, SchemaFlowBatch):
            raise TypeError("batch must be SchemaFlowBatch")
        dataset_ids = set(batch.dataset_ids)
        if len(dataset_ids) != 1:
            raise ValueError("supervised batches must contain exactly one dataset")
        dataset_id = next(iter(dataset_ids))
        if dataset_id not in self.label_spaces:
            raise ValueError(f"no native label head is configured for {dataset_id}")
        encoder_output = self.encoder(batch)
        representation = self.adapter(encoder_output.flow_embedding)
        native_head = self.native_heads[self._dataset_module_keys[dataset_id]]
        return SchemaFlowSupervisedOutput(
            dataset_id=dataset_id,
            representation=representation,
            binary_logits=self.binary_head(representation),
            coarse_logits=self.coarse_head(representation),
            native_logits=native_head(representation),
            encoder_output=encoder_output,
        )


def compute_supervised_loss(
    output: SchemaFlowSupervisedOutput,
    targets: SupervisedTargets,
    *,
    binary_weight: float = 1.0,
    coarse_weight: float = 1.0,
    native_weight: float = 1.0,
) -> SchemaFlowSupervisedLoss:
    if not isinstance(output, SchemaFlowSupervisedOutput):
        raise TypeError("output must be SchemaFlowSupervisedOutput")
    if not isinstance(targets, SupervisedTargets):
        raise TypeError("targets must be SupervisedTargets")
    if output.dataset_id != targets.dataset_id:
        raise ValueError("output and targets use different datasets")
    weights = (binary_weight, coarse_weight, native_weight)
    if not all(isinstance(value, (int, float)) and value >= 0 for value in weights):
        raise ValueError("loss weights must be non-negative numbers")
    if not any(value > 0 for value in weights):
        raise ValueError("at least one loss weight must be positive")
    expected_rows = targets.binary.shape[0]
    logits = (output.binary_logits, output.coarse_logits, output.native_logits)
    if not all(value.ndim == 2 and value.shape[0] == expected_rows for value in logits):
        raise ValueError("logits and targets must have aligned batch dimensions")

    binary_loss = F.cross_entropy(output.binary_logits, targets.binary)
    coarse_loss = F.cross_entropy(output.coarse_logits, targets.coarse)
    native_loss = F.cross_entropy(output.native_logits, targets.native)
    total = (
        float(binary_weight) * binary_loss
        + float(coarse_weight) * coarse_loss
        + float(native_weight) * native_loss
    )
    return SchemaFlowSupervisedLoss(
        total=total,
        binary=binary_loss,
        coarse=coarse_loss,
        native=native_loss,
    )


__all__ = [
    "ResidualAdapter",
    "SchemaFlowSupervisedLoss",
    "SchemaFlowSupervisedModel",
    "SchemaFlowSupervisedOutput",
    "compute_supervised_loss",
]
