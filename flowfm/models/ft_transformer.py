"""Standalone single-task FT-Transformer baseline over canonical feature tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flowfm.data.schemaflow_batch import SchemaFlowBatch
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.models.schemaflow import SchemaFlowEncoder, SchemaFlowEncoderOutput


@dataclass(frozen=True)
class FTTransformerClassifierOutput:
    logits: torch.Tensor
    representation: torch.Tensor
    encoder_output: SchemaFlowEncoderOutput


class FTTransformerClassifier(nn.Module):
    """Unpretrained FT-Transformer with one independently trained task head."""

    def __init__(
        self,
        tokenizer_metadata: TokenizerMetadata,
        *,
        num_classes: int,
        registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
        d_token: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than one")
        self.num_classes = num_classes
        self.d_token = d_token
        self.encoder = SchemaFlowEncoder(
            tokenizer_metadata,
            registry=registry,
            d_model=d_token,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classification_head = nn.Linear(d_token, num_classes)

    def forward(
        self,
        batch: SchemaFlowBatch,
        *,
        return_layer_states: bool = False,
    ) -> FTTransformerClassifierOutput:
        encoder_output = self.encoder(
            batch,
            return_layer_states=return_layer_states,
        )
        representation = encoder_output.flow_embedding
        return FTTransformerClassifierOutput(
            logits=self.classification_head(representation),
            representation=representation,
            encoder_output=encoder_output,
        )


__all__ = [
    "FTTransformerClassifier",
    "FTTransformerClassifierOutput",
]
