"""Schema-aware BiGRU baseline using the frozen SchemaFlow token boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from flowfm.data.schemaflow_batch import SchemaFlowBatch
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)
from flowfm.models.schemaflow import SchemaFlowEncoder


@dataclass(frozen=True)
class SchemaBiGRUEncoderOutput:
    representation: torch.Tensor
    sequence_embeddings: torch.Tensor
    sequence_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    input_feature_embeddings: torch.Tensor
    value_embeddings: torch.Tensor


@dataclass(frozen=True)
class SchemaBiGRUClassifierOutput:
    logits: torch.Tensor
    representation: torch.Tensor
    encoder_output: SchemaBiGRUEncoderOutput


class SchemaBiGRUEncoder(SchemaFlowEncoder):
    """Replace the Feature-set Transformer with a canonical-order BiGRU.

    The inherited embedding modules are the frozen SchemaFlow token boundary.
    Transformer modules are removed during construction and are not parameters
    of this baseline. Schema-absent fields are packed out of the GRU sequence;
    present-but-missing fields remain as missing-state tokens.
    """

    def __init__(
        self,
        tokenizer_metadata: TokenizerMetadata,
        *,
        registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
        d_model: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError("dropout must be a number in the interval [0, 1)")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be a number in the interval [0, 1)")

        # Reuse the exact SchemaFlow embedding initialization and validation.
        # The temporary backbone is removed before the BiGRU is registered.
        super().__init__(
            tokenizer_metadata,
            registry=registry,
            d_model=d_model,
            nhead=1,
            num_layers=1,
            dim_feedforward=d_model,
            dropout=0.0,
        )
        del self.flow_token
        del self.layers

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = hidden_size * 2
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )

        max_feature_id = max(spec.feature_id for spec in registry.features)
        canonical_ranks = torch.full(
            (max_feature_id + 1,),
            len(registry.model_features) + 1,
            dtype=torch.int64,
        )
        for rank, spec in enumerate(registry.model_features):
            canonical_ranks[spec.feature_id] = rank
        self.register_buffer(
            "canonical_rank_by_feature_id",
            canonical_ranks,
            persistent=False,
        )

    def _input_feature_embeddings(
        self, batch: SchemaFlowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_batch(batch)
        present = batch.schema_present_mask.unsqueeze(-1)
        value_embeddings = self._value_embeddings(batch)
        token_sum = (
            self.feature_identity_embedding(batch.feature_ids)
            + self.feature_type_embedding(batch.feature_type_ids)
            + self.missing_state_embedding(batch.missing_mask.to(torch.int64))
            + value_embeddings
        )
        return self.token_norm(token_sum) * present, value_embeddings

    def _canonical_packed_sequence(
        self,
        batch: SchemaFlowBatch,
        token_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ranks = self.canonical_rank_by_feature_id[batch.feature_ids]
        order = torch.argsort(ranks, dim=1, stable=True)
        ordered_tokens = token_embeddings.gather(
            1,
            order.unsqueeze(-1).expand(-1, -1, token_embeddings.shape[-1]),
        )
        ordered_present = batch.schema_present_mask.gather(1, order)
        lengths = ordered_present.sum(dim=1).to(torch.int64)
        if (lengths == 0).any():
            raise ValueError("each sample must contain at least one schema-present feature")

        compact_positions = ordered_present.cumsum(dim=1) - 1
        compact = torch.zeros_like(ordered_tokens)
        batch_positions = torch.arange(
            batch.batch_size,
            device=batch.feature_ids.device,
        ).unsqueeze(1).expand_as(ordered_present)
        compact[
            batch_positions[ordered_present],
            compact_positions[ordered_present],
        ] = ordered_tokens[ordered_present]
        sequence_mask = (
            torch.arange(
                batch.feature_count,
                device=batch.feature_ids.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return compact, sequence_mask, lengths

    def forward(self, batch: SchemaFlowBatch) -> SchemaBiGRUEncoderOutput:
        input_features, value_embeddings = self._input_feature_embeddings(batch)
        compact, sequence_mask, lengths = self._canonical_packed_sequence(
            batch,
            input_features,
        )
        self.gru.flatten_parameters()
        packed = pack_padded_sequence(
            compact,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        sequence_embeddings, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=batch.feature_count,
        )
        masked = sequence_embeddings * sequence_mask.unsqueeze(-1)
        representation = masked.sum(dim=1) / lengths.unsqueeze(1)
        return SchemaBiGRUEncoderOutput(
            representation=representation,
            sequence_embeddings=sequence_embeddings,
            sequence_mask=sequence_mask,
            sequence_lengths=lengths,
            input_feature_embeddings=input_features,
            value_embeddings=value_embeddings,
        )


class SchemaBiGRUClassifier(nn.Module):
    """Single-task classifier; binary, coarse, and native tasks train separately."""

    def __init__(
        self,
        tokenizer_metadata: TokenizerMetadata,
        *,
        num_classes: int,
        registry: CanonicalFeatureRegistry = SCHEMAFLOW_MAIN_REGISTRY,
        d_model: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than one")
        self.encoder = SchemaBiGRUEncoder(
            tokenizer_metadata,
            registry=registry,
            d_model=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classification_head = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, batch: SchemaFlowBatch) -> SchemaBiGRUClassifierOutput:
        encoder_output = self.encoder(batch)
        representation = encoder_output.representation
        return SchemaBiGRUClassifierOutput(
            logits=self.classification_head(representation),
            representation=representation,
            encoder_output=encoder_output,
        )


__all__ = [
    "SchemaBiGRUClassifier",
    "SchemaBiGRUClassifierOutput",
    "SchemaBiGRUEncoder",
    "SchemaBiGRUEncoderOutput",
]
