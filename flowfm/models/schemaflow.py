"""Schema-aware feature token encoder with a Feature-set Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flowfm.data.schemaflow_batch import (
    CATEGORICAL_FEATURE_TYPE_ID,
    NUMERIC_FEATURE_TYPE_ID,
    SchemaFlowBatch,
)
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)


@dataclass(frozen=True)
class SchemaFlowEncoderOutput:
    flow_embedding: torch.Tensor
    feature_embeddings: torch.Tensor
    input_feature_embeddings: torch.Tensor
    value_embeddings: torch.Tensor
    layer_feature_embeddings: tuple[torch.Tensor, ...]


class SchemaFlowEncoder(nn.Module):
    """Encode an unordered set of canonical feature tokens."""

    def __init__(
        self,
        tokenizer_metadata: TokenizerMetadata,
        *,
        registry: CanonicalFeatureRegistry = (
            SCHEMAFLOW_MAIN_REGISTRY
        ),
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(
            tokenizer_metadata, TokenizerMetadata
        ):
            raise TypeError(
                "tokenizer_metadata must be TokenizerMetadata"
            )
        if not isinstance(registry, CanonicalFeatureRegistry):
            raise TypeError(
                "registry must be CanonicalFeatureRegistry"
            )
        if (
            tokenizer_metadata.registry_version
            != registry.version
            or tokenizer_metadata.registry_hash
            != registry.registry_hash
        ):
            raise ValueError(
                "tokenizer metadata does not match feature registry"
            )
        for name, value in (
            ("d_model", d_model),
            ("nhead", nhead),
            ("num_layers", num_layers),
            ("dim_feedforward", dim_feedforward),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer"
                )
        if d_model % nhead:
            raise ValueError(
                "d_model must be divisible by nhead"
            )
        if not isinstance(dropout, (int, float)) or not (
            0 <= dropout < 1
        ):
            raise ValueError(
                "dropout must be in the interval [0, 1)"
            )

        categorical_specs = (
            registry.categorical_model_features
        )
        expected_vocabularies = {
            spec.name for spec in categorical_specs
        }
        if (
            set(tokenizer_metadata.categorical_vocabularies)
            != expected_vocabularies
        ):
            raise ValueError(
                "tokenizer vocabularies do not match registry categories"
            )
        for spec in categorical_specs:
            vocabulary = (
                tokenizer_metadata.categorical_vocabularies[
                    spec.name
                ]
            )
            if (
                vocabulary.feature_id != spec.feature_id
                or vocabulary.value_policy
                != spec.value_policy
            ):
                raise ValueError(
                    f"tokenizer vocabulary mismatch for {spec.name}"
                )

        self.tokenizer_metadata = tokenizer_metadata
        self.registry = registry
        self.d_model = d_model
        max_feature_id = max(
            spec.feature_id for spec in registry.features
        )
        self.feature_identity_embedding = nn.Embedding(
            max_feature_id + 1,
            d_model,
        )
        self.feature_type_embedding = nn.Embedding(
            2,
            d_model,
        )
        self.missing_state_embedding = nn.Embedding(
            2,
            d_model,
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.category_embeddings = nn.ModuleDict(
            {
                str(spec.feature_id): nn.Embedding(
                    tokenizer_metadata.categorical_vocabularies[
                        spec.name
                    ].table_size,
                    d_model,
                    padding_idx=0,
                )
                for spec in categorical_specs
            }
        )
        self.token_norm = nn.LayerNorm(d_model)
        self.flow_token = nn.Parameter(
            torch.empty(1, 1, d_model)
        )
        nn.init.trunc_normal_(
            self.flow_token,
            std=0.02,
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def _validate_batch(self, batch: SchemaFlowBatch) -> None:
        if not isinstance(batch, SchemaFlowBatch):
            raise TypeError(
                "batch must be a SchemaFlowBatch"
            )
        model_specs = {
            spec.feature_id: spec
            for spec in self.registry.model_features
        }
        unique_ids = set(batch.feature_ids.detach().cpu().view(-1).tolist())
        unknown_ids = unique_ids - set(model_specs)
        if unknown_ids:
            raise ValueError(
                "unregistered feature ID: "
                + ", ".join(str(value) for value in sorted(unknown_ids))
            )
        for row in batch.feature_ids:
            if torch.unique(row).numel() != row.numel():
                raise ValueError(
                    "feature IDs must be unique within each sample"
                )
        for feature_id, spec in model_specs.items():
            positions = batch.feature_ids == feature_id
            if not positions.any():
                continue
            expected_type = (
                CATEGORICAL_FEATURE_TYPE_ID
                if spec.feature_type == "categorical"
                else NUMERIC_FEATURE_TYPE_ID
            )
            if (
                batch.feature_type_ids[positions]
                != expected_type
            ).any():
                raise ValueError(
                    f"feature ID/type mismatch for {spec.name}"
                )
            if spec.feature_type == "categorical":
                category_ids = batch.category_ids[positions]
                table_size = self.category_embeddings[
                    str(feature_id)
                ].num_embeddings
                if (
                    (category_ids < 0).any()
                    or (category_ids >= table_size).any()
                ):
                    raise ValueError(
                        f"category ID out of range for {spec.name}"
                    )

    def _value_embeddings(
        self, batch: SchemaFlowBatch
    ) -> torch.Tensor:
        if not torch.isfinite(batch.numeric_values).all():
            raise ValueError(
                "numeric values must be finite before numeric projection"
            )
        numeric_projected = self.numeric_projection(
            batch.numeric_values.unsqueeze(-1)
        )
        numeric_observed = (
            (
                batch.feature_type_ids
                == NUMERIC_FEATURE_TYPE_ID
            )
            & batch.schema_present_mask
            & ~batch.missing_mask
        )
        value_embeddings = numeric_projected * (
            numeric_observed.unsqueeze(-1)
        )
        for spec in self.registry.categorical_model_features:
            positions = batch.feature_ids == spec.feature_id
            if not positions.any():
                continue
            embedded = self.category_embeddings[
                str(spec.feature_id)
            ](batch.category_ids[positions])
            value_embeddings[positions] = embedded
        return value_embeddings * (
            batch.schema_present_mask.unsqueeze(-1)
        )

    def forward(
        self,
        batch: SchemaFlowBatch,
        *,
        return_layer_states: bool = False,
    ) -> SchemaFlowEncoderOutput:
        self._validate_batch(batch)
        present = batch.schema_present_mask.unsqueeze(-1)
        value_embeddings = self._value_embeddings(batch)
        token_sum = (
            self.feature_identity_embedding(batch.feature_ids)
            + self.feature_type_embedding(
                batch.feature_type_ids
            )
            + self.missing_state_embedding(
                batch.missing_mask.to(torch.int64)
            )
            + value_embeddings
        )
        input_features = self.token_norm(token_sum) * present

        flow = self.flow_token.expand(
            batch.batch_size,
            -1,
            -1,
        )
        hidden = torch.cat((flow, input_features), dim=1)
        key_padding_mask = torch.cat(
            (
                torch.zeros(
                    batch.batch_size,
                    1,
                    dtype=torch.bool,
                    device=batch.feature_ids.device,
                ),
                ~batch.schema_present_mask,
            ),
            dim=1,
        )
        layer_states = []
        for layer in self.layers:
            hidden = layer(
                hidden,
                src_key_padding_mask=key_padding_mask,
            )
            feature_hidden = hidden[:, 1:] * present
            hidden = torch.cat(
                (hidden[:, :1], feature_hidden),
                dim=1,
            )
            if return_layer_states:
                layer_states.append(feature_hidden)

        feature_embeddings = hidden[:, 1:] * present
        return SchemaFlowEncoderOutput(
            flow_embedding=hidden[:, 0],
            feature_embeddings=feature_embeddings,
            input_feature_embeddings=input_features,
            value_embeddings=value_embeddings,
            layer_feature_embeddings=tuple(layer_states),
        )


__all__ = [
    "SchemaFlowEncoder",
    "SchemaFlowEncoderOutput",
]
