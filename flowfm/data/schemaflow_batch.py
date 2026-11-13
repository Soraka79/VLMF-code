"""Canonical PyTorch batch contract for SchemaFlow-FM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from flowfm.data.contracts import FlowSample
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
)


NUMERIC_FEATURE_TYPE_ID = 0
CATEGORICAL_FEATURE_TYPE_ID = 1


@dataclass(frozen=True)
class SchemaFlowBatch:
    feature_ids: torch.Tensor
    numeric_values: torch.Tensor
    category_ids: torch.Tensor
    feature_type_ids: torch.Tensor
    missing_mask: torch.Tensor
    schema_present_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        tensor_fields = (
            "feature_ids",
            "numeric_values",
            "category_ids",
            "feature_type_ids",
            "missing_mask",
            "schema_present_mask",
        )
        tensors = {
            name: getattr(self, name) for name in tensor_fields
        }
        if not all(
            isinstance(value, torch.Tensor)
            for value in tensors.values()
        ):
            raise TypeError(
                "SchemaFlowBatch tensor fields must be torch tensors"
            )
        shapes = {tuple(value.shape) for value in tensors.values()}
        if len(shapes) != 1:
            raise ValueError(
                "SchemaFlowBatch tensors must have identical shapes"
            )
        shape = next(iter(shapes))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(
                "SchemaFlowBatch tensors must be non-empty rank-2 tensors"
            )
        expected_dtypes = {
            "feature_ids": torch.int64,
            "numeric_values": torch.float32,
            "category_ids": torch.int64,
            "feature_type_ids": torch.int64,
            "missing_mask": torch.bool,
            "schema_present_mask": torch.bool,
        }
        for name, expected in expected_dtypes.items():
            if tensors[name].dtype != expected:
                raise ValueError(
                    f"{name} must use dtype {expected}"
                )
        devices = {
            value.device for value in tensors.values()
        }
        if len(devices) != 1:
            raise ValueError(
                "SchemaFlowBatch tensors must share one device"
            )
        if not torch.isfinite(self.numeric_values).all():
            raise ValueError(
                "numeric_values must contain only finite safe fills"
            )
        if (self.feature_ids <= 0).any():
            raise ValueError(
                "feature_ids must contain positive canonical IDs"
            )
        if (self.category_ids < 0).any():
            raise ValueError(
                "category_ids must be non-negative"
            )
        if not torch.all(
            (self.feature_type_ids == NUMERIC_FEATURE_TYPE_ID)
            | (
                self.feature_type_ids
                == CATEGORICAL_FEATURE_TYPE_ID
            )
        ):
            raise ValueError(
                "feature_type_ids contain unsupported values"
            )
        numeric = (
            self.feature_type_ids == NUMERIC_FEATURE_TYPE_ID
        )
        categorical = (
            self.feature_type_ids
            == CATEGORICAL_FEATURE_TYPE_ID
        )
        if (self.category_ids[numeric] != 0).any():
            raise ValueError(
                "numeric features must use category ID 0"
            )
        if (self.numeric_values[categorical] != 0).any():
            raise ValueError(
                "categorical features must use numeric value 0"
            )
        if (
            self.category_ids[categorical & self.missing_mask]
            != 0
        ).any():
            raise ValueError(
                "missing categorical features must use category ID 0"
            )
        if (
            self.numeric_values[numeric & self.missing_mask]
            != 0
        ).any():
            raise ValueError(
                "missing numeric features must use safe fill 0"
            )
        absent = ~self.schema_present_mask
        if not self.missing_mask[absent].all():
            raise ValueError(
                "schema-absent features must also be missing"
            )
        if (self.numeric_values[absent] != 0).any():
            raise ValueError(
                "schema-absent numeric values must be zero"
            )
        if (self.category_ids[absent] != 0).any():
            raise ValueError(
                "schema-absent category IDs must be zero"
            )
        if len(self.sample_ids) != shape[0]:
            raise ValueError(
                "sample_ids must align with batch dimension"
            )
        if len(self.dataset_ids) != shape[0]:
            raise ValueError(
                "dataset_ids must align with batch dimension"
            )
        for field_name in ("sample_ids", "dataset_ids"):
            values = tuple(getattr(self, field_name))
            if not all(
                isinstance(value, str) and value
                for value in values
            ):
                raise ValueError(
                    f"{field_name} must contain non-empty strings"
                )
            object.__setattr__(self, field_name, values)

    @property
    def batch_size(self) -> int:
        return self.feature_ids.shape[0]

    @property
    def feature_count(self) -> int:
        return self.feature_ids.shape[1]

    def to(self, device: Any) -> "SchemaFlowBatch":
        return SchemaFlowBatch(
            feature_ids=self.feature_ids.to(device),
            numeric_values=self.numeric_values.to(device),
            category_ids=self.category_ids.to(device),
            feature_type_ids=self.feature_type_ids.to(device),
            missing_mask=self.missing_mask.to(device),
            schema_present_mask=(
                self.schema_present_mask.to(device)
            ),
            sample_ids=self.sample_ids,
            dataset_ids=self.dataset_ids,
        )

    def permute_features(
        self, permutation: torch.Tensor | Sequence[int]
    ) -> "SchemaFlowBatch":
        permutation = torch.as_tensor(
            permutation,
            dtype=torch.int64,
        )
        if permutation.ndim != 1 or permutation.numel() != (
            self.feature_count
        ):
            raise ValueError(
                "permutation must contain one index per feature"
            )
        expected = torch.arange(
            self.feature_count,
            dtype=torch.int64,
        )
        if not torch.equal(
            torch.sort(permutation.cpu()).values,
            expected,
        ):
            raise ValueError(
                "permutation must contain every feature index once"
            )
        index = permutation.to(self.feature_ids.device)
        return SchemaFlowBatch(
            feature_ids=self.feature_ids.index_select(1, index),
            numeric_values=self.numeric_values.index_select(
                1, index
            ),
            category_ids=self.category_ids.index_select(1, index),
            feature_type_ids=self.feature_type_ids.index_select(
                1, index
            ),
            missing_mask=self.missing_mask.index_select(1, index),
            schema_present_mask=(
                self.schema_present_mask.index_select(1, index)
            ),
            sample_ids=self.sample_ids,
            dataset_ids=self.dataset_ids,
        )


def _numeric_value(value: Any, *, missing: bool) -> float:
    if missing:
        return 0.0
    if isinstance(value, bool):
        raise ValueError(
            "observed numeric values must be finite real numbers"
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "observed numeric values must be finite real numbers"
        ) from exc
    if not math.isfinite(normalized):
        raise ValueError(
            "observed numeric values must be finite"
        )
    return normalized


def collate_schemaflow_samples(
    samples: Sequence[FlowSample],
    tokenizer_metadata: TokenizerMetadata,
    *,
    registry: CanonicalFeatureRegistry = (
        SCHEMAFLOW_MAIN_REGISTRY
    ),
) -> SchemaFlowBatch:
    """Map local sample order into a stable canonical feature batch."""

    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError(
            "samples must be a non-empty sequence"
        )
    if not isinstance(tokenizer_metadata, TokenizerMetadata):
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
    expected_vocabularies = {
        spec.name
        for spec in registry.categorical_model_features
    }
    if (
        set(tokenizer_metadata.categorical_vocabularies)
        != expected_vocabularies
    ):
        raise ValueError(
            "tokenizer vocabularies do not match registry categories"
        )
    for spec in registry.categorical_model_features:
        vocabulary = (
            tokenizer_metadata.categorical_vocabularies[
                spec.name
            ]
        )
        if (
            vocabulary.feature_id != spec.feature_id
            or vocabulary.value_policy != spec.value_policy
        ):
            raise ValueError(
                f"tokenizer vocabulary mismatch for {spec.name}"
            )

    model_specs = registry.model_features
    registered_names = set(registry.by_name)
    base_feature_ids = [
        spec.feature_id for spec in model_specs
    ]
    base_type_ids = [
        (
            CATEGORICAL_FEATURE_TYPE_ID
            if spec.feature_type == "categorical"
            else NUMERIC_FEATURE_TYPE_ID
        )
        for spec in model_specs
    ]
    rows = {
        "feature_ids": [],
        "numeric_values": [],
        "category_ids": [],
        "feature_type_ids": [],
        "missing_mask": [],
        "schema_present_mask": [],
    }
    sample_ids = []
    dataset_ids = []

    for sample in samples:
        if not isinstance(sample, FlowSample):
            raise TypeError(
                "samples must contain FlowSample values"
            )
        if len(set(sample.feature_ids)) != len(
            sample.feature_ids
        ):
            raise ValueError(
                "sample feature_ids must be unique"
            )
        unregistered = set(sample.feature_ids) - registered_names
        if unregistered:
            raise ValueError(
                "sample contains unregistered feature: "
                + ", ".join(sorted(unregistered))
            )
        positions = {
            name: index
            for index, name in enumerate(sample.feature_ids)
        }
        numeric_row = []
        category_row = []
        missing_row = []
        present_row = []

        for spec in model_specs:
            index = positions.get(spec.name)
            if index is None:
                numeric_row.append(0.0)
                category_row.append(0)
                missing_row.append(True)
                present_row.append(False)
                continue

            expected_categorical = (
                spec.feature_type == "categorical"
            )
            if (
                sample.categorical_mask[index]
                != expected_categorical
            ):
                raise ValueError(
                    f"feature type mismatch for {spec.name}"
                )
            missing = sample.missing_mask[index]
            present_row.append(True)
            missing_row.append(missing)
            if expected_categorical:
                local_id = sample.features[index]
                if type(local_id) is not int or local_id < 0:
                    raise ValueError(
                        f"invalid local category ID for {spec.name}"
                    )
                if missing and local_id != 0:
                    raise ValueError(
                        f"missing category ID for {spec.name} must be 0"
                    )
                if not missing and local_id == 0:
                    raise ValueError(
                        f"observed category ID for {spec.name} cannot be 0"
                    )
                vocabulary = (
                    tokenizer_metadata.categorical_vocabularies[
                        spec.name
                    ]
                )
                category_row.append(
                    vocabulary.encode(
                        sample.category_tokens[index]
                    )
                )
                numeric_row.append(0.0)
            else:
                numeric_row.append(
                    _numeric_value(
                        sample.features[index],
                        missing=missing,
                    )
                )
                category_row.append(0)

        rows["feature_ids"].append(base_feature_ids)
        rows["numeric_values"].append(numeric_row)
        rows["category_ids"].append(category_row)
        rows["feature_type_ids"].append(base_type_ids)
        rows["missing_mask"].append(missing_row)
        rows["schema_present_mask"].append(present_row)
        sample_ids.append(sample.sample_id)
        dataset_ids.append(sample.dataset_id)

    return SchemaFlowBatch(
        feature_ids=torch.tensor(
            rows["feature_ids"],
            dtype=torch.int64,
        ),
        numeric_values=torch.tensor(
            rows["numeric_values"],
            dtype=torch.float32,
        ),
        category_ids=torch.tensor(
            rows["category_ids"],
            dtype=torch.int64,
        ),
        feature_type_ids=torch.tensor(
            rows["feature_type_ids"],
            dtype=torch.int64,
        ),
        missing_mask=torch.tensor(
            rows["missing_mask"],
            dtype=torch.bool,
        ),
        schema_present_mask=torch.tensor(
            rows["schema_present_mask"],
            dtype=torch.bool,
        ),
        sample_ids=tuple(sample_ids),
        dataset_ids=tuple(dataset_ids),
    )


__all__ = [
    "CATEGORICAL_FEATURE_TYPE_ID",
    "NUMERIC_FEATURE_TYPE_ID",
    "SchemaFlowBatch",
    "collate_schemaflow_samples",
]
