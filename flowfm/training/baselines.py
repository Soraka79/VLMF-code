"""Leakage-safe tabular matrices for classical baseline models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from flowfm.data.contracts import FlowSample


@dataclass(frozen=True)
class TabularFeatureEncoder:
    """Encode FlowSamples without using identity or post-train information.

    Numeric values are copied from the train-only preprocessor output, categorical
    IDs are one-hot encoded using only IDs observed in the fit samples, and each
    feature receives an explicit missing indicator. Unknown validation/test IDs
    map to the reserved category ID 1.
    """

    feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    categorical_values: Mapping[str, tuple[int, ...]]
    unknown_category_id: int = 1
    category_encoders: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def fit(
        cls,
        samples: Sequence[FlowSample],
        *,
        category_encoders: Mapping[str, object] | None = None,
    ) -> "TabularFeatureEncoder":
        values = tuple(samples)
        if not values:
            raise ValueError("fit samples must be non-empty")
        first = values[0]
        if not isinstance(first, FlowSample):
            raise TypeError("fit samples must contain FlowSample values")
        feature_names = tuple(sorted(first.feature_ids))
        if len(feature_names) != len(first.feature_ids):
            raise ValueError("feature_ids must be unique")
        first_by_name = _sample_index(first, feature_names)
        categorical_names = tuple(
            name for name in feature_names if first_by_name[name][1]
        )
        encoders = {} if category_encoders is None else dict(category_encoders)
        if set(encoders) != set(categorical_names) and encoders:
            raise ValueError("category_encoders must match categorical features")
        category_values = {}
        for name in categorical_names:
            encoder = encoders.get(name)
            if encoder is None:
                category_values[name] = {0, 1}
                continue
            table_size = getattr(encoder, "table_size", None)
            if type(table_size) is not int or table_size < 2:
                raise ValueError(f"invalid category table for {name}")
            if not callable(getattr(encoder, "encode", None)):
                raise TypeError(f"category encoder for {name} lacks encode()")
            category_values[name] = set(range(table_size))
        for sample in values:
            _validate_sample_schema(sample, feature_names, first_by_name)
            for name in categorical_names:
                index, _ = _sample_index(sample, feature_names)[name]
                if name in encoders:
                    encoded = encoders[name].encode(sample.category_tokens[index])
                    if type(encoded) is not int or encoded < 0:
                        raise ValueError(f"invalid encoded category for {name}")
                    category_values[name].add(encoded)
                    continue
                raw = sample.features[index]
                if type(raw) is bool or not isinstance(raw, (int, np.integer)):
                    raise ValueError(
                        f"categorical feature {name} must contain integer IDs"
                    )
                category_values[name].add(int(raw))
        normalized = {
            name: tuple(sorted(ids))
            for name, ids in sorted(category_values.items())
        }
        return cls(
            feature_names=feature_names,
            categorical_feature_names=categorical_names,
            categorical_values=MappingProxyType(normalized),
            category_encoders=MappingProxyType(encoders),
        )

    @property
    def numeric_feature_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.feature_names
            if name not in self.categorical_feature_names
        )

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        names = []
        for name in self.feature_names:
            if name in self.categorical_feature_names:
                names.extend(
                    f"{name}=={category}"
                    for category in self.categorical_values[name]
                )
            else:
                names.append(name)
            names.append(f"{name}__missing")
        return tuple(names)

    def transform(self, samples: Sequence[FlowSample]) -> np.ndarray:
        values = tuple(samples)
        if not values:
            return np.empty((0, len(self.output_feature_names)), dtype=np.float32)
        rows = []
        categorical_names = set(self.categorical_feature_names)
        expected_types = {
            name: (0, name in categorical_names)
            for name in self.feature_names
        }
        for sample in values:
            if not isinstance(sample, FlowSample):
                raise TypeError("transform samples must contain FlowSample values")
            _validate_sample_schema(sample, self.feature_names, expected_types)
            index_by_name = _sample_index(sample, self.feature_names)
            row = []
            for name in self.feature_names:
                index, categorical = index_by_name[name]
                missing = bool(sample.missing_mask[index])
                raw = sample.features[index]
                if categorical:
                    if missing:
                        category = 0
                    elif name in self.category_encoders:
                        category = self.category_encoders[name].encode(
                            sample.category_tokens[index]
                        )
                        if type(category) is not int:
                            raise ValueError(
                                f"encoded category for {name} must be an integer"
                            )
                    else:
                        if type(raw) is bool or not isinstance(raw, (int, np.integer)):
                            raise ValueError(
                                f"categorical feature {name} must contain integer IDs"
                            )
                        category = int(raw)
                    if category not in self.categorical_values[name]:
                        category = self.unknown_category_id
                    row.extend(
                        1.0 if category == known else 0.0
                        for known in self.categorical_values[name]
                    )
                else:
                    numeric = 0.0 if missing else float(raw)
                    if not np.isfinite(numeric):
                        raise ValueError(
                            f"numeric feature {name} must be finite"
                        )
                    row.append(numeric)
                row.append(1.0 if missing else 0.0)
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "feature_names": list(self.feature_names),
            "categorical_feature_names": list(self.categorical_feature_names),
            "categorical_values": {
                name: list(values)
                for name, values in sorted(self.categorical_values.items())
            },
            "unknown_category_id": self.unknown_category_id,
            "output_feature_names": list(self.output_feature_names),
            "categorical_encoding": (
                "canonical_vocabulary"
                if self.category_encoders
                else "train_observed_ids"
            ),
            "fit_scope": "train_only",
        }


def inverse_frequency_sample_weights(targets: Sequence[int]) -> np.ndarray:
    values = np.asarray(tuple(targets))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty rank-1 sequence")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("targets must contain integer class IDs")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("targets must contain non-negative class IDs")
    counts = np.bincount(values)
    observed = counts > 0
    class_weights = np.zeros(len(counts), dtype=np.float64)
    class_weights[observed] = values.size / (
        observed.sum() * counts[observed]
    )
    return class_weights[values]


def _sample_index(
    sample: FlowSample,
    feature_names: Sequence[str],
) -> dict[str, tuple[int, bool]]:
    if len(sample.feature_ids) != len(set(sample.feature_ids)):
        raise ValueError("feature_ids must be unique")
    positions = {name: index for index, name in enumerate(sample.feature_ids)}
    if set(positions) != set(feature_names):
        raise ValueError("sample feature names do not match fitted schema")
    return {
        name: (
            positions[name],
            bool(sample.categorical_mask[positions[name]]),
        )
        for name in feature_names
    }


def _validate_sample_schema(
    sample: FlowSample,
    feature_names: Sequence[str],
    expected: Mapping[str, tuple[int, bool]],
) -> None:
    current = _sample_index(sample, feature_names)
    for name in feature_names:
        if current[name][1] != expected[name][1]:
            raise ValueError(f"feature type mismatch for {name}")


__all__ = ["TabularFeatureEncoder", "inverse_frequency_sample_weights"]
