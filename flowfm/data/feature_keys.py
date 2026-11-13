"""Stable feature-key generation before labels, scaling, encoding, or splitting."""

from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from flowfm.data.identity import stable_hash


FEATURE_KEY_NAMESPACE = "flowfm-feature-key-v1"


def canonicalize_value(value: Any, value_type: str) -> str:
    """Return a typed, deterministic token for one raw model-visible value."""

    if value is None or value == "":
        return "m:"

    if value_type == "numeric":
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid numeric feature values")
        try:
            if isinstance(value, Decimal):
                number = value
            elif isinstance(value, int):
                number = Decimal(value)
            elif isinstance(value, float):
                number = Decimal(str(value))
            elif isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return "m:"
                number = Decimal(stripped)
            else:
                raise ValueError(
                    f"unsupported numeric feature value type: {type(value).__name__}"
                )
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid numeric feature value: {value!r}") from exc
        if not number.is_finite():
            raise ValueError("non-finite numeric feature values are not allowed")
        if number == 0:
            return "n:0"
        text = format(number.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"n:{text}"

    if value_type == "string":
        if not isinstance(value, str):
            raise ValueError("string feature values must be strings")
        text = unicodedata.normalize("NFC", value.strip())
        if not text:
            return "m:"
        return f"s:{len(text)}:{text}"

    raise ValueError(f"unsupported feature value type: {value_type}")


def compute_feature_key(record: Mapping[str, Any], schema: Any) -> str:
    """Hash only the ordered model-visible raw inputs defined by ``schema``."""

    canonical_features = [
        [spec.name, canonicalize_value(record.get(spec.name), spec.value_type)]
        for spec in schema.model_features
    ]
    payload = {
        "dataset_id": schema.dataset_id,
        "schema_version": schema.version,
        "features": canonical_features,
    }
    return stable_hash(payload, FEATURE_KEY_NAMESPACE)


__all__ = ["FEATURE_KEY_NAMESPACE", "canonicalize_value", "compute_feature_key"]
