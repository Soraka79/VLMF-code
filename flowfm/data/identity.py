"""Stable, portable identities for dataset records."""

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be hashed canonically")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mappings require string keys")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for supported JSON-like Python values.

    Mapping keys must be strings. Tuples are normalized to arrays, while
    non-finite floats and non-JSON values are rejected.
    """

    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any, namespace: str) -> str:
    """Return a namespace-separated lowercase SHA-256 digest."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    namespace_bytes = namespace.encode("utf-8")
    payload = canonical_json(value).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(namespace_bytes).to_bytes(8, "big"))
    digest.update(namespace_bytes)
    digest.update(payload)
    return digest.hexdigest()


def make_sample_id(record: Any, dataset_id: str, schema_version: str) -> str:
    """Create an ID bound to a record, dataset, and schema version."""

    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version must be a non-empty string")
    return stable_hash(
        {
            "dataset_id": dataset_id,
            "record": record,
            "schema_version": schema_version,
        },
        namespace="flowfm.sample-id.v1",
    )
