"""Stable canonical feature identities and train-only categorical metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

from flowfm.data.identity import stable_hash


FEATURE_REGISTRY_FORMAT_VERSION = 1
CATEGORY_VOCABULARY_FORMAT_VERSION = 1
TOKENIZER_METADATA_FORMAT_VERSION = 1
MISSING_CATEGORY_ID = 0
UNKNOWN_CATEGORY_ID = 1
PORT_BUCKET_POLICY_VERSION = "stable_port_buckets_v1"
DEFAULT_MAX_KNOWN_CATEGORIES = 4096

_REGISTRY_HASH_NAMESPACE = "flowfm-canonical-feature-registry-v1"
_VOCABULARY_HASH_NAMESPACE = "flowfm-categorical-vocabulary-v1"
_TOKENIZER_HASH_NAMESPACE = "flowfm-tokenizer-metadata-v1"
_MISSING_TOKEN = "<MISSING>"
_UNKNOWN_TOKEN = "<UNKNOWN>"


@dataclass(frozen=True)
class CanonicalFeatureSpec:
    feature_id: int
    name: str
    feature_type: str
    value_policy: str
    model_visible: bool = True

    def __post_init__(self) -> None:
        if type(self.feature_id) is not int or self.feature_id <= 0:
            raise ValueError("feature_id must be a positive integer")
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name.strip() != self.name
        ):
            raise ValueError(
                "feature name must be non-empty without surrounding whitespace"
            )
        if self.feature_type not in {"numeric", "categorical"}:
            raise ValueError("feature_type must be numeric or categorical")
        if type(self.model_visible) is not bool:
            raise ValueError("model_visible must be boolean")
        if self.feature_type == "numeric":
            if self.value_policy != "continuous":
                raise ValueError(
                    "categorical value policies require a categorical "
                    "feature"
                )
        elif self.value_policy not in {
            "vocabulary",
            PORT_BUCKET_POLICY_VERSION,
            "audit_only",
        }:
            raise ValueError(
                "categorical features use an unsupported value policy"
            )
        if self.value_policy == "audit_only" and self.model_visible:
            raise ValueError("audit_only features cannot be model-visible")
        if self.value_policy != "audit_only" and not self.model_visible:
            raise ValueError("non-audit features must be model-visible")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "feature_type": self.feature_type,
            "value_policy": self.value_policy,
            "model_visible": self.model_visible,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CanonicalFeatureSpec":
        required = {
            "feature_id",
            "name",
            "feature_type",
            "value_policy",
            "model_visible",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("canonical feature payload has invalid fields")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CanonicalFeatureRegistry:
    version: str
    features: tuple[CanonicalFeatureSpec, ...]
    _by_id: Mapping[int, CanonicalFeatureSpec] = field(
        init=False, repr=False, compare=False
    )
    _by_name: Mapping[str, CanonicalFeatureSpec] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("registry version must be a non-empty string")
        if not isinstance(self.features, (list, tuple)) or not self.features:
            raise ValueError(
                "registry features must be a non-empty sequence"
            )
        normalized = tuple(
            sorted(self.features, key=lambda spec: spec.feature_id)
        )
        if not all(
            isinstance(spec, CanonicalFeatureSpec) for spec in normalized
        ):
            raise TypeError(
                "registry features must be CanonicalFeatureSpec values"
            )
        by_id = {}
        by_name = {}
        for spec in normalized:
            if spec.feature_id in by_id:
                raise ValueError(f"duplicate feature_id: {spec.feature_id}")
            if spec.name in by_name:
                raise ValueError(f"duplicate feature name: {spec.name}")
            by_id[spec.feature_id] = spec
            by_name[spec.name] = spec
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    @property
    def by_id(self) -> Mapping[int, CanonicalFeatureSpec]:
        return self._by_id

    @property
    def by_name(self) -> Mapping[str, CanonicalFeatureSpec]:
        return self._by_name

    @property
    def model_features(self) -> tuple[CanonicalFeatureSpec, ...]:
        return tuple(spec for spec in self.features if spec.model_visible)

    @property
    def categorical_model_features(
        self,
    ) -> tuple[CanonicalFeatureSpec, ...]:
        return tuple(
            spec
            for spec in self.model_features
            if spec.feature_type == "categorical"
        )

    @property
    def registry_hash(self) -> str:
        return stable_hash(self.to_dict(), _REGISTRY_HASH_NAMESPACE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FEATURE_REGISTRY_FORMAT_VERSION,
            "version": self.version,
            "features": [spec.to_dict() for spec in self.features],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CanonicalFeatureRegistry":
        required = {"format_version", "version", "features"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("canonical registry payload has invalid fields")
        if payload["format_version"] != FEATURE_REGISTRY_FORMAT_VERSION:
            raise ValueError(
                "unsupported canonical registry format version"
            )
        features = payload["features"]
        if not isinstance(features, list):
            raise ValueError(
                "canonical registry features must be a list"
            )
        return cls(
            version=payload["version"],
            features=tuple(
                CanonicalFeatureSpec.from_dict(item) for item in features
            ),
        )


def _validate_fit_inputs(
    spec: CanonicalFeatureSpec,
    counts: Mapping[str, int],
    *,
    training_row_count: int,
    training_sample_ids_hash: str,
    fit_scope: str,
) -> dict[str, int]:
    if not isinstance(spec, CanonicalFeatureSpec):
        raise TypeError("spec must be a CanonicalFeatureSpec")
    if spec.feature_type != "categorical" or not spec.model_visible:
        raise ValueError(
            "vocabularies require a model-visible categorical feature"
        )
    if fit_scope != "train_only":
        raise ValueError("fit_scope must be train_only")
    if type(training_row_count) is not int or training_row_count <= 0:
        raise ValueError(
            "training_row_count must be a positive integer"
        )
    if (
        not isinstance(training_sample_ids_hash, str)
        or not training_sample_ids_hash
    ):
        raise ValueError(
            "training_sample_ids_hash must be a non-empty string"
        )
    if not isinstance(counts, Mapping):
        raise TypeError("category counts must be a mapping")
    normalized = {}
    for token, count in counts.items():
        if not isinstance(token, str) or not token:
            raise ValueError(
                "category tokens must be non-empty strings"
            )
        if token == "m:":
            raise ValueError(
                "missing token must not enter a known-category vocabulary"
            )
        if type(count) is not int or count <= 0:
            raise ValueError(
                "category counts must be positive integers"
            )
        normalized[token] = count
    if sum(normalized.values()) > training_row_count:
        raise ValueError(
            "category counts exceed training_row_count"
        )
    return normalized


def _parse_port(token: Any) -> int | None:
    if token is None or token == "m:":
        return None
    if isinstance(token, bool):
        return -1
    if isinstance(token, int):
        value = Decimal(token)
    elif isinstance(token, str) and token.startswith("n:"):
        try:
            value = Decimal(token[2:])
        except InvalidOperation:
            return -1
    else:
        return -1
    if not value.is_finite() or value != value.to_integral_value():
        return -1
    port = int(value)
    if not 0 <= port <= 65535:
        return -1
    return port


def _port_bucket_index(port: int) -> int:
    if 0 <= port <= 1023:
        return port
    if port <= 49151:
        return 1024 + (port - 1024) // 256
    return 1212 + (port - 49152) // 256


def _port_bucket_labels() -> tuple[str, ...]:
    labels = [f"port:{port}" for port in range(1024)]
    labels.extend(
        f"port:{start}-{min(start + 255, 49151)}"
        for start in range(1024, 49152, 256)
    )
    labels.extend(
        f"port:{start}-{min(start + 255, 65535)}"
        for start in range(49152, 65536, 256)
    )
    if len(labels) != 1276:
        raise RuntimeError(
            "stable port policy produced an unexpected bucket count"
        )
    return tuple(labels)


_PORT_BUCKET_LABELS = _port_bucket_labels()


@dataclass(frozen=True)
class CategoricalVocabulary:
    feature_id: int
    feature_name: str
    value_policy: str
    id_to_token: tuple[str, ...]
    known_frequencies: tuple[int, ...]
    training_row_count: int
    training_sample_ids_hash: str
    max_known_categories: int
    overflow_unique_count: int
    overflow_frequency: int
    fit_scope: str = "train_only"
    selection_rule: str = ""
    vocabulary_hash: str = ""
    _token_to_id: Mapping[str, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.feature_id) is not int or self.feature_id <= 0:
            raise ValueError(
                "vocabulary feature_id must be a positive integer"
            )
        if (
            not isinstance(self.feature_name, str)
            or not self.feature_name
        ):
            raise ValueError(
                "vocabulary feature_name must be non-empty"
            )
        if self.value_policy not in {
            "vocabulary",
            PORT_BUCKET_POLICY_VERSION,
        }:
            raise ValueError(
                "unsupported categorical value policy"
            )
        if not isinstance(self.id_to_token, (list, tuple)):
            raise ValueError("id_to_token must be a sequence")
        normalized_tokens = tuple(self.id_to_token)
        if normalized_tokens[:2] != (_MISSING_TOKEN, _UNKNOWN_TOKEN):
            raise ValueError(
                "reserved category token definitions are invalid"
            )
        known_tokens = normalized_tokens[2:]
        if len(set(known_tokens)) != len(known_tokens):
            raise ValueError(
                "known category tokens must be unique"
            )
        if not all(
            isinstance(token, str) and token for token in known_tokens
        ):
            raise ValueError(
                "known category tokens must be non-empty strings"
            )
        normalized_frequencies = tuple(self.known_frequencies)
        if len(normalized_frequencies) != len(known_tokens):
            raise ValueError(
                "known_frequencies must align with known category tokens"
            )
        if not all(
            type(value) is int and value >= 0
            for value in normalized_frequencies
        ):
            raise ValueError(
                "known category frequencies must be non-negative integers"
            )
        for field_name in (
            "training_row_count",
            "max_known_categories",
            "overflow_unique_count",
            "overflow_frequency",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )
        if (
            self.training_row_count <= 0
            or self.max_known_categories <= 0
        ):
            raise ValueError(
                "training_row_count and max_known_categories must be positive"
            )
        if len(known_tokens) > self.max_known_categories:
            raise ValueError(
                "known categories exceed max_known_categories"
            )
        if (
            sum(normalized_frequencies) + self.overflow_frequency
            > self.training_row_count
        ):
            raise ValueError(
                "vocabulary frequencies exceed training_row_count"
            )
        if self.fit_scope != "train_only":
            raise ValueError("fit_scope must be train_only")
        if (
            not isinstance(self.training_sample_ids_hash, str)
            or not self.training_sample_ids_hash
        ):
            raise ValueError(
                "training_sample_ids_hash must be non-empty"
            )
        if (
            not isinstance(self.selection_rule, str)
            or not self.selection_rule
        ):
            raise ValueError("selection_rule must be non-empty")
        token_to_id = {
            token: index
            for index, token in enumerate(known_tokens, start=2)
        }
        object.__setattr__(self, "id_to_token", normalized_tokens)
        object.__setattr__(
            self, "known_frequencies", normalized_frequencies
        )
        object.__setattr__(
            self, "_token_to_id", MappingProxyType(token_to_id)
        )
        expected_hash = stable_hash(
            self._hash_payload(), _VOCABULARY_HASH_NAMESPACE
        )
        if (
            self.vocabulary_hash
            and self.vocabulary_hash != expected_hash
        ):
            raise ValueError(
                "vocabulary_hash does not match vocabulary content"
            )
        object.__setattr__(
            self, "vocabulary_hash", expected_hash
        )

    @property
    def table_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, token: Any) -> int:
        if token is None or token == "m:":
            return MISSING_CATEGORY_ID
        if self.value_policy == PORT_BUCKET_POLICY_VERSION:
            port = _parse_port(token)
            if port is None:
                return MISSING_CATEGORY_ID
            if port < 0:
                return UNKNOWN_CATEGORY_ID
            return _port_bucket_index(port) + 2
        if not isinstance(token, str):
            return UNKNOWN_CATEGORY_ID
        return self._token_to_id.get(
            token, UNKNOWN_CATEGORY_ID
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "format_version": CATEGORY_VOCABULARY_FORMAT_VERSION,
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "value_policy": self.value_policy,
            "id_to_token": list(self.id_to_token),
            "known_frequencies": list(self.known_frequencies),
            "training_row_count": self.training_row_count,
            "training_sample_ids_hash": self.training_sample_ids_hash,
            "max_known_categories": self.max_known_categories,
            "overflow_unique_count": self.overflow_unique_count,
            "overflow_frequency": self.overflow_frequency,
            "fit_scope": self.fit_scope,
            "selection_rule": self.selection_rule,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["vocabulary_hash"] = self.vocabulary_hash
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CategoricalVocabulary":
        required = {
            "format_version",
            "feature_id",
            "feature_name",
            "value_policy",
            "id_to_token",
            "known_frequencies",
            "training_row_count",
            "training_sample_ids_hash",
            "max_known_categories",
            "overflow_unique_count",
            "overflow_frequency",
            "fit_scope",
            "selection_rule",
            "vocabulary_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError(
                "categorical vocabulary payload has invalid fields"
            )
        if (
            payload["format_version"]
            != CATEGORY_VOCABULARY_FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported categorical vocabulary format version"
            )
        values = dict(payload)
        values.pop("format_version")
        return cls(**values)


def fit_categorical_vocabulary(
    spec: CanonicalFeatureSpec,
    counts: Mapping[str, int],
    *,
    training_row_count: int,
    training_sample_ids_hash: str,
    max_known_categories: int = DEFAULT_MAX_KNOWN_CATEGORIES,
    fit_scope: str = "train_only",
) -> CategoricalVocabulary:
    normalized = _validate_fit_inputs(
        spec,
        counts,
        training_row_count=training_row_count,
        training_sample_ids_hash=training_sample_ids_hash,
        fit_scope=fit_scope,
    )
    if spec.value_policy != "vocabulary":
        raise ValueError(
            "feature must use the vocabulary value policy"
        )
    if (
        type(max_known_categories) is not int
        or max_known_categories <= 0
    ):
        raise ValueError(
            "max_known_categories must be a positive integer"
        )
    ranked = sorted(
        normalized, key=lambda token: (-normalized[token], token)
    )
    retained = tuple(sorted(ranked[:max_known_categories]))
    overflow = ranked[max_known_categories:]
    return CategoricalVocabulary(
        feature_id=spec.feature_id,
        feature_name=spec.name,
        value_policy=spec.value_policy,
        id_to_token=(_MISSING_TOKEN, _UNKNOWN_TOKEN, *retained),
        known_frequencies=tuple(
            normalized[token] for token in retained
        ),
        training_row_count=training_row_count,
        training_sample_ids_hash=training_sample_ids_hash,
        max_known_categories=max_known_categories,
        overflow_unique_count=len(overflow),
        overflow_frequency=sum(
            normalized[token] for token in overflow
        ),
        fit_scope=fit_scope,
        selection_rule=(
            "frequency_desc_token_asc_then_id_token_asc"
        ),
    )


def build_port_vocabulary(
    spec: CanonicalFeatureSpec,
    counts: Mapping[str, int],
    *,
    training_row_count: int,
    training_sample_ids_hash: str,
    fit_scope: str = "train_only",
) -> CategoricalVocabulary:
    normalized = _validate_fit_inputs(
        spec,
        counts,
        training_row_count=training_row_count,
        training_sample_ids_hash=training_sample_ids_hash,
        fit_scope=fit_scope,
    )
    if spec.value_policy != PORT_BUCKET_POLICY_VERSION:
        raise ValueError(
            "feature must use the stable port bucket policy"
        )
    frequencies = [0] * len(_PORT_BUCKET_LABELS)
    invalid_tokens = []
    invalid_frequency = 0
    for token, count in normalized.items():
        port = _parse_port(token)
        if port is None or port < 0:
            invalid_tokens.append(token)
            invalid_frequency += count
        else:
            frequencies[_port_bucket_index(port)] += count
    return CategoricalVocabulary(
        feature_id=spec.feature_id,
        feature_name=spec.name,
        value_policy=spec.value_policy,
        id_to_token=(
            _MISSING_TOKEN,
            _UNKNOWN_TOKEN,
            *_PORT_BUCKET_LABELS,
        ),
        known_frequencies=tuple(frequencies),
        training_row_count=training_row_count,
        training_sample_ids_hash=training_sample_ids_hash,
        max_known_categories=len(_PORT_BUCKET_LABELS),
        overflow_unique_count=len(invalid_tokens),
        overflow_frequency=invalid_frequency,
        fit_scope=fit_scope,
        selection_rule="fixed_stable_port_buckets_v1",
    )


@dataclass(frozen=True)
class TokenizerMetadata:
    registry_version: str
    registry_hash: str
    schema_version: str
    schema_hash: str
    categorical_vocabularies: Mapping[
        str, CategoricalVocabulary
    ]
    data_manifest: Mapping[str, Any]
    split_manifest_sha256: str
    preprocessor_hash: str
    training_sample_ids_hash: str
    metadata_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "registry_version",
            "registry_hash",
            "schema_version",
            "schema_hash",
            "split_manifest_sha256",
            "preprocessor_hash",
            "training_sample_ids_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{field_name} must be a non-empty string"
                )
        if not isinstance(
            self.categorical_vocabularies, Mapping
        ):
            raise TypeError(
                "categorical_vocabularies must be a mapping"
            )
        normalized_vocabularies = {}
        for name, vocabulary in sorted(
            self.categorical_vocabularies.items()
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "categorical vocabulary keys must be non-empty strings"
                )
            if not isinstance(
                vocabulary, CategoricalVocabulary
            ):
                raise TypeError(
                    "categorical vocabularies must contain "
                    "CategoricalVocabulary"
                )
            if name != vocabulary.feature_name:
                raise ValueError(
                    "categorical vocabulary key does not match feature name"
                )
            if (
                vocabulary.training_sample_ids_hash
                != self.training_sample_ids_hash
            ):
                raise ValueError(
                    "vocabulary training sample hash does not match metadata"
                )
            normalized_vocabularies[name] = vocabulary
        if (
            not isinstance(self.data_manifest, Mapping)
            or not self.data_manifest
        ):
            raise ValueError(
                "data_manifest must be a non-empty mapping"
            )
        object.__setattr__(
            self,
            "categorical_vocabularies",
            MappingProxyType(normalized_vocabularies),
        )
        object.__setattr__(
            self,
            "data_manifest",
            MappingProxyType(dict(self.data_manifest)),
        )
        expected_hash = stable_hash(
            self._hash_payload(), _TOKENIZER_HASH_NAMESPACE
        )
        if (
            self.metadata_hash
            and self.metadata_hash != expected_hash
        ):
            raise ValueError(
                "metadata_hash does not match tokenizer metadata"
            )
        object.__setattr__(
            self, "metadata_hash", expected_hash
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "format_version": TOKENIZER_METADATA_FORMAT_VERSION,
            "registry_version": self.registry_version,
            "registry_hash": self.registry_hash,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "reserved_category_ids": {
                "missing_padding": MISSING_CATEGORY_ID,
                "unknown": UNKNOWN_CATEGORY_ID,
            },
            "categorical_vocabularies": {
                name: vocabulary.to_dict()
                for name, vocabulary
                in self.categorical_vocabularies.items()
            },
            "data_manifest": dict(self.data_manifest),
            "split_manifest_sha256": (
                self.split_manifest_sha256
            ),
            "preprocessor_hash": self.preprocessor_hash,
            "training_sample_ids_hash": (
                self.training_sample_ids_hash
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["metadata_hash"] = self.metadata_hash
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TokenizerMetadata":
        required = {
            "format_version",
            "registry_version",
            "registry_hash",
            "schema_version",
            "schema_hash",
            "reserved_category_ids",
            "categorical_vocabularies",
            "data_manifest",
            "split_manifest_sha256",
            "preprocessor_hash",
            "training_sample_ids_hash",
            "metadata_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError(
                "tokenizer metadata payload has invalid fields"
            )
        if (
            payload["format_version"]
            != TOKENIZER_METADATA_FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported tokenizer metadata format version"
            )
        if payload["reserved_category_ids"] != {
            "missing_padding": MISSING_CATEGORY_ID,
            "unknown": UNKNOWN_CATEGORY_ID,
        }:
            raise ValueError(
                "reserved category IDs do not match implementation"
            )
        raw_vocabularies = payload["categorical_vocabularies"]
        if not isinstance(raw_vocabularies, Mapping):
            raise ValueError(
                "categorical_vocabularies must be a mapping"
            )
        return cls(
            registry_version=payload["registry_version"],
            registry_hash=payload["registry_hash"],
            schema_version=payload["schema_version"],
            schema_hash=payload["schema_hash"],
            categorical_vocabularies={
                name: CategoricalVocabulary.from_dict(vocabulary)
                for name, vocabulary in raw_vocabularies.items()
            },
            data_manifest=payload["data_manifest"],
            split_manifest_sha256=(
                payload["split_manifest_sha256"]
            ),
            preprocessor_hash=payload["preprocessor_hash"],
            training_sample_ids_hash=(
                payload["training_sample_ids_hash"]
            ),
            metadata_hash=payload["metadata_hash"],
        )


def _spec(
    feature_id: int,
    name: str,
    feature_type: str,
    value_policy: str | None = None,
    *,
    model_visible: bool = True,
) -> CanonicalFeatureSpec:
    if value_policy is None:
        value_policy = (
            "continuous"
            if feature_type == "numeric"
            else "vocabulary"
        )
    return CanonicalFeatureSpec(
        feature_id,
        name,
        feature_type,
        value_policy,
        model_visible,
    )


# IDs are explicit and append-only. The audit-only DNS transaction
# identifier keeps its ID gap instead of renumbering model tokens.
SCHEMAFLOW_MAIN_REGISTRY = CanonicalFeatureRegistry(
    "schemaflow-main-features-v1",
    (
        _spec(
            1,
            "L4_SRC_PORT",
            "categorical",
            PORT_BUCKET_POLICY_VERSION,
        ),
        _spec(
            2,
            "L4_DST_PORT",
            "categorical",
            PORT_BUCKET_POLICY_VERSION,
        ),
        _spec(3, "PROTOCOL", "categorical"),
        _spec(4, "L7_PROTO", "categorical"),
        _spec(5, "IN_BYTES", "numeric"),
        _spec(6, "IN_PKTS", "numeric"),
        _spec(7, "OUT_BYTES", "numeric"),
        _spec(8, "OUT_PKTS", "numeric"),
        _spec(9, "TCP_FLAGS", "categorical"),
        _spec(10, "CLIENT_TCP_FLAGS", "categorical"),
        _spec(11, "SERVER_TCP_FLAGS", "categorical"),
        _spec(12, "FLOW_DURATION_MILLISECONDS", "numeric"),
        _spec(13, "DURATION_IN", "numeric"),
        _spec(14, "DURATION_OUT", "numeric"),
        _spec(15, "MIN_TTL", "numeric"),
        _spec(16, "MAX_TTL", "numeric"),
        _spec(17, "LONGEST_FLOW_PKT", "numeric"),
        _spec(18, "SHORTEST_FLOW_PKT", "numeric"),
        _spec(19, "MIN_IP_PKT_LEN", "numeric"),
        _spec(20, "MAX_IP_PKT_LEN", "numeric"),
        _spec(21, "SRC_TO_DST_SECOND_BYTES", "numeric"),
        _spec(22, "DST_TO_SRC_SECOND_BYTES", "numeric"),
        _spec(23, "RETRANSMITTED_IN_BYTES", "numeric"),
        _spec(24, "RETRANSMITTED_IN_PKTS", "numeric"),
        _spec(25, "RETRANSMITTED_OUT_BYTES", "numeric"),
        _spec(26, "RETRANSMITTED_OUT_PKTS", "numeric"),
        _spec(27, "SRC_TO_DST_AVG_THROUGHPUT", "numeric"),
        _spec(28, "DST_TO_SRC_AVG_THROUGHPUT", "numeric"),
        _spec(29, "NUM_PKTS_UP_TO_128_BYTES", "numeric"),
        _spec(30, "NUM_PKTS_128_TO_256_BYTES", "numeric"),
        _spec(31, "NUM_PKTS_256_TO_512_BYTES", "numeric"),
        _spec(32, "NUM_PKTS_512_TO_1024_BYTES", "numeric"),
        _spec(33, "NUM_PKTS_1024_TO_1514_BYTES", "numeric"),
        _spec(34, "TCP_WIN_MAX_IN", "numeric"),
        _spec(35, "TCP_WIN_MAX_OUT", "numeric"),
        _spec(36, "ICMP_TYPE", "categorical"),
        _spec(37, "ICMP_IPV4_TYPE", "categorical"),
        _spec(
            38,
            "DNS_QUERY_ID",
            "categorical",
            "audit_only",
            model_visible=False,
        ),
        _spec(39, "DNS_QUERY_TYPE", "categorical"),
        _spec(40, "DNS_TTL_ANSWER", "numeric"),
        _spec(41, "FTP_COMMAND_RET_CODE", "categorical"),
    ),
)


__all__ = [
    "CATEGORY_VOCABULARY_FORMAT_VERSION",
    "DEFAULT_MAX_KNOWN_CATEGORIES",
    "FEATURE_REGISTRY_FORMAT_VERSION",
    "MISSING_CATEGORY_ID",
    "PORT_BUCKET_POLICY_VERSION",
    "SCHEMAFLOW_MAIN_REGISTRY",
    "TOKENIZER_METADATA_FORMAT_VERSION",
    "UNKNOWN_CATEGORY_ID",
    "CanonicalFeatureRegistry",
    "CanonicalFeatureSpec",
    "CategoricalVocabulary",
    "TokenizerMetadata",
    "build_port_vocabulary",
    "fit_categorical_vocabulary",
]
