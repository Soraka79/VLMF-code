"""Raw and initial model-visible schema for NF-UNSW-NB15-v2."""

from __future__ import annotations

from dataclasses import dataclass

from flowfm.data.identity import stable_hash


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    value_type: str = "numeric"
    categorical: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("feature names must be non-empty without surrounding whitespace")
        if self.value_type not in {"numeric", "string"}:
            raise ValueError(f"unsupported feature value type: {self.value_type}")


@dataclass(frozen=True)
class DatasetFeatureSchema:
    dataset_id: str
    version: str
    raw_features: tuple[FeatureSpec, ...]
    model_features: tuple[FeatureSpec, ...]
    trace_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.version:
            raise ValueError("dataset_id and version must be non-empty")
        raw_names = self.raw_feature_names
        model_names = self.model_feature_names
        if len(set(raw_names)) != len(raw_names):
            raise ValueError("raw feature names must be unique")
        if len(set(model_names)) != len(model_names):
            raise ValueError("model feature names must be unique")
        if not set(model_names).issubset(raw_names):
            raise ValueError("model features must be a subset of raw features")
        if not set(self.trace_feature_names).issubset(raw_names):
            raise ValueError("trace features must be a subset of raw features")
        if set(model_names).intersection(self.trace_feature_names):
            raise ValueError("trace features cannot also be model-visible features")

    @property
    def raw_feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.raw_features)

    @property
    def model_feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.model_features)

    @property
    def categorical_mask(self) -> tuple[bool, ...]:
        return tuple(spec.categorical for spec in self.model_features)

    @property
    def schema_hash(self) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "raw_features": [
                [spec.name, spec.value_type, spec.categorical]
                for spec in self.raw_features
            ],
            "model_features": [
                [spec.name, spec.value_type, spec.categorical]
                for spec in self.model_features
            ],
            "trace_feature_names": list(self.trace_feature_names),
        }
        return stable_hash(payload, "flowfm-schema-v1")


RAW_FEATURE_NAMES = (
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES",
    "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
)

TRACE_FEATURE_NAMES = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")

CATEGORICAL_FEATURE_NAMES = {
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
}

RAW_FEATURES = tuple(
    FeatureSpec(
        name=name,
        value_type="string" if name in TRACE_FEATURE_NAMES else "numeric",
        categorical=name in CATEGORICAL_FEATURE_NAMES,
    )
    for name in RAW_FEATURE_NAMES
)
MODEL_FEATURES = tuple(
    spec for spec in RAW_FEATURES if spec.name not in TRACE_FEATURE_NAMES
)

NF_UNSW_NB15_V2_SCHEMA = DatasetFeatureSchema(
    dataset_id="NF-UNSW-NB15-v2",
    version="nf-unsw-nb15-v2-model-41-v1",
    raw_features=RAW_FEATURES,
    model_features=MODEL_FEATURES,
    trace_feature_names=TRACE_FEATURE_NAMES,
)


__all__ = [
    "DatasetFeatureSchema", "FeatureSpec", "NF_UNSW_NB15_V2_SCHEMA",
]
