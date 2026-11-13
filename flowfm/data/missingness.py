"""Deterministic field-missing perturbations for robustness evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from types import MappingProxyType
from typing import Sequence

from flowfm.data.contracts import FlowSample
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY


MISSINGNESS_ALGORITHM = "sha256-sample-feature-ranking-v1"
STRUCTURED_MISSINGNESS_ALGORITHM = (
    "fixed-netflow-v2-collection-field-groups-v1"
)
_HASH_NAMESPACE = "schemaflow-field-missingness-v1"
STRUCTURED_MISSINGNESS_GROUPS = MappingProxyType(
    {
        "transport_protocol": (
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
        ),
        "traffic_volume_timing": (
            "IN_BYTES",
            "IN_PKTS",
            "OUT_BYTES",
            "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS",
            "DURATION_IN",
            "DURATION_OUT",
            "SRC_TO_DST_SECOND_BYTES",
            "DST_TO_SRC_SECOND_BYTES",
            "SRC_TO_DST_AVG_THROUGHPUT",
            "DST_TO_SRC_AVG_THROUGHPUT",
        ),
        "tcp_state_retransmission": (
            "TCP_FLAGS",
            "CLIENT_TCP_FLAGS",
            "SERVER_TCP_FLAGS",
            "RETRANSMITTED_IN_BYTES",
            "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES",
            "RETRANSMITTED_OUT_PKTS",
            "TCP_WIN_MAX_IN",
            "TCP_WIN_MAX_OUT",
        ),
        "packet_length_distribution": (
            "MIN_TTL",
            "MAX_TTL",
            "LONGEST_FLOW_PKT",
            "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN",
            "MAX_IP_PKT_LEN",
            "NUM_PKTS_UP_TO_128_BYTES",
            "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES",
            "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES",
        ),
        "application_protocol_metadata": (
            "ICMP_TYPE",
            "ICMP_IPV4_TYPE",
            "DNS_QUERY_TYPE",
            "DNS_TTL_ANSWER",
            "FTP_COMMAND_RET_CODE",
        ),
    }
)


def _validate_structured_groups() -> None:
    flattened = tuple(
        feature
        for features in STRUCTURED_MISSINGNESS_GROUPS.values()
        for feature in features
    )
    expected = tuple(
        spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.model_features
    )
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise RuntimeError(
            "structured missingness groups must partition all NetFlow-v2 "
            "model-visible fields"
        )


_validate_structured_groups()


def _validate_fraction(fraction: float) -> float:
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("missing fraction must be numeric")
    value = float(fraction)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("missing fraction must be between zero and one")
    return value


def _rank(seed: int, sample_id: str, feature_id: str) -> bytes:
    value = "\0".join((_HASH_NAMESPACE, str(seed), sample_id, feature_id))
    return hashlib.sha256(value.encode("utf-8")).digest()


def apply_deterministic_field_missingness(
    samples: Sequence[FlowSample],
    *,
    fraction: float,
    seed: int,
    eligible_feature_ids: Sequence[str] | None = None,
) -> tuple[tuple[FlowSample, ...], dict[str, object]]:
    """Hide an exact floor(fraction * feature_count) fields in each sample."""
    if type(seed) is not int or seed < 0:
        raise ValueError("missingness seed must be a non-negative integer")
    fraction = _validate_fraction(fraction)
    values = tuple(samples)
    if not values:
        raise ValueError("missingness evaluation requires non-empty samples")
    if any(not isinstance(sample, FlowSample) for sample in values):
        raise TypeError("missingness samples must contain FlowSample values")

    if eligible_feature_ids is None:
        eligible = tuple(values[0].feature_ids)
    else:
        eligible = tuple(eligible_feature_ids)
    if not eligible or len(set(eligible)) != len(eligible):
        raise ValueError("eligible feature IDs must be non-empty and unique")

    hidden_count = math.floor(fraction * len(eligible))
    feature_selection_counts = Counter()
    assignment_records = []
    perturbed = []
    preexisting_missing_selected = 0

    for sample in values:
        index_by_name = {
            name: index for index, name in enumerate(sample.feature_ids)
        }
        if len(index_by_name) != len(sample.feature_ids):
            raise ValueError("sample feature IDs must be unique")
        absent = [name for name in eligible if name not in index_by_name]
        if absent:
            raise ValueError(
                "eligible feature IDs are absent from sample: " + ", ".join(absent)
            )
        selected = tuple(
            sorted(
                eligible,
                key=lambda name: (_rank(seed, sample.sample_id, name), name),
            )[:hidden_count]
        )
        features = list(sample.features)
        missing_mask = list(sample.missing_mask)
        category_tokens = list(sample.category_tokens)
        for name in selected:
            index = index_by_name[name]
            preexisting_missing_selected += int(missing_mask[index])
            features[index] = 0
            missing_mask[index] = True
            category_tokens[index] = None
            feature_selection_counts[name] += 1
        if len(set(selected)) != hidden_count:
            raise RuntimeError("missingness selection did not produce an exact field count")
        perturbed.append(
            replace(
                sample,
                features=tuple(features),
                missing_mask=tuple(missing_mask),
                category_tokens=tuple(category_tokens),
            )
        )
        assignment_records.append((sample.sample_id, selected))

    assignment_digest = hashlib.sha256()
    for sample_id, selected in sorted(assignment_records):
        assignment_digest.update(
            json.dumps(
                [sample_id, list(selected)],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        assignment_digest.update(b"\n")

    report = {
        "format_version": 1,
        "algorithm": MISSINGNESS_ALGORITHM,
        "seed": seed,
        "fraction": fraction,
        "sample_count": len(values),
        "eligible_feature_count": len(eligible),
        "hidden_fields_per_sample": hidden_count,
        "total_selected_fields": hidden_count * len(values),
        "preexisting_missing_selected": preexisting_missing_selected,
        "selected_assignments_sha256": assignment_digest.hexdigest(),
        "eligible_feature_ids": list(eligible),
        "feature_selection_counts": {
            name: feature_selection_counts[name] for name in eligible
        },
        "training_data_perturbed": False,
    }
    return tuple(perturbed), report


def apply_structured_field_group_missingness(
    samples: Sequence[FlowSample],
    *,
    group_name: str,
) -> tuple[tuple[FlowSample, ...], dict[str, object]]:
    """Hide one frozen collection-mechanism field group in every sample."""

    if (
        not isinstance(group_name, str)
        or group_name not in STRUCTURED_MISSINGNESS_GROUPS
    ):
        raise ValueError(
            "structured missingness group must be one of: "
            + ", ".join(STRUCTURED_MISSINGNESS_GROUPS)
        )
    values = tuple(samples)
    if not values:
        raise ValueError("structured missingness requires non-empty samples")
    if any(not isinstance(sample, FlowSample) for sample in values):
        raise TypeError("structured missingness samples must contain FlowSample values")

    selected = STRUCTURED_MISSINGNESS_GROUPS[group_name]
    feature_selection_counts = Counter()
    preexisting_missing_selected = 0
    perturbed = []
    assignment_records = []
    for sample in values:
        index_by_name = {
            name: index for index, name in enumerate(sample.feature_ids)
        }
        if len(index_by_name) != len(sample.feature_ids):
            raise ValueError("sample feature IDs must be unique")
        absent = [name for name in selected if name not in index_by_name]
        if absent:
            raise ValueError(
                "structured group fields are absent from sample: "
                + ", ".join(absent)
            )
        features = list(sample.features)
        missing_mask = list(sample.missing_mask)
        category_tokens = list(sample.category_tokens)
        for name in selected:
            index = index_by_name[name]
            preexisting_missing_selected += int(missing_mask[index])
            features[index] = 0
            missing_mask[index] = True
            category_tokens[index] = None
            feature_selection_counts[name] += 1
        perturbed.append(
            replace(
                sample,
                features=tuple(features),
                missing_mask=tuple(missing_mask),
                category_tokens=tuple(category_tokens),
            )
        )
        assignment_records.append(
            (sample.sample_id, group_name, selected)
        )

    assignment_digest = hashlib.sha256()
    for sample_id, assigned_group, assigned_fields in sorted(assignment_records):
        assignment_digest.update(
            json.dumps(
                [sample_id, assigned_group, list(assigned_fields)],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        assignment_digest.update(b"\n")

    return tuple(perturbed), {
        "format_version": 1,
        "algorithm": STRUCTURED_MISSINGNESS_ALGORITHM,
        "group_name": group_name,
        "sample_count": len(values),
        "hidden_fields_per_sample": len(selected),
        "total_selected_fields": len(selected) * len(values),
        "preexisting_missing_selected": preexisting_missing_selected,
        "selected_assignments_sha256": assignment_digest.hexdigest(),
        "selected_feature_ids": list(selected),
        "feature_selection_counts": {
            name: feature_selection_counts[name] for name in selected
        },
        "training_data_perturbed": False,
    }


__all__ = [
    "MISSINGNESS_ALGORITHM",
    "STRUCTURED_MISSINGNESS_ALGORITHM",
    "STRUCTURED_MISSINGNESS_GROUPS",
    "apply_deterministic_field_missingness",
    "apply_structured_field_group_missingness",
]
