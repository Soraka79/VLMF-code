"""Exporter-profile field loss and observed-mask condition resolution.

This module extends the frozen v1 missingness conditions for the revision
experiments. Profiles describe which NetFlow-v2 fields a collector state no
longer exports; the resolver maps the observed missing-field set of one record
to a registered condition, or to ``None`` when the caller must apply the
fixed-fusion fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from flowfm.data.contracts import FlowSample
from flowfm.data.missingness import STRUCTURED_MISSINGNESS_GROUPS
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY


EXPORTER_MISSINGNESS_ALGORITHM = "fixed-netflow-v2-exporter-profiles-v1"
INTERMITTENT_MISSINGNESS_ALGORITHM = "sha256-sample-intermittent-group-v1"
_INTERMITTENT_NAMESPACE = "flowfm-intermittent-field-loss-v1"
MODEL_FIELDS = tuple(spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.model_features)

# Fields kept by the twelve-field nProbe template of the NF-v1 datasets
# (Sarhan et al. 2021), excluding the two IPv4 addresses that are never model
# inputs here.
_NF_V1_OBSERVED = (
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
)

EXPORTER_PROFILES = MappingProxyType(
    {
        # Exporter downgraded to the NF-v1 basic template.
        "nf_v1_basic_template": tuple(
            name for name in MODEL_FIELDS if name not in _NF_V1_OBSERVED
        ),
        # Deep packet inspection disabled: payload-derived fields vanish while
        # header-derived ICMP fields remain.
        "dpi_disabled": (
            "L7_PROTO",
            "DNS_QUERY_TYPE",
            "DNS_TTL_ANSWER",
            "FTP_COMMAND_RET_CODE",
        ),
        # Asymmetric routing or a single-direction tap: every field that needs
        # server-to-client packets is lost.
        "client_to_server_only": (
            "OUT_BYTES",
            "OUT_PKTS",
            "DURATION_OUT",
            "DST_TO_SRC_SECOND_BYTES",
            "DST_TO_SRC_AVG_THROUGHPUT",
            "SERVER_TCP_FLAGS",
            "RETRANSMITTED_OUT_BYTES",
            "RETRANSMITTED_OUT_PKTS",
            "TCP_WIN_MAX_OUT",
            "DNS_TTL_ANSWER",
            "FTP_COMMAND_RET_CODE",
        ),
    }
)


def _union(*names: str) -> tuple[str, ...]:
    fields = set()
    for name in names:
        fields.update(
            STRUCTURED_MISSINGNESS_GROUPS.get(name) or EXPORTER_PROFILES[name]
        )
    return tuple(name for name in MODEL_FIELDS if name in fields)


# Overlapping failures that are never registered. The first has the same size
# as the 50% random condition, so a count-based rule absorbs it.
COMPOUND_PROFILES = MappingProxyType(
    {
        "compound_tcp_state_packet_length": _union(
            "tcp_state_retransmission", "packet_length_distribution"
        ),
        "compound_transport_application": _union(
            "transport_protocol", "application_protocol_metadata"
        ),
        "compound_dpi_client_to_server": _union(
            "dpi_disabled", "client_to_server_only"
        ),
    }
)

# A parser that intermittently drops the application metadata group.
INTERMITTENT_PROFILES = MappingProxyType(
    {
        "intermittent_application_metadata": MappingProxyType(
            {"group": "application_protocol_metadata", "fraction": 0.2}
        ),
    }
)


def _validate_field_sets() -> None:
    for registry in (EXPORTER_PROFILES, COMPOUND_PROFILES):
        for name, fields in registry.items():
            if not fields or len(set(fields)) != len(fields):
                raise RuntimeError(f"profile {name} must list unique fields")
            unknown = set(fields) - set(MODEL_FIELDS)
            if unknown:
                raise RuntimeError(
                    f"profile {name} names non-model fields: {sorted(unknown)}"
                )
    for name, spec in INTERMITTENT_PROFILES.items():
        if spec["group"] not in STRUCTURED_MISSINGNESS_GROUPS:
            raise RuntimeError(f"intermittent profile {name} has unknown group")
        if not 0.0 < spec["fraction"] < 1.0:
            raise RuntimeError(f"intermittent profile {name} fraction is invalid")


_validate_field_sets()


def _hide_fields(samples, selections, *, condition, algorithm, extra=None):
    """Hide ``selections[i]`` fields in sample ``i`` and report the assignment."""

    perturbed = []
    assignment_records = []
    feature_selection_counts = Counter()
    preexisting_missing_selected = 0
    for sample, selected in zip(samples, selections):
        index_by_name = {name: index for index, name in enumerate(sample.feature_ids)}
        absent = [name for name in selected if name not in index_by_name]
        if absent:
            raise ValueError(
                "profile fields are absent from sample: " + ", ".join(absent)
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
        assignment_records.append((sample.sample_id, condition, list(selected)))

    digest = hashlib.sha256()
    for record in sorted(assignment_records):
        digest.update(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    report = {
        "format_version": 1,
        "algorithm": algorithm,
        "condition": condition,
        "sample_count": len(perturbed),
        "samples_with_hidden_fields": sum(bool(s) for s in selections),
        "total_selected_fields": sum(len(s) for s in selections),
        "preexisting_missing_selected": preexisting_missing_selected,
        "selected_assignments_sha256": digest.hexdigest(),
        "feature_selection_counts": dict(sorted(feature_selection_counts.items())),
        "training_data_perturbed": False,
    }
    report.update(extra or {})
    return tuple(perturbed), report


def _require_samples(samples) -> tuple[FlowSample, ...]:
    values = tuple(samples)
    if not values:
        raise ValueError("profile missingness requires non-empty samples")
    if any(not isinstance(sample, FlowSample) for sample in values):
        raise TypeError("profile missingness samples must be FlowSample values")
    return values


def apply_profile_missingness(
    samples: Sequence[FlowSample], *, profile: str
) -> tuple[tuple[FlowSample, ...], dict[str, object]]:
    """Hide one exporter or compound profile's fields in every sample."""

    fields = EXPORTER_PROFILES.get(profile) or COMPOUND_PROFILES.get(profile)
    if fields is None:
        raise ValueError(f"unknown exporter or compound profile: {profile}")
    values = _require_samples(samples)
    return _hide_fields(
        values,
        [fields] * len(values),
        condition=profile,
        algorithm=EXPORTER_MISSINGNESS_ALGORITHM,
        extra={"selected_feature_ids": list(fields)},
    )


def _intermittent_hit(seed: int, sample_id: str, fraction: float) -> bool:
    value = "\0".join((_INTERMITTENT_NAMESPACE, str(seed), sample_id))
    draw = int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
    return draw < int(fraction * 2**64)


def apply_intermittent_missingness(
    samples: Sequence[FlowSample], *, profile: str, seed: int
) -> tuple[tuple[FlowSample, ...], dict[str, object]]:
    """Hide one structured group in a deterministic per-record subset."""

    if type(seed) is not int or seed < 0:
        raise ValueError("intermittent seed must be a non-negative integer")
    spec = INTERMITTENT_PROFILES.get(profile)
    if spec is None:
        raise ValueError(f"unknown intermittent profile: {profile}")
    values = _require_samples(samples)
    fields = STRUCTURED_MISSINGNESS_GROUPS[spec["group"]]
    selections = [
        fields if _intermittent_hit(seed, sample.sample_id, spec["fraction"]) else ()
        for sample in values
    ]
    return _hide_fields(
        values,
        selections,
        condition=profile,
        algorithm=INTERMITTENT_MISSINGNESS_ALGORITHM,
        extra={
            "seed": seed,
            "group": spec["group"],
            "fraction": spec["fraction"],
            "selected_feature_ids": list(fields),
        },
    )


@dataclass(frozen=True)
class ConditionRule:
    """Match an observed missing-field set exactly or by its size."""

    name: str
    missing_fields: frozenset[str] | None = None
    missing_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("condition rule name must be non-empty")
        if (self.missing_fields is None) == (self.missing_count is None):
            raise ValueError("condition rule needs exactly one matching mode")
        if self.missing_fields is not None:
            unknown = set(self.missing_fields) - set(MODEL_FIELDS)
            if unknown:
                raise ValueError(f"rule {self.name} names non-model fields")
        if self.missing_count is not None and not (
            0 < self.missing_count < len(MODEL_FIELDS)
        ):
            raise ValueError(f"rule {self.name} has an invalid missing count")


class ConditionResolver:
    """Resolve observed masks: exact field sets first, then missing counts."""

    def __init__(self, rules: Iterable[ConditionRule]):
        self.rules = tuple(rules)
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("condition rule names must be unique")
        self._exact: dict[frozenset[str], str] = {}
        self._count: dict[int, str] = {}
        for rule in self.rules:
            if rule.missing_fields is not None:
                if rule.missing_fields in self._exact:
                    raise ValueError(
                        f"rules {self._exact[rule.missing_fields]} and "
                        f"{rule.name} share one missing-field set"
                    )
                self._exact[rule.missing_fields] = rule.name
            else:
                if rule.missing_count in self._count:
                    raise ValueError(
                        f"rules {self._count[rule.missing_count]} and "
                        f"{rule.name} share one missing count"
                    )
                self._count[rule.missing_count] = rule.name

    @property
    def condition_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.rules)

    def resolve(self, missing_fields: Iterable[str]) -> str | None:
        observed = frozenset(missing_fields) & frozenset(MODEL_FIELDS)
        name = self._exact.get(observed)
        if name is not None:
            return name
        return self._count.get(len(observed))

    def resolve_sample(self, sample: FlowSample) -> str | None:
        return self.resolve(
            name
            for name, missing in zip(sample.feature_ids, sample.missing_mask)
            if missing
        )

    def resolve_matrix(
        self, missing: np.ndarray, field_order: Sequence[str] = MODEL_FIELDS
    ) -> np.ndarray:
        """Resolve a boolean ``(records, fields)`` mask; unresolved rows are ``""``."""

        values = np.asarray(missing, dtype=bool)
        if values.ndim != 2 or values.shape[1] != len(field_order):
            raise ValueError("mask matrix must align with the field order")
        patterns, inverse = np.unique(values, axis=0, return_inverse=True)
        names = np.asarray(field_order)
        resolved = [self.resolve(names[row]) or "" for row in patterns]
        return np.asarray(resolved, dtype=object)[inverse.reshape(-1)]


BASE_RANDOM_FRACTIONS = MappingProxyType(
    {"missing_30pct": 0.3, "missing_50pct": 0.5}
)


def build_supplementary_resolver(
    registered_profiles: Sequence[str] = (),
) -> ConditionResolver:
    """Clean, two random rates, five structured groups, plus chosen profiles."""

    rules = [ConditionRule("clean", missing_fields=frozenset())]
    for name, fraction in BASE_RANDOM_FRACTIONS.items():
        # Same count rule as apply_deterministic_field_missingness.
        count = math.floor(fraction * len(MODEL_FIELDS))
        rules.append(ConditionRule(name, missing_count=count))
    for group, fields in STRUCTURED_MISSINGNESS_GROUPS.items():
        rules.append(
            ConditionRule(f"structured__{group}", missing_fields=frozenset(fields))
        )
    for profile in registered_profiles:
        if profile not in EXPORTER_PROFILES:
            raise ValueError(f"only exporter profiles can be registered: {profile}")
        rules.append(
            ConditionRule(
                f"profile__{profile}",
                missing_fields=frozenset(EXPORTER_PROFILES[profile]),
            )
        )
    return ConditionResolver(rules)


def profile_field_sets() -> Mapping[str, tuple[str, ...]]:
    """All declared fixed field sets, for protocol snapshots."""

    return MappingProxyType({**EXPORTER_PROFILES, **COMPOUND_PROFILES})


__all__ = [
    "BASE_RANDOM_FRACTIONS",
    "COMPOUND_PROFILES",
    "ConditionResolver",
    "ConditionRule",
    "EXPORTER_MISSINGNESS_ALGORITHM",
    "EXPORTER_PROFILES",
    "INTERMITTENT_MISSINGNESS_ALGORITHM",
    "INTERMITTENT_PROFILES",
    "MODEL_FIELDS",
    "apply_intermittent_missingness",
    "apply_profile_missingness",
    "build_supplementary_resolver",
    "profile_field_sets",
]
