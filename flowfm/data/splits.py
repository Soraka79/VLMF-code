"""Deterministic, group-safe dataset split manifests."""

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping as TypingMapping, Sequence, Tuple

from .identity import canonical_json, stable_hash


_SPLIT_NAMES = ("train", "val", "test")
_MANIFEST_FIELDS = frozenset(
    {
        "splits",
        "ratios",
        "seed",
        "schema_version",
        "source_hash",
        "sample_groups",
        "group_assignments",
    }
)


def _validated_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    if not isinstance(ratios, Mapping) or set(ratios) != set(_SPLIT_NAMES):
        raise ValueError("ratios must contain exactly train, val, and test")
    result = {}
    for name in _SPLIT_NAMES:
        value = ratios[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("split ratios must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("split ratios must be finite and non-negative")
        result[name] = value
    if result["train"] <= 0:
        raise ValueError("the train ratio must be positive")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1")
    return result


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SplitManifest:
    """Serializable split membership plus group-leakage audit mappings."""

    splits: TypingMapping[str, Sequence[str]]
    ratios: TypingMapping[str, float]
    seed: int
    schema_version: str
    source_hash: str
    sample_groups: TypingMapping[str, str]
    group_assignments: TypingMapping[str, str]

    def __post_init__(self) -> None:
        ratios = _validated_ratios(self.ratios)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        _non_empty_string(self.schema_version, "schema_version")
        _non_empty_string(self.source_hash, "source_hash")
        if not isinstance(self.splits, Mapping) or set(self.splits) != set(_SPLIT_NAMES):
            raise ValueError("splits must contain exactly train, val, and test")
        if not isinstance(self.sample_groups, Mapping):
            raise ValueError("sample_groups must be a mapping")
        if not isinstance(self.group_assignments, Mapping):
            raise ValueError("group_assignments must be a mapping")

        normalized_splits = {}
        seen = set()
        for split_name in _SPLIT_NAMES:
            sample_ids = self.splits[split_name]
            if not isinstance(sample_ids, (list, tuple)):
                raise ValueError("split members must be lists of sample IDs")
            normalized_ids = []
            for sample_id in sample_ids:
                sample_id = _non_empty_string(sample_id, "sample_id")
                if sample_id in seen:
                    raise ValueError(f"sample_id occurs in multiple splits: {sample_id}")
                seen.add(sample_id)
                normalized_ids.append(sample_id)
            normalized_splits[split_name] = sorted(normalized_ids)

        sample_groups = {}
        for sample_id, group_id in self.sample_groups.items():
            sample_id = _non_empty_string(sample_id, "sample_id")
            group_id = _non_empty_string(group_id, "group_id")
            sample_groups[sample_id] = group_id
        if set(sample_groups) != seen:
            raise ValueError("sample_groups must describe every and only assigned sample")

        group_assignments = {}
        for group_id, split_name in self.group_assignments.items():
            group_id = _non_empty_string(group_id, "group_id")
            if split_name not in _SPLIT_NAMES:
                raise ValueError(f"invalid group split assignment: {split_name}")
            group_assignments[group_id] = split_name
        if set(group_assignments) != set(sample_groups.values()):
            raise ValueError("group_assignments must describe every and only assigned group")

        for split_name, sample_ids in normalized_splits.items():
            for sample_id in sample_ids:
                group_id = sample_groups[sample_id]
                if group_assignments[group_id] != split_name:
                    raise ValueError(f"group crosses splits: {group_id}")

        immutable_splits = {
            name: tuple(normalized_splits[name]) for name in _SPLIT_NAMES
        }
        object.__setattr__(self, "splits", MappingProxyType(immutable_splits))
        object.__setattr__(self, "ratios", MappingProxyType(dict(ratios)))
        object.__setattr__(
            self,
            "sample_groups",
            MappingProxyType(dict(sorted(sample_groups.items()))),
        )
        object.__setattr__(
            self,
            "group_assignments",
            MappingProxyType(dict(sorted(group_assignments.items()))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_assignments": dict(self.group_assignments),
            "ratios": dict(self.ratios),
            "sample_groups": dict(self.sample_groups),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_hash": self.source_hash,
            "splits": {name: list(self.splits[name]) for name in _SPLIT_NAMES},
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> "SplitManifest":
        def reject_duplicate_keys(pairs: Any) -> Dict[str, Any]:
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key: {key}")
                result[key] = item
            return result

        data = json.loads(value, object_pairs_hook=reject_duplicate_keys)
        if not isinstance(data, dict):
            raise ValueError("split manifest JSON must contain an object")
        fields = set(data)
        if fields != _MANIFEST_FIELDS:
            missing = sorted(_MANIFEST_FIELDS - fields)
            extra = sorted(fields - _MANIFEST_FIELDS)
            raise ValueError(
                f"invalid manifest top-level fields; missing={missing}, extra={extra}"
            )
        return cls(**data)


def deterministic_group_split(
    records: Any,
    ratios: Mapping[str, float],
    seed: int,
    schema_version: str,
    source_hash: str,
) -> SplitManifest:
    """Assign mapping records with ``sample_id`` and ``group_id`` by group.

    Groups are processed largest-first, with seed-derived ordering for equal
    sizes. Each group goes to the split that minimizes total squared deviation
    from target sample counts at that step. Because groups remain indivisible,
    an oversized group can make the requested ratios impossible to realize
    exactly.
    """

    checked_ratios = _validated_ratios(ratios)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _non_empty_string(schema_version, "schema_version")
    _non_empty_string(source_hash, "source_hash")

    sample_groups = {}
    record_content = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("split records must be mappings")
        if "sample_id" not in record or "group_id" not in record:
            raise ValueError("each record requires sample_id and group_id")
        sample_id = _non_empty_string(record["sample_id"], "sample_id")
        group_id = _non_empty_string(record["group_id"], "group_id")
        content = canonical_json(record)
        previous = sample_groups.get(sample_id)
        if previous is not None and previous != group_id:
            raise ValueError(f"sample_id has conflicting group_id values: {sample_id}")
        if sample_id in record_content and record_content[sample_id] != content:
            raise ValueError(f"sample_id has conflicting record content: {sample_id}")
        sample_groups[sample_id] = group_id
        record_content[sample_id] = content

    group_sizes = {}
    for group_id in sample_groups.values():
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1

    ordered_groups = sorted(
        group_sizes,
        key=lambda group_id: (
            -group_sizes[group_id],
            stable_hash(
                {"group_id": group_id, "seed": seed},
                namespace="flowfm.group-order.v1",
            ),
            group_id,
        ),
    )
    total_samples = len(sample_groups)
    targets = {
        name: checked_ratios[name] * total_samples for name in _SPLIT_NAMES
    }
    counts = dict.fromkeys(_SPLIT_NAMES, 0)
    group_assignments = {}
    for group_id in ordered_groups:
        group_size = group_sizes[group_id]

        def assignment_key(split_name: str) -> Tuple[float, float, int]:
            candidate_error = sum(
                (
                    counts[name]
                    + (group_size if name == split_name else 0)
                    - targets[name]
                )
                ** 2
                for name in _SPLIT_NAMES
            )
            remaining_deficit = targets[split_name] - counts[split_name]
            return (
                candidate_error,
                -remaining_deficit,
                _SPLIT_NAMES.index(split_name),
            )

        split_name = min(_SPLIT_NAMES, key=assignment_key)
        group_assignments[group_id] = split_name
        counts[split_name] += group_size

    splits = {name: [] for name in _SPLIT_NAMES}
    for sample_id, group_id in sorted(sample_groups.items()):
        splits[group_assignments[group_id]].append(sample_id)

    return SplitManifest(
        splits=splits,
        ratios=checked_ratios,
        seed=seed,
        schema_version=schema_version,
        source_hash=source_hash,
        sample_groups=sample_groups,
        group_assignments=group_assignments,
    )

def deterministic_stratified_group_split(
    records: Any,
    ratios: Mapping[str, float],
    seed: int,
    schema_version: str,
    source_hash: str,
) -> SplitManifest:
    """Assign groups while balancing total and per-stratum sample counts."""

    checked_ratios = _validated_ratios(ratios)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _non_empty_string(schema_version, "schema_version")
    _non_empty_string(source_hash, "source_hash")

    sample_groups = {}
    sample_strata = {}
    record_content = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("split records must be mappings")
        if "sample_id" not in record or "group_id" not in record:
            raise ValueError("each record requires sample_id and group_id")
        sample_id = _non_empty_string(record["sample_id"], "sample_id")
        group_id = _non_empty_string(record["group_id"], "group_id")
        try:
            stratum = record["stratum"]
        except KeyError as exc:
            raise ValueError("each stratified split record requires stratum") from exc
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("stratum must be a non-empty string")

        content = canonical_json(record)
        previous_group = sample_groups.get(sample_id)
        if previous_group is not None and previous_group != group_id:
            raise ValueError(f"sample_id has conflicting group_id values: {sample_id}")
        previous_stratum = sample_strata.get(sample_id)
        if previous_stratum is not None and previous_stratum != stratum:
            raise ValueError(f"sample_id has conflicting stratum values: {sample_id}")
        if sample_id in record_content and record_content[sample_id] != content:
            raise ValueError(f"sample_id has conflicting record content: {sample_id}")
        sample_groups[sample_id] = group_id
        sample_strata[sample_id] = stratum
        record_content[sample_id] = content

    group_members = {}
    group_strata = {}
    total_strata = Counter(sample_strata.values())
    for sample_id, group_id in sample_groups.items():
        group_members.setdefault(group_id, []).append(sample_id)
        group_strata.setdefault(group_id, Counter())[sample_strata[sample_id]] += 1

    ordered_groups = sorted(
        group_members,
        key=lambda group_id: (
            min(total_strata[stratum] for stratum in group_strata[group_id]),
            -len(group_members[group_id]),
            stable_hash(
                {"group_id": group_id, "seed": seed},
                namespace="flowfm.stratified-group-order.v1",
            ),
            group_id,
        ),
    )
    total_samples = len(sample_groups)
    size_targets = {
        split_name: checked_ratios[split_name] * total_samples
        for split_name in _SPLIT_NAMES
    }
    stratum_targets = {
        stratum: {
            split_name: checked_ratios[split_name] * count
            for split_name in _SPLIT_NAMES
        }
        for stratum, count in total_strata.items()
    }
    split_sizes = dict.fromkeys(_SPLIT_NAMES, 0)
    split_strata = {
        split_name: Counter() for split_name in _SPLIT_NAMES
    }
    group_assignments = {}

    for group_id in ordered_groups:
        group_size = len(group_members[group_id])
        histogram = group_strata[group_id]

        def assignment_key(split_name: str):
            candidate_sizes = dict(split_sizes)
            candidate_sizes[split_name] += group_size
            size_error = sum(
                (
                    candidate_sizes[name] - size_targets[name]
                )
                ** 2
                / max(total_samples, 1) ** 2
                for name in _SPLIT_NAMES
            )
            stratum_error = 0.0
            for stratum, total in total_strata.items():
                for name in _SPLIT_NAMES:
                    candidate = split_strata[name][stratum]
                    if name == split_name:
                        candidate += histogram[stratum]
                    stratum_error += (
                        candidate - stratum_targets[stratum][name]
                    ) ** 2 / max(total, 1) ** 2
            remaining_deficit = sum(
                (
                    stratum_targets[stratum][split_name]
                    - split_strata[split_name][stratum]
                )
                * count
                for stratum, count in histogram.items()
            )
            return (
                stratum_error + size_error,
                -remaining_deficit,
                _SPLIT_NAMES.index(split_name),
            )

        split_name = min(_SPLIT_NAMES, key=assignment_key)
        group_assignments[group_id] = split_name
        split_sizes[split_name] += group_size
        split_strata[split_name].update(histogram)

    splits = {name: [] for name in _SPLIT_NAMES}
    for sample_id, group_id in sorted(sample_groups.items()):
        splits[group_assignments[group_id]].append(sample_id)

    return SplitManifest(
        splits=splits,
        ratios=checked_ratios,
        seed=seed,
        schema_version=schema_version,
        source_hash=source_hash,
        sample_groups=sample_groups,
        group_assignments=group_assignments,
    )
