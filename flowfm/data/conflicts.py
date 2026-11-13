"""Deterministic, privacy-safe feature-label conflict aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping


LABEL_FIELDS = ("binary_label", "coarse_label", "native_label")
REQUIRED_FIELDS = (
    "sample_id",
    "feature_key",
    "binary_label",
    "coarse_label",
    "native_label",
    "source_file",
    "source_row_id",
)
VIEW_NAMES = ("raw_full", "conflict_free_core", "conflict_quarantine")


def _required_text(record: Mapping[str, Any], field: str) -> str:
    if field not in record:
        raise ValueError(f"record is missing required field: {field}")
    value = record[field]
    if value is None:
        raise ValueError(f"{field} must be a non-empty value")
    if isinstance(value, str):
        normalized = value.strip()
    elif field in LABEL_FIELDS or field == "source_row_id":
        normalized = str(value)
    else:
        raise ValueError(f"{field} must be a non-empty string")
    if not normalized:
        raise ValueError(f"{field} must be a non-empty value")
    return normalized


def _optional_text(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _upper_bound(groups: Iterable["ConflictGroup"], field: str) -> dict[str, Any]:
    group_list = tuple(groups)
    total_rows = sum(group.group_size for group in group_list)
    majority_correct = sum(
        max(group.label_counts[field].values()) for group in group_list
    )
    conflicting = tuple(
        group for group in group_list if field in group.conflict_fields
    )
    conflict_rows = sum(group.group_size for group in conflicting)
    conflict_correct = sum(
        max(group.label_counts[field].values()) for group in conflicting
    )
    return {
        "majority_correct_rows": majority_correct,
        "irreducible_error_rows": total_rows - majority_correct,
        "accuracy_upper_bound": (
            majority_correct / total_rows if total_rows else 0.0
        ),
        "conflict_only_majority_correct_rows": conflict_correct,
        "conflict_only_rows": conflict_rows,
        "conflict_only_accuracy_upper_bound": (
            conflict_correct / conflict_rows if conflict_rows else 1.0
        ),
    }


def _label_pair_counts(
    groups: Iterable["ConflictGroup"], field: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for group in groups:
        label_counts = group.label_counts[field]
        if len(label_counts) < 2:
            continue
        for first, second in combinations(sorted(label_counts), 2):
            key = f"{first} | {second}"
            entry = counts.setdefault(
                key,
                {"group_count": 0, "cross_label_row_pairs": 0},
            )
            entry["group_count"] += 1
            entry["cross_label_row_pairs"] += (
                label_counts[first] * label_counts[second]
            )
    return dict(sorted(counts.items()))


def _has_benign_attack_conflict(group: "ConflictGroup") -> bool:
    labels = {label.casefold() for label in group.label_counts["binary_label"]}
    benign = bool(labels.intersection({"0", "benign", "normal"}))
    attack = bool(labels.intersection({"1", "attack", "malicious"}))
    return benign and attack


def _has_attack_family_conflict(group: "ConflictGroup") -> bool:
    if "native_label" not in group.conflict_fields:
        return False
    binary_labels = {
        label.casefold() for label in group.label_counts["binary_label"]
    }
    return bool(binary_labels) and binary_labels.issubset(
        {"1", "attack", "malicious"}
    )


@dataclass(frozen=True)
class ConflictGroup:
    feature_key: str
    sample_ids: tuple[str, ...]
    source_refs: tuple[tuple[str, str], ...]
    group_size: int
    conflict_flag: bool
    conflict_fields: tuple[str, ...]
    conflict_label_count: int
    label_counts: dict[str, dict[str, int]]
    host_pair_hashes: tuple[str, ...]
    five_tuple_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_key": self.feature_key,
            "sample_ids": list(self.sample_ids),
            "source_refs": [
                {"source_file": source_file, "source_row_id": source_row_id}
                for source_file, source_row_id in self.source_refs
            ],
            "group_size": self.group_size,
            "conflict_flag": self.conflict_flag,
            "conflict_fields": list(self.conflict_fields),
            "conflict_label_count": self.conflict_label_count,
            "label_counts": {
                field: dict(counts)
                for field, counts in sorted(self.label_counts.items())
            },
            "host_pair_hashes": list(self.host_pair_hashes),
            "five_tuple_hashes": list(self.five_tuple_hashes),
            "deterministic_classifier_upper_bounds": {
                field: {
                    "majority_correct_rows": max(self.label_counts[field].values()),
                    "irreducible_error_rows": self.group_size
                    - max(self.label_counts[field].values()),
                    "accuracy_upper_bound": max(
                        self.label_counts[field].values()
                    )
                    / self.group_size,
                }
                for field in LABEL_FIELDS
            },
        }


@dataclass(frozen=True)
class ConflictAudit:
    groups: tuple[ConflictGroup, ...]
    summary: dict[str, Any]
    _views: dict[str, dict[str, tuple[str, ...]]]

    def group_by_key(self, feature_key: str) -> ConflictGroup:
        for group in self.groups:
            if group.feature_key == feature_key:
                return group
        raise KeyError(f"unknown feature_key: {feature_key}")

    def view_manifest(self, name: str) -> dict[str, tuple[str, ...]]:
        try:
            view = self._views[name]
        except KeyError as exc:
            raise KeyError(f"unknown conflict view: {name}") from exc
        return {
            "feature_keys": tuple(view["feature_keys"]),
            "sample_ids": tuple(view["sample_ids"]),
        }

    def split_records(self, view_name: str) -> tuple[dict[str, str], ...]:
        view = self.view_manifest(view_name)
        sample_groups = {
            sample_id: group.feature_key
            for group in self.groups
            for sample_id in group.sample_ids
        }
        return tuple(
            {"sample_id": sample_id, "group_id": sample_groups[sample_id]}
            for sample_id in view["sample_ids"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "groups": [group.to_dict() for group in self.groups],
            "views": {
                name: {
                    "feature_keys": list(self._views[name]["feature_keys"]),
                    "sample_ids": list(self._views[name]["sample_ids"]),
                }
                for name in VIEW_NAMES
            },
        }


def _build_group(feature_key: str, rows: list[dict[str, str | None]]) -> ConflictGroup:
    ordered = sorted(rows, key=lambda item: item["sample_id"] or "")
    label_counts = {
        field: _sorted_counts(str(item[field]) for item in ordered)
        for field in LABEL_FIELDS
    }
    conflict_fields = tuple(
        field for field in LABEL_FIELDS if len(label_counts[field]) > 1
    )
    source_refs = tuple(
        (str(item["source_file"]), str(item["source_row_id"]))
        for item in ordered
    )
    return ConflictGroup(
        feature_key=feature_key,
        sample_ids=tuple(str(item["sample_id"]) for item in ordered),
        source_refs=source_refs,
        group_size=len(ordered),
        conflict_flag=bool(conflict_fields),
        conflict_fields=conflict_fields,
        conflict_label_count=len(label_counts["native_label"]),
        label_counts=label_counts,
        host_pair_hashes=tuple(
            sorted(
                {
                    str(item["host_pair_hash"])
                    for item in ordered
                    if item["host_pair_hash"] is not None
                }
            )
        ),
        five_tuple_hashes=tuple(
            sorted(
                {
                    str(item["five_tuple_hash"])
                    for item in ordered
                    if item["five_tuple_hash"] is not None
                }
            )
        ),
    )


def _build_views(groups: tuple[ConflictGroup, ...]):
    selections = {
        "raw_full": groups,
        "conflict_free_core": tuple(
            group for group in groups if not group.conflict_flag
        ),
        "conflict_quarantine": tuple(
            group for group in groups if group.conflict_flag
        ),
    }
    return {
        name: {
            "feature_keys": tuple(group.feature_key for group in selected),
            "sample_ids": tuple(
                sorted(
                    sample_id
                    for group in selected
                    for sample_id in group.sample_ids
                )
            ),
        }
        for name, selected in selections.items()
    }


def analyze_conflicts(records: Iterable[Mapping[str, Any]]) -> ConflictAudit:
    """Aggregate model-visible feature collisions without copying raw features."""

    grouped: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    sample_ids = set()
    source_refs = set()
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise TypeError("conflict records must be mappings")
        normalized = {
            field: _required_text(raw_record, field) for field in REQUIRED_FIELDS
        }
        sample_id = str(normalized["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        source_ref = (
            str(normalized["source_file"]),
            str(normalized["source_row_id"]),
        )
        if source_ref in source_refs:
            raise ValueError(
                "duplicate source reference: {}:{}".format(*source_ref)
            )
        source_refs.add(source_ref)
        normalized["host_pair_hash"] = _optional_text(
            raw_record, "host_pair_hash"
        )
        normalized["five_tuple_hash"] = _optional_text(
            raw_record, "five_tuple_hash"
        )
        grouped[str(normalized["feature_key"])].append(normalized)

    groups = tuple(
        _build_group(feature_key, grouped[feature_key])
        for feature_key in sorted(grouped)
    )
    total_rows = sum(group.group_size for group in groups)
    conflict_groups = tuple(group for group in groups if group.conflict_flag)
    conflict_rows = sum(group.group_size for group in conflict_groups)
    size_distribution = _sorted_counts(str(group.group_size) for group in groups)
    summary = {
        "total_rows": total_rows,
        "unique_feature_keys": len(groups),
        "duplicate_excess_rows": total_rows - len(groups),
        "conflict_feature_keys": len(conflict_groups),
        "conflict_rows": conflict_rows,
        "conflict_row_ratio": conflict_rows / total_rows if total_rows else 0.0,
        "group_size_distribution": size_distribution,
        "label_pair_counts": {
            field: _label_pair_counts(groups, field) for field in LABEL_FIELDS
        },
        "benign_attack_conflict_groups": sum(
            _has_benign_attack_conflict(group) for group in groups
        ),
        "attack_family_conflict_groups": sum(
            _has_attack_family_conflict(group) for group in groups
        ),
        "deterministic_classifier_upper_bounds": {
            field: _upper_bound(groups, field) for field in LABEL_FIELDS
        },
    }
    return ConflictAudit(
        groups=groups,
        summary=summary,
        _views=_build_views(groups),
    )


__all__ = [
    "ConflictAudit",
    "ConflictGroup",
    "LABEL_FIELDS",
    "analyze_conflicts",
]
