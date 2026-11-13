"""Dataset-neutral leakage-safe indexing for registered NetFlow-v2 data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from flowfm.data.feature_keys import compute_feature_key
from flowfm.data.identity import make_sample_id, stable_hash
from flowfm.data.labels import DatasetLabelSpace


@dataclass(frozen=True)
class NetFlowV2IndexedRecord:
    sample_id: str
    feature_key: str
    group_id: str
    binary_label: str
    coarse_label: str
    native_label: str
    source_file: str
    source_row_id: str
    host_pair_hash: str
    five_tuple_hash: str

    def to_split_record(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "stratum": self.native_label,
        }


class _StableUnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self._parent.setdefault(value, value)
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        smaller, larger = sorted((first_root, second_root))
        self._parent[larger] = smaller


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _binary_label(value: Any) -> str:
    normalized = _required_text(str(value), "binary label").casefold()
    if normalized in {"0", "benign", "normal"}:
        return "benign"
    if normalized in {"1", "attack", "malicious"}:
        return "attack"
    raise ValueError(f"unsupported binary label: {value!r}")


def _privacy_hash(namespace: str, suffix: str, payload: Any) -> str:
    return stable_hash(payload, f"{namespace}.{suffix}")


def index_netflow_v2_rows(
    rows: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    schema,
    label_space: DatasetLabelSpace,
    conflict_feature_keys: set[str],
    source_file_id: str,
    sensitive_hash_namespace: str,
    leakage_group_namespace: str,
    coarse_label_map: Mapping[str, str] | None = None,
    record_type=NetFlowV2IndexedRecord,
):
    """Index rows without publishing raw addresses or crossing leakage groups."""

    if not isinstance(label_space, DatasetLabelSpace):
        raise TypeError("label_space must be a DatasetLabelSpace")
    if (
        label_space.dataset_id != schema.dataset_id
        or label_space.schema_version != schema.version
    ):
        raise ValueError("label space does not match the registered Schema")
    source_file = _required_text(source_file_id, "source_file_id")
    sensitive_namespace = _required_text(
        sensitive_hash_namespace,
        "sensitive_hash_namespace",
    )
    group_namespace = _required_text(
        leakage_group_namespace,
        "leakage_group_namespace",
    )
    if not isinstance(conflict_feature_keys, set):
        raise TypeError("conflict_feature_keys must be a set")
    mapping = label_space.native_to_coarse if coarse_label_map is None else coarse_label_map
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("coarse_label_map must be a non-empty mapping")
    normalized_coarse_map = {
        _required_text(native, "coarse label key"): _required_text(
            coarse,
            "coarse label value",
        )
        for native, coarse in mapping.items()
    }

    seen_source_refs = set()
    union_find = _StableUnionFind()
    pending = []
    for source_row_value, row in rows:
        source_row_id = _required_text(source_row_value, "source_row_id")
        source_ref = (source_file, source_row_id)
        if source_ref in seen_source_refs:
            raise ValueError(
                f"duplicate source reference: {source_file}:{source_row_id}"
            )
        seen_source_refs.add(source_ref)
        if not isinstance(row, Mapping):
            raise TypeError("NetFlow-v2 rows must be mappings")

        feature_key = compute_feature_key(row, schema)
        if feature_key in conflict_feature_keys:
            continue
        native_label = _required_text(row.get("Attack"), "native label")
        try:
            coarse_label = normalized_coarse_map[native_label]
        except KeyError as exc:
            raise ValueError(
                f"native label is missing from coarse label map: {native_label}"
            ) from exc
        if native_label not in label_space.native_labels:
            raise ValueError(f"unknown native label: {native_label}")
        if label_space.native_to_coarse[native_label] != coarse_label:
            raise ValueError("coarse label map does not match label space")
        binary_label = _binary_label(row.get("Label"))
        expected_binary = "benign" if coarse_label == "benign" else "attack"
        if binary_label != expected_binary:
            raise ValueError("binary/native label hierarchy is inconsistent")

        source = _required_text(row.get("IPV4_SRC_ADDR"), "IPV4_SRC_ADDR")
        destination = _required_text(
            row.get("IPV4_DST_ADDR"),
            "IPV4_DST_ADDR",
        )
        host_pair_hash = _privacy_hash(
            sensitive_namespace,
            "host-pair",
            sorted((source, destination)),
        )
        five_tuple_hash = _privacy_hash(
            sensitive_namespace,
            "five-tuple",
            {
                "src": source,
                "src_port": _required_text(
                    row.get("L4_SRC_PORT"),
                    "L4_SRC_PORT",
                ),
                "dst": destination,
                "dst_port": _required_text(
                    row.get("L4_DST_PORT"),
                    "L4_DST_PORT",
                ),
                "protocol": _required_text(row.get("PROTOCOL"), "PROTOCOL"),
            },
        )
        feature_node = f"feature:{feature_key}"
        tuple_node = f"five-tuple:{five_tuple_hash}"
        union_find.union(feature_node, tuple_node)
        pending.append(
            {
                "sample_id": make_sample_id(
                    {
                        "source_file": source_file,
                        "source_row_id": source_row_id,
                    },
                    schema.dataset_id,
                    schema.version,
                ),
                "feature_key": feature_key,
                "feature_node": feature_node,
                "binary_label": binary_label,
                "coarse_label": coarse_label,
                "native_label": native_label,
                "source_file": source_file,
                "source_row_id": source_row_id,
                "host_pair_hash": host_pair_hash,
                "five_tuple_hash": five_tuple_hash,
            }
        )

    indexed = []
    for values in pending:
        component_root = union_find.find(values.pop("feature_node"))
        indexed.append(
            record_type(
                group_id=stable_hash(component_root, group_namespace),
                **values,
            )
        )
    return tuple(sorted(indexed, key=lambda record: record.sample_id))


def select_netflow_v2_development_records(
    indexed_records,
    *,
    max_samples: int,
    seed: int,
    order_namespace: str,
    required_native_labels=None,
):
    """Select whole leakage groups with rare native labels processed first."""

    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive non-boolean integer")
    normalized_namespace = _required_text(order_namespace, "order_namespace")
    groups = {}
    native_label_counts = Counter()
    for record in indexed_records:
        groups.setdefault(record.group_id, []).append(record)
        native_label_counts[record.native_label] += 1

    for group_id, members in groups.items():
        if len(members) > max_samples:
            raise ValueError(
                f"leakage group {group_id} size {len(members)} exceeds "
                f"max_samples {max_samples}"
            )

    required_group_ids = set()
    if required_native_labels is not None:
        if isinstance(required_native_labels, (str, bytes)) or not isinstance(
            required_native_labels,
            (list, tuple),
        ):
            raise TypeError("required_native_labels must be a list or tuple")
        required = tuple(
            _required_text(value, "required native label")
            for value in required_native_labels
        )
        if len(set(required)) != len(required):
            raise ValueError("required_native_labels must not contain duplicates")
        label_bits = {label: 1 << index for index, label in enumerate(required)}
        full_mask = (1 << len(required)) - 1
        best_group_by_mask = {}
        for group_id, members in groups.items():
            mask = 0
            for record in members:
                mask |= label_bits.get(record.native_label, 0)
            if not mask:
                continue
            key = (
                len(members),
                stable_hash(
                    {"group_id": group_id, "seed": seed},
                    f"{normalized_namespace}.required-label-cover",
                ),
                group_id,
            )
            previous = best_group_by_mask.get(mask)
            if previous is None or key < previous[0]:
                best_group_by_mask[mask] = (key, group_id)
        available_mask = 0
        for mask in best_group_by_mask:
            available_mask |= mask
        if available_mask != full_mask:
            missing = [
                label
                for label, bit in label_bits.items()
                if not available_mask & bit
            ]
            raise ValueError(
                f"required native labels are absent from indexed records: {missing}"
            )

        candidates = sorted(
            (
                key[0],
                key[1],
                group_id,
                mask,
            )
            for mask, (key, group_id) in best_group_by_mask.items()
        )
        best_cover = {0: (0, ())}
        for group_size, _, group_id, group_mask in candidates:
            for covered_mask, (cost, group_ids) in tuple(best_cover.items()):
                combined_mask = covered_mask | group_mask
                if combined_mask == covered_mask:
                    continue
                candidate = (cost + group_size, tuple(sorted((*group_ids, group_id))))
                if candidate[0] > max_samples:
                    continue
                previous = best_cover.get(combined_mask)
                if previous is None or candidate < previous:
                    best_cover[combined_mask] = candidate
        if full_mask not in best_cover:
            raise ValueError(
                "required native labels cannot be covered within max_samples"
            )
        required_group_ids.update(best_cover[full_mask][1])

    ordered_group_ids = sorted(
        groups,
        key=lambda group_id: (
            min(
                native_label_counts[record.native_label]
                for record in groups[group_id]
            ),
            -len(groups[group_id]),
            stable_hash(
                {"group_id": group_id, "seed": seed},
                normalized_namespace,
            ),
            group_id,
        ),
    )
    selected = [
        record
        for group_id in sorted(required_group_ids)
        for record in groups[group_id]
    ]
    for group_id in ordered_group_ids:
        if group_id in required_group_ids:
            continue
        members = groups[group_id]
        if len(selected) + len(members) <= max_samples:
            selected.extend(members)
    return tuple(sorted(selected, key=lambda record: record.sample_id))


def _native_proportional_targets(native_label_counts, max_samples, required):
    total = sum(native_label_counts.values())
    if total <= max_samples:
        return dict(native_label_counts)
    targets = {}
    remainders = {}
    for label, count in native_label_counts.items():
        target, remainder = divmod(count * max_samples, total)
        targets[label] = target
        remainders[label] = remainder
    remaining = max_samples - sum(targets.values())
    for label in sorted(remainders, key=lambda value: (-remainders[value], value)):
        if remaining == 0:
            break
        targets[label] += 1
        remaining -= 1
    for label in required:
        if targets.get(label, 0) > 0:
            continue
        donors = [
            candidate
            for candidate, target in targets.items()
            if target > 1 and candidate not in required
        ]
        if not donors:
            donors = [
                candidate
                for candidate, target in targets.items()
                if target > 1 and candidate != label
            ]
        if not donors:
            raise ValueError(
                "required native labels cannot receive positive proportional targets"
            )
        donor = max(
            donors,
            key=lambda candidate: (
                targets[candidate],
                native_label_counts[candidate],
                candidate,
            ),
        )
        targets[donor] -= 1
        targets[label] = 1
    if sum(targets.values()) != max_samples:
        raise RuntimeError("native proportional targets do not sum to max_samples")
    return targets


def select_netflow_v2_proportional_development_records(
    indexed_records,
    *,
    max_samples: int,
    seed: int,
    order_namespace: str,
    required_native_labels=None,
):
    """Select whole groups while preserving native row proportions."""

    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive non-boolean integer")
    normalized_namespace = _required_text(order_namespace, "order_namespace")
    groups = {}
    native_label_counts = Counter()
    for record in indexed_records:
        groups.setdefault(record.group_id, []).append(record)
        native_label_counts[record.native_label] += 1
    for group_id, members in groups.items():
        if len(members) > max_samples:
            raise ValueError(
                f"leakage group {group_id} size {len(members)} exceeds "
                f"max_samples {max_samples}"
            )

    if required_native_labels is None:
        required = ()
    else:
        if isinstance(required_native_labels, (str, bytes)) or not isinstance(
            required_native_labels,
            (list, tuple),
        ):
            raise TypeError("required_native_labels must be a list or tuple")
        required = tuple(
            _required_text(value, "required native label")
            for value in required_native_labels
        )
        if len(set(required)) != len(required):
            raise ValueError("required_native_labels must not contain duplicates")
        missing = sorted(set(required) - set(native_label_counts))
        if missing:
            raise ValueError(
                f"required native labels are absent from indexed records: {missing}"
            )

    targets = _native_proportional_targets(
        native_label_counts,
        max_samples,
        required,
    )
    group_label_counts = {
        group_id: Counter(record.native_label for record in members)
        for group_id, members in groups.items()
    }

    required_group_ids = set()
    if required:
        label_bits = {label: 1 << index for index, label in enumerate(required)}
        full_mask = (1 << len(required)) - 1
        best_group_by_mask = {}
        for group_id, members in groups.items():
            mask = 0
            for record in members:
                mask |= label_bits.get(record.native_label, 0)
            if not mask:
                continue
            key = (
                len(members),
                stable_hash(
                    {"group_id": group_id, "seed": seed},
                    f"{normalized_namespace}.required-label-cover",
                ),
                group_id,
            )
            previous = best_group_by_mask.get(mask)
            if previous is None or key < previous[0]:
                best_group_by_mask[mask] = (key, group_id)
        candidates = sorted(
            (key[0], key[1], group_id, mask)
            for mask, (key, group_id) in best_group_by_mask.items()
        )
        best_cover = {0: (0, ())}
        for group_size, _, group_id, group_mask in candidates:
            for covered_mask, (cost, group_ids) in tuple(best_cover.items()):
                combined_mask = covered_mask | group_mask
                if combined_mask == covered_mask:
                    continue
                candidate = (
                    cost + group_size,
                    tuple(sorted((*group_ids, group_id))),
                )
                if candidate[0] > max_samples:
                    continue
                previous = best_cover.get(combined_mask)
                if previous is None or candidate < previous:
                    best_cover[combined_mask] = candidate
        if full_mask not in best_cover:
            raise ValueError(
                "required native labels cannot be covered within max_samples"
            )
        required_group_ids.update(best_cover[full_mask][1])

    selected_group_ids = set(required_group_ids)
    selected_counts = Counter()
    selected_size = 0
    for group_id in selected_group_ids:
        selected_counts.update(group_label_counts[group_id])
        selected_size += len(groups[group_id])

    ordered_group_ids = sorted(
        groups,
        key=lambda group_id: (
            stable_hash(
                {"group_id": group_id, "seed": seed},
                normalized_namespace,
            ),
            group_id,
        ),
    )
    for group_id in ordered_group_ids:
        if group_id in selected_group_ids:
            continue
        members = groups[group_id]
        if selected_size + len(members) > max_samples:
            continue
        counts = group_label_counts[group_id]
        if all(
            count <= max(targets.get(label, 0) - selected_counts[label], 0)
            for label, count in counts.items()
        ):
            selected_group_ids.add(group_id)
            selected_counts.update(counts)
            selected_size += len(members)

    for group_id in ordered_group_ids:
        if group_id in selected_group_ids:
            continue
        members = groups[group_id]
        if selected_size + len(members) > max_samples:
            continue
        counts = group_label_counts[group_id]
        before = sum(
            abs(targets.get(label, 0) - selected_counts[label])
            for label in native_label_counts
        )
        after = sum(
            abs(
                targets.get(label, 0)
                - selected_counts[label]
                - counts.get(label, 0)
            )
            for label in native_label_counts
        )
        if after < before:
            selected_group_ids.add(group_id)
            selected_counts.update(counts)
            selected_size += len(members)

    if selected_size < max_samples:
        for group_id in ordered_group_ids:
            if group_id in selected_group_ids:
                continue
            members = groups[group_id]
            if selected_size + len(members) <= max_samples:
                selected_group_ids.add(group_id)
                selected_counts.update(group_label_counts[group_id])
                selected_size += len(members)
            if selected_size == max_samples:
                break

    selected = [
        record
        for group_id in selected_group_ids
        for record in groups[group_id]
    ]
    return tuple(sorted(selected, key=lambda record: record.sample_id))


__all__ = [
    "NetFlowV2IndexedRecord",
    "index_netflow_v2_rows",
    "select_netflow_v2_development_records",
    "select_netflow_v2_proportional_development_records",
]
