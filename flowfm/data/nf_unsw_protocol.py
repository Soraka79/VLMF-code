"""Leakage-safe indexing compatibility wrappers for NF-UNSW-NB15-v2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from flowfm.data.labels import NF_UNSW_LABEL_SPACE
from flowfm.data.netflow_v2_protocol import (
    index_netflow_v2_rows,
    select_netflow_v2_development_records,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


LEAKAGE_GROUP_NAMESPACE = "flowfm-nf-unsw-leakage-group-v1"
DEVELOPMENT_GROUP_ORDER_NAMESPACE = (
    "flowfm.nf-unsw-development-group-order.v1"
)


@dataclass(frozen=True)
class NFUNSWIndexedRecord:
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


def index_nf_unsw_rows(
    rows: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    conflict_feature_keys: set[str],
    source_file_id: str,
    coarse_label_map: Mapping[str, str],
    sensitive_hash_namespace: str,
) -> tuple[NFUNSWIndexedRecord, ...]:
    """Index rows using the frozen NF-UNSW identity and namespace contract."""

    return index_netflow_v2_rows(
        rows,
        schema=NF_UNSW_NB15_V2_SCHEMA,
        label_space=NF_UNSW_LABEL_SPACE,
        conflict_feature_keys=conflict_feature_keys,
        source_file_id=source_file_id,
        sensitive_hash_namespace=sensitive_hash_namespace,
        leakage_group_namespace=LEAKAGE_GROUP_NAMESPACE,
        coarse_label_map=coarse_label_map,
        record_type=NFUNSWIndexedRecord,
    )


def select_nf_unsw_development_records(
    indexed_records: Iterable[NFUNSWIndexedRecord],
    *,
    max_samples: int,
    seed: int,
) -> tuple[NFUNSWIndexedRecord, ...]:
    """Select whole groups using the frozen NF-UNSW ordering namespace."""

    return select_netflow_v2_development_records(
        indexed_records,
        max_samples=max_samples,
        seed=seed,
        order_namespace=DEVELOPMENT_GROUP_ORDER_NAMESPACE,
    )


__all__ = [
    "DEVELOPMENT_GROUP_ORDER_NAMESPACE",
    "LEAKAGE_GROUP_NAMESPACE",
    "NFUNSWIndexedRecord",
    "index_nf_unsw_rows",
    "select_nf_unsw_development_records",
]
