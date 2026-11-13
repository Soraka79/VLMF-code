"""Strict, transactional NF-CSE-CIC-IDS2018-v2 development split publication."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flowfm.data.audit import load_audit_config
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import NF_CSE_LABEL_SPACE
from flowfm.data.netflow_v2_protocol import (
    index_netflow_v2_rows,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_unsw_split import (
    _git_commit,
    _hash_file,
    _label_counts,
    _load_json_object,
    _load_yaml,
    _require_text,
    _validated_ratios,
    _write_text,
)
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
from flowfm.data.splits import deterministic_stratified_group_split
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


LEAKAGE_GROUP_NAMESPACE = "flowfm-nf-cse-cic-ids2018-v2-leakage-group-v1"
DEVELOPMENT_GROUP_ORDER_NAMESPACE = (
    "flowfm.nf-cse-cic-ids2018-v2-development-group-order.v2-native-proportional"
)
SPLIT_CONFIG_NAMESPACE = "flowfm-nf-cse-cic-ids2018-v2-development-split-config-v2"
SPLIT_SOURCE_NAMESPACE = "flowfm-nf-cse-cic-ids2018-v2-development-split-source-v2"
SPLIT_FORMAT_VERSION = 1
_CONFIG_FIELDS = frozenset(
    {
        "audit_config",
        "audit_report_dir",
        "view",
        "max_samples",
        "seed",
        "selection_policy",
        "ratios",
    }
)
_SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_AUDIT_REPORTS = (
    "conflict_summary.json",
    "logical_views.json",
)


@dataclass(frozen=True)
class NFCSEDevelopmentSplitConfig:
    audit_config_path: Path
    audit_report_dir: Path
    view: str
    max_samples: int
    seed: int
    selection_policy: str
    ratios: dict[str, float]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_config": str(self.audit_config_path),
            "audit_report_dir": str(self.audit_report_dir),
            "max_samples": self.max_samples,
            "ratios": dict(self.ratios),
            "seed": self.seed,
            "selection_policy": self.selection_policy,
            "view": self.view,
        }


def load_nf_cse_development_split_config(path) -> NFCSEDevelopmentSplitConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"split config must be an existing file: {config_path}")
    payload = _load_yaml(config_path)
    fields = set(payload)
    if fields != _CONFIG_FIELDS:
        missing = sorted(_CONFIG_FIELDS - fields)
        extra = sorted(fields - _CONFIG_FIELDS)
        raise ValueError(
            f"invalid split config fields; missing={missing}, extra={extra}"
        )

    audit_config_path = Path(
        _require_text(payload["audit_config"], "audit_config")
    ).expanduser().resolve()
    if not audit_config_path.is_file():
        raise ValueError(
            f"audit_config must reference an existing file: {audit_config_path}"
        )
    audit_report_dir = Path(
        _require_text(payload["audit_report_dir"], "audit_report_dir")
    ).expanduser().resolve()
    if not audit_report_dir.is_dir():
        raise ValueError(
            "audit_report_dir must reference an existing directory: "
            f"{audit_report_dir}"
        )
    for filename in _REQUIRED_AUDIT_REPORTS:
        if not (audit_report_dir / filename).is_file():
            raise ValueError(
                f"audit_report_dir is missing required report: {filename}"
            )

    view = _require_text(payload["view"], "view")
    if view != "conflict_free_core":
        raise ValueError("view must be conflict_free_core for development publication")
    max_samples = payload["max_samples"]
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive non-boolean integer")
    if max_samples > 100_000:
        raise ValueError("max_samples must not exceed the development cap of 100000")
    seed = payload["seed"]
    if type(seed) is not int:
        raise ValueError("seed must be an integer and not a boolean")
    if seed != 42:
        raise ValueError("NF-CSE development seed must be 42")
    selection_policy = _require_text(
        payload["selection_policy"], "selection_policy"
    )
    if selection_policy != "native_proportional":
        raise ValueError("selection_policy must be native_proportional")
    ratios = _validated_ratios(payload["ratios"])
    if ratios != {"train": 0.6, "val": 0.2, "test": 0.2}:
        raise ValueError("NF-CSE development ratios must be 0.6/0.2/0.2")

    normalized = {
        "audit_config": str(audit_config_path),
        "audit_report_dir": str(audit_report_dir),
        "max_samples": max_samples,
        "ratios": ratios,
        "seed": seed,
        "selection_policy": selection_policy,
        "view": view,
    }
    return NFCSEDevelopmentSplitConfig(
        audit_config_path=audit_config_path,
        audit_report_dir=audit_report_dir,
        view=view,
        max_samples=max_samples,
        seed=seed,
        selection_policy=selection_policy,
        ratios=ratios,
        config_hash=stable_hash(normalized, SPLIT_CONFIG_NAMESPACE),
    )


def _validated_audit_state(config: NFCSEDevelopmentSplitConfig):
    audit_config = load_audit_config(config.audit_config_path)
    if audit_config.dataset_id != NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id:
        raise ValueError("audit config dataset_id is not NF-CSE-CIC-IDS2018-v2")
    if audit_config.schema_version != NF_CSE_CIC_IDS2018_V2_SCHEMA.version:
        raise ValueError("audit config does not use the registered NF-CSE Schema")
    if audit_config.schema.schema_hash != NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash:
        raise ValueError("audit config Schema hash does not match NF-CSE")
    if audit_config.coarse_label_map != dict(NF_CSE_LABEL_SPACE.native_to_coarse):
        raise ValueError("audit coarse label map does not match NF-CSE label space")

    summary_path = config.audit_report_dir / "conflict_summary.json"
    views_path = config.audit_report_dir / "logical_views.json"
    summary = _load_json_object(summary_path, "conflict summary")
    views = _load_json_object(views_path, "logical views")
    expected = {
        "dataset_id": audit_config.dataset_id,
        "schema_version": audit_config.schema_version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "config_hash": audit_config.config_hash,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"conflict summary {field} does not match audit config")
    if summary.get("raw_mutated") is not False:
        raise ValueError("conflict summary must prove the raw CSV was not mutated")
    raw_sha1 = _hash_file(audit_config.raw_csv, "sha1")
    if summary.get("raw_sha1_before") != raw_sha1 or summary.get(
        "raw_sha1_after"
    ) != raw_sha1:
        raise ValueError("raw CSV SHA1 does not match the conflict audit")
    source_manifest_sha256 = _hash_file(audit_config.source_manifest, "sha256")
    if summary.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("source manifest hash does not match the conflict audit")
    if views.get("dataset_id") != audit_config.dataset_id:
        raise ValueError("logical views dataset_id does not match audit config")
    if views.get("schema_version") != audit_config.schema_version:
        raise ValueError("logical views schema_version does not match audit config")
    try:
        conflict_keys = set(views["conflict_quarantine"]["feature_keys"])
        expected_view_rows = int(views[config.view]["rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("logical views do not contain valid conflict-free metadata") from exc
    if not all(isinstance(key, str) and key for key in conflict_keys):
        raise ValueError("logical views contain invalid conflict feature keys")
    return {
        "audit_config": audit_config,
        "audit_summary_sha256": _hash_file(summary_path, "sha256"),
        "conflict_feature_keys": conflict_keys,
        "expected_view_rows": expected_view_rows,
        "raw_csv_sha1": raw_sha1,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _raw_rows(audit_config):
    expected_header = (
        *NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_feature_names,
        audit_config.binary_label_column,
        audit_config.native_label_column,
    )
    with audit_config.raw_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header does not match the registered NF-CSE Schema")
        for row_number, row in enumerate(reader, start=1):
            yield str(row_number), row


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def publish_nf_cse_development_split(
    config_path,
    output_path,
    *,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Publish a deterministic conflict-free NF-CSE development split."""

    config = load_nf_cse_development_split_config(config_path)
    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if output_dir.exists():
        raise ValueError(f"split output directory already exists: {output_dir}")
    audit_state = _validated_audit_state(config)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text("NF-CSE split publication incomplete\n", encoding="utf-8")

    audit_config = audit_state["audit_config"]
    indexed = index_netflow_v2_rows(
        _raw_rows(audit_config),
        schema=NF_CSE_CIC_IDS2018_V2_SCHEMA,
        label_space=NF_CSE_LABEL_SPACE,
        conflict_feature_keys=audit_state["conflict_feature_keys"],
        source_file_id=audit_config.source_file_id,
        coarse_label_map=audit_config.coarse_label_map,
        sensitive_hash_namespace=audit_config.sensitive_hash_namespace,
        leakage_group_namespace=LEAKAGE_GROUP_NAMESPACE,
    )
    if len(indexed) != audit_state["expected_view_rows"]:
        raise RuntimeError(
            "indexed conflict-free row count does not match the audited logical view"
        )
    selected = select_netflow_v2_proportional_development_records(
        indexed,
        max_samples=config.max_samples,
        seed=config.seed,
        order_namespace=DEVELOPMENT_GROUP_ORDER_NAMESPACE,
        required_native_labels=NF_CSE_LABEL_SPACE.native_labels,
    )

    source_hash = stable_hash(
        {
            "audit_config_hash": audit_config.config_hash,
            "audit_summary_sha256": audit_state["audit_summary_sha256"],
            "raw_csv_sha1": audit_state["raw_csv_sha1"],
            "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
            "source_manifest_sha256": audit_state["source_manifest_sha256"],
            "view": config.view,
            "selection_policy": config.selection_policy,
        },
        SPLIT_SOURCE_NAMESPACE,
    )
    manifest = deterministic_stratified_group_split(
        (record.to_split_record() for record in selected),
        ratios=config.ratios,
        seed=config.seed,
        schema_version=NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        source_hash=source_hash,
    )

    config_snapshot = output_dir / "config.snapshot.json"
    data_manifest_path = output_dir / "data_manifest.json"
    selected_path = output_dir / "selected_records.jsonl"
    manifest_path = output_dir / "split_manifest.json"
    summary_path = output_dir / "split_summary.json"
    write_json_snapshot(config_snapshot, config.to_dict())
    _write_bytes(data_manifest_path, audit_config.source_manifest.read_bytes())
    _write_text(
        selected_path,
        "".join(canonical_json(asdict(record)) + "\n" for record in selected),
    )
    _write_text(manifest_path, manifest.to_json() + "\n")

    records_by_sample = {record.sample_id: record for record in selected}
    split_counts = {}
    for split_name in _SPLIT_NAMES:
        members = [
            records_by_sample[sample_id]
            for sample_id in manifest.splits[split_name]
        ]
        split_counts[split_name] = {
            "rows": len(members),
            "groups": len({record.group_id for record in members}),
            "binary_labels": _label_counts(members, "binary_label"),
            "coarse_labels": _label_counts(members, "coarse_label"),
            "native_labels": _label_counts(members, "native_label"),
        }
    summary = {
        "format_version": SPLIT_FORMAT_VERSION,
        "dataset_id": audit_config.dataset_id,
        "view": config.view,
        "selected_rows": len(selected),
        "selected_groups": len({record.group_id for record in selected}),
        "max_samples": config.max_samples,
        "seed": config.seed,
        "selection_policy": config.selection_policy,
        "ratios": dict(config.ratios),
        "split_counts": split_counts,
        "binary_labels": _label_counts(selected, "binary_label"),
        "coarse_labels": _label_counts(selected, "coarse_label"),
        "native_labels": _label_counts(selected, "native_label"),
        "config_hash": config.config_hash,
        "audit_config_hash": audit_config.config_hash,
        "audit_summary_sha256": audit_state["audit_summary_sha256"],
        "source_manifest_sha256": audit_state["source_manifest_sha256"],
        "raw_csv_sha1": audit_state["raw_csv_sha1"],
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_CSE_LABEL_SPACE.label_space_hash,
        "feature_key_namespace": audit_config.feature_key_namespace,
        "leakage_group_namespace": LEAKAGE_GROUP_NAMESPACE,
        "development_group_order_namespace": DEVELOPMENT_GROUP_ORDER_NAMESPACE,
        "source_hash": source_hash,
        "selected_records_sha256": _hash_file(selected_path, "sha256"),
        "split_manifest_sha256": _hash_file(manifest_path, "sha256"),
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
    }
    write_json_snapshot(summary_path, summary)
    marker.unlink()
    return {
        "config_hash": config.config_hash,
        "selected_rows": len(selected),
        "selected_groups": summary["selected_groups"],
        "selected_records_sha256": summary["selected_records_sha256"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
    }


__all__ = [
    "DEVELOPMENT_GROUP_ORDER_NAMESPACE",
    "LEAKAGE_GROUP_NAMESPACE",
    "NFCSEDevelopmentSplitConfig",
    "SPLIT_CONFIG_NAMESPACE",
    "SPLIT_FORMAT_VERSION",
    "SPLIT_SOURCE_NAMESPACE",
    "load_nf_cse_development_split_config",
    "publish_nf_cse_development_split",
]
