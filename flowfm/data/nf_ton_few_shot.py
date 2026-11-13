"""Derive group-safe NF-ToN few-shot training splits from a frozen split."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.netflow_v2_protocol import (
    NetFlowV2IndexedRecord,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_ton_split import (
    SPLIT_CONFIG_NAMESPACE,
    _git_commit,
    _hash_file,
    _label_counts,
    _load_json_object,
    _write_bytes,
    _write_text,
)
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


FEW_SHOT_FORMAT_VERSION = 1
FEW_SHOT_SOURCE_NAMESPACE = "flowfm-nf-ton-iot-v2-few-shot-source-v1"
FEW_SHOT_GROUP_ORDER_NAMESPACE = (
    "flowfm.nf-ton-iot-v2-few-shot-train-group-order.v1"
)
_SPLIT_NAMES = ("train", "val", "test")


def _load_records(path: Path) -> tuple[NetFlowV2IndexedRecord, ...]:
    records = []
    seen_sample_ids = set()
    seen_source_refs = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                payload = json.loads(line)
                record = NetFlowV2IndexedRecord(**payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid selected record JSON at line {line_number}"
                ) from exc
            if record.sample_id in seen_sample_ids:
                raise ValueError(
                    f"duplicate selected sample_id: {record.sample_id}"
                )
            source_ref = (record.source_file, record.source_row_id)
            if source_ref in seen_source_refs:
                raise ValueError(
                    "duplicate selected source reference: "
                    f"{record.source_file}:{record.source_row_id}"
                )
            seen_sample_ids.add(record.sample_id)
            seen_source_refs.add(source_ref)
            records.append(record)
    return tuple(records)


def _validate_source_split(source_dir: Path) -> dict[str, Any]:
    required = {
        "config": source_dir / "config.snapshot.json",
        "data_manifest": source_dir / "data_manifest.json",
        "selected": source_dir / "selected_records.jsonl",
        "manifest": source_dir / "split_manifest.json",
        "summary": source_dir / "split_summary.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise ValueError(
                f"source split is missing required {label} file: {path.name}"
            )
    if (source_dir / "REJECTED.json").exists():
        raise ValueError("source split is explicitly rejected")
    if (source_dir / ".incomplete").exists():
        raise ValueError("source split is incomplete")

    config = _load_json_object(required["config"], "source split config")
    summary = _load_json_object(required["summary"], "source split summary")
    if stable_hash(config, SPLIT_CONFIG_NAMESPACE) != summary.get("config_hash"):
        raise ValueError("source split config hash does not match summary")
    expected_identity = {
        "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
        "schema_version": NF_TON_IOT_V2_SCHEMA.version,
        "schema_hash": NF_TON_IOT_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_TON_LABEL_SPACE.label_space_hash,
    }
    for field, expected in expected_identity.items():
        if summary.get(field) != expected:
            raise ValueError(f"source split {field} does not match implementation")

    selected_sha256 = _hash_file(required["selected"], "sha256")
    manifest_sha256 = _hash_file(required["manifest"], "sha256")
    data_manifest_sha256 = _hash_file(required["data_manifest"], "sha256")
    expected_hashes = {
        "selected_records_sha256": selected_sha256,
        "split_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": data_manifest_sha256,
    }
    for field, expected in expected_hashes.items():
        if summary.get(field) != expected:
            raise ValueError(f"source split {field} does not match artifacts")

    manifest = SplitManifest.from_json(
        required["manifest"].read_text(encoding="utf-8")
    )
    if manifest.source_hash != summary.get("source_hash"):
        raise ValueError("source split manifest source_hash does not match summary")
    if manifest.seed != summary.get("seed"):
        raise ValueError("source split manifest seed does not match summary")
    if dict(manifest.ratios) != summary.get("ratios"):
        raise ValueError("source split manifest ratios do not match summary")

    records = _load_records(required["selected"])
    records_by_sample = {record.sample_id: record for record in records}
    assigned_ids = {
        sample_id
        for split_name in _SPLIT_NAMES
        for sample_id in manifest.splits[split_name]
    }
    if assigned_ids != set(records_by_sample):
        raise ValueError(
            "source selected records and split manifest sample IDs do not match"
        )
    for sample_id, group_id in manifest.sample_groups.items():
        if records_by_sample[sample_id].group_id != group_id:
            raise ValueError("source selected record group_id does not match manifest")

    return {
        "config": config,
        "summary": summary,
        "manifest": manifest,
        "records_by_sample": records_by_sample,
        "paths": required,
        "selected_sha256": selected_sha256,
        "manifest_sha256": manifest_sha256,
        "data_manifest_sha256": data_manifest_sha256,
    }


def publish_nf_ton_few_shot_split(
    source_split_dir,
    output_path,
    *,
    train_percent: int,
    seed: int = 42,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Keep source validation/test membership and subsample whole train groups."""

    if type(train_percent) is not int or not 1 <= train_percent <= 100:
        raise ValueError("train_percent must be an integer from 1 through 100")
    if type(seed) is not int:
        raise ValueError("seed must be an integer and not a boolean")

    source_dir = validate_output_path(
        source_split_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not source_dir.is_dir():
        raise ValueError(
            f"source_split_dir must reference an existing directory: {source_dir}"
        )
    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if output_dir.exists():
        raise ValueError(f"few-shot output directory already exists: {output_dir}")

    source = _validate_source_split(source_dir)
    source_manifest = source["manifest"]
    records_by_sample = source["records_by_sample"]
    source_train = tuple(
        records_by_sample[sample_id]
        for sample_id in source_manifest.splits["train"]
    )
    source_labels = {record.native_label for record in source_train}
    missing_labels = sorted(set(NF_TON_LABEL_SPACE.native_labels) - source_labels)
    if missing_labels:
        raise ValueError(
            f"source train split is missing required native labels: {missing_labels}"
        )
    target_train_rows = max(
        len(NF_TON_LABEL_SPACE.native_labels),
        len(source_train) * train_percent // 100,
    )
    selected_train = select_netflow_v2_proportional_development_records(
        source_train,
        max_samples=target_train_rows,
        seed=seed,
        order_namespace=FEW_SHOT_GROUP_ORDER_NAMESPACE,
        required_native_labels=NF_TON_LABEL_SPACE.native_labels,
    )
    selected_train_ids = {record.sample_id for record in selected_train}
    derived_splits = {
        "train": sorted(selected_train_ids),
        "val": list(source_manifest.splits["val"]),
        "test": list(source_manifest.splits["test"]),
    }
    derived_ids = {
        sample_id
        for split_name in _SPLIT_NAMES
        for sample_id in derived_splits[split_name]
    }
    selected = tuple(
        records_by_sample[sample_id] for sample_id in sorted(derived_ids)
    )

    source_hash = stable_hash(
        {
            "source_split_config_hash": source["summary"]["config_hash"],
            "source_split_manifest_sha256": source["manifest_sha256"],
            "source_selected_records_sha256": source["selected_sha256"],
            "train_percent": train_percent,
            "target_train_rows": target_train_rows,
            "seed": seed,
            "selection_policy": "native_proportional_whole_groups",
        },
        FEW_SHOT_SOURCE_NAMESPACE,
    )
    manifest = SplitManifest(
        splits=derived_splits,
        ratios=source_manifest.ratios,
        seed=seed,
        schema_version=source_manifest.schema_version,
        source_hash=source_hash,
        sample_groups={
            sample_id: source_manifest.sample_groups[sample_id]
            for sample_id in derived_ids
        },
        group_assignments={
            source_manifest.sample_groups[sample_id]: split_name
            for split_name in _SPLIT_NAMES
            for sample_id in derived_splits[split_name]
        },
    )

    config_snapshot = {
        "audit_config": source["config"]["audit_config"],
        "preserve_source_eval": True,
        "seed": seed,
        "selection_policy": "native_proportional_whole_groups",
        "source_split_dir": str(source_dir),
        "source_split_manifest_sha256": source["manifest_sha256"],
        "source_selected_records_sha256": source["selected_sha256"],
        "train_percent": train_percent,
    }
    config_hash = stable_hash(config_snapshot, SPLIT_CONFIG_NAMESPACE)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text(
        "NF-ToN few-shot split publication incomplete\n",
        encoding="utf-8",
    )
    write_json_snapshot(output_dir / "config.snapshot.json", config_snapshot)
    _write_bytes(
        output_dir / "data_manifest.json",
        source["paths"]["data_manifest"].read_bytes(),
    )
    selected_path = output_dir / "selected_records.jsonl"
    manifest_path = output_dir / "split_manifest.json"
    _write_text(
        selected_path,
        "".join(canonical_json(asdict(record)) + "\n" for record in selected),
    )
    _write_text(manifest_path, manifest.to_json() + "\n")

    split_counts = {}
    for split_name in _SPLIT_NAMES:
        members = [
            records_by_sample[value] for value in manifest.splits[split_name]
        ]
        split_counts[split_name] = {
            "rows": len(members),
            "groups": len({record.group_id for record in members}),
            "binary_labels": _label_counts(members, "binary_label"),
            "coarse_labels": _label_counts(members, "coarse_label"),
            "native_labels": _label_counts(members, "native_label"),
        }
    selected_train_counts = Counter(record.native_label for record in selected_train)
    summary = {
        "format_version": FEW_SHOT_FORMAT_VERSION,
        "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
        "derivation": "few_shot_train_only",
        "preserve_source_eval": True,
        "train_percent": train_percent,
        "source_train_rows": len(source_train),
        "target_train_rows": target_train_rows,
        "selected_train_rows": len(selected_train),
        "selected_train_groups": len({record.group_id for record in selected_train}),
        "selected_rows": len(selected),
        "selected_groups": len({record.group_id for record in selected}),
        "seed": seed,
        "ratios": dict(manifest.ratios),
        "split_counts": split_counts,
        "native_labels": dict(sorted(selected_train_counts.items())),
        "config_hash": config_hash,
        "source_split_config_hash": source["summary"]["config_hash"],
        "source_split_manifest_sha256": source["manifest_sha256"],
        "source_selected_records_sha256": source["selected_sha256"],
        "audit_config_hash": source["summary"]["audit_config_hash"],
        "source_manifest_sha256": source["data_manifest_sha256"],
        "raw_csv_sha1": source["summary"]["raw_csv_sha1"],
        "schema_version": NF_TON_IOT_V2_SCHEMA.version,
        "schema_hash": NF_TON_IOT_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_TON_LABEL_SPACE.label_space_hash,
        "source_hash": source_hash,
        "selected_records_sha256": _hash_file(selected_path, "sha256"),
        "split_manifest_sha256": _hash_file(manifest_path, "sha256"),
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
    }
    write_json_snapshot(output_dir / "split_summary.json", summary)
    marker.unlink()
    return {
        "train_percent": train_percent,
        "target_train_rows": target_train_rows,
        "selected_train_rows": len(selected_train),
        "selected_train_groups": summary["selected_train_groups"],
        "validation_rows": len(manifest.splits["val"]),
        "test_rows": len(manifest.splits["test"]),
        "selected_records_sha256": summary["selected_records_sha256"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
    }


__all__ = [
    "FEW_SHOT_FORMAT_VERSION",
    "FEW_SHOT_GROUP_ORDER_NAMESPACE",
    "FEW_SHOT_SOURCE_NAMESPACE",
    "publish_nf_ton_few_shot_split",
]
