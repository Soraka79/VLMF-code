"""Group-safe NF-CSE few-shot training views over a frozen split."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import NF_CSE_LABEL_SPACE
from flowfm.data.netflow_v2_protocol import (
    NetFlowV2IndexedRecord,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_cse_split import SPLIT_CONFIG_NAMESPACE
from flowfm.data.nf_unsw_split import (
    _git_commit,
    _hash_file,
    _label_counts,
    _load_json_object,
    _write_text,
)
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
    NF_CSE_CIC_IDS2018_V2_SCHEMA,
)
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


FEW_SHOT_FORMAT_VERSION = 1
FEW_SHOT_SOURCE_NAMESPACE = "flowfm-nf-cse-confirmatory-few-shot-source-v1"
FEW_SHOT_GROUP_ORDER_NAMESPACE = (
    "flowfm.nf-cse-confirmatory-few-shot-train-group-order.v1"
)
SELECTION_RECORDS_FILENAME = "selection_records.jsonl"
SELECTION_MANIFEST_FILENAME = "selection_manifest.json"
SELECTION_SUMMARY_FILENAME = "selection_summary.json"
_SPLIT_NAMES = ("train", "val", "test")


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _load_records(path: Path) -> tuple[NetFlowV2IndexedRecord, ...]:
    records = []
    seen_samples = set()
    seen_sources = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = NetFlowV2IndexedRecord(**json.loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid selected record JSON at line {line_number}"
                ) from exc
            source_ref = (record.source_file, record.source_row_id)
            if record.sample_id in seen_samples or source_ref in seen_sources:
                raise ValueError("selected records contain duplicate identities")
            seen_samples.add(record.sample_id)
            seen_sources.add(source_ref)
            records.append(record)
    return tuple(records)


def _validate_source(source_dir: Path):
    paths = {
        "config": source_dir / "config.snapshot.json",
        "data_manifest": source_dir / "data_manifest.json",
        "selected": source_dir / "selected_records.jsonl",
        "manifest": source_dir / "split_manifest.json",
        "summary": source_dir / "split_summary.json",
        "overlap": source_dir / "overlap_audit.json",
        "seal": source_dir / "sealed_test.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("source confirmatory split artifacts are incomplete")
    if (source_dir / ".incomplete").exists() or (
        source_dir / "REJECTED.json"
    ).exists():
        raise ValueError("source confirmatory split is incomplete or rejected")
    config = _load_json_object(paths["config"], "source config")
    summary = _load_json_object(paths["summary"], "source summary")
    overlap = _load_json_object(paths["overlap"], "source overlap audit")
    seal = _load_json_object(paths["seal"], "source sealed test")
    if stable_hash(config, SPLIT_CONFIG_NAMESPACE) != summary.get("config_hash"):
        raise ValueError("source split config hash does not match summary")
    expected = {
        "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_CSE_LABEL_SPACE.label_space_hash,
        "selected_records_sha256": _hash_file(paths["selected"], "sha256"),
        "split_manifest_sha256": _hash_file(paths["manifest"], "sha256"),
        "source_manifest_sha256": _hash_file(
            paths["data_manifest"], "sha256"
        ),
        "overlap_audit_sha256": _hash_file(paths["overlap"], "sha256"),
        "sealed_test_sha256": _hash_file(paths["seal"], "sha256"),
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"source split {field} does not match artifacts")
    if overlap.get("verified_zero_overlap") is not True:
        raise ValueError("source split does not prove zero historical overlap")
    if seal.get("status") != "sealed_before_model_selection":
        raise ValueError("source test is not sealed")
    manifest = SplitManifest.from_json(paths["manifest"].read_text(encoding="utf-8"))
    records = _load_records(paths["selected"])
    by_sample = {record.sample_id: record for record in records}
    assigned = {
        sample_id
        for split_name in _SPLIT_NAMES
        for sample_id in manifest.splits[split_name]
    }
    if assigned != set(by_sample):
        raise ValueError("source selected records do not match split manifest")
    return {
        "config": config,
        "overlap": overlap,
        "seal": seal,
        "summary": summary,
        "manifest": manifest,
        "records_by_sample": by_sample,
        "paths": paths,
    }


def _group_label_counts(records, field):
    groups = {}
    for record in records:
        groups.setdefault(getattr(record, field), set()).add(record.group_id)
    return {label: len(groups[label]) for label in sorted(groups)}


def publish_nf_cse_few_shot_split(
    source_split_dir,
    output_path,
    *,
    train_percent: int,
    seed: int,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Subsample whole training groups while preserving sealed val/test members."""

    if type(train_percent) is not int or not 1 <= train_percent <= 100:
        raise ValueError("train_percent must be an integer from 1 through 100")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    source_dir = validate_output_path(
        source_split_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not source_dir.is_dir():
        raise ValueError("source_split_dir must be an existing directory")
    if output_dir.exists():
        raise ValueError(f"few-shot output already exists: {output_dir}")
    source = _validate_source(source_dir)
    source_manifest = source["manifest"]
    records_by_sample = source["records_by_sample"]
    source_train = tuple(
        records_by_sample[sample_id]
        for sample_id in source_manifest.splits["train"]
    )
    source_native_labels = {record.native_label for record in source_train}
    source_coarse_labels = {record.coarse_label for record in source_train}
    required_coarse_labels = set(NF_CSE_LABEL_SPACE.native_to_coarse.values())
    missing_coarse = required_coarse_labels - source_coarse_labels
    if missing_coarse:
        raise ValueError(
            f"source train is missing coarse labels: {sorted(missing_coarse)}"
        )
    required_native_representatives = tuple(
        next(
            native
            for native in NF_CSE_LABEL_SPACE.native_labels
            if native in source_native_labels
            and NF_CSE_LABEL_SPACE.native_to_coarse[native] == coarse
        )
        for coarse in NF_CSE_LABEL_SPACE.coarse_labels
        if coarse in required_coarse_labels
    )
    target_rows = max(
        len(required_coarse_labels),
        len(source_train) * train_percent // 100,
    )
    selected_train = select_netflow_v2_proportional_development_records(
        source_train,
        max_samples=target_rows,
        seed=seed,
        order_namespace=FEW_SHOT_GROUP_ORDER_NAMESPACE,
        required_native_labels=required_native_representatives,
    )
    if len(selected_train) < max(1, int(target_rows * 0.99)):
        raise RuntimeError("few-shot group selection fell more than 1% below target")
    derived_splits = {
        "train": sorted(record.sample_id for record in selected_train),
        "val": list(source_manifest.splits["val"]),
        "test": list(source_manifest.splits["test"]),
    }
    derived_ids = {
        sample_id
        for split_name in _SPLIT_NAMES
        for sample_id in derived_splits[split_name]
    }
    selected = tuple(records_by_sample[value] for value in sorted(derived_ids))
    source_hash = stable_hash(
        {
            "source_config_hash": source["summary"]["config_hash"],
            "source_split_manifest_sha256": source["summary"][
                "split_manifest_sha256"
            ],
            "source_selected_records_sha256": source["summary"][
                "selected_records_sha256"
            ],
            "train_percent": train_percent,
            "target_train_rows": target_rows,
            "seed": seed,
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
        "sealed_test": True,
        "seed": seed,
        "selection_policy": "native_proportional_whole_train_groups",
        "source_overlap_audit_sha256": source["summary"][
            "overlap_audit_sha256"
        ],
        "source_zero_overlap_verified": source["overlap"][
            "verified_zero_overlap"
        ],
        "source_sealed_test_sha256": source["summary"]["sealed_test_sha256"],
        "source_sealed_test_status": source["seal"]["status"],
        "source_split_dir": str(source_dir),
        "source_selected_rows": source["summary"]["selected_rows"],
        "source_split_summary_sha256": _hash_file(
            source["paths"]["summary"], "sha256"
        ),
        "source_split_manifest_sha256": source["summary"][
            "split_manifest_sha256"
        ],
        "source_selected_records_sha256": source["summary"][
            "selected_records_sha256"
        ],
        "train_percent": train_percent,
    }
    config_hash = stable_hash(config_snapshot, SPLIT_CONFIG_NAMESPACE)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text("NF-CSE few-shot split incomplete\n", encoding="utf-8")
    write_json_snapshot(output_dir / "config.snapshot.json", config_snapshot)
    _write_bytes(
        output_dir / "data_manifest.json",
        source["paths"]["data_manifest"].read_bytes(),
    )
    selected_path = output_dir / "selected_records.jsonl"
    selection_records_path = output_dir / SELECTION_RECORDS_FILENAME
    manifest_path = output_dir / "split_manifest.json"
    selection_manifest_path = output_dir / SELECTION_MANIFEST_FILENAME
    _write_text(
        selected_path,
        "".join(canonical_json(asdict(record)) + "\n" for record in selected),
    )
    _write_text(manifest_path, manifest.to_json() + "\n")
    selection_ids = set(derived_splits["train"]) | set(derived_splits["val"])
    selection_manifest = SplitManifest(
        splits={
            "train": derived_splits["train"],
            "val": derived_splits["val"],
            "test": [],
        },
        ratios=manifest.ratios,
        seed=manifest.seed,
        schema_version=manifest.schema_version,
        source_hash=manifest.source_hash,
        sample_groups={
            sample_id: manifest.sample_groups[sample_id]
            for sample_id in selection_ids
        },
        group_assignments={
            group_id: split_name
            for group_id, split_name in manifest.group_assignments.items()
            if split_name in {"train", "val"}
        },
    )
    _write_text(
        selection_manifest_path,
        selection_manifest.to_json() + "\n",
    )
    selection_records = tuple(
        record for record in selected if record.sample_id in selection_ids
    )
    _write_text(
        selection_records_path,
        "".join(
            canonical_json(asdict(record)) + "\n"
            for record in selection_records
        ),
    )
    split_counts = {}
    for split_name in _SPLIT_NAMES:
        members = tuple(
            records_by_sample[sample_id]
            for sample_id in manifest.splits[split_name]
        )
        split_counts[split_name] = {
            "rows": len(members),
            "groups": len({record.group_id for record in members}),
            "binary_labels": _label_counts(members, "binary_label"),
            "coarse_labels": _label_counts(members, "coarse_label"),
            "coarse_label_groups": _group_label_counts(members, "coarse_label"),
            "native_labels": _label_counts(members, "native_label"),
        }
    train_native = Counter(record.native_label for record in selected_train)
    selection_summary = {
        "format_version": FEW_SHOT_FORMAT_VERSION,
        "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
        "purpose": "sealed_test_train_validation_selection_view",
        "selection_scope": "train_validation_only",
        "materialized_splits": ["train", "val"],
        "test_labels_present": False,
        "test_statistics_present": False,
        "selected_rows": len(selection_records),
        "seed": seed,
        "ratios": dict(manifest.ratios),
        "split_counts": {
            split_name: split_counts[split_name]
            for split_name in ("train", "val")
        },
        "config_hash": config_hash,
        "source_hash": source_hash,
        "source_selected_records_sha256": source["summary"][
            "selected_records_sha256"
        ],
        "source_split_summary_sha256": config_snapshot[
            "source_split_summary_sha256"
        ],
        "source_split_manifest_sha256": source["summary"][
            "split_manifest_sha256"
        ],
        "source_overlap_audit_sha256": source["summary"][
            "overlap_audit_sha256"
        ],
        "source_sealed_test_sha256": source["summary"][
            "sealed_test_sha256"
        ],
        "source_zero_overlap_verified": config_snapshot[
            "source_zero_overlap_verified"
        ],
        "source_sealed_test_status": config_snapshot[
            "source_sealed_test_status"
        ],
        "audit_config_hash": source["summary"]["audit_config_hash"],
        "source_manifest_sha256": source["summary"][
            "source_manifest_sha256"
        ],
        "raw_csv_sha1": source["summary"]["raw_csv_sha1"],
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_CSE_LABEL_SPACE.label_space_hash,
        "selected_records_sha256": _hash_file(
            selection_records_path, "sha256"
        ),
        "split_manifest_sha256": _hash_file(
            selection_manifest_path, "sha256"
        ),
        "sealed_split_manifest_sha256": _hash_file(
            manifest_path, "sha256"
        ),
        "sealed_selected_records_sha256": _hash_file(
            selected_path, "sha256"
        ),
    }
    selection_summary_path = write_json_snapshot(
        output_dir / SELECTION_SUMMARY_FILENAME,
        selection_summary,
    )
    summary = {
        "format_version": FEW_SHOT_FORMAT_VERSION,
        "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
        "derivation": "few_shot_train_only",
        "preserve_source_eval": True,
        "sealed_test": True,
        "train_percent": train_percent,
        "source_train_rows": len(source_train),
        "target_train_rows": target_rows,
        "selected_train_rows": len(selected_train),
        "selected_train_groups": len(
            {record.group_id for record in selected_train}
        ),
        "selected_rows": len(selected),
        "selected_groups": len({record.group_id for record in selected}),
        "seed": seed,
        "ratios": dict(manifest.ratios),
        "split_counts": split_counts,
        "native_labels": dict(sorted(train_native.items())),
        "required_coarse_labels": sorted(required_coarse_labels),
        "required_native_representatives": list(
            required_native_representatives
        ),
        "config_hash": config_hash,
        "source_split_config_hash": source["summary"]["config_hash"],
        "source_split_summary_sha256": config_snapshot[
            "source_split_summary_sha256"
        ],
        "source_split_manifest_sha256": source["summary"][
            "split_manifest_sha256"
        ],
        "source_selected_records_sha256": source["summary"][
            "selected_records_sha256"
        ],
        "source_overlap_audit_sha256": source["summary"][
            "overlap_audit_sha256"
        ],
        "source_sealed_test_sha256": source["summary"]["sealed_test_sha256"],
        "audit_config_hash": source["summary"]["audit_config_hash"],
        "source_manifest_sha256": source["summary"]["source_manifest_sha256"],
        "raw_csv_sha1": source["summary"]["raw_csv_sha1"],
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_CSE_LABEL_SPACE.label_space_hash,
        "source_hash": source_hash,
        "selected_records_sha256": _hash_file(selected_path, "sha256"),
        "selection_records_sha256": selection_summary[
            "selected_records_sha256"
        ],
        "selection_summary_sha256": _hash_file(
            selection_summary_path, "sha256"
        ),
        "selection_manifest_sha256": selection_summary[
            "split_manifest_sha256"
        ],
        "split_manifest_sha256": _hash_file(manifest_path, "sha256"),
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
    }
    write_json_snapshot(output_dir / "split_summary.json", summary)
    marker.unlink()
    return {
        "train_percent": train_percent,
        "target_train_rows": target_rows,
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
    "SELECTION_MANIFEST_FILENAME",
    "SELECTION_RECORDS_FILENAME",
    "SELECTION_SUMMARY_FILENAME",
    "publish_nf_cse_few_shot_split",
]
