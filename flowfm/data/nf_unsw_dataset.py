"""Validated NF-UNSW development samples from frozen split artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flowfm.data.audit import load_audit_config
from flowfm.data.contracts import FlowSample
from flowfm.data.feature_keys import compute_feature_key
from flowfm.data.identity import stable_hash
from flowfm.data.nf_unsw_preprocessing import TRAINING_SAMPLE_IDS_NAMESPACE
from flowfm.data.nf_unsw_preprocessing import load_nf_unsw_preprocessor
from flowfm.data.nf_unsw_split import SPLIT_CONFIG_NAMESPACE
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import validate_output_path


_SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_SELECTED_FIELDS = (
    "sample_id",
    "feature_key",
    "group_id",
    "binary_label",
    "coarse_label",
    "native_label",
    "source_file",
    "source_row_id",
)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _load_selected_records(path: Path) -> tuple[dict[str, str], ...]:
    records = []
    seen_sample_ids = set()
    seen_source_refs = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid selected record JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("selected records must contain JSON objects")
            normalized = {}
            for field in _REQUIRED_SELECTED_FIELDS:
                value = record.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"selected record {field} must be a non-empty string"
                    )
                normalized[field] = value
            if normalized["sample_id"] in seen_sample_ids:
                raise ValueError(
                    f"duplicate selected sample_id: {normalized['sample_id']}"
                )
            source_ref = (normalized["source_file"], normalized["source_row_id"])
            if source_ref in seen_source_refs:
                raise ValueError(
                    "duplicate selected source reference: {}:{}".format(*source_ref)
                )
            seen_sample_ids.add(normalized["sample_id"])
            seen_source_refs.add(source_ref)
            records.append(normalized)
    return tuple(records)


def _normalize_binary_label(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if normalized in {"0", "benign", "normal"}:
        return "benign"
    if normalized in {"1", "attack", "malicious"}:
        return "attack"
    raise ValueError(f"unsupported binary label: {value!r}")


@dataclass(frozen=True)
class NFUNSWDevelopmentSamples:
    splits: Mapping[str, tuple[FlowSample, ...]]
    preprocessor_hash: str
    split_manifest_sha256: str
    raw_csv_sha1: str
    schema_version: str
    training_sample_ids_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.splits, Mapping) or set(self.splits) != set(_SPLIT_NAMES):
            raise ValueError("splits must contain exactly train, val, and test")
        normalized = {}
        seen = set()
        for split_name in _SPLIT_NAMES:
            samples = self.splits[split_name]
            if not isinstance(samples, (list, tuple)):
                raise ValueError("dataset split samples must be sequences")
            normalized_samples = tuple(samples)
            for sample in normalized_samples:
                if not isinstance(sample, FlowSample):
                    raise TypeError("dataset splits must contain FlowSample values")
                if sample.sample_id in seen:
                    raise ValueError(
                        f"sample_id occurs in multiple dataset splits: {sample.sample_id}"
                    )
                seen.add(sample.sample_id)
            normalized[split_name] = normalized_samples
        for field_name in (
            "preprocessor_hash",
            "split_manifest_sha256",
            "raw_csv_sha1",
            "schema_version",
            "training_sample_ids_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "splits", MappingProxyType(normalized))

    @property
    def sample_count(self) -> int:
        return sum(len(samples) for samples in self.splits.values())


def load_nf_unsw_development_samples(
    split_dir,
    preprocessor_dir,
    *,
    data_root=Path("/root/autodl-tmp"),
) -> NFUNSWDevelopmentSamples:
    """Load privacy-safe FlowSamples after validating all frozen artifacts."""

    split_root = validate_output_path(
        split_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    preprocessor_root = validate_output_path(
        preprocessor_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not split_root.is_dir():
        raise ValueError(f"split_dir must reference an existing directory: {split_root}")
    if not preprocessor_root.is_dir():
        raise ValueError(
            f"preprocessor_dir must reference an existing directory: {preprocessor_root}"
        )

    split_files = {
        "config": split_root / "config.snapshot.json",
        "selected": split_root / "selected_records.jsonl",
        "manifest": split_root / "split_manifest.json",
        "summary": split_root / "split_summary.json",
    }
    preprocessor_files = {
        "config": preprocessor_root / "config.snapshot.json",
        "preprocessor": preprocessor_root / "preprocessor.json",
        "summary": preprocessor_root / "preprocessor_summary.json",
    }
    for prefix, files in (
        ("split", split_files),
        ("preprocessor", preprocessor_files),
    ):
        for label, path in files.items():
            if not path.is_file():
                raise ValueError(
                    f"published artifacts are missing required {prefix} {label}: {path}"
                )
    for root in (split_root, preprocessor_root):
        if (root / ".incomplete").exists():
            raise ValueError(f"published artifact directory is incomplete: {root}")

    split_config = _load_json_object(split_files["config"], "split config snapshot")
    split_summary = _load_json_object(split_files["summary"], "split summary")
    preprocessor_config = _load_json_object(
        preprocessor_files["config"], "preprocessor config snapshot"
    )
    preprocessor_summary = _load_json_object(
        preprocessor_files["summary"], "preprocessor summary"
    )
    if stable_hash(split_config, SPLIT_CONFIG_NAMESPACE) != split_summary.get(
        "config_hash"
    ):
        raise ValueError("split config hash does not match split summary")
    if preprocessor_config.get("fit_scope") != "train_only":
        raise ValueError("preprocessor config fit_scope must be train_only")
    try:
        configured_split_root = Path(
            preprocessor_config["source_split_dir"]
        ).expanduser().resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError("preprocessor config is missing source_split_dir") from exc
    if configured_split_root != split_root:
        raise ValueError("preprocessor config references a different split directory")

    selected_sha256 = _hash_file(split_files["selected"])
    manifest_sha256 = _hash_file(split_files["manifest"])
    for summary_name, summary in (
        ("split summary", split_summary),
        ("preprocessor summary", preprocessor_summary),
    ):
        if summary.get("selected_records_sha256") != selected_sha256:
            raise ValueError(
                f"{summary_name} selected record hash does not match artifacts"
            )
        if summary.get("split_manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"{summary_name} split manifest hash does not match artifacts"
            )

    expected_identity = {
        "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
        "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
        "schema_hash": NF_UNSW_NB15_V2_SCHEMA.schema_hash,
    }
    for summary_name, summary in (
        ("split summary", split_summary),
        ("preprocessor summary", preprocessor_summary),
    ):
        for field, expected in expected_identity.items():
            if summary.get(field) != expected:
                raise ValueError(f"{summary_name} {field} does not match implementation")
    if preprocessor_summary.get("fit_scope") != "train_only":
        raise ValueError("preprocessor summary fit_scope must be train_only")
    if preprocessor_summary.get("split_config_hash") != split_summary.get(
        "config_hash"
    ):
        raise ValueError("preprocessor split config hash does not match split summary")

    manifest = SplitManifest.from_json(
        split_files["manifest"].read_text(encoding="utf-8")
    )
    if manifest.schema_version != NF_UNSW_NB15_V2_SCHEMA.version:
        raise ValueError("split manifest schema_version does not match implementation")
    if manifest.source_hash != split_summary.get("source_hash"):
        raise ValueError("split manifest source_hash does not match split summary")
    if manifest.seed != split_summary.get("seed"):
        raise ValueError("split manifest seed does not match split summary")
    if dict(manifest.ratios) != split_summary.get("ratios"):
        raise ValueError("split manifest ratios do not match split summary")

    selected_records = _load_selected_records(split_files["selected"])
    records_by_sample = {
        record["sample_id"]: record for record in selected_records
    }
    assigned_ids = {
        sample_id
        for sample_ids in manifest.splits.values()
        for sample_id in sample_ids
    }
    if assigned_ids != set(records_by_sample):
        raise ValueError("selected records and split manifest sample IDs do not match")

    preprocessor_sha256 = _hash_file(preprocessor_files["preprocessor"])
    if preprocessor_summary.get("preprocessor_file_sha256") != preprocessor_sha256:
        raise ValueError("preprocessor file hash does not match preprocessor summary")
    preprocessor = load_nf_unsw_preprocessor(preprocessor_files["preprocessor"])
    if preprocessor_summary.get("preprocessor_hash") != preprocessor.preprocessor_hash:
        raise ValueError("preprocessor hash does not match preprocessor summary")
    expected_counts = {
        "training_rows": len(manifest.splits["train"]),
        "validation_rows": len(manifest.splits["val"]),
        "test_rows": len(manifest.splits["test"]),
    }
    for field, expected in expected_counts.items():
        if preprocessor_summary.get(field) != expected:
            raise ValueError(f"preprocessor summary {field} does not match split manifest")
    if preprocessor.training_row_count != expected_counts["training_rows"]:
        raise ValueError("preprocessor training_row_count does not match train split")
    training_sample_ids_hash = stable_hash(
        list(manifest.splits["train"]),
        TRAINING_SAMPLE_IDS_NAMESPACE,
    )
    if preprocessor_summary.get("training_sample_ids_hash") != training_sample_ids_hash:
        raise ValueError("preprocessor training_sample_ids_hash does not match train split")

    try:
        audit_config_path = split_config["audit_config"]
    except KeyError as exc:
        raise ValueError("split config snapshot is missing audit_config") from exc
    audit_config = load_audit_config(audit_config_path)
    for summary_name, summary in (
        ("split summary", split_summary),
        ("preprocessor summary", preprocessor_summary),
    ):
        if summary.get("audit_config_hash") != audit_config.config_hash:
            raise ValueError(f"{summary_name} audit config hash does not match current config")
    raw_sha1 = _hash_file(audit_config.raw_csv, "sha1")
    for summary_name, summary in (
        ("split summary", split_summary),
        ("preprocessor summary", preprocessor_summary),
    ):
        if summary.get("raw_csv_sha1") != raw_sha1:
            raise ValueError(f"{summary_name} raw CSV SHA1 does not match current data")

    feature_group_sizes = Counter(
        record["feature_key"] for record in selected_records
    )
    records_by_source_row = {}
    for record in selected_records:
        if record["source_file"] != audit_config.source_file_id:
            raise ValueError("selected source_file does not match audit config")
        source_row_id = record["source_row_id"]
        if not source_row_id.isdigit() or int(source_row_id) < 1:
            raise ValueError("selected source_row_id must be a positive integer string")
        records_by_source_row[int(source_row_id)] = record

    expected_header = (
        *NF_UNSW_NB15_V2_SCHEMA.raw_feature_names,
        audit_config.binary_label_column,
        audit_config.native_label_column,
    )
    samples_by_id = {}
    with audit_config.raw_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header does not match the registered schema")
        for row_number, row in enumerate(reader, start=1):
            record = records_by_source_row.get(row_number)
            if record is None:
                continue
            if compute_feature_key(row, NF_UNSW_NB15_V2_SCHEMA) != record["feature_key"]:
                raise ValueError("raw row feature_key does not match selected record")
            native_label = row[audit_config.native_label_column].strip()
            if native_label != record["native_label"]:
                raise ValueError("raw row native label does not match selected record")
            if audit_config.coarse_label_map.get(native_label) != record["coarse_label"]:
                raise ValueError("raw row coarse label does not match selected record")
            if _normalize_binary_label(
                row[audit_config.binary_label_column]
            ) != record["binary_label"]:
                raise ValueError("raw row binary label does not match selected record")
            encoded = preprocessor.transform(row)
            sample = FlowSample(
                features=encoded.features,
                feature_ids=encoded.feature_ids,
                categorical_mask=encoded.categorical_mask,
                missing_mask=encoded.missing_mask,
                category_tokens=encoded.category_tokens,
                binary_label=record["binary_label"],
                coarse_label=record["coarse_label"],
                native_label=record["native_label"],
                dataset_id=NF_UNSW_NB15_V2_SCHEMA.dataset_id,
                sample_id=record["sample_id"],
                feature_key=record["feature_key"],
                conflict_flag=False,
                conflict_group_size=feature_group_sizes[record["feature_key"]],
                conflict_label_count=1,
                source_file=record["source_file"],
                source_row_id=record["source_row_id"],
            )
            samples_by_id[sample.sample_id] = sample

    missing_ids = sorted(assigned_ids - samples_by_id.keys())
    if missing_ids:
        raise ValueError(f"selected source rows are missing from raw CSV: {missing_ids[:5]}")
    splits = {
        split_name: tuple(
            samples_by_id[sample_id] for sample_id in manifest.splits[split_name]
        )
        for split_name in _SPLIT_NAMES
    }
    return NFUNSWDevelopmentSamples(
        splits=splits,
        preprocessor_hash=preprocessor.preprocessor_hash,
        split_manifest_sha256=manifest_sha256,
        raw_csv_sha1=raw_sha1,
        schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
        training_sample_ids_hash=training_sample_ids_hash,
    )


__all__ = [
    "NFUNSWDevelopmentSamples",
    "load_nf_unsw_development_samples",
]
