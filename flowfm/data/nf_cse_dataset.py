"""Validated NF-CSE development samples from frozen split artifacts."""

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
from flowfm.data.nf_cse_few_shot import (
    SELECTION_MANIFEST_FILENAME,
    SELECTION_RECORDS_FILENAME,
    SELECTION_SUMMARY_FILENAME,
)
from flowfm.data.nf_cse_preprocessing import (
    SELECTION_FEATURES_FILENAME,
    TRAINING_SAMPLE_IDS_NAMESPACE,
    load_nf_cse_preprocessor,
)
from flowfm.data.nf_cse_split import SPLIT_CONFIG_NAMESPACE
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
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


def _load_selection_features(path: Path) -> dict[int, dict[str, str]]:
    expected = set(NF_CSE_CIC_IDS2018_V2_SCHEMA.model_feature_names)
    values = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                payload = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid selection feature JSON at line {line_number}"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != {
                "features",
                "source_row_id",
            }:
                raise ValueError("selection feature rows have invalid fields")
            source_row_id = payload["source_row_id"]
            features = payload["features"]
            if (
                not isinstance(source_row_id, str)
                or not source_row_id.isdigit()
                or int(source_row_id) < 1
            ):
                raise ValueError("selection feature source_row_id is invalid")
            if (
                not isinstance(features, dict)
                or set(features) != expected
                or not all(isinstance(value, str) for value in features.values())
            ):
                raise ValueError(
                    "selection feature values do not match the registered schema"
                )
            numeric_row_id = int(source_row_id)
            if numeric_row_id in values:
                raise ValueError("selection feature source rows are duplicated")
            values[numeric_row_id] = features
    return values


@dataclass(frozen=True)
class NFCSEDevelopmentSamples:
    splits: Mapping[str, tuple[FlowSample, ...]]
    preprocessor_hash: str
    split_manifest_sha256: str
    raw_csv_sha1: str
    schema_version: str
    training_sample_ids_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.splits, Mapping)
            or not self.splits
            or not set(self.splits).issubset(_SPLIT_NAMES)
        ):
            raise ValueError("splits must contain a non-empty subset of train, val, test")
        normalized = {}
        seen = set()
        for split_name in _SPLIT_NAMES:
            if split_name not in self.splits:
                continue
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


def load_nf_cse_development_samples(
    split_dir,
    preprocessor_dir,
    *,
    data_root=Path("/root/autodl-tmp"),
    split_names=_SPLIT_NAMES,
) -> NFCSEDevelopmentSamples:
    """Load privacy-safe FlowSamples after validating all frozen artifacts."""

    if isinstance(split_names, (str, bytes)) or not isinstance(
        split_names, (list, tuple)
    ):
        raise TypeError("split_names must be a list or tuple")
    requested_splits = tuple(split_names)
    if (
        not requested_splits
        or len(set(requested_splits)) != len(requested_splits)
        or any(name not in _SPLIT_NAMES for name in requested_splits)
    ):
        raise ValueError("split_names must be a unique non-empty subset of train, val, test")

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
        "data_manifest": split_root / "data_manifest.json",
    }
    preprocessor_files = {
        "config": preprocessor_root / "config.snapshot.json",
        "preprocessor": preprocessor_root / "preprocessor.json",
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
    if (split_root / "REJECTED.json").exists():
        raise ValueError("split artifact directory is explicitly rejected")

    split_config = _load_json_object(split_files["config"], "split config snapshot")
    selection_view = (
        requested_splits == ("train", "val")
        and split_config.get("sealed_test") is True
    )
    split_files.update(
        {
            "manifest": split_root
            / (
                SELECTION_MANIFEST_FILENAME
                if selection_view
                else "split_manifest.json"
            ),
            "selected": split_root
            / (
                SELECTION_RECORDS_FILENAME
                if selection_view
                else "selected_records.jsonl"
            ),
            "summary": split_root
            / (
                SELECTION_SUMMARY_FILENAME
                if selection_view
                else "split_summary.json"
            ),
        }
    )
    preprocessor_files["summary"] = preprocessor_root / (
        SELECTION_SUMMARY_FILENAME
        if selection_view
        else "preprocessor_summary.json"
    )
    for prefix, files in (
        (
            "split",
            {
                key: split_files[key]
                for key in ("manifest", "selected", "summary")
            },
        ),
        ("preprocessor", {"summary": preprocessor_files["summary"]}),
    ):
        for label, path in files.items():
            if not path.is_file():
                raise ValueError(
                    f"published artifacts are missing required {prefix} {label}: {path}"
                )
    split_summary = _load_json_object(split_files["summary"], "split summary")
    preprocessor_config = _load_json_object(
        preprocessor_files["config"], "preprocessor config snapshot"
    )
    preprocessor_summary = _load_json_object(
        preprocessor_files["summary"], "preprocessor summary"
    )
    if selection_view:
        for summary_name, summary in (
            ("split selection summary", split_summary),
            ("preprocessor selection summary", preprocessor_summary),
        ):
            if (
                summary.get("selection_scope") != "train_validation_only"
                or summary.get("materialized_splits") != ["train", "val"]
                or summary.get("test_labels_present") is not False
                or summary.get("test_statistics_present") is not False
            ):
                raise ValueError(
                    f"{summary_name} does not preserve the sealed-test boundary"
                )
        if preprocessor_summary.get(
            "split_selection_summary_sha256"
        ) != _hash_file(split_files["summary"]):
            raise ValueError(
                "preprocessor selection summary does not lock split selection summary"
            )
        if preprocessor_summary.get(
            "sealed_split_manifest_sha256"
        ) != split_summary.get("sealed_split_manifest_sha256"):
            raise ValueError("selection summaries disagree on sealed manifest hash")
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
    data_manifest_sha256 = _hash_file(split_files["data_manifest"])
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
        if summary.get("source_manifest_sha256") != data_manifest_sha256:
            raise ValueError(
                f"{summary_name} data manifest hash does not match artifacts"
            )

    expected_identity = {
        "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
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
    if manifest.schema_version != NF_CSE_CIC_IDS2018_V2_SCHEMA.version:
        raise ValueError("split manifest schema_version does not match implementation")
    if manifest.source_hash != split_summary.get("source_hash"):
        raise ValueError("split manifest source_hash does not match split summary")
    if manifest.seed != split_summary.get("seed"):
        raise ValueError("split manifest seed does not match split summary")
    if dict(manifest.ratios) != split_summary.get("ratios"):
        raise ValueError("split manifest ratios do not match split summary")
    if selection_view and manifest.splits["test"]:
        raise ValueError("selection manifest contains sealed test members")

    selected_records = _load_selected_records(split_files["selected"])
    records_by_sample = {
        record["sample_id"]: record for record in selected_records
    }
    assigned_ids = {
        sample_id
        for split_name in (
            requested_splits if selection_view else _SPLIT_NAMES
        )
        for sample_id in manifest.splits[split_name]
    }
    if assigned_ids != set(records_by_sample):
        raise ValueError("selected records and split manifest sample IDs do not match")

    preprocessor_sha256 = _hash_file(preprocessor_files["preprocessor"])
    if preprocessor_summary.get("preprocessor_file_sha256") != preprocessor_sha256:
        raise ValueError("preprocessor file hash does not match preprocessor summary")
    preprocessor = load_nf_cse_preprocessor(preprocessor_files["preprocessor"])
    if preprocessor_summary.get("preprocessor_hash") != preprocessor.preprocessor_hash:
        raise ValueError("preprocessor hash does not match preprocessor summary")
    expected_counts = {
        "training_rows": len(manifest.splits["train"]),
        "validation_rows": len(manifest.splits["val"]),
    }
    if not selection_view:
        expected_counts["test_rows"] = len(manifest.splits["test"])
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
    if selection_view:
        raw_sha1 = split_summary.get("raw_csv_sha1")
        if (
            not isinstance(raw_sha1, str)
            or preprocessor_summary.get("raw_csv_sha1") != raw_sha1
        ):
            raise ValueError("selection summaries do not lock one raw CSV SHA1")
    else:
        raw_sha1 = _hash_file(audit_config.raw_csv, "sha1")
        for summary_name, summary in (
            ("split summary", split_summary),
            ("preprocessor summary", preprocessor_summary),
        ):
            if summary.get("raw_csv_sha1") != raw_sha1:
                raise ValueError(
                    f"{summary_name} raw CSV SHA1 does not match current data"
                )

    requested_ids = {
        sample_id
        for split_name in requested_splits
        for sample_id in manifest.splits[split_name]
    }
    feature_group_sizes = Counter(
        record["feature_key"]
        for record in selected_records
        if record["sample_id"] in requested_ids
    )
    records_by_source_row = {}
    for record in selected_records:
        if record["sample_id"] not in requested_ids:
            continue
        if record["source_file"] != audit_config.source_file_id:
            raise ValueError("selected source_file does not match audit config")
        source_row_id = record["source_row_id"]
        if not source_row_id.isdigit() or int(source_row_id) < 1:
            raise ValueError("selected source_row_id must be a positive integer string")
        records_by_source_row[int(source_row_id)] = record

    samples_by_id = {}
    if selection_view:
        selection_features_path = (
            preprocessor_root / SELECTION_FEATURES_FILENAME
        )
        if not selection_features_path.is_file():
            raise ValueError("sealed selection feature cache is missing")
        if preprocessor_summary.get("selection_features_sha256") != _hash_file(
            selection_features_path
        ):
            raise ValueError("selection feature cache hash does not match summary")
        if (
            preprocessor_summary.get(
                "selection_features_model_payload_label_fields_present"
            )
            is not False
            or preprocessor_summary.get(
                "selection_features_model_payload_trace_fields_present"
            )
            is not False
            or preprocessor_summary.get(
                "selection_features_join_key_is_model_input"
            )
            is not False
            or preprocessor_summary.get("selection_features_join_key")
            != "source_row_id"
            or preprocessor_summary.get(
                "selection_features_model_payload_feature_names"
            )
            != list(NF_CSE_CIC_IDS2018_V2_SCHEMA.model_feature_names)
        ):
            raise ValueError("selection feature cache violates the sealed-test contract")
        cached_rows = _load_selection_features(selection_features_path)
        if set(cached_rows) != set(records_by_source_row):
            raise ValueError("selection feature cache rows do not match selection records")
        raw_rows = sorted(cached_rows.items())
    else:
        expected_header = (
            *NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_feature_names,
            audit_config.binary_label_column,
            audit_config.native_label_column,
        )
        raw_rows = []
        with audit_config.raw_csv.open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != expected_header:
                raise ValueError("raw CSV header does not match the registered schema")
            raw_rows = [
                (row_number, row)
                for row_number, row in enumerate(reader, start=1)
                if row_number in records_by_source_row
            ]

    for row_number, row in raw_rows:
        record = records_by_source_row.get(row_number)
        if record is None:
            continue
        if compute_feature_key(row, NF_CSE_CIC_IDS2018_V2_SCHEMA) != record["feature_key"]:
            raise ValueError("raw row feature_key does not match selected record")
        if not selection_view:
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
            dataset_id=NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
            sample_id=record["sample_id"],
            feature_key=record["feature_key"],
            conflict_flag=False,
            conflict_group_size=feature_group_sizes[record["feature_key"]],
            conflict_label_count=1,
            source_file=record["source_file"],
            source_row_id=record["source_row_id"],
        )
        samples_by_id[sample.sample_id] = sample

    missing_ids = sorted(requested_ids - samples_by_id.keys())
    if missing_ids:
        raise ValueError(f"selected source rows are missing from raw CSV: {missing_ids[:5]}")
    splits = {
        split_name: tuple(
            samples_by_id[sample_id] for sample_id in manifest.splits[split_name]
        )
        for split_name in requested_splits
    }
    return NFCSEDevelopmentSamples(
        splits=splits,
        preprocessor_hash=preprocessor.preprocessor_hash,
        split_manifest_sha256=manifest_sha256,
        raw_csv_sha1=raw_sha1,
        schema_version=NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        training_sample_ids_hash=training_sample_ids_hash,
    )


__all__ = [
    "NFCSEDevelopmentSamples",
    "load_nf_cse_development_samples",
]
