"""Train-only numeric scaling and categorical vocabularies for NF-CSE."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flowfm.data.audit import load_audit_config
from flowfm.data.feature_keys import canonicalize_value
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.nf_cse_few_shot import (
    SELECTION_MANIFEST_FILENAME,
    SELECTION_RECORDS_FILENAME,
    SELECTION_SUMMARY_FILENAME,
)
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
from flowfm.data.nf_cse_split import SPLIT_CONFIG_NAMESPACE
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


PREPROCESSOR_FORMAT_VERSION = 1
PREPROCESSOR_HASH_NAMESPACE = "flowfm-nf-cse-cic-ids2018-v2-preprocessor-v1"
MISSING_CATEGORY_ID = 0
UNKNOWN_CATEGORY_ID = 1
TRAINING_SAMPLE_IDS_NAMESPACE = "flowfm-nf-cse-cic-ids2018-v2-training-sample-ids-v1"
PREPROCESSOR_PUBLICATION_FORMAT_VERSION = 1
SELECTION_FEATURES_FILENAME = "selection_features.jsonl"


def _decimal_text(value: Decimal) -> str:
    token = canonicalize_value(value, "numeric")
    return token.removeprefix("n:")


def _numeric_feature_names() -> tuple[str, ...]:
    return tuple(
        spec.name for spec in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features
        if not spec.categorical
    )


def _categorical_feature_names() -> tuple[str, ...]:
    return tuple(
        spec.name for spec in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features
        if spec.categorical
    )


@dataclass(frozen=True)
class EncodedNFCSEFeatures:
    features: tuple[float | int, ...]
    feature_ids: tuple[str, ...]
    categorical_mask: tuple[bool, ...]
    missing_mask: tuple[bool, ...]
    category_tokens: tuple[str | None, ...]


@dataclass(frozen=True)
class NFCSETrainOnlyPreprocessor:
    training_row_count: int
    numeric_statistics: Mapping[str, Mapping[str, Any]]
    categorical_vocabularies: Mapping[str, tuple[str, ...]]
    preprocessor_hash: str = ""
    _category_ids: Mapping[str, Mapping[str, int]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.training_row_count) is not int or self.training_row_count <= 0:
            raise ValueError("training_row_count must be a positive integer")
        expected_numeric = set(_numeric_feature_names())
        expected_categorical = set(_categorical_feature_names())
        if set(self.numeric_statistics) != expected_numeric:
            raise ValueError("numeric_statistics do not match the NF-CSE schema")
        if set(self.categorical_vocabularies) != expected_categorical:
            raise ValueError("categorical_vocabularies do not match the NF-CSE schema")

        normalized_statistics = {}
        for feature_name in sorted(expected_numeric):
            raw = self.numeric_statistics[feature_name]
            if not isinstance(raw, Mapping):
                raise ValueError("numeric feature statistics must be mappings")
            required = {"observed_count", "missing_count", "min", "max", "mean"}
            if set(raw) != required:
                raise ValueError("numeric feature statistics have invalid fields")
            observed = raw["observed_count"]
            missing = raw["missing_count"]
            if type(observed) is not int or type(missing) is not int:
                raise ValueError("numeric feature counts must be integers")
            if observed < 0 or missing < 0 or observed + missing != self.training_row_count:
                raise ValueError("numeric feature counts do not match training_row_count")
            values = (raw["min"], raw["max"], raw["mean"])
            if observed == 0:
                if values != (None, None, None):
                    raise ValueError("all-missing numeric statistics must use null values")
            else:
                if not all(isinstance(value, str) and value for value in values):
                    raise ValueError("observed numeric statistics must be strings")
                minimum, maximum, _ = (Decimal(value) for value in values)
                if minimum > maximum:
                    raise ValueError("numeric minimum cannot exceed maximum")
            normalized_statistics[feature_name] = MappingProxyType(
                {
                    "observed_count": observed,
                    "missing_count": missing,
                    "min": raw["min"],
                    "max": raw["max"],
                    "mean": raw["mean"],
                }
            )

        normalized_vocabularies = {}
        category_ids = {}
        for feature_name in sorted(expected_categorical):
            vocabulary = self.categorical_vocabularies[feature_name]
            if not isinstance(vocabulary, (list, tuple)):
                raise ValueError("categorical vocabularies must be sequences")
            normalized = tuple(vocabulary)
            if normalized != tuple(sorted(set(normalized))):
                raise ValueError("categorical vocabularies must be sorted and unique")
            if any(
                not isinstance(token, str) or not token or token == "m:"
                for token in normalized
            ):
                raise ValueError("categorical vocabularies contain invalid tokens")
            normalized_vocabularies[feature_name] = normalized
            category_ids[feature_name] = MappingProxyType(
                {
                    token: index
                    for index, token in enumerate(normalized, start=2)
                }
            )

        object.__setattr__(
            self,
            "numeric_statistics",
            MappingProxyType(normalized_statistics),
        )
        object.__setattr__(
            self,
            "categorical_vocabularies",
            MappingProxyType(normalized_vocabularies),
        )
        object.__setattr__(self, "_category_ids", MappingProxyType(category_ids))
        expected_hash = stable_hash(self.to_dict(), PREPROCESSOR_HASH_NAMESPACE)
        if self.preprocessor_hash and self.preprocessor_hash != expected_hash:
            raise ValueError("preprocessor_hash does not match preprocessor content")
        object.__setattr__(self, "preprocessor_hash", expected_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PREPROCESSOR_FORMAT_VERSION,
            "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
            "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
            "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
            "training_row_count": self.training_row_count,
            "missing_category_id": MISSING_CATEGORY_ID,
            "unknown_category_id": UNKNOWN_CATEGORY_ID,
            "numeric_statistics": {
                feature_name: dict(statistics)
                for feature_name, statistics in sorted(
                    self.numeric_statistics.items()
                )
            },
            "categorical_vocabularies": {
                feature_name: list(vocabulary)
                for feature_name, vocabulary in sorted(
                    self.categorical_vocabularies.items()
                )
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        preprocessor_hash: str,
    ) -> "NFCSETrainOnlyPreprocessor":
        required = {
            "format_version",
            "dataset_id",
            "schema_version",
            "schema_hash",
            "training_row_count",
            "missing_category_id",
            "unknown_category_id",
            "numeric_statistics",
            "categorical_vocabularies",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("preprocessor payload has invalid fields")
        expected_values = {
            "format_version": PREPROCESSOR_FORMAT_VERSION,
            "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
            "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
            "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
            "missing_category_id": MISSING_CATEGORY_ID,
            "unknown_category_id": UNKNOWN_CATEGORY_ID,
        }
        for field_name, expected in expected_values.items():
            if payload[field_name] != expected:
                raise ValueError(f"preprocessor {field_name} does not match implementation")
        return cls(
            training_row_count=payload["training_row_count"],
            numeric_statistics=payload["numeric_statistics"],
            categorical_vocabularies=payload["categorical_vocabularies"],
            preprocessor_hash=preprocessor_hash,
        )

    def transform(self, row: Mapping[str, Any]) -> EncodedNFCSEFeatures:
        if not isinstance(row, Mapping):
            raise TypeError("NF-CSE feature rows must be mappings")
        features = []
        missing_mask = []
        category_tokens = []
        for spec in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features:
            token = canonicalize_value(row.get(spec.name), spec.value_type)
            missing = token == "m:"
            missing_mask.append(missing)
            if spec.categorical:
                category_tokens.append(None if missing else token)
                if missing:
                    features.append(MISSING_CATEGORY_ID)
                else:
                    features.append(
                        self._category_ids[spec.name].get(
                            token,
                            UNKNOWN_CATEGORY_ID,
                        )
                    )
                continue

            category_tokens.append(None)
            statistics = self.numeric_statistics[spec.name]
            if missing or statistics["observed_count"] == 0:
                features.append(0.0)
                continue
            value = Decimal(token.removeprefix("n:"))
            minimum = Decimal(statistics["min"])
            maximum = Decimal(statistics["max"])
            if maximum == minimum:
                features.append(0.0)
            else:
                features.append(float((value - minimum) / (maximum - minimum)))

        return EncodedNFCSEFeatures(
            features=tuple(features),
            feature_ids=NF_CSE_CIC_IDS2018_V2_SCHEMA.model_feature_names,
            categorical_mask=NF_CSE_CIC_IDS2018_V2_SCHEMA.categorical_mask,
            missing_mask=tuple(missing_mask),
            category_tokens=tuple(category_tokens),
        )


def fit_nf_cse_preprocessor(
    training_rows: Iterable[Mapping[str, Any]],
) -> NFCSETrainOnlyPreprocessor:
    numeric_accumulators = {
        feature_name: {
            "observed_count": 0,
            "missing_count": 0,
            "min": None,
            "max": None,
            "sum": Decimal(0),
        }
        for feature_name in _numeric_feature_names()
    }
    vocabularies = {
        feature_name: set() for feature_name in _categorical_feature_names()
    }
    training_row_count = 0
    for row in training_rows:
        training_row_count += 1
        if not isinstance(row, Mapping):
            raise TypeError("training rows must be mappings")
        for spec in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features:
            token = canonicalize_value(row.get(spec.name), spec.value_type)
            if spec.categorical:
                if token != "m:":
                    vocabularies[spec.name].add(token)
                continue

            accumulator = numeric_accumulators[spec.name]
            if token == "m:":
                accumulator["missing_count"] += 1
                continue
            value = Decimal(token.removeprefix("n:"))
            accumulator["observed_count"] += 1
            accumulator["sum"] += value
            if accumulator["min"] is None or value < accumulator["min"]:
                accumulator["min"] = value
            if accumulator["max"] is None or value > accumulator["max"]:
                accumulator["max"] = value

    if training_row_count == 0:
        raise ValueError("training data must contain at least one row")

    numeric_statistics = {}
    for feature_name, accumulator in numeric_accumulators.items():
        observed = accumulator["observed_count"]
        if observed == 0:
            minimum = maximum = mean = None
        else:
            with localcontext() as context:
                context.prec = 50
                mean_value = accumulator["sum"] / Decimal(observed)
            minimum = _decimal_text(accumulator["min"])
            maximum = _decimal_text(accumulator["max"])
            mean = _decimal_text(mean_value)
        numeric_statistics[feature_name] = {
            "observed_count": observed,
            "missing_count": accumulator["missing_count"],
            "min": minimum,
            "max": maximum,
            "mean": mean,
        }

    return NFCSETrainOnlyPreprocessor(
        training_row_count=training_row_count,
        numeric_statistics=numeric_statistics,
        categorical_vocabularies={
            feature_name: tuple(sorted(tokens))
            for feature_name, tokens in vocabularies.items()
        },
    )


def save_nf_cse_preprocessor(path, preprocessor) -> Path:
    if not isinstance(preprocessor, NFCSETrainOnlyPreprocessor):
        raise TypeError("preprocessor must be an NFCSETrainOnlyPreprocessor")
    return write_json_snapshot(
        path,
        {
            "preprocessor": preprocessor.to_dict(),
            "preprocessor_hash": preprocessor.preprocessor_hash,
        },
    )


def load_nf_cse_preprocessor(path) -> NFCSETrainOnlyPreprocessor:
    source = Path(path).expanduser().resolve()

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid preprocessor file: {source}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "preprocessor",
        "preprocessor_hash",
    }:
        raise ValueError("preprocessor file has invalid fields")
    preprocessor_hash = payload["preprocessor_hash"]
    if not isinstance(preprocessor_hash, str):
        raise ValueError("preprocessor_hash must be a string")
    return NFCSETrainOnlyPreprocessor.from_dict(
        payload["preprocessor"],
        preprocessor_hash=preprocessor_hash,
    )


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _git_commit(repo_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _load_selected_records(path: Path) -> tuple[dict[str, Any], ...]:
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
            try:
                sample_id = record["sample_id"]
                source_file = record["source_file"]
                source_row_id = record["source_row_id"]
            except KeyError as exc:
                raise ValueError("selected record is missing source identity") from exc
            if not all(
                isinstance(value, str) and value
                for value in (sample_id, source_file, source_row_id)
            ):
                raise ValueError("selected record source identity must be non-empty")
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate selected sample_id: {sample_id}")
            source_ref = (source_file, source_row_id)
            if source_ref in seen_source_refs:
                raise ValueError(
                    f"duplicate selected source reference: {source_file}:{source_row_id}"
                )
            seen_sample_ids.add(sample_id)
            seen_source_refs.add(source_ref)
            records.append(record)
    return tuple(records)


def _training_rows(audit_config, source_row_ids: set[int]):
    expected_header = (
        *NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_feature_names,
        audit_config.binary_label_column,
        audit_config.native_label_column,
    )
    seen = set()
    with audit_config.raw_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header does not match the registered schema")
        for row_number, row in enumerate(reader, start=1):
            if row_number in source_row_ids:
                seen.add(row_number)
                yield row
    missing = sorted(source_row_ids - seen)
    if missing:
        raise ValueError(
            f"training source rows are missing from raw CSV: {missing[:5]}"
        )


def _selection_model_rows(audit_config, source_row_ids: set[int]):
    expected_header = (
        *NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_feature_names,
        audit_config.binary_label_column,
        audit_config.native_label_column,
    )
    selected = {}
    with audit_config.raw_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header does not match the registered schema")
        for row_number, row in enumerate(reader, start=1):
            if row_number not in source_row_ids:
                continue
            selected[row_number] = {
                name: row[name]
                for name in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_feature_names
            }
    missing = sorted(source_row_ids - set(selected))
    if missing:
        raise ValueError(
            f"selection source rows are missing from raw CSV: {missing[:5]}"
        )
    return selected


def _categorical_training_statistics(training_rows):
    counters = {
        name: Counter() for name in _categorical_feature_names()
    }
    missing_counts = dict.fromkeys(_categorical_feature_names(), 0)
    for row in training_rows:
        for spec in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features:
            if not spec.categorical:
                continue
            token = canonicalize_value(row.get(spec.name), spec.value_type)
            if token == "m:":
                missing_counts[spec.name] += 1
            else:
                counters[spec.name][token] += 1
    statistics = {}
    for feature_name in sorted(counters):
        frequencies = tuple(counters[feature_name].values())
        histogram = Counter(frequencies)
        statistics[feature_name] = {
            "observed_count": sum(frequencies),
            "missing_count": missing_counts[feature_name],
            "unique_count": len(frequencies),
            "singleton_count": histogram.get(1, 0),
            "min_frequency": min(frequencies) if frequencies else 0,
            "max_frequency": max(frequencies) if frequencies else 0,
            "frequency_histogram": {
                str(frequency): unique_categories
                for frequency, unique_categories in sorted(histogram.items())
            },
        }
    return statistics


def publish_nf_cse_train_only_preprocessor(
    split_dir,
    output_path,
    *,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Fit only the manifest train split and publish immutable preprocessing state."""

    source_dir = validate_output_path(
        split_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not source_dir.is_dir():
        raise ValueError(f"split_dir must reference an existing directory: {source_dir}")
    required = {
        "config": source_dir / "config.snapshot.json",
        "data_manifest": source_dir / "data_manifest.json",
        "selected": source_dir / "selected_records.jsonl",
        "manifest": source_dir / "split_manifest.json",
        "summary": source_dir / "split_summary.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise ValueError(f"split_dir is missing required {label} file: {path.name}")
    if (source_dir / "REJECTED.json").exists():
        raise ValueError("split_dir is explicitly rejected")
    if (source_dir / ".incomplete").exists():
        raise ValueError("split_dir is incomplete")

    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if output_dir.exists():
        raise ValueError(f"preprocessor output directory already exists: {output_dir}")

    split_config = _load_json_object(required["config"], "split config snapshot")
    split_summary = _load_json_object(required["summary"], "split summary")
    sealed_selection = split_config.get("sealed_test") is True
    selection_records_path = source_dir / SELECTION_RECORDS_FILENAME
    selection_manifest_path = source_dir / SELECTION_MANIFEST_FILENAME
    selection_summary_path = source_dir / SELECTION_SUMMARY_FILENAME
    if sealed_selection and (
        not selection_records_path.is_file()
        or not selection_manifest_path.is_file()
        or not selection_summary_path.is_file()
    ):
        raise ValueError("sealed split is missing train-validation selection artifacts")
    split_selection_summary = (
        _load_json_object(selection_summary_path, "split selection summary")
        if sealed_selection
        else None
    )
    if stable_hash(split_config, SPLIT_CONFIG_NAMESPACE) != split_summary.get(
        "config_hash"
    ):
        raise ValueError("split config hash does not match split summary")
    if split_summary.get("dataset_id") != NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id:
        raise ValueError("split summary dataset_id does not match NF-CSE")
    if split_summary.get("schema_hash") != NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash:
        raise ValueError("split summary schema_hash does not match implementation")
    selected_sha256 = _hash_file(required["selected"])
    manifest_sha256 = _hash_file(required["manifest"])
    data_manifest_sha256 = _hash_file(required["data_manifest"])
    if split_summary.get("selected_records_sha256") != selected_sha256:
        raise ValueError("selected record hash does not match split summary")
    if split_summary.get("split_manifest_sha256") != manifest_sha256:
        raise ValueError("split manifest hash does not match split summary")
    if split_summary.get("source_manifest_sha256") != data_manifest_sha256:
        raise ValueError("data manifest hash does not match split summary")

    manifest = SplitManifest.from_json(required["manifest"].read_text(encoding="utf-8"))
    if manifest.schema_version != NF_CSE_CIC_IDS2018_V2_SCHEMA.version:
        raise ValueError("split manifest schema_version does not match implementation")
    if manifest.source_hash != split_summary.get("source_hash"):
        raise ValueError("split manifest source_hash does not match split summary")
    if manifest.seed != split_summary.get("seed"):
        raise ValueError("split manifest seed does not match split summary")
    if dict(manifest.ratios) != split_summary.get("ratios"):
        raise ValueError("split manifest ratios do not match split summary")
    selection_manifest = None
    selection_manifest_sha256 = None
    if sealed_selection:
        selection_manifest = SplitManifest.from_json(
            selection_manifest_path.read_text(encoding="utf-8")
        )
        if selection_manifest.splits["test"]:
            raise ValueError("selection manifest must not contain test members")
        for split_name in ("train", "val"):
            if selection_manifest.splits[split_name] != manifest.splits[split_name]:
                raise ValueError(
                    f"selection manifest {split_name} does not match sealed manifest"
                )
        if (
            selection_manifest.seed != manifest.seed
            or selection_manifest.schema_version != manifest.schema_version
            or selection_manifest.source_hash != manifest.source_hash
            or dict(selection_manifest.ratios) != dict(manifest.ratios)
        ):
            raise ValueError("selection manifest provenance does not match sealed manifest")
        selection_manifest_sha256 = _hash_file(selection_manifest_path)
    selected_records = _load_selected_records(required["selected"])
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
    try:
        audit_config_path = split_config["audit_config"]
    except KeyError as exc:
        raise ValueError("split config snapshot is missing audit_config") from exc
    audit_config = load_audit_config(audit_config_path)
    if audit_config.config_hash != split_summary.get("audit_config_hash"):
        raise ValueError("audit config hash does not match split summary")
    raw_sha1 = _hash_file(audit_config.raw_csv, "sha1")
    if split_summary.get("raw_csv_sha1") != raw_sha1:
        raise ValueError("raw CSV SHA1 does not match split summary")

    training_sample_ids = tuple(manifest.splits["train"])
    training_source_row_ids = set()
    for sample_id in training_sample_ids:
        record = records_by_sample[sample_id]
        if record["source_file"] != audit_config.source_file_id:
            raise ValueError("selected source_file does not match audit config")
        source_row_id = record["source_row_id"]
        if not source_row_id.isdigit() or int(source_row_id) < 1:
            raise ValueError("selected source_row_id must be a positive integer string")
        training_source_row_ids.add(int(source_row_id))
    if len(training_source_row_ids) != len(training_sample_ids):
        raise ValueError("training samples do not map to unique source rows")

    selection_records = None
    selection_source_row_ids = None
    if sealed_selection:
        selection_records = _load_selected_records(selection_records_path)
        expected_selection_ids = {
            sample_id
            for split_name in ("train", "val")
            for sample_id in selection_manifest.splits[split_name]
        }
        if {
            record["sample_id"] for record in selection_records
        } != expected_selection_ids:
            raise ValueError(
                "selection records must contain exactly train and validation"
            )
        selection_source_row_ids = set()
        for record in selection_records:
            if record["source_file"] != audit_config.source_file_id:
                raise ValueError("selection source_file does not match audit config")
            source_row_id = record["source_row_id"]
            if not source_row_id.isdigit() or int(source_row_id) < 1:
                raise ValueError(
                    "selection source_row_id must be a positive integer string"
                )
            selection_source_row_ids.add(int(source_row_id))
        if len(selection_source_row_ids) != len(selection_records):
            raise ValueError("selection records must map to unique source rows")
        expected_selection = {
            "selection_scope": "train_validation_only",
            "materialized_splits": ["train", "val"],
            "test_labels_present": False,
            "test_statistics_present": False,
            "selected_records_sha256": _hash_file(selection_records_path),
            "split_manifest_sha256": selection_manifest_sha256,
            "sealed_split_manifest_sha256": manifest_sha256,
            "source_manifest_sha256": data_manifest_sha256,
            "config_hash": split_summary.get("config_hash"),
        }
        for field, expected in expected_selection.items():
            if split_selection_summary.get(field) != expected:
                raise ValueError(
                    f"split selection summary {field} does not match artifacts"
                )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text(
        "NF-CSE train-only preprocessor publication incomplete\n",
        encoding="utf-8",
    )
    selection_model_rows = None
    if sealed_selection:
        selection_model_rows = _selection_model_rows(
            audit_config, selection_source_row_ids
        )
        training_rows = tuple(
            selection_model_rows[row_id]
            for row_id in sorted(training_source_row_ids)
        )
    else:
        training_rows = tuple(
            _training_rows(audit_config, training_source_row_ids)
        )
    preprocessor = fit_nf_cse_preprocessor(training_rows)
    categorical_training_statistics = _categorical_training_statistics(
        training_rows
    )
    config_snapshot = {
        "fit_scope": "train_only",
        "source_split_dir": str(source_dir),
    }
    write_json_snapshot(output_dir / "config.snapshot.json", config_snapshot)
    preprocessor_path = save_nf_cse_preprocessor(
        output_dir / "preprocessor.json",
        preprocessor,
    )
    preprocessor_file_sha256 = _hash_file(preprocessor_path)
    preprocessor_selection_summary_path = None
    selection_features_path = None
    if sealed_selection:
        selection_features_path = output_dir / SELECTION_FEATURES_FILENAME
        selection_features_path.write_text(
            "".join(
                canonical_json(
                    {
                        "features": selection_model_rows[source_row_id],
                        "source_row_id": str(source_row_id),
                    }
                )
                + "\n"
                for source_row_id in sorted(selection_model_rows)
            ),
            encoding="utf-8",
        )
        selection_records_sha256 = _hash_file(selection_records_path)
        split_selection_summary_sha256 = _hash_file(selection_summary_path)
        preprocessor_selection_summary_path = write_json_snapshot(
            output_dir / SELECTION_SUMMARY_FILENAME,
            {
                "format_version": PREPROCESSOR_PUBLICATION_FORMAT_VERSION,
                "dataset_id": audit_config.dataset_id,
                "fit_scope": "train_only",
                "selection_scope": "train_validation_only",
                "materialized_splits": ["train", "val"],
                "test_labels_present": False,
                "test_statistics_present": False,
                "training_rows": len(manifest.splits["train"]),
                "validation_rows": len(manifest.splits["val"]),
                "training_sample_ids_hash": stable_hash(
                    list(manifest.splits["train"]),
                    TRAINING_SAMPLE_IDS_NAMESPACE,
                ),
                "preprocessor_hash": preprocessor.preprocessor_hash,
                "preprocessor_file_sha256": preprocessor_file_sha256,
                "selection_features_file": SELECTION_FEATURES_FILENAME,
                "selection_features_rows": len(selection_model_rows),
                "selection_features_sha256": _hash_file(
                    selection_features_path
                ),
                "selection_features_join_key": "source_row_id",
                "selection_features_join_key_is_model_input": False,
                "selection_features_model_payload_label_fields_present": False,
                "selection_features_model_payload_trace_fields_present": False,
                "selection_features_model_payload_feature_names": list(
                    NF_CSE_CIC_IDS2018_V2_SCHEMA.model_feature_names
                ),
                "selected_records_sha256": selection_records_sha256,
                "split_selection_summary_sha256": (
                    split_selection_summary_sha256
                ),
                "split_manifest_sha256": selection_manifest_sha256,
                "sealed_split_manifest_sha256": manifest_sha256,
                "source_manifest_sha256": data_manifest_sha256,
                "split_config_hash": split_summary.get("config_hash"),
                "audit_config_hash": split_summary.get("audit_config_hash"),
                "raw_csv_sha1": raw_sha1,
                "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
                "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
            },
        )
    summary = {
        "format_version": PREPROCESSOR_PUBLICATION_FORMAT_VERSION,
        "dataset_id": audit_config.dataset_id,
        "fit_scope": "train_only",
        "training_rows": len(training_sample_ids),
        "validation_rows": len(manifest.splits["val"]),
        "test_rows": len(manifest.splits["test"]),
        "training_sample_ids_hash": stable_hash(
            list(training_sample_ids),
            TRAINING_SAMPLE_IDS_NAMESPACE,
        ),
        "preprocessor_hash": preprocessor.preprocessor_hash,
        "preprocessor_file_sha256": preprocessor_file_sha256,
        "selected_records_sha256": selected_sha256,
        "split_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": data_manifest_sha256,
        "split_config_hash": split_summary.get("config_hash"),
        "audit_config_hash": split_summary.get("audit_config_hash"),
        "raw_csv_sha1": raw_sha1,
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "selection_records_sha256": (
            _hash_file(selection_records_path) if sealed_selection else None
        ),
        "split_selection_summary_sha256": (
            _hash_file(selection_summary_path) if sealed_selection else None
        ),
        "selection_manifest_sha256": (
            selection_manifest_sha256 if sealed_selection else None
        ),
        "preprocessor_selection_summary_sha256": (
            _hash_file(preprocessor_selection_summary_path)
            if preprocessor_selection_summary_path is not None
            else None
        ),
        "selection_features_sha256": (
            _hash_file(selection_features_path)
            if selection_features_path is not None
            else None
        ),
        "categorical_vocabulary_sizes": {
            name: len(vocabulary)
            for name, vocabulary in sorted(
                preprocessor.categorical_vocabularies.items()
            )
        },
        "categorical_training_statistics": (
            categorical_training_statistics
        ),
        "numeric_statistics": {
            name: dict(statistics)
            for name, statistics in sorted(
                preprocessor.numeric_statistics.items()
            )
        },
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
    }
    write_json_snapshot(output_dir / "preprocessor_summary.json", summary)
    marker.unlink()
    return {
        "fit_scope": "train_only",
        "training_rows": len(training_sample_ids),
        "preprocessor_hash": preprocessor.preprocessor_hash,
        "preprocessor_file_sha256": summary["preprocessor_file_sha256"],
        "split_manifest_sha256": manifest_sha256,
    }


__all__ = [
    "EncodedNFCSEFeatures",
    "MISSING_CATEGORY_ID",
    "NFCSETrainOnlyPreprocessor",
    "PREPROCESSOR_FORMAT_VERSION",
    "PREPROCESSOR_HASH_NAMESPACE",
    "PREPROCESSOR_PUBLICATION_FORMAT_VERSION",
    "SELECTION_FEATURES_FILENAME",
    "TRAINING_SAMPLE_IDS_NAMESPACE",
    "UNKNOWN_CATEGORY_ID",
    "fit_nf_cse_preprocessor",
    "load_nf_cse_preprocessor",
    "publish_nf_cse_train_only_preprocessor",
    "save_nf_cse_preprocessor",
]
