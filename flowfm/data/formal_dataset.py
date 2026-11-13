"""Train-only views and samples for sealed cross-dataset formal cohorts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow.parquet as pq

from flowfm.data.contracts import FlowSample
from flowfm.data.feature_keys import canonicalize_value
from flowfm.data.formal_cohort import (
    FORMAL_FEW_SHOT_NAMESPACE,
    FormalDatasetSpec,
    get_formal_dataset_spec,
    validate_formal_overlap_contract,
)
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.netflow_v2_protocol import (
    NetFlowV2IndexedRecord,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_unsw_split import _hash_file
from flowfm.data.schemaflow_registry import (
    DEFAULT_MAX_KNOWN_CATEGORIES,
    PORT_BUCKET_POLICY_VERSION,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
    build_port_vocabulary,
    fit_categorical_vocabulary,
)
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


FORMAL_PREPROCESSOR_FORMAT_VERSION = 2
FORMAL_PREPROCESSOR_HASH_NAMESPACES = MappingProxyType(
    {
        1: "flowfm-formal-preprocessor-v1",
        2: "flowfm-formal-preprocessor-v2",
    }
)
LEGACY_NUMERIC_SCALING_POLICY = "unbounded_train_minmax_v1"
NUMERIC_SCALING_POLICY = "train_minmax_clip_0_1_v1"
FORMAL_TRAINING_VIEW_VERSION = "cross-dataset-formal-training-view-v2"
MISSING_CATEGORY_ID = 0
UNKNOWN_CATEGORY_ID = 1
TRAIN_PERCENTS = (1, 5, 10)
_CATEGORY_CAPS = MappingProxyType(
    {
        "PROTOCOL": 256,
        "L7_PROTO": 512,
        "TCP_FLAGS": 512,
        "CLIENT_TCP_FLAGS": 512,
        "SERVER_TCP_FLAGS": 512,
        "ICMP_TYPE": 512,
        "ICMP_IPV4_TYPE": 512,
        "DNS_QUERY_TYPE": 512,
        "FTP_COMMAND_RET_CODE": 512,
    }
)


def _decimal_text(value: Decimal) -> str:
    return canonicalize_value(value, "numeric").removeprefix("n:")


def _numeric_names(spec: FormalDatasetSpec):
    return tuple(
        feature.name for feature in spec.schema.model_features
        if not feature.categorical
    )


def _categorical_names(spec: FormalDatasetSpec):
    return tuple(
        feature.name for feature in spec.schema.model_features
        if feature.categorical
    )


@dataclass(frozen=True)
class EncodedFormalFeatures:
    features: tuple[float | int, ...]
    feature_ids: tuple[str, ...]
    categorical_mask: tuple[bool, ...]
    missing_mask: tuple[bool, ...]
    category_tokens: tuple[str | None, ...]


@dataclass(frozen=True)
class FormalTrainOnlyPreprocessor:
    dataset_key: str
    dataset_id: str
    schema_version: str
    schema_hash: str
    training_row_count: int
    numeric_statistics: Mapping[str, Mapping[str, Any]]
    categorical_vocabularies: Mapping[str, tuple[str, ...]]
    format_version: int = FORMAL_PREPROCESSOR_FORMAT_VERSION
    numeric_scaling_policy: str = NUMERIC_SCALING_POLICY
    preprocessor_hash: str = ""
    _category_ids: Mapping[str, Mapping[str, int]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if (
            type(self.format_version) is not int
            or self.format_version not in FORMAL_PREPROCESSOR_HASH_NAMESPACES
        ):
            raise ValueError("formal preprocessor format changed")
        expected_scaling_policy = (
            LEGACY_NUMERIC_SCALING_POLICY
            if self.format_version == 1
            else NUMERIC_SCALING_POLICY
        )
        if (
            type(self.numeric_scaling_policy) is not str
            or self.numeric_scaling_policy != expected_scaling_policy
        ):
            raise ValueError("formal numeric scaling policy changed")
        spec = get_formal_dataset_spec(self.dataset_key)
        expected = (spec.dataset_id, spec.schema.version, spec.schema.schema_hash)
        if (self.dataset_id, self.schema_version, self.schema_hash) != expected:
            raise ValueError("formal preprocessor dataset identity changed")
        if type(self.training_row_count) is not int or self.training_row_count <= 0:
            raise ValueError("training_row_count must be positive")
        numeric_names = set(_numeric_names(spec))
        categorical_names = set(_categorical_names(spec))
        if set(self.numeric_statistics) != numeric_names:
            raise ValueError("numeric statistics do not match formal schema")
        if set(self.categorical_vocabularies) != categorical_names:
            raise ValueError("categorical vocabularies do not match formal schema")
        normalized_statistics = {}
        for name in sorted(numeric_names):
            values = self.numeric_statistics[name]
            required = {"observed_count", "missing_count", "min", "max", "mean"}
            if not isinstance(values, Mapping) or set(values) != required:
                raise ValueError("numeric statistics fields are invalid")
            observed = values["observed_count"]
            missing = values["missing_count"]
            if (
                type(observed) is not int
                or type(missing) is not int
                or observed < 0
                or missing < 0
                or observed + missing != self.training_row_count
            ):
                raise ValueError("numeric counts do not match training rows")
            if observed == 0:
                if (values["min"], values["max"], values["mean"]) != (
                    None,
                    None,
                    None,
                ):
                    raise ValueError("all-missing numeric statistics must be null")
            else:
                if not all(
                    isinstance(values[field], str) and values[field]
                    for field in ("min", "max", "mean")
                ):
                    raise ValueError("observed numeric statistics must be strings")
                if Decimal(values["min"]) > Decimal(values["max"]):
                    raise ValueError("numeric minimum exceeds maximum")
            normalized_statistics[name] = MappingProxyType(dict(values))
        normalized_vocabularies = {}
        category_ids = {}
        for name in sorted(categorical_names):
            values = tuple(self.categorical_vocabularies[name])
            if values != tuple(sorted(set(values))):
                raise ValueError("categorical vocabulary must be sorted and unique")
            if any(not isinstance(value, str) or not value or value == "m:" for value in values):
                raise ValueError("categorical vocabulary contains invalid tokens")
            normalized_vocabularies[name] = values
            category_ids[name] = MappingProxyType(
                {value: index for index, value in enumerate(values, start=2)}
            )
        object.__setattr__(
            self, "numeric_statistics", MappingProxyType(normalized_statistics)
        )
        object.__setattr__(
            self,
            "categorical_vocabularies",
            MappingProxyType(normalized_vocabularies),
        )
        object.__setattr__(self, "_category_ids", MappingProxyType(category_ids))
        expected_hash = stable_hash(
            self.to_dict(),
            FORMAL_PREPROCESSOR_HASH_NAMESPACES[self.format_version],
        )
        if self.preprocessor_hash and self.preprocessor_hash != expected_hash:
            raise ValueError("formal preprocessor hash changed")
        object.__setattr__(self, "preprocessor_hash", expected_hash)

    def to_dict(self):
        payload = {
            "format_version": self.format_version,
            "dataset_key": self.dataset_key,
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "training_row_count": self.training_row_count,
            "missing_category_id": MISSING_CATEGORY_ID,
            "unknown_category_id": UNKNOWN_CATEGORY_ID,
            "numeric_statistics": {
                name: dict(values)
                for name, values in sorted(self.numeric_statistics.items())
            },
            "categorical_vocabularies": {
                name: list(values)
                for name, values in sorted(self.categorical_vocabularies.items())
            },
        }
        if self.format_version >= 2:
            payload["numeric_scaling_policy"] = self.numeric_scaling_policy
        return payload

    @classmethod
    def from_dict(cls, payload, *, preprocessor_hash):
        base_required = {
            "format_version",
            "dataset_key",
            "dataset_id",
            "schema_version",
            "schema_hash",
            "training_row_count",
            "missing_category_id",
            "unknown_category_id",
            "numeric_statistics",
            "categorical_vocabularies",
        }
        if not isinstance(payload, Mapping):
            raise ValueError("formal preprocessor payload fields are invalid")
        format_version = payload.get("format_version")
        required = base_required | (
            {"numeric_scaling_policy"} if format_version == 2 else set()
        )
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("formal preprocessor payload fields are invalid")
        if (
            type(format_version) is not int
            or format_version not in FORMAL_PREPROCESSOR_HASH_NAMESPACES
            or payload["missing_category_id"] != MISSING_CATEGORY_ID
            or payload["unknown_category_id"] != UNKNOWN_CATEGORY_ID
        ):
            raise ValueError("formal preprocessor format changed")
        values = dict(payload)
        for key in ("missing_category_id", "unknown_category_id"):
            values.pop(key)
        if format_version == 1:
            values["numeric_scaling_policy"] = LEGACY_NUMERIC_SCALING_POLICY
        return cls(**values, preprocessor_hash=preprocessor_hash)

    def transform(self, row: Mapping[str, Any]) -> EncodedFormalFeatures:
        spec = get_formal_dataset_spec(self.dataset_key)
        features = []
        missing_mask = []
        category_tokens = []
        for feature in spec.schema.model_features:
            token = canonicalize_value(row.get(feature.name), feature.value_type)
            missing = token == "m:"
            missing_mask.append(missing)
            if feature.categorical:
                category_tokens.append(None if missing else token)
                features.append(
                    MISSING_CATEGORY_ID
                    if missing
                    else self._category_ids[feature.name].get(
                        token, UNKNOWN_CATEGORY_ID
                    )
                )
                continue
            category_tokens.append(None)
            statistics = self.numeric_statistics[feature.name]
            if missing or statistics["observed_count"] == 0:
                features.append(0.0)
                continue
            value = Decimal(token.removeprefix("n:"))
            minimum = Decimal(statistics["min"])
            maximum = Decimal(statistics["max"])
            if maximum == minimum:
                features.append(0.0)
                continue
            scaled = (value - minimum) / (maximum - minimum)
            if self.numeric_scaling_policy == NUMERIC_SCALING_POLICY:
                scaled = min(max(scaled, Decimal(0)), Decimal(1))
            features.append(float(scaled))
        return EncodedFormalFeatures(
            features=tuple(features),
            feature_ids=spec.schema.model_feature_names,
            categorical_mask=spec.schema.categorical_mask,
            missing_mask=tuple(missing_mask),
            category_tokens=tuple(category_tokens),
        )


def fit_formal_preprocessor(
    key: str, training_rows: Iterable[Mapping[str, Any]]
) -> FormalTrainOnlyPreprocessor:
    spec = get_formal_dataset_spec(key)
    numeric = {
        name: {
            "observed_count": 0,
            "missing_count": 0,
            "min": None,
            "max": None,
            "sum": Decimal(0),
        }
        for name in _numeric_names(spec)
    }
    vocabularies = {name: set() for name in _categorical_names(spec)}
    count = 0
    for row in training_rows:
        count += 1
        for feature in spec.schema.model_features:
            token = canonicalize_value(row.get(feature.name), feature.value_type)
            if feature.categorical:
                if token != "m:":
                    vocabularies[feature.name].add(token)
                continue
            accumulator = numeric[feature.name]
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
    if count <= 0:
        raise ValueError("formal preprocessor requires training rows")
    statistics = {}
    for name, accumulator in numeric.items():
        observed = accumulator["observed_count"]
        if observed:
            with localcontext() as context:
                context.prec = 50
                mean = accumulator["sum"] / Decimal(observed)
            minimum_text = _decimal_text(accumulator["min"])
            maximum_text = _decimal_text(accumulator["max"])
            mean_text = _decimal_text(mean)
        else:
            minimum_text = maximum_text = mean_text = None
        statistics[name] = {
            "observed_count": observed,
            "missing_count": accumulator["missing_count"],
            "min": minimum_text,
            "max": maximum_text,
            "mean": mean_text,
        }
    return FormalTrainOnlyPreprocessor(
        dataset_key=key,
        dataset_id=spec.dataset_id,
        schema_version=spec.schema.version,
        schema_hash=spec.schema.schema_hash,
        training_row_count=count,
        numeric_statistics=statistics,
        categorical_vocabularies={
            name: tuple(sorted(values)) for name, values in vocabularies.items()
        },
    )


def save_formal_preprocessor(path, preprocessor):
    return write_json_snapshot(
        path,
        {
            "preprocessor": preprocessor.to_dict(),
            "preprocessor_hash": preprocessor.preprocessor_hash,
        },
    )


def load_formal_preprocessor(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "preprocessor",
        "preprocessor_hash",
    }:
        raise ValueError("formal preprocessor file fields are invalid")
    return FormalTrainOnlyPreprocessor.from_dict(
        payload["preprocessor"], preprocessor_hash=payload["preprocessor_hash"]
    )


def _load_records(path):
    values = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            values.append(NetFlowV2IndexedRecord(**payload))
    return tuple(values)


def _skip_groups_exceeding_budget(records, max_samples):
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    group_sizes = Counter(record.group_id for record in records)
    oversized = {
        group_id: size
        for group_id, size in group_sizes.items()
        if size > max_samples
    }
    eligible = tuple(
        record for record in records if record.group_id not in oversized
    )
    if not eligible:
        raise ValueError("no whole leakage group fits the training budget")
    return eligible, {
        "policy": "skip_whole_group_before_group_safe_sampling",
        "max_samples": max_samples,
        "skipped_group_count": len(oversized),
        "skipped_row_count": sum(oversized.values()),
        "largest_skipped_group_rows": max(oversized.values(), default=0),
        "eligible_source_train_rows": len(eligible),
    }


def _feature_rows(path: Path, selected_ids: set[str]):
    found = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        for row in batch.to_pylist():
            sample_id = row.pop("sample_id")
            if sample_id not in selected_ids:
                continue
            if sample_id in found:
                raise ValueError("feature cache repeats a selected sample")
            found.add(sample_id)
            yield sample_id, row
    missing = selected_ids - found
    if missing:
        raise ValueError(f"feature cache is missing selected samples: {len(missing)}")


def publish_formal_training_view(
    key: str,
    cohort_dir,
    output_path,
    *,
    train_percent: int,
    data_root=Path("/root/autodl-tmp"),
):
    """Freeze one group-safe train percentage without reading sealed test files."""

    if train_percent not in TRAIN_PERCENTS:
        raise ValueError("train_percent must be one of 1, 5, 10")
    spec = get_formal_dataset_spec(key)
    cohort = validate_output_path(
        cohort_dir, enforce_autodl_data_disk=True, data_root=data_root
    )
    output = validate_output_path(
        output_path, enforce_autodl_data_disk=True, data_root=data_root
    )
    if output.exists():
        raise ValueError("formal training view output already exists")
    required = {
        "config": cohort / "config.snapshot.json",
        "records": cohort / "selection_records.jsonl",
        "manifest": cohort / "selection_manifest.json",
        "features": cohort / "selection_features.parquet",
        "overlap": cohort / "overlap_audit.json",
        "seal": cohort / "sealed_test.json",
        "data_manifest": cohort / "data_manifest.json",
    }
    if any(not path.is_file() for path in required.values()):
        raise ValueError("formal cohort selection artifacts are incomplete")
    if (cohort / ".incomplete").exists():
        raise ValueError("formal cohort is incomplete")
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    if config.get("dataset_key") != key or config.get("sealed_test") is not True:
        raise ValueError("formal cohort config does not match requested dataset")
    overlap = json.loads(required["overlap"].read_text(encoding="utf-8"))
    validate_formal_overlap_contract(spec, overlap)
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if manifest.get("test_members_present") is not False or set(
        manifest.get("splits", {})
    ) != {"train", "val"}:
        raise ValueError("selection manifest violates the sealed-test boundary")
    records = _load_records(required["records"])
    records_by_id = {record.sample_id: record for record in records}
    source_train_ids = tuple(manifest["splits"]["train"])
    source_val_ids = tuple(manifest["splits"]["val"])
    if set(records_by_id) != set(source_train_ids) | set(source_val_ids):
        raise ValueError("selection records do not match selection manifest")
    source_train = tuple(records_by_id[value] for value in source_train_ids)
    target_rows = max(
        len(spec.task_coarse_labels), len(source_train) * train_percent // 100
    )
    eligible_source_train, oversized_group_handling = (
        _skip_groups_exceeding_budget(source_train, target_rows)
    )
    available_native = {
        record.native_label for record in eligible_source_train
    }
    available_coarse = {
        spec.source_label_space.native_to_coarse[native]
        for native in available_native
    }
    missing_coarse = sorted(
        set(spec.task_coarse_labels) - available_coarse
    )
    if missing_coarse:
        raise ValueError(
            "whole-group training budget excludes required coarse labels: "
            + ", ".join(missing_coarse)
        )
    required_representatives = tuple(
        next(
            native
            for native in spec.task_native_labels
            if native in available_native
            and spec.source_label_space.native_to_coarse[native] == coarse
        )
        for coarse in spec.task_coarse_labels
    )
    selected_train = select_netflow_v2_proportional_development_records(
        eligible_source_train,
        max_samples=target_rows,
        seed=spec.split_seed,
        order_namespace=f"{FORMAL_FEW_SHOT_NAMESPACE}.{spec.dataset_id}.{train_percent}",
        required_native_labels=required_representatives,
    )
    if len(selected_train) < int(target_rows * 0.99):
        raise RuntimeError("formal few-shot view is more than 1% below target")
    train_ids = tuple(sorted(record.sample_id for record in selected_train))
    selected_ids = set(train_ids) | set(source_val_ids)
    view_records = tuple(records_by_id[value] for value in sorted(selected_ids))
    view_manifest = {
        "format_version": 1,
        "protocol_version": FORMAL_TRAINING_VIEW_VERSION,
        "dataset_key": key,
        "dataset_id": spec.dataset_id,
        "schema_version": spec.schema.version,
        "source_cohort": str(cohort),
        "source_selection_manifest_sha256": _hash_file(
            required["manifest"], "sha256"
        ),
        "data_split_seed": spec.split_seed,
        "train_percent": train_percent,
        "splits": {"train": list(train_ids), "val": list(source_val_ids)},
        "sample_groups": {
            sample_id: records_by_id[sample_id].group_id
            for sample_id in sorted(selected_ids)
        },
        "test_members_present": False,
    }
    train_id_set = set(train_ids)
    training_rows = [
        row for _, row in _feature_rows(required["features"], train_id_set)
    ]
    preprocessor = fit_formal_preprocessor(key, training_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    marker = output / ".incomplete"
    marker.write_text("formal training view incomplete\n", encoding="ascii")
    records_path = output / "selection_records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in view_records:
            stream.write(canonical_json(record.__dict__) + "\n")
    manifest_path = write_json_snapshot(
        output / "selection_manifest.json", view_manifest
    )
    preprocessor_path = save_formal_preprocessor(
        output / "preprocessor.json", preprocessor
    )
    summary = {
        "format_version": 1,
        "protocol_version": FORMAL_TRAINING_VIEW_VERSION,
        "status": "train_validation_view_published",
        "dataset_key": key,
        "dataset_id": spec.dataset_id,
        "task_label_space": spec.task_label_space.to_dict(),
        "train_percent": train_percent,
        "data_split_seed": spec.split_seed,
        "source_train_rows": len(source_train),
        "oversized_group_handling": oversized_group_handling,
        "selected_train_rows": len(selected_train),
        "selected_train_groups": len({r.group_id for r in selected_train}),
        "validation_rows": len(source_val_ids),
        "train_coarse_counts": dict(
            sorted(Counter(r.coarse_label for r in selected_train).items())
        ),
        "train_native_counts": dict(
            sorted(Counter(r.native_label for r in selected_train).items())
        ),
        "required_native_representatives": list(required_representatives),
        "preprocessing_fit_scope": "selected_target_train_only",
        "preprocessor_format_version": preprocessor.format_version,
        "numeric_scaling_policy": preprocessor.numeric_scaling_policy,
        "preprocessor_hash": preprocessor.preprocessor_hash,
        "preprocessor_sha256": _hash_file(preprocessor_path, "sha256"),
        "selection_records_sha256": _hash_file(records_path, "sha256"),
        "selection_manifest_sha256": _hash_file(manifest_path, "sha256"),
        "selection_features_sha256": _hash_file(required["features"], "sha256"),
        "source_overlap_audit_sha256": _hash_file(required["overlap"], "sha256"),
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent": spec.requires_zero_historical_overlap,
        "sealed_test_expected_sha256": _hash_file(required["seal"], "sha256"),
        "sealed_test_files_read": False,
        "test_labels_or_statistics_present": False,
    }
    write_json_snapshot(output / "training_view_summary.json", summary)
    marker.unlink()
    return summary


@dataclass(frozen=True)
class FormalSelectionSamples:
    dataset_key: str
    splits: Mapping[str, tuple[FlowSample, ...]]
    groups: Mapping[str, tuple[str, ...]]
    preprocessor: FormalTrainOnlyPreprocessor
    tokenizer_metadata: TokenizerMetadata
    label_space: Any


@dataclass(frozen=True)
class FormalSealedTestRows:
    dataset_key: str
    records: tuple[NetFlowV2IndexedRecord, ...]
    feature_rows: Mapping[str, Mapping[str, Any]]
    sample_order: tuple[str, ...]
    group_ids: tuple[str, ...]
    seal_sha256: str
    records_sha256: str
    manifest_sha256: str
    features_sha256: str


def _category_cap(name):
    return _CATEGORY_CAPS.get(name, DEFAULT_MAX_KNOWN_CATEGORIES)


def _build_tokenizer(
    spec,
    train_samples,
    preprocessor,
    data_manifest,
    split_manifest_sha256,
    training_ids_hash,
):
    categorical_specs = SCHEMAFLOW_MAIN_REGISTRY.categorical_model_features
    counts = {value.name: Counter() for value in categorical_specs}
    for sample in train_samples:
        positions = {name: index for index, name in enumerate(sample.feature_ids)}
        for feature in categorical_specs:
            index = positions[feature.name]
            if sample.missing_mask[index]:
                continue
            token = sample.category_tokens[index]
            if token not in preprocessor.categorical_vocabularies[feature.name]:
                raise ValueError("training token is absent from preprocessor vocabulary")
            counts[feature.name][token] += 1
    vocabularies = {}
    for feature in categorical_specs:
        arguments = {
            "training_row_count": len(train_samples),
            "training_sample_ids_hash": training_ids_hash,
        }
        if feature.value_policy == PORT_BUCKET_POLICY_VERSION:
            vocabulary = build_port_vocabulary(
                feature, counts[feature.name], **arguments
            )
        else:
            vocabulary = fit_categorical_vocabulary(
                feature,
                counts[feature.name],
                max_known_categories=_category_cap(feature.name),
                **arguments,
            )
        vocabularies[feature.name] = vocabulary
    return TokenizerMetadata(
        registry_version=SCHEMAFLOW_MAIN_REGISTRY.version,
        registry_hash=SCHEMAFLOW_MAIN_REGISTRY.registry_hash,
        schema_version=spec.schema.version,
        schema_hash=spec.schema.schema_hash,
        categorical_vocabularies=vocabularies,
        data_manifest=dict(data_manifest),
        split_manifest_sha256=split_manifest_sha256,
        preprocessor_hash=preprocessor.preprocessor_hash,
        training_sample_ids_hash=training_ids_hash,
    )


def load_formal_selection_samples(
    key: str,
    cohort_dir,
    view_dir,
    *,
    data_root=Path("/root/autodl-tmp"),
) -> FormalSelectionSamples:
    """Load only train/validation records and the non-test feature cache."""

    spec = get_formal_dataset_spec(key)
    cohort = validate_output_path(
        cohort_dir, enforce_autodl_data_disk=True, data_root=data_root
    )
    view = validate_output_path(
        view_dir, enforce_autodl_data_disk=True, data_root=data_root
    )
    required = {
        "records": view / "selection_records.jsonl",
        "manifest": view / "selection_manifest.json",
        "summary": view / "training_view_summary.json",
        "preprocessor": view / "preprocessor.json",
        "features": cohort / "selection_features.parquet",
        "data_manifest": cohort / "data_manifest.json",
    }
    if any(not path.is_file() for path in required.values()):
        raise ValueError("formal selection view is incomplete")
    if (view / ".incomplete").exists():
        raise ValueError("formal selection view is incomplete")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_key") != key
        or manifest.get("test_members_present") is not False
        or set(manifest.get("splits", {})) != {"train", "val"}
        or summary.get("sealed_test_files_read") is not False
    ):
        raise ValueError("formal selection view violates sealed scope")
    records = _load_records(required["records"])
    records_by_id = {record.sample_id: record for record in records}
    assigned = set(manifest["splits"]["train"]) | set(manifest["splits"]["val"])
    if assigned != set(records_by_id):
        raise ValueError("formal view records do not match manifest")
    preprocessor = load_formal_preprocessor(required["preprocessor"])
    rows_by_id = dict(_feature_rows(required["features"], assigned))
    feature_sizes = Counter(record.feature_key for record in records)
    samples_by_id = {}
    for sample_id, record in records_by_id.items():
        encoded = preprocessor.transform(rows_by_id[sample_id])
        samples_by_id[sample_id] = FlowSample(
            features=encoded.features,
            feature_ids=encoded.feature_ids,
            categorical_mask=encoded.categorical_mask,
            missing_mask=encoded.missing_mask,
            category_tokens=encoded.category_tokens,
            binary_label=record.binary_label,
            coarse_label=record.coarse_label,
            native_label=record.native_label,
            dataset_id=spec.dataset_id,
            sample_id=record.sample_id,
            feature_key=record.feature_key,
            conflict_flag=False,
            conflict_group_size=feature_sizes[record.feature_key],
            conflict_label_count=1,
            source_file=record.source_file,
            source_row_id=record.source_row_id,
        )
    splits = {
        name: tuple(samples_by_id[value] for value in manifest["splits"][name])
        for name in ("train", "val")
    }
    groups = {
        name: tuple(
            records_by_id[value].group_id for value in manifest["splits"][name]
        )
        for name in ("train", "val")
    }
    training_ids_hash = stable_hash(
        list(manifest["splits"]["train"]),
        f"flowfm-formal-training-sample-ids-v1.{spec.dataset_id}",
    )
    data_manifest = json.loads(required["data_manifest"].read_text(encoding="utf-8"))
    tokenizer = _build_tokenizer(
        spec,
        splits["train"],
        preprocessor,
        data_manifest,
        _hash_file(required["manifest"], "sha256"),
        training_ids_hash,
    )
    return FormalSelectionSamples(
        dataset_key=key,
        splits=MappingProxyType(splits),
        groups=MappingProxyType(groups),
        preprocessor=preprocessor,
        tokenizer_metadata=tokenizer,
        label_space=spec.task_label_space,
    )


def open_formal_sealed_test_rows(
    key: str,
    cohort_dir,
    *,
    data_root=Path("/root/autodl-tmp"),
) -> FormalSealedTestRows:
    """Open the three sealed test artifacts once for a globally locked matrix."""

    spec = get_formal_dataset_spec(key)
    cohort = validate_output_path(
        cohort_dir, enforce_autodl_data_disk=True, data_root=data_root
    )
    paths = {
        "seal": cohort / "sealed_test.json",
        "records": cohort / "sealed_test_records.jsonl",
        "manifest": cohort / "sealed_test_manifest.json",
        "features": cohort / "sealed_test_features.parquet",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("sealed formal test artifacts are incomplete")
    if (cohort / ".incomplete").exists():
        raise ValueError("sealed formal cohort is incomplete")
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    expected = {
        "sealed_test_records_sha256": _hash_file(paths["records"], "sha256"),
        "sealed_test_manifest_sha256": _hash_file(paths["manifest"], "sha256"),
        "sealed_test_features_sha256": _hash_file(paths["features"], "sha256"),
    }
    if (
        seal.get("status") != spec.seal_status
        or seal.get("selection_access_allowed") is not False
        or seal.get("opened_for_evaluation") is not False
        or seal.get("dataset_id") != spec.dataset_id
        or any(seal.get(field) != value for field, value in expected.items())
    ):
        raise ValueError("sealed formal test lock is invalid")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_id") != spec.dataset_id
        or manifest.get("split") != "test"
        or manifest.get("split_seed") != spec.split_seed
    ):
        raise ValueError("sealed formal test manifest changed")
    sample_order = tuple(manifest.get("sample_ids", ()))
    if not sample_order or len(set(sample_order)) != len(sample_order):
        raise ValueError("sealed formal test sample order is invalid")
    records = _load_records(paths["records"])
    records_by_id = {record.sample_id: record for record in records}
    if len(records_by_id) != len(records) or set(records_by_id) != set(sample_order):
        raise ValueError("sealed formal records do not match test manifest")
    feature_rows = dict(_feature_rows(paths["features"], set(sample_order)))
    ordered_records = tuple(records_by_id[sample_id] for sample_id in sample_order)
    return FormalSealedTestRows(
        dataset_key=key,
        records=ordered_records,
        feature_rows=MappingProxyType(feature_rows),
        sample_order=sample_order,
        group_ids=tuple(record.group_id for record in ordered_records),
        seal_sha256=_hash_file(paths["seal"], "sha256"),
        records_sha256=expected["sealed_test_records_sha256"],
        manifest_sha256=expected["sealed_test_manifest_sha256"],
        features_sha256=expected["sealed_test_features_sha256"],
    )


def materialize_formal_test_samples(
    rows: FormalSealedTestRows,
    preprocessor: FormalTrainOnlyPreprocessor,
) -> tuple[FlowSample, ...]:
    """Apply one locked train-only preprocessor to already opened test rows."""

    if not isinstance(rows, FormalSealedTestRows):
        raise TypeError("rows must be FormalSealedTestRows")
    if not isinstance(preprocessor, FormalTrainOnlyPreprocessor):
        raise TypeError("preprocessor must be FormalTrainOnlyPreprocessor")
    if rows.dataset_key != preprocessor.dataset_key:
        raise ValueError("test rows and preprocessor belong to different datasets")
    spec = get_formal_dataset_spec(rows.dataset_key)
    feature_sizes = Counter(record.feature_key for record in rows.records)
    samples = []
    for record in rows.records:
        encoded = preprocessor.transform(rows.feature_rows[record.sample_id])
        samples.append(
            FlowSample(
                features=encoded.features,
                feature_ids=encoded.feature_ids,
                categorical_mask=encoded.categorical_mask,
                missing_mask=encoded.missing_mask,
                category_tokens=encoded.category_tokens,
                binary_label=record.binary_label,
                coarse_label=record.coarse_label,
                native_label=record.native_label,
                dataset_id=spec.dataset_id,
                sample_id=record.sample_id,
                feature_key=record.feature_key,
                conflict_flag=False,
                conflict_group_size=feature_sizes[record.feature_key],
                conflict_label_count=1,
                source_file=record.source_file,
                source_row_id=record.source_row_id,
            )
        )
    return tuple(samples)


__all__ = [
    "FormalSelectionSamples",
    "FormalSealedTestRows",
    "FormalTrainOnlyPreprocessor",
    "TRAIN_PERCENTS",
    "fit_formal_preprocessor",
    "load_formal_preprocessor",
    "load_formal_selection_samples",
    "materialize_formal_test_samples",
    "open_formal_sealed_test_rows",
    "publish_formal_training_view",
    "save_formal_preprocessor",
]
