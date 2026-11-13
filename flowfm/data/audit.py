"""Read-only, reproducible conflict audit for unified NetFlow CSV data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from flowfm.data.feature_keys import FEATURE_KEY_NAMESPACE, compute_feature_key
from flowfm.data.identity import make_sample_id, stable_hash
from flowfm.data.schemas.registry import get_dataset_schema
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


AUDIT_CONFIG_NAMESPACE = "flowfm-audit-config-v1"
AUDIT_FORMAT_VERSION = 1
_CONFIG_FIELDS = frozenset(
    {
        "dataset_id",
        "raw_csv",
        "source_manifest",
        "source_file_id",
        "schema_version",
        "feature_key_namespace",
        "binary_label_column",
        "native_label_column",
        "coarse_label_mapping_version",
        "coarse_label_map",
        "sensitive_hash_namespace",
        "max_examples",
    }
)
_LABEL_FIELDS = ("binary_label", "coarse_label", "native_label")
_PARQUET_BATCH_SIZE = 10000


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _require_data_disk_file(value: Any, field: str) -> Path:
    path = validate_output_path(
        _require_text(value, field),
        enforce_autodl_data_disk=True,
    )
    if not path.is_file():
        raise ValueError(f"{field} must reference an existing file: {path}")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for conflict audit configuration") from exc

    class AuditLoader(yaml.SafeLoader):
        def compose_node(self, parent, index):
            if self.check_event(yaml.AliasEvent):
                event = self.get_event()
                raise ValueError(f"YAML aliases are not allowed: *{event.anchor}")
            return super().compose_node(parent, index)

        def construct_mapping(self, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("YAML mapping keys must be strings")
                if key in seen:
                    raise ValueError(f"duplicate YAML key: {key}")
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=AuditLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid audit YAML configuration: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("audit configuration must be a mapping")
    return payload


@dataclass(frozen=True)
class AuditConfig:
    dataset_id: str
    schema: Any
    raw_csv: Path
    source_manifest: Path
    source_file_id: str
    schema_version: str
    feature_key_namespace: str
    binary_label_column: str
    native_label_column: str
    coarse_label_mapping_version: str
    coarse_label_map: dict[str, str]
    sensitive_hash_namespace: str
    max_examples: int
    config_hash: str


def load_audit_config(path) -> AuditConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"audit config must be an existing file: {config_path}")
    payload = _load_yaml(config_path)
    fields = set(payload)
    if fields != _CONFIG_FIELDS:
        missing = sorted(_CONFIG_FIELDS - fields)
        extra = sorted(fields - _CONFIG_FIELDS)
        raise ValueError(
            f"invalid audit config fields; missing={missing}, extra={extra}"
        )

    dataset_id = _require_text(payload["dataset_id"], "dataset_id")
    schema_version = _require_text(payload["schema_version"], "schema_version")
    schema = get_dataset_schema(dataset_id, schema_version)
    feature_key_namespace = _require_text(
        payload["feature_key_namespace"], "feature_key_namespace"
    )
    if feature_key_namespace != FEATURE_KEY_NAMESPACE:
        raise ValueError("feature_key_namespace does not match the implementation")

    mapping = payload["coarse_label_map"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("coarse_label_map must be a non-empty mapping")
    normalized_mapping = {}
    for native_label, coarse_label in mapping.items():
        normalized_mapping[
            _require_text(native_label, "coarse_label_map key")
        ] = _require_text(coarse_label, "coarse_label_map value")

    max_examples = payload["max_examples"]
    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a non-negative integer")

    normalized_for_hash = {
        **payload,
        "raw_csv": str(Path(payload["raw_csv"]).expanduser().resolve()),
        "source_manifest": str(
            Path(payload["source_manifest"]).expanduser().resolve()
        ),
        "coarse_label_map": dict(sorted(normalized_mapping.items())),
    }
    return AuditConfig(
        dataset_id=dataset_id,
        schema=schema,
        raw_csv=_require_data_disk_file(payload["raw_csv"], "raw_csv"),
        source_manifest=_require_data_disk_file(
            payload["source_manifest"], "source_manifest"
        ),
        source_file_id=_require_text(payload["source_file_id"], "source_file_id"),
        schema_version=schema_version,
        feature_key_namespace=feature_key_namespace,
        binary_label_column=_require_text(
            payload["binary_label_column"], "binary_label_column"
        ),
        native_label_column=_require_text(
            payload["native_label_column"], "native_label_column"
        ),
        coarse_label_mapping_version=_require_text(
            payload["coarse_label_mapping_version"],
            "coarse_label_mapping_version",
        ),
        coarse_label_map=normalized_mapping,
        sensitive_hash_namespace=_require_text(
            payload["sensitive_hash_namespace"], "sensitive_hash_namespace"
        ),
        max_examples=max_examples,
        config_hash=stable_hash(normalized_for_hash, AUDIT_CONFIG_NAMESPACE),
    )


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_manifest(config: AuditConfig) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(config.source_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid source manifest: {config.source_manifest}") from exc
    if not isinstance(payload, dict) or payload.get("dataset_id") != config.dataset_id:
        raise ValueError("source manifest dataset_id does not match audit config")
    try:
        archive_sha256 = _require_text(payload["archive"]["sha256"], "manifest archive sha256")
        csv_manifest = payload["bag"]["csv"]
        expected_sha1 = _require_text(csv_manifest["sha1"], "manifest csv sha1")
        expected_rows = int(csv_manifest["rows_excluding_header"])
        expected_columns = int(csv_manifest["columns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source manifest is missing valid bag.csv metadata") from exc
    payload["_validated_csv"] = {
        "sha1": expected_sha1,
        "rows": expected_rows,
        "columns": expected_columns,
    }
    payload["_validated_archive"] = {"sha256": archive_sha256}
    return payload, _hash_file(config.source_manifest, "sha256")


class _LabelState:
    __slots__ = ("first", "first_count", "counts")

    def __init__(self, label: str):
        self.first = label
        self.first_count = 1
        self.counts = None

    def add(self, label: str) -> None:
        if self.counts is None:
            if label == self.first:
                self.first_count += 1
                return
            self.counts = {self.first: self.first_count, label: 1}
            return
        self.counts[label] = self.counts.get(label, 0) + 1

    def as_counts(self) -> dict[str, int]:
        if self.counts is None:
            return {self.first: self.first_count}
        return dict(sorted(self.counts.items()))

    @property
    def label_count(self) -> int:
        return 1 if self.counts is None else len(self.counts)

    @property
    def majority_count(self) -> int:
        if self.counts is None:
            return self.first_count
        return max(self.counts.values())


class _GroupState:
    __slots__ = ("size", "binary", "coarse", "native")

    def __init__(self, binary: str, coarse: str, native: str):
        self.size = 1
        self.binary = _LabelState(binary)
        self.coarse = _LabelState(coarse)
        self.native = _LabelState(native)

    def add(self, binary: str, coarse: str, native: str) -> None:
        self.size += 1
        self.binary.add(binary)
        self.coarse.add(coarse)
        self.native.add(native)

    def label_state(self, field: str) -> _LabelState:
        return getattr(self, field.removesuffix("_label"))

    @property
    def conflict_fields(self) -> tuple[str, ...]:
        return tuple(
            field
            for field in _LABEL_FIELDS
            if self.label_state(field).label_count > 1
        )

    @property
    def conflict_flag(self) -> bool:
        return bool(self.conflict_fields)


def _labels(row: Mapping[str, str], config: AuditConfig):
    binary = row[config.binary_label_column].strip()
    native = row[config.native_label_column].strip()
    if not binary or not native:
        raise ValueError("audit label columns must not contain empty values")
    try:
        coarse = config.coarse_label_map[native]
    except KeyError as exc:
        raise ValueError(f"native label is missing from coarse_label_map: {native}") from exc
    return binary, coarse, native


def _expected_header(config: AuditConfig) -> tuple[str, ...]:
    return (
        *config.schema.raw_feature_names,
        config.binary_label_column,
        config.native_label_column,
    )


def _first_pass(config: AuditConfig):
    groups: dict[str, _GroupState] = {}
    row_count = 0
    with config.raw_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _expected_header(config):
            raise ValueError("raw CSV header does not match the registered schema")
        for row_count, row in enumerate(reader, start=1):
            feature_key = compute_feature_key(row, config.schema)
            binary, coarse, native = _labels(row, config)
            state = groups.get(feature_key)
            if state is None:
                groups[feature_key] = _GroupState(binary, coarse, native)
            else:
                state.add(binary, coarse, native)
    return groups, row_count, len(reader.fieldnames or ())


def _pair_counts(groups: Mapping[str, _GroupState], field: str):
    totals: dict[str, dict[str, int]] = {}
    for state in groups.values():
        counts = state.label_state(field).as_counts()
        for first, second in combinations(sorted(counts), 2):
            key = f"{first} | {second}"
            entry = totals.setdefault(
                key, {"group_count": 0, "cross_label_row_pairs": 0}
            )
            entry["group_count"] += 1
            entry["cross_label_row_pairs"] += counts[first] * counts[second]
    return dict(sorted(totals.items()))


def _upper_bound(groups: Mapping[str, _GroupState], field: str):
    total_rows = sum(state.size for state in groups.values())
    majority = sum(
        state.label_state(field).majority_count for state in groups.values()
    )
    conflicting = tuple(
        state
        for state in groups.values()
        if state.label_state(field).label_count > 1
    )
    conflict_rows = sum(state.size for state in conflicting)
    conflict_majority = sum(
        state.label_state(field).majority_count for state in conflicting
    )
    return {
        "majority_correct_rows": majority,
        "irreducible_error_rows": total_rows - majority,
        "accuracy_upper_bound": majority / total_rows if total_rows else 0.0,
        "conflict_only_majority_correct_rows": conflict_majority,
        "conflict_only_rows": conflict_rows,
        "conflict_only_accuracy_upper_bound": (
            conflict_majority / conflict_rows if conflict_rows else 1.0
        ),
    }


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


def _privacy_hash(config: AuditConfig, suffix: str, payload: Any) -> str:
    return stable_hash(
        payload,
        f"{config.sensitive_hash_namespace}.{suffix}",
    )


def _row_metadata(row: Mapping[str, str], config: AuditConfig):
    source = row["IPV4_SRC_ADDR"].strip()
    destination = row["IPV4_DST_ADDR"].strip()
    host_pair = sorted((source, destination))
    host_pair_hash = _privacy_hash(config, "host-pair", host_pair)
    five_tuple_hash = _privacy_hash(
        config,
        "five-tuple",
        {
            "src": source,
            "src_port": row["L4_SRC_PORT"].strip(),
            "dst": destination,
            "dst_port": row["L4_DST_PORT"].strip(),
            "protocol": row["PROTOCOL"].strip(),
        },
    )
    return host_pair_hash, five_tuple_hash


def _parquet_schema(pa):
    return pa.schema(
        [
            ("feature_key", pa.string()),
            ("sample_id", pa.string()),
            ("source_file", pa.string()),
            ("source_row_id", pa.string()),
            ("binary_label", pa.string()),
            ("coarse_label", pa.string()),
            ("native_label", pa.string()),
            ("conflict_flag", pa.bool_()),
            ("conflict_group_size", pa.int64()),
            ("conflict_label_count", pa.int64()),
            ("conflict_fields", pa.list_(pa.string())),
            ("host_pair_hash", pa.string()),
            ("five_tuple_hash", pa.string()),
            ("native_majority_count", pa.int64()),
            ("native_accuracy_upper_bound", pa.float64()),
        ]
    )


def _write_parquet_batch(writer, pa, rows, schema):
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def _second_pass(config: AuditConfig, groups, output_dir: Path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("PyArrow is required to publish conflict Parquet") from exc

    conflict_keys = {
        feature_key for feature_key, state in groups.items() if state.conflict_flag
    }
    schema = _parquet_schema(pa)
    writer = pq.ParquetWriter(
        output_dir / "conflict_groups.parquet",
        schema,
        compression="zstd",
    )
    batch = []
    examples = {}
    source_counts = Counter()
    host_pair_counts = Counter()
    five_tuple_counts = Counter()
    try:
        with config.raw_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row_number, row in enumerate(reader, start=1):
                feature_key = compute_feature_key(row, config.schema)
                if feature_key not in conflict_keys:
                    continue
                state = groups[feature_key]
                binary, coarse, native = _labels(row, config)
                host_pair_hash, five_tuple_hash = _row_metadata(row, config)
                sample_id = make_sample_id(
                    {
                        "source_file": config.source_file_id,
                        "source_row_id": str(row_number),
                    },
                    config.dataset_id,
                    config.schema_version,
                )
                native_majority = state.native.majority_count
                published = {
                    "feature_key": feature_key,
                    "sample_id": sample_id,
                    "source_file": config.source_file_id,
                    "source_row_id": str(row_number),
                    "binary_label": binary,
                    "coarse_label": coarse,
                    "native_label": native,
                    "conflict_flag": True,
                    "conflict_group_size": state.size,
                    "conflict_label_count": state.native.label_count,
                    "conflict_fields": list(state.conflict_fields),
                    "host_pair_hash": host_pair_hash,
                    "five_tuple_hash": five_tuple_hash,
                    "native_majority_count": native_majority,
                    "native_accuracy_upper_bound": native_majority / state.size,
                }
                batch.append(published)
                if feature_key not in examples:
                    examples[feature_key] = {
                        **published,
                        "label_counts_json": json.dumps(
                            {
                                field: state.label_state(field).as_counts()
                                for field in _LABEL_FIELDS
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                source_counts[config.source_file_id] += 1
                host_pair_counts[host_pair_hash] += 1
                five_tuple_counts[five_tuple_hash] += 1
                if len(batch) >= _PARQUET_BATCH_SIZE:
                    _write_parquet_batch(writer, pa, batch, schema)
        _write_parquet_batch(writer, pa, batch, schema)
    finally:
        writer.close()
    selected_examples = [examples[key] for key in sorted(examples)[: config.max_examples]]
    return {
        "examples": selected_examples,
        "source_file_distribution": dict(sorted(source_counts.items())),
        "host_pair_distribution": dict(sorted(host_pair_counts.items())),
        "five_tuple_distribution": dict(sorted(five_tuple_counts.items())),
        "conflict_feature_keys": sorted(conflict_keys),
    }


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _write_label_pairs(path: Path, pair_counts) -> None:
    rows = []
    for field in _LABEL_FIELDS:
        for pair, counts in pair_counts[field].items():
            first, second = pair.split(" | ", 1)
            rows.append(
                {
                    "label_field": field,
                    "label_a": first,
                    "label_b": second,
                    **counts,
                }
            )
    _write_csv(
        path,
        (
            "label_field",
            "label_a",
            "label_b",
            "group_count",
            "cross_label_row_pairs",
        ),
        rows,
    )


def _write_examples(path: Path, examples) -> None:
    fields = (
        "feature_key",
        "source_file",
        "source_row_id",
        "binary_label",
        "coarse_label",
        "native_label",
        "conflict_group_size",
        "conflict_label_count",
        "conflict_fields",
        "host_pair_hash",
        "five_tuple_hash",
        "native_accuracy_upper_bound",
        "label_counts_json",
    )
    rows = []
    for example in examples:
        rows.append(
            {
                field: (
                    "|".join(example[field])
                    if field == "conflict_fields"
                    else example[field]
                )
                for field in fields
            }
        )
    _write_csv(path, fields, rows)


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    bounds = summary["deterministic_classifier_upper_bounds"]
    lines = [
        f"# Data quality report: {summary['dataset_id']}",
        "",
        "## Conflict audit",
        "",
        f"- Total rows (N): {summary['total_rows']}",
        f"- Unique feature keys (U): {summary['unique_feature_keys']}",
        f"- Conflict feature keys (K): {summary['conflict_feature_keys']}",
        f"- Conflict rows (R): {summary['conflict_rows']}",
        f"- Duplicate excess rows (E=N-U): {summary['duplicate_excess_rows']}",
        f"- Conflict row ratio: {summary['conflict_row_ratio']:.8f}",
        "",
        "## Deterministic-classifier upper bounds",
        "",
        "| Label field | All-row upper bound | Conflict-only upper bound | Irreducible rows |",
        "|---|---:|---:|---:|",
    ]
    for field in _LABEL_FIELDS:
        value = bounds[field]
        lines.append(
            "| {} | {:.8f} | {:.8f} | {} |".format(
                field,
                value["accuracy_upper_bound"],
                value["conflict_only_accuracy_upper_bound"],
                value["irreducible_error_rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- `raw_full`: preserves all original records and labels.",
            "- `conflict_free_core`: excludes every row in every conflicting feature-key group.",
            "- `conflict_quarantine`: contains every row in every conflicting feature-key group.",
            "- `raw_grouped`: sensitivity protocol; each feature key must remain in one split.",
            "- No labels were changed and the raw CSV hash was unchanged.",
            "- Raw IP addresses and host names are not published in these reports.",
            "",
        ]
    )
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())


def _environment_versions() -> dict[str, Any]:
    import pyarrow
    import yaml

    try:
        import torch
    except ImportError:
        torch_version = None
        cuda_runtime = None
        cuda_available = None
    else:
        torch_version = str(torch.__version__)
        cuda_runtime = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pyyaml_version": str(yaml.__version__),
        "pyarrow_version": str(pyarrow.__version__),
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_available": cuda_available,
    }


def run_conflict_audit(config_path, output_path, repo_dir=None):
    """Run a two-pass read-only audit and publish reports transactionally."""

    config = load_audit_config(config_path)
    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
    )
    if output_dir.exists():
        raise ValueError(f"audit output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text("conflict audit publication incomplete\n", encoding="utf-8")

    source_manifest, source_manifest_hash = _load_source_manifest(config)
    expected = source_manifest["_validated_csv"]
    raw_sha1_before = _hash_file(config.raw_csv, "sha1")
    if raw_sha1_before != expected["sha1"]:
        raise ValueError("raw CSV SHA1 does not match the source manifest")

    groups, row_count, column_count = _first_pass(config)
    if row_count != expected["rows"]:
        raise ValueError("raw CSV row count does not match the source manifest")
    if column_count != expected["columns"]:
        raise ValueError("raw CSV column count does not match the source manifest")

    conflict_groups = {
        key: state for key, state in groups.items() if state.conflict_flag
    }
    conflict_rows = sum(state.size for state in conflict_groups.values())
    group_size_distribution = dict(
        sorted(Counter(state.size for state in groups.values()).items())
    )
    pair_counts = {
        field: _pair_counts(groups, field) for field in _LABEL_FIELDS
    }
    second_pass = _second_pass(config, groups, output_dir)
    raw_sha1_after = _hash_file(config.raw_csv, "sha1")
    if raw_sha1_after != raw_sha1_before:
        raise RuntimeError("raw CSV changed during read-only conflict audit")

    binary_conflicts = sum(
        state.binary.label_count > 1 for state in groups.values()
    )
    attack_family_conflicts = sum(
        state.native.label_count > 1
        and set(label.casefold() for label in state.binary.as_counts()).issubset(
            {"1", "attack", "malicious"}
        )
        for state in groups.values()
    )
    summary = {
        "format_version": AUDIT_FORMAT_VERSION,
        "dataset_id": config.dataset_id,
        "total_rows": row_count,
        "unique_feature_keys": len(groups),
        "duplicate_excess_rows": row_count - len(groups),
        "conflict_feature_keys": len(conflict_groups),
        "conflict_rows": conflict_rows,
        "conflict_row_ratio": conflict_rows / row_count if row_count else 0.0,
        "group_size_distribution": {
            str(size): count for size, count in group_size_distribution.items()
        },
        "label_pair_counts": pair_counts,
        "benign_attack_conflict_groups": binary_conflicts,
        "attack_family_conflict_groups": attack_family_conflicts,
        "deterministic_classifier_upper_bounds": {
            field: _upper_bound(groups, field) for field in _LABEL_FIELDS
        },
        "source_file_distribution": second_pass["source_file_distribution"],
        "host_pair_distribution": second_pass["host_pair_distribution"],
        "five_tuple_distribution": second_pass["five_tuple_distribution"],
        "time_distribution": {
            "available": False,
            "reason": f"{config.dataset_id} model schema has no timestamp field",
        },
        "raw_sha1_before": raw_sha1_before,
        "raw_sha1_after": raw_sha1_after,
        "source_manifest_sha256": source_manifest_hash,
        "raw_archive_sha256_manifest": source_manifest["_validated_archive"]["sha256"],
        "schema_version": config.schema_version,
        "schema_hash": config.schema.schema_hash,
        "feature_key_namespace": config.feature_key_namespace,
        "coarse_label_mapping_version": config.coarse_label_mapping_version,
        "config_hash": config.config_hash,
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
        "conflict_policy": "quarantine_entire_feature_key_group",
        "raw_mutated": False,
    }
    summary.update(_environment_versions())
    write_json_snapshot(output_dir / "conflict_summary.json", summary)
    _write_label_pairs(output_dir / "conflict_label_pairs.csv", pair_counts)
    _write_examples(output_dir / "conflict_examples.csv", second_pass["examples"])
    logical_views = {
        "dataset_id": config.dataset_id,
        "schema_version": config.schema_version,
        "feature_key_namespace": config.feature_key_namespace,
        "raw_full": {"rows": row_count, "policy": "include_all"},
        "conflict_free_core": {
            "rows": row_count - conflict_rows,
            "policy": "exclude_entire_conflicting_feature_key_groups",
        },
        "conflict_quarantine": {
            "rows": conflict_rows,
            "feature_keys": second_pass["conflict_feature_keys"],
            "policy": "include_entire_conflicting_feature_key_groups",
        },
    }
    write_json_snapshot(output_dir / "logical_views.json", logical_views)
    _write_markdown(output_dir / "data_quality_report.md", summary)
    marker.unlink()
    return {
        "dataset_id": config.dataset_id,
        "output": str(output_dir),
        "conflict_feature_keys": len(conflict_groups),
        "conflict_rows": conflict_rows,
        "config_hash": config.config_hash,
    }


__all__ = ["AuditConfig", "load_audit_config", "run_conflict_audit"]
