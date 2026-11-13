"""Strict, transactional NF-UNSW development split publication."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from flowfm.data.audit import load_audit_config
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.nf_unsw_protocol import (
    DEVELOPMENT_GROUP_ORDER_NAMESPACE,
    LEAKAGE_GROUP_NAMESPACE,
    index_nf_unsw_rows,
    select_nf_unsw_development_records,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
from flowfm.data.splits import deterministic_stratified_group_split
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


SPLIT_CONFIG_NAMESPACE = "flowfm-nf-unsw-development-split-config-v1"
SPLIT_SOURCE_NAMESPACE = "flowfm-nf-unsw-development-split-source-v1"
SPLIT_FORMAT_VERSION = 1
_CONFIG_FIELDS = frozenset(
    {
        "audit_config",
        "audit_report_dir",
        "view",
        "max_samples",
        "seed",
        "ratios",
    }
)
_SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_AUDIT_REPORTS = (
    "conflict_summary.json",
    "logical_views.json",
)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field} must be a non-empty string without surrounding whitespace"
        )
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for NF-UNSW split configuration") from exc

    class SplitLoader(yaml.SafeLoader):
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
            payload = yaml.load(stream, Loader=SplitLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid split YAML configuration: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("NF-UNSW split configuration must be a mapping")
    return payload


def _validated_ratios(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_SPLIT_NAMES):
        raise ValueError("ratios must contain exactly train, val, and test")
    ratios = {}
    for split_name in _SPLIT_NAMES:
        ratio = value[split_name]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("split ratios must be numeric and not boolean")
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError("split ratios must be finite and non-negative")
        ratios[split_name] = ratio
    if ratios["train"] <= 0:
        raise ValueError("the train ratio must be positive")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1")
    return ratios


def _hash_file(path: Path, algorithm: str) -> str:
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
        raise ValueError(f"{label} must contain a JSON object: {path}")
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


@dataclass(frozen=True)
class NFUNSWDevelopmentSplitConfig:
    audit_config_path: Path
    audit_report_dir: Path
    view: str
    max_samples: int
    seed: int
    ratios: dict[str, float]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_config": str(self.audit_config_path),
            "audit_report_dir": str(self.audit_report_dir),
            "max_samples": self.max_samples,
            "ratios": dict(self.ratios),
            "seed": self.seed,
            "view": self.view,
        }


def load_nf_unsw_development_split_config(path) -> NFUNSWDevelopmentSplitConfig:
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
        raise ValueError(f"audit_config must reference an existing file: {audit_config_path}")
    audit_report_dir = Path(
        _require_text(payload["audit_report_dir"], "audit_report_dir")
    ).expanduser().resolve()
    if not audit_report_dir.is_dir():
        raise ValueError(
            f"audit_report_dir must reference an existing directory: {audit_report_dir}"
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
    ratios = _validated_ratios(payload["ratios"])

    normalized = {
        "audit_config": str(audit_config_path),
        "audit_report_dir": str(audit_report_dir),
        "max_samples": max_samples,
        "ratios": ratios,
        "seed": seed,
        "view": view,
    }
    return NFUNSWDevelopmentSplitConfig(
        audit_config_path=audit_config_path,
        audit_report_dir=audit_report_dir,
        view=view,
        max_samples=max_samples,
        seed=seed,
        ratios=ratios,
        config_hash=stable_hash(normalized, SPLIT_CONFIG_NAMESPACE),
    )


def _validated_audit_state(config: NFUNSWDevelopmentSplitConfig):
    audit_config = load_audit_config(config.audit_config_path)
    summary_path = config.audit_report_dir / "conflict_summary.json"
    views_path = config.audit_report_dir / "logical_views.json"
    summary = _load_json_object(summary_path, "conflict summary")
    views = _load_json_object(views_path, "logical views")

    expected = {
        "dataset_id": audit_config.dataset_id,
        "schema_version": audit_config.schema_version,
        "schema_hash": NF_UNSW_NB15_V2_SCHEMA.schema_hash,
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
        "audit_summary": summary,
        "audit_summary_sha256": _hash_file(summary_path, "sha256"),
        "conflict_feature_keys": conflict_keys,
        "expected_view_rows": expected_view_rows,
        "raw_csv_sha1": raw_sha1,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _raw_rows(audit_config):
    expected_header = (
        *NF_UNSW_NB15_V2_SCHEMA.raw_feature_names,
        audit_config.binary_label_column,
        audit_config.native_label_column,
    )
    with audit_config.raw_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header does not match the registered schema")
        for row_number, row in enumerate(reader, start=1):
            yield str(row_number), row


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _label_counts(records, field: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(record, field) for record in records).items()))


def publish_nf_unsw_development_split(
    config_path,
    output_path,
    *,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Publish a deterministic conflict-free development index and split."""

    config = load_nf_unsw_development_split_config(config_path)
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
    marker.write_text("NF-UNSW split publication incomplete\n", encoding="utf-8")

    audit_config = audit_state["audit_config"]
    indexed = index_nf_unsw_rows(
        _raw_rows(audit_config),
        conflict_feature_keys=audit_state["conflict_feature_keys"],
        source_file_id=audit_config.source_file_id,
        coarse_label_map=audit_config.coarse_label_map,
        sensitive_hash_namespace=audit_config.sensitive_hash_namespace,
    )
    if len(indexed) != audit_state["expected_view_rows"]:
        raise RuntimeError(
            "indexed conflict-free row count does not match the audited logical view"
        )
    selected = select_nf_unsw_development_records(
        indexed,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    source_hash = stable_hash(
        {
            "audit_config_hash": audit_config.config_hash,
            "audit_summary_sha256": audit_state["audit_summary_sha256"],
            "raw_csv_sha1": audit_state["raw_csv_sha1"],
            "schema_hash": NF_UNSW_NB15_V2_SCHEMA.schema_hash,
            "source_manifest_sha256": audit_state["source_manifest_sha256"],
            "view": config.view,
        },
        SPLIT_SOURCE_NAMESPACE,
    )
    manifest = deterministic_stratified_group_split(
        (record.to_split_record() for record in selected),
        ratios=config.ratios,
        seed=config.seed,
        schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
        source_hash=source_hash,
    )

    config_snapshot = output_dir / "config.snapshot.json"
    selected_path = output_dir / "selected_records.jsonl"
    manifest_path = output_dir / "split_manifest.json"
    summary_path = output_dir / "split_summary.json"
    write_json_snapshot(config_snapshot, config.to_dict())
    selected_text = "".join(
        canonical_json(asdict(record)) + "\n" for record in selected
    )
    _write_text(selected_path, selected_text)
    _write_text(manifest_path, manifest.to_json() + "\n")

    records_by_sample = {record.sample_id: record for record in selected}
    split_counts = {}
    for split_name in _SPLIT_NAMES:
        members = [records_by_sample[sample_id] for sample_id in manifest.splits[split_name]]
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
        "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
        "schema_hash": NF_UNSW_NB15_V2_SCHEMA.schema_hash,
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
    "NFUNSWDevelopmentSplitConfig",
    "SPLIT_CONFIG_NAMESPACE",
    "SPLIT_FORMAT_VERSION",
    "SPLIT_SOURCE_NAMESPACE",
    "load_nf_unsw_development_split_config",
    "publish_nf_unsw_development_split",
]
