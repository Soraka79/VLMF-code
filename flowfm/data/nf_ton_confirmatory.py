"""Independent, group-disjoint NF-ToN confirmatory cohort artifacts."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flowfm.data.contracts import FlowSample
from flowfm.data.feature_keys import compute_feature_key
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.netflow_v2_protocol import (
    index_netflow_v2_rows,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_ton_dataset import _load_selected_records
from flowfm.data.nf_ton_preprocessing import load_nf_ton_preprocessor
from flowfm.data.nf_ton_split import (
    LEAKAGE_GROUP_NAMESPACE,
    _raw_rows,
    _validated_audit_state,
)
from flowfm.data.nf_unsw_split import (
    _git_commit,
    _hash_file,
    _label_counts,
    _load_json_object,
    _load_yaml,
    _require_text,
)
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


COHORT_CONFIG_NAMESPACE = "flowfm-nf-ton-iot-v2-confirmatory-config-v1"
COHORT_ORDER_NAMESPACE = "flowfm.nf-ton-iot-v2-confirmatory-group-order.v1"
COHORT_FORMAT_VERSION = 1
_CONFIG_FIELDS = {
    "audit_config",
    "audit_report_dir",
    "exclusion_split_dirs",
    "max_samples",
    "seed",
    "selection_policy",
    "view",
}
_REQUIRED_FILES = (
    "cohort_manifest.json",
    "config.snapshot.json",
    "data_manifest.json",
    "overlap_audit.json",
    "selected_records.jsonl",
)


@dataclass(frozen=True)
class NFTONConfirmatoryConfig:
    audit_config_path: Path
    audit_report_dir: Path
    exclusion_split_dirs: tuple[Path, ...]
    view: str
    max_samples: int
    seed: int
    selection_policy: str
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_config": str(self.audit_config_path),
            "audit_report_dir": str(self.audit_report_dir),
            "exclusion_split_dirs": [
                str(path) for path in self.exclusion_split_dirs
            ],
            "max_samples": self.max_samples,
            "seed": self.seed,
            "selection_policy": self.selection_policy,
            "view": self.view,
        }


@dataclass(frozen=True)
class NFTONConfirmatorySamples:
    samples: tuple[FlowSample, ...]
    group_ids: tuple[str, ...]
    cohort_manifest_sha256: str
    preprocessor_hash: str
    raw_csv_sha1: str

    def __post_init__(self) -> None:
        if not self.samples or len(self.samples) != len(self.group_ids):
            raise ValueError("confirmatory samples and group IDs must align")
        if any(not isinstance(value, str) or not value for value in self.group_ids):
            raise ValueError("confirmatory group IDs must be non-empty strings")
        for field in (
            "cohort_manifest_sha256",
            "preprocessor_hash",
            "raw_csv_sha1",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty string")


def load_nf_ton_confirmatory_config(path) -> NFTONConfirmatoryConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"cohort config must be an existing file: {config_path}")
    payload = _load_yaml(config_path)
    if set(payload) != _CONFIG_FIELDS:
        raise ValueError(
            "invalid confirmatory config fields; missing={}, extra={}".format(
                sorted(_CONFIG_FIELDS - set(payload)),
                sorted(set(payload) - _CONFIG_FIELDS),
            )
        )
    audit_config_path = Path(
        _require_text(payload["audit_config"], "audit_config")
    ).expanduser().resolve()
    audit_report_dir = Path(
        _require_text(payload["audit_report_dir"], "audit_report_dir")
    ).expanduser().resolve()
    if not audit_config_path.is_file():
        raise ValueError("audit_config must reference an existing file")
    if not audit_report_dir.is_dir():
        raise ValueError("audit_report_dir must reference an existing directory")
    raw_exclusions = payload["exclusion_split_dirs"]
    if not isinstance(raw_exclusions, list) or not raw_exclusions:
        raise ValueError("exclusion_split_dirs must be a non-empty list")
    exclusions = tuple(
        Path(_require_text(value, "exclusion split directory"))
        .expanduser()
        .resolve()
        for value in raw_exclusions
    )
    if len(set(exclusions)) != len(exclusions):
        raise ValueError("exclusion_split_dirs must be unique")
    if any(not path.is_dir() for path in exclusions):
        raise ValueError("every exclusion split directory must exist")
    view = _require_text(payload["view"], "view")
    if view != "conflict_free_core":
        raise ValueError("confirmatory view must be conflict_free_core")
    max_samples = payload["max_samples"]
    if type(max_samples) is not int or not 0 < max_samples <= 500_000:
        raise ValueError("confirmatory max_samples must be in [1, 500000]")
    seed = payload["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("confirmatory seed must be a non-negative integer")
    selection_policy = _require_text(
        payload["selection_policy"], "selection_policy"
    )
    if selection_policy != "native_proportional_group_safe":
        raise ValueError(
            "selection_policy must be native_proportional_group_safe"
        )
    normalized = {
        "audit_config": str(audit_config_path),
        "audit_report_dir": str(audit_report_dir),
        "exclusion_split_dirs": [str(path) for path in exclusions],
        "max_samples": max_samples,
        "seed": seed,
        "selection_policy": selection_policy,
        "view": view,
    }
    return NFTONConfirmatoryConfig(
        audit_config_path=audit_config_path,
        audit_report_dir=audit_report_dir,
        exclusion_split_dirs=exclusions,
        view=view,
        max_samples=max_samples,
        seed=seed,
        selection_policy=selection_policy,
        config_hash=stable_hash(normalized, COHORT_CONFIG_NAMESPACE),
    )


def _load_exclusions(config: NFTONConfirmatoryConfig, audit_state):
    sample_ids: set[str] = set()
    feature_keys: set[str] = set()
    group_ids: set[str] = set()
    source_refs: set[tuple[str, str]] = set()
    native_counts = Counter()
    artifacts = []
    for split_dir in config.exclusion_split_dirs:
        if (split_dir / ".incomplete").exists():
            raise ValueError(f"exclusion split is incomplete: {split_dir}")
        rejected_path = split_dir / "REJECTED.json"
        artifact_status = "completed"
        rejected_sha256 = None
        if rejected_path.exists():
            rejected = _load_json_object(rejected_path, "exclusion rejection")
            if (
                rejected.get("status") != "rejected"
                or rejected.get("downstream_use_allowed") is not False
            ):
                raise ValueError("exclusion rejection marker is malformed")
            artifact_status = "rejected_but_defensively_excluded"
            rejected_sha256 = _hash_file(rejected_path, "sha256")
        selected_path = split_dir / "selected_records.jsonl"
        summary_path = split_dir / "split_summary.json"
        data_manifest_path = split_dir / "data_manifest.json"
        for path in (selected_path, summary_path, data_manifest_path):
            if not path.is_file():
                raise ValueError(f"exclusion split is missing {path.name}: {split_dir}")
        summary = _load_json_object(summary_path, "exclusion split summary")
        expected = {
            "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
            "schema_version": NF_TON_IOT_V2_SCHEMA.version,
            "schema_hash": NF_TON_IOT_V2_SCHEMA.schema_hash,
            "raw_csv_sha1": audit_state["raw_csv_sha1"],
            "source_manifest_sha256": audit_state["source_manifest_sha256"],
        }
        for field, value in expected.items():
            if summary.get(field) != value:
                raise ValueError(
                    f"exclusion split {field} does not match the audited source"
                )
        namespace = summary.get("leakage_group_namespace")
        if namespace is None:
            config_snapshot_path = split_dir / "config.snapshot.json"
            if not config_snapshot_path.is_file():
                raise ValueError(
                    "derived exclusion is missing its config snapshot"
                )
            derived_config = _load_json_object(
                config_snapshot_path, "derived exclusion config"
            )
            if summary.get("derivation") != "few_shot_train_only":
                raise ValueError(
                    "exclusion split does not declare the leakage namespace"
                )
            try:
                source_dir = Path(
                    derived_config["source_split_dir"]
                ).expanduser().resolve()
                source_selected_hash = derived_config[
                    "source_selected_records_sha256"
                ]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "derived exclusion does not identify its source split"
                ) from exc
            if source_dir not in config.exclusion_split_dirs:
                raise ValueError(
                    "derived exclusion source must also be explicitly excluded"
                )
            source_summary = _load_json_object(
                source_dir / "split_summary.json", "derived source summary"
            )
            if (
                source_summary.get("leakage_group_namespace")
                != LEAKAGE_GROUP_NAMESPACE
                or source_summary.get("selected_records_sha256")
                != source_selected_hash
                or summary.get("source_selected_records_sha256")
                != source_selected_hash
            ):
                raise ValueError(
                    "derived exclusion does not trace to the audited group namespace"
                )
        elif namespace != LEAKAGE_GROUP_NAMESPACE:
            raise ValueError(
                "exclusion split leakage namespace does not match implementation"
            )
        selected_sha256 = _hash_file(selected_path, "sha256")
        if summary.get("selected_records_sha256") != selected_sha256:
            raise ValueError("exclusion selected-record hash does not match summary")
        if _hash_file(data_manifest_path, "sha256") != audit_state[
            "source_manifest_sha256"
        ]:
            raise ValueError("exclusion data manifest does not match source")
        records = _load_selected_records(selected_path)
        if namespace is None and any(
            record["sample_id"] not in sample_ids
            or record["feature_key"] not in feature_keys
            or record["group_id"] not in group_ids
            or (record["source_file"], record["source_row_id"])
            not in source_refs
            for record in records
        ):
            raise ValueError(
                "derived exclusion contains records outside its declared source"
            )
        before = (
            len(sample_ids),
            len(feature_keys),
            len(group_ids),
            len(source_refs),
        )
        for record in records:
            if record["sample_id"] not in sample_ids:
                native_counts[record["native_label"]] += 1
        sample_ids.update(record["sample_id"] for record in records)
        feature_keys.update(record["feature_key"] for record in records)
        group_ids.update(record["group_id"] for record in records)
        source_refs.update(
            (record["source_file"], record["source_row_id"])
            for record in records
        )
        after = (
            len(sample_ids),
            len(feature_keys),
            len(group_ids),
            len(source_refs),
        )
        artifacts.append(
            {
                "path": str(split_dir),
                "artifact_status": artifact_status,
                "rejected_marker_sha256": rejected_sha256,
                "selected_records_sha256": selected_sha256,
                "summary_sha256": _hash_file(summary_path, "sha256"),
                "rows": len(records),
                "new_unique_sample_ids": after[0] - before[0],
                "new_unique_feature_keys": after[1] - before[1],
                "new_unique_group_ids": after[2] - before[2],
                "new_unique_source_refs": after[3] - before[3],
            }
        )
    return {
        "sample_ids": sample_ids,
        "feature_keys": feature_keys,
        "group_ids": group_ids,
        "source_refs": source_refs,
        "native_counts": native_counts,
        "artifacts": artifacts,
    }


def _source_native_counts(audit_state) -> dict[str, int]:
    source = _load_json_object(
        audit_state["audit_config"].source_manifest, "source manifest"
    )
    try:
        binary = source["label_counts"]["binary"]
        attacks = source["label_counts"]["native_attack"]
        counts = {"Benign": int(binary["benign"])}
        counts.update({label: int(attacks[label]) for label in attacks})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "source manifest does not provide complete native label counts"
        ) from exc
    if set(counts) != set(NF_TON_LABEL_SPACE.native_labels):
        raise ValueError("source native counts do not match the label space")
    if any(value <= 0 for value in counts.values()):
        raise ValueError("source native counts must be positive")
    return counts


def _write_selected_records(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(canonical_json(asdict(record)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _distribution(counts, total: int) -> dict[str, float]:
    return {
        label: float(counts.get(label, 0) / total)
        for label in NF_TON_LABEL_SPACE.native_labels
    }


def publish_nf_ton_confirmatory_cohort(
    config_path,
    output_path,
    *,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Publish a locked cohort outside every declared historical split."""

    config = load_nf_ton_confirmatory_config(config_path)
    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if output_dir.exists():
        raise ValueError(f"cohort output directory already exists: {output_dir}")
    audit_state = _validated_audit_state(config)
    exclusions = _load_exclusions(config, audit_state)
    source_native_counts = _source_native_counts(audit_state)
    exhausted_labels = [
        label
        for label in NF_TON_LABEL_SPACE.native_labels
        if exclusions["native_counts"][label] >= source_native_counts[label]
    ]
    if exhausted_labels:
        raise ValueError(
            "independent native cohort is impossible because declared historical "
            f"exclusions exhaust full-source rows for labels: {exhausted_labels}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text("NF-ToN confirmatory cohort incomplete\n", encoding="utf-8")

    indexed = index_netflow_v2_rows(
        _raw_rows(audit_state["audit_config"]),
        schema=NF_TON_IOT_V2_SCHEMA,
        label_space=NF_TON_LABEL_SPACE,
        conflict_feature_keys=audit_state["conflict_feature_keys"],
        source_file_id=audit_state["audit_config"].source_file_id,
        coarse_label_map=audit_state["audit_config"].coarse_label_map,
        sensitive_hash_namespace=(
            audit_state["audit_config"].sensitive_hash_namespace
        ),
        leakage_group_namespace=LEAKAGE_GROUP_NAMESPACE,
    )
    if len(indexed) != audit_state["expected_view_rows"]:
        raise RuntimeError("indexed row count does not match the audited view")

    eligible_counts = Counter()
    eligible_rows = 0

    def eligible_records():
        nonlocal eligible_rows
        for record in indexed:
            source_ref = (record.source_file, record.source_row_id)
            overlaps = (
                record.sample_id in exclusions["sample_ids"],
                record.feature_key in exclusions["feature_keys"],
                record.group_id in exclusions["group_ids"],
                source_ref in exclusions["source_refs"],
            )
            if any(overlaps):
                continue
            eligible_rows += 1
            eligible_counts[record.native_label] += 1
            yield record

    selected = select_netflow_v2_proportional_development_records(
        eligible_records(),
        max_samples=config.max_samples,
        seed=config.seed,
        order_namespace=COHORT_ORDER_NAMESPACE,
        required_native_labels=NF_TON_LABEL_SPACE.native_labels,
    )
    if len(selected) != config.max_samples:
        raise RuntimeError("confirmatory selection did not reach max_samples")

    overlap = {
        "sample_ids": sum(
            record.sample_id in exclusions["sample_ids"] for record in selected
        ),
        "feature_keys": sum(
            record.feature_key in exclusions["feature_keys"] for record in selected
        ),
        "group_ids": sum(
            record.group_id in exclusions["group_ids"] for record in selected
        ),
        "source_refs": sum(
            (record.source_file, record.source_row_id) in exclusions["source_refs"]
            for record in selected
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"confirmatory cohort overlaps exclusions: {overlap}")

    config_path_out = output_dir / "config.snapshot.json"
    data_manifest_path = output_dir / "data_manifest.json"
    selected_path = output_dir / "selected_records.jsonl"
    overlap_path = output_dir / "overlap_audit.json"
    manifest_path = output_dir / "cohort_manifest.json"
    write_json_snapshot(config_path_out, config.to_dict())
    data_manifest_path.write_bytes(
        audit_state["audit_config"].source_manifest.read_bytes()
    )
    _write_selected_records(selected_path, selected)

    selected_counts = Counter(record.native_label for record in selected)
    selected_distribution = _distribution(selected_counts, len(selected))
    eligible_distribution = _distribution(eligible_counts, eligible_rows)
    overlap_audit = {
        "format_version": COHORT_FORMAT_VERSION,
        "exclusion_artifacts": exclusions["artifacts"],
        "excluded_unique_counts": {
            "sample_ids": len(exclusions["sample_ids"]),
            "feature_keys": len(exclusions["feature_keys"]),
            "group_ids": len(exclusions["group_ids"]),
            "source_refs": len(exclusions["source_refs"]),
        },
        "excluded_unique_native_labels": dict(
            sorted(exclusions["native_counts"].items())
        ),
        "full_source_native_labels": source_native_counts,
        "selected_overlap_counts": overlap,
        "verified_zero_overlap": not any(overlap.values()),
    }
    write_json_snapshot(overlap_path, overlap_audit)
    manifest = {
        "format_version": COHORT_FORMAT_VERSION,
        "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
        "purpose": "independent_a0_mr_confirmation",
        "selection_scope": "unused_groups_outside_all_declared_splits",
        "selection_policy": config.selection_policy,
        "selection_order_namespace": COHORT_ORDER_NAMESPACE,
        "seed": config.seed,
        "selected_rows": len(selected),
        "selected_groups": len({record.group_id for record in selected}),
        "selected_feature_keys": len(
            {record.feature_key for record in selected}
        ),
        "eligible_rows_after_exclusion": eligible_rows,
        "native_labels": _label_counts(selected, "native_label"),
        "eligible_native_labels": dict(sorted(eligible_counts.items())),
        "selected_native_distribution": selected_distribution,
        "eligible_native_distribution": eligible_distribution,
        "max_absolute_native_proportion_delta": max(
            abs(selected_distribution[label] - eligible_distribution[label])
            for label in NF_TON_LABEL_SPACE.native_labels
        ),
        "config_hash": config.config_hash,
        "audit_config_hash": audit_state["audit_config"].config_hash,
        "audit_summary_sha256": audit_state["audit_summary_sha256"],
        "raw_csv_sha1": audit_state["raw_csv_sha1"],
        "source_manifest_sha256": audit_state["source_manifest_sha256"],
        "schema_version": NF_TON_IOT_V2_SCHEMA.version,
        "schema_hash": NF_TON_IOT_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_TON_LABEL_SPACE.label_space_hash,
        "feature_key_namespace": (
            audit_state["audit_config"].feature_key_namespace
        ),
        "leakage_group_namespace": LEAKAGE_GROUP_NAMESPACE,
        "selected_records_sha256": _hash_file(selected_path, "sha256"),
        "overlap_audit_sha256": _hash_file(overlap_path, "sha256"),
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
    }
    write_json_snapshot(manifest_path, manifest)
    marker.unlink()
    return {
        "output": str(output_dir),
        "selected_rows": len(selected),
        "selected_groups": manifest["selected_groups"],
        "selected_records_sha256": manifest["selected_records_sha256"],
        "verified_zero_overlap": True,
        "max_absolute_native_proportion_delta": manifest[
            "max_absolute_native_proportion_delta"
        ],
    }


def _normalize_binary_label(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if normalized in {"0", "benign", "normal"}:
        return "benign"
    if normalized in {"1", "attack", "malicious"}:
        return "attack"
    raise ValueError(f"unsupported binary label: {value!r}")


def load_nf_ton_confirmatory_samples(
    cohort_dir,
    preprocessor_dir,
    *,
    data_root=Path("/root/autodl-tmp"),
) -> NFTONConfirmatorySamples:
    """Load a published cohort with the original train-only preprocessor."""

    cohort_root = validate_output_path(
        cohort_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    preprocessor_root = validate_output_path(
        preprocessor_dir,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not cohort_root.is_dir() or not preprocessor_root.is_dir():
        raise ValueError("cohort and preprocessor directories must exist")
    if (cohort_root / ".incomplete").exists():
        raise ValueError("confirmatory cohort is incomplete")
    for filename in _REQUIRED_FILES:
        if not (cohort_root / filename).is_file():
            raise ValueError(f"confirmatory cohort is missing {filename}")

    config_payload = _load_json_object(
        cohort_root / "config.snapshot.json", "cohort config"
    )
    manifest = _load_json_object(
        cohort_root / "cohort_manifest.json", "cohort manifest"
    )
    overlap = _load_json_object(
        cohort_root / "overlap_audit.json", "cohort overlap audit"
    )
    if stable_hash(config_payload, COHORT_CONFIG_NAMESPACE) != manifest.get(
        "config_hash"
    ):
        raise ValueError("cohort config hash does not match manifest")
    if overlap.get("verified_zero_overlap") is not True or any(
        overlap.get("selected_overlap_counts", {}).values()
    ):
        raise ValueError("cohort does not prove zero overlap")
    selected_path = cohort_root / "selected_records.jsonl"
    overlap_path = cohort_root / "overlap_audit.json"
    if _hash_file(selected_path, "sha256") != manifest.get(
        "selected_records_sha256"
    ):
        raise ValueError("cohort selected-record hash does not match")
    if _hash_file(overlap_path, "sha256") != manifest.get(
        "overlap_audit_sha256"
    ):
        raise ValueError("cohort overlap audit hash does not match")
    if not 0 < manifest.get("selected_groups", 0) <= manifest.get(
        "selected_rows", 0
    ):
        raise ValueError("cohort selected group count is invalid")

    audit_config = _load_json_object(
        cohort_root / "config.snapshot.json", "cohort config"
    )["audit_config"]
    from flowfm.data.audit import load_audit_config

    audit = load_audit_config(audit_config)
    raw_sha1 = _hash_file(audit.raw_csv, "sha1")
    expected_identity = {
        "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
        "schema_version": NF_TON_IOT_V2_SCHEMA.version,
        "schema_hash": NF_TON_IOT_V2_SCHEMA.schema_hash,
        "raw_csv_sha1": raw_sha1,
        "source_manifest_sha256": _hash_file(audit.source_manifest, "sha256"),
        "leakage_group_namespace": LEAKAGE_GROUP_NAMESPACE,
    }
    for field, value in expected_identity.items():
        if manifest.get(field) != value:
            raise ValueError(f"cohort manifest {field} does not match source")
    if _hash_file(cohort_root / "data_manifest.json", "sha256") != manifest[
        "source_manifest_sha256"
    ]:
        raise ValueError("cohort data manifest does not match source")

    preprocessor_path = preprocessor_root / "preprocessor.json"
    preprocessor_summary_path = preprocessor_root / "preprocessor_summary.json"
    if not preprocessor_path.is_file() or not preprocessor_summary_path.is_file():
        raise ValueError("train-only preprocessor artifacts are incomplete")
    preprocessor_summary = _load_json_object(
        preprocessor_summary_path, "preprocessor summary"
    )
    if preprocessor_summary.get("fit_scope") != "train_only":
        raise ValueError("confirmatory preprocessing must remain train-only")
    if preprocessor_summary.get("preprocessor_file_sha256") != _hash_file(
        preprocessor_path, "sha256"
    ):
        raise ValueError("preprocessor file hash does not match summary")
    preprocessor = load_nf_ton_preprocessor(preprocessor_path)
    if preprocessor.preprocessor_hash != preprocessor_summary.get(
        "preprocessor_hash"
    ):
        raise ValueError("preprocessor hash does not match summary")

    records = _load_selected_records(selected_path)
    if len(records) != manifest.get("selected_rows"):
        raise ValueError("cohort selected row count does not match manifest")
    feature_group_sizes = Counter(record["feature_key"] for record in records)
    records_by_source_row = {}
    for record in records:
        if record["source_file"] != audit.source_file_id:
            raise ValueError("cohort source_file does not match audit config")
        if not record["source_row_id"].isdigit():
            raise ValueError("cohort source row IDs must be positive integers")
        records_by_source_row[int(record["source_row_id"])] = record

    expected_header = (
        *NF_TON_IOT_V2_SCHEMA.raw_feature_names,
        audit.binary_label_column,
        audit.native_label_column,
    )
    samples_by_id = {}
    with audit.raw_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header does not match the registered schema")
        for row_number, row in enumerate(reader, start=1):
            record = records_by_source_row.get(row_number)
            if record is None:
                continue
            if compute_feature_key(row, NF_TON_IOT_V2_SCHEMA) != record[
                "feature_key"
            ]:
                raise ValueError("cohort raw feature key does not match artifact")
            native_label = row[audit.native_label_column].strip()
            if native_label != record["native_label"]:
                raise ValueError("cohort native label does not match raw source")
            if audit.coarse_label_map[native_label] != record["coarse_label"]:
                raise ValueError("cohort coarse label does not match raw source")
            if _normalize_binary_label(row[audit.binary_label_column]) != record[
                "binary_label"
            ]:
                raise ValueError("cohort binary label does not match raw source")
            encoded = preprocessor.transform(row)
            samples_by_id[record["sample_id"]] = FlowSample(
                features=encoded.features,
                feature_ids=encoded.feature_ids,
                categorical_mask=encoded.categorical_mask,
                missing_mask=encoded.missing_mask,
                category_tokens=encoded.category_tokens,
                binary_label=record["binary_label"],
                coarse_label=record["coarse_label"],
                native_label=record["native_label"],
                dataset_id=NF_TON_IOT_V2_SCHEMA.dataset_id,
                sample_id=record["sample_id"],
                feature_key=record["feature_key"],
                conflict_flag=False,
                conflict_group_size=feature_group_sizes[record["feature_key"]],
                conflict_label_count=1,
                source_file=record["source_file"],
                source_row_id=record["source_row_id"],
            )
    missing = [
        record["sample_id"]
        for record in records
        if record["sample_id"] not in samples_by_id
    ]
    if missing:
        raise ValueError(f"cohort rows are missing from raw CSV: {missing[:5]}")
    return NFTONConfirmatorySamples(
        samples=tuple(samples_by_id[record["sample_id"]] for record in records),
        group_ids=tuple(record["group_id"] for record in records),
        cohort_manifest_sha256=_hash_file(
            cohort_root / "cohort_manifest.json", "sha256"
        ),
        preprocessor_hash=preprocessor.preprocessor_hash,
        raw_csv_sha1=raw_sha1,
    )


__all__ = [
    "COHORT_CONFIG_NAMESPACE",
    "COHORT_FORMAT_VERSION",
    "COHORT_ORDER_NAMESPACE",
    "NFTONConfirmatoryConfig",
    "NFTONConfirmatorySamples",
    "load_nf_ton_confirmatory_config",
    "load_nf_ton_confirmatory_samples",
    "publish_nf_ton_confirmatory_cohort",
]
