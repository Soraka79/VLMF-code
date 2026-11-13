"""Record-independent formal cohorts for the cross-dataset A0-MR matrix."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from flowfm.data.audit import load_audit_config
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import (
    NF_BOT_LABEL_SPACE,
    NF_CSE_LABEL_SPACE,
    NF_TON_LABEL_SPACE,
    NF_UNSW_LABEL_SPACE,
    DatasetLabelSpace,
    coarse_task_label_space,
)
from flowfm.data.netflow_v2_protocol import (
    index_netflow_v2_rows,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_unsw_split import _hash_file
from flowfm.data.schemas.nf_bot_iot_v2 import NF_BOT_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
from flowfm.data.splits import SplitManifest, deterministic_stratified_group_split
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


FORMAL_COHORT_FORMAT_VERSION = 1
FORMAL_COHORT_PROTOCOL_VERSION = "cross-dataset-formal-cohort-v1"
FORMAL_RESTART_COHORT_PROTOCOL_VERSION = (
    "cross-dataset-disclosed-full-source-restart-cohort-v1"
)
FORMAL_REVISION_COHORT_PROTOCOL_VERSION = (
    "cross-dataset-revision-resample-cohort-v1"
)
HISTORY_POLICY_EXCLUDE = "exclude_all_discovered_record_and_group_exposure"
HISTORY_POLICY_DISCLOSED_RESTART = (
    "reuse_retired_historical_exposure_with_disclosure"
)
FORMAL_SOURCE_HASH_NAMESPACE = "flowfm-cross-dataset-formal-source-v1"
FORMAL_SELECTION_ORDER_NAMESPACE = (
    "flowfm-cross-dataset-formal-native-proportional-order-v1"
)
FORMAL_FEW_SHOT_NAMESPACE = "flowfm-cross-dataset-formal-few-shot-v1"
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_NAMES = ("train", "val", "test")
HISTORY_IDENTITY_FIELDS = (
    "sample_id",
    "source_row_id",
    "feature_key",
    "group_id",
)
_STRUCTURED_NAMES = {
    "config.snapshot.json",
    "config.json",
    "config.yaml",
    "data_manifest.json",
    "formal_provenance.json",
    "pretrained_initialization.json",
}


@dataclass(frozen=True)
class FormalDatasetSpec:
    key: str
    dataset_id: str
    schema: Any
    source_label_space: DatasetLabelSpace
    task_coarse_labels: tuple[str, ...]
    task_mapping_version: str
    split_seed: int
    audit_config_relative: str
    audit_report_relative: str
    leakage_group_namespace: str
    cohort_name: str
    rare_coarse_labels: tuple[str, ...]
    max_samples: int = 1_000_000
    protocol_version: str = FORMAL_COHORT_PROTOCOL_VERSION
    historical_exposure_policy: str = HISTORY_POLICY_EXCLUDE

    def __post_init__(self) -> None:
        if self.historical_exposure_policy not in {
            HISTORY_POLICY_EXCLUDE,
            HISTORY_POLICY_DISCLOSED_RESTART,
        }:
            raise ValueError("unsupported formal historical-exposure policy")
        if not self.protocol_version:
            raise ValueError("formal protocol_version must be non-empty")

    @property
    def task_label_space(self) -> DatasetLabelSpace:
        return coarse_task_label_space(
            self.source_label_space,
            self.task_coarse_labels,
            mapping_version=self.task_mapping_version,
        )

    @property
    def task_native_labels(self) -> tuple[str, ...]:
        return self.task_label_space.native_labels

    @property
    def requires_zero_historical_overlap(self) -> bool:
        return self.historical_exposure_policy == HISTORY_POLICY_EXCLUDE

    @property
    def feasibility_status(self) -> str:
        if self.requires_zero_historical_overlap:
            return "eligible_for_exact_group_safe_build"
        return "eligible_for_disclosed_full_source_restart"

    @property
    def seal_status(self) -> str:
        if self.requires_zero_historical_overlap:
            return "sealed_before_any_model_selection"
        return "sealed_before_any_restart_model_selection"


FORMAL_DATASETS = {
    "nf_bot": FormalDatasetSpec(
        key="nf_bot",
        dataset_id="NF-BoT-IoT-v2",
        schema=NF_BOT_IOT_V2_SCHEMA,
        source_label_space=NF_BOT_LABEL_SPACE,
        task_coarse_labels=("benign", "dos", "scan", "malware"),
        task_mapping_version="nf-bot-iot-v2-formal-shared-coarse-four-class-v1",
        split_seed=20260727,
        audit_config_relative="configs/data/nf_bot_iot_v2_audit.yaml",
        audit_report_relative=(
            "reports/data_audit/NF-BoT-IoT-v2/"
            "nf-bot-iot-v2-model-41-v1-28ce004"
        ),
        leakage_group_namespace="flowfm-nf-bot-iot-v2-leakage-group-v1",
        cohort_name="formal-unused-1m-seed20260727-shared-coarse-v1",
        rare_coarse_labels=("benign", "malware"),
    ),
    "nf_cse": FormalDatasetSpec(
        key="nf_cse",
        dataset_id="NF-CSE-CIC-IDS2018-v2",
        schema=NF_CSE_CIC_IDS2018_V2_SCHEMA,
        source_label_space=NF_CSE_LABEL_SPACE,
        task_coarse_labels=(
            "benign",
            "dos",
            "brute_force",
            "exploit",
            "botnet",
        ),
        task_mapping_version=(
            "nf-cse-cic-ids2018-v2-formal-shared-coarse-five-class-v2"
        ),
        split_seed=20260731,
        audit_config_relative="configs/data/nf_cse_cic_ids2018_v2.yaml",
        audit_report_relative=(
            "reports/data_audit/NF-CSE-CIC-IDS2018-v2/"
            "nf-cse-cic-ids2018-v2-model-41-v1-61a3179"
        ),
        leakage_group_namespace="flowfm-nf-cse-cic-ids2018-v2-leakage-group-v1",
        cohort_name="formal-unused-1m-seed20260731-shared-coarse-v2",
        rare_coarse_labels=("brute_force", "exploit", "botnet"),
    ),
    "nf_unsw": FormalDatasetSpec(
        key="nf_unsw",
        dataset_id="NF-UNSW-NB15-v2",
        schema=NF_UNSW_NB15_V2_SCHEMA,
        source_label_space=NF_UNSW_LABEL_SPACE,
        task_coarse_labels=(
            "benign",
            "dos",
            "scan",
            "exploit",
            "backdoor",
            "malware",
        ),
        task_mapping_version="nf-unsw-nb15-v2-formal-natural-six-class-v1",
        split_seed=20260728,
        audit_config_relative="configs/data/nf_unsw_nb15_v2.yaml",
        audit_report_relative=(
            "reports/data_audit/NF-UNSW-NB15-v2/model-41-v1-e20be31"
        ),
        leakage_group_namespace="flowfm-nf-unsw-leakage-group-v1",
        cohort_name="formal-unused-1m-seed20260728-shared-coarse-v1",
        rare_coarse_labels=("dos", "scan", "backdoor", "malware"),
    ),
    "nf_unsw_restart": FormalDatasetSpec(
        key="nf_unsw_restart",
        dataset_id="NF-UNSW-NB15-v2",
        schema=NF_UNSW_NB15_V2_SCHEMA,
        source_label_space=NF_UNSW_LABEL_SPACE,
        task_coarse_labels=(
            "benign",
            "dos",
            "scan",
            "exploit",
            "backdoor",
            "malware",
        ),
        task_mapping_version=(
            "nf-unsw-nb15-v2-formal-restart-natural-six-class-v1"
        ),
        split_seed=20260801,
        audit_config_relative="configs/data/nf_unsw_nb15_v2.yaml",
        audit_report_relative=(
            "reports/data_audit/NF-UNSW-NB15-v2/model-41-v1-e20be31"
        ),
        leakage_group_namespace="flowfm-nf-unsw-leakage-group-v1",
        cohort_name="formal-restart-1m-seed20260801-shared-coarse-v1",
        rare_coarse_labels=("dos", "scan", "backdoor", "malware"),
        protocol_version=FORMAL_RESTART_COHORT_PROTOCOL_VERSION,
        historical_exposure_policy=HISTORY_POLICY_DISCLOSED_RESTART,
    ),
    "nf_ton": FormalDatasetSpec(
        key="nf_ton",
        dataset_id="NF-ToN-IoT-v2",
        schema=NF_TON_IOT_V2_SCHEMA,
        source_label_space=NF_TON_LABEL_SPACE,
        task_coarse_labels=(
            "benign",
            "dos",
            "scan",
            "brute_force",
            "exploit",
        ),
        task_mapping_version="nf-ton-iot-v2-formal-shared-coarse-five-class-v1",
        split_seed=20260729,
        audit_config_relative="configs/data/nf_ton_iot_v2.yaml",
        audit_report_relative=(
            "reports/data_audit/NF-ToN-IoT-v2/"
            "nf-ton-iot-v2-model-41-v1-7ed0e56"
        ),
        leakage_group_namespace="flowfm-nf-ton-iot-v2-leakage-group-v1",
        cohort_name="formal-unused-1m-seed20260729-shared-coarse-v1",
        rare_coarse_labels=(),
    ),
}


# Revision resampling cohorts: same task, audit, and selection procedure as the
# formal cohorts, drawn with new split seeds from the full conflict-free pool.
# Overlap with earlier cohorts is permitted and recorded, not excluded. The
# first three seeds are used now; the last two stay unopened in reserve.
REVISION_COHORT_SEEDS = (20261018, 20261019, 20261020, 20261021, 20261022)
REVISION_ACTIVE_COHORT_COUNT = 3


def revision_cohort_key(base_key: str, index: int) -> str:
    return f"{base_key}_rev{index}"


def _revision_specs():
    specs = {}
    for base_key in ("nf_bot", "nf_cse", "nf_ton"):
        base = FORMAL_DATASETS[base_key]
        for index, seed in enumerate(REVISION_COHORT_SEEDS, start=1):
            key = revision_cohort_key(base_key, index)
            specs[key] = replace(
                base,
                key=key,
                split_seed=seed,
                cohort_name=f"formal-revision-1m-seed{seed}-shared-coarse-v1",
                protocol_version=FORMAL_REVISION_COHORT_PROTOCOL_VERSION,
                historical_exposure_policy=HISTORY_POLICY_DISCLOSED_RESTART,
            )
    return specs


FORMAL_DATASETS.update(_revision_specs())


def get_formal_dataset_spec(key: str) -> FormalDatasetSpec:
    try:
        return FORMAL_DATASETS[key]
    except KeyError as exc:
        raise ValueError(
            "dataset must be one of: " + ", ".join(FORMAL_DATASETS)
        ) from exc


def validate_formal_overlap_contract(
    spec: FormalDatasetSpec, overlap: dict[str, Any]
) -> None:
    """Validate internal isolation and the declared historical-exposure mode."""

    if overlap.get("verified_zero_internal_overlap") is not True:
        raise ValueError("formal cohort does not prove zero internal overlap")
    if spec.requires_zero_historical_overlap:
        if overlap.get("verified_zero_historical_overlap") is not True:
            raise ValueError("formal cohort does not prove zero historical overlap")
        policy = overlap.get("historical_exposure_policy")
        if policy not in (None, HISTORY_POLICY_EXCLUDE):
            raise ValueError("formal cohort historical-exposure policy changed")
        return
    if (
        overlap.get("historical_exposure_policy")
        != HISTORY_POLICY_DISCLOSED_RESTART
        or overlap.get("historical_overlap_expected") is not True
        or overlap.get("historically_independent") is not False
    ):
        raise ValueError("restart cohort does not disclose historical exposure")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _load_structured(path: Path):
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _source_native_counts(source_manifest, label_space) -> Counter:
    payload = _load_json(source_manifest)
    if payload.get("dataset_id") != label_space.dataset_id:
        raise ValueError("source Manifest dataset_id does not match label space")
    try:
        counts = Counter(
            {
                label: int(value)
                for label, value in payload["label_counts"][
                    "native_attack"
                ].items()
            }
        )
        benign_count = int(payload["label_counts"]["binary"]["benign"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source Manifest has invalid label counts") from exc
    if "Benign" in label_space.native_labels and "Benign" not in counts:
        counts["Benign"] = benign_count
    if set(counts) != set(label_space.native_labels):
        raise ValueError("source counts do not cover the versioned native labels")
    return counts


def _coarse_counts(native_counts, label_space) -> Counter:
    counts = Counter()
    for label, count in native_counts.items():
        counts[label_space.native_to_coarse[label]] += int(count)
    return counts


def _validated_audit_state(
    spec: FormalDatasetSpec,
    *,
    repo_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    audit_config_path = (repo_dir / spec.audit_config_relative).resolve()
    report_dir = (project_root / spec.audit_report_relative).resolve()
    if not audit_config_path.is_file() or not report_dir.is_dir():
        raise ValueError("formal audit inputs are missing")
    audit_config = load_audit_config(audit_config_path)
    if (
        audit_config.dataset_id != spec.dataset_id
        or audit_config.schema_version != spec.schema.version
        or audit_config.schema.schema_hash != spec.schema.schema_hash
        or audit_config.coarse_label_map
        != dict(spec.source_label_space.native_to_coarse)
    ):
        raise ValueError("audit config does not match the formal dataset spec")
    summary_path = report_dir / "conflict_summary.json"
    views_path = report_dir / "logical_views.json"
    conflict_path = report_dir / "conflict_groups.parquet"
    if not all(path.is_file() for path in (summary_path, views_path, conflict_path)):
        raise ValueError("formal audit report is incomplete")
    summary = _load_json(summary_path)
    views = _load_json(views_path)
    expected = {
        "dataset_id": spec.dataset_id,
        "schema_version": spec.schema.version,
        "schema_hash": spec.schema.schema_hash,
        "config_hash": audit_config.config_hash,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"audit summary {field} changed")
    raw_sha1 = _hash_file(audit_config.raw_csv, "sha1")
    if (
        summary.get("raw_mutated") is not False
        or summary.get("raw_sha1_before") != raw_sha1
        or summary.get("raw_sha1_after") != raw_sha1
    ):
        raise ValueError("raw data no longer matches the conflict audit")
    source_manifest_sha256 = _hash_file(
        audit_config.source_manifest, "sha256"
    )
    if summary.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("source Manifest no longer matches the conflict audit")
    try:
        conflict_keys = set(views["conflict_quarantine"]["feature_keys"])
        core_rows = int(views["conflict_free_core"]["rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("logical views are invalid") from exc
    conflict_table = pq.read_table(
        conflict_path, columns=["native_label", "coarse_label", "feature_key"]
    )
    if conflict_table.num_rows != int(summary["conflict_rows"]):
        raise ValueError("conflict parquet row count changed")
    parquet_keys = set(conflict_table.column("feature_key").to_pylist())
    if parquet_keys != conflict_keys:
        raise ValueError("conflict parquet keys differ from logical view")
    source_native = _source_native_counts(
        audit_config.source_manifest, spec.source_label_space
    )
    if sum(source_native.values()) != int(summary["total_rows"]):
        raise ValueError("source label counts do not match audited total rows")
    conflict_native = Counter(
        conflict_table.column("native_label").to_pylist()
    )
    core_native = source_native - conflict_native
    if sum(core_native.values()) != core_rows:
        raise ValueError("derived conflict-free label counts do not match view")
    return {
        "audit_config": audit_config,
        "audit_config_path": audit_config_path,
        "report_dir": report_dir,
        "summary_path": summary_path,
        "views_path": views_path,
        "conflict_path": conflict_path,
        "summary": summary,
        "conflict_feature_keys": conflict_keys,
        "source_native_counts": source_native,
        "conflict_native_counts": conflict_native,
        "core_native_counts": core_native,
        "raw_csv_sha1": raw_sha1,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _load_selected_records(path: Path):
    required = {
        "sample_id",
        "source_row_id",
        "source_file",
        "feature_key",
        "group_id",
        "native_label",
        "coarse_label",
        "binary_label",
    }
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid selected record at {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"selected record fields are incomplete: {path}")
            if any(
                not isinstance(record[field], str) or not record[field]
                for field in required
            ):
                raise ValueError(f"selected record contains invalid text: {path}")
            yield record


def _normalize_split_reference(value: str, split_root: Path) -> str:
    candidate = Path(value).expanduser().resolve()
    root = split_root.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return str(candidate)
    if not relative.parts:
        return str(candidate)
    return str(root / relative.parts[0])


def _history_inventory(
    spec: FormalDatasetSpec,
    *,
    project_root: Path,
) -> dict[str, Any]:
    split_root = project_root / "data" / "splits" / spec.dataset_id
    selected_paths = (
        sorted(split_root.glob("*/selected_records.jsonl"))
        if split_root.is_dir()
        else []
    )
    incomplete_dirs = (
        sorted(path.parent for path in split_root.glob("*/.incomplete"))
        if split_root.is_dir()
        else []
    )
    records_by_sample = {}
    identities = {field: set() for field in HISTORY_IDENTITY_FIELDS}
    source_refs = set()
    artifacts = []
    for path in selected_paths:
        before = {field: len(values) for field, values in identities.items()}
        row_count = 0
        for record in _load_selected_records(path):
            row_count += 1
            sample_id = record["sample_id"]
            previous = records_by_sample.get(sample_id)
            identity = {
                field: record[field] for field in HISTORY_IDENTITY_FIELDS
            }
            identity.update(
                {
                    "source_file": record["source_file"],
                    "native_label": record["native_label"],
                    "coarse_label": record["coarse_label"],
                }
            )
            if previous is not None and previous != identity:
                raise ValueError("historical sample identity is inconsistent")
            records_by_sample[sample_id] = identity
            for field in HISTORY_IDENTITY_FIELDS:
                identities[field].add(record[field])
            source_refs.add((record["source_file"], record["source_row_id"]))
        artifacts.append(
            {
                "split_dir": str(path.parent.resolve()),
                "selected_records_sha256": _hash_file(path, "sha256"),
                "rows": row_count,
                "rejected": (path.parent / "REJECTED.json").is_file(),
                "new_unique_identities": {
                    field: len(identities[field]) - before[field]
                    for field in HISTORY_IDENTITY_FIELDS
                },
            }
        )

    provenance = []
    referenced_target_splits = set()
    for root in (
        project_root / "data" / "preprocessing" / spec.dataset_id,
        project_root / "results",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.name not in _STRUCTURED_NAMES
                or path.stat().st_size > 10_000_000
            ):
                continue
            payload = _load_structured(path)
            if payload is None:
                continue
            strings = tuple(_walk_strings(payload))
            if not any(spec.dataset_id in value for value in strings):
                continue
            referenced = sorted(
                {
                    _normalize_split_reference(value, split_root)
                    for value in strings
                    if value.startswith("/")
                    and "/data/splits/" in value
                    and spec.dataset_id in value
                }
            )
            referenced_target_splits.update(referenced)
            provenance.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _hash_file(path, "sha256"),
                    "referenced_target_splits": referenced,
                }
            )
    discovered_dirs = {str(path.parent.resolve()) for path in selected_paths}
    untracked_references = sorted(referenced_target_splits - discovered_dirs)
    if untracked_references:
        raise ValueError(
            "provenance references target splits without selected records: "
            + ", ".join(untracked_references)
        )
    native_counts = Counter(
        record["native_label"] for record in records_by_sample.values()
    )
    coarse_counts = Counter(
        record["coarse_label"] for record in records_by_sample.values()
    )
    return {
        "split_root": str(split_root.resolve()),
        "selected_artifacts": artifacts,
        "incomplete_split_dirs_without_selected_records": [
            str(path.resolve()) for path in incomplete_dirs
        ],
        "structured_provenance_artifacts": provenance,
        "untracked_referenced_split_dirs": [],
        "records_by_sample": records_by_sample,
        "identities": identities,
        "source_refs": source_refs,
        "native_counts": native_counts,
        "coarse_counts": coarse_counts,
    }


def _public_history(history) -> dict[str, Any]:
    return {
        "discovery_rule": (
            "all selected_records.jsonl under the target split root plus "
            "structured preprocessing/result provenance references"
        ),
        "split_root": history["split_root"],
        "selected_artifacts": history["selected_artifacts"],
        "incomplete_split_dirs_without_selected_records": history[
            "incomplete_split_dirs_without_selected_records"
        ],
        "structured_provenance_artifacts": history[
            "structured_provenance_artifacts"
        ],
        "untracked_referenced_split_dirs": [],
        "unique_identity_counts": {
            field: len(history["identities"][field])
            for field in HISTORY_IDENTITY_FIELDS
        },
        "unique_source_refs": len(history["source_refs"]),
        "native_record_counts": dict(sorted(history["native_counts"].items())),
        "coarse_record_counts": dict(sorted(history["coarse_counts"].items())),
        "complete": True,
    }


def audit_formal_dataset(
    key: str,
    output_path,
    *,
    repo_dir,
    project_root,
    data_root=Path("/root/autodl-tmp"),
) -> dict[str, Any]:
    """Publish history exposure and a pre-model feasibility decision."""

    spec = get_formal_dataset_spec(key)
    repo = Path(repo_dir).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    output = validate_output_path(
        output_path, enforce_autodl_data_disk=True, data_root=data_root
    )
    if output.exists():
        raise ValueError(f"formal audit output already exists: {output}")
    audit = _validated_audit_state(spec, repo_dir=repo, project_root=project)
    history = _history_inventory(spec, project_root=project)
    historical_conflict_overlap = sum(
        record["feature_key"] in audit["conflict_feature_keys"]
        for record in history["records_by_sample"].values()
    )
    if historical_conflict_overlap:
        raise ValueError("historical selected records overlap conflict quarantine")

    upper_native = audit["core_native_counts"] - history["native_counts"]
    upper_coarse = _coarse_counts(upper_native, spec.source_label_space)
    core_coarse = _coarse_counts(
        audit["core_native_counts"], spec.source_label_space
    )
    exhausted_natural = sorted(
        label
        for label in set(spec.source_label_space.native_to_coarse.values())
        if upper_coarse[label] <= 0
    )
    independent_exhausted_required = [
        label for label in spec.task_coarse_labels if upper_coarse[label] <= 0
    ]
    restart_exhausted_required = [
        label for label in spec.task_coarse_labels if core_coarse[label] <= 0
    ]
    exhausted_required = (
        independent_exhausted_required
        if spec.requires_zero_historical_overlap
        else restart_exhausted_required
    )
    status = (
        "not_evaluable_required_class_exhausted"
        if exhausted_required
        else spec.feasibility_status
    )
    reason = None
    if exhausted_required:
        reason = (
            "At least one predeclared natural shared-coarse class has zero "
            + (
                "conflict-free records after defensively excluding all "
                "historical record exposure. The task cannot be made "
                "record-independent."
                if spec.requires_zero_historical_overlap
                else "records in the conflict-free full-source restart pool."
            )
        )
    feasibility = {
        "format_version": FORMAL_COHORT_FORMAT_VERSION,
        "protocol_version": spec.protocol_version,
        "dataset_key": key,
        "dataset_id": spec.dataset_id,
        "status": status,
        "reason": reason,
        "claim_scope": (
            "not_evaluable_required_class_exhausted"
            if exhausted_required
            else (
                "historically_independent_confirmatory_target"
                if spec.requires_zero_historical_overlap
                else "full_source_leakage_safe_restart_not_historically_independent"
            )
        ),
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent_confirmatory_target": (
            spec.requires_zero_historical_overlap and not exhausted_required
        ),
        "historical_overlap_expected": (
            not spec.requires_zero_historical_overlap
        ),
        "retired_prior_protocol_records_reusable": (
            not spec.requires_zero_historical_overlap
        ),
        "split_seed": spec.split_seed,
        "max_samples": spec.max_samples,
        "task_coarse_labels_predeclared": list(spec.task_coarse_labels),
        "task_native_labels_predeclared": list(spec.task_native_labels),
        "source_native_counts": dict(
            sorted(audit["source_native_counts"].items())
        ),
        "conflict_quarantine_native_counts": dict(
            sorted(audit["conflict_native_counts"].items())
        ),
        "conflict_free_native_counts": dict(
            sorted(audit["core_native_counts"].items())
        ),
        "historically_exposed_native_counts": dict(
            sorted(history["native_counts"].items())
        ),
        "record_level_remaining_upper_bound_native": dict(
            sorted(upper_native.items())
        ),
        "record_level_remaining_upper_bound_coarse": dict(
            sorted(upper_coarse.items())
        ),
        "full_source_conflict_free_coarse_counts": dict(
            sorted(core_coarse.items())
        ),
        "selection_pool_native_counts": dict(
            sorted(
                (
                    upper_native
                    if spec.requires_zero_historical_overlap
                    else audit["core_native_counts"]
                ).items()
            )
        ),
        "selection_pool_coarse_counts": dict(
            sorted(
                (
                    upper_coarse
                    if spec.requires_zero_historical_overlap
                    else core_coarse
                ).items()
            )
        ),
        "natural_coarse_labels_exhausted_before_modeling": exhausted_natural,
        "natural_coarse_labels_exhausted_under_independent_exclusion": (
            exhausted_natural
        ),
        "required_coarse_labels_exhausted_under_independent_exclusion": (
            independent_exhausted_required
        ),
        "required_coarse_labels_exhausted_before_modeling": exhausted_required,
        "exact_group_closure_counts_require_build": not exhausted_required,
        "model_results_read": False,
        "test_artifacts_created_or_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    marker = output / ".incomplete"
    marker.write_text("formal historical audit incomplete\n", encoding="ascii")
    history_payload = _public_history(history)
    write_json_snapshot(output / "historical_exposure.json", history_payload)
    write_json_snapshot(output / "feasibility.json", feasibility)
    write_json_snapshot(
        output / "audit_provenance.json",
        {
            "audit_config": str(audit["audit_config_path"]),
            "audit_config_hash": audit["audit_config"].config_hash,
            "audit_report_dir": str(audit["report_dir"]),
            "conflict_summary_sha256": _hash_file(
                audit["summary_path"], "sha256"
            ),
            "logical_views_sha256": _hash_file(audit["views_path"], "sha256"),
            "conflict_groups_sha256": _hash_file(
                audit["conflict_path"], "sha256"
            ),
            "raw_csv_sha1": audit["raw_csv_sha1"],
            "source_manifest_sha256": audit["source_manifest_sha256"],
            "historical_exposure_policy": spec.historical_exposure_policy,
        },
    )
    marker.unlink()
    return feasibility


def _raw_rows(audit_config, schema):
    expected_header = (
        *schema.raw_feature_names,
        audit_config.binary_label_column,
        audit_config.native_label_column,
    )
    with audit_config.raw_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("raw CSV header differs from the registered schema")
        for row_number, row in enumerate(reader, start=1):
            yield str(row_number), row


def _pairwise_overlap(records_by_split, attribute):
    values = {
        name: {
            (
                (record.source_file, record.source_row_id)
                if attribute == "source_ref"
                else getattr(record, attribute)
            )
            for record in members
        }
        for name, members in records_by_split.items()
    }
    return {
        f"{left}__{right}": len(values[left] & values[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(canonical_json(record))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_feature_cache(
    audit_config,
    schema,
    records,
    selection_ids,
    test_ids,
    output: Path,
) -> dict[str, Any]:
    by_row = {int(record.source_row_id): record for record in records}
    selection_path = output / "selection_features.parquet"
    test_path = output / "sealed_test_features.parquet"
    parquet_schema = pa.schema(
        [pa.field("sample_id", pa.string(), nullable=False)]
        + [pa.field(name, pa.string()) for name in schema.model_feature_names]
    )
    writers = {
        "selection": pq.ParquetWriter(
            selection_path, parquet_schema, compression="zstd"
        ),
        "test": pq.ParquetWriter(test_path, parquet_schema, compression="zstd"),
    }
    buffers = {"selection": [], "test": []}
    counts = Counter()

    def flush(name):
        if buffers[name]:
            writers[name].write_table(
                pa.Table.from_pylist(buffers[name], schema=parquet_schema)
            )
            buffers[name].clear()

    try:
        for row_number, row in _raw_rows(audit_config, schema):
            record = by_row.get(int(row_number))
            if record is None:
                continue
            if record.sample_id in selection_ids:
                view = "selection"
            elif record.sample_id in test_ids:
                view = "test"
            else:
                raise RuntimeError("selected record is absent from split manifests")
            buffers[view].append(
                {
                    "sample_id": record.sample_id,
                    **{
                        name: (None if row.get(name) is None else str(row[name]))
                        for name in schema.model_feature_names
                    },
                }
            )
            counts[view] += 1
            if len(buffers[view]) >= 50_000:
                flush(view)
        flush("selection")
        flush("test")
    finally:
        for writer in writers.values():
            writer.close()
    if counts["selection"] != len(selection_ids) or counts["test"] != len(test_ids):
        raise RuntimeError("feature cache did not materialize every selected row")
    return {
        "selection_features": {
            "path": str(selection_path),
            "rows": counts["selection"],
            "sha256": _hash_file(selection_path, "sha256"),
            "bytes": selection_path.stat().st_size,
        },
        "sealed_test_features": {
            "path": str(test_path),
            "rows": counts["test"],
            "sha256": _hash_file(test_path, "sha256"),
            "bytes": test_path.stat().st_size,
        },
    }


def publish_formal_cohort(
    key: str,
    output_path,
    *,
    audit_dir,
    repo_dir,
    project_root,
    data_root=Path("/root/autodl-tmp"),
) -> dict[str, Any]:
    """Build one untouched group-disjoint cohort and seal its test view."""

    spec = get_formal_dataset_spec(key)
    repo = Path(repo_dir).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    prior_audit = Path(audit_dir).expanduser().resolve()
    feasibility = _load_json(prior_audit / "feasibility.json")
    if (
        feasibility.get("dataset_key") != key
        or feasibility.get("protocol_version") != spec.protocol_version
        or feasibility.get(
            "historical_exposure_policy", HISTORY_POLICY_EXCLUDE
        )
        != spec.historical_exposure_policy
    ):
        raise ValueError("formal preflight belongs to another dataset")
    if feasibility.get("status") != spec.feasibility_status:
        raise ValueError("formal preflight does not permit cohort construction")
    output = validate_output_path(
        output_path, enforce_autodl_data_disk=True, data_root=data_root
    )
    if output.exists():
        raise ValueError(f"formal cohort output already exists: {output}")
    audit = _validated_audit_state(spec, repo_dir=repo, project_root=project)
    history = _history_inventory(spec, project_root=project)

    indexed = index_netflow_v2_rows(
        _raw_rows(audit["audit_config"], spec.schema),
        schema=spec.schema,
        label_space=spec.source_label_space,
        conflict_feature_keys=audit["conflict_feature_keys"],
        source_file_id=audit["audit_config"].source_file_id,
        coarse_label_map=audit["audit_config"].coarse_label_map,
        sensitive_hash_namespace=(
            audit["audit_config"].sensitive_hash_namespace
        ),
        leakage_group_namespace=spec.leakage_group_namespace,
    )
    if len(indexed) != sum(audit["core_native_counts"].values()):
        raise RuntimeError("indexed conflict-free rows do not match audit")
    history_ids = history["identities"]
    task_labels = set(spec.task_coarse_labels)
    task_core_native = Counter()
    task_core_coarse = Counter()
    eligible_native = Counter()
    eligible_coarse = Counter()
    eligible = []
    for record in indexed:
        if record.coarse_label not in task_labels:
            continue
        task_core_native[record.native_label] += 1
        task_core_coarse[record.coarse_label] += 1
        excluded = spec.requires_zero_historical_overlap and (
            record.sample_id in history_ids["sample_id"]
            or record.source_row_id in history_ids["source_row_id"]
            or record.feature_key in history_ids["feature_key"]
            or record.group_id in history_ids["group_id"]
            or (record.source_file, record.source_row_id)
            in history["source_refs"]
        )
        if excluded:
            continue
        eligible.append(record)
        eligible_native[record.native_label] += 1
        eligible_coarse[record.coarse_label] += 1
    missing = [
        label for label in spec.task_coarse_labels if eligible_coarse[label] <= 0
    ]
    if missing:
        raise ValueError(
            "exact historical group closure exhausts required labels: "
            + ", ".join(missing)
        )
    selected = select_netflow_v2_proportional_development_records(
        eligible,
        max_samples=spec.max_samples,
        seed=spec.split_seed,
        order_namespace=(
            f"{FORMAL_SELECTION_ORDER_NAMESPACE}.{spec.dataset_id}"
        ),
        required_native_labels=tuple(
            label
            for label in spec.task_native_labels
            if eligible_native[label] > 0
        ),
    )
    if len(selected) < int(spec.max_samples * 0.99):
        raise RuntimeError("group-safe cohort is more than 1% below target size")
    selected_overlap = {
        "sample_id": sum(
            record.sample_id in history_ids["sample_id"] for record in selected
        ),
        "source_row_id": sum(
            record.source_row_id in history_ids["source_row_id"]
            for record in selected
        ),
        "feature_key": sum(
            record.feature_key in history_ids["feature_key"] for record in selected
        ),
        "group_id": sum(
            record.group_id in history_ids["group_id"] for record in selected
        ),
        "source_ref": sum(
            (record.source_file, record.source_row_id) in history["source_refs"]
            for record in selected
        ),
    }
    if spec.requires_zero_historical_overlap and any(selected_overlap.values()):
        raise RuntimeError("formal cohort overlaps historical exposure")
    source_hash = stable_hash(
        {
            "protocol_version": spec.protocol_version,
            "dataset_id": spec.dataset_id,
            "audit_config_hash": audit["audit_config"].config_hash,
            "conflict_summary_sha256": _hash_file(
                audit["summary_path"], "sha256"
            ),
            "historical_selected_sha256": [
                artifact["selected_records_sha256"]
                for artifact in history["selected_artifacts"]
            ],
            "historical_exposure_sha256": _hash_file(
                prior_audit / "historical_exposure.json", "sha256"
            ),
            "feasibility_sha256": _hash_file(
                prior_audit / "feasibility.json", "sha256"
            ),
            "audit_provenance_sha256": _hash_file(
                prior_audit / "audit_provenance.json", "sha256"
            ),
            "task_mapping_version": spec.task_mapping_version,
            "task_coarse_labels": list(spec.task_coarse_labels),
            "selection_seed": spec.split_seed,
            "max_samples": spec.max_samples,
            "historical_exposure_policy": spec.historical_exposure_policy,
        },
        FORMAL_SOURCE_HASH_NAMESPACE,
    )
    manifest = deterministic_stratified_group_split(
        (
            {
                "sample_id": record.sample_id,
                "group_id": record.group_id,
                "stratum": record.coarse_label,
            }
            for record in selected
        ),
        ratios=SPLIT_RATIOS,
        seed=spec.split_seed,
        schema_version=spec.schema.version,
        source_hash=source_hash,
    )
    by_sample = {record.sample_id: record for record in selected}
    by_split = {
        name: tuple(by_sample[sample_id] for sample_id in manifest.splits[name])
        for name in SPLIT_NAMES
    }
    pairwise = {
        field: _pairwise_overlap(by_split, field)
        for field in ("sample_id", "feature_key", "group_id", "source_ref")
    }
    if any(value for values in pairwise.values() for value in values.values()):
        raise RuntimeError("formal cohort internal splits overlap")
    split_coarse = {
        name: Counter(record.coarse_label for record in members)
        for name, members in by_split.items()
    }
    split_native = {
        name: Counter(record.native_label for record in members)
        for name, members in by_split.items()
    }
    split_groups = {
        name: len({record.group_id for record in members})
        for name, members in by_split.items()
    }
    unsupported = {
        name: [label for label in spec.task_coarse_labels if counts[label] == 0]
        for name, counts in split_coarse.items()
    }
    unsupported = {name: labels for name, labels in unsupported.items() if labels}
    if unsupported:
        raise ValueError(
            "group-safe split omits required labels: " + json.dumps(unsupported)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    marker = output / ".incomplete"
    marker.write_text("formal cohort publication incomplete\n", encoding="ascii")
    config = {
        "format_version": FORMAL_COHORT_FORMAT_VERSION,
        "protocol_version": spec.protocol_version,
        "dataset_key": key,
        "dataset_id": spec.dataset_id,
        "audit_config": str(audit["audit_config_path"]),
        "audit_report_dir": str(audit["report_dir"]),
        "historical_audit_dir": str(prior_audit),
        "view": "conflict_free_core",
        "selection_policy": (
            "native_proportional_group_safe_after_historical_exclusion"
            if spec.requires_zero_historical_overlap
            else "native_proportional_group_safe_full_source_restart"
        ),
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent": spec.requires_zero_historical_overlap,
        "task_mapping_version": spec.task_mapping_version,
        "task_coarse_labels": list(spec.task_coarse_labels),
        "max_samples": spec.max_samples,
        "ratios": SPLIT_RATIOS,
        "seed": spec.split_seed,
        "sealed_test": True,
    }
    write_json_snapshot(output / "config.snapshot.json", config)
    shutil.copyfile(
        audit["audit_config"].source_manifest, output / "data_manifest.json"
    )
    selected_path = output / "selected_records.jsonl"
    _write_jsonl(selected_path, (record.__dict__ for record in selected))
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    selection_ids = set(manifest.splits["train"]) | set(manifest.splits["val"])
    test_ids = set(manifest.splits["test"])
    selection_records = tuple(
        record for record in selected if record.sample_id in selection_ids
    )
    test_records = tuple(record for record in selected if record.sample_id in test_ids)
    selection_records_path = output / "selection_records.jsonl"
    sealed_records_path = output / "sealed_test_records.jsonl"
    _write_jsonl(
        selection_records_path,
        (record.__dict__ for record in selection_records),
    )
    _write_jsonl(
        sealed_records_path, (record.__dict__ for record in test_records)
    )
    selection_manifest_path = output / "selection_manifest.json"
    write_json_snapshot(
        selection_manifest_path,
        {
            "format_version": 1,
            "selection_scope": ["train", "validation"],
            "schema_version": manifest.schema_version,
            "source_hash": manifest.source_hash,
            "seed": manifest.seed,
            "ratios_within_selection_view": {"train": 0.75, "val": 0.25},
            "splits": {
                "train": list(manifest.splits["train"]),
                "val": list(manifest.splits["val"]),
            },
            "sample_groups": {
                sample_id: manifest.sample_groups[sample_id]
                for sample_id in sorted(selection_ids)
            },
            "group_assignments": {
                group_id: split_name
                for group_id, split_name in manifest.group_assignments.items()
                if split_name != "test"
            },
            "test_members_present": False,
        },
    )
    sealed_manifest_path = output / "sealed_test_manifest.json"
    write_json_snapshot(
        sealed_manifest_path,
        {
            "format_version": 1,
            "dataset_id": spec.dataset_id,
            "schema_version": spec.schema.version,
            "source_hash": source_hash,
            "split_seed": spec.split_seed,
            "split": "test",
            "sample_ids": list(manifest.splits["test"]),
        },
    )
    feature_cache = _write_feature_cache(
        audit["audit_config"],
        spec.schema,
        selected,
        selection_ids,
        test_ids,
        output,
    )
    overlap = {
        "format_version": 1,
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historical_overlap_expected": (
            not spec.requires_zero_historical_overlap
        ),
        "history_identity_overlap_counts": selected_overlap,
        "internal_split_overlap_counts": pairwise,
        "verified_zero_historical_overlap": not any(selected_overlap.values()),
        "historically_independent": (
            spec.requires_zero_historical_overlap
            and not any(selected_overlap.values())
        ),
        "verified_zero_internal_overlap": not any(
            value for values in pairwise.values() for value in values.values()
        ),
    }
    validate_formal_overlap_contract(spec, overlap)
    write_json_snapshot(output / "overlap_audit.json", overlap)
    summary = {
        "format_version": FORMAL_COHORT_FORMAT_VERSION,
        "protocol_version": spec.protocol_version,
        "status": (
            "formal_cohort_published_and_test_sealed"
            if spec.requires_zero_historical_overlap
            else "restart_cohort_published_and_test_sealed"
        ),
        "dataset_id": spec.dataset_id,
        "schema_version": spec.schema.version,
        "schema_hash": spec.schema.schema_hash,
        "task_label_space": spec.task_label_space.to_dict(),
        "source_hash": source_hash,
        "seed": spec.split_seed,
        "ratios": SPLIT_RATIOS,
        "selected_rows": len(selected),
        "selected_native_counts": dict(
            sorted(Counter(r.native_label for r in selected).items())
        ),
        "selected_coarse_counts": dict(
            sorted(Counter(r.coarse_label for r in selected).items())
        ),
        "split_rows": {name: len(rows) for name, rows in by_split.items()},
        "split_group_counts": split_groups,
        "split_native_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in split_native.items()
        },
        "split_coarse_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in split_coarse.items()
        },
        "task_core_native_counts": dict(sorted(task_core_native.items())),
        "task_core_coarse_counts": dict(sorted(task_core_coarse.items())),
        "eligible_native_counts_after_group_closure": dict(
            sorted(eligible_native.items())
        ),
        "eligible_coarse_counts_after_group_closure": dict(
            sorted(eligible_coarse.items())
        ),
        "selected_records_sha256": _hash_file(selected_path, "sha256"),
        "split_manifest_sha256": _hash_file(manifest_path, "sha256"),
        "selection_records_sha256": _hash_file(
            selection_records_path, "sha256"
        ),
        "selection_manifest_sha256": _hash_file(
            selection_manifest_path, "sha256"
        ),
        "sealed_test_records_sha256": _hash_file(
            sealed_records_path, "sha256"
        ),
        "sealed_test_manifest_sha256": _hash_file(
            sealed_manifest_path, "sha256"
        ),
        "data_manifest_sha256": _hash_file(
            output / "data_manifest.json", "sha256"
        ),
        "feature_cache": feature_cache,
        "raw_csv_sha1": audit["raw_csv_sha1"],
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent": spec.requires_zero_historical_overlap,
        "historical_exposure_sha256": _hash_file(
            prior_audit / "historical_exposure.json", "sha256"
        ),
        "feasibility_sha256": _hash_file(
            prior_audit / "feasibility.json", "sha256"
        ),
        "audit_provenance_sha256": _hash_file(
            prior_audit / "audit_provenance.json", "sha256"
        ),
        "overlap_audit_sha256": _hash_file(
            output / "overlap_audit.json", "sha256"
        ),
    }
    write_json_snapshot(output / "cohort_summary.json", summary)
    seal = {
        "format_version": 1,
        "status": spec.seal_status,
        "selection_access_allowed": False,
        "opened_for_evaluation": False,
        "dataset_id": spec.dataset_id,
        "test_rows": len(test_ids),
        "sealed_test_records_sha256": summary["sealed_test_records_sha256"],
        "sealed_test_manifest_sha256": summary["sealed_test_manifest_sha256"],
        "sealed_test_features_sha256": feature_cache[
            "sealed_test_features"
        ]["sha256"],
        "full_split_manifest_sha256": summary["split_manifest_sha256"],
        "created_without_model_predictions": True,
        "historical_exposure_preexists": (
            not spec.requires_zero_historical_overlap
        ),
        "historically_independent": spec.requires_zero_historical_overlap,
    }
    write_json_snapshot(output / "sealed_test.json", seal)
    marker.unlink()
    return summary


__all__ = [
    "FORMAL_COHORT_PROTOCOL_VERSION",
    "FORMAL_RESTART_COHORT_PROTOCOL_VERSION",
    "FORMAL_REVISION_COHORT_PROTOCOL_VERSION",
    "REVISION_ACTIVE_COHORT_COUNT",
    "REVISION_COHORT_SEEDS",
    "FORMAL_DATASETS",
    "HISTORY_POLICY_DISCLOSED_RESTART",
    "HISTORY_POLICY_EXCLUDE",
    "FormalDatasetSpec",
    "audit_formal_dataset",
    "get_formal_dataset_spec",
    "publish_formal_cohort",
    "revision_cohort_key",
    "validate_formal_overlap_contract",
]
