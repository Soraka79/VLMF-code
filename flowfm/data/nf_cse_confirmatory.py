"""Independent NF-CSE shared-coarse confirmation split publication."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import NF_CSE_LABEL_SPACE, SHARED_COARSE_LABELS
from flowfm.data.netflow_v2_protocol import (
    index_netflow_v2_rows,
    select_netflow_v2_proportional_development_records,
)
from flowfm.data.nf_cse_dataset import _load_selected_records
from flowfm.data.nf_cse_split import (
    LEAKAGE_GROUP_NAMESPACE,
    SPLIT_CONFIG_NAMESPACE,
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
    _validated_ratios,
    _write_text,
)
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
    NF_CSE_CIC_IDS2018_V2_SCHEMA,
)
from flowfm.data.splits import deterministic_stratified_group_split
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot


CONFIRMATORY_ORDER_NAMESPACE = (
    "flowfm.nf-cse-cic-ids2018-v2-confirmatory-group-order.v1"
)
CONFIRMATORY_SOURCE_NAMESPACE = (
    "flowfm-nf-cse-cic-ids2018-v2-confirmatory-split-source-v1"
)
CONFIRMATORY_FORMAT_VERSION = 1
SUPPORTED_COARSE_LABELS = tuple(
    label
    for label in SHARED_COARSE_LABELS
    if label in set(NF_CSE_LABEL_SPACE.native_to_coarse.values())
)
_CONFIG_FIELDS = {
    "audit_config",
    "audit_report_dir",
    "exclusion_split_dirs",
    "history_root",
    "max_samples",
    "ratios",
    "seed",
    "selection_policy",
    "view",
}
_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class NFCSEConfirmatorySplitConfig:
    audit_config_path: Path
    audit_report_dir: Path
    exclusion_split_dirs: tuple[Path, ...]
    history_root: Path
    view: str
    max_samples: int
    ratios: dict[str, float]
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
            "history_root": str(self.history_root),
            "max_samples": self.max_samples,
            "ratios": dict(self.ratios),
            "seed": self.seed,
            "selection_policy": self.selection_policy,
            "view": self.view,
        }


def load_nf_cse_confirmatory_split_config(path) -> NFCSEConfirmatorySplitConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"confirmatory split config does not exist: {config_path}")
    payload = _load_yaml(config_path)
    if set(payload) != _CONFIG_FIELDS:
        raise ValueError(
            "invalid confirmatory split config fields; missing={}, extra={}".format(
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
    history_root = Path(
        _require_text(payload["history_root"], "history_root")
    ).expanduser().resolve()
    if not audit_config_path.is_file():
        raise ValueError("audit_config must reference an existing file")
    if not audit_report_dir.is_dir():
        raise ValueError("audit_report_dir must reference an existing directory")
    if not history_root.is_dir():
        raise ValueError("history_root must reference an existing directory")
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
    if type(max_samples) is not int or not 0 < max_samples <= 2_000_000:
        raise ValueError("confirmatory max_samples must be in [1, 2000000]")
    seed = payload["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("confirmatory seed must be a non-negative integer")
    ratios = _validated_ratios(payload["ratios"])
    if ratios != {"train": 0.6, "val": 0.2, "test": 0.2}:
        raise ValueError("confirmatory ratios must be 0.6/0.2/0.2")
    selection_policy = _require_text(
        payload["selection_policy"], "selection_policy"
    )
    if selection_policy != "native_proportional_group_safe_after_exclusion":
        raise ValueError(
            "selection_policy must be "
            "native_proportional_group_safe_after_exclusion"
        )
    normalized = {
        "audit_config": str(audit_config_path),
        "audit_report_dir": str(audit_report_dir),
        "exclusion_split_dirs": [str(path) for path in exclusions],
        "history_root": str(history_root),
        "max_samples": max_samples,
        "ratios": ratios,
        "seed": seed,
        "selection_policy": selection_policy,
        "view": view,
    }
    return NFCSEConfirmatorySplitConfig(
        audit_config_path=audit_config_path,
        audit_report_dir=audit_report_dir,
        exclusion_split_dirs=exclusions,
        history_root=history_root,
        view=view,
        max_samples=max_samples,
        ratios=ratios,
        seed=seed,
        selection_policy=selection_policy,
        config_hash=stable_hash(normalized, SPLIT_CONFIG_NAMESPACE),
    )


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _load_structured_config(path: Path):
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _history_inventory(config: NFCSEConfirmatorySplitConfig) -> dict[str, Any]:
    dataset_id = NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id
    split_root = config.history_root / "data" / "splits" / dataset_id
    discovered = set()
    if split_root.is_dir():
        discovered.update(
            path.parent.resolve()
            for path in split_root.rglob("selected_records.jsonl")
        )
    configured = set(config.exclusion_split_dirs)
    unexcluded = sorted(discovered - configured)
    if unexcluded:
        raise ValueError(
            "historical NF-CSE split artifacts are not excluded: "
            + ", ".join(str(path) for path in unexcluded)
        )

    candidate_roots = (
        config.history_root / "data" / "preprocessing" / dataset_id,
        config.history_root / "results",
    )
    artifact_files = set()
    for root in candidate_roots:
        if not root.is_dir():
            continue
        for name in ("config.snapshot.json", "config.json", "config.yaml"):
            artifact_files.update(root.rglob(name))
    references = []
    referenced_splits = set()
    split_prefix = str(split_root.resolve()) + os.sep
    for path in sorted(artifact_files):
        payload = _load_structured_config(path)
        if payload is None:
            continue
        strings = tuple(_walk_strings(payload))
        matched = dataset_id in strings
        paths = set()
        for value in strings:
            if not value.startswith("/"):
                continue
            resolved = str(Path(value).expanduser().resolve())
            if resolved.startswith(split_prefix):
                paths.add(resolved)
                referenced_splits.add(Path(resolved))
                matched = True
        if matched:
            references.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _hash_file(path, "sha256"),
                    "referenced_split_dirs": sorted(paths),
                }
            )
    missing_references = sorted(referenced_splits - configured)
    if missing_references:
        raise ValueError(
            "historical result/preprocessing provenance references unexcluded "
            "NF-CSE splits: "
            + ", ".join(str(path) for path in missing_references)
        )
    return {
        "history_root": str(config.history_root),
        "discovery_rule": (
            "all selected_records.jsonl below data/splits/NF-CSE-CIC-IDS2018-v2 "
            "plus structured result/preprocessor config references"
        ),
        "discovered_split_dirs": [str(path) for path in sorted(discovered)],
        "configured_exclusion_split_dirs": [
            str(path) for path in config.exclusion_split_dirs
        ],
        "structured_provenance_artifacts": references,
        "unexcluded_discovered_split_dirs": [],
        "unexcluded_referenced_split_dirs": [],
        "complete": True,
    }


def _load_exclusions(config: NFCSEConfirmatorySplitConfig, audit_state):
    sample_ids: set[str] = set()
    feature_keys: set[str] = set()
    group_ids: set[str] = set()
    source_refs: set[tuple[str, str]] = set()
    native_counts = Counter()
    coarse_counts = Counter()
    artifacts = []
    for split_dir in config.exclusion_split_dirs:
        if (split_dir / ".incomplete").exists():
            raise ValueError(f"exclusion split is incomplete: {split_dir}")
        rejected_path = split_dir / "REJECTED.json"
        artifact_status = "completed"
        rejected_sha256 = None
        if rejected_path.exists():
            rejected = _load_json_object(rejected_path, "exclusion rejection")
            current_marker = (
                rejected.get("status") == "rejected"
                and rejected.get("downstream_use_allowed") is False
            )
            legacy_marker = (
                rejected.get("status")
                == "rejected_for_training_and_experiment_use"
                and isinstance(rejected.get("reason"), str)
                and bool(rejected["reason"])
                and isinstance(rejected.get("required_action"), str)
                and bool(rejected["required_action"])
            )
            if not (current_marker or legacy_marker):
                raise ValueError("exclusion rejection marker is malformed")
            artifact_status = "rejected_but_defensively_excluded"
            rejected_sha256 = _hash_file(rejected_path, "sha256")
        required = {
            "selected": split_dir / "selected_records.jsonl",
            "summary": split_dir / "split_summary.json",
            "manifest": split_dir / "data_manifest.json",
        }
        if any(not path.is_file() for path in required.values()):
            raise ValueError(f"exclusion split artifacts are incomplete: {split_dir}")
        summary = _load_json_object(required["summary"], "exclusion summary")
        expected = {
            "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
            "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
            "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
            "raw_csv_sha1": audit_state["raw_csv_sha1"],
            "source_manifest_sha256": audit_state["source_manifest_sha256"],
        }
        for field, value in expected.items():
            if summary.get(field) != value:
                raise ValueError(
                    f"exclusion split {field} does not match the audited source"
                )
        namespace = summary.get("leakage_group_namespace")
        if namespace not in (None, LEAKAGE_GROUP_NAMESPACE):
            raise ValueError("exclusion split leakage group namespace changed")
        if namespace is None and summary.get("derivation") != "few_shot_train_only":
            raise ValueError("exclusion split does not prove its leakage namespace")
        selected_sha256 = _hash_file(required["selected"], "sha256")
        if summary.get("selected_records_sha256") != selected_sha256:
            raise ValueError("exclusion selected-record hash does not match summary")
        if _hash_file(required["manifest"], "sha256") != audit_state[
            "source_manifest_sha256"
        ]:
            raise ValueError("exclusion data manifest does not match source")
        records = _load_selected_records(required["selected"])
        before = (
            len(sample_ids),
            len(feature_keys),
            len(group_ids),
            len(source_refs),
        )
        for record in records:
            if record["sample_id"] not in sample_ids:
                native_counts[record["native_label"]] += 1
                coarse_counts[record["coarse_label"]] += 1
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
                "rejection_status": (
                    rejected.get("status") if rejected_path.exists() else None
                ),
                "rejected_marker_sha256": rejected_sha256,
                "selected_records_sha256": selected_sha256,
                "summary_sha256": _hash_file(required["summary"], "sha256"),
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
        "coarse_counts": coarse_counts,
        "artifacts": artifacts,
    }


def _source_native_counts(audit_state) -> dict[str, int]:
    source = _load_json_object(
        audit_state["audit_config"].source_manifest, "source manifest"
    )
    try:
        counts = {
            label: int(value)
            for label, value in source["label_counts"]["native_attack"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source manifest has invalid native label counts") from exc
    if set(counts) != set(NF_CSE_LABEL_SPACE.native_labels):
        raise ValueError("source native counts do not match NF-CSE label space")
    return counts


def _coarse_counts(native_counts) -> dict[str, int]:
    counts = Counter()
    for native_label, count in native_counts.items():
        counts[NF_CSE_LABEL_SPACE.native_to_coarse[native_label]] += count
    return {label: counts[label] for label in SUPPORTED_COARSE_LABELS}


def _distribution(counts, total, labels):
    return {label: float(counts.get(label, 0) / total) for label in labels}


def _availability_audit(source_counts, view_counts, eligible_counts):
    labels = tuple(source_counts)
    if set(view_counts) - set(labels) or set(eligible_counts) - set(labels):
        raise ValueError("availability counts contain unknown labels")
    for label in labels:
        source = int(source_counts[label])
        view = int(view_counts.get(label, 0))
        eligible = int(eligible_counts.get(label, 0))
        if not 0 <= eligible <= view <= source:
            raise RuntimeError(
                f"invalid availability counts for {label}: "
                f"source={source}, view={view}, eligible={eligible}"
            )
    return {
        "conflict_quarantine_exhausted_labels": [
            label for label in labels if view_counts.get(label, 0) == 0
        ],
        "historically_exhausted_labels": [
            label
            for label in labels
            if view_counts.get(label, 0) > 0
            and eligible_counts.get(label, 0) == 0
        ],
        "available_labels": [
            label for label in labels if eligible_counts.get(label, 0) > 0
        ],
    }


def _group_label_counts(records, field):
    groups = {}
    for record in records:
        groups.setdefault(getattr(record, field), set()).add(record.group_id)
    return {label: len(groups.get(label, ())) for label in sorted(groups)}


def _pairwise_overlap(records_by_split, attribute):
    values = {
        split_name: {
            (
                (record.source_file, record.source_row_id)
                if attribute == "source_ref"
                else getattr(record, attribute)
            )
            for record in records
        }
        for split_name, records in records_by_split.items()
    }
    result = {}
    for index, first in enumerate(_SPLIT_NAMES):
        for second in _SPLIT_NAMES[index + 1 :]:
            result[f"{first}__{second}"] = len(values[first] & values[second])
    return result


def publish_nf_cse_confirmatory_split(
    config_path,
    output_path,
    *,
    repo_dir=None,
    data_root=Path("/root/autodl-tmp"),
):
    """Publish an unused-record, group-safe NF-CSE train/val/sealed-test split."""

    config = load_nf_cse_confirmatory_split_config(config_path)
    output_dir = validate_output_path(
        output_path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if output_dir.exists():
        raise ValueError(f"confirmatory split output already exists: {output_dir}")
    audit_state = _validated_audit_state(config)
    history = _history_inventory(config)
    exclusions = _load_exclusions(config, audit_state)
    source_native_counts = _source_native_counts(audit_state)
    source_coarse_counts = _coarse_counts(source_native_counts)

    indexed = index_netflow_v2_rows(
        _raw_rows(audit_state["audit_config"]),
        schema=NF_CSE_CIC_IDS2018_V2_SCHEMA,
        label_space=NF_CSE_LABEL_SPACE,
        conflict_feature_keys=audit_state["conflict_feature_keys"],
        source_file_id=audit_state["audit_config"].source_file_id,
        coarse_label_map=audit_state["audit_config"].coarse_label_map,
        sensitive_hash_namespace=(
            audit_state["audit_config"].sensitive_hash_namespace
        ),
        leakage_group_namespace=LEAKAGE_GROUP_NAMESPACE,
    )
    if len(indexed) != audit_state["expected_view_rows"]:
        raise RuntimeError("indexed rows do not match the audited conflict-free view")

    view_native = Counter()
    view_coarse = Counter()
    eligible_native = Counter()
    eligible_coarse = Counter()
    eligible_rows = 0

    def eligible_records():
        nonlocal eligible_rows
        for record in indexed:
            view_native[record.native_label] += 1
            view_coarse[record.coarse_label] += 1
            if (
                record.sample_id in exclusions["sample_ids"]
                or record.feature_key in exclusions["feature_keys"]
                or record.group_id in exclusions["group_ids"]
                or (record.source_file, record.source_row_id)
                in exclusions["source_refs"]
            ):
                continue
            eligible_rows += 1
            eligible_native[record.native_label] += 1
            eligible_coarse[record.coarse_label] += 1
            yield record

    selected = select_netflow_v2_proportional_development_records(
        eligible_records(),
        max_samples=config.max_samples,
        seed=config.seed,
        order_namespace=CONFIRMATORY_ORDER_NAMESPACE,
        required_native_labels=None,
    )
    native_availability = _availability_audit(
        source_native_counts,
        view_native,
        eligible_native,
    )
    coarse_availability = _availability_audit(
        source_coarse_counts,
        view_coarse,
        eligible_coarse,
    )
    exhausted_native = native_availability["historically_exhausted_labels"]
    exhausted_coarse = coarse_availability["historically_exhausted_labels"]
    if coarse_availability["conflict_quarantine_exhausted_labels"]:
        raise ValueError(
            "conflict-free view exhausts required NF-CSE coarse labels: "
            + ", ".join(
                coarse_availability["conflict_quarantine_exhausted_labels"]
            )
        )
    if exhausted_coarse:
        raise ValueError(
            "historical exclusions exhaust required NF-CSE coarse labels: "
            + ", ".join(exhausted_coarse)
        )
    if len(selected) < max(1, int(config.max_samples * 0.99)):
        raise RuntimeError("group-safe selection fell more than 1% below max_samples")
    selected_coarse = Counter(record.coarse_label for record in selected)
    missing_selected_coarse = [
        label for label in SUPPORTED_COARSE_LABELS if selected_coarse[label] == 0
    ]
    if missing_selected_coarse:
        raise RuntimeError(
            "proportional selection omitted required NF-CSE coarse labels: "
            + ", ".join(missing_selected_coarse)
        )

    selected_overlap = {
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
    if any(selected_overlap.values()):
        raise RuntimeError(f"confirmatory records overlap history: {selected_overlap}")

    source_hash = stable_hash(
        {
            "audit_config_hash": audit_state["audit_config"].config_hash,
            "audit_summary_sha256": audit_state["audit_summary_sha256"],
            "exclusion_selected_sha256": [
                artifact["selected_records_sha256"]
                for artifact in exclusions["artifacts"]
            ],
            "history_inventory": history,
            "raw_csv_sha1": audit_state["raw_csv_sha1"],
            "selection_policy": config.selection_policy,
            "source_manifest_sha256": audit_state["source_manifest_sha256"],
            "view": config.view,
        },
        CONFIRMATORY_SOURCE_NAMESPACE,
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
        ratios=config.ratios,
        seed=config.seed,
        schema_version=NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        source_hash=source_hash,
    )
    records_by_sample = {record.sample_id: record for record in selected}
    records_by_split = {
        split_name: tuple(
            records_by_sample[sample_id]
            for sample_id in manifest.splits[split_name]
        )
        for split_name in _SPLIT_NAMES
    }
    missing_split_coarse = {
        split_name: [
            label
            for label in SUPPORTED_COARSE_LABELS
            if not any(record.coarse_label == label for record in members)
        ]
        for split_name, members in records_by_split.items()
    }
    missing_split_coarse = {
        split_name: labels
        for split_name, labels in missing_split_coarse.items()
        if labels
    }
    if missing_split_coarse:
        raise RuntimeError(
            "confirmatory split does not cover every coarse label: "
            f"{missing_split_coarse}"
        )
    internal_overlap = {
        attribute: _pairwise_overlap(records_by_split, attribute)
        for attribute in ("sample_id", "feature_key", "group_id", "source_ref")
    }
    if any(
        value
        for pairwise in internal_overlap.values()
        for value in pairwise.values()
    ):
        raise RuntimeError(f"confirmatory split has internal overlap: {internal_overlap}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / ".incomplete"
    marker.write_text("NF-CSE confirmatory split incomplete\n", encoding="utf-8")
    config_out = output_dir / "config.snapshot.json"
    data_manifest_path = output_dir / "data_manifest.json"
    selected_path = output_dir / "selected_records.jsonl"
    manifest_path = output_dir / "split_manifest.json"
    summary_path = output_dir / "split_summary.json"
    overlap_path = output_dir / "overlap_audit.json"
    seal_path = output_dir / "sealed_test.json"
    write_json_snapshot(config_out, config.to_dict())
    data_manifest_path.write_bytes(
        audit_state["audit_config"].source_manifest.read_bytes()
    )
    _write_text(
        selected_path,
        "".join(canonical_json(asdict(record)) + "\n" for record in selected),
    )
    _write_text(manifest_path, manifest.to_json() + "\n")

    split_counts = {}
    for split_name, members in records_by_split.items():
        split_counts[split_name] = {
            "rows": len(members),
            "groups": len({record.group_id for record in members}),
            "binary_labels": _label_counts(members, "binary_label"),
            "coarse_labels": _label_counts(members, "coarse_label"),
            "coarse_label_groups": _group_label_counts(members, "coarse_label"),
            "native_labels": _label_counts(members, "native_label"),
            "native_label_groups": _group_label_counts(members, "native_label"),
        }
    overlap_audit = {
        "format_version": CONFIRMATORY_FORMAT_VERSION,
        "history_inventory": history,
        "exclusion_artifacts": exclusions["artifacts"],
        "excluded_unique_counts": {
            "sample_ids": len(exclusions["sample_ids"]),
            "feature_keys": len(exclusions["feature_keys"]),
            "group_ids": len(exclusions["group_ids"]),
            "source_refs": len(exclusions["source_refs"]),
        },
        "excluded_native_label_counts": dict(
            sorted(exclusions["native_counts"].items())
        ),
        "excluded_coarse_label_counts": dict(
            sorted(exclusions["coarse_counts"].items())
        ),
        "full_source_native_label_counts": source_native_counts,
        "full_source_coarse_label_counts": source_coarse_counts,
        "conflict_free_view_native_label_counts": {
            label: view_native[label]
            for label in NF_CSE_LABEL_SPACE.native_labels
        },
        "conflict_free_view_coarse_label_counts": {
            label: view_coarse[label] for label in SUPPORTED_COARSE_LABELS
        },
        "eligible_after_history_native_label_counts": {
            label: eligible_native[label]
            for label in NF_CSE_LABEL_SPACE.native_labels
        },
        "eligible_after_history_coarse_label_counts": {
            label: eligible_coarse[label] for label in SUPPORTED_COARSE_LABELS
        },
        "conflict_quarantine_exhausted_native_labels": native_availability[
            "conflict_quarantine_exhausted_labels"
        ],
        "conflict_quarantine_exhausted_coarse_labels": coarse_availability[
            "conflict_quarantine_exhausted_labels"
        ],
        "historically_exhausted_native_labels": exhausted_native,
        "historically_exhausted_coarse_labels": exhausted_coarse,
        "selected_vs_history_overlap_counts": selected_overlap,
        "internal_cross_split_overlap_counts": internal_overlap,
        "verified_zero_overlap": not any(selected_overlap.values()) and not any(
            value
            for pairwise in internal_overlap.values()
            for value in pairwise.values()
        ),
    }
    write_json_snapshot(overlap_path, overlap_audit)
    sealed_test = {
        "format_version": 1,
        "status": "sealed_before_model_selection",
        "selection_access_allowed": False,
        "test_rows": len(manifest.splits["test"]),
        "test_sample_order_sha256": stable_hash(
            list(manifest.splits["test"]),
            "flowfm-nf-cse-confirmatory-test-sample-order-v1",
        ),
        "test_target_order_sha256": stable_hash(
            [
                records_by_sample[sample_id].coarse_label
                for sample_id in manifest.splits["test"]
            ],
            "flowfm-nf-cse-confirmatory-test-target-order-v1",
        ),
        "opened_for_evaluation": False,
    }
    write_json_snapshot(seal_path, sealed_test)
    selected_native = Counter(record.native_label for record in selected)
    summary = {
        "format_version": CONFIRMATORY_FORMAT_VERSION,
        "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
        "purpose": "record_level_independent_a0_mr_shared_coarse_confirmation",
        "view": config.view,
        "selected_rows": len(selected),
        "selected_groups": len({record.group_id for record in selected}),
        "max_samples": config.max_samples,
        "seed": config.seed,
        "selection_policy": config.selection_policy,
        "selection_order_namespace": CONFIRMATORY_ORDER_NAMESPACE,
        "ratios": dict(config.ratios),
        "split_counts": split_counts,
        "supported_coarse_labels": list(SUPPORTED_COARSE_LABELS),
        "conflict_quarantine_exhausted_native_labels": native_availability[
            "conflict_quarantine_exhausted_labels"
        ],
        "historically_exhausted_native_labels": exhausted_native,
        "available_native_labels": native_availability["available_labels"],
        "coarse_label_mapping_version": NF_CSE_LABEL_SPACE.mapping_version,
        "native_labels": dict(sorted(selected_native.items())),
        "coarse_labels": dict(sorted(selected_coarse.items())),
        "eligible_native_distribution": _distribution(
            eligible_native, eligible_rows, NF_CSE_LABEL_SPACE.native_labels
        ),
        "selected_native_distribution": _distribution(
            selected_native, len(selected), NF_CSE_LABEL_SPACE.native_labels
        ),
        "eligible_coarse_distribution": _distribution(
            eligible_coarse, eligible_rows, SUPPORTED_COARSE_LABELS
        ),
        "selected_coarse_distribution": _distribution(
            selected_coarse, len(selected), SUPPORTED_COARSE_LABELS
        ),
        "config_hash": config.config_hash,
        "audit_config_hash": audit_state["audit_config"].config_hash,
        "audit_summary_sha256": audit_state["audit_summary_sha256"],
        "source_manifest_sha256": audit_state["source_manifest_sha256"],
        "raw_csv_sha1": audit_state["raw_csv_sha1"],
        "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
        "schema_hash": NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
        "label_space_hash": NF_CSE_LABEL_SPACE.label_space_hash,
        "leakage_group_namespace": LEAKAGE_GROUP_NAMESPACE,
        "source_hash": source_hash,
        "selected_records_sha256": _hash_file(selected_path, "sha256"),
        "split_manifest_sha256": _hash_file(manifest_path, "sha256"),
        "overlap_audit_sha256": _hash_file(overlap_path, "sha256"),
        "sealed_test_sha256": _hash_file(seal_path, "sha256"),
        "test_access_during_publication": "manifest_and_predeclared_counts_only",
        "git_commit": _git_commit(
            Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        ),
    }
    write_json_snapshot(summary_path, summary)
    marker.unlink()
    return {
        "selected_rows": len(selected),
        "selected_groups": summary["selected_groups"],
        "split_counts": split_counts,
        "selected_records_sha256": summary["selected_records_sha256"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
        "verified_zero_overlap": overlap_audit["verified_zero_overlap"],
        "sealed_test_sha256": summary["sealed_test_sha256"],
    }


__all__ = [
    "CONFIRMATORY_FORMAT_VERSION",
    "CONFIRMATORY_ORDER_NAMESPACE",
    "CONFIRMATORY_SOURCE_NAMESPACE",
    "NFCSEConfirmatorySplitConfig",
    "SUPPORTED_COARSE_LABELS",
    "load_nf_cse_confirmatory_split_config",
    "publish_nf_cse_confirmatory_split",
]
