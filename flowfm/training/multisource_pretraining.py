"""Two-or-more dataset train-only self-supervised pretraining runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from flowfm.config import FlowConfig
from flowfm.data.identity import stable_hash
from flowfm.data.multisource_tokenization import (
    build_multisource_tokenizer_metadata,
)
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.experiment.metadata import validate_output_path
from flowfm.training.pretraining import run_pretraining
from flowfm.training.supervised import _clean_git_commit, _sha256_file


MULTISOURCE_DATASET_ID = "SchemaFlow-MultiSource-UNSW-CSE-v1"
_SPLIT_NAMESPACE = "schemaflow-multisource-splits-v1"


def _dataset_components(dataset_id):
    if dataset_id == "NF-UNSW-NB15-v2":
        from flowfm.data.labels import NF_UNSW_LABEL_SPACE
        from flowfm.data.nf_unsw_dataset import load_nf_unsw_development_samples
        from flowfm.data.nf_unsw_preprocessing import load_nf_unsw_preprocessor
        from flowfm.data.nf_unsw_tokenization import (
            build_nf_unsw_tokenizer_metadata,
        )

        return (
            NF_UNSW_LABEL_SPACE,
            load_nf_unsw_development_samples,
            load_nf_unsw_preprocessor,
            build_nf_unsw_tokenizer_metadata,
        )
    if dataset_id == "NF-CSE-CIC-IDS2018-v2":
        from flowfm.data.labels import NF_CSE_LABEL_SPACE
        from flowfm.data.nf_cse_dataset import load_nf_cse_development_samples
        from flowfm.data.nf_cse_preprocessing import load_nf_cse_preprocessor
        from flowfm.data.nf_cse_tokenization import build_nf_cse_tokenizer_metadata

        return (
            NF_CSE_LABEL_SPACE,
            load_nf_cse_development_samples,
            load_nf_cse_preprocessor,
            build_nf_cse_tokenizer_metadata,
        )
    raise ValueError(f"unsupported multi-source dataset_id: {dataset_id}")


def _load_source(source, *, data_root):
    if not isinstance(source, Mapping):
        raise TypeError("multi-source entries must be mappings")
    dataset_id = source["dataset_id"]
    label_space, load_dataset, load_preprocessor, build_tokenizer = (
        _dataset_components(dataset_id)
    )
    split_dir = validate_output_path(
        source["split_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    preprocessor_dir = validate_output_path(
        source["preprocessor_dir"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    manifest_path = validate_output_path(
        source["data_manifest"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    split_manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file() or not split_manifest_path.is_file():
        raise ValueError("multi-source Manifest path does not exist")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(
        split_manifest_path.read_text(encoding="utf-8")
    )
    dataset = load_dataset(split_dir, preprocessor_dir, data_root=data_root)
    preprocessor = load_preprocessor(preprocessor_dir / "preprocessor.json")
    tokenizer = build_tokenizer(
        dataset, preprocessor, data_manifest=data_manifest
    )
    train = tuple(dataset.splits["train"])
    return {
        "dataset_id": dataset_id,
        "label_space": label_space,
        "train": train,
        "tokenizer": tokenizer,
        "data_manifest": data_manifest,
        "data_manifest_path": str(manifest_path),
        "data_manifest_sha256": _sha256_file(manifest_path),
        "split_manifest": split_manifest,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "preprocessor_dir": str(preprocessor_dir),
    }


def run_multisource_pretrain(
    config: FlowConfig,
    *,
    repo_dir,
    config_path,
    resume_from=None,
    data_root=Path("/root/autodl-tmp"),
):
    if not isinstance(config, FlowConfig) or config.workflow != "pretrain":
        raise TypeError("config must be a validated pretrain FlowConfig")
    if config.data["dataset_id"] != MULTISOURCE_DATASET_ID:
        raise ValueError("multi-source pretraining dataset_id is unsupported")
    sources = config.data.get("sources")
    if not isinstance(sources, (list, tuple)) or len(sources) < 2:
        raise ValueError("multi-source pretraining requires at least two sources")
    loaded = tuple(_load_source(source, data_root=data_root) for source in sources)
    source_ids = tuple(item["dataset_id"] for item in loaded)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("multi-source dataset_ids must be unique")
    if set(source_ids) != {"NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2"}:
        raise ValueError("pilot sources must be exactly NF-UNSW and NF-CSE")

    declared_manifest_path = validate_output_path(
        config.data["data_manifest"],
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    declared_manifest = json.loads(
        declared_manifest_path.read_text(encoding="utf-8")
    )
    if declared_manifest.get("dataset_id") != MULTISOURCE_DATASET_ID:
        raise ValueError("declared multi-source Manifest dataset_id is invalid")
    if (
        declared_manifest.get("fit_scope") != "source_train_only"
        or declared_manifest.get("target_data_used") is not False
        or declared_manifest.get("uses_validation_or_test") is not False
    ):
        raise ValueError("declared multi-source Manifest violates train-only scope")
    declared_sources = declared_manifest.get("sources")
    if not isinstance(declared_sources, list) or {
        source.get("dataset_id")
        for source in declared_sources
        if isinstance(source, Mapping)
        and source.get("split_role") == "train_only"
    } != {"NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2"}:
        raise ValueError("declared multi-source Manifest sources are invalid")
    if len(declared_sources) != 2:
        raise ValueError("declared multi-source Manifest must contain two sources")

    source_records = []
    split_records = []
    source_samples = {}
    source_tokenizers = {}
    label_spaces = {}
    for item in sorted(loaded, key=lambda value: value["dataset_id"]):
        dataset_id = item["dataset_id"]
        source_samples[dataset_id] = item["train"]
        source_tokenizers[dataset_id] = item["tokenizer"]
        label_spaces[dataset_id] = item["label_space"]
        source_records.append(
            {
                "dataset_id": dataset_id,
                "data_manifest": item["data_manifest"],
                "data_manifest_path": item["data_manifest_path"],
                "data_manifest_sha256": item["data_manifest_sha256"],
                "preprocessor_dir": item["preprocessor_dir"],
                "preprocessor_hash": item["tokenizer"].preprocessor_hash,
                "train_sample_count": len(item["train"]),
                "training_sample_ids_hash": (
                    item["tokenizer"].training_sample_ids_hash
                ),
            }
        )
        split_records.append(
            {
                "dataset_id": dataset_id,
                "split_manifest_path": item["split_manifest_path"],
                "split_manifest_sha256": item["split_manifest_sha256"],
                "train_sample_count": len(item["train"]),
            }
        )

    aggregate_manifest = {
        "format_version": 1,
        "dataset_id": MULTISOURCE_DATASET_ID,
        "declared_manifest": declared_manifest,
        "sources": source_records,
        "fit_scope": "source_train_only",
        "uses_validation_or_test": False,
    }
    aggregate_split_manifest = {
        "format_version": 1,
        "dataset_id": MULTISOURCE_DATASET_ID,
        "sources": split_records,
        "uses_validation_or_test": False,
    }
    aggregate_split_hash = stable_hash(
        aggregate_split_manifest, _SPLIT_NAMESPACE
    )
    tokenizer = build_multisource_tokenizer_metadata(
        source_samples,
        source_tokenizers,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        schema_version=config.schema["version"],
        data_manifest=aggregate_manifest,
        split_manifest_sha256=aggregate_split_hash,
    )
    combined_train = tuple(
        sample
        for dataset_id in sorted(source_samples)
        for sample in source_samples[dataset_id]
    )
    git_commit = _clean_git_commit(Path(repo_dir).expanduser().resolve())
    return run_pretraining(
        config,
        splits={"train": combined_train},
        tokenizer_metadata=tokenizer,
        label_space=label_spaces,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        data_manifest=aggregate_manifest,
        split_manifest=aggregate_split_manifest,
        repo_dir=repo_dir,
        data_root=data_root,
        git_commit=git_commit,
        config_path=config_path,
        resume_from=resume_from,
    )


__all__ = ["MULTISOURCE_DATASET_ID", "run_multisource_pretrain"]
