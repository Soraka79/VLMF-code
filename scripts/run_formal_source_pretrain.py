#!/usr/bin/env python3
"""Train and lock the CSE-only source encoder for the NF-UNSW restart."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml

from flowfm.config import load_config
from flowfm.data.formal_cohort import get_formal_dataset_spec
from flowfm.data.formal_dataset import load_formal_selection_samples
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.training.pretraining import run_pretraining


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_REPO_ROOT = _PROJECT_ROOT / "repository"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_SOURCE_ROOT = (
    _RESULTS_ROOT
    / "cross-dataset-formal-source-pretrain-v1"
    / "nf_cse_for_nf_unsw_restart"
)
_CSE_COHORT = (
    _PROJECT_ROOT
    / "data"
    / "splits"
    / "NF-CSE-CIC-IDS2018-v2"
    / "formal-unused-1m-seed20260731-shared-coarse-v2"
)
_CSE_VIEW = _PROJECT_ROOT / "data" / "formal_views_v1" / "nf_cse" / "train-10pct"
_SOURCE_KEY = "nf_cse"
_TARGET_KEY = "nf_unsw_restart"
_PRETRAIN_SEED = 20260801


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path):
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise RuntimeError(f"formal source artifact is missing: {value}")
    return {
        "path": str(value),
        "bytes": value.stat().st_size,
        "sha256": _sha256_file(value),
    }


def _git_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("formal source pretraining requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("formal source pretraining could not resolve Git HEAD")
    return commit


def _payload(source_spec, cohort_dir, view_dir):
    return {
        "workflow": "pretrain",
        "experiment": {
            "name": "schemaflow-nf-cse-source-for-nf-unsw-restart",
            "output_root": str(_RESULTS_ROOT),
            "seed": _PRETRAIN_SEED,
            "enforce_autodl_data_disk": True,
        },
        "schema": {"version": source_spec.schema.version},
        "runtime": {"device": "cuda", "execution": "train"},
        "data": {
            "dataset_id": source_spec.dataset_id,
            "split_dir": str(view_dir),
            "preprocessor_dir": str(view_dir),
            "data_manifest": str(cohort_dir / "data_manifest.json"),
        },
        "model": {
            "name": "schemaflow_fm_pretrainer",
            "d_model": 128,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "adapter_dim": 32,
        },
        "training": {
            "epochs": 1,
            "batch_size": 256,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "scheduler_gamma": 0.95,
            "shuffle": True,
            "class_balance": "none",
            "mask_fraction": 0.3,
            "schema_corruption_fraction": 0.1,
            "contrastive_temperature": 0.1,
            "loss_weights": {"binary": 1.0, "coarse": 1.0, "native": 1.0},
        },
    }


def _write_config(path, payload):
    value = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if path.exists() and path.read_text(encoding="utf-8") != value:
        raise RuntimeError("formal source pretraining config changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _validate_lock(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(payload["checkpoint"]["path"])
    if (
        payload.get("source_dataset_key") != _SOURCE_KEY
        or payload.get("target_dataset_key") != _TARGET_KEY
        or payload.get("target_excluded") is not True
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != payload["checkpoint"]["bytes"]
        or _sha256_file(checkpoint) != payload["checkpoint"]["sha256"]
    ):
        raise RuntimeError("formal source checkpoint lock is invalid")
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    data = checkpoint_payload.get("config", {}).get("data", {})
    if (
        checkpoint_payload.get("config", {}).get("workflow") != "pretrain"
        or data.get("dataset_id") != "NF-CSE-CIC-IDS2018-v2"
        or "sources" in data
    ):
        raise RuntimeError("locked source checkpoint is not CSE-only pretraining")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the locked CSE-only source encoder for NF-UNSW restart"
    )
    parser.add_argument("--source-root", type=Path, default=_SOURCE_ROOT)
    parser.add_argument("--cohort-dir", type=Path, default=_CSE_COHORT)
    parser.add_argument("--view-dir", type=Path, default=_CSE_VIEW)
    return parser


def run(arguments):
    source_root = validate_output_path(
        arguments.source_root,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    cohort_dir = validate_output_path(
        arguments.cohort_dir,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    view_dir = validate_output_path(
        arguments.view_dir,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    lock_path = source_root / "source_checkpoint_lock.json"
    if lock_path.is_file():
        payload = _validate_lock(lock_path)
        marker = source_root / ".incomplete"
        if marker.exists():
            marker.unlink()
        return payload

    source_spec = get_formal_dataset_spec(_SOURCE_KEY)
    target_spec = get_formal_dataset_spec(_TARGET_KEY)
    if source_spec.dataset_id == target_spec.dataset_id:
        raise RuntimeError("formal source dataset is not target-excluded")
    samples = load_formal_selection_samples(
        _SOURCE_KEY, cohort_dir, view_dir, data_root=_DATA_ROOT
    )
    source_root.mkdir(parents=True, exist_ok=True)
    marker = source_root / ".incomplete"
    marker.write_text("formal source pretraining incomplete\n", encoding="ascii")
    config_path = source_root / "config.yaml"
    _write_config(config_path, _payload(source_spec, cohort_dir, view_dir))
    config = load_config(config_path, expected_workflow="pretrain")
    data_manifest = json.loads(
        (cohort_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (view_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    git_commit = _git_commit()
    result = run_pretraining(
        config,
        splits={"train": samples.splits["train"]},
        tokenizer_metadata=samples.tokenizer_metadata,
        label_space=samples.label_space,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        repo_dir=_REPO_ROOT,
        data_root=_DATA_ROOT,
        git_commit=git_commit,
        config_path=config_path,
    )
    run_dir = Path(result["run_dir"])
    checkpoint = Path(result["checkpoint"])
    provenance = {
        "format_version": 1,
        "workflow": "formal_target_excluded_source_pretraining",
        "source_dataset_key": _SOURCE_KEY,
        "source_dataset_id": source_spec.dataset_id,
        "target_dataset_key": _TARGET_KEY,
        "target_dataset_id": target_spec.dataset_id,
        "source_cohort": str(cohort_dir),
        "source_training_view": str(view_dir),
        "source_train_rows": len(samples.splits["train"]),
        "validation_samples_materialized_but_not_used": True,
        "sealed_test_files_read": False,
        "target_excluded": True,
        "git_commit": git_commit,
    }
    provenance_path = write_json_snapshot(
        run_dir / "formal_source_provenance.json", provenance
    )
    lock = {
        "format_version": 1,
        "status": "target_excluded_source_checkpoint_locked",
        "source_dataset_key": _SOURCE_KEY,
        "source_dataset_id": source_spec.dataset_id,
        "target_dataset_key": _TARGET_KEY,
        "target_dataset_id": target_spec.dataset_id,
        "target_excluded": True,
        "pretraining_seed": _PRETRAIN_SEED,
        "checkpoint": _artifact(checkpoint),
        "config": _artifact(config_path),
        "run_config": _artifact(run_dir / "config.yaml"),
        "metrics": _artifact(run_dir / "metrics.json"),
        "provenance": _artifact(provenance_path),
        "source_view_manifest": _artifact(view_dir / "selection_manifest.json"),
        "source_view_summary": _artifact(view_dir / "training_view_summary.json"),
        "source_cohort_data_manifest": _artifact(
            cohort_dir / "data_manifest.json"
        ),
        "source_test_files_read": False,
        "git_commit": git_commit,
    }
    write_json_snapshot(lock_path, lock)
    marker.unlink()
    return _validate_lock(lock_path)


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
