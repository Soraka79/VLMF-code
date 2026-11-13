#!/usr/bin/env python3
"""Evaluate a trained NF-ToN SchemaFlow checkpoint under field missingness."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from flowfm.config import load_config
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.missingness import apply_deterministic_field_missingness
from flowfm.data.nf_ton_dataset import load_nf_ton_development_samples
from flowfm.data.nf_ton_preprocessing import load_nf_ton_preprocessor
from flowfm.data.nf_ton_tokenization import build_nf_ton_tokenizer_metadata
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.training.checkpoints import load_checkpoint
from flowfm.training.supervised import (
    _build_model,
    _configure_adaptation,
    _evaluate,
    _model_data_metadata,
    _sha256_file,
)


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/Res-AE-Attn/results",
        type=Path,
    )
    parser.add_argument("--missing-fraction", default=0.3, type=float)
    parser.add_argument("--missing-seed", default=17, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_state(repo_dir: Path) -> dict[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise RuntimeError("missingness evaluation requires a clean Git worktree")
    return {"commit": commit, "dirty": False, "status": []}


def _metric_deltas(clean, missing):
    keys = ("macro_f1", "mcc", "balanced_accuracy")
    return {
        task: {
            key: float(missing[task][key] - clean[task][key])
            for key in keys
        }
        for task in ("binary", "coarse", "native")
    }


def main() -> int:
    args = _args()
    if (
        args.missing_seed < 0
        or not 0.0 <= args.missing_fraction <= 1.0
    ):
        raise SystemExit("invalid missing-seed or missing-fraction")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    started = time.perf_counter()
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    data_root = Path("/root/autodl-tmp").resolve()
    if not output_root.is_relative_to(data_root):
        raise SystemExit("output-root must be located below /root/autodl-tmp")
    if not checkpoint_path.is_file():
        raise SystemExit("checkpoint does not exist")

    config = load_config(config_path, expected_workflow="finetune")
    if config.data["dataset_id"] != NF_TON_LABEL_SPACE.dataset_id:
        raise SystemExit("this focused evaluator requires NF-ToN")
    split_dir = Path(config.data["split_dir"]).expanduser().resolve()
    preprocessor_dir = Path(config.data["preprocessor_dir"]).expanduser().resolve()
    manifest_path = Path(config.data["data_manifest"]).expanduser().resolve()
    split_manifest_path = split_dir / "split_manifest.json"
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    dataset = load_nf_ton_development_samples(
        split_dir,
        preprocessor_dir,
        data_root=data_root,
    )
    preprocessor = load_nf_ton_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer = build_nf_ton_tokenizer_metadata(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )
    eligible = tuple(
        spec.name
        for spec in SCHEMAFLOW_MAIN_REGISTRY.features
        if spec.model_visible
    )
    missing_validation, validation_report = (
        apply_deterministic_field_missingness(
            dataset.splits["val"],
            fraction=args.missing_fraction,
            seed=args.missing_seed,
            eligible_feature_ids=eligible,
        )
    )
    missing_test, test_report = apply_deterministic_field_missingness(
        dataset.splits["test"],
        fraction=args.missing_fraction,
        seed=args.missing_seed,
        eligible_feature_ids=eligible,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _build_model(
        config,
        tokenizer,
        NF_TON_LABEL_SPACE,
        SCHEMAFLOW_MAIN_REGISTRY,
    ).to(device)
    adaptation = _configure_adaptation(
        model,
        config.training.get("adaptation_mode", "full"),
    )
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training["learning_rate"],
        weight_decay=config.training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.training["scheduler_gamma"],
    )
    model_data_metadata = _model_data_metadata(
        SCHEMAFLOW_MAIN_REGISTRY,
        tokenizer,
        NF_TON_LABEL_SPACE,
    )
    payload = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
        restore_rng=False,
        expected_model_data_metadata=model_data_metadata,
    )
    checkpoint_adaptation = payload["extra_state"].get("adaptation")
    if checkpoint_adaptation is not None and checkpoint_adaptation != adaptation:
        raise RuntimeError("checkpoint adaptation policy does not match config")

    batch_size = config.training["batch_size"]
    clean_validation, clean_validation_seconds = _evaluate(
        model, dataset.splits["val"], tokenizer, NF_TON_LABEL_SPACE,
        SCHEMAFLOW_MAIN_REGISTRY, device, batch_size,
    )
    missing_validation_metrics, missing_validation_seconds = _evaluate(
        model, missing_validation, tokenizer, NF_TON_LABEL_SPACE,
        SCHEMAFLOW_MAIN_REGISTRY, device, batch_size,
    )
    clean_test, clean_test_seconds = _evaluate(
        model, dataset.splits["test"], tokenizer, NF_TON_LABEL_SPACE,
        SCHEMAFLOW_MAIN_REGISTRY, device, batch_size,
    )
    missing_test_metrics, missing_test_seconds = _evaluate(
        model, missing_test, tokenizer, NF_TON_LABEL_SPACE,
        SCHEMAFLOW_MAIN_REGISTRY, device, batch_size,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / (
        "schemaflow-nf-ton-missing-fields-eval-"
        f"{timestamp}-{git['commit'][:12]}-{uuid.uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    marker = run_dir / ".incomplete"
    marker.write_text("missing-field evaluation incomplete\n", encoding="utf-8")
    try:
        evaluation_config = {
            "format_version": 1,
            "workflow": "missing_field_evaluation",
            "dataset_id": NF_TON_LABEL_SPACE.dataset_id,
            "source_training_config": config.to_dict(),
            "source_training_config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "missing_fraction": args.missing_fraction,
            "missing_seed": args.missing_seed,
            "scope": ["validation", "test"],
            "training_data_perturbed": False,
            "device": args.device,
            "git": git,
        }
        metrics = {
            "format_version": 1,
            "validation": {
                "clean": clean_validation,
                "missing": missing_validation_metrics,
                "missing_minus_clean": _metric_deltas(
                    clean_validation, missing_validation_metrics
                ),
            },
            "test": {
                "clean": clean_test,
                "missing": missing_test_metrics,
                "missing_minus_clean": _metric_deltas(
                    clean_test, missing_test_metrics
                ),
            },
        }
        missingness = {
            "training_data_perturbed": False,
            "validation": validation_report,
            "test": test_report,
        }
        efficiency = {
            "wall_seconds": time.perf_counter() - started,
            "clean_validation_seconds": clean_validation_seconds,
            "missing_validation_seconds": missing_validation_seconds,
            "clean_test_seconds": clean_test_seconds,
            "missing_test_seconds": missing_test_seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else "CPU"
            ),
        }
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": evaluation_config["checkpoint_sha256"],
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "git_commit": payload["git_commit"],
            "schema_version": payload["schema_version"],
            "adaptation": checkpoint_adaptation,
            "restore_verified": True,
        }
        _write_json(run_dir / "config.json", evaluation_config)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "missingness.json", missingness)
        predictions_dir = run_dir / "predictions"
        predictions_dir.mkdir()
        _write_json(
            predictions_dir / "validation_summary.json",
            {
                "sample_count": len(dataset.splits["val"]),
                "clean": clean_validation,
                "missing": missing_validation_metrics,
            },
        )
        _write_json(
            predictions_dir / "test_summary.json",
            {
                "sample_count": len(dataset.splits["test"]),
                "clean": clean_test,
                "missing": missing_test_metrics,
            },
        )
        _write_json(run_dir / "checkpoint_provenance.json", checkpoint)
        _write_json(run_dir / "efficiency.json", efficiency)
        _write_json(run_dir / "label_space.json", NF_TON_LABEL_SPACE.to_dict())
        _write_json(
            run_dir / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
        )
        shutil.copyfile(manifest_path, run_dir / "data_manifest.json")
        shutil.copyfile(split_manifest_path, run_dir / "split_manifest.json")
        (run_dir / "git_commit.txt").write_text(
            git["commit"] + "\n", encoding="utf-8"
        )
        marker.unlink()
    except Exception:
        raise

    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "checkpoint_restore_verified": True,
                "metrics": metrics,
                "missingness": missingness,
                "efficiency": efficiency,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
