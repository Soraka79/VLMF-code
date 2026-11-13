#!/usr/bin/env python3
"""Train one validation-only neural model for a formal dataset setting."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml

from flowfm.config import load_config
from flowfm.data.formal_dataset import load_formal_selection_samples
from flowfm.data.formal_cohort import FORMAL_DATASETS, get_formal_dataset_spec
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.training.baselines import TabularFeatureEncoder
from flowfm.training.neural_baselines import (
    categorical_training_audit,
    project_model_visible_samples,
    run_neural_baseline_training,
)
from flowfm.training.supervised import run_supervised_training


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_REPO_ROOT = _PROJECT_ROOT / "repository"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_DEFAULT_SOURCE_CHECKPOINT = (
    _RESULTS_ROOT
    / "schemaflow-nf-unsw-pretrain-smoke-20260722T054417924340Z-"
    "9965a490-7de8cdfb4f42/checkpoints/epoch-0001.pt"
)
_MODEL_SEEDS = (17, 42, 20260725)
_TRAIN_PERCENTS = (1, 5, 10)
_SCHEMAFLOW_MODEL = {
    "name": "schemaflow_fm_supervised",
    "d_model": 128,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "adapter_dim": 32,
}
_FT_MODEL = {
    "d_token": 64,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
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
        raise RuntimeError("formal training requires a clean AutoDL worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("formal training could not resolve the Git commit")
    return commit


def _verify_source_checkpoint(source_checkpoint, target_dataset_id):
    source_checkpoint = Path(source_checkpoint).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise RuntimeError("target-excluded pretraining checkpoint is missing")
    payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True
    )
    config = payload.get("config", {})
    data = config.get("data", {})
    sources = data.get("sources")
    if sources is None:
        source_dataset_ids = (data.get("dataset_id"),)
    elif isinstance(sources, list):
        source_dataset_ids = tuple(
            source.get("dataset_id") for source in sources
            if isinstance(source, dict)
        )
    else:
        source_dataset_ids = ()
    if (
        config.get("workflow") != "pretrain"
        or len(source_dataset_ids) != 1
        or not isinstance(source_dataset_ids[0], str)
        or not source_dataset_ids[0]
        or target_dataset_id in source_dataset_ids
    ):
        raise RuntimeError(
            "pretraining checkpoint is not single-source target-excluded"
        )
    return {
        "source_checkpoint": str(source_checkpoint),
        "source_dataset_id": source_dataset_ids[0],
        "target_dataset_id": target_dataset_id,
        "target_excluded": True,
    }


def _training_payload(arguments, spec, view_dir, *, pretrained):
    return {
        "workflow": "finetune",
        "experiment": {
            "name": (
                f"schemaflow-{arguments.dataset}-formal-"
                f"{arguments.train_percent}pct-seed{arguments.model_seed}-"
                + ("target-excluded-pretrained" if pretrained else "scratch")
            ),
            "output_root": str(_RESULTS_ROOT),
            "seed": arguments.model_seed,
            "enforce_autodl_data_disk": True,
        },
        "schema": {"version": spec.schema.version},
        "runtime": {"device": "cuda", "execution": "train"},
        "data": {
            "dataset_id": spec.dataset_id,
            "split_dir": str(view_dir),
            "preprocessor_dir": str(view_dir),
            "data_manifest": str(arguments.cohort_dir / "data_manifest.json"),
        },
        "model": dict(_SCHEMAFLOW_MODEL),
        "training": {
            "epochs": 50 if pretrained else 20,
            "batch_size": 64,
            "learning_rate": 0.001 if pretrained else 0.0003,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "scheduler_gamma": 0.98 if pretrained else 0.95,
            "shuffle": True,
            "class_balance": "inverse_frequency",
            "evaluation_scope": "validation_only",
            "classification_loss": "weighted_cross_entropy",
            "consistency_weight": 0.0,
            "loss_weights": {"binary": 1.0, "coarse": 1.0, "native": 1.0},
        },
    }


def _write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if path.exists() and path.read_text(encoding="utf-8") != value:
        raise RuntimeError("published formal training config changed")
    path.write_text(value, encoding="utf-8")


def _schemaflow(arguments, samples, spec, git_commit, *, pretrained):
    model_name = "schemaflow_pretrained" if pretrained else "schemaflow_scratch"
    config_path = (
        arguments.matrix_root
        / "configs"
        / (
            f"pct-{arguments.train_percent:02d}-seed-{arguments.model_seed}-"
            f"{model_name}.yaml"
        )
    )
    payload = _training_payload(
        arguments, spec, arguments.view_dir, pretrained=pretrained
    )
    _write_yaml(config_path, payload)
    config = load_config(config_path, expected_workflow="finetune")
    data_manifest = json.loads(
        (arguments.cohort_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (arguments.view_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    initialization = None
    if pretrained:
        initialization = _verify_source_checkpoint(
            arguments.source_checkpoint, spec.dataset_id
        )
    result = run_supervised_training(
        config,
        splits=samples.splits,
        tokenizer_metadata=samples.tokenizer_metadata,
        label_space=samples.label_space,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        repo_dir=_REPO_ROOT,
        data_root=_DATA_ROOT,
        git_commit=git_commit,
        config_path=config_path,
        initialize_from=arguments.source_checkpoint if pretrained else None,
        allow_cross_dataset_init=pretrained,
    )
    run_dir = Path(result["run_dir"])
    provenance = {
        "format_version": 1,
        "workflow": "cross_dataset_formal_neural_training",
        "dataset_key": arguments.dataset,
        "dataset_id": spec.dataset_id,
        "task_label_space": samples.label_space.to_dict(),
        "model": model_name,
        "model_seed": arguments.model_seed,
        "data_split_seed": spec.split_seed,
        "train_percent": arguments.train_percent,
        "selection_scope": ["train", "validation"],
        "test_samples_materialized": False,
        "sealed_test_files_read": False,
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent": spec.requires_zero_historical_overlap,
        "initialization_contract": initialization,
        "git_commit": git_commit,
    }
    write_json_snapshot(run_dir / "formal_provenance.json", provenance)
    return result


def _ft_transformer(arguments, samples, spec, git_commit):
    train = project_model_visible_samples(samples.splits["train"])
    validation = project_model_visible_samples(samples.splits["val"])
    encoder = TabularFeatureEncoder.fit(
        train,
        category_encoders=samples.tokenizer_metadata.categorical_vocabularies,
    )
    encoder_metadata = encoder.to_dict()
    encoder_metadata["output_dimension"] = len(encoder.output_feature_names)
    categorical_names = tuple(
        feature.name
        for feature in spec.schema.model_features
        if feature.categorical
    )
    result = run_neural_baseline_training(
        architecture="ft_transformer",
        task="coarse",
        splits={"train": train, "validation": validation},
        tokenizer_metadata=samples.tokenizer_metadata,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        label_space=samples.label_space,
        legacy_encoder_metadata=encoder_metadata,
        legacy_matrices=None,
        categorical_audit=categorical_training_audit(
            samples.splits["train"],
            categorical_names=categorical_names,
            registry=SCHEMAFLOW_MAIN_REGISTRY,
        ),
        data_manifest_path=arguments.cohort_dir / "data_manifest.json",
        split_manifest_path=arguments.view_dir / "selection_manifest.json",
        output_root=_RESULTS_ROOT,
        repo_dir=_REPO_ROOT,
        git_commit=git_commit,
        device="cuda",
        seed=arguments.model_seed,
        epochs=20,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.01,
        gradient_clip=1.0,
        model_config=_FT_MODEL,
        evaluation_scope="validation_only",
        data_root=_DATA_ROOT,
    )
    run_dir = Path(result["run_dir"])
    write_json_snapshot(
        run_dir / "formal_provenance.json",
        {
            "format_version": 1,
            "workflow": "cross_dataset_formal_neural_training",
            "dataset_key": arguments.dataset,
            "dataset_id": spec.dataset_id,
            "task_label_space": samples.label_space.to_dict(),
            "model": "ft_transformer",
            "model_config": _FT_MODEL,
            "model_seed": arguments.model_seed,
            "data_split_seed": spec.split_seed,
            "train_percent": arguments.train_percent,
            "selection_scope": ["train", "validation"],
            "test_samples_materialized": False,
            "sealed_test_files_read": False,
            "historical_exposure_policy": spec.historical_exposure_policy,
            "historically_independent": spec.requires_zero_historical_overlap,
            "git_commit": git_commit,
        },
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train one formal validation-only neural model"
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(FORMAL_DATASETS),
        required=True,
    )
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=_DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument("--model-seed", type=int, choices=_MODEL_SEEDS, required=True)
    parser.add_argument(
        "--train-percent", type=int, choices=_TRAIN_PERCENTS, required=True
    )
    parser.add_argument(
        "--model",
        choices=("schemaflow_scratch", "schemaflow_pretrained", "ft_transformer"),
        required=True,
    )
    return parser


def run(arguments):
    arguments.cohort_dir = validate_output_path(
        arguments.cohort_dir,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    arguments.view_dir = validate_output_path(
        arguments.view_dir,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    arguments.matrix_root = validate_output_path(
        arguments.matrix_root,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    arguments.source_checkpoint = validate_output_path(
        arguments.source_checkpoint,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    spec = get_formal_dataset_spec(arguments.dataset)
    samples = load_formal_selection_samples(
        arguments.dataset,
        arguments.cohort_dir,
        arguments.view_dir,
        data_root=_DATA_ROOT,
    )
    git_commit = _git_commit()
    if arguments.model == "schemaflow_scratch":
        return _schemaflow(arguments, samples, spec, git_commit, pretrained=False)
    if arguments.model == "schemaflow_pretrained":
        return _schemaflow(arguments, samples, spec, git_commit, pretrained=True)
    return _ft_transformer(arguments, samples, spec, git_commit)


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
