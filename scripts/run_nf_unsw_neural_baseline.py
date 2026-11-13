"""Run one fair single-task NF-UNSW neural strong baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from flowfm.data.labels import NF_TON_LABEL_SPACE, NF_UNSW_LABEL_SPACE
from flowfm.data.nf_ton_dataset import load_nf_ton_development_samples
from flowfm.data.nf_ton_preprocessing import load_nf_ton_preprocessor
from flowfm.data.nf_ton_tokenization import build_nf_ton_tokenizer_metadata
from flowfm.data.nf_unsw_dataset import load_nf_unsw_development_samples
from flowfm.data.nf_unsw_preprocessing import load_nf_unsw_preprocessor
from flowfm.data.nf_unsw_tokenization import build_nf_unsw_tokenizer_metadata
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.training.baselines import TabularFeatureEncoder
from flowfm.training.neural_baselines import (
    LEGACY_INPUT_DIM,
    categorical_training_audit,
    project_model_visible_samples,
    run_neural_baseline_training,
)


DEFAULT_SPLIT_DIR = (
    "/root/autodl-tmp/Res-AE-Attn/data/splits/NF-UNSW-NB15-v2/"
    "development-conflict-free-core-100k-seed42-91bec69"
)
DEFAULT_PREPROCESSOR_DIR = (
    "/root/autodl-tmp/Res-AE-Attn/data/processed/NF-UNSW-NB15-v2/"
    "development-train-only-preprocessor-d3e0749"
)
DEFAULT_DATA_MANIFEST = (
    "/root/autodl-tmp/Res-AE-Attn/data/manifests/"
    "NF-UNSW-NB15-v2.source.json"
)


def dataset_components(name: str):
    if name == "nf_unsw":
        return (
            NF_UNSW_LABEL_SPACE,
            load_nf_unsw_development_samples,
            load_nf_unsw_preprocessor,
            build_nf_unsw_tokenizer_metadata,
        )
    if name == "nf_ton":
        return (
            NF_TON_LABEL_SPACE,
            load_nf_ton_development_samples,
            load_nf_ton_preprocessor,
            build_nf_ton_tokenizer_metadata,
        )
    raise ValueError(f"unsupported dataset: {name}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one NF-UNSW neural strong-baseline task"
    )
    parser.add_argument(
        "--dataset",
        choices=("nf_unsw", "nf_ton"),
        default="nf_unsw",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("rescae_bigru", "ft_transformer", "schema_bigru"),
    )
    parser.add_argument(
        "--task", required=True, choices=("binary", "coarse", "native")
    )
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--preprocessor-dir", default=DEFAULT_PREPROCESSOR_DIR)
    parser.add_argument("--data-manifest", default=DEFAULT_DATA_MANIFEST)
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/Res-AE-Attn/results",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=0.01)
    parser.add_argument("--gradient-clip", type=_positive_float, default=1.0)
    parser.add_argument("--resume-from", type=Path)
    return parser


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.model == "rescae_bigru":
        return {}
    if args.model == "ft_transformer":
        return {
            "d_token": 64,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        }
    if args.model == "schema_bigru":
        return {
            "d_model": 128,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
        }
    raise ValueError(f"unsupported model: {args.model}")


def _clean_git_commit(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("refusing neural baseline training from a dirty worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("unable to resolve a full Git commit")
    return commit


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.seed < 0:
        raise ValueError("seed must be non-negative")
    label_space, load_dataset, load_preprocessor, build_tokenizer = (
        dataset_components(arguments.dataset)
    )
    data_root = Path("/root/autodl-tmp").resolve()
    split_dir = Path(arguments.split_dir).expanduser().resolve()
    preprocessor_dir = Path(arguments.preprocessor_dir).expanduser().resolve()
    data_manifest_path = Path(arguments.data_manifest).expanduser().resolve()
    output_root = Path(arguments.output_root).expanduser().resolve()
    for name, path in (
        ("split-dir", split_dir),
        ("preprocessor-dir", preprocessor_dir),
        ("data-manifest", data_manifest_path),
        ("output-root", output_root),
    ):
        if not path.is_relative_to(data_root):
            raise ValueError(f"{name} must be located below /root/autodl-tmp")
    split_manifest_path = split_dir / "split_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    dataset = load_dataset(
        split_dir,
        preprocessor_dir,
        data_root=data_root,
    )
    preprocessor = load_preprocessor(
        preprocessor_dir / "preprocessor.json"
    )
    tokenizer = build_tokenizer(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )

    audit = categorical_training_audit(dataset.splits["train"])
    print(
        "CATEGORICAL_TRAIN_ONLY_AUDIT "
        + json.dumps(audit, sort_keys=True, allow_nan=False),
        flush=True,
    )

    projected = {
        name: project_model_visible_samples(samples)
        for name, samples in (
            ("train", dataset.splits["train"]),
            ("validation", dataset.splits["val"]),
            ("test", dataset.splits["test"]),
        )
    }
    encoder = TabularFeatureEncoder.fit(
        projected["train"],
        category_encoders=tokenizer.categorical_vocabularies,
    )
    encoder_metadata = encoder.to_dict()
    encoder_metadata["output_dimension"] = len(encoder.output_feature_names)
    if (
        arguments.model == "rescae_bigru"
        and encoder_metadata["output_dimension"] != LEGACY_INPUT_DIM
    ):
        raise RuntimeError("legacy train-only encoder must remain exactly 3,690-wide")
    legacy_matrices = None
    if arguments.model == "rescae_bigru":
        legacy_matrices = {
            name: encoder.transform(samples)
            for name, samples in projected.items()
        }

    repo_dir = Path(__file__).resolve().parents[1]
    git_commit = _clean_git_commit(repo_dir)
    return run_neural_baseline_training(
        architecture=arguments.model,
        task=arguments.task,
        splits=projected,
        tokenizer_metadata=tokenizer,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        label_space=label_space,
        legacy_encoder_metadata=encoder_metadata,
        legacy_matrices=legacy_matrices,
        categorical_audit=audit,
        data_manifest_path=data_manifest_path,
        split_manifest_path=split_manifest_path,
        output_root=output_root,
        data_root=data_root,
        repo_dir=repo_dir,
        git_commit=git_commit,
        device=arguments.device,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip=arguments.gradient_clip,
        model_config=model_config_from_args(arguments),
        resume_from=(
            None
            if arguments.resume_from is None
            else arguments.resume_from.expanduser().resolve()
        ),
    )


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run(arguments)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
