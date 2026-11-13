#!/usr/bin/env python3
"""Train the NF-CSE five-class FT-Transformer without opening sealed test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flowfm.data.labels import NF_CSE_SHARED_COARSE_LABEL_SPACE
from flowfm.data.nf_cse_few_shot import SELECTION_MANIFEST_FILENAME
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
    NF_CSE_CIC_IDS2018_V2_SCHEMA,
)
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.training.baselines import TabularFeatureEncoder
from flowfm.training.neural_baselines import (
    categorical_training_audit,
    project_model_visible_samples,
    run_neural_baseline_training,
)
from scripts.run_nf_cse_a0_mr_confirmation import (
    _git_state,
    _load_selection_inputs,
    _sha256_file,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_DEFAULT_OUTPUT_ROOT = _DATA_ROOT / "Res-AE-Attn" / "results"
_MODEL_CONFIG = {
    "d_token": 64,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one validation-only NF-CSE FT-Transformer baseline"
    )
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--gradient-clip", default=1.0, type=float)
    parser.add_argument("--resume-from", type=Path)
    return parser


def _formal_provenance(inputs, arguments, git_commit):
    split_config = inputs["split_config"]
    return {
        "format_version": 1,
        "workflow": "nf_cse_formal_ft_transformer_validation_only",
        "task": "shared_coarse_nf_cse_supported_five_class",
        "label_names": list(NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels),
        "label_order_rule": (
            "global_shared_coarse_order_filtered_to_natural_support"
        ),
        "selection_scope": ["train", "validation"],
        "materialized_splits": ["train", "val"],
        "test_manifest_members_read": False,
        "sealed_test_file_content_read": False,
        "test_samples_materialized": False,
        "test_labels_present": False,
        "data_config_path": str(arguments.data_config.resolve()),
        "data_config_sha256": _sha256_file(arguments.data_config.resolve()),
        "split_dir": str(inputs["split_dir"]),
        "preprocessor_dir": str(inputs["preprocessor_dir"]),
        "base_confirmatory_split_dir": str(inputs["source_split_dir"]),
        "selection_manifest_sha256": _sha256_file(inputs["split_manifest_path"]),
        "selection_records_sha256": _sha256_file(
            inputs["split_selection_records_path"]
        ),
        "selection_features_sha256": _sha256_file(inputs["selection_features_path"]),
        "base_overlap_audit_sha256": _sha256_file(inputs["overlap_path"]),
        "base_sealed_test_expected_sha256": split_config[
            "source_sealed_test_sha256"
        ],
        "model_config": dict(_MODEL_CONFIG),
        "training": {
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "gradient_clip": arguments.gradient_clip,
            "class_balance": "inverse_frequency_train_only",
            "evaluation_scope": "validation_only",
        },
        "git_commit": git_commit,
    }


def run(arguments: argparse.Namespace):
    if arguments.seed < 0:
        raise ValueError("seed must be non-negative")
    config_path = arguments.data_config.expanduser().resolve()
    output_root = arguments.output_root.expanduser().resolve()
    if not output_root.is_relative_to(_DATA_ROOT):
        raise ValueError("output-root must be below /root/autodl-tmp")
    inputs = _load_selection_inputs(config_path, config_path)
    data_config = inputs["configs"][0]
    if data_config.experiment["seed"] != arguments.seed:
        raise ValueError("FT seed must match the formal data config seed")

    original_train = inputs["dataset"].splits["train"]
    projected = {
        "train": project_model_visible_samples(original_train),
        "validation": project_model_visible_samples(
            inputs["dataset"].splits["val"]
        ),
    }
    encoder = TabularFeatureEncoder.fit(
        projected["train"],
        category_encoders=inputs["tokenizer"].categorical_vocabularies,
    )
    encoder_metadata = encoder.to_dict()
    encoder_metadata["output_dimension"] = len(encoder.output_feature_names)
    encoder_metadata["fit_scope"] = "train_only"
    categorical_names = tuple(
        spec.name
        for spec in NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features
        if spec.categorical
    )
    audit = categorical_training_audit(
        original_train,
        categorical_names=categorical_names,
        registry=SCHEMAFLOW_MAIN_REGISTRY,
    )
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    summary = run_neural_baseline_training(
        architecture="ft_transformer",
        task="coarse",
        splits=projected,
        tokenizer_metadata=inputs["tokenizer"],
        registry=SCHEMAFLOW_MAIN_REGISTRY,
        label_space=NF_CSE_SHARED_COARSE_LABEL_SPACE,
        legacy_encoder_metadata=encoder_metadata,
        legacy_matrices=None,
        categorical_audit=audit,
        data_manifest_path=inputs["data_manifest_path"],
        split_manifest_path=inputs["split_manifest_path"],
        output_root=output_root,
        repo_dir=repo_dir,
        git_commit=git["commit"],
        device=arguments.device,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip=arguments.gradient_clip,
        model_config=_MODEL_CONFIG,
        evaluation_scope="validation_only",
        data_root=_DATA_ROOT,
        resume_from=(
            None
            if arguments.resume_from is None
            else arguments.resume_from.expanduser().resolve()
        ),
    )
    run_dir = Path(summary["run_dir"])
    provenance = _formal_provenance(inputs, arguments, git["commit"])
    (run_dir / "formal_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "formal_provenance.sha256").write_text(
        hashlib.sha256((run_dir / "formal_provenance.json").read_bytes()).hexdigest()
        + "  formal_provenance.json\n",
        encoding="ascii",
    )
    (run_dir / "resume_command.txt").write_text(
        "python -m scripts.run_nf_cse_ft_transformer "
        f"--data-config {config_path} --seed {arguments.seed} "
        f"--epochs {arguments.epochs} --batch-size {arguments.batch_size} "
        f"--learning-rate {arguments.learning_rate} "
        f"--weight-decay {arguments.weight_decay} "
        f"--gradient-clip {arguments.gradient_clip} "
        f"--device {arguments.device} --resume-from {summary['checkpoint']}\n",
        encoding="utf-8",
    )
    return summary


def main(argv=None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
