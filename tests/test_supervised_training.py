import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from flowfm.config import load_config
from flowfm.cli import build_parser, dispatch
from flowfm.data.contracts import FlowSample
from flowfm.data.labels import DatasetLabelSpace
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)
from flowfm.training.supervised import (
    run_supervised_training,
    validate_supervised_splits,
)


def _sample(sample_id, token, binary, coarse, native):
    return FlowSample(
        features=(1.0 + int(sample_id), 1),
        feature_ids=("BYTES", "PROTOCOL"),
        categorical_mask=(False, True),
        missing_mask=(False, False),
        category_tokens=(None, token),
        binary_label=binary,
        coarse_label=coarse,
        native_label=native,
        dataset_id="fixture",
        sample_id=sample_id,
        feature_key=f"feature-{sample_id}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id,
    )


def _fixture():
    registry = CanonicalFeatureRegistry(
        "training-fixture-registry-v1",
        (
            CanonicalFeatureSpec(7, "BYTES", "numeric", "continuous"),
            CanonicalFeatureSpec(3, "PROTOCOL", "categorical", "vocabulary"),
        ),
    )
    training_hash = "training-fixture-sample-hash"
    vocabulary = fit_categorical_vocabulary(
        registry.by_name["PROTOCOL"],
        {"n:6": 2, "n:17": 2},
        training_row_count=4,
        training_sample_ids_hash=training_hash,
    )
    metadata = TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version="training-fixture-schema-v1",
        schema_hash="training-fixture-schema-hash",
        categorical_vocabularies={"PROTOCOL": vocabulary},
        data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
        split_manifest_sha256="split-hash",
        preprocessor_hash="preprocessor-hash",
        training_sample_ids_hash=training_hash,
    )
    label_space = DatasetLabelSpace(
        dataset_id="fixture",
        schema_version=metadata.schema_version,
        mapping_version="training-fixture-labels-v1",
        binary_labels=("benign", "attack"),
        coarse_labels=("benign", "dos"),
        native_labels=("Benign", "DoS"),
        native_to_coarse={"Benign": "benign", "DoS": "dos"},
    )
    train = (
        _sample("1", "n:6", "benign", "benign", "Benign"),
        _sample("2", "n:17", "attack", "dos", "DoS"),
        _sample("3", "n:6", "benign", "benign", "Benign"),
        _sample("4", "n:17", "attack", "dos", "DoS"),
    )
    val = (
        _sample("5", "n:6", "benign", "benign", "Benign"),
        _sample("6", "n:999", "attack", "dos", "DoS"),
    )
    return registry, metadata, label_space, {"train": train, "val": val}


def _config_text(output_root):
    return f"""workflow: finetune
experiment:
  name: supervised-fixture
  output_root: {output_root}
  seed: 17
  enforce_autodl_data_disk: true
schema:
  version: training-fixture-schema-v1
runtime:
  device: cpu
  execution: train
data:
  dataset_id: fixture
  split_dir: /root/autodl-tmp/fixture-split
  preprocessor_dir: /root/autodl-tmp/fixture-preprocessor
  data_manifest: /root/autodl-tmp/fixture-manifest.json
model:
  name: schemaflow_fm_supervised
  d_model: 8
  nhead: 2
  num_layers: 1
  dim_feedforward: 16
  dropout: 0.0
  adapter_dim: 4
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.01
  gradient_clip: 1.0
  scheduler_gamma: 0.9
  shuffle: true
  class_balance: inverse_frequency
  loss_weights:
    binary: 1.0
    coarse: 1.0
    native: 1.0
"""


class SupervisedTrainingConfigTests(unittest.TestCase):
    def test_finetune_accepts_safe_training_objective_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            path = Path(directory) / "finetune.yaml"
            path.write_text(
                _config_text(output_root)
                .replace(
                    "  class_balance: inverse_frequency\n",
                    "  class_balance: none\n"
                    "  classification_loss: balanced_softmax\n"
                    "  consistency_weight: 0.10\n"
                    "  consistency_missing_seed: 42\n"
                    "  early_stopping_patience: 2\n"
                    "  early_stopping_interval: 1\n"
                    "  early_stopping_min_delta: 0.0001\n"
                    "  evaluation_scope: validation_only\n",
                ),
                encoding="utf-8",
            )

            config = load_config(path, expected_workflow="finetune")

        self.assertEqual(config.training["classification_loss"], "balanced_softmax")
        self.assertEqual(config.training["consistency_weight"], 0.10)

    def test_balanced_softmax_rejects_inverse_frequency_double_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            path = Path(directory) / "finetune.yaml"
            path.write_text(
                _config_text(output_root).replace(
                    "  class_balance: inverse_frequency\n",
                    "  class_balance: inverse_frequency\n"
                    "  classification_loss: balanced_softmax\n",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires.*class_balance none"):
                load_config(path, expected_workflow="finetune")

    def test_finetune_accepts_validation_only_evaluation_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            path = Path(directory) / "finetune.yaml"
            path.write_text(
                _config_text(output_root).replace(
                    "  loss_weights:\n",
                    "  evaluation_scope: validation_only\n  loss_weights:\n",
                ),
                encoding="utf-8",
            )
            config = load_config(path, expected_workflow="finetune")

        self.assertEqual(config.training["evaluation_scope"], "validation_only")

    def test_train_execution_requires_workflow_specific_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                """workflow: finetune
experiment: {name: invalid, output_root: /root/autodl-tmp/results, seed: 1}
schema: {version: v1}
runtime: {device: cpu, execution: train}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "data.*model.*training"):
                load_config(path, expected_workflow="finetune")

    def test_finetune_dispatches_train_execution_to_supervised_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            path = Path(directory) / "finetune.yaml"
            path.write_text(
                _config_text(output_root).replace(
                    "dataset_id: fixture",
                    "dataset_id: NF-UNSW-NB15-v2",
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                ["finetune", "--config", str(path), "--resume-from", "checkpoint.pt"]
            )
            with mock.patch(
                "flowfm.training.supervised.run_nf_unsw_finetune",
                return_value={"status": "completed", "run_dir": "run"},
            ) as runner:
                result = dispatch(args)
            self.assertEqual(result["status"], "completed")
            runner.assert_called_once()
            self.assertEqual(runner.call_args.kwargs["resume_from"], Path("checkpoint.pt").resolve())

    def test_finetune_dispatches_pretrained_encoder_initializer(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            path = Path(directory) / "finetune.yaml"
            path.write_text(
                _config_text(output_root).replace(
                    "dataset_id: fixture",
                    "dataset_id: NF-UNSW-NB15-v2",
                ),
                encoding="utf-8",
            )
            try:
                args = build_parser().parse_args(
                    [
                        "finetune",
                        "--config",
                        str(path),
                        "--initialize-from",
                        "pretrain.pt",
                    ]
                )
            except SystemExit as exc:
                self.fail(f"finetune pretrained initializer is missing: {exc}")
            with mock.patch(
                "flowfm.training.supervised.run_nf_unsw_finetune",
                return_value={"status": "completed", "run_dir": "run"},
            ) as runner:
                result = dispatch(args)
            self.assertEqual(result["status"], "completed")
            runner.assert_called_once()
            self.assertEqual(
                runner.call_args.kwargs["initialize_from"],
                Path("pretrain.pt").resolve(),
            )
            self.assertIsNone(runner.call_args.kwargs["resume_from"])


class SupervisedTrainingTests(unittest.TestCase):
    def test_feature_key_repeats_within_split_are_allowed_but_not_across_splits(self):
        _, _, label_space, splits = _fixture()
        duplicate_train = replace(
            splits["train"][1],
            feature_key=splits["train"][0].feature_key,
        )
        allowed = {
            "train": (splits["train"][0], duplicate_train),
            "val": splits["val"],
        }
        validate_supervised_splits(allowed, label_space)

        leaking_val = replace(
            splits["val"][0],
            feature_key=splits["train"][0].feature_key,
        )
        with self.assertRaisesRegex(ValueError, "feature_key crosses"):
            validate_supervised_splits(
                {"train": splits["train"], "val": (leaking_val,)},
                label_space,
            )

    def test_one_epoch_publishes_complete_artifacts_and_verified_checkpoint(self):
        registry, metadata, label_space, splits = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "autodl-tmp"
            output_root = data_root / "results"
            config_path = Path(directory) / "finetune.yaml"
            config_path.write_text(_config_text(output_root), encoding="utf-8")
            config = load_config(config_path, expected_workflow="finetune")

            summary = run_supervised_training(
                config,
                splits=splits,
                tokenizer_metadata=metadata,
                label_space=label_space,
                registry=registry,
                data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
                split_manifest={
                    "train": [sample.sample_id for sample in splits["train"]],
                    "val": [sample.sample_id for sample in splits["val"]],
                },
                repo_dir=Path(__file__).resolve().parents[1],
                data_root=data_root,
                git_commit="c" * 40,
            )

            run_dir = Path(summary["run_dir"])
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(summary["checkpoint_verified"])
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertEqual(summary["global_step"], 2)
            self.assertFalse((run_dir / ".incomplete").exists())
            for relative in (
                "config.yaml",
                "git_commit.txt",
                "environment.txt",
                "data_manifest.json",
                "split_manifest.json",
                "schema_version.txt",
                "metrics.json",
                "train.log",
                "efficiency.json",
                "resume_command.txt",
                "checkpoints/epoch-0001.pt",
                "checkpoints/index.json",
                "predictions/validation_summary.json",
            ):
                with self.subTest(relative=relative):
                    self.assertTrue((run_dir / relative).is_file())

            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["epoch"], 1)
            self.assertIn("macro_f1", metrics["validation"]["native"])
            self.assertIn("mcc", metrics["validation"]["binary"])
            self.assertIn("pr_auc", metrics["validation"]["binary"])
            self.assertIn("ece", metrics["validation"]["binary"])
            self.assertIn("brier", metrics["validation"]["binary"])
            efficiency = json.loads((run_dir / "efficiency.json").read_text())
            self.assertGreater(efficiency["train_samples_per_second"], 0)
            self.assertGreater(efficiency["parameter_count"], 0)

            resumed = run_supervised_training(
                config,
                splits=splits,
                tokenizer_metadata=metadata,
                label_space=label_space,
                registry=registry,
                data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
                split_manifest={
                    "train": [sample.sample_id for sample in splits["train"]],
                    "val": [sample.sample_id for sample in splits["val"]],
                },
                repo_dir=Path(__file__).resolve().parents[1],
                data_root=data_root,
                git_commit="c" * 40,
                config_path=config_path,
                resume_from=summary["checkpoint"],
            )
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(resumed["global_step"], 4)


if __name__ == "__main__":
    unittest.main()
