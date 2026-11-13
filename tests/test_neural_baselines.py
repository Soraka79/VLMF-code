import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flowfm.data.labels import NF_UNSW_LABEL_SPACE
from flowfm.training.neural_baselines import (
    LEGACY_INPUT_DIM,
    build_legacy_model,
    build_neural_model,
    build_neural_model_data_metadata,
    categorical_training_audit,
    inverse_frequency_class_weights,
    legacy_input_tensor,
    neural_probabilities,
    run_neural_baseline_training,
    task_class_count,
    train_neural_batch,
)
from flowfm.training.checkpoints import load_checkpoint, save_checkpoint


class NeuralLegacyContractTests(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def test_legacy_input_requires_exact_finite_float32_matrix(self):
        for device in self._devices():
            with self.subTest(device=device):
                matrix = np.zeros((2, LEGACY_INPUT_DIM), dtype=np.float32)
                tensor = legacy_input_tensor(matrix, device)
                self.assertEqual(tuple(tensor.shape), (2, 1, 1, 3690))
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device.type, device)
        with self.assertRaisesRegex(ValueError, "3,690"):
            legacy_input_tensor(
                np.zeros((2, LEGACY_INPUT_DIM - 1), dtype=np.float32),
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "float32"):
            legacy_input_tensor(
                np.zeros((2, LEGACY_INPUT_DIM), dtype=np.float64),
                "cpu",
            )
        invalid = np.zeros((2, LEGACY_INPUT_DIM), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            legacy_input_tensor(invalid, "cpu")

    def test_task_class_counts_are_frozen(self):
        self.assertEqual(task_class_count("binary"), 2)
        self.assertEqual(task_class_count("coarse"), 8)
        self.assertEqual(task_class_count("native"), 10)
        with self.assertRaises(ValueError):
            task_class_count("all")

    def test_legacy_structure_output_and_parameter_counts_remain_exact(self):
        expected_parameters = {
            "binary": 134275,
            "coarse": 135049,
            "native": 135307,
        }
        for task, parameter_count in expected_parameters.items():
            with self.subTest(task=task):
                model = build_legacy_model(task)
                self.assertFalse(model.use_attention_block)
                self.assertIsNone(model.feature_adapter)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    parameter_count,
                )
                inputs = torch.zeros(2, 1, 1, LEGACY_INPUT_DIM)
                encoded = model.rescae(inputs, mode="encode")
                sequence = encoded.squeeze(2).permute(0, 2, 1)
                logits = model(inputs)
                self.assertEqual(tuple(encoded.shape), (2, 32, 1, 923))
                self.assertEqual(tuple(sequence.shape), (2, 923, 32))
                self.assertEqual(
                    tuple(logits.shape),
                    (2, len(getattr(NF_UNSW_LABEL_SPACE, f"{task}_labels"))),
                )

    def test_cpu_and_cuda_forward_backward_optimizer_step(self):
        for device in self._devices():
            with self.subTest(device=device):
                torch.manual_seed(53)
                model = build_legacy_model("binary").to(device)
                inputs = legacy_input_tensor(
                    np.random.default_rng(53).random(
                        (2, LEGACY_INPUT_DIM),
                        dtype=np.float32,
                    ),
                    device,
                )
                labels = torch.tensor([0, 1], device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(logits).all().item())
                self.assertTrue(torch.isfinite(loss).item())


class NeuralSingleTaskTrainingTests(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def test_inverse_frequency_class_weights_use_selected_training_targets(self):
        weights, counts = inverse_frequency_class_weights(
            np.asarray((0, 0, 0, 1), dtype=np.int64),
            class_count=4,
            device="cpu",
        )
        self.assertEqual(counts, (3, 1, 0, 0))
        self.assertAlmostEqual(float(weights[0]), 2.0 / 3.0)
        self.assertAlmostEqual(float(weights[1]), 2.0)
        self.assertEqual(float(weights[2]), 0.0)
        self.assertEqual(float(weights[3]), 0.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            inverse_frequency_class_weights(
                np.asarray((0, 4), dtype=np.int64),
                class_count=2,
                device="cpu",
            )

    def test_all_architectures_complete_cpu_and_cuda_single_task_step(self):
        from tests.test_schemaflow_model import fixture

        registry, metadata, batch = fixture()
        rng = np.random.default_rng(59)
        legacy_matrix = rng.random((2, LEGACY_INPUT_DIM), dtype=np.float32)
        targets = np.asarray((0, 1), dtype=np.int64)
        for device in self._devices():
            for architecture in (
                "rescae_bigru",
                "ft_transformer",
                "schema_bigru",
            ):
                with self.subTest(device=device, architecture=architecture):
                    torch.manual_seed(59)
                    kwargs = {}
                    inputs = batch.to(device)
                    if architecture == "rescae_bigru":
                        inputs = legacy_input_tensor(legacy_matrix, device)
                    elif architecture == "ft_transformer":
                        kwargs = {
                            "d_token": 16,
                            "nhead": 4,
                            "num_layers": 2,
                            "dim_feedforward": 32,
                            "dropout": 0.0,
                        }
                    else:
                        kwargs = {
                            "d_model": 16,
                            "hidden_size": 8,
                            "num_layers": 2,
                            "dropout": 0.0,
                        }
                    model = build_neural_model(
                        architecture,
                        "binary",
                        tokenizer_metadata=metadata,
                        registry=registry,
                        model_config=kwargs,
                    ).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    weights, _ = inverse_frequency_class_weights(
                        targets,
                        class_count=2,
                        device=device,
                    )
                    result = train_neural_batch(
                        model,
                        architecture,
                        inputs,
                        torch.as_tensor(targets, device=device),
                        weights,
                        optimizer,
                        gradient_clip=1.0,
                    )
                    self.assertEqual(tuple(result.logits.shape), (2, 2))
                    self.assertTrue(torch.isfinite(result.logits).all().item())
                    self.assertTrue(np.isfinite(result.loss))
                    probabilities = neural_probabilities(result.logits)
                    self.assertEqual(tuple(probabilities.shape), (2, 2))
                    torch.testing.assert_close(
                        probabilities.sum(dim=1),
                        torch.ones(2, device=device),
                    )

    def test_build_model_rejects_multitask_and_unknown_architecture(self):
        from tests.test_schemaflow_model import fixture

        registry, metadata, _ = fixture()
        with self.assertRaises(ValueError):
            build_neural_model(
                "ft_transformer",
                "all",
                tokenizer_metadata=metadata,
                registry=registry,
            )
        with self.assertRaisesRegex(ValueError, "unsupported architecture"):
            build_neural_model(
                "unknown",
                "binary",
                tokenizer_metadata=metadata,
                registry=registry,
            )

    def test_checkpoint_metadata_freezes_single_task_and_fair_input_contract(self):
        from tests.test_schemaflow_model import fixture

        registry, tokenizer, _ = fixture()
        metadata = build_neural_model_data_metadata(
            architecture="ft_transformer",
            task="binary",
            model_config={"d_token": 16, "dropout": 0.0},
            registry=registry,
            tokenizer_metadata=tokenizer,
            legacy_encoder_metadata={
                "format_version": 1,
                "fit_scope": "train_only",
                "output_dimension": 2476,
            },
            class_counts=(3, 1),
            class_weights=(2.0 / 3.0, 2.0),
            categorical_audit=(
                {
                    "feature_id": 3,
                    "feature_name": "PROTOCOL",
                    "model_visible": True,
                    "vocabulary_size": 2,
                    "fit_scope": "train_only",
                },
            ),
        )

        contract = metadata["training_contract"]
        self.assertEqual(contract["architecture"], "ft_transformer")
        self.assertEqual(contract["task"], "binary")
        self.assertEqual(contract["task_labels"], ["benign", "attack"])
        self.assertEqual(contract["legacy_encoder"]["output_dimension"], 2476)
        self.assertEqual(contract["class_counts_train_only"], [3, 1])
        self.assertEqual(
            metadata["categorical_table_sizes"],
            {
                name: vocabulary.table_size
                for name, vocabulary in tokenizer.categorical_vocabularies.items()
            },
        )
        self.assertEqual(
            contract["data_manifest"], tokenizer.data_manifest
        )
        self.assertEqual(
            contract["split_manifest_sha256"],
            tokenizer.split_manifest_sha256,
        )

    def test_checkpoint_rejects_task_or_legacy_width_before_mutating_model(self):
        from tests.test_schemaflow_model import fixture

        registry, tokenizer, _ = fixture()
        metadata = build_neural_model_data_metadata(
            architecture="ft_transformer",
            task="binary",
            model_config={"d_token": 16},
            registry=registry,
            tokenizer_metadata=tokenizer,
            legacy_encoder_metadata={
                "format_version": 1,
                "fit_scope": "train_only",
                "output_dimension": LEGACY_INPUT_DIM,
            },
            class_counts=(3, 1),
            class_weights=(2.0 / 3.0, 2.0),
            categorical_audit=(),
        )
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.pt"
            save_checkpoint(
                path,
                model,
                epoch=1,
                global_step=1,
                git_commit="a" * 40,
                schema_version=tokenizer.schema_version,
                seed=17,
                config={"architecture": "ft_transformer", "task": "binary"},
                model_data_metadata=metadata,
                optimizer=optimizer,
            )
            for field, value in (("task", "coarse"), ("output_dimension", 3689)):
                with self.subTest(field=field):
                    incompatible = copy.deepcopy(metadata)
                    if field == "task":
                        incompatible["training_contract"][field] = value
                        incompatible["training_contract"]["task_labels"] = [
                            "benign",
                            "dos",
                        ]
                    else:
                        incompatible["training_contract"]["legacy_encoder"][
                            field
                        ] = value
                    fresh = torch.nn.Linear(2, 2)
                    fresh_optimizer = torch.optim.AdamW(
                        fresh.parameters(), lr=1e-3
                    )
                    before = {
                        name: tensor.detach().clone()
                        for name, tensor in fresh.state_dict().items()
                    }
                    with self.assertRaisesRegex(
                        ValueError, "model_data_metadata mismatch"
                    ):
                        load_checkpoint(
                            path,
                            fresh,
                            optimizer=fresh_optimizer,
                            expected_model_data_metadata=incompatible,
                        )
                    for name, tensor in fresh.state_dict().items():
                        torch.testing.assert_close(tensor, before[name])

    def test_one_epoch_publishes_verified_single_task_result_directory(self):
        from tests.test_supervised_training import _fixture, _sample

        registry, tokenizer, label_space, fixture_splits = _fixture()
        splits = {
            "train": fixture_splits["train"],
            "validation": fixture_splits["val"],
            "test": (
                _sample("7", "n:6", "benign", "benign", "Benign"),
                _sample("8", "n:999", "attack", "dos", "DoS"),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "autodl-tmp"
            data_root.mkdir()
            data_manifest_path = data_root / "source.json"
            split_manifest_path = data_root / "split.json"
            data_bytes = b'{"dataset_id":"fixture","sha256":"data-hash"}\n'
            split_bytes = b'{"split":"fixture"}\n'
            data_manifest_path.write_bytes(data_bytes)
            split_manifest_path.write_bytes(split_bytes)
            tokenizer = replace(
                tokenizer,
                split_manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
                metadata_hash="",
            )
            summary = run_neural_baseline_training(
                architecture="ft_transformer",
                task="binary",
                splits=splits,
                tokenizer_metadata=tokenizer,
                registry=registry,
                label_space=label_space,
                legacy_encoder_metadata={
                    "format_version": 1,
                    "fit_scope": "train_only",
                    "output_dimension": LEGACY_INPUT_DIM,
                },
                legacy_matrices=None,
                categorical_audit=(),
                data_manifest_path=data_manifest_path,
                split_manifest_path=split_manifest_path,
                output_root=data_root / "results",
                data_root=data_root,
                repo_dir=Path(__file__).resolve().parents[1],
                git_commit="b" * 40,
                device="cpu",
                seed=17,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.01,
                gradient_clip=1.0,
                model_config={
                    "d_token": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "dropout": 0.0,
                },
            )

            run_dir = Path(summary["run_dir"])
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(summary["checkpoint_verified"])
            self.assertFalse((run_dir / ".incomplete").exists())
            self.assertEqual(
                (run_dir / "data_manifest.json").read_bytes(), data_bytes
            )
            self.assertEqual(
                (run_dir / "split_manifest.json").read_bytes(), split_bytes
            )
            for relative in (
                "config.json",
                "label_space.json",
                "feature_registry.json",
                "tokenizer_metadata.json",
                "categorical_training_audit.json",
                "metrics.json",
                "environment.json",
                "efficiency.json",
                "train.log",
                "resume_command.txt",
                "checkpoints/epoch-0001.pt",
                "checkpoints/index.json",
                "predictions/validation.jsonl",
                "predictions/test.jsonl",
            ):
                with self.subTest(relative=relative):
                    self.assertTrue((run_dir / relative).is_file())
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["architecture"], "ft_transformer")
            self.assertEqual(config["task"], "binary")
            self.assertEqual(config["epochs_this_run"], 1)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertIn("macro_f1", metrics["validation"])
            self.assertIn("mcc", metrics["test"])

    def test_validation_only_training_never_requires_or_publishes_test(self):
        from tests.test_supervised_training import _fixture

        registry, tokenizer, label_space, fixture_splits = _fixture()
        splits = {
            "train": fixture_splits["train"],
            "validation": fixture_splits["val"],
        }
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "autodl-tmp"
            data_root.mkdir()
            data_manifest_path = data_root / "source.json"
            split_manifest_path = data_root / "selection.json"
            data_bytes = b'{"dataset_id":"fixture","sha256":"data-hash"}\n'
            split_bytes = b'{"split":"train-validation-only"}\n'
            data_manifest_path.write_bytes(data_bytes)
            split_manifest_path.write_bytes(split_bytes)
            tokenizer = replace(
                tokenizer,
                split_manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
                metadata_hash="",
            )
            summary = run_neural_baseline_training(
                architecture="ft_transformer",
                task="binary",
                splits=splits,
                tokenizer_metadata=tokenizer,
                registry=registry,
                label_space=label_space,
                legacy_encoder_metadata={
                    "format_version": 1,
                    "fit_scope": "train_only",
                    "output_dimension": LEGACY_INPUT_DIM,
                },
                legacy_matrices=None,
                categorical_audit=(),
                data_manifest_path=data_manifest_path,
                split_manifest_path=split_manifest_path,
                output_root=data_root / "results",
                data_root=data_root,
                repo_dir=Path(__file__).resolve().parents[1],
                git_commit="c" * 40,
                device="cpu",
                seed=17,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.01,
                gradient_clip=1.0,
                model_config={
                    "d_token": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "dropout": 0.0,
                },
                evaluation_scope="validation_only",
            )

            run_dir = Path(summary["run_dir"])
            config = json.loads((run_dir / "config.json").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(config["evaluation_scope"], "validation_only")
            self.assertIn("validation", metrics)
            self.assertNotIn("test", metrics)
            self.assertFalse((run_dir / "predictions/test.jsonl").exists())
            checkpoint = torch.load(
                summary["checkpoint"], map_location="cpu", weights_only=True
            )
            self.assertNotIn("test", checkpoint["extra_state"]["metrics"])

            with self.assertRaisesRegex(ValueError, "evaluation_scope"):
                run_neural_baseline_training(
                    architecture="ft_transformer",
                    task="binary",
                    splits={**splits, "test": fixture_splits["val"]},
                    tokenizer_metadata=tokenizer,
                    registry=registry,
                    label_space=label_space,
                    legacy_encoder_metadata={
                        "format_version": 1,
                        "fit_scope": "train_only",
                        "output_dimension": LEGACY_INPUT_DIM,
                    },
                    legacy_matrices=None,
                    categorical_audit=(),
                    data_manifest_path=data_manifest_path,
                    split_manifest_path=split_manifest_path,
                    output_root=data_root / "results",
                    data_root=data_root,
                    repo_dir=Path(__file__).resolve().parents[1],
                    git_commit="c" * 40,
                    device="cpu",
                    seed=17,
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    weight_decay=0.01,
                    gradient_clip=1.0,
                    model_config={
                        "d_token": 8,
                        "nhead": 2,
                        "num_layers": 1,
                        "dim_feedforward": 16,
                        "dropout": 0.0,
                    },
                    evaluation_scope="validation_only",
                )
class CategoricalTrainingAuditTests(unittest.TestCase):
    def _full_schema_sample(self, sample_id, token_offset):
        from flowfm.data.contracts import FlowSample
        from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA

        features = []
        categorical_mask = []
        category_tokens = []
        for index, spec in enumerate(NF_UNSW_NB15_V2_SCHEMA.model_features):
            categorical_mask.append(spec.categorical)
            if spec.categorical:
                features.append(2)
                category_tokens.append(f"n:{index + token_offset}")
            else:
                features.append(float(index))
                category_tokens.append(None)
        return FlowSample(
            features=tuple(features),
            feature_ids=NF_UNSW_NB15_V2_SCHEMA.model_feature_names,
            categorical_mask=tuple(categorical_mask),
            missing_mask=tuple(False for _ in features),
            category_tokens=tuple(category_tokens),
            binary_label="benign",
            coarse_label="benign",
            native_label="Benign",
            dataset_id=NF_UNSW_LABEL_SPACE.dataset_id,
            sample_id=sample_id,
            feature_key=f"key-{sample_id}",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="fixture.csv",
            source_row_id=sample_id,
        )

    def test_audit_reports_all_twelve_raw_categories_before_model_build(self):
        rows = categorical_training_audit(
            (
                self._full_schema_sample("1", 0),
                self._full_schema_sample("2", 100),
            )
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            [row["feature_name"] for row in rows],
            [
                "L4_SRC_PORT",
                "L4_DST_PORT",
                "PROTOCOL",
                "L7_PROTO",
                "TCP_FLAGS",
                "CLIENT_TCP_FLAGS",
                "SERVER_TCP_FLAGS",
                "ICMP_TYPE",
                "ICMP_IPV4_TYPE",
                "DNS_QUERY_ID",
                "DNS_QUERY_TYPE",
                "FTP_COMMAND_RET_CODE",
            ],
        )
        dns = next(row for row in rows if row["feature_name"] == "DNS_QUERY_ID")
        self.assertFalse(dns["model_visible"])
        self.assertEqual(dns["policy"], "audit_only")
        self.assertEqual(sum(row["model_visible"] for row in rows), 11)
        for row in rows:
            self.assertEqual(row["training_rows"], 2)
            self.assertEqual(row["vocabulary_size"], 2)
            self.assertEqual(row["singletons"], 2)
            self.assertEqual(row["frequency_le_5"], 2)
            self.assertEqual(row["missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
