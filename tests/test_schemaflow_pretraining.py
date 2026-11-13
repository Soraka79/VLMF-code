import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
import yaml

from flowfm.data.contracts import FlowSample
from flowfm.data.multisource_tokenization import (
    build_multisource_tokenizer_metadata,
)
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)
from flowfm.data.labels import DatasetLabelSpace
from flowfm.config import load_config
from flowfm.models.supervised import SchemaFlowSupervisedModel
from flowfm.cli import build_parser, dispatch
from flowfm.training import checkpoints as checkpoint_io
from flowfm.training.pretraining import (
    SchemaFlowPretrainer,
    _pretraining_model_data_metadata,
    _validate_pretraining_samples,
    compute_pretraining_loss,
    make_pretraining_view,
    run_pretraining,
)
from flowfm.training.multisource_pretraining import MULTISOURCE_DATASET_ID
from flowfm.training.supervised import _model_data_metadata


def _fixture():
    registry = CanonicalFeatureRegistry(
        "pretraining-fixture-registry-v1",
        (
            CanonicalFeatureSpec(7, "BYTES", "numeric", "continuous"),
            CanonicalFeatureSpec(3, "PROTOCOL", "categorical", "vocabulary"),
        ),
    )
    training_hash = "pretraining-fixture-hash"
    vocabulary = fit_categorical_vocabulary(
        registry.by_name["PROTOCOL"],
        {"n:6": 2, "n:17": 2},
        training_row_count=4,
        training_sample_ids_hash=training_hash,
    )
    metadata = TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version="pretraining-fixture-schema-v1",
        schema_hash="pretraining-fixture-schema-hash",
        categorical_vocabularies={"PROTOCOL": vocabulary},
        data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
        split_manifest_sha256="split-hash",
        preprocessor_hash="preprocessor-hash",
        training_sample_ids_hash=training_hash,
    )
    samples = (
        FlowSample(
            features=(1.0, 2),
            feature_ids=("BYTES", "PROTOCOL"),
            categorical_mask=(False, True),
            missing_mask=(False, False),
            category_tokens=(None, "n:6"),
            binary_label="benign",
            coarse_label="benign",
            native_label="Benign",
            dataset_id="fixture",
            sample_id="1",
            feature_key="feature-1",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="fixture.csv",
            source_row_id="1",
        ),
        FlowSample(
            features=(2, 3.0),
            feature_ids=("PROTOCOL", "BYTES"),
            categorical_mask=(True, False),
            missing_mask=(False, False),
            category_tokens=("n:17", None),
            binary_label="attack",
            coarse_label="dos",
            native_label="DoS",
            dataset_id="fixture",
            sample_id="2",
            feature_key="feature-2",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="fixture.csv",
            source_row_id="2",
        ),
    )
    batch = collate_schemaflow_samples(samples, metadata, registry=registry)
    return registry, metadata, batch, samples


def _multisource_fixture():
    registry, metadata, _, samples = _fixture()
    source_samples = {
        "source-a": (
            replace(samples[0], dataset_id="source-a", sample_id="shared"),
        ),
        "source-b": (
            replace(samples[1], dataset_id="source-b", sample_id="shared"),
        ),
    }
    source_tokenizers = {
        dataset_id: replace(
            metadata,
            schema_version=f"{dataset_id}-schema-v1",
            schema_hash=f"{dataset_id}-schema-hash",
            data_manifest={"dataset_id": dataset_id},
            preprocessor_hash=f"{dataset_id}-preprocessor-hash",
            metadata_hash="",
        )
        for dataset_id in source_samples
    }
    label_spaces = {
        dataset_id: DatasetLabelSpace(
            dataset_id=dataset_id,
            schema_version=f"{dataset_id}-schema-v1",
            mapping_version=f"{dataset_id}-labels-v1",
            binary_labels=("benign", "attack"),
            coarse_labels=("benign", "dos"),
            native_labels=("Benign", "DoS"),
            native_to_coarse={"Benign": "benign", "DoS": "dos"},
        )
        for dataset_id in source_samples
    }
    return registry, source_samples, source_tokenizers, label_spaces


class SchemaFlowPretrainingTests(unittest.TestCase):
    def _exercise(self, device):
        registry, metadata, batch, _ = _fixture()
        generator = torch.Generator(device="cpu").manual_seed(19)
        view_one = make_pretraining_view(
            batch,
            mask_fraction=1.0,
            schema_corruption_fraction=0.0,
            generator=generator,
        )
        view_two = make_pretraining_view(
            batch,
            mask_fraction=0.5,
            schema_corruption_fraction=0.5,
            generator=generator,
        )
        self.assertTrue(view_one.reconstruction_mask.any().item())
        self.assertTrue(torch.isfinite(view_one.numeric_targets).all().item())
        self.assertTrue(torch.equal(view_one.batch.numeric_values[view_one.reconstruction_mask & (batch.feature_type_ids == 0)], torch.zeros_like(view_one.batch.numeric_values[view_one.reconstruction_mask & (batch.feature_type_ids == 0)])))

        model = SchemaFlowPretrainer(
            metadata,
            registry=registry,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            projection_dim=8,
        ).to(device)
        output_one = model(view_one.batch.to(device))
        output_two = model(view_two.batch.to(device))
        loss = compute_pretraining_loss(
            view_one.to(device),
            output_one,
            view_two.to(device),
            output_two,
        )
        self.assertTrue(torch.isfinite(loss.total).item())
        self.assertGreater(loss.reconstruction.item(), 0.0)
        self.assertGreater(loss.contrastive.item(), 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss.total.backward()
        optimizer.step()

    def test_cpu_masked_reconstruction_schema_corruption_and_infonce(self):
        self._exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_masked_reconstruction_schema_corruption_and_infonce(self):
        self._exercise("cuda")

    def test_corruption_fractions_are_strict(self):
        _, _, batch, _ = _fixture()
        for kwargs in (
            {"mask_fraction": -0.1},
            {"mask_fraction": 1.1},
            {"schema_corruption_fraction": -0.1},
            {"schema_corruption_fraction": 1.1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "fraction"):
                    make_pretraining_view(batch, **kwargs)

    def test_pretraining_checkpoint_initializes_only_supervised_encoder(self):
        registry, metadata, _, _ = _fixture()
        label_space = DatasetLabelSpace(
            dataset_id="fixture",
            schema_version=metadata.schema_version,
            mapping_version="pretraining-fixture-labels-v1",
            binary_labels=("benign", "attack"),
            coarse_labels=("benign", "dos"),
            native_labels=("Benign", "DoS"),
            native_to_coarse={"Benign": "benign", "DoS": "dos"},
        )
        model_config = {
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        }
        source = SchemaFlowPretrainer(
            metadata,
            registry=registry,
            **model_config,
            projection_dim=8,
        )
        with torch.no_grad():
            for parameter in source.encoder.parameters():
                parameter.fill_(0.125)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.9,
        )
        metadata_payload = _model_data_metadata(
            registry,
            metadata,
            label_space,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pretrain.pt"
            checkpoint_io.save_checkpoint(
                checkpoint_path,
                source,
                epoch=3,
                global_step=9,
                git_commit="a" * 40,
                schema_version=metadata.schema_version,
                seed=19,
                config={
                    "workflow": "pretrain",
                    "model": {
                        "name": "schemaflow_fm_pretrainer",
                        **model_config,
                    },
                },
                model_data_metadata=metadata_payload,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            target = SchemaFlowSupervisedModel(
                metadata,
                {label_space.dataset_id: label_space},
                registry=registry,
                **model_config,
                adapter_dim=4,
            )
            unrelated_before = {
                name: value.detach().clone()
                for name, value in target.state_dict().items()
                if not name.startswith("encoder.")
            }
            loader = getattr(checkpoint_io, "load_pretrained_encoder", None)
            self.assertIsNotNone(loader, "pretrained encoder loader is missing")
            provenance = loader(
                checkpoint_path,
                target.encoder,
                expected_model_data_metadata=metadata_payload,
                expected_model_config=model_config,
            )
            for name, value in source.encoder.state_dict().items():
                self.assertTrue(
                    torch.equal(target.encoder.state_dict()[name], value)
                )
            for name, value in unrelated_before.items():
                self.assertTrue(torch.equal(target.state_dict()[name], value))
            self.assertEqual(provenance["source_epoch"], 3)
            self.assertEqual(provenance["source_global_step"], 9)
            self.assertEqual(provenance["source_git_commit"], "a" * 40)

    def test_multisource_tokenizer_fits_union_of_source_train_only_samples(self):
        registry, source_samples, source_tokenizers, label_spaces = (
            _multisource_fixture()
        )
        metadata = build_multisource_tokenizer_metadata(
            source_samples,
            source_tokenizers,
            registry=registry,
            schema_version="multisource-schema-v1",
            data_manifest={"dataset_id": "multi"},
            split_manifest_sha256="multisource-split-hash",
        )
        vocabulary = metadata.categorical_vocabularies["PROTOCOL"]
        self.assertEqual(vocabulary.training_row_count, 2)
        self.assertIn("n:6", vocabulary.id_to_token)
        self.assertIn("n:17", vocabulary.id_to_token)
        self.assertNotIn("n:99", vocabulary.id_to_token)
        model_data = _pretraining_model_data_metadata(
            registry, metadata, label_spaces
        )
        normalized = checkpoint_io._normalize_model_data_metadata(
            model_data, torch
        )
        self.assertEqual(set(normalized["label_spaces"]), set(label_spaces))
        self.assertEqual(normalized["format_version"], 2)

    def test_multisource_sample_validation_uses_dataset_sample_identity(self):
        _, source_samples, _, label_spaces = _multisource_fixture()
        combined = source_samples["source-a"] + source_samples["source-b"]
        _validate_pretraining_samples({"train": combined}, label_spaces)
        undeclared = replace(combined[1], dataset_id="held-out-target")
        with self.assertRaisesRegex(ValueError, "does not match label space"):
            _validate_pretraining_samples(
                {"train": (combined[0], undeclared)}, label_spaces
            )
        duplicate = replace(combined[1], dataset_id="source-a")
        with self.assertRaisesRegex(ValueError, "identity is duplicated"):
            _validate_pretraining_samples(
                {"train": (combined[0], duplicate)}, label_spaces
            )

    def test_multisource_config_rejects_incomplete_or_duplicate_sources(self):
        project_root = Path(__file__).resolve().parents[1]
        config_path = (
            project_root
            / "configs"
            / "pretrain_nf_unsw_nf_cse_schemaflow_pilot.yaml"
        )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        load_config(config_path, expected_workflow="pretrain")
        fewer = json.loads(json.dumps(payload))
        fewer["data"]["sources"] = fewer["data"]["sources"][:1]
        duplicate = json.loads(json.dumps(payload))
        duplicate["data"]["sources"][1]["dataset_id"] = (
            duplicate["data"]["sources"][0]["dataset_id"]
        )
        malformed = json.loads(json.dumps(payload))
        malformed["data"]["sources"][1].pop("data_manifest")
        for label, invalid in (
            ("fewer", fewer),
            ("duplicate", duplicate),
            ("malformed", malformed),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path, expected_workflow="pretrain")

    def test_pretrain_dispatches_multisource_config_to_multisource_runner(self):
        project_root = Path(__file__).resolve().parents[1]
        path = (
            project_root
            / "configs"
            / "pretrain_nf_unsw_nf_cse_schemaflow_pilot.yaml"
        )
        args = build_parser().parse_args(["pretrain", "--config", str(path)])
        with mock.patch(
            "flowfm.training.multisource_pretraining.run_multisource_pretrain",
            return_value={"status": "completed", "run_dir": "run"},
        ) as runner:
            result = dispatch(args)
        self.assertEqual(result["status"], "completed")
        runner.assert_called_once()
        self.assertEqual(
            runner.call_args.args[0].data["dataset_id"],
            MULTISOURCE_DATASET_ID,
        )

    def test_pretrain_runner_publishes_verified_checkpoint_without_validation_data(self):
        registry, metadata, _, samples = _fixture()
        label_space = DatasetLabelSpace(
            dataset_id="fixture",
            schema_version=metadata.schema_version,
            mapping_version="pretraining-fixture-labels-v1",
            binary_labels=("benign", "attack"),
            coarse_labels=("benign", "dos"),
            native_labels=("Benign", "DoS"),
            native_to_coarse={"Benign": "benign", "DoS": "dos"},
        )
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "autodl-tmp"
            config_path = Path(directory) / "pretrain.yaml"
            config_path.write_text(
                f"""workflow: pretrain
experiment:
  name: pretrain-fixture
  output_root: {data_root / 'results'}
  seed: 19
  enforce_autodl_data_disk: true
schema:
  version: {metadata.schema_version}
runtime:
  device: cpu
  execution: train
data:
  dataset_id: fixture
  split_dir: /root/autodl-tmp/fixture-split
  preprocessor_dir: /root/autodl-tmp/fixture-preprocessor
  data_manifest: /root/autodl-tmp/fixture-manifest.json
model:
  name: schemaflow_fm_pretrainer
  d_model: 16
  nhead: 4
  num_layers: 1
  dim_feedforward: 32
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
  class_balance: none
  reconstruction_weight: 1.0
  contrastive_weight: 0.0
  loss_weights:
    binary: 1.0
    coarse: 1.0
    native: 1.0
""",
                encoding="utf-8",
            )
            try:
                config = load_config(config_path, expected_workflow="pretrain")
            except ValueError as exc:
                self.fail(f"pretraining objective weights are unsupported: {exc}")
            summary = run_pretraining(
                config,
                splits={"train": samples},
                tokenizer_metadata=metadata,
                label_space=label_space,
                registry=registry,
                data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
                split_manifest={"train": [sample.sample_id for sample in samples]},
                repo_dir=Path(__file__).resolve().parents[1],
                data_root=data_root,
                git_commit="d" * 40,
                config_path=config_path,
            )
            run_dir = Path(summary["run_dir"])
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(summary["checkpoint_verified"])
            self.assertEqual(summary["global_step"], 1)
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertTrue(
                (run_dir / "predictions/pretraining_summary.json").is_file()
            )
            self.assertTrue((run_dir / "checkpoints/index.json").is_file())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["objective"]["reconstruction_weight"], 1.0)
            self.assertEqual(metrics["objective"]["contrastive_weight"], 0.0)
            self.assertIn("reconstruction", metrics["pretraining"])
            self.assertIn("contrastive", metrics["pretraining"])

    def test_pretrain_dispatches_train_execution_to_pretraining_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.yaml"
            path.write_text(
                """workflow: pretrain
experiment:
  name: pretrain-cli
  output_root: /root/autodl-tmp/results
  seed: 1
schema: {version: v1}
runtime: {device: cpu, execution: train}
data:
  dataset_id: fixture
  split_dir: /root/autodl-tmp/split
  preprocessor_dir: /root/autodl-tmp/preprocessor
  data_manifest: /root/autodl-tmp/manifest.json
model:
  name: schemaflow_fm_pretrainer
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
  weight_decay: 0.0
  gradient_clip: 1.0
  scheduler_gamma: 0.9
  shuffle: true
  class_balance: none
  loss_weights: {binary: 1.0, coarse: 1.0, native: 1.0}
""",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                ["pretrain", "--config", str(path), "--resume-from", "checkpoint.pt"]
            )
            with mock.patch(
                "flowfm.training.pretraining.run_nf_unsw_pretrain",
                return_value={"status": "completed", "run_dir": "run"},
            ) as runner:
                result = dispatch(args)
            self.assertEqual(result["status"], "completed")
            runner.assert_called_once()
            self.assertEqual(
                runner.call_args.kwargs["resume_from"],
                Path("checkpoint.pt").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
