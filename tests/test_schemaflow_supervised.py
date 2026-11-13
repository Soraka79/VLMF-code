import unittest

import torch

from flowfm.data.contracts import FlowSample
from flowfm.data.labels import (
    DatasetLabelSpace,
    NF_UNSW_LABEL_SPACE,
    encode_supervised_targets,
)
from flowfm.data.schemaflow_batch import collate_schemaflow_samples
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)
from flowfm.models.supervised import (
    SchemaFlowSupervisedModel,
    compute_supervised_loss,
)
from flowfm.models.registry import MODEL_REGISTRY


def _sample(sample_id, token, binary, coarse, native, dataset_id="fixture"):
    return FlowSample(
        features=(1.5, 1),
        feature_ids=("BYTES", "PROTOCOL"),
        categorical_mask=(False, True),
        missing_mask=(False, False),
        category_tokens=(None, token),
        binary_label=binary,
        coarse_label=coarse,
        native_label=native,
        dataset_id=dataset_id,
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
        "supervised-fixture-registry-v1",
        (
            CanonicalFeatureSpec(7, "BYTES", "numeric", "continuous"),
            CanonicalFeatureSpec(3, "PROTOCOL", "categorical", "vocabulary"),
        ),
    )
    training_hash = "supervised-training-hash"
    vocabulary = fit_categorical_vocabulary(
        registry.by_name["PROTOCOL"],
        {"n:6": 2},
        training_row_count=2,
        training_sample_ids_hash=training_hash,
    )
    metadata = TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version="supervised-fixture-schema-v1",
        schema_hash="supervised-fixture-schema-hash",
        categorical_vocabularies={"PROTOCOL": vocabulary},
        data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
        split_manifest_sha256="split-hash",
        preprocessor_hash="preprocessor-hash",
        training_sample_ids_hash=training_hash,
    )
    label_space = DatasetLabelSpace(
        dataset_id="fixture",
        schema_version=metadata.schema_version,
        mapping_version="fixture-labels-v1",
        binary_labels=("benign", "attack"),
        coarse_labels=("benign", "dos"),
        native_labels=("Benign", "DoS"),
        native_to_coarse={"Benign": "benign", "DoS": "dos"},
    )
    samples = (
        _sample("1", "n:999", "attack", "dos", "DoS"),
        _sample("2", "n:6", "benign", "benign", "Benign"),
    )
    batch = collate_schemaflow_samples(samples, metadata, registry=registry)
    return registry, metadata, label_space, samples, batch


class DatasetLabelSpaceTests(unittest.TestCase):
    def test_nf_unsw_labels_are_explicit_versioned_and_not_train_fitted(self):
        self.assertEqual(NF_UNSW_LABEL_SPACE.dataset_id, "NF-UNSW-NB15-v2")
        self.assertEqual(
            NF_UNSW_LABEL_SPACE.mapping_version,
            "nf-unsw-nb15-v2-label-space-v2",
        )
        self.assertEqual(NF_UNSW_LABEL_SPACE.binary_labels, ("benign", "attack"))
        self.assertEqual(
            NF_UNSW_LABEL_SPACE.coarse_labels,
            (
                "benign",
                "dos",
                "scan",
                "brute_force",
                "exploit",
                "botnet",
                "backdoor",
                "malware",
            ),
        )
        self.assertEqual(len(NF_UNSW_LABEL_SPACE.native_labels), 10)
        restored = DatasetLabelSpace.from_dict(NF_UNSW_LABEL_SPACE.to_dict())
        self.assertEqual(restored, NF_UNSW_LABEL_SPACE)
        self.assertEqual(restored.label_space_hash, NF_UNSW_LABEL_SPACE.label_space_hash)

    def test_target_encoding_rejects_unknown_or_inconsistent_labels(self):
        _, _, label_space, samples, _ = _fixture()
        targets = encode_supervised_targets(samples, label_space)
        self.assertEqual(targets.binary.tolist(), [1, 0])
        self.assertEqual(targets.coarse.tolist(), [1, 0])
        self.assertEqual(targets.native.tolist(), [1, 0])

        invalid = _sample("3", "n:6", "attack", "benign", "Benign")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            encode_supervised_targets((invalid,), label_space)
        unknown = _sample("4", "n:6", "attack", "dos", "Unknown")
        with self.assertRaisesRegex(ValueError, "unknown native"):
            encode_supervised_targets((unknown,), label_space)


class SchemaFlowSupervisedModelTests(unittest.TestCase):
    def _exercise(self, device):
        registry, metadata, label_space, samples, batch = _fixture()
        model = SchemaFlowSupervisedModel(
            metadata,
            {"fixture": label_space},
            registry=registry,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            adapter_dim=8,
        ).to(device)
        device_batch = batch.to(device)
        targets = encode_supervised_targets(samples, label_space, device=device)
        output = model(device_batch)

        self.assertEqual(output.binary_logits.shape, (2, 2))
        self.assertEqual(output.coarse_logits.shape, (2, 2))
        self.assertEqual(output.native_logits.shape, (2, 2))
        self.assertEqual(output.dataset_id, "fixture")
        losses = compute_supervised_loss(output, targets)
        self.assertTrue(torch.isfinite(losses.total).item())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses.total.backward()
        self.assertGreater(
            model.encoder.category_embeddings["3"].weight.grad[1].abs().sum().item(),
            0.0,
        )
        optimizer.step()
        self.assertTrue(
            torch.equal(
                model.encoder.category_embeddings["3"].weight[0],
                torch.zeros_like(model.encoder.category_embeddings["3"].weight[0]),
            )
        )

    def test_cpu_forward_backward_and_optimizer_step(self):
        self._exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_forward_backward_and_optimizer_step(self):
        self._exercise("cuda")

    def test_model_rejects_mixed_dataset_batches(self):
        registry, metadata, label_space, _, batch = _fixture()
        model = SchemaFlowSupervisedModel(
            metadata,
            {"fixture": label_space},
            registry=registry,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            adapter_dim=4,
        )
        mixed = type(batch)(
            feature_ids=batch.feature_ids,
            numeric_values=batch.numeric_values,
            category_ids=batch.category_ids,
            feature_type_ids=batch.feature_type_ids,
            missing_mask=batch.missing_mask,
            schema_present_mask=batch.schema_present_mask,
            sample_ids=batch.sample_ids,
            dataset_ids=("fixture", "other"),
        )
        with self.assertRaisesRegex(ValueError, "one dataset"):
            model(mixed)

    def test_supervised_model_is_publicly_exported_and_registered(self):
        from flowfm.models import (
            SchemaFlowSupervisedModel as ExportedSchemaFlowSupervisedModel,
        )

        self.assertIs(ExportedSchemaFlowSupervisedModel, SchemaFlowSupervisedModel)
        self.assertIn("schemaflow_fm_supervised", MODEL_REGISTRY.names())


if __name__ == "__main__":
    unittest.main()
