import unittest
from dataclasses import replace

import torch
from flowfm.models import SchemaFlowEncoder as ExportedSchemaFlowEncoder

from flowfm.data.contracts import FlowSample
from flowfm.data.schemaflow_batch import (
    collate_schemaflow_samples,
)
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)
from flowfm.models.schemaflow import (
    SchemaFlowEncoder,
    SchemaFlowEncoderOutput,
)
from flowfm.models.registry import MODEL_REGISTRY


def make_sample(
    *,
    sample_id,
    feature_ids,
    features,
    categorical_mask,
    missing_mask,
    category_tokens,
):
    return FlowSample(
        features=features,
        feature_ids=feature_ids,
        categorical_mask=categorical_mask,
        missing_mask=missing_mask,
        category_tokens=category_tokens,
        binary_label="attack",
        coarse_label="dos",
        native_label="DoS",
        dataset_id="fixture",
        sample_id=sample_id,
        feature_key=f"feature-{sample_id}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id,
    )


def fixture():
    registry = CanonicalFeatureRegistry(
        "model-fixture-registry-v1",
        (
            CanonicalFeatureSpec(
                7, "BYTES", "numeric", "continuous"
            ),
            CanonicalFeatureSpec(
                3, "PROTOCOL", "categorical", "vocabulary"
            ),
            CanonicalFeatureSpec(
                9, "LATENCY", "numeric", "continuous"
            ),
            CanonicalFeatureSpec(
                5, "TCP_FLAGS", "categorical", "vocabulary"
            ),
            CanonicalFeatureSpec(
                11,
                "DNS_QUERY_ID",
                "categorical",
                "audit_only",
                model_visible=False,
            ),
        ),
    )
    training_hash = "training-hash"
    protocol = fit_categorical_vocabulary(
        registry.by_name["PROTOCOL"],
        {"n:6": 2},
        training_row_count=2,
        training_sample_ids_hash=training_hash,
    )
    flags = fit_categorical_vocabulary(
        registry.by_name["TCP_FLAGS"],
        {"n:19": 1, "n:27": 1},
        training_row_count=2,
        training_sample_ids_hash=training_hash,
    )
    metadata = TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version="model-fixture-schema-v1",
        schema_hash="model-fixture-schema-hash",
        categorical_vocabularies={
            "PROTOCOL": protocol,
            "TCP_FLAGS": flags,
        },
        data_manifest={
            "dataset_id": "fixture",
            "sha256": "data-hash",
        },
        split_manifest_sha256="split-hash",
        preprocessor_hash="preprocessor-hash",
        training_sample_ids_hash=training_hash,
    )
    first = make_sample(
        sample_id="1",
        feature_ids=(
            "BYTES",
            "PROTOCOL",
            "TCP_FLAGS",
            "LATENCY",
            "DNS_QUERY_ID",
        ),
        features=(1.5, 1, 3, float("nan"), 2),
        categorical_mask=(False, True, True, False, True),
        missing_mask=(False, False, False, True, False),
        category_tokens=(None, "n:999", "n:27", None, "n:42"),
    )
    second = make_sample(
        sample_id="2",
        feature_ids=(
            "PROTOCOL",
            "TCP_FLAGS",
            "LATENCY",
        ),
        features=(2, 0, float("nan")),
        categorical_mask=(True, True, False),
        missing_mask=(False, True, True),
        category_tokens=("n:6", None, None),
    )
    batch = collate_schemaflow_samples(
        (first, second),
        metadata,
        registry=registry,
    )
    return registry, metadata, batch


def make_model(registry, metadata, device):
    torch.manual_seed(17)
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
    return SchemaFlowEncoder(
        metadata,
        registry=registry,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
    ).to(device)


class SchemaFlowEncoderTests(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def _exercise_forward_backward_step(self, device):
        registry, metadata, batch = fixture()
        batch = batch.to(device)
        model = make_model(registry, metadata, device)
        model.train()

        self.assertEqual(
            set(model.category_embeddings.keys()),
            {"3", "5"},
        )
        self.assertIsNot(
            model.category_embeddings["3"],
            model.category_embeddings["5"],
        )
        for embedding in model.category_embeddings.values():
            self.assertTrue(
                torch.equal(
                    embedding.weight[0],
                    torch.zeros_like(embedding.weight[0]),
                )
            )

        output = model(batch, return_layer_states=True)

        self.assertIsInstance(output, SchemaFlowEncoderOutput)
        self.assertEqual(output.flow_embedding.shape, (2, 16))
        self.assertEqual(output.feature_embeddings.shape, (2, 4, 16))
        self.assertEqual(output.value_embeddings.shape, (2, 4, 16))
        self.assertEqual(len(output.layer_feature_embeddings), 2)
        self.assertTrue(torch.isfinite(output.flow_embedding).all().item())
        self.assertTrue(torch.isfinite(output.feature_embeddings).all().item())

        latency_index = batch.feature_ids[0].tolist().index(9)
        bytes_index = batch.feature_ids[0].tolist().index(7)
        flags_index = batch.feature_ids[0].tolist().index(5)
        self.assertTrue(
            torch.equal(
                output.value_embeddings[:, latency_index],
                torch.zeros_like(
                    output.value_embeddings[:, latency_index]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                output.value_embeddings[1, flags_index],
                torch.zeros_like(
                    output.value_embeddings[1, flags_index]
                ),
            )
        )
        self.assertFalse(batch.schema_present_mask[1, bytes_index].item())
        for hidden in (
            output.input_feature_embeddings,
            *output.layer_feature_embeddings,
            output.feature_embeddings,
        ):
            self.assertTrue(
                torch.equal(
                    hidden[1, bytes_index],
                    torch.zeros_like(hidden[1, bytes_index]),
                )
            )

        present = batch.schema_present_mask
        token_means = output.input_feature_embeddings.mean(dim=-1)
        self.assertTrue(
            torch.allclose(
                token_means[present],
                torch.zeros_like(token_means[present]),
                atol=1e-5,
                rtol=0,
            )
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = (
            output.flow_embedding.square().mean()
            + output.feature_embeddings.square().mean()
        )
        loss.backward()
        unknown_gradient = (
            model.category_embeddings["3"].weight.grad[1]
        )
        self.assertGreater(
            unknown_gradient.abs().sum().item(),
            0.0,
        )
        optimizer.step()

        for embedding in model.category_embeddings.values():
            self.assertTrue(
                torch.equal(
                    embedding.weight[0],
                    torch.zeros_like(embedding.weight[0]),
                )
            )

    def test_cpu_forward_backward_and_optimizer_step(self):
        self._exercise_forward_backward_step("cpu")

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for AutoDL acceptance",
    )
    def test_cuda_forward_backward_and_optimizer_step(self):
        self._exercise_forward_backward_step("cuda")

    def test_synchronized_feature_permutation_preserves_flow_representation(self):
        for device in self._devices():
            with self.subTest(device=device):
                registry, metadata, batch = fixture()
                batch = batch.to(device)
                model = make_model(registry, metadata, device)
                model.eval()
                permutation = torch.tensor(
                    [2, 0, 3, 1],
                    device=device,
                )

                with torch.no_grad():
                    original = model(batch)
                    permuted = model(
                        batch.permute_features(permutation)
                    )

                tolerance = 2e-5 if device == "cuda" else 1e-6
                torch.testing.assert_close(
                    permuted.flow_embedding,
                    original.flow_embedding,
                    atol=tolerance,
                    rtol=tolerance,
                )
                torch.testing.assert_close(
                    permuted.feature_embeddings,
                    original.feature_embeddings.index_select(
                        1, permutation
                    ),
                    atol=tolerance,
                    rtol=tolerance,
                )

    def test_rejects_invalid_category_id_and_feature_id_type_misalignment(self):
        for device in self._devices():
            with self.subTest(device=device):
                registry, metadata, batch = fixture()
                batch = batch.to(device)
                model = make_model(registry, metadata, device)

                invalid_categories = batch.category_ids.clone()
                invalid_categories[0, 0] = (
                    metadata.categorical_vocabularies[
                        "PROTOCOL"
                    ].table_size
                )
                with self.assertRaisesRegex(
                    ValueError, "category ID"
                ):
                    model(
                        replace(
                            batch,
                            category_ids=invalid_categories,
                        )
                    )

                misaligned_ids = batch.feature_ids.clone()
                first = misaligned_ids[:, 0].clone()
                misaligned_ids[:, 0] = misaligned_ids[:, 2]
                misaligned_ids[:, 2] = first
                with self.assertRaisesRegex(
                    ValueError, "feature ID/type mismatch"
                ):
                    model(
                        replace(
                            batch,
                            feature_ids=misaligned_ids,
                        )
                    )

                unknown_ids = batch.feature_ids.clone()
                unknown_ids[:, 0] = 999
                with self.assertRaisesRegex(
                    ValueError, "unregistered feature ID"
                ):
                    model(
                        replace(
                            batch,
                            feature_ids=unknown_ids,
                        )
                    )

    def test_model_registry_creates_schemaflow_encoder(self):
        registry, metadata, _ = fixture()

        self.assertIn("schemaflow_fm", MODEL_REGISTRY.names())
        model = MODEL_REGISTRY.create(
            "schemaflow_fm",
            metadata,
            registry=registry,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )

        self.assertIsInstance(model, SchemaFlowEncoder)
        self.assertIs(ExportedSchemaFlowEncoder, SchemaFlowEncoder)


if __name__ == "__main__":
    unittest.main()
