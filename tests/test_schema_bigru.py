import unittest
from dataclasses import replace

import torch

from flowfm.models.registry import MODEL_REGISTRY
from tests.test_schemaflow_model import fixture


def _make_encoder(registry, metadata, device):
    from flowfm.models.schema_bigru import SchemaBiGRUEncoder

    torch.manual_seed(31)
    if device == "cuda":
        torch.cuda.manual_seed_all(31)
    return SchemaBiGRUEncoder(
        metadata,
        registry=registry,
        d_model=16,
        hidden_size=8,
        num_layers=2,
        dropout=0.0,
    ).to(device)


def _select_features(batch, indices):
    index = torch.as_tensor(indices, dtype=torch.int64, device=batch.feature_ids.device)
    return type(batch)(
        feature_ids=batch.feature_ids.index_select(1, index),
        numeric_values=batch.numeric_values.index_select(1, index),
        category_ids=batch.category_ids.index_select(1, index),
        feature_type_ids=batch.feature_type_ids.index_select(1, index),
        missing_mask=batch.missing_mask.index_select(1, index),
        schema_present_mask=batch.schema_present_mask.index_select(1, index),
        sample_ids=batch.sample_ids,
        dataset_ids=batch.dataset_ids,
    )


class SchemaBiGRUEncoderTests(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def _exercise_forward_backward(self, device):
        from flowfm.models.schema_bigru import SchemaBiGRUEncoderOutput

        registry, metadata, batch = fixture()
        model = _make_encoder(registry, metadata, device)
        batch = batch.to(device)
        output = model(batch)

        self.assertIsInstance(output, SchemaBiGRUEncoderOutput)
        self.assertEqual(output.representation.shape, (2, 16))
        self.assertEqual(output.sequence_embeddings.shape, (2, 4, 16))
        self.assertEqual(output.sequence_lengths.tolist(), [4, 3])
        self.assertEqual(output.sequence_mask.sum(dim=1).tolist(), [4, 3])
        self.assertTrue(torch.isfinite(output.representation).all().item())

        forbidden = (
            torch.nn.MultiheadAttention,
            torch.nn.TransformerEncoderLayer,
        )
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        self.assertFalse(hasattr(model, "adapter"))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output.representation.square().mean().backward()
        self.assertGreater(
            model.category_embeddings["3"].weight.grad[1].abs().sum().item(),
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
        self._exercise_forward_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_forward_backward_and_optimizer_step(self):
        self._exercise_forward_backward("cuda")

    def test_input_permutation_is_restored_to_versioned_canonical_order(self):
        for device in self._devices():
            with self.subTest(device=device):
                registry, metadata, batch = fixture()
                model = _make_encoder(registry, metadata, device)
                model.eval()
                batch = batch.to(device)
                permutation = torch.tensor([2, 0, 3, 1], device=device)
                with torch.no_grad():
                    original = model(batch)
                    permuted = model(batch.permute_features(permutation))
                tolerance = 2e-5 if device == "cuda" else 1e-6
                torch.testing.assert_close(
                    permuted.representation,
                    original.representation,
                    atol=tolerance,
                    rtol=tolerance,
                )
                torch.testing.assert_close(
                    permuted.sequence_embeddings,
                    original.sequence_embeddings,
                    atol=tolerance,
                    rtol=tolerance,
                )

    def test_schema_absent_fields_are_excluded_but_missing_present_fields_remain(self):
        registry, metadata, batch = fixture()
        model = _make_encoder(registry, metadata, "cpu")
        model.eval()
        bytes_index = batch.feature_ids[0].tolist().index(7)
        latency_index = batch.feature_ids[0].tolist().index(9)

        numeric = batch.numeric_values.clone()
        numeric[:, bytes_index] = 0
        missing = batch.missing_mask.clone()
        missing[:, bytes_index] = True
        present = batch.schema_present_mask.clone()
        present[:, bytes_index] = False
        dense_absent = replace(
            batch,
            numeric_values=numeric,
            missing_mask=missing,
            schema_present_mask=present,
        )
        reduced = _select_features(
            dense_absent,
            [index for index in range(batch.feature_count) if index != bytes_index],
        )

        with torch.no_grad():
            dense_output = model(dense_absent)
            reduced_output = model(reduced)
        torch.testing.assert_close(
            dense_output.representation,
            reduced_output.representation,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(dense_output.sequence_lengths.tolist(), [3, 3])
        self.assertTrue(dense_absent.missing_mask[0, latency_index].item())
        self.assertTrue(dense_absent.schema_present_mask[0, latency_index].item())

    def test_rejects_samples_with_no_schema_present_features(self):
        registry, metadata, batch = fixture()
        model = _make_encoder(registry, metadata, "cpu")
        numeric = torch.zeros_like(batch.numeric_values)
        categories = torch.zeros_like(batch.category_ids)
        missing = torch.ones_like(batch.missing_mask)
        present = torch.zeros_like(batch.schema_present_mask)
        empty = replace(
            batch,
            numeric_values=numeric,
            category_ids=categories,
            missing_mask=missing,
            schema_present_mask=present,
        )
        with self.assertRaisesRegex(ValueError, "at least one schema-present"):
            model(empty)


class SchemaBiGRUClassifierTests(unittest.TestCase):
    def test_independent_classifier_head_is_exported_and_registered(self):
        from flowfm.models import SchemaBiGRUClassifier
        from flowfm.models.schema_bigru import SchemaBiGRUClassifierOutput

        registry, metadata, batch = fixture()
        model = SchemaBiGRUClassifier(
            metadata,
            num_classes=3,
            registry=registry,
            d_model=16,
            hidden_size=8,
            num_layers=2,
            dropout=0.0,
        )
        output = model(batch)
        self.assertIsInstance(output, SchemaBiGRUClassifierOutput)
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertEqual(output.representation.shape, (2, 16))
        self.assertIn("schema_bigru", MODEL_REGISTRY.names())
        registered = MODEL_REGISTRY.create(
            "schema_bigru",
            metadata,
            num_classes=3,
            registry=registry,
            d_model=16,
            hidden_size=8,
            num_layers=2,
            dropout=0.0,
        )
        self.assertIsInstance(registered, SchemaBiGRUClassifier)


if __name__ == "__main__":
    unittest.main()
