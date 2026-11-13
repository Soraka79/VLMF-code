import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from flowfm.models.registry import MODEL_REGISTRY
from tests.test_schemaflow_model import fixture


def _make_model(registry, metadata, device, num_classes=3):
    from flowfm.models.ft_transformer import FTTransformerClassifier

    torch.manual_seed(47)
    if device == "cuda":
        torch.cuda.manual_seed_all(47)
    return FTTransformerClassifier(
        metadata,
        num_classes=num_classes,
        registry=registry,
        d_token=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
    ).to(device)


class FTTransformerClassifierTests(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def _exercise(self, device):
        from flowfm.models.ft_transformer import FTTransformerClassifierOutput

        registry, metadata, batch = fixture()
        model = _make_model(registry, metadata, device)
        batch = batch.to(device)
        output = model(batch)
        self.assertIsInstance(output, FTTransformerClassifierOutput)
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertEqual(output.representation.shape, (2, 16))
        self.assertEqual(output.encoder_output.feature_embeddings.shape, (2, 4, 16))
        self.assertFalse(hasattr(model, "adapter"))
        self.assertFalse(hasattr(model, "pretraining_head"))

        labels = torch.tensor([0, 2], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = F.cross_entropy(output.logits, labels)
        loss.backward()
        self.assertGreater(
            model.encoder.category_embeddings["3"].weight.grad[1].abs().sum().item(),
            0.0,
        )
        optimizer.step()
        for embedding in model.encoder.category_embeddings.values():
            self.assertTrue(
                torch.equal(
                    embedding.weight[0],
                    torch.zeros_like(embedding.weight[0]),
                )
            )
        self.assertTrue(torch.isfinite(loss).item())

    def test_cpu_forward_backward_optimizer_step(self):
        self._exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_forward_backward_optimizer_step(self):
        self._exercise("cuda")

    def test_synchronized_permutation_preserves_representation(self):
        for device in self._devices():
            with self.subTest(device=device):
                registry, metadata, batch = fixture()
                model = _make_model(registry, metadata, device)
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

    def test_absent_feature_states_remain_zero(self):
        registry, metadata, batch = fixture()
        model = _make_model(registry, metadata, "cpu")
        category_ids = batch.category_ids.clone()
        category_ids[1, -1] = 0
        missing_mask = batch.missing_mask.clone()
        missing_mask[1, -1] = True
        present_mask = batch.schema_present_mask.clone()
        present_mask[1, -1] = False
        absent = replace(
            batch,
            category_ids=category_ids,
            missing_mask=missing_mask,
            schema_present_mask=present_mask,
        )
        output = model(absent, return_layer_states=True)
        self.assertTrue(
            torch.equal(
                output.encoder_output.input_feature_embeddings[1, -1],
                torch.zeros(16),
            )
        )
        for state in output.encoder_output.layer_feature_embeddings:
            self.assertTrue(torch.equal(state[1, -1], torch.zeros(16)))
        self.assertTrue(
            torch.equal(
                output.encoder_output.feature_embeddings[1, -1],
                torch.zeros(16),
            )
        )

    def test_invalid_class_count_is_rejected_and_registry_is_explicit(self):
        registry, metadata, _ = fixture()
        from flowfm.models.ft_transformer import FTTransformerClassifier

        for value in (True, 0, 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "greater than one"):
                    FTTransformerClassifier(
                        metadata,
                        num_classes=value,
                        registry=registry,
                    )
        self.assertIn("ft_transformer_baseline", MODEL_REGISTRY.names())
        registered = MODEL_REGISTRY.create(
            "ft_transformer_baseline",
            metadata,
            num_classes=3,
            registry=registry,
            d_token=16,
            nhead=4,
            num_layers=2,
            dim_feedforward=32,
            dropout=0.0,
        )
        self.assertIsInstance(registered, FTTransformerClassifier)


if __name__ == "__main__":
    unittest.main()
