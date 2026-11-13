import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised in dependency-poor environments
    torch = None
    F = None

from flowfm.models.registry import MODEL_REGISTRY


@unittest.skipIf(torch is None, "PyTorch is required for legacy model smoke tests")
class LegacyModelSmokeTests(unittest.TestCase):
    def _run_step(self, device):
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        model = MODEL_REGISTRY.create(
            "rescae_bigru_legacy",
            input_channels=1,
            num_features=32,
            num_classes=3,
            use_attention_block=True,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.rand(4, 1, 1, 32, device=device)
        labels = torch.tensor([0, 1, 2, 1], device=device)

        before = next(model.parameters()).detach().clone()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        after = next(model.parameters()).detach()

        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.equal(before, after))

        return float(loss.detach().cpu())

    def test_cpu_forward_backward_optimizer_step(self):
        loss = self._run_step(torch.device("cpu"))
        self.assertGreater(loss, 0.0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for the CUDA legacy smoke test",
    )
    def test_cuda_forward_backward_optimizer_step(self):
        loss = self._run_step(torch.device("cuda"))
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
