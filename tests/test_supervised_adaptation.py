import tempfile
import unittest
from pathlib import Path

from flowfm.config import load_config
from flowfm.models.supervised import SchemaFlowSupervisedModel
from flowfm.training.supervised import _configure_adaptation
from tests.test_supervised_training import _config_text, _fixture


class SupervisedAdaptationTests(unittest.TestCase):
    def _model(self):
        registry, metadata, label_space, _ = _fixture()
        model = SchemaFlowSupervisedModel(
            metadata,
            {label_space.dataset_id: label_space},
            registry=registry,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            adapter_dim=4,
        )
        return model

    def test_adapter_policy_freezes_shared_encoder_but_trains_categories(self):
        model = self._model()
        adaptation = _configure_adaptation(model, "adapter_category_heads")

        for name, parameter in model.named_parameters():
            if name.startswith("encoder.category_embeddings."):
                self.assertTrue(parameter.requires_grad, name)
            elif name.startswith("encoder."):
                self.assertFalse(parameter.requires_grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)
        self.assertLess(
            adaptation["trainable_parameter_count"],
            adaptation["total_parameter_count"],
        )
        self.assertGreater(
            adaptation["trainable_category_parameter_count"],
            0,
        )

    def test_config_accepts_only_supported_finetune_adaptation_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.yaml"
            valid_path.write_text(
                _config_text(root / "results").replace(
                    "  class_balance: inverse_frequency\n",
                    "  class_balance: inverse_frequency\n"
                    "  adaptation_mode: adapter_category_heads\n",
                ),
                encoding="utf-8",
            )
            config = load_config(valid_path, expected_workflow="finetune")
            self.assertEqual(
                config.training["adaptation_mode"],
                "adapter_category_heads",
            )

            invalid_path = root / "invalid.yaml"
            invalid_path.write_text(
                valid_path.read_text(encoding="utf-8").replace(
                    "adapter_category_heads",
                    "freeze_everything",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "adaptation_mode"):
                load_config(invalid_path, expected_workflow="finetune")


if __name__ == "__main__":
    unittest.main()
