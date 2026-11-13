import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised by this environment
    torch = None

try:
    import numpy
except ImportError:  # pragma: no cover - exercised by this environment
    numpy = None

from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)
from flowfm.data.labels import DatasetLabelSpace
from flowfm.training.checkpoints import load_checkpoint, save_checkpoint


_TEST_GIT_COMMIT = "b" * 40


def _model_data_metadata(schema_version):
    protocol = CanonicalFeatureSpec(3, "PROTOCOL", "categorical", "vocabulary")
    registry = CanonicalFeatureRegistry("torch-checkpoint-registry-v1", (protocol,))
    vocabulary = fit_categorical_vocabulary(
        protocol,
        {"i:6": 8, "i:17": 4},
        training_row_count=12,
        training_sample_ids_hash="torch-train-hash",
        max_known_categories=1,
    )
    tokenizer = TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version=schema_version,
        schema_hash="torch-schema-hash",
        categorical_vocabularies={"PROTOCOL": vocabulary},
        data_manifest={"dataset": "torch-test", "sha256": "manifest-hash"},
        split_manifest_sha256="split-hash",
        preprocessor_hash="preprocessor-hash",
        training_sample_ids_hash="torch-train-hash",
    )
    label_space = DatasetLabelSpace(
        dataset_id="torch-test",
        schema_version=schema_version,
        mapping_version="torch-labels-v1",
        binary_labels=("benign", "attack"),
        coarse_labels=("benign", "dos"),
        native_labels=("Benign", "DoS"),
        native_to_coarse={"Benign": "benign", "DoS": "dos"},
    )
    return {
        "format_version": 2,
        "feature_registry": registry.to_dict(),
        "tokenizer_metadata": tokenizer.to_dict(),
        "categorical_table_sizes": {"PROTOCOL": vocabulary.table_size},
        "label_spaces": {"torch-test": label_space.to_dict()},
    }


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RealTorchCheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_next_step(self):
        torch.manual_seed(123)
        random.seed(123)
        if numpy is not None:
            numpy.random.seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4, 1),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        inputs = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        targets = torch.tensor([[0.25], [-1.0]])

        self._step(model, optimizer, inputs, targets)
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                epoch=1,
                global_step=1,
                git_commit=_TEST_GIT_COMMIT,
                schema_version="test-v1",
                seed=123,
                config={"lr": 0.1},
                model_data_metadata=_model_data_metadata("test-v1"),
                optimizer=optimizer,
                scheduler=scheduler,
            )

            expected_python_random = random.random()
            expected_numpy_random = None if numpy is None else numpy.random.random()
            expected_torch_random = torch.rand(3)
            expected_loss = self._step(model, optimizer, inputs, targets)
            scheduler.step()
            expected_parameters = [value.detach().clone() for value in model.parameters()]
            expected_optimizer = optimizer.state_dict()

            resumed_model = torch.nn.Sequential(
                torch.nn.Linear(2, 4),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4, 1),
            )
            resumed_optimizer = torch.optim.SGD(
                resumed_model.parameters(), lr=0.1, momentum=0.9
            )
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1, gamma=0.5
            )
            metadata = load_checkpoint(
                path,
                resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                expected_model_data_metadata=_model_data_metadata("test-v1"),
            )
            resumed_python_random = random.random()
            resumed_numpy_random = None if numpy is None else numpy.random.random()
            resumed_torch_random = torch.rand(3)
            resumed_loss = self._step(resumed_model, resumed_optimizer, inputs, targets)
            resumed_scheduler.step()

        self.assertEqual(metadata["epoch"], 1)
        self.assertEqual(resumed_python_random, expected_python_random)
        self.assertEqual(resumed_numpy_random, expected_numpy_random)
        self.assertTrue(torch.equal(resumed_torch_random, expected_torch_random))
        self.assertAlmostEqual(resumed_loss, expected_loss, places=7)
        for resumed, expected in zip(resumed_model.parameters(), expected_parameters):
            self.assertTrue(torch.equal(resumed, expected))
        self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
        self._assert_nested_equal(resumed_optimizer.state_dict(), expected_optimizer)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cuda_map_location_keeps_rng_states_on_cpu(self):
        model = torch.nn.Linear(2, 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.tensor([[1.0, -2.0], [0.5, 3.0]], device="cuda")
        targets = torch.tensor([[0.25], [-1.0]], device="cuda")
        self._step(model, optimizer, inputs, targets)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda-checkpoint.pt"
            save_checkpoint(
                path,
                model,
                epoch=1,
                global_step=1,
                git_commit=_TEST_GIT_COMMIT,
                schema_version="test-v2",
                seed=123,
                config={},
                model_data_metadata=_model_data_metadata("test-v2"),
                optimizer=optimizer,
            )
            restored_model = torch.nn.Linear(2, 1).cuda()
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=0.01
            )
            payload = load_checkpoint(
                path,
                restored_model,
                optimizer=restored_optimizer,
                map_location="cuda",
                expected_model_data_metadata=_model_data_metadata("test-v2"),
            )
            restored_loss = self._step(
                restored_model,
                restored_optimizer,
                inputs,
                targets,
            )

        self.assertTrue(torch.isfinite(torch.tensor(restored_loss)))
        self.assertEqual(
            payload["model_data_metadata"],
            _model_data_metadata("test-v2"),
        )
        self.assertEqual(payload["torch_cpu_rng_state"].device.type, "cpu")
        self.assertTrue(payload["torch_cuda_rng_states"])
        for state in payload["torch_cuda_rng_states"]:
            self.assertEqual(state.device.type, "cpu")

    def test_reduce_lr_on_plateau_infinite_state_roundtrips(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
        )
        self.assertEqual(scheduler.state_dict()["best"], float("inf"))
        self.assertEqual(scheduler.state_dict()["mode_worse"], float("inf"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plateau-checkpoint.pt"
            save_checkpoint(
                path,
                model,
                epoch=0,
                global_step=0,
                git_commit=_TEST_GIT_COMMIT,
                schema_version="test-v2",
                seed=123,
                config={},
                model_data_metadata=_model_data_metadata("test-v2"),
                optimizer=optimizer,
                scheduler=scheduler,
            )
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
            restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                restored_optimizer,
                mode="min",
            )
            load_checkpoint(
                path,
                restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                expected_model_data_metadata=_model_data_metadata("test-v2"),
            )

        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    @staticmethod
    def _step(model, optimizer, inputs, targets):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _assert_nested_equal(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self._assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self._assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
