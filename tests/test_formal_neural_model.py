import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.run_formal_neural_model import _verify_source_checkpoint


class FormalNeuralModelTests(unittest.TestCase):
    def _checkpoint(self, directory, dataset_id):
        path = Path(directory) / "source.pt"
        torch.save(
            {
                "config": {
                    "workflow": "pretrain",
                    "data": {"dataset_id": dataset_id},
                }
            },
            path,
        )
        return path

    def test_single_source_checkpoint_must_exclude_target(self):
        with TemporaryDirectory() as directory:
            path = self._checkpoint(directory, "NF-CSE-CIC-IDS2018-v2")

            contract = _verify_source_checkpoint(path, "NF-UNSW-NB15-v2")

            self.assertEqual(
                contract["source_dataset_id"], "NF-CSE-CIC-IDS2018-v2"
            )
            self.assertTrue(contract["target_excluded"])

    def test_same_target_source_checkpoint_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = self._checkpoint(directory, "NF-UNSW-NB15-v2")

            with self.assertRaisesRegex(RuntimeError, "target-excluded"):
                _verify_source_checkpoint(path, "NF-UNSW-NB15-v2")


if __name__ == "__main__":
    unittest.main()
