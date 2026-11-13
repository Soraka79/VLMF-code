import tempfile
import unittest
from pathlib import Path

import numpy as np

from flowfm.training.probability_fusion import default_alpha_grid
from scripts.run_nf_cse_a0_mr_confirmation import (
    _CONDITIONS,
    _FIXED_A0,
    _select_a0_mr,
    _write_prediction_lock,
)


class NFCSEA0MRConfirmationTests(unittest.TestCase):
    def _outputs(self):
        targets = np.tile(np.arange(5, dtype=np.int64), 4)
        probabilities = np.eye(5, dtype=np.float64)[targets]
        return {
            condition: {
                "xgboost": probabilities,
                "schemaflow_scratch": probabilities,
                "schemaflow_pretrained": probabilities,
                "targets": targets,
                "sample_order_sha256": f"order-{condition}",
            }
            for condition in _CONDITIONS
        }

    def test_validation_routing_falls_back_to_frozen_a0_without_gain(self):
        selection = _select_a0_mr(
            self._outputs(),
            default_alpha_grid(0.05),
            candidate_min_gain=0.005,
            pool_min_gain=0.002,
        )

        self.assertFalse(selection["routing_gate"]["passed"])
        self.assertEqual(selection["locked_policy"]["policies"], _FIXED_A0)
        self.assertFalse(selection["test_samples_materialized"])
        self.assertFalse(selection["test_metrics_used"])

    def test_prediction_archive_is_locked_before_metrics(self):
        outputs = self._outputs()
        groups = tuple(f"{index // 2:064x}" for index in range(20))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            reloaded, reloaded_groups, lock = _write_prediction_lock(
                run_dir, outputs, groups
            )

            self.assertEqual(lock["status"], "locked_before_metric_computation")
            self.assertEqual(reloaded_groups, groups)
            self.assertEqual(tuple(reloaded), _CONDITIONS)
            self.assertTrue((run_dir / "predictions.npz").is_file())
            self.assertTrue((run_dir / "prediction_lock.json").is_file())


if __name__ == "__main__":
    unittest.main()
