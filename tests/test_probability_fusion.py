import unittest

import numpy as np

from flowfm.training.probability_fusion import (
    default_alpha_grid,
    evaluate_policy,
    pool_probabilities,
    probability_sha256,
    select_fusion,
)


class ProbabilityFusionTests(unittest.TestCase):
    def setUp(self):
        self.tree = np.asarray(
            (
                (0.90, 0.05, 0.05),
                (0.05, 0.90, 0.05),
                (0.50, 0.10, 0.40),
            ),
            dtype=np.float64,
        )
        self.neural = np.asarray(
            (
                (0.45, 0.50, 0.05),
                (0.10, 0.80, 0.10),
                (0.10, 0.10, 0.80),
            ),
            dtype=np.float64,
        )
        self.targets = np.asarray((0, 1, 2), dtype=np.int64)
        self.labels = ("benign", "dos", "scan")

    def test_default_grid_has_stable_endpoints(self):
        grid = default_alpha_grid()

        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[-1], 1.0)
        self.assertEqual(len(grid), 21)
        self.assertIn(0.55, grid)

    def test_linear_pool_uses_alpha_as_xgboost_weight(self):
        pooled = pool_probabilities(
            self.tree,
            self.neural,
            alpha=0.25,
            method="linear",
        )

        np.testing.assert_allclose(
            pooled,
            0.25 * self.tree + 0.75 * self.neural,
            atol=1e-12,
            rtol=0,
        )

    def test_log_pool_is_finite_and_normalized_with_zero_probabilities(self):
        tree = np.asarray(((1.0, 0.0), (0.0, 1.0)))
        neural = np.asarray(((0.6, 0.4), (0.4, 0.6)))

        pooled = pool_probabilities(tree, neural, alpha=0.5, method="log")

        self.assertTrue(np.isfinite(pooled).all())
        np.testing.assert_allclose(pooled.sum(axis=1), 1.0)

    def test_selection_uses_validation_metrics_and_can_exploit_complementarity(self):
        selection = select_fusion(
            self.tree,
            self.neural,
            self.targets,
            self.labels,
            alpha_grid=(0.0, 0.5, 1.0),
            methods=("linear",),
        )

        self.assertEqual(selection["selection_scope"], "validation_only")
        self.assertEqual(selection["selected"]["alpha"], 0.5)
        self.assertEqual(selection["selected"]["metrics"]["macro_f1"], 1.0)
        self.assertEqual(len(selection["candidates"]), 3)

    def test_locked_policy_is_evaluated_without_reselection(self):
        metrics = evaluate_policy(
            self.tree,
            self.neural,
            self.targets,
            self.labels,
            {"method": "linear", "alpha": 1.0},
        )

        self.assertLess(metrics["macro_f1"], 1.0)

    def test_probability_hash_covers_values_and_shape(self):
        first = probability_sha256(self.tree)
        second = probability_sha256(self.tree.copy())
        changed = self.tree.copy()
        changed[0] = (0.80, 0.10, 0.10)

        self.assertEqual(first, second)
        self.assertNotEqual(first, probability_sha256(changed))

    def test_invalid_probability_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            pool_probabilities(
                np.asarray(((0.8, 0.3),)),
                np.asarray(((0.5, 0.5),)),
                alpha=0.5,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            pool_probabilities(
                np.asarray(((0.5, 0.5),)),
                np.asarray(((0.2, 0.3, 0.5),)),
                alpha=0.5,
            )
        with self.assertRaisesRegex(ValueError, "alpha"):
            pool_probabilities(self.tree, self.neural, alpha=1.1)


if __name__ == "__main__":
    unittest.main()
