import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flowfm.training.learned_ensemble import (
    CONDITIONS,
    C_GRID,
    GATE_CONTINUOUS_INDICES,
    candidate_ranking_key,
    dataset_stratified_paired_setting_bootstrap,
    fit_learned_gate,
    fit_logistic_stacker,
    fold_assignment_sha256,
    gate_feature_names,
    gate_features,
    gate_targets,
    make_group_folds,
    predict_locked_gate,
    predict_locked_stacker,
    stacking_features,
)
from scripts.run_learned_ensemble import (
    FROZEN_PROTOCOL_SHA256,
    LearnedAccessAudit,
)


def _branch_probabilities(targets, *, tree_preferred):
    targets = np.asarray(targets, dtype=np.int64)
    class_count = int(targets.max()) + 1
    probabilities = np.full(
        (targets.size, class_count), 0.2 / (class_count - 1), dtype=np.float64
    )
    preferred = np.asarray(tree_preferred, dtype=bool)
    true_probability = np.where(preferred, 0.8, 0.55)
    probabilities[:] = ((1.0 - true_probability) / (class_count - 1))[:, None]
    probabilities[np.arange(targets.size), targets] = true_probability
    return probabilities


def _synthetic_inputs(sample_count=60):
    targets = np.arange(sample_count, dtype=np.int64) % 3
    tree_preferred = (np.arange(sample_count) % 2) == 0
    by_condition = {}
    for index, condition in enumerate(CONDITIONS):
        preferred = np.roll(tree_preferred, index)
        by_condition[condition] = {
            "xgboost": _branch_probabilities(
                targets, tree_preferred=preferred
            ),
            "schemaflow": _branch_probabilities(
                targets, tree_preferred=~preferred
            ),
        }
    groups = np.asarray([f"group-{index:03d}" for index in range(sample_count)])
    folds = make_group_folds(targets, groups)
    return targets, groups, folds, by_condition


class LearnedEnsembleTests(unittest.TestCase):
    def test_stacking_features_are_clipped_log_probabilities(self):
        tree = np.asarray(((1.0, 0.0), (0.25, 0.75)))
        neural = np.asarray(((0.5, 0.5), (0.0, 1.0)))

        features = stacking_features(tree, neural)

        self.assertEqual(features.shape, (2, 4))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(features[0, 0], 0.0)
        self.assertAlmostEqual(features[0, 1], np.log(1e-12))

    def test_gate_features_match_the_frozen_low_capacity_schema(self):
        tree = np.asarray(((0.8, 0.2), (0.4, 0.6)))
        neural = np.asarray(((0.7, 0.3), (0.9, 0.1)))

        features = gate_features(tree, neural, condition_index=1)

        self.assertEqual(features.shape, (2, 11))
        self.assertEqual(len(gate_feature_names()), 11)
        self.assertEqual(GATE_CONTINUOUS_INDICES, tuple(range(3, 10)))
        np.testing.assert_array_equal(features[:, :3], ((0, 1, 0), (0, 1, 0)))
        np.testing.assert_array_equal(features[:, -1], (1, 0))

    def test_gate_target_ties_are_excluded_only_from_fitting(self):
        tree = np.asarray(((0.8, 0.2), (0.5, 0.5), (0.2, 0.8)))
        neural = np.asarray(((0.7, 0.3), (0.5, 0.5), (0.4, 0.6)))
        targets = np.asarray((0, 1, 1))

        binary, eligible = gate_targets(tree, neural, targets)

        np.testing.assert_array_equal(binary, (1, 0, 1))
        np.testing.assert_array_equal(eligible, (True, False, True))

    def test_group_folds_are_deterministic_and_group_exclusive(self):
        targets = np.tile(np.arange(3), 20)
        groups = np.repeat(np.arange(30), 2)

        first = make_group_folds(targets, groups)
        second = make_group_folds(targets, groups)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(fold_assignment_sha256(first), fold_assignment_sha256(second))
        for group in np.unique(groups):
            self.assertEqual(np.unique(first[groups == group]).size, 1)

    def test_candidate_tie_prefers_smaller_C(self):
        rows = [
            {
                "C": C,
                "mean_macro_f1": 0.5,
                "mean_mcc": 0.4,
                "mean_balanced_accuracy": 0.3,
            }
            for C in C_GRID
        ]

        ranked = sorted(rows, key=candidate_ranking_key, reverse=True)

        self.assertEqual(ranked[0]["C"], 0.01)

    def test_fitted_models_use_complete_grid_and_locked_inference(self):
        targets, _, folds, inputs = _synthetic_inputs()

        stacker = fit_logistic_stacker(inputs, targets, folds)
        gate = fit_learned_gate(inputs, targets, folds)

        self.assertEqual(
            {row["C"] for row in stacker["oof_candidates"]}, set(C_GRID)
        )
        self.assertEqual({row["C"] for row in gate["oof_candidates"]}, set(C_GRID))
        self.assertLess(stacker["locked_inference_max_absolute_error"], 1e-12)
        self.assertLess(gate["locked_inference_max_absolute_error"], 1e-12)
        serialized_stacker = json.loads(json.dumps(stacker["model"]))
        serialized_gate = json.loads(json.dumps(gate["model"]))
        tree = inputs["clean"]["xgboost"]
        neural = inputs["clean"]["schemaflow"]
        stacker_probabilities = predict_locked_stacker(
            serialized_stacker, tree, neural
        )
        gate_probabilities = predict_locked_gate(
            serialized_gate, tree, neural, condition_index=0
        )
        np.testing.assert_allclose(stacker_probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(gate_probabilities.sum(axis=1), 1.0)

    def test_audit_rejects_test_access_until_all_54_models_are_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = LearnedAccessAudit()
            audit.lock_protocol(
                root / "protocol.json",
                FROZEN_PROTOCOL_SHA256,
                root / "source-protocol.json",
            )
            with self.assertRaisesRegex(RuntimeError, "54 model locks"):
                audit.enable_test_access(FROZEN_PROTOCOL_SHA256, "abc", 52)
            audit.lock_models(root / "analysis_lock.json", "abc", 54)
            audit.enable_test_access(FROZEN_PROTOCOL_SHA256, "abc", 54)
            audit.test_read(root / "sealed_test" / "predictions.npz", "test")
            self.assertEqual(audit.events[-1]["event"], "sealed_test_input_read")

    def test_dataset_stratified_bootstrap_is_deterministic_and_has_no_p_value(self):
        values = {
            "A": np.linspace(-0.01, 0.01, 9),
            "B": np.linspace(0.00, 0.02, 9),
            "C": np.linspace(0.01, 0.03, 9),
        }

        first = dataset_stratified_paired_setting_bootstrap(
            values, datasets=("A", "B", "C"), resamples=2000, seed=7
        )
        second = dataset_stratified_paired_setting_bootstrap(
            values, datasets=("A", "B", "C"), resamples=2000, seed=7
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_equal_dataset_mean"], 0.01)
        self.assertFalse(first["p_value_inferred"])


if __name__ == "__main__":
    unittest.main()
