import unittest

from scripts.summarize_cross_dataset_a0_mr import (
    _BOT_ROOT,
    _TON_ROOT,
    _comparison,
    _predeclared_rare_labels,
    _seed_stability,
    dataset_stratified_paired_bootstrap,
)


class CrossDatasetSummaryTests(unittest.TestCase):
    def test_summary_uses_completed_bot_root_and_ton_retry_root(self):
        self.assertEqual(_BOT_ROOT.name, "nf_bot")
        self.assertEqual(_BOT_ROOT.parent.name, "cross-dataset-formal-matrix-v1")
        self.assertEqual(_TON_ROOT.name, "nf_ton")
        self.assertEqual(_TON_ROOT.parent.name, "cross-dataset-formal-matrix-v3")

    def test_new_dataset_rare_labels_come_from_frozen_dataset_spec(self):
        self.assertEqual(
            _predeclared_rare_labels({"dataset": "nf_bot"}),
            ("benign", "malware"),
        )
        self.assertEqual(
            _predeclared_rare_labels({"dataset": "nf_ton"}),
            (),
        )

    def test_stratified_bootstrap_equal_weights_datasets(self):
        result = dataset_stratified_paired_bootstrap(
            {
                "first": [0.01, 0.02, 0.03],
                "second": [0.02, 0.03, 0.04],
                "third": [0.03, 0.04, 0.05],
            },
            resamples=2000,
            seed=42,
        )

        self.assertAlmostEqual(result["observed_mean_gain"], 0.03)
        self.assertGreater(result["ci95_lower"], 0.0)
        self.assertEqual(result["dataset_count"], 3)

    def test_seed_stability_accepts_both_locked_setting_id_formats(self):
        cse = {
            f"seed-{seed}-pct-{percent:02d}": 0.01
            for percent in (1, 5, 10)
            for seed in (17, 42, 20260725)
        }
        current = {
            f"pct-{percent:02d}-seed-{seed}": 0.01
            for percent in (1, 5, 10)
            for seed in (17, 42, 20260725)
        }

        self.assertEqual(_seed_stability(cse, cse=True), _seed_stability(current))

    def test_comparison_marks_unavailable_lightgbm_as_na(self):
        macro = {
            condition: {"xgboost": 0.5, "fixed_a0": 0.51, "a0_mr": 0.52}
            for condition in (
                "clean",
                "missing_30pct",
                "missing_50pct",
                "structured__one",
            )
        }
        result = _comparison(
            {"retention": {}, "rare_recall": {}},
            macro,
            lightgbm_available=False,
        )

        self.assertIsNone(
            result["random_condition_mean_a0_mr_deltas"]["lightgbm"]
        )
        self.assertIsNone(result["random_conditions_noninferior"]["lightgbm"])


if __name__ == "__main__":
    unittest.main()
