import unittest

import numpy as np

from flowfm.training.vlmf_strengthening import (
    ALPHA_GRID,
    MAIN_CONDITIONS,
    complementarity_payload,
    route_stability_record,
    select_branch_policies,
    select_threshold_variant,
    threshold_variants,
    validate_condition_searches,
)
from scripts.run_vlmf_strengthening_suite import (
    DATASETS,
    _paired_comparison_summary,
    protocol_payload,
)


def complete_searches(overrides=None):
    overrides = overrides or {}
    searches = {}
    for condition in MAIN_CONDITIONS:
        candidates = []
        for method in ("linear", "log"):
            for alpha in ALPHA_GRID:
                row = {
                    "method": method,
                    "alpha": alpha,
                    "macro_f1": 0.2,
                    "mcc": 0.1,
                    "balanced_accuracy": 0.1,
                }
                row.update(overrides.get((condition, method, alpha), {}))
                candidates.append(row)
        ranked = sorted(
            candidates,
            key=lambda row: (
                row["macro_f1"],
                row["mcc"],
                row["balanced_accuracy"],
                row["method"] == "linear",
                row["alpha"],
            ),
            reverse=True,
        )
        searches[condition] = {
            "methods": ["linear", "log"],
            "alpha_grid": list(ALPHA_GRID),
            "selected": {
                "method": ranked[0]["method"],
                "alpha": ranked[0]["alpha"],
            },
            "candidates": candidates,
        }
    return searches


class VLMFStrengtheningTests(unittest.TestCase):
    def test_paired_summary_names_its_actual_difference_column(self):
        rows = []
        for dataset in DATASETS:
            for setting in range(9):
                for condition in MAIN_CONDITIONS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "setting_id": f"setting-{setting}",
                            "train_percent": 10,
                            "model_seed": setting,
                            "condition": condition,
                            "canonical_minus_variant_macro_f1": 0.01,
                        }
                    )

        result = _paired_comparison_summary(
            rows, "canonical_minus_variant_macro_f1"
        )

        self.assertEqual(
            result["dataset_stratified_paired_setting_bootstrap"]["pairing_unit"],
            "same_setting_three_condition_mean_canonical_minus_variant_macro_f1",
        )

    def test_protocol_freezes_scope_variants_and_test_oracle_boundary(self):
        protocol = protocol_payload()

        self.assertEqual(protocol["scope"]["setting_count"], 27)
        self.assertEqual(tuple(protocol["scope"]["conditions"]), MAIN_CONDITIONS)
        self.assertFalse(protocol["scope"]["base_models_retrained"])
        self.assertFalse(protocol["scope"]["learned_models_included"])
        self.assertEqual(
            len(protocol["threshold_sensitivity"]["variants"]), 14
        )
        self.assertEqual(
            protocol["route_stability"]["test_oracle_role"],
            "post_lock_diagnostic_only",
        )
        self.assertFalse(
            protocol["route_stability"]["oracle_used_to_redefine_or_select_method"]
        )
        self.assertEqual(
            protocol["aggregation"]["bootstrap"]["resamples"], 20000
        )
        self.assertEqual(protocol["aggregation"]["bootstrap"]["seed"], 20260730)
        self.assertFalse(
            protocol["aggregation"]["bootstrap"]["p_value_inferred"]
        )

    def test_complete_candidate_pool_is_required(self):
        searches = complete_searches()
        searches["missing_50pct"]["candidates"].pop()

        with self.assertRaisesRegex(ValueError, "candidate pool mismatch"):
            validate_condition_searches(searches)

    def test_global_and_conditioned_branch_selection_are_distinct(self):
        overrides = {}
        branch_scores = {
            "clean": (0.80, 0.70),
            "missing_30pct": (0.40, 0.65),
            "missing_50pct": (0.40, 0.65),
        }
        for condition, (tree, neural) in branch_scores.items():
            overrides[(condition, "linear", 1.0)] = {
                "macro_f1": tree,
                "mcc": tree,
                "balanced_accuracy": tree,
            }
            overrides[(condition, "linear", 0.0)] = {
                "macro_f1": neural,
                "mcc": neural,
                "balanced_accuracy": neural,
            }

        selected = select_branch_policies(complete_searches(overrides))

        self.assertEqual(selected["global"]["branch"], "schemaflow_pretrained")
        self.assertEqual(
            selected["conditioned"]["clean"]["branch"], "xgboost"
        )
        self.assertEqual(
            selected["conditioned"]["missing_30pct"]["branch"],
            "schemaflow_pretrained",
        )
        self.assertEqual(
            selected["conditioned"]["missing_50pct"]["branch"],
            "schemaflow_pretrained",
        )

    def test_branch_ties_prefer_xgboost(self):
        overrides = {}
        for condition in MAIN_CONDITIONS:
            for alpha in (0.0, 1.0):
                overrides[(condition, "linear", alpha)] = {
                    "macro_f1": 0.8,
                    "mcc": 0.7,
                    "balanced_accuracy": 0.6,
                }
        selected = select_branch_policies(complete_searches(overrides))

        self.assertEqual(selected["global"]["branch"], "xgboost")
        self.assertEqual(
            {row["branch"] for row in selected["conditioned"].values()},
            {"xgboost"},
        )

    def test_canonical_gate_passes_and_failed_gate_uses_fixed_fallback(self):
        overrides = {}
        for condition in MAIN_CONDITIONS:
            overrides[(condition, "linear", 0.8)] = {
                "macro_f1": 0.65,
                "mcc": 0.6,
                "balanced_accuracy": 0.6,
            }
        searches = complete_searches(overrides)
        global_policy = {"method": "linear", "alpha": 0.9}
        passing_fixed = {
            name: {"macro_f1": 0.60} for name in MAIN_CONDITIONS
        }
        canonical = next(row for row in threshold_variants() if row["name"] == "canonical")

        passed = select_threshold_variant(
            searches, passing_fixed, global_policy, canonical
        )
        self.assertTrue(passed["gate_passed"])
        self.assertEqual(
            {row["alpha"] for row in passed["locked_policies"].values()},
            {0.8},
        )

        failing_fixed = {
            name: {"macro_f1": 0.66} for name in MAIN_CONDITIONS
        }
        failed = select_threshold_variant(
            searches, failing_fixed, global_policy, canonical
        )
        self.assertFalse(failed["gate_passed"])
        self.assertEqual(
            {row["alpha"] for row in failed["locked_policies"].values()},
            {0.55},
        )

    def test_global_fallback_and_forced_simplification_are_explicit(self):
        searches = complete_searches()
        fixed = {name: {"macro_f1": 0.9} for name in MAIN_CONDITIONS}
        global_policy = {"method": "log", "alpha": 0.75}
        global_variant = next(
            row
            for row in threshold_variants()
            if row["name"] == "fallback_global_validation_fusion"
        )
        forced_variant = next(
            row
            for row in threshold_variants()
            if row["name"] == "always_conditioned_linear_or_log"
        )

        fallback = select_threshold_variant(
            searches, fixed, global_policy, global_variant
        )
        forced = select_threshold_variant(
            searches, fixed, global_policy, forced_variant
        )

        self.assertFalse(fallback["gate_passed"])
        self.assertEqual(
            {
                (row["method"], row["alpha"])
                for row in fallback["locked_policies"].values()
            },
            {("log", 0.75)},
        )
        self.assertTrue(forced["gate_passed"])

    def test_complementarity_partition_is_exhaustive(self):
        targets = np.asarray((0, 0, 1, 1, 0), dtype=np.int64)
        tree_predictions = np.asarray((0, 0, 0, 0, 1), dtype=np.int64)
        neural_predictions = np.asarray((0, 1, 1, 0, 0), dtype=np.int64)
        tree = np.eye(2, dtype=np.float64)[tree_predictions]
        neural = np.eye(2, dtype=np.float64)[neural_predictions]

        result = complementarity_payload(tree, neural, targets)

        self.assertEqual(result["counts"]["both_correct"], 1)
        self.assertEqual(result["counts"]["xgboost_only_correct"], 1)
        self.assertEqual(result["counts"]["schemaflow_only_correct"], 2)
        self.assertEqual(result["counts"]["both_wrong_same_prediction"], 1)
        self.assertEqual(result["counts"]["both_wrong_different_prediction"], 0)
        self.assertAlmostEqual(
            result["rates"]["prediction_disagreement_rate"], 3 / 5
        )
        self.assertAlmostEqual(result["rates"]["oracle_branch_accuracy"], 4 / 5)

    def test_route_stability_reports_validation_and_deployed_regret(self):
        ranking = [
            {
                "method": "log",
                "alpha": 0.4,
                "macro_f1": 0.80,
                "mcc": 0.7,
                "balanced_accuracy": 0.7,
            },
            {
                "method": "linear",
                "alpha": 0.6,
                "macro_f1": 0.78,
                "mcc": 0.7,
                "balanced_accuracy": 0.7,
            },
            {
                "method": "linear",
                "alpha": 0.55,
                "macro_f1": 0.75,
                "mcc": 0.7,
                "balanced_accuracy": 0.7,
            },
        ]

        result = route_stability_record(
            {"method": "linear", "alpha": 0.6},
            ranking,
            deployed_policy={"method": "linear", "alpha": 0.55},
        )

        self.assertFalse(result["validation_exact_match"])
        self.assertEqual(result["validation_policy_test_rank"], 2)
        self.assertAlmostEqual(
            result["validation_policy_test_macro_f1_regret"], 0.02
        )
        self.assertEqual(result["deployed_policy_test_rank"], 3)
        self.assertAlmostEqual(
            result["deployed_policy_test_macro_f1_regret"], 0.05
        )


if __name__ == "__main__":
    unittest.main()
