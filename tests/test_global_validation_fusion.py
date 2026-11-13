import unittest

import numpy as np

from scripts.run_global_validation_fusion import (
    ALPHA_GRID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    DATASETS,
    METHODS,
    AccessAudit,
    dataset_stratified_paired_setting_bootstrap,
    evaluate_global_policy,
    protocol_payload,
    select_global_policy,
)


def candidate_searches(
    overrides=None,
    *,
    methods=("linear", "log"),
    alpha_grid=(0.0, 1.0),
):
    overrides = overrides or {}
    result = {}
    for condition in CONDITIONS:
        candidates = []
        for method in methods:
            for alpha in alpha_grid:
                values = {
                    "macro_f1": 0.5,
                    "mcc": 0.4,
                    "balanced_accuracy": 0.3,
                }
                values.update(overrides.get((condition, method, alpha), {}))
                candidates.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        **values,
                    }
                )
        result[condition] = {
            "methods": list(methods),
            "alpha_grid": list(alpha_grid),
            "candidates": candidates,
        }
    return result


class GlobalValidationFusionTests(unittest.TestCase):
    def test_protocol_locks_exact_scope_and_bootstrap(self):
        protocol = protocol_payload()

        self.assertEqual(protocol["experiment_name"], "global_validation_fusion_v1")
        self.assertEqual(protocol["scope"]["setting_count"], 27)
        self.assertEqual(tuple(protocol["scope"]["conditions"]), CONDITIONS)
        self.assertEqual(tuple(protocol["selection"]["methods"]), METHODS)
        self.assertEqual(tuple(protocol["selection"]["alpha_grid"]), ALPHA_GRID)
        self.assertFalse(protocol["selection"]["vlmf_admission_gate_used"])
        self.assertFalse(protocol["selection"]["fixed_alpha_fallback_used"])
        self.assertEqual(
            protocol["aggregation"]["bootstrap"]["resamples"],
            BOOTSTRAP_RESAMPLES,
        )
        self.assertEqual(
            protocol["aggregation"]["bootstrap"]["seed"], BOOTSTRAP_SEED
        )
        self.assertFalse(protocol["aggregation"]["bootstrap"]["p_value_inferred"])

    def test_selection_averages_all_three_conditions_before_ranking(self):
        overrides = {}
        for condition in CONDITIONS:
            overrides[(condition, "log", 0.0)] = {"macro_f1": 0.6}
        overrides[("clean", "linear", 1.0)] = {"macro_f1": 0.9}
        overrides[("missing_30pct", "linear", 1.0)] = {"macro_f1": 0.4}
        overrides[("missing_50pct", "linear", 1.0)] = {"macro_f1": 0.4}

        selected = select_global_policy(
            candidate_searches(overrides),
            methods=("linear", "log"),
            alpha_grid=(0.0, 1.0),
        )

        self.assertEqual(selected["policy"]["method"], "log")
        self.assertEqual(selected["policy"]["alpha"], 0.0)
        self.assertAlmostEqual(
            selected["validation_objective"]["mean_macro_f1"], 0.6
        )

    def test_tie_break_uses_mcc_then_balanced_accuracy(self):
        overrides = {}
        for condition in CONDITIONS:
            overrides[(condition, "log", 0.0)] = {
                "macro_f1": 0.7,
                "mcc": 0.7,
                "balanced_accuracy": 0.2,
            }
            overrides[(condition, "linear", 1.0)] = {
                "macro_f1": 0.7,
                "mcc": 0.6,
                "balanced_accuracy": 0.9,
            }
        selected = select_global_policy(
            candidate_searches(overrides),
            methods=("linear", "log"),
            alpha_grid=(0.0, 1.0),
        )
        self.assertEqual(selected["policy"], {
            "method": "log",
            "alpha": 0.0,
            "alpha_semantics": "xgboost_probability_weight",
            "applies_to_test_conditions": list(CONDITIONS),
        })
        self.assertEqual(
            selected["selection_basis"]["deciding_criterion_vs_runner_up"],
            "three_condition_mean_mcc",
        )

        for condition in CONDITIONS:
            overrides[(condition, "log", 0.0)]["mcc"] = 0.6
        selected = select_global_policy(
            candidate_searches(overrides),
            methods=("linear", "log"),
            alpha_grid=(0.0, 1.0),
        )
        self.assertEqual(selected["policy"]["method"], "linear")
        self.assertEqual(
            selected["selection_basis"]["deciding_criterion_vs_runner_up"],
            "three_condition_mean_balanced_accuracy",
        )

    def test_final_ties_prefer_linear_then_higher_xgboost_alpha(self):
        selected = select_global_policy(
            candidate_searches(),
            methods=("linear", "log"),
            alpha_grid=(0.0, 1.0),
        )

        self.assertEqual(selected["policy"]["method"], "linear")
        self.assertEqual(selected["policy"]["alpha"], 1.0)
        self.assertEqual(
            selected["selection_basis"]["deciding_criterion_vs_runner_up"],
            "higher_xgboost_alpha",
        )
        self.assertEqual(
            selected["selection_basis"]["complete_candidate_ranking"][2][
                "method"
            ],
            "log",
        )

    def test_incomplete_candidate_pool_is_rejected(self):
        searches = candidate_searches()
        searches["missing_50pct"]["candidates"].pop()

        with self.assertRaisesRegex(RuntimeError, "candidate pool mismatch"):
            select_global_policy(
                searches,
                methods=("linear", "log"),
                alpha_grid=(0.0, 1.0),
            )

    def test_audit_forbids_test_read_until_both_locks_exist(self):
        audit = AccessAudit()

        with self.assertRaisesRegex(RuntimeError, "before policy lock"):
            audit.test_read("sealed_test/predictions.npz", "test")
        audit.lock_protocol("protocol.json", "a" * 64)
        audit.validation_read("selection.json", "validation")
        with self.assertRaisesRegex(RuntimeError, "protocol and policy locks"):
            audit.enable_test_access("a" * 64, "b" * 64)
        audit.lock_policies("policies.json", "b" * 64)
        audit.enable_test_access("a" * 64, "b" * 64)
        audit.test_read("sealed_test/predictions.npz", "test")

        policy_sequence = next(
            row["sequence"] for row in audit.events if row["event"] == "policies_locked"
        )
        test_sequence = next(
            row["sequence"]
            for row in audit.events
            if row["event"] == "sealed_test_input_read"
        )
        self.assertGreater(test_sequence, policy_sequence)

    def test_one_policy_is_applied_to_all_test_conditions(self):
        targets = np.asarray((0, 1, 0, 1), dtype=np.int64)
        tree = np.asarray(
            ((0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.3, 0.7)),
            dtype=np.float64,
        )
        neural = np.asarray(
            ((0.6, 0.4), (0.4, 0.6), (0.7, 0.3), (0.2, 0.8)),
            dtype=np.float64,
        )
        probabilities = {
            condition: {
                "xgboost": tree,
                "schemaflow_pretrained": neural,
            }
            for condition in CONDITIONS
        }

        result = evaluate_global_policy(
            probabilities,
            targets,
            2,
            {"method": "linear", "alpha": 0.75},
        )

        self.assertEqual(tuple(result), CONDITIONS)
        self.assertEqual(
            {
                (row["applied_policy"]["method"], row["applied_policy"]["alpha"])
                for row in result.values()
            },
            {("linear", 0.75)},
        )

    def test_bootstrap_is_dataset_stratified_equal_weight_and_deterministic(self):
        values = {
            dataset: np.full(9, offset, dtype=np.float64)
            for dataset, offset in zip(DATASETS, (0.03, -0.01, 0.01))
        }

        first = dataset_stratified_paired_setting_bootstrap(values)
        second = dataset_stratified_paired_setting_bootstrap(values)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_equal_dataset_mean"], 0.01)
        self.assertAlmostEqual(first["ci95_lower"], 0.01)
        self.assertAlmostEqual(first["ci95_upper"], 0.01)
        self.assertEqual(first["resamples"], 20_000)
        self.assertEqual(first["seed"], 20260730)
        self.assertFalse(first["p_value_inferred"])


if __name__ == "__main__":
    unittest.main()
