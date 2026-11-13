import unittest

from scripts import run_nf_cse_a0_mr_safe as safe


def _metric(macro_f1, *, nll=0.5, ece=0.06, recall=0.5):
    return {
        "macro_f1": macro_f1,
        "mcc": macro_f1,
        "nll": nll,
        "ece": ece,
        "per_class_support": {
            "brute_force": 200,
            "exploit": 200,
            "botnet": 200,
        },
        "per_class_recall": {
            "brute_force": recall,
            "exploit": recall,
            "botnet": recall,
        },
    }


def _result(train_percent):
    safe_values = {
        "clean": 0.81,
        "missing_30pct": 0.73,
        "missing_50pct": 0.51,
        **{condition: 0.72 for condition in safe._FORMAL_CONDITIONS[3:]},
    }
    routed = {
        condition: _metric(safe_values[condition])
        for condition in safe._FORMAL_CONDITIONS
    }
    calibrated = {
        condition: _metric(safe_values[condition], nll=0.45, ece=0.05)
        for condition in safe._FORMAL_CONDITIONS
    }
    xgboost = {
        condition: _metric(0.70) for condition in safe._FORMAL_CONDITIONS
    }
    return {
        "train_percent": train_percent,
        "candidate": safe._CANDIDATES[1],
        "safe_route_and_calibration": {
            "routed_metrics": routed,
            "calibrated_metrics": calibrated,
            "xgboost_metrics": xgboost,
        },
    }


class NFCSEA0MRSafeProtocolTests(unittest.TestCase):
    def test_candidate_grid_is_only_the_predeclared_two_by_two_ablation(self):
        self.assertEqual(len(safe._CANDIDATES), 8)
        self.assertEqual(
            {candidate["variant"] for candidate in safe._CANDIDATES},
            {"V0", "V1", "V2", "V3"},
        )
        self.assertEqual(
            {
                candidate["consistency_weight"]
                for candidate in safe._CANDIDATES
                if candidate["variant"] in {"V2", "V3"}
            },
            {0.05, 0.10, 0.20},
        )

    def test_read_guard_blocks_old_test_material_but_allows_selection_config(self):
        self.assertIsNotNone(
            safe._forbidden_reason(safe._BASE_SPLIT / "sealed_test.json")
        )
        self.assertIsNotNone(
            safe._forbidden_reason(
                safe._RESULTS_ROOT
                / "nf-cse-formal-matrix-v2/state/metrics.json"
            )
        )
        self.assertIsNone(
            safe._forbidden_reason(
                safe._RESULTS_ROOT
                / "nf-cse-formal-matrix-v2/configs/seed-42-pct-01-pretrained.yaml"
            )
        )

    def test_aggregate_seed42_gate_uses_both_training_percentages(self):
        current_values = {
            "clean": 0.80,
            "missing_30pct": 0.68,
            "missing_50pct": 0.48,
            **{condition: 0.70 for condition in safe._FORMAL_CONDITIONS[3:]},
        }
        current = [
            {
                condition: _metric(current_values[condition])
                for condition in safe._FORMAL_CONDITIONS
            }
            for _ in safe._TRAIN_PERCENTS
        ]

        gate = safe._aggregate_gate([_result(1), _result(5)], current)

        self.assertTrue(gate["passed"])
        self.assertEqual(set(gate["random_deltas_by_setting"]), {"1", "5"})
        self.assertTrue(all(gate["checks"].values()))


if __name__ == "__main__":
    unittest.main()
