import unittest

import numpy as np

from flowfm.training.confirmatory_statistics import (
    confirmatory_metric_payload,
    evaluate_confirmatory_gate,
    paired_cluster_bootstrap_macro_f1,
)


class ConfirmatoryStatisticsTests(unittest.TestCase):
    def test_confirmatory_metrics_add_calibration_and_per_class_details(self):
        targets = np.asarray((0, 1, 1), dtype=np.int64)
        probabilities = np.asarray(
            ((0.8, 0.2), (0.3, 0.7), (0.6, 0.4)), dtype=np.float64
        )

        metrics = confirmatory_metric_payload(
            probabilities, targets, ("benign", "attack")
        )

        self.assertIn("nll", metrics)
        self.assertEqual(set(metrics["per_class_precision"]), {"benign", "attack"})
        self.assertEqual(set(metrics["per_class_f1"]), {"benign", "attack"})

    def test_paired_cluster_bootstrap_detects_positive_gain(self):
        targets = np.tile(np.arange(3, dtype=np.int64), 200)
        weak_predictions = targets.copy()
        weak_predictions[::5] = (weak_predictions[::5] + 1) % 3
        strong = np.eye(3, dtype=np.float64)[targets]
        weak = np.eye(3, dtype=np.float64)[weak_predictions]
        outputs = {
            condition: {"targets": targets}
            for condition in ("clean", "missing_30pct", "missing_50pct")
        }
        first = {condition: strong for condition in outputs}
        second = {condition: weak for condition in outputs}
        groups = tuple(f"{index // 2:064x}" for index in range(len(targets)))

        result = paired_cluster_bootstrap_macro_f1(
            outputs,
            first,
            second,
            groups,
            class_count=3,
            resamples=500,
            seed=7,
            batch_size=25,
        )

        self.assertEqual(result["cluster_count"], 300)
        self.assertGreater(
            result["summaries"]["mean_across_conditions"]["ci95_lower"],
            0.0,
        )
        repeated = paired_cluster_bootstrap_macro_f1(
            outputs,
            first,
            second,
            groups,
            class_count=3,
            resamples=500,
            seed=7,
            batch_size=25,
        )
        self.assertEqual(result, repeated)

    def test_gate_is_conjunctive_and_predeclared(self):
        labels = (
            "Benign",
            "backdoor",
            "ddos",
            "dos",
            "injection",
            "mitm",
            "password",
            "ransomware",
            "scanning",
            "xss",
        )

        def metrics(value, rare_recall):
            return {
                "macro_f1": value,
                "per_class_recall": {
                    label: rare_recall if label in {
                        "backdoor", "mitm", "ransomware"
                    } else 0.8
                    for label in labels
                },
            }

        xgboost = {
            "clean": metrics(0.50, 0.20),
            "missing_30pct": metrics(0.40, 0.15),
            "missing_50pct": metrics(0.30, 0.10),
        }
        fixed = {
            "clean": metrics(0.51, 0.20),
            "missing_30pct": metrics(0.405, 0.15),
            "missing_50pct": metrics(0.305, 0.10),
        }
        locked = {
            "clean": metrics(0.52, 0.21),
            "missing_30pct": metrics(0.42, 0.16),
            "missing_50pct": metrics(0.32, 0.11),
        }
        bootstrap = {
            "summaries": {
                "mean_across_conditions": {"ci95_lower": 0.005}
            }
        }

        gate = evaluate_confirmatory_gate(
            xgboost_metrics=xgboost,
            fixed_a0_metrics=fixed,
            locked_metrics=locked,
            bootstrap=bootstrap,
        )

        self.assertTrue(gate["passed"])
        self.assertTrue(gate["full_matrix_allowed"])
        locked["missing_50pct"] = metrics(0.29, 0.11)
        failed = evaluate_confirmatory_gate(
            xgboost_metrics=xgboost,
            fixed_a0_metrics=fixed,
            locked_metrics=locked,
            bootstrap=bootstrap,
        )
        self.assertFalse(failed["passed"])

    def test_gate_accepts_predeclared_nf_cse_rare_coarse_labels(self):
        labels = ("benign", "dos", "brute_force", "exploit", "botnet")

        def metrics(value, recall):
            return {
                "macro_f1": value,
                "per_class_recall": {label: recall for label in labels},
            }

        xgboost = {
            "clean": metrics(0.50, 0.50),
            "missing_30pct": metrics(0.40, 0.40),
            "missing_50pct": metrics(0.30, 0.30),
        }
        fixed = {
            "clean": metrics(0.50, 0.50),
            "missing_30pct": metrics(0.40, 0.40),
            "missing_50pct": metrics(0.30, 0.30),
        }
        locked = {
            "clean": metrics(0.52, 0.51),
            "missing_30pct": metrics(0.42, 0.41),
            "missing_50pct": metrics(0.32, 0.31),
        }
        gate = evaluate_confirmatory_gate(
            xgboost_metrics=xgboost,
            fixed_a0_metrics=fixed,
            locked_metrics=locked,
            bootstrap={
                "summaries": {
                    "mean_across_conditions": {"ci95_lower": 0.01}
                }
            },
            rare_labels=("brute_force", "exploit", "botnet"),
            rare_label_definition=(
                "official full-source coarse prevalence below 1%"
            ),
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(
            gate["thresholds"]["rare_labels"],
            ["brute_force", "exploit", "botnet"],
        )


if __name__ == "__main__":
    unittest.main()
