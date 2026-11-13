import unittest

import numpy as np
import torch

from flowfm.data.contracts import FlowSample
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.training.safe_fusion import (
    CONSISTENCY_CONDITIONS,
    apply_scalar_temperature,
    balanced_softmax_cross_entropy,
    deterministic_paired_missing_view,
    evaluate_seed42_setting_gate,
    fit_scalar_temperature,
    jensen_shannon_consistency,
    safe_route_and_calibrate,
)


def _full_sample(index):
    specs = SCHEMAFLOW_MAIN_REGISTRY.model_features
    categorical = tuple(spec.feature_type == "categorical" for spec in specs)
    return FlowSample(
        features=tuple(1 if value else float(index + 1) for value in categorical),
        feature_ids=tuple(spec.name for spec in specs),
        categorical_mask=categorical,
        missing_mask=(False,) * len(specs),
        category_tokens=tuple("n:1" if value else None for value in categorical),
        binary_label="attack",
        coarse_label="dos",
        native_label="DoS",
        dataset_id="fixture",
        sample_id=f"sample-{index:04d}",
        feature_key=f"feature-{index:04d}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=str(index),
    )


def _gate_metric(macro_f1, *, nll=0.5, ece=0.06, recall=0.5):
    return {
        "macro_f1": macro_f1,
        "nll": nll,
        "ece": ece,
        "per_class_support": {"exploit": 200},
        "per_class_recall": {"exploit": recall},
    }


class SafeTrainingObjectiveTests(unittest.TestCase):
    def test_balanced_softmax_adds_log_train_priors(self):
        logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float64)
        targets = torch.tensor([0, 1])
        counts = torch.tensor([1.0, 3.0], dtype=torch.float64)

        actual = balanced_softmax_cross_entropy(logits, targets, counts)
        expected = torch.nn.functional.cross_entropy(
            logits + counts.log().unsqueeze(0), targets
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_js_consistency_is_symmetric_and_zero_for_equal_logits(self):
        first = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
        second = torch.tensor([[-0.5, 1.5], [1.0, 0.0]])

        forward = jensen_shannon_consistency(first, second)
        reverse = jensen_shannon_consistency(second, first)

        self.assertGreater(float(forward), 0.0)
        self.assertTrue(torch.allclose(forward, reverse))
        self.assertLess(float(jensen_shannon_consistency(first, first)), 1e-7)

    def test_paired_view_is_deterministic_and_covers_frozen_conditions(self):
        samples = tuple(_full_sample(index) for index in range(700))

        first, first_report = deterministic_paired_missing_view(samples, seed=42)
        second, second_report = deterministic_paired_missing_view(samples, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(set(first_report["condition_counts"]), set(CONSISTENCY_CONDITIONS))
        self.assertEqual(sum(first_report["condition_counts"].values()), len(samples))
        self.assertTrue(any(sample.missing_mask != source.missing_mask for sample, source in zip(first, samples)))
        self.assertFalse(first_report["validation_or_test_records_used"])


class SafeRoutingAndCalibrationTests(unittest.TestCase):
    def test_scalar_temperature_improves_validation_nll_without_changing_argmax(self):
        probabilities = np.asarray(
            [[0.99, 0.01], [0.99, 0.01], [0.80, 0.20], [0.20, 0.80]],
            dtype=np.float64,
        )
        targets = np.asarray([0, 1, 0, 1], dtype=np.int64)

        fit = fit_scalar_temperature(probabilities, targets)
        calibrated = apply_scalar_temperature(probabilities, fit["temperature"])

        self.assertTrue(fit["improved_nll"])
        self.assertLess(fit["nll_after"], fit["nll_before"])
        np.testing.assert_array_equal(
            calibrated.argmax(axis=1), probabilities.argmax(axis=1)
        )

    def test_nonpositive_bootstrap_lower_bound_forces_xgboost_fallback(self):
        targets = np.asarray([index % 2 for index in range(200)], dtype=np.int64)
        probabilities = np.asarray(
            [[0.8, 0.2] if target == 0 else [0.2, 0.8] for target in targets],
            dtype=np.float64,
        )
        outputs = {"clean": {"targets": targets}}

        payload, routed, _ = safe_route_and_calibrate(
            outputs,
            {"clean": probabilities.copy()},
            {"clean": probabilities.copy()},
            tuple(f"group-{index}" for index in range(len(targets))),
            ("benign", "exploit"),
            rare_labels=("exploit",),
            bootstrap_resamples=200,
        )

        self.assertEqual(payload["decisions"]["clean"]["route"], "xgboost_fallback")
        np.testing.assert_array_equal(routed["clean"], probabilities)

    def test_seed42_gate_applies_every_predeclared_requirement(self):
        conditions = (
            "clean",
            "missing_30pct",
            "missing_50pct",
            "structured__transport_protocol",
            "structured__traffic_volume_timing",
            "structured__tcp_state_retransmission",
            "structured__packet_length_distribution",
            "structured__application_protocol_metadata",
        )
        current_values = {
            "clean": 0.80,
            "missing_30pct": 0.68,
            "missing_50pct": 0.48,
            **{condition: 0.70 for condition in conditions[3:]},
        }
        safe_values = {
            "clean": 0.81,
            "missing_30pct": 0.73,
            "missing_50pct": 0.51,
            **{condition: 0.72 for condition in conditions[3:]},
        }
        current = {
            condition: _gate_metric(current_values[condition])
            for condition in conditions
        }
        safe = {
            condition: _gate_metric(safe_values[condition]) for condition in conditions
        }
        calibrated = {
            condition: _gate_metric(safe_values[condition], nll=0.45, ece=0.05)
            for condition in conditions
        }
        xgboost = {condition: _gate_metric(0.70) for condition in conditions}

        result = evaluate_seed42_setting_gate(
            current_a0_mr_metrics=current,
            safe_metrics=safe,
            calibrated_metrics=calibrated,
            xgboost_metrics=xgboost,
            rare_labels=("exploit",),
        )

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
