import unittest

import numpy as np

from flowfm.data.contracts import FlowSample
from flowfm.data.labels import NF_UNSW_LABEL_SPACE
from flowfm.training.baseline_evaluation import (
    metric_payload,
    target_arrays,
    task_labels,
    task_names,
)


def _sample(sample_id, binary, coarse, native):
    return FlowSample(
        features=(1.0,),
        feature_ids=("IN_BYTES",),
        categorical_mask=(False,),
        missing_mask=(False,),
        category_tokens=(None,),
        binary_label=binary,
        coarse_label=coarse,
        native_label=native,
        dataset_id=NF_UNSW_LABEL_SPACE.dataset_id,
        sample_id=sample_id,
        feature_key=f"feature-{sample_id}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id,
    )


class BaselineEvaluationTests(unittest.TestCase):
    def test_task_selection_and_labels_are_versioned(self):
        self.assertEqual(task_names("all"), ("binary", "coarse", "native"))
        for task in ("binary", "coarse", "native"):
            with self.subTest(task=task):
                self.assertEqual(task_names(task), (task,))
                self.assertEqual(
                    task_labels(task),
                    tuple(getattr(NF_UNSW_LABEL_SPACE, f"{task}_labels")),
                )
        with self.assertRaises(ValueError):
            task_names("unsupported")
        with self.assertRaises(ValueError):
            task_labels("unsupported")

    def test_targets_use_the_frozen_label_space(self):
        samples = (
            _sample("1", "benign", "benign", "Benign"),
            _sample("2", "attack", "dos", "DoS"),
            _sample("3", "attack", "scan", "Reconnaissance"),
            _sample("4", "attack", "backdoor", "Backdoor"),
        )
        targets = target_arrays(samples)
        self.assertEqual(targets["binary"].tolist(), [0, 1, 1, 1])
        self.assertEqual(targets["coarse"].tolist(), [0, 1, 2, 6])
        self.assertEqual(targets["native"].tolist(), [0, 4, 5, 7])

    def test_metric_payload_is_complete_and_probability_based(self):
        probabilities = np.asarray(
            (
                (0.90, 0.05, 0.05),
                (0.10, 0.80, 0.10),
                (0.05, 0.05, 0.90),
                (0.10, 0.20, 0.70),
            ),
            dtype=np.float64,
        )
        metrics = metric_payload(
            probabilities,
            np.asarray((0, 1, 2, 2), dtype=np.int64),
            ("benign", "dos", "scan"),
        )
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        for name in ("mcc", "macro_pr_auc", "ece", "brier"):
            self.assertTrue(np.isfinite(metrics[name]), name)

    def test_metric_payload_rejects_invalid_probability_contract(self):
        targets = np.asarray((0, 1), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "probability columns"):
            metric_payload(
                np.asarray(((0.7, 0.3), (0.2, 0.8))),
                targets,
                ("benign", "dos", "scan"),
            )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            metric_payload(
                np.asarray(((0.8, 0.3), (0.2, 0.8))),
                targets,
                ("benign", "attack"),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            metric_payload(
                np.asarray(((np.nan, np.nan), (0.2, 0.8))),
                targets,
                ("benign", "attack"),
            )


class _ProbabilityModel:
    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self.classes_ = np.arange(self._probabilities.shape[1], dtype=np.int64)

    def predict_proba(self, matrix):
        if len(matrix) != len(self._probabilities):
            raise AssertionError("matrix row mismatch")
        return self._probabilities


class BaselineMetricEquivalenceTests(unittest.TestCase):
    def test_shared_metrics_match_the_existing_tabular_contract(self):
        from scripts.run_nf_unsw_tabular_baseline import _metric_payload

        probabilities = np.asarray(
            ((0.8, 0.2), (0.1, 0.9), (0.4, 0.6)),
            dtype=np.float64,
        )
        targets = np.asarray((0, 1, 0), dtype=np.int64)
        labels = ("benign", "attack")
        existing = _metric_payload(
            _ProbabilityModel(probabilities),
            np.zeros((3, 1), dtype=np.float32),
            targets,
            labels,
        )
        self.assertEqual(metric_payload(probabilities, targets, labels), existing)


if __name__ == "__main__":
    unittest.main()
