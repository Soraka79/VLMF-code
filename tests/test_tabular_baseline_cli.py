import tempfile
import unittest
from pathlib import Path

import numpy as np

from flowfm.data.contracts import FlowSample
from flowfm.data.labels import NF_UNSW_LABEL_SPACE
from scripts.run_nf_unsw_tabular_baseline import (
    _copy_manifest,
    _metric_payload,
    _target_arrays,
    _task_names,
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


class _ProbabilityModel:
    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self.classes_ = np.arange(self._probabilities.shape[1], dtype=np.int64)

    def predict_proba(self, matrix):
        if len(matrix) != len(self._probabilities):
            raise AssertionError("matrix rows do not match fixture probabilities")
        return self._probabilities


class TabularBaselineCLITests(unittest.TestCase):
    def test_task_selection_is_explicit_and_stable(self):
        self.assertEqual(_task_names("all"), ("binary", "coarse", "native"))
        for task in ("binary", "coarse", "native"):
            with self.subTest(task=task):
                self.assertEqual(_task_names(task), (task,))
        with self.assertRaises(ValueError):
            _task_names("unsupported")

    def test_manifest_publication_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            destination = root / "result" / "split_manifest.json"
            payload = b'{  "z": 1,\n"a": [2, 3]}\n'
            source.write_bytes(payload)

            _copy_manifest(source, destination)

            self.assertEqual(destination.read_bytes(), payload)

    def test_targets_use_the_versioned_nf_unsw_label_space(self):
        samples = (
            _sample("1", "benign", "benign", "Benign"),
            _sample("2", "attack", "dos", "DoS"),
            _sample("3", "attack", "scan", "Reconnaissance"),
            _sample("4", "attack", "backdoor", "Backdoor"),
        )

        targets = _target_arrays(samples, NF_UNSW_LABEL_SPACE)

        self.assertEqual(targets["binary"].tolist(), [0, 1, 1, 1])
        self.assertEqual(targets["coarse"].tolist(), [0, 1, 2, 6])
        self.assertEqual(targets["native"].tolist(), [0, 4, 5, 7])

    def test_multiclass_metrics_publish_complete_fairness_evidence(self):
        model = _ProbabilityModel(
            (
                (0.90, 0.05, 0.05),
                (0.10, 0.80, 0.10),
                (0.05, 0.05, 0.90),
                (0.10, 0.20, 0.70),
            )
        )
        targets = np.asarray((0, 1, 2, 2), dtype=np.int64)

        metrics = _metric_payload(
            model,
            np.zeros((4, 1), dtype=np.float32),
            targets,
            ("benign", "dos", "scan"),
        )

        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(
            metrics["per_class_recall"],
            {"benign": 1.0, "dos": 1.0, "scan": 1.0},
        )
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
        )
        self.assertEqual(metrics["class_counts"], {"benign": 1, "dos": 1, "scan": 2})
        for name in ("mcc", "macro_pr_auc", "ece", "brier"):
            self.assertTrue(np.isfinite(metrics[name]), name)

    def test_metrics_reject_probability_columns_that_do_not_match_labels(self):
        model = _ProbabilityModel(((0.7, 0.3), (0.2, 0.8)))
        with self.assertRaisesRegex(ValueError, "probability columns"):
            _metric_payload(
                model,
                np.zeros((2, 1), dtype=np.float32),
                np.asarray((0, 1), dtype=np.int64),
                ("benign", "dos", "scan"),
            )


if __name__ == "__main__":
    unittest.main()
