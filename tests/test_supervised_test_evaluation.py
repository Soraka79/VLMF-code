import json
import tempfile
import unittest
from pathlib import Path

from flowfm.config import load_config
from flowfm.training.supervised import run_supervised_training
from tests.test_supervised_training import _config_text, _fixture, _sample


class SupervisedTestEvaluationTests(unittest.TestCase):
    def test_publishes_metrics_and_efficiency_for_frozen_test_split(self):
        registry, metadata, label_space, splits = _fixture()
        splits = {
            **splits,
            "test": (
                _sample("7", "n:17", "attack", "dos", "DoS"),
                _sample("8", "n:6", "benign", "benign", "Benign"),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "autodl-tmp"
            output_root = data_root / "results"
            config_path = Path(directory) / "finetune.yaml"
            config_path.write_text(_config_text(output_root), encoding="utf-8")
            config = load_config(config_path, expected_workflow="finetune")

            summary = run_supervised_training(
                config,
                splits=splits,
                tokenizer_metadata=metadata,
                label_space=label_space,
                registry=registry,
                data_manifest={"dataset_id": "fixture", "sha256": "data-hash"},
                split_manifest={
                    name: [sample.sample_id for sample in values]
                    for name, values in splits.items()
                },
                repo_dir=Path(__file__).resolve().parents[1],
                data_root=data_root,
                git_commit="d" * 40,
            )

            run_dir = Path(summary["run_dir"])
            metrics = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            test_summary = json.loads(
                (run_dir / "predictions/test_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            efficiency = json.loads(
                (run_dir / "efficiency.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["test"]["native"]["sample_count"], 2)
            self.assertEqual(test_summary["sample_count"], 2)
            self.assertIn("macro_f1", test_summary["metrics"]["native"])
            self.assertGreater(efficiency["test_samples_per_second"], 0)
            self.assertGreater(efficiency["test_seconds_per_sample"], 0)


if __name__ == "__main__":
    unittest.main()
