import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flowfm.cli import build_parser, dispatch
from flowfm.config import load_config
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.training.supervised import run_nf_ton_finetune
from scripts.smoke_schemaflow_nf_ton import build_parser as build_smoke_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_text(output_root):
    return f"""workflow: finetune
experiment:
  name: schemaflow-nf-ton-test
  output_root: {output_root}
  seed: 17
  enforce_autodl_data_disk: true
schema:
  version: nf-ton-iot-v2-model-41-v1
runtime:
  device: cpu
  execution: train
data:
  dataset_id: NF-ToN-IoT-v2
  split_dir: /root/autodl-tmp/nf-ton-split
  preprocessor_dir: /root/autodl-tmp/nf-ton-preprocessor
  data_manifest: /root/autodl-tmp/nf-ton-manifest.json
model:
  name: schemaflow_fm_supervised
  d_model: 8
  nhead: 2
  num_layers: 1
  dim_feedforward: 16
  dropout: 0.0
  adapter_dim: 4
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.01
  gradient_clip: 1.0
  scheduler_gamma: 0.9
  shuffle: true
  class_balance: inverse_frequency
  loss_weights:
    binary: 1.0
    coarse: 1.0
    native: 1.0
"""


class NFTONTrainingContractTests(unittest.TestCase):
    def test_label_space_and_real_smoke_config_are_frozen(self):
        self.assertEqual(NF_TON_LABEL_SPACE.dataset_id, "NF-ToN-IoT-v2")
        self.assertEqual(len(NF_TON_LABEL_SPACE.binary_labels), 2)
        self.assertEqual(len(NF_TON_LABEL_SPACE.coarse_labels), 8)
        self.assertEqual(len(NF_TON_LABEL_SPACE.native_labels), 10)
        config = load_config(
            PROJECT_ROOT / "configs/finetune_nf_ton_schemaflow_smoke.yaml",
            expected_workflow="finetune",
        )
        self.assertEqual(config.data["dataset_id"], "NF-ToN-IoT-v2")
        self.assertEqual(config.schema["version"], "nf-ton-iot-v2-model-41-v1")
        self.assertEqual(config.training["epochs"], 1)
        self.assertEqual(config.training["batch_size"], 256)
        self.assertEqual(config.runtime["device"], "cuda")

    def test_finetune_cli_dispatches_nf_ton_dataset_to_thin_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "finetune.yaml"
            config_path.write_text(
                _config_text(Path(directory) / "results"),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                ["finetune", "--config", str(config_path)]
            )
            with mock.patch(
                "flowfm.training.supervised.run_nf_ton_finetune",
                return_value={"status": "completed", "run_dir": "run"},
            ) as runner:
                result = dispatch(args)
            self.assertEqual(result["status"], "completed")
            runner.assert_called_once()

    def test_nf_ton_runner_rejects_another_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "finetune.yaml"
            config_path.write_text(
                _config_text(Path(directory) / "results").replace(
                    "NF-ToN-IoT-v2",
                    "NF-UNSW-NB15-v2",
                    1,
                ),
                encoding="utf-8",
            )
            config = load_config(config_path, expected_workflow="finetune")
            with self.assertRaisesRegex(ValueError, "NF-ToN"):
                run_nf_ton_finetune(
                    config,
                    repo_dir=PROJECT_ROOT,
                    config_path=config_path,
                )

    def test_real_smoke_parser_requires_frozen_inputs_and_device(self):
        parser = build_smoke_parser()
        arguments = parser.parse_args(
            [
                "--split-dir",
                "/root/autodl-tmp/split",
                "--preprocessor-dir",
                "/root/autodl-tmp/preprocessor",
                "--data-manifest",
                "/root/autodl-tmp/manifest.json",
                "--device",
                "cuda",
            ]
        )
        self.assertEqual(arguments.device, "cuda")
        self.assertEqual(arguments.train_samples, 4)
        self.assertEqual(arguments.validation_samples, 4)


if __name__ == "__main__":
    unittest.main()
