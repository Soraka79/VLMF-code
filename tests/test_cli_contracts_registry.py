import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from flowfm.cli import build_parser, dispatch
from flowfm.data.contracts import FlowSample


class CliTests(unittest.TestCase):
    def test_all_phase_one_commands_parse_config_paths(self):
        parser = build_parser()

        for command in ("pretrain", "finetune", "evaluate", "ablate"):
            with self.subTest(command=command):
                args = parser.parse_args([command, "--config", "configs/run.json"])
                self.assertEqual(args.command, command)
                self.assertEqual(args.config, Path("configs/run.json"))

    def test_module_entrypoint_reports_parsed_request_for_all_commands(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for command in ("pretrain", "finetune", "evaluate", "ablate"):
                with self.subTest(command=command):
                    config_path = Path(directory) / f"{command}.yaml"
                    config_path.write_text(
                        f"""workflow: {command}
experiment:
  name: cli-test
  output_root: /tmp/cli-test
  seed: 1
schema:
  version: phase1-v1
runtime:
  device: cpu
""",
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "flowfm",
                            command,
                            "--config",
                            str(config_path),
                        ],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    summary = json.loads(result.stdout)
                    self.assertEqual(summary["command"], command)
                    self.assertEqual(summary["config"], str(config_path.resolve()))
                    self.assertEqual(summary["experiment"], "cli-test")
                    self.assertEqual(summary["schema_version"], "phase1-v1")
                    self.assertEqual(summary["device"], "cpu")


class FlowSampleTests(unittest.TestCase):
    def test_exposes_required_fields(self):
        field_names = {field.name for field in fields(FlowSample)}

        self.assertEqual(
            field_names,
            {
                "features",
                "feature_ids",
                "categorical_mask",
                "missing_mask",
                "category_tokens",
                "binary_label",
                "coarse_label",
                "native_label",
                "dataset_id",
                "sample_id",
                "feature_key",
                "conflict_flag",
                "conflict_group_size",
                "conflict_label_count",
                "source_file",
                "source_row_id",
            },
        )

    def test_accepts_equal_feature_aligned_lengths(self):
        sample = FlowSample(
            features=[1.5, 2.5],
            feature_ids=["duration", "bytes"],
            categorical_mask=[False, False],
            missing_mask=[False, True],
            category_tokens=[None, None],
            binary_label=1,
            coarse_label="attack",
            native_label="DoS",
            dataset_id="cicids2017",
            sample_id="sample-1",
            feature_key="feature-key-1",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="raw.csv",
            source_row_id="1",
        )

        self.assertEqual(len(sample.features), 2)
    def test_rejects_category_tokens_inconsistent_with_type_or_missingness(self):
        common = dict(
            features=[1],
            feature_ids=["PROTOCOL"],
            binary_label=1,
            coarse_label="attack",
            native_label="DoS",
            dataset_id="fixture",
            sample_id="sample-1",
            feature_key="feature-key-1",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="raw.csv",
            source_row_id="1",
        )
        invalid = (
            {
                "categorical_mask": [False],
                "missing_mask": [False],
                "category_tokens": ["n:6"],
            },
            {
                "categorical_mask": [True],
                "missing_mask": [False],
                "category_tokens": [None],
            },
            {
                "categorical_mask": [True],
                "missing_mask": [True],
                "category_tokens": ["n:6"],
            },
        )

        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "category_tokens"):
                    FlowSample(**common, **values)


    def test_rejects_mismatched_feature_aligned_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            FlowSample(
                features=[1.5, 2.5],
                feature_ids=["duration"],
                categorical_mask=[False, False],
                missing_mask=[False, True],
                category_tokens=[None, None],
                binary_label=1,
                coarse_label="attack",
                native_label="DoS",
                dataset_id="cicids2017",
                sample_id="sample-1",
                feature_key="feature-key-1",
                conflict_flag=False,
                conflict_group_size=1,
                conflict_label_count=1,
                source_file="raw.csv",
                source_row_id="1",
            )

    def test_rejects_empty_dataset_or_sample_ids(self):
        common = dict(
            features=[1.5],
            feature_ids=["duration"],
            categorical_mask=[False],
            missing_mask=[False],
            category_tokens=[None],
            binary_label=0,
            coarse_label="benign",
            native_label="BENIGN",
            feature_key="feature-key-1",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="raw.csv",
            source_row_id="1",
        )

        with self.assertRaisesRegex(ValueError, "dataset_id"):
            FlowSample(dataset_id="", sample_id="sample-1", **common)
        with self.assertRaisesRegex(ValueError, "sample_id"):
            FlowSample(dataset_id="cicids2017", sample_id="", **common)

    def test_rejects_empty_conflict_identity_and_source_fields(self):
        common = dict(
            features=[1.5],
            feature_ids=["duration"],
            categorical_mask=[False],
            missing_mask=[False],
            category_tokens=[None],
            binary_label=0,
            coarse_label="benign",
            native_label="BENIGN",
            dataset_id="cicids2017",
            sample_id="sample-1",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
        )

        for field in ("feature_key", "source_file", "source_row_id"):
            values = {
                "feature_key": "feature-key-1",
                "source_file": "raw.csv",
                "source_row_id": "1",
            }
            values[field] = ""
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    FlowSample(**common, **values)

    def test_rejects_invalid_conflict_metadata(self):
        common = dict(
            features=[1.5],
            feature_ids=["duration"],
            categorical_mask=[False],
            missing_mask=[False],
            category_tokens=[None],
            binary_label=1,
            coarse_label="attack",
            native_label="DoS",
            dataset_id="cicids2017",
            sample_id="sample-1",
            feature_key="feature-key-1",
            source_file="raw.csv",
            source_row_id="1",
        )

        invalid_cases = (
            {
                "conflict_flag": 1,
                "conflict_group_size": 1,
                "conflict_label_count": 1,
            },
            {
                "conflict_flag": False,
                "conflict_group_size": 0,
                "conflict_label_count": 1,
            },
            {
                "conflict_flag": False,
                "conflict_group_size": 1,
                "conflict_label_count": 0,
            },
            {
                "conflict_flag": True,
                "conflict_group_size": 2,
                "conflict_label_count": 3,
            },
        )
        for metadata in invalid_cases:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    FlowSample(**common, **metadata)


class LegacyRegistryTests(unittest.TestCase):
    def test_importing_registry_does_not_import_legacy_module(self):
        script = """
import sys
from flowfm.models.registry import MODEL_REGISTRY

assert "rescae_bigru_legacy" in MODEL_REGISTRY.names()
assert "shared_rescae_bigru" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_factory_imports_controlled_module_on_demand(self):
        script = """
import importlib.abc
import importlib.util
import json
import sys


class LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "shared_rescae_bigru":
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        class ResCAE_BiGRU:
            def __init__(self, marker):
                self.marker = marker

        module.ResCAE_BiGRU = ResCAE_BiGRU


sys.meta_path.insert(0, LegacyModuleFinder())
from flowfm.models.registry import MODEL_REGISTRY
assert "shared_rescae_bigru" not in sys.modules
model = MODEL_REGISTRY.create("rescae_bigru_legacy", "loaded-on-demand")
print(json.dumps({
    "legacy_loaded": "shared_rescae_bigru" in sys.modules,
    "marker": model.marker,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {"legacy_loaded": True, "marker": "loaded-on-demand"},
        )


if __name__ == "__main__":
    unittest.main()
