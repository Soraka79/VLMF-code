import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from flowfm.data.audit import run_conflict_audit
from flowfm.data.feature_keys import FEATURE_KEY_NAMESPACE
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.nf_ton_dataset import load_nf_ton_development_samples
from flowfm.data.nf_ton_preprocessing import (
    MISSING_CATEGORY_ID,
    UNKNOWN_CATEGORY_ID,
    fit_nf_ton_preprocessor,
    load_nf_ton_preprocessor,
    publish_nf_ton_train_only_preprocessor,
)
from flowfm.data.nf_ton_split import (
    load_nf_ton_development_split_config,
    publish_nf_ton_development_split,
)
from flowfm.data.nf_ton_tokenization import build_nf_ton_tokenizer_metadata
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.splits import SplitManifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))
OUTPUT_FILES = (
    "config.snapshot.json",
    "data_manifest.json",
    "selected_records.jsonl",
    "split_manifest.json",
    "split_summary.json",
)
PREPROCESSOR_OUTPUT_FILES = (
    "config.snapshot.json",
    "preprocessor.json",
    "preprocessor_summary.json",
)


def hash_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class NFTONSplitPublicationTests(unittest.TestCase):
    def setUp(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-nf-ton-access-",
            dir=DATA_TMP_ROOT,
        )
        self.root = Path(self.temporary.name)
        self.raw_csv = self.root / "NF-ToN-IoT-v2.csv"
        self.source_manifest = self.root / "source.json"
        self.audit_config = self.root / "audit.yaml"
        self.audit_report = self.root / "audit-report"
        self.split_config = self.root / "split.yaml"
        self._write_raw_csv()
        self._write_source_manifest()
        self._write_audit_config()
        run_conflict_audit(
            self.audit_config,
            self.audit_report,
            repo_dir=PROJECT_ROOT,
        )
        self._write_split_config(max_samples=11)

    def tearDown(self):
        self.temporary.cleanup()

    def _base_row(
        self,
        *,
        source,
        destination,
        source_port,
        in_bytes,
        native,
        binary="1",
    ):
        row = {
            name: str(index + 1)
            for index, name in enumerate(
                NF_TON_IOT_V2_SCHEMA.model_feature_names
            )
        }
        row.update(
            {
                "IPV4_SRC_ADDR": source,
                "IPV4_DST_ADDR": destination,
                "L4_SRC_PORT": source_port,
                "L4_DST_PORT": "80",
                "PROTOCOL": "6",
                "IN_BYTES": in_bytes,
                "Label": binary,
                "Attack": native,
            }
        )
        return row

    def _write_raw_csv(self):
        labels = NF_TON_LABEL_SPACE.native_labels
        rows = []
        for index, native in enumerate(labels):
            rows.append(
                self._base_row(
                    source=f"10.0.{index}.1",
                    destination=f"10.0.{index}.2",
                    source_port=str(2000 + index),
                    in_bytes=str(100 + index),
                    native=native,
                    binary="0" if native == "Benign" else "1",
                )
            )
        duplicate = dict(rows[-1])
        duplicate["IPV4_SRC_ADDR"] = "192.0.2.10"
        duplicate["IPV4_DST_ADDR"] = "198.51.100.20"
        rows.append(duplicate)
        with self.raw_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    *NF_TON_IOT_V2_SCHEMA.raw_feature_names,
                    "Label",
                    "Attack",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_source_manifest(self):
        payload = {
            "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
            "archive": {"sha256": "a" * 64},
            "bag": {
                "csv": {
                    "relative_path": self.raw_csv.name,
                    "rows_excluding_header": 11,
                    "columns": 45,
                    "sha1": hash_file(self.raw_csv, "sha1"),
                }
            },
        }
        self.source_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_audit_config(self):
        payload = {
            "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
            "raw_csv": str(self.raw_csv),
            "source_manifest": str(self.source_manifest),
            "source_file_id": self.raw_csv.name,
            "schema_version": NF_TON_IOT_V2_SCHEMA.version,
            "feature_key_namespace": FEATURE_KEY_NAMESPACE,
            "binary_label_column": "Label",
            "native_label_column": "Attack",
            "coarse_label_mapping_version": "test-nf-ton-coarse-v1",
            "coarse_label_map": dict(NF_TON_LABEL_SPACE.native_to_coarse),
            "sensitive_hash_namespace": "test-nf-ton-sensitive-v1",
            "max_examples": 2,
        }
        self.audit_config.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def _write_split_config(
        self, *, max_samples, selection_policy=None, extra=None
    ):
        payload = {
            "audit_config": str(self.audit_config),
            "audit_report_dir": str(self.audit_report),
            "view": "conflict_free_core",
            "max_samples": max_samples,
            "seed": 42,
            "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        }
        if selection_policy is not None:
            payload["selection_policy"] = selection_policy
        if extra:
            payload.update(extra)
        self.split_config.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def test_loads_strict_nf_ton_development_config(self):
        config = load_nf_ton_development_split_config(self.split_config)
        self.assertEqual(config.audit_config_path, self.audit_config.resolve())
        self.assertEqual(config.audit_report_dir, self.audit_report.resolve())
        self.assertEqual(config.max_samples, 11)
        self.assertEqual(config.seed, 42)
        self.assertEqual(
            config.ratios,
            {"train": 0.6, "val": 0.2, "test": 0.2},
        )
        self._write_split_config(max_samples=100001)
        with self.assertRaisesRegex(ValueError, "100000"):
            load_nf_ton_development_split_config(self.split_config)

    def test_loads_native_proportional_pilot_config(self):
        self._write_split_config(
            max_samples=11,
            selection_policy="native_proportional",
        )
        config = load_nf_ton_development_split_config(self.split_config)
        self.assertEqual(config.selection_policy, "native_proportional")
        self.assertEqual(
            config.to_dict()["selection_policy"],
            "native_proportional",
        )
        self._write_split_config(max_samples=11, selection_policy="invalid")
        with self.assertRaisesRegex(ValueError, "selection_policy"):
            load_nf_ton_development_split_config(self.split_config)

    def test_publication_is_deterministic_group_safe_and_privacy_safe(self):
        first_output = self.root / "split-first"
        second_output = self.root / "split-second"
        first = publish_nf_ton_development_split(
            self.split_config,
            first_output,
            repo_dir=PROJECT_ROOT,
        )
        second = publish_nf_ton_development_split(
            self.split_config,
            second_output,
            repo_dir=PROJECT_ROOT,
        )
        self.assertEqual(first, second)
        for filename in OUTPUT_FILES:
            self.assertEqual(
                (first_output / filename).read_bytes(),
                (second_output / filename).read_bytes(),
                filename,
            )
        self.assertEqual(
            (first_output / "data_manifest.json").read_bytes(),
            self.source_manifest.read_bytes(),
        )
        self.assertFalse((first_output / ".incomplete").exists())

        records = [
            json.loads(line)
            for line in (first_output / "selected_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        manifest = SplitManifest.from_json(
            (first_output / "split_manifest.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (first_output / "split_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(records), 11)
        self.assertEqual(summary["dataset_id"], "NF-ToN-IoT-v2")
        self.assertEqual(summary["selected_rows"], 11)
        self.assertEqual(summary["schema_hash"], NF_TON_IOT_V2_SCHEMA.schema_hash)
        self.assertEqual(set(summary["native_labels"]), set(NF_TON_LABEL_SPACE.native_labels))
        sample_split = {
            sample_id: split_name
            for split_name, sample_ids in manifest.splits.items()
            for sample_id in sample_ids
        }
        group_splits = {}
        feature_splits = {}
        for record in records:
            split_name = sample_split[record["sample_id"]]
            group_splits.setdefault(record["group_id"], set()).add(split_name)
            feature_splits.setdefault(record["feature_key"], set()).add(split_name)
        self.assertTrue(all(len(values) == 1 for values in group_splits.values()))
        self.assertTrue(all(len(values) == 1 for values in feature_splits.values()))
        serialized = "".join(
            (first_output / filename).read_text(encoding="utf-8")
            for filename in OUTPUT_FILES
        )
        self.assertNotIn("192.0.2.10", serialized)
        self.assertNotIn("198.51.100.20", serialized)

    def test_cli_and_no_overwrite_contract(self):
        output = self.root / "split-cli"
        command = [
            sys.executable,
            "-m",
            "flowfm",
            "prepare-nf-ton-split",
            "--config",
            str(self.split_config),
            "--output",
            str(output),
        ]
        first = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        second = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(second.returncode, 2)
        self.assertIn("already exists", second.stderr)

    def test_train_only_preprocessor_dataset_and_tokenizer_contract(self):
        split_output = self.root / "split-for-preprocessor"
        publish_nf_ton_development_split(
            self.split_config,
            split_output,
            repo_dir=PROJECT_ROOT,
        )
        first_output = self.root / "preprocessor-first"
        second_output = self.root / "preprocessor-second"
        first = publish_nf_ton_train_only_preprocessor(
            split_output,
            first_output,
            repo_dir=PROJECT_ROOT,
        )
        second = publish_nf_ton_train_only_preprocessor(
            split_output,
            second_output,
            repo_dir=PROJECT_ROOT,
        )
        self.assertEqual(first, second)
        for filename in PREPROCESSOR_OUTPUT_FILES:
            self.assertEqual(
                (first_output / filename).read_bytes(),
                (second_output / filename).read_bytes(),
                filename,
            )
        self.assertFalse((first_output / ".incomplete").exists())

        preprocessor = load_nf_ton_preprocessor(
            first_output / "preprocessor.json"
        )
        manifest = SplitManifest.from_json(
            (split_output / "split_manifest.json").read_text()
        )
        selected = [
            json.loads(line)
            for line in (split_output / "selected_records.jsonl")
            .read_text()
            .splitlines()
        ]
        source_row_by_sample = {
            record["sample_id"]: record["source_row_id"]
            for record in selected
        }
        with self.raw_csv.open("r", encoding="utf-8", newline="") as stream:
            raw_by_row = {
                str(row_number): row
                for row_number, row in enumerate(csv.DictReader(stream), start=1)
            }
        train_rows = [
            raw_by_row[source_row_by_sample[sample_id]]
            for sample_id in manifest.splits["train"]
        ]
        expected_train_only = fit_nf_ton_preprocessor(train_rows)
        heldout_probe = dict(train_rows[0])
        heldout_probe["PROTOCOL"] = "999"
        heldout_probe["IN_BYTES"] = "999999999"
        leaked_all_selected = fit_nf_ton_preprocessor(
            (*train_rows, heldout_probe)
        )
        self.assertEqual(
            preprocessor.preprocessor_hash,
            expected_train_only.preprocessor_hash,
        )
        self.assertNotEqual(
            preprocessor.preprocessor_hash,
            leaked_all_selected.preprocessor_hash,
        )

        unseen = dict(train_rows[0])
        unseen["PROTOCOL"] = "999"
        protocol_index = NF_TON_IOT_V2_SCHEMA.model_feature_names.index(
            "PROTOCOL"
        )
        encoded_unseen = preprocessor.transform(unseen)
        self.assertEqual(
            encoded_unseen.features[protocol_index],
            UNKNOWN_CATEGORY_ID,
        )
        missing_numeric = dict(train_rows[0])
        missing_numeric["IN_BYTES"] = ""
        in_bytes_index = NF_TON_IOT_V2_SCHEMA.model_feature_names.index(
            "IN_BYTES"
        )
        encoded_missing = preprocessor.transform(missing_numeric)
        self.assertTrue(encoded_missing.missing_mask[in_bytes_index])
        self.assertEqual(encoded_missing.features[in_bytes_index], 0.0)
        missing_category = dict(train_rows[0])
        missing_category["PROTOCOL"] = ""
        encoded_missing_category = preprocessor.transform(missing_category)
        self.assertEqual(
            encoded_missing_category.features[protocol_index],
            MISSING_CATEGORY_ID,
        )

        summary = json.loads(
            (first_output / "preprocessor_summary.json").read_text()
        )
        category_statistics = summary["categorical_training_statistics"]
        self.assertEqual(len(category_statistics), 12)
        self.assertEqual(
            set(category_statistics),
            {
                spec.name
                for spec in NF_TON_IOT_V2_SCHEMA.model_features
                if spec.categorical
            },
        )
        for statistics in category_statistics.values():
            self.assertEqual(
                statistics["observed_count"] + statistics["missing_count"],
                preprocessor.training_row_count,
            )
            self.assertGreaterEqual(statistics["unique_count"], 0)

        dataset = load_nf_ton_development_samples(
            split_output,
            first_output,
        )
        self.assertEqual(dataset.sample_count, len(selected))
        for samples in dataset.splits.values():
            for sample in samples:
                self.assertEqual(
                    sample.feature_ids,
                    NF_TON_IOT_V2_SCHEMA.model_feature_names,
                )
                self.assertNotIn("192.0.2.10", repr(sample))
                self.assertNotIn("198.51.100.20", repr(sample))

        data_manifest = json.loads(
            (split_output / "data_manifest.json").read_text()
        )
        metadata = build_nf_ton_tokenizer_metadata(
            dataset,
            preprocessor,
            data_manifest=data_manifest,
            max_known_categories=2,
        )
        self.assertEqual(len(metadata.categorical_vocabularies), 11)
        self.assertNotIn("DNS_QUERY_ID", metadata.categorical_vocabularies)
        for feature_name, vocabulary in metadata.categorical_vocabularies.items():
            spec = SCHEMAFLOW_MAIN_REGISTRY.by_name[feature_name]
            self.assertEqual(vocabulary.feature_id, spec.feature_id)
            self.assertEqual(vocabulary.encode(None), MISSING_CATEGORY_ID)
            if spec.value_policy == "vocabulary":
                self.assertLessEqual(vocabulary.table_size, 4)
                self.assertEqual(
                    vocabulary.encode("never-observed-category"),
                    UNKNOWN_CATEGORY_ID,
                )
        self.assertEqual(
            metadata.categorical_vocabularies["L4_SRC_PORT"].table_size,
            1278,
        )
        default_metadata = build_nf_ton_tokenizer_metadata(
            dataset,
            preprocessor,
            data_manifest=data_manifest,
        )
        self.assertEqual(
            default_metadata.categorical_vocabularies[
                "PROTOCOL"
            ].max_known_categories,
            256,
        )
        self.assertEqual(
            default_metadata.categorical_vocabularies[
                "L7_PROTO"
            ].max_known_categories,
            512,
        )

    def test_train_only_preprocessor_cli_and_no_overwrite(self):
        split_output = self.root / "split-for-preprocessor-cli"
        publish_nf_ton_development_split(
            self.split_config,
            split_output,
            repo_dir=PROJECT_ROOT,
        )
        output = self.root / "preprocessor-cli"
        command = [
            sys.executable,
            "-m",
            "flowfm",
            "fit-nf-ton-preprocessor",
            "--split-dir",
            str(split_output),
            "--output",
            str(output),
        ]
        first = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        second = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(second.returncode, 2)
        self.assertIn("already exists", second.stderr)


if __name__ == "__main__":
    unittest.main()
