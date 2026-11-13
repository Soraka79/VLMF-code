import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from flowfm.data.feature_keys import FEATURE_KEY_NAMESPACE
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))
REQUIRED_REPORTS = (
    "conflict_summary.json",
    "conflict_groups.parquet",
    "conflict_label_pairs.csv",
    "conflict_examples.csv",
    "data_quality_report.md",
)


def sha1_file(path):
    digest = hashlib.sha1()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class AuditConflictsCliTests(unittest.TestCase):
    def setUp(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-audit-test-",
            dir=DATA_TMP_ROOT,
        )
        self.root = Path(self.temporary.name)
        self.raw_csv = self.root / "synthetic.csv"
        self.manifest = self.root / "source.json"
        self.config = self.root / "audit.yaml"
        self._write_raw_csv()
        self._write_manifest()
        self._write_config()

    def tearDown(self):
        self.temporary.cleanup()

    def _base_row(self):
        row = {
            name: str(index + 1)
            for index, name in enumerate(
                NF_UNSW_NB15_V2_SCHEMA.model_feature_names
            )
        }
        row.update(
            {
                "IPV4_SRC_ADDR": "192.0.2.10",
                "IPV4_DST_ADDR": "198.51.100.20",
                "Label": "1",
                "Attack": "Exploits",
            }
        )
        return row

    def _write_raw_csv(self):
        first = self._base_row()
        second = dict(first)
        second.update(
            {
                "IPV4_SRC_ADDR": "203.0.113.11",
                "IPV4_DST_ADDR": "203.0.113.12",
                "Attack": "Fuzzers",
            }
        )
        third = dict(first)
        third[NF_UNSW_NB15_V2_SCHEMA.model_feature_names[0]] = "2"
        third.update(
            {
                "IPV4_SRC_ADDR": "192.0.2.30",
                "IPV4_DST_ADDR": "198.51.100.40",
                "Label": "0",
                "Attack": "Benign",
            }
        )
        fieldnames = [
            *NF_UNSW_NB15_V2_SCHEMA.raw_feature_names,
            "Label",
            "Attack",
        ]
        with self.raw_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows((first, second, third))

    def _write_manifest(self):
        payload = {
            "dataset_id": "NF-UNSW-NB15-v2",
            "archive": {
                "sha256": "a" * 64,
            },
            "bag": {
                "csv": {
                    "relative_path": self.raw_csv.name,
                    "rows_excluding_header": 3,
                    "columns": 45,
                    "sha1": sha1_file(self.raw_csv),
                }
            },
        }
        self.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_config(self):
        payload = {
            "dataset_id": "NF-UNSW-NB15-v2",
            "raw_csv": str(self.raw_csv),
            "source_manifest": str(self.manifest),
            "source_file_id": self.raw_csv.name,
            "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
            "feature_key_namespace": FEATURE_KEY_NAMESPACE,
            "binary_label_column": "Label",
            "native_label_column": "Attack",
            "coarse_label_mapping_version": "test-coarse-v1",
            "coarse_label_map": {
                "Benign": "benign",
                "Exploits": "exploit",
                "Fuzzers": "exploit",
            },
            "sensitive_hash_namespace": "test-sensitive-v1",
            "max_examples": 5,
        }
        self.config.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def _rewrite_as_nf_ton(self):
        first = self._base_row()
        first["Attack"] = "injection"
        second = dict(first)
        second.update(
            {
                "IPV4_SRC_ADDR": "203.0.113.11",
                "IPV4_DST_ADDR": "203.0.113.12",
                "Attack": "xss",
            }
        )
        third = dict(first)
        third[NF_TON_IOT_V2_SCHEMA.model_feature_names[0]] = "2"
        third.update(
            {
                "IPV4_SRC_ADDR": "192.0.2.30",
                "IPV4_DST_ADDR": "198.51.100.40",
                "Label": "0",
                "Attack": "Benign",
            }
        )
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
            writer.writerows((first, second, third))

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["dataset_id"] = NF_TON_IOT_V2_SCHEMA.dataset_id
        manifest["bag"]["csv"]["sha1"] = sha1_file(self.raw_csv)
        self.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["dataset_id"] = NF_TON_IOT_V2_SCHEMA.dataset_id
        config["schema_version"] = NF_TON_IOT_V2_SCHEMA.version
        config["coarse_label_map"] = {
            "Benign": "benign",
            "injection": "exploit",
            "xss": "exploit",
        }
        self.config.write_text(
            yaml.safe_dump(config, sort_keys=True),
            encoding="utf-8",
        )

    def _run(self, output):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "flowfm",
                "audit-conflicts",
                "--config",
                str(self.config),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    def test_cli_generates_required_reports_and_expected_counts(self):
        output = self.root / "report-one"
        result = self._run(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        for filename in REQUIRED_REPORTS:
            self.assertTrue((output / filename).is_file(), filename)
        self.assertFalse((output / ".incomplete").exists())

        summary = json.loads((output / "conflict_summary.json").read_text())
        self.assertEqual(summary["dataset_id"], "NF-UNSW-NB15-v2")
        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["unique_feature_keys"], 2)
        self.assertEqual(summary["duplicate_excess_rows"], 1)
        self.assertEqual(summary["conflict_feature_keys"], 1)
        self.assertEqual(summary["conflict_rows"], 2)
        self.assertEqual(summary["raw_sha1_before"], sha1_file(self.raw_csv))
        self.assertEqual(summary["raw_sha1_after"], sha1_file(self.raw_csv))
        self.assertEqual(summary["schema_hash"], NF_UNSW_NB15_V2_SCHEMA.schema_hash)
        self.assertIn("git_commit", summary)
        self.assertIn("config_hash", summary)
        self.assertEqual(summary["raw_archive_sha256_manifest"], "a" * 64)
        self.assertEqual(summary["pyarrow_version"], "17.0.0")
        self.assertIn("python_version", summary)
        self.assertIn("pyyaml_version", summary)

        table = pq.read_table(output / "conflict_groups.parquet")
        self.assertEqual(table.num_rows, 2)
        self.assertEqual(
            set(table.column("native_label").to_pylist()),
            {"Exploits", "Fuzzers"},
        )
        self.assertEqual(set(table.column("conflict_group_size").to_pylist()), {2})
        self.assertEqual(set(table.column("conflict_label_count").to_pylist()), {2})

    def test_reports_are_privacy_safe_and_deterministic(self):
        first_output = self.root / "report-first"
        second_output = self.root / "report-second"
        first = self._run(first_output)
        second = self._run(second_output)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)

        for filename in (
            "conflict_summary.json",
            "conflict_label_pairs.csv",
            "conflict_examples.csv",
            "data_quality_report.md",
        ):
            self.assertEqual(
                (first_output / filename).read_bytes(),
                (second_output / filename).read_bytes(),
                filename,
            )
        self.assertEqual(
            pq.read_table(first_output / "conflict_groups.parquet").to_pylist(),
            pq.read_table(second_output / "conflict_groups.parquet").to_pylist(),
        )

        sensitive_values = (
            "192.0.2.10",
            "198.51.100.20",
            "203.0.113.11",
            "203.0.113.12",
        )
        for report in first_output.iterdir():
            if report.suffix == ".parquet":
                serialized = json.dumps(pq.read_table(report).to_pylist())
            else:
                serialized = report.read_text(encoding="utf-8")
            for value in sensitive_values:
                self.assertNotIn(value, serialized, report.name)

    def test_nf_ton_registered_schema_generates_the_same_audit_contract(self):
        self._rewrite_as_nf_ton()
        output = self.root / "report-nf-ton"
        result = self._run(output)
        self.assertEqual(result.returncode, 0, result.stderr)

        summary = json.loads((output / "conflict_summary.json").read_text())
        self.assertEqual(summary["dataset_id"], NF_TON_IOT_V2_SCHEMA.dataset_id)
        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["unique_feature_keys"], 2)
        self.assertEqual(summary["duplicate_excess_rows"], 1)
        self.assertEqual(summary["conflict_feature_keys"], 1)
        self.assertEqual(summary["conflict_rows"], 2)
        self.assertEqual(summary["schema_hash"], NF_TON_IOT_V2_SCHEMA.schema_hash)
        self.assertIn(
            NF_TON_IOT_V2_SCHEMA.dataset_id,
            summary["time_distribution"]["reason"],
        )

    def test_dataset_and_schema_version_must_select_one_registered_schema(self):
        from flowfm.data.audit import load_audit_config

        self._rewrite_as_nf_ton()
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["schema_version"] = NF_UNSW_NB15_V2_SCHEMA.version
        self.config.write_text(
            yaml.safe_dump(config, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "unregistered dataset Schema"):
            load_audit_config(self.config)

    def test_source_manifest_dataset_must_match_the_resolved_schema(self):
        self._rewrite_as_nf_ton()
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["dataset_id"] = NF_UNSW_NB15_V2_SCHEMA.dataset_id
        self.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output = self.root / "report-manifest-mismatch"
        result = self._run(output)
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "source manifest dataset_id does not match audit config",
            result.stderr,
        )

    def test_output_outside_autodl_data_disk_is_rejected_without_side_effects(self):
        output = Path("/tmp") / f"flowfm-forbidden-{os.getpid()}"
        result = self._run(output)

        self.assertEqual(result.returncode, 2)
        self.assertIn("/root/autodl-tmp", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
