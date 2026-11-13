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
from flowfm.data.nf_unsw_preprocessing import (
    UNKNOWN_CATEGORY_ID,
    fit_nf_unsw_preprocessor,
    load_nf_unsw_preprocessor,
    publish_nf_unsw_train_only_preprocessor,
)
from flowfm.data.nf_unsw_split import (
    load_nf_unsw_development_split_config,
    publish_nf_unsw_development_split,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
from flowfm.data.splits import SplitManifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))
PREPROCESSOR_OUTPUT_FILES = (
    "config.snapshot.json",
    "preprocessor.json",
    "preprocessor_summary.json",
)
OUTPUT_FILES = (
    "config.snapshot.json",
    "selected_records.jsonl",
    "split_manifest.json",
    "split_summary.json",
)
COARSE_LABEL_MAP = {
    "Benign": "benign",
    "DoS": "dos",
    "Exploits": "exploit",
    "Fuzzers": "exploit",
    "Worms": "malware",
}


def hash_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class NFUNSWDevelopmentSplitPublicationTests(unittest.TestCase):
    def setUp(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-nf-unsw-split-",
            dir=DATA_TMP_ROOT,
        )
        self.root = Path(self.temporary.name)
        self.raw_csv = self.root / "synthetic.csv"
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
        self._write_split_config(max_samples=9)

    def tearDown(self):
        self.temporary.cleanup()

    def _base_row(
        self,
        *,
        source,
        destination,
        source_port="1000",
        destination_port="80",
        protocol="6",
        in_bytes="10",
        binary="1",
        native="Exploits",
    ):
        row = {
            name: str(index + 1)
            for index, name in enumerate(
                NF_UNSW_NB15_V2_SCHEMA.model_feature_names
            )
        }
        row.update(
            {
                "IPV4_SRC_ADDR": source,
                "IPV4_DST_ADDR": destination,
                "L4_SRC_PORT": source_port,
                "L4_DST_PORT": destination_port,
                "PROTOCOL": protocol,
                "IN_BYTES": in_bytes,
                "Label": binary,
                "Attack": native,
            }
        )
        return row

    def _write_raw_csv(self):
        conflict_first = self._base_row(
            source="192.0.2.10",
            destination="198.51.100.20",
            native="Exploits",
        )
        conflict_second = dict(conflict_first)
        conflict_second.update(
            {
                "IPV4_SRC_ADDR": "203.0.113.10",
                "IPV4_DST_ADDR": "203.0.113.20",
                "Attack": "Fuzzers",
            }
        )
        benign_first = self._base_row(
            source="10.0.0.1",
            destination="10.0.0.2",
            source_port="2000",
            in_bytes="20",
            binary="0",
            native="Benign",
        )
        benign_second = self._base_row(
            source="10.0.0.1",
            destination="10.0.0.2",
            source_port="2000",
            in_bytes="21",
            binary="0",
            native="Benign",
        )
        rows = (
            conflict_first,
            conflict_second,
            benign_first,
            benign_second,
            self._base_row(
                source="10.0.1.1",
                destination="10.0.1.2",
                source_port="3000",
                in_bytes="30",
                native="DoS",
            ),
            self._base_row(
                source="10.0.2.1",
                destination="10.0.2.2",
                source_port="4000",
                in_bytes="40",
                native="Exploits",
            ),
            self._base_row(
                source="10.0.3.1",
                destination="10.0.3.2",
                source_port="5000",
                in_bytes="50",
                native="Worms",
            ),
            self._base_row(
                source="10.0.4.1",
                destination="10.0.4.2",
                source_port="6000",
                destination_port="81",
                in_bytes="60",
                binary="0",
                native="Benign",
            ),
            self._base_row(
                source="10.0.5.1",
                destination="10.0.5.2",
                source_port="7000",
                destination_port="82",
                in_bytes="70",
                binary="0",
                native="Benign",
            ),
            self._base_row(
                source="10.0.6.1",
                destination="10.0.6.2",
                source_port="8000",
                destination_port="83",
                in_bytes="80",
                binary="0",
                native="Benign",
            ),
            self._base_row(
                source="10.0.7.1",
                destination="10.0.7.2",
                source_port="9000",
                destination_port="84",
                in_bytes="90",
                binary="0",
                native="Benign",
            ),
        )
        fieldnames = [
            *NF_UNSW_NB15_V2_SCHEMA.raw_feature_names,
            "Label",
            "Attack",
        ]
        with self.raw_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_source_manifest(self):
        payload = {
            "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
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
            "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
            "raw_csv": str(self.raw_csv),
            "source_manifest": str(self.source_manifest),
            "source_file_id": self.raw_csv.name,
            "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
            "feature_key_namespace": FEATURE_KEY_NAMESPACE,
            "binary_label_column": "Label",
            "native_label_column": "Attack",
            "coarse_label_mapping_version": "test-coarse-v1",
            "coarse_label_map": COARSE_LABEL_MAP,
            "sensitive_hash_namespace": "test-sensitive-v1",
            "max_examples": 2,
        }
        self.audit_config.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def _write_split_config(self, *, max_samples, extra=None):
        payload = {
            "audit_config": str(self.audit_config),
            "audit_report_dir": str(self.audit_report),
            "view": "conflict_free_core",
            "max_samples": max_samples,
            "seed": 11,
            "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        }
        if extra:
            payload.update(extra)
        self.split_config.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def test_loads_strict_config_by_referencing_the_existing_audit_config(self):
        config = load_nf_unsw_development_split_config(self.split_config)

        self.assertEqual(config.audit_config_path, self.audit_config.resolve())
        self.assertEqual(config.audit_report_dir, self.audit_report.resolve())
        self.assertEqual(config.view, "conflict_free_core")
        self.assertEqual(config.max_samples, 9)
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.ratios, {"train": 0.6, "val": 0.2, "test": 0.2})
        self.assertRegex(config.config_hash, r"^[0-9a-f]{64}$")

        self._write_split_config(max_samples=9, extra={"unexpected": True})
        with self.assertRaisesRegex(ValueError, "split config fields"):
            load_nf_unsw_development_split_config(self.split_config)

    def test_rejects_values_outside_the_development_protocol(self):
        invalid_cases = (
            ({"max_samples": 0}, "max_samples"),
            ({"max_samples": 100001}, "100000"),
            ({"seed": True}, "seed"),
            ({"view": "raw_full"}, "conflict_free_core"),
            ({"ratios": {"train": 0.8, "val": 0.2, "test": 0.2}}, "sum"),
        )
        for changes, message in invalid_cases:
            with self.subTest(changes=changes):
                self._write_split_config(max_samples=9, extra=changes)
                with self.assertRaisesRegex(ValueError, message):
                    load_nf_unsw_development_split_config(self.split_config)

    def test_publishes_deterministic_group_safe_privacy_safe_artifacts(self):
        first_output = self.root / "split-first"
        second_output = self.root / "split-second"

        first = publish_nf_unsw_development_split(
            self.split_config,
            first_output,
            repo_dir=PROJECT_ROOT,
        )
        second = publish_nf_unsw_development_split(
            self.split_config,
            second_output,
            repo_dir=PROJECT_ROOT,
        )

        self.assertEqual(first, second)
        for filename in OUTPUT_FILES:
            self.assertTrue((first_output / filename).is_file(), filename)
            self.assertEqual(
                (first_output / filename).read_bytes(),
                (second_output / filename).read_bytes(),
                filename,
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
        selected_ids = {record["sample_id"] for record in records}
        assigned_ids = {
            sample_id
            for sample_ids in manifest.splits.values()
            for sample_id in sample_ids
        }
        self.assertEqual(selected_ids, assigned_ids)
        self.assertEqual(len(records), 9)
        self.assertEqual(summary["selected_rows"], 9)
        self.assertEqual(summary["selected_groups"], 8)
        self.assertEqual(summary["view"], "conflict_free_core")
        self.assertEqual(summary["raw_csv_sha1"], hash_file(self.raw_csv, "sha1"))
        for field in (
            "config_hash",
            "audit_config_hash",
            "audit_summary_sha256",
            "source_manifest_sha256",
            "schema_hash",
            "selected_records_sha256",
            "split_manifest_sha256",
        ):
            self.assertRegex(summary[field], r"^[0-9a-f]{64}$", field)

        conflict_keys = set(
            json.loads(
                (self.audit_report / "logical_views.json").read_text()
            )["conflict_quarantine"]["feature_keys"]
        )
        self.assertTrue(conflict_keys.isdisjoint(record["feature_key"] for record in records))
        sample_split = {
            sample_id: split_name
            for split_name, sample_ids in manifest.splits.items()
            for sample_id in sample_ids
        }
        feature_splits = {}
        for record in records:
            feature_splits.setdefault(record["feature_key"], set()).add(
                sample_split[record["sample_id"]]
            )
        self.assertTrue(all(len(splits) == 1 for splits in feature_splits.values()))

        serialized = "".join(
            (first_output / filename).read_text(encoding="utf-8")
            for filename in OUTPUT_FILES
        )
        for address in (
            "192.0.2.10",
            "198.51.100.20",
            "203.0.113.10",
            "203.0.113.20",
            "10.0.0.1",
            "10.0.0.2",
        ):
            self.assertNotIn(address, serialized)

    def test_cli_publishes_the_same_split_contract(self):
        output = self.root / "split-cli"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flowfm",
                "prepare-nf-unsw-split",
                "--config",
                str(self.split_config),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["selected_rows"], 9)
        self.assertEqual(summary["selected_groups"], 8)
        for filename in OUTPUT_FILES:
            self.assertTrue((output / filename).is_file(), filename)

    def test_failed_publication_keeps_incomplete_marker(self):
        self._write_split_config(max_samples=1)
        output = self.root / "split-incomplete"

        with self.assertRaisesRegex(ValueError, "max_samples"):
            publish_nf_unsw_development_split(
                self.split_config,
                output,
                repo_dir=PROJECT_ROOT,
            )

        self.assertTrue(output.is_dir())
        self.assertTrue((output / ".incomplete").is_file())
        self.assertFalse((output / "split_manifest.json").exists())

    def test_train_only_preprocessor_publication_is_deterministic_and_bound_to_split(self):
        split_output = self.root / "split-for-preprocessor"
        publish_nf_unsw_development_split(
            self.split_config,
            split_output,
            repo_dir=PROJECT_ROOT,
        )
        first_output = self.root / "preprocessor-first"
        second_output = self.root / "preprocessor-second"

        first = publish_nf_unsw_train_only_preprocessor(
            split_output,
            first_output,
            repo_dir=PROJECT_ROOT,
        )
        second = publish_nf_unsw_train_only_preprocessor(
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

        preprocessor = load_nf_unsw_preprocessor(
            first_output / "preprocessor.json"
        )
        summary = json.loads(
            (first_output / "preprocessor_summary.json").read_text()
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
            record["sample_id"]: record["source_row_id"] for record in selected
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
        all_rows = [raw_by_row[record["source_row_id"]] for record in selected]
        expected_train_only = fit_nf_unsw_preprocessor(train_rows)
        leaked_all_selected = fit_nf_unsw_preprocessor(all_rows)

        self.assertEqual(preprocessor.preprocessor_hash, expected_train_only.preprocessor_hash)
        self.assertNotEqual(preprocessor.preprocessor_hash, leaked_all_selected.preprocessor_hash)
        self.assertEqual(preprocessor.training_row_count, len(manifest.splits["train"]))
        self.assertEqual(summary["fit_scope"], "train_only")
        self.assertEqual(summary["training_rows"], len(manifest.splits["train"]))
        self.assertEqual(summary["preprocessor_hash"], preprocessor.preprocessor_hash)
        split_summary = json.loads((split_output / "split_summary.json").read_text())
        self.assertEqual(
            summary["split_manifest_sha256"],
            split_summary["split_manifest_sha256"],
        )
        self.assertEqual(
            summary["selected_records_sha256"],
            split_summary["selected_records_sha256"],
        )

        unknown_observed = False
        for split_name in ("val", "test"):
            for sample_id in manifest.splits[split_name]:
                encoded = preprocessor.transform(
                    raw_by_row[source_row_by_sample[sample_id]]
                )
                unknown_observed = unknown_observed or any(
                    categorical and value == UNKNOWN_CATEGORY_ID
                    for categorical, value in zip(
                        encoded.categorical_mask,
                        encoded.features,
                    )
                )
        self.assertTrue(unknown_observed)

    def test_train_only_preprocessor_rejects_tampered_split_config_snapshot(self):
        split_output = self.root / "split-with-tampered-config"
        publish_nf_unsw_development_split(
            self.split_config,
            split_output,
            repo_dir=PROJECT_ROOT,
        )
        config_path = split_output / "config.snapshot.json"
        config_snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        config_snapshot["seed"] += 1
        config_path.write_text(
            json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output = self.root / "preprocessor-tampered-config"

        with self.assertRaisesRegex(ValueError, "split config hash"):
            publish_nf_unsw_train_only_preprocessor(
                split_output,
                output,
                repo_dir=PROJECT_ROOT,
            )
        self.assertFalse(output.exists())

    def test_train_only_preprocessor_rejects_changed_audit_config(self):
        split_output = self.root / "split-with-changed-audit-config"
        publish_nf_unsw_development_split(
            self.split_config,
            split_output,
            repo_dir=PROJECT_ROOT,
        )
        audit_payload = yaml.safe_load(self.audit_config.read_text(encoding="utf-8"))
        audit_payload["coarse_label_mapping_version"] = "changed-after-split"
        self.audit_config.write_text(
            yaml.safe_dump(audit_payload, sort_keys=True),
            encoding="utf-8",
        )
        output = self.root / "preprocessor-changed-audit-config"

        with self.assertRaisesRegex(ValueError, "audit config hash"):
            publish_nf_unsw_train_only_preprocessor(
                split_output,
                output,
                repo_dir=PROJECT_ROOT,
            )
        self.assertFalse(output.exists())

    def test_train_only_preprocessor_cli_publishes_artifacts(self):
        split_output = self.root / "split-for-preprocessor-cli"
        publish_nf_unsw_development_split(
            self.split_config,
            split_output,
            repo_dir=PROJECT_ROOT,
        )
        output = self.root / "preprocessor-cli"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flowfm",
                "fit-nf-unsw-preprocessor",
                "--split-dir",
                str(split_output),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["fit_scope"], "train_only")
        for filename in PREPROCESSOR_OUTPUT_FILES:
            self.assertTrue((output / filename).is_file(), filename)

    def test_output_outside_data_disk_is_rejected_without_side_effects(self):
        output = Path("/tmp") / f"flowfm-nf-unsw-split-forbidden-{os.getpid()}"

        with self.assertRaisesRegex(ValueError, "/root/autodl-tmp"):
            publish_nf_unsw_development_split(
                self.split_config,
                output,
                repo_dir=PROJECT_ROOT,
            )

        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
