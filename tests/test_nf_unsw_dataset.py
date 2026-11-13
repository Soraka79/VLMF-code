import csv
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import yaml

from flowfm.data.audit import load_audit_config
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.nf_unsw_dataset import load_nf_unsw_development_samples
from flowfm.data.nf_unsw_preprocessing import (
    TRAINING_SAMPLE_IDS_NAMESPACE,
    fit_nf_unsw_preprocessor,
    save_nf_unsw_preprocessor,
)
from flowfm.data.nf_unsw_protocol import index_nf_unsw_rows
from flowfm.data.nf_unsw_split import SPLIT_CONFIG_NAMESPACE
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import write_json_snapshot


DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))
COARSE_LABEL_MAP = {
    "Benign": "benign",
    "DoS": "dos",
    "Worms": "malware",
}


def hash_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class NFUNSWDevelopmentDatasetTests(unittest.TestCase):
    def setUp(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-nf-unsw-dataset-",
            dir=DATA_TMP_ROOT,
        )
        self.root = Path(self.temporary.name)
        self.raw_csv = self.root / "synthetic.csv"
        self.audit_config_path = self.root / "audit.yaml"
        self.source_manifest = self.root / "source.json"
        self.split_dir = self.root / "split"
        self.preprocessor_dir = self.root / "preprocessor"
        self.rows = self._write_raw_csv()
        self._write_audit_config()
        self._publish_split_and_preprocessor()

    def tearDown(self):
        self.temporary.cleanup()

    def _row(self, *, source, destination, source_port, in_bytes, binary, native):
        row = {
            name: str(index + 1)
            for index, name in enumerate(NF_UNSW_NB15_V2_SCHEMA.model_feature_names)
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
        rows = (
            self._row(
                source="192.0.2.1",
                destination="198.51.100.1",
                source_port="1000",
                in_bytes="10",
                binary="0",
                native="Benign",
            ),
            self._row(
                source="192.0.2.2",
                destination="198.51.100.2",
                source_port="2000",
                in_bytes="20",
                binary="1",
                native="DoS",
            ),
            self._row(
                source="192.0.2.3",
                destination="198.51.100.3",
                source_port="3000",
                in_bytes="30",
                binary="1",
                native="Worms",
            ),
        )
        with self.raw_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    *NF_UNSW_NB15_V2_SCHEMA.raw_feature_names,
                    "Label",
                    "Attack",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def _write_audit_config(self):
        write_json_snapshot(
            self.source_manifest,
            {
                "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
                "archive": {"sha256": "a" * 64},
                "bag": {
                    "csv": {
                        "sha1": hash_file(self.raw_csv, "sha1"),
                        "rows_excluding_header": len(self.rows),
                        "columns": 45,
                    }
                },
            },
        )
        payload = {
            "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
            "raw_csv": str(self.raw_csv),
            "source_manifest": str(self.source_manifest),
            "source_file_id": self.raw_csv.name,
            "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
            "feature_key_namespace": "flowfm-feature-key-v1",
            "binary_label_column": "Label",
            "native_label_column": "Attack",
            "coarse_label_mapping_version": "test-coarse-v1",
            "coarse_label_map": COARSE_LABEL_MAP,
            "sensitive_hash_namespace": "test-sensitive-v1",
            "max_examples": 0,
        }
        self.audit_config_path.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def _publish_split_and_preprocessor(self):
        audit_config = load_audit_config(self.audit_config_path)
        indexed = index_nf_unsw_rows(
            ((str(index), row) for index, row in enumerate(self.rows, start=1)),
            conflict_feature_keys=set(),
            source_file_id=audit_config.source_file_id,
            coarse_label_map=audit_config.coarse_label_map,
            sensitive_hash_namespace=audit_config.sensitive_hash_namespace,
        )
        by_source_row = {record.source_row_id: record for record in indexed}
        train_record = by_source_row["1"]
        val_record = by_source_row["2"]
        test_record = by_source_row["3"]
        ratios = {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3}
        manifest = SplitManifest(
            splits={
                "train": [train_record.sample_id],
                "val": [val_record.sample_id],
                "test": [test_record.sample_id],
            },
            ratios=ratios,
            seed=7,
            schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
            source_hash="synthetic-source-hash",
            sample_groups={record.sample_id: record.group_id for record in indexed},
            group_assignments={
                train_record.group_id: "train",
                val_record.group_id: "val",
                test_record.group_id: "test",
            },
        )
        split_config = {
            "audit_config": str(self.audit_config_path),
            "audit_report_dir": str(self.root / "audit-report"),
            "max_samples": 3,
            "ratios": ratios,
            "seed": 7,
            "view": "conflict_free_core",
        }
        self.split_dir.mkdir()
        write_json_snapshot(self.split_dir / "config.snapshot.json", split_config)
        selected_path = self.split_dir / "selected_records.jsonl"
        selected_path.write_text(
            "".join(canonical_json(asdict(record)) + "\n" for record in indexed),
            encoding="utf-8",
        )
        manifest_path = self.split_dir / "split_manifest.json"
        manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
        split_summary = {
            "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
            "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
            "schema_hash": NF_UNSW_NB15_V2_SCHEMA.schema_hash,
            "config_hash": stable_hash(split_config, SPLIT_CONFIG_NAMESPACE),
            "audit_config_hash": audit_config.config_hash,
            "raw_csv_sha1": hash_file(self.raw_csv, "sha1"),
            "selected_records_sha256": hash_file(selected_path),
            "split_manifest_sha256": hash_file(manifest_path),
            "source_hash": manifest.source_hash,
            "seed": manifest.seed,
            "ratios": ratios,
        }
        write_json_snapshot(self.split_dir / "split_summary.json", split_summary)

        preprocessor = fit_nf_unsw_preprocessor((self.rows[0],))
        self.preprocessor_dir.mkdir()
        write_json_snapshot(
            self.preprocessor_dir / "config.snapshot.json",
            {
                "fit_scope": "train_only",
                "source_split_dir": str(self.split_dir.resolve()),
            },
        )
        preprocessor_path = save_nf_unsw_preprocessor(
            self.preprocessor_dir / "preprocessor.json",
            preprocessor,
        )
        write_json_snapshot(
            self.preprocessor_dir / "preprocessor_summary.json",
            {
                "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
                "fit_scope": "train_only",
                "training_rows": 1,
                "validation_rows": 1,
                "test_rows": 1,
                "training_sample_ids_hash": stable_hash(
                    list(manifest.splits["train"]),
                    TRAINING_SAMPLE_IDS_NAMESPACE,
                ),
                "preprocessor_hash": preprocessor.preprocessor_hash,
                "preprocessor_file_sha256": hash_file(preprocessor_path),
                "split_config_hash": split_summary["config_hash"],
                "selected_records_sha256": split_summary["selected_records_sha256"],
                "split_manifest_sha256": split_summary["split_manifest_sha256"],
                "audit_config_hash": audit_config.config_hash,
                "raw_csv_sha1": split_summary["raw_csv_sha1"],
                "schema_version": NF_UNSW_NB15_V2_SCHEMA.version,
                "schema_hash": NF_UNSW_NB15_V2_SCHEMA.schema_hash,
            },
        )

    def test_loads_aligned_privacy_safe_flow_samples(self):
        dataset = load_nf_unsw_development_samples(
            self.split_dir,
            self.preprocessor_dir,
        )

        self.assertEqual(
            {name: len(samples) for name, samples in dataset.splits.items()},
            {"train": 1, "val": 1, "test": 1},
        )
        samples = [sample for split in dataset.splits.values() for sample in split]
        self.assertEqual(len({sample.sample_id for sample in samples}), 3)
        for sample in samples:
            self.assertEqual(len(sample.features), 41)
            self.assertEqual(sample.feature_ids, NF_UNSW_NB15_V2_SCHEMA.model_feature_names)
            self.assertEqual(len(sample.categorical_mask), 41)
            self.assertEqual(len(sample.category_tokens), 41)
            self.assertEqual(len(sample.missing_mask), 41)
            self.assertFalse(sample.conflict_flag)
            self.assertEqual(sample.conflict_label_count, 1)
            self.assertNotIn("192.0.2.", repr(sample))
            self.assertNotIn("198.51.100.", repr(sample))
        self.assertEqual(dataset.splits["train"][0].binary_label, "benign")
        self.assertEqual(dataset.splits["val"][0].coarse_label, "dos")
        self.assertEqual(dataset.splits["test"][0].native_label, "Worms")
        train = dataset.splits["train"][0]
        val = dataset.splits["val"][0]
        source_port_index = val.feature_ids.index("L4_SRC_PORT")
        self.assertEqual(train.category_tokens[source_port_index], "n:1000")
        self.assertEqual(val.features[source_port_index], 1)
        self.assertEqual(val.category_tokens[source_port_index], "n:2000")
        self.assertEqual(
            dataset.training_sample_ids_hash,
            stable_hash([train.sample_id], TRAINING_SAMPLE_IDS_NAMESPACE),
        )
        self.assertFalse(any(dataset.splits["val"][0].missing_mask))
        self.assertTrue(
            any(
                categorical and value == 1
                for categorical, value in zip(
                    dataset.splits["val"][0].categorical_mask,
                    dataset.splits["val"][0].features,
                )
            )
        )

    def test_rejects_preprocessor_bound_to_a_different_split(self):
        summary_path = self.preprocessor_dir / "preprocessor_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["split_manifest_sha256"] = "0" * 64
        write_json_snapshot(summary_path, summary)

        with self.assertRaisesRegex(ValueError, "split manifest hash"):
            load_nf_unsw_development_samples(
                self.split_dir,
                self.preprocessor_dir,
            )

    def test_rejects_raw_data_changed_after_publication(self):
        with self.raw_csv.open("a", encoding="utf-8") as stream:
            stream.write("tampered\n")

        with self.assertRaisesRegex(ValueError, "raw CSV SHA1"):
            load_nf_unsw_development_samples(
                self.split_dir,
                self.preprocessor_dir,
            )


if __name__ == "__main__":
    unittest.main()
