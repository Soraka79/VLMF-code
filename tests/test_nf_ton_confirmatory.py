import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from flowfm.data.audit import run_conflict_audit
from flowfm.data.feature_keys import FEATURE_KEY_NAMESPACE
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.nf_ton_confirmatory import (
    load_nf_ton_confirmatory_samples,
    publish_nf_ton_confirmatory_cohort,
)
from flowfm.data.nf_ton_preprocessing import (
    publish_nf_ton_train_only_preprocessor,
)
from flowfm.data.nf_ton_split import publish_nf_ton_development_split
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))


def _hash_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class NFTONConfirmatoryCohortTests(unittest.TestCase):
    def setUp(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-nf-ton-confirmatory-", dir=DATA_TMP_ROOT
        )
        self.root = Path(self.temporary.name)
        self.raw_csv = self.root / "NF-ToN-IoT-v2.csv"
        self.source_manifest = self.root / "source.json"
        self.audit_config = self.root / "audit.yaml"
        self.audit_report = self.root / "audit-report"
        self.split_config = self.root / "split.yaml"
        self.exclusion_split = self.root / "historical-split"
        self.preprocessor = self.root / "preprocessor"
        self.cohort_config = self.root / "confirmatory.yaml"
        self._write_raw()
        self._write_source_manifest()
        self._write_audit_config()
        run_conflict_audit(
            self.audit_config, self.audit_report, repo_dir=PROJECT_ROOT
        )
        self._write_split_config()
        publish_nf_ton_development_split(
            self.split_config,
            self.exclusion_split,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        publish_nf_ton_train_only_preprocessor(
            self.exclusion_split,
            self.preprocessor,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        (self.exclusion_split / "REJECTED.json").write_text(
            json.dumps(
                {
                    "status": "rejected",
                    "downstream_use_allowed": False,
                    "reason": "test defensive exclusion",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_cohort_config()

    def tearDown(self):
        self.temporary.cleanup()

    def _row(self, label, copy_index):
        label_index = NF_TON_LABEL_SPACE.native_labels.index(label)
        row = {
            name: str(feature_index + 1 + copy_index * 100 + label_index)
            for feature_index, name in enumerate(
                NF_TON_IOT_V2_SCHEMA.model_feature_names
            )
        }
        row.update(
            {
                "IPV4_SRC_ADDR": f"10.{copy_index}.{label_index}.1",
                "IPV4_DST_ADDR": f"10.{copy_index}.{label_index}.2",
                "L4_SRC_PORT": str(2000 + copy_index * 100 + label_index),
                "L4_DST_PORT": "80",
                "PROTOCOL": "6",
                "Label": "0" if label == "Benign" else "1",
                "Attack": label,
            }
        )
        return row

    def _write_raw(self):
        rows = [
            self._row(label, copy_index)
            for copy_index in range(3)
            for label in NF_TON_LABEL_SPACE.native_labels
        ]
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
        self.source_manifest.write_text(
            json.dumps(
                {
                    "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
                    "archive": {"sha256": "a" * 64},
                    "bag": {
                        "csv": {
                            "relative_path": self.raw_csv.name,
                            "rows_excluding_header": 30,
                            "columns": 45,
                            "sha1": _hash_file(self.raw_csv, "sha1"),
                        }
                    },
                    "label_counts": {
                        "binary": {"benign": 3, "attack": 27},
                        "native_attack": {
                            label: 3
                            for label in NF_TON_LABEL_SPACE.native_labels
                            if label != "Benign"
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_audit_config(self):
        self.audit_config.write_text(
            yaml.safe_dump(
                {
                    "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
                    "raw_csv": str(self.raw_csv),
                    "source_manifest": str(self.source_manifest),
                    "source_file_id": self.raw_csv.name,
                    "schema_version": NF_TON_IOT_V2_SCHEMA.version,
                    "feature_key_namespace": FEATURE_KEY_NAMESPACE,
                    "binary_label_column": "Label",
                    "native_label_column": "Attack",
                    "coarse_label_mapping_version": "test-coarse-v1",
                    "coarse_label_map": dict(
                        NF_TON_LABEL_SPACE.native_to_coarse
                    ),
                    "sensitive_hash_namespace": "test-sensitive-v1",
                    "max_examples": 2,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_split_config(self):
        self.split_config.write_text(
            yaml.safe_dump(
                {
                    "audit_config": str(self.audit_config),
                    "audit_report_dir": str(self.audit_report),
                    "view": "conflict_free_core",
                    "max_samples": 10,
                    "seed": 42,
                    "selection_policy": "native_proportional",
                    "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_cohort_config(self):
        self.cohort_config.write_text(
            yaml.safe_dump(
                {
                    "audit_config": str(self.audit_config),
                    "audit_report_dir": str(self.audit_report),
                    "exclusion_split_dirs": [str(self.exclusion_split)],
                    "view": "conflict_free_core",
                    "max_samples": 10,
                    "seed": 20260724,
                    "selection_policy": "native_proportional_group_safe",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_publication_is_deterministic_disjoint_and_loadable(self):
        first = self.root / "cohort-first"
        second = self.root / "cohort-second"
        first_summary = publish_nf_ton_confirmatory_cohort(
            self.cohort_config,
            first,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        second_summary = publish_nf_ton_confirmatory_cohort(
            self.cohort_config,
            second,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )

        self.assertTrue(first_summary["verified_zero_overlap"])
        self.assertEqual(first_summary["selected_rows"], 10)
        overlap = json.loads((first / "overlap_audit.json").read_text())
        self.assertEqual(
            overlap["exclusion_artifacts"][0]["artifact_status"],
            "rejected_but_defensively_excluded",
        )
        for filename in (
            "cohort_manifest.json",
            "config.snapshot.json",
            "data_manifest.json",
            "overlap_audit.json",
            "selected_records.jsonl",
        ):
            self.assertEqual(
                (first / filename).read_bytes(),
                (second / filename).read_bytes(),
            )
        old_records = {
            json.loads(line)["group_id"]
            for line in (self.exclusion_split / "selected_records.jsonl")
            .read_text()
            .splitlines()
        }
        new_records = [
            json.loads(line)
            for line in (first / "selected_records.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertTrue(
            old_records.isdisjoint(record["group_id"] for record in new_records)
        )
        loaded = load_nf_ton_confirmatory_samples(
            first,
            self.preprocessor,
            data_root=self.root,
        )
        self.assertEqual(len(loaded.samples), 10)
        self.assertEqual(len(loaded.group_ids), 10)
        self.assertEqual(
            {sample.native_label for sample in loaded.samples},
            set(NF_TON_LABEL_SPACE.native_labels),
        )

    def test_publication_refuses_overwrite(self):
        output = self.root / "cohort"
        publish_nf_ton_confirmatory_cohort(
            self.cohort_config,
            output,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            publish_nf_ton_confirmatory_cohort(
                self.cohort_config,
                output,
                repo_dir=PROJECT_ROOT,
                data_root=self.root,
            )

    def test_preflight_rejects_when_historical_split_exhausts_labels(self):
        all_split_config = self.root / "all-split.yaml"
        payload = yaml.safe_load(self.split_config.read_text())
        payload["max_samples"] = 30
        all_split_config.write_text(
            yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
        )
        all_split = self.root / "all-historical-split"
        publish_nf_ton_development_split(
            all_split_config,
            all_split,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        blocked_config = self.root / "blocked-confirmatory.yaml"
        cohort_payload = yaml.safe_load(self.cohort_config.read_text())
        cohort_payload["exclusion_split_dirs"] = [str(all_split)]
        blocked_config.write_text(
            yaml.safe_dump(cohort_payload, sort_keys=True), encoding="utf-8"
        )
        output = self.root / "blocked-cohort"

        with self.assertRaisesRegex(ValueError, "exhaust full-source rows"):
            publish_nf_ton_confirmatory_cohort(
                blocked_config,
                output,
                repo_dir=PROJECT_ROOT,
                data_root=self.root,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
