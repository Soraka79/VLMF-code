import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from flowfm.cli import build_parser
from flowfm.data.audit import run_conflict_audit
from flowfm.data.feature_keys import FEATURE_KEY_NAMESPACE
from flowfm.data.labels import NF_CSE_LABEL_SPACE
from flowfm.data.nf_cse_confirmatory import (
    SUPPORTED_COARSE_LABELS,
    _availability_audit,
    publish_nf_cse_confirmatory_split,
)
from flowfm.data.nf_cse_dataset import load_nf_cse_development_samples
from flowfm.data.nf_cse_few_shot import (
    SELECTION_MANIFEST_FILENAME,
    SELECTION_RECORDS_FILENAME,
    SELECTION_SUMMARY_FILENAME,
    publish_nf_cse_few_shot_split,
)
from flowfm.data.nf_cse_preprocessing import (
    SELECTION_FEATURES_FILENAME,
    publish_nf_cse_train_only_preprocessor,
)
from flowfm.data.nf_cse_split import publish_nf_cse_development_split
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
    NF_CSE_CIC_IDS2018_V2_SCHEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))


def _hash_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class NFCSEConfirmatorySplitTests(unittest.TestCase):
    COPY_COUNT = 9

    def setUp(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-nf-cse-confirmatory-", dir=DATA_TMP_ROOT
        )
        self.root = Path(self.temporary.name)
        self.history = self.root / "history"
        self.history.mkdir()
        self.raw = self.root / "NF-CSE-CIC-IDS2018-v2.csv"
        self.source = self.root / "source.json"
        self.audit_config = self.root / "audit.yaml"
        self.audit_report = self.root / "audit-report"
        self.old_config = self.root / "old-split.yaml"
        self.old_split = (
            self.history
            / "data"
            / "splits"
            / NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id
            / "old-100k"
        )
        self.confirm_config = self.root / "confirm.yaml"
        self._write_raw()
        self._write_source()
        self._write_audit_config()
        run_conflict_audit(
            self.audit_config, self.audit_report, repo_dir=PROJECT_ROOT
        )
        self._write_old_config()
        publish_nf_cse_development_split(
            self.old_config,
            self.old_split,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        (self.old_split / "REJECTED.json").write_text(
            json.dumps(
                {
                    "status": "rejected_for_training_and_experiment_use",
                    "reason": "legacy rejected split",
                    "required_action": "republish",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_confirm_config()

    def tearDown(self):
        self.temporary.cleanup()

    def _row(self, native, copy_index, label_index):
        row = {
            name: str(index + 1 + copy_index * 1000 + label_index * 10)
            for index, name in enumerate(
                NF_CSE_CIC_IDS2018_V2_SCHEMA.model_feature_names
            )
        }
        row.update(
            {
                "IPV4_SRC_ADDR": f"10.{copy_index}.{label_index}.1",
                "IPV4_DST_ADDR": f"10.{copy_index}.{label_index}.2",
                "L4_SRC_PORT": str(2000 + copy_index * 100 + label_index),
                "L4_DST_PORT": "443",
                "PROTOCOL": "6",
                "Label": "0" if native == "Benign" else "1",
                "Attack": native,
            }
        )
        return row

    def _write_raw(self):
        rows = [
            self._row(native, copy_index, label_index)
            for copy_index in range(self.COPY_COUNT)
            for label_index, native in enumerate(NF_CSE_LABEL_SPACE.native_labels)
        ]
        with self.raw.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    *NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_feature_names,
                    "Label",
                    "Attack",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_source(self):
        native = {
            label: self.COPY_COUNT for label in NF_CSE_LABEL_SPACE.native_labels
        }
        self.source.write_text(
            json.dumps(
                {
                    "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
                    "archive": {"sha256": "a" * 64},
                    "bag": {
                        "csv": {
                            "relative_path": self.raw.name,
                            "rows_excluding_header": (
                                self.COPY_COUNT
                                * len(NF_CSE_LABEL_SPACE.native_labels)
                            ),
                            "columns": 45,
                            "sha1": _hash_file(self.raw, "sha1"),
                        }
                    },
                    "label_counts": {
                        "binary": {
                            "benign": self.COPY_COUNT,
                            "attack": self.COPY_COUNT
                            * (len(NF_CSE_LABEL_SPACE.native_labels) - 1),
                        },
                        "native_attack": native,
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
                    "dataset_id": NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
                    "raw_csv": str(self.raw),
                    "source_manifest": str(self.source),
                    "source_file_id": self.raw.name,
                    "schema_version": NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
                    "feature_key_namespace": FEATURE_KEY_NAMESPACE,
                    "binary_label_column": "Label",
                    "native_label_column": "Attack",
                    "coarse_label_mapping_version": "test-coarse-v1",
                    "coarse_label_map": dict(NF_CSE_LABEL_SPACE.native_to_coarse),
                    "sensitive_hash_namespace": "test-sensitive-v1",
                    "max_examples": 2,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_old_config(self):
        self.old_config.write_text(
            yaml.safe_dump(
                {
                    "audit_config": str(self.audit_config),
                    "audit_report_dir": str(self.audit_report),
                    "view": "conflict_free_core",
                    "max_samples": 15,
                    "seed": 42,
                    "selection_policy": "native_proportional",
                    "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_confirm_config(self):
        self.confirm_config.write_text(
            yaml.safe_dump(
                {
                    "audit_config": str(self.audit_config),
                    "audit_report_dir": str(self.audit_report),
                    "exclusion_split_dirs": [str(self.old_split)],
                    "history_root": str(self.history),
                    "max_samples": 90,
                    "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                    "seed": 20260725,
                    "selection_policy": (
                        "native_proportional_group_safe_after_exclusion"
                    ),
                    "view": "conflict_free_core",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_publication_is_deterministic_and_zero_overlap(self):
        first = self.root / "confirm-first"
        second = self.root / "confirm-second"
        first_result = publish_nf_cse_confirmatory_split(
            self.confirm_config,
            first,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        second_result = publish_nf_cse_confirmatory_split(
            self.confirm_config,
            second,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        self.assertEqual(first_result, second_result)
        self.assertTrue(first_result["verified_zero_overlap"])
        for filename in (
            "config.snapshot.json",
            "data_manifest.json",
            "selected_records.jsonl",
            "split_manifest.json",
            "split_summary.json",
            "overlap_audit.json",
            "sealed_test.json",
        ):
            self.assertEqual(
                (first / filename).read_bytes(),
                (second / filename).read_bytes(),
            )
        summary = json.loads((first / "split_summary.json").read_text())
        overlap = json.loads((first / "overlap_audit.json").read_text())
        self.assertEqual(
            tuple(summary["supported_coarse_labels"]), SUPPORTED_COARSE_LABELS
        )
        self.assertTrue(overlap["history_inventory"]["complete"])
        self.assertEqual(
            overlap["selected_vs_history_overlap_counts"],
            {"feature_keys": 0, "group_ids": 0, "sample_ids": 0, "source_refs": 0},
        )

    def test_history_inventory_rejects_an_unexcluded_split(self):
        unexcluded = self.old_split.parent / "unexcluded"
        unexcluded.mkdir()
        (unexcluded / "selected_records.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not excluded"):
            publish_nf_cse_confirmatory_split(
                self.confirm_config,
                self.root / "blocked",
                repo_dir=PROJECT_ROOT,
                data_root=self.root,
            )

    def test_cli_exposes_confirmatory_commands(self):
        parser = build_parser()
        confirm = parser.parse_args(
            [
                "prepare-nf-cse-confirmatory-split",
                "--config",
                str(self.confirm_config),
                "--output",
                str(self.root / "out"),
            ]
        )
        few_shot = parser.parse_args(
            [
                "derive-nf-cse-few-shot",
                "--source-split-dir",
                str(self.root),
                "--train-percent",
                "1",
                "--seed",
                "17",
                "--output",
                str(self.root / "few"),
            ]
        )
        self.assertEqual(confirm.command, "prepare-nf-cse-confirmatory-split")
        self.assertEqual(few_shot.command, "derive-nf-cse-few-shot")

    def test_availability_uses_conflict_free_view_before_history(self):
        availability = _availability_audit(
            {"native-a": 10, "native-b": 3, "native-c": 2},
            {"native-a": 8, "native-b": 2},
            {"native-a": 7},
        )
        self.assertEqual(
            availability,
            {
                "conflict_quarantine_exhausted_labels": ["native-c"],
                "historically_exhausted_labels": ["native-b"],
                "available_labels": ["native-a"],
            },
        )

    def test_selection_view_never_opens_sealed_test_or_raw_csv(self):
        confirm = self.root / "confirm-selection-boundary"
        few_shot = self.root / "few-shot-selection-boundary"
        preprocessor = self.root / "preprocessor-selection-boundary"
        publish_nf_cse_confirmatory_split(
            self.confirm_config,
            confirm,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        publish_nf_cse_few_shot_split(
            confirm,
            few_shot,
            train_percent=50,
            seed=17,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )
        publish_nf_cse_train_only_preprocessor(
            few_shot,
            preprocessor,
            repo_dir=PROJECT_ROOT,
            data_root=self.root,
        )

        full_manifest = json.loads(
            (few_shot / "split_manifest.json").read_text(encoding="utf-8")
        )
        selection_manifest = json.loads(
            (few_shot / SELECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(selection_manifest["splits"]["test"], [])
        self.assertFalse(
            set(full_manifest["splits"]["test"])
            & set(selection_manifest["sample_groups"])
        )
        selection_summary = json.loads(
            (few_shot / SELECTION_SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(set(selection_summary["split_counts"]), {"train", "val"})
        self.assertFalse(selection_summary["test_labels_present"])
        self.assertFalse(selection_summary["test_statistics_present"])
        selection_records = [
            json.loads(line)
            for line in (few_shot / SELECTION_RECORDS_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            {record["sample_id"] for record in selection_records},
            set(selection_manifest["splits"]["train"])
            | set(selection_manifest["splits"]["val"]),
        )

        cached_features = [
            json.loads(line)
            for line in (preprocessor / SELECTION_FEATURES_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        forbidden = {
            "Attack",
            "Label",
            "IPV4_SRC_ADDR",
            "IPV4_DST_ADDR",
            "sample_id",
            "flow_id",
            "dataset_id",
        }
        self.assertTrue(cached_features)
        self.assertTrue(
            all(
                not forbidden.intersection(row["features"])
                and set(row) == {"features", "source_row_id"}
                for row in cached_features
            )
        )
        preprocessor_selection_summary = json.loads(
            (preprocessor / SELECTION_SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        self.assertFalse(
            preprocessor_selection_summary[
                "selection_features_join_key_is_model_input"
            ]
        )
        self.assertFalse(
            preprocessor_selection_summary[
                "selection_features_model_payload_label_fields_present"
            ]
        )
        self.assertFalse(
            preprocessor_selection_summary[
                "selection_features_model_payload_trace_fields_present"
            ]
        )

        for name in (
            "selected_records.jsonl",
            "split_manifest.json",
            "split_summary.json",
        ):
            (few_shot / name).write_text("sealed artifact opened\n", encoding="utf-8")
        self.raw.write_text("raw CSV opened\n", encoding="utf-8")
        dataset = load_nf_cse_development_samples(
            few_shot,
            preprocessor,
            data_root=self.root,
            split_names=("train", "val"),
        )
        self.assertEqual(
            dataset.sample_count,
            len(selection_manifest["splits"]["train"])
            + len(selection_manifest["splits"]["val"]),
        )


if __name__ == "__main__":
    unittest.main()
