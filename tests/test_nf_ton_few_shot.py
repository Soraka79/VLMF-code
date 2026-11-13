import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from flowfm.cli import build_parser
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.netflow_v2_protocol import NetFlowV2IndexedRecord
from flowfm.data.nf_ton_few_shot import publish_nf_ton_few_shot_split
from flowfm.data.nf_ton_split import SPLIT_CONFIG_NAMESPACE, _hash_file
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.splits import SplitManifest
from flowfm.experiment.metadata import write_json_snapshot


class NFTONFewShotSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="flowfm-nf-ton-few-shot-"
        )
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.records = self._records()
        self.manifest = self._write_source_split()

    def tearDown(self):
        self.temporary.cleanup()

    def _record(self, split_name, group_index, label_index, source_row_id):
        native_label = NF_TON_LABEL_SPACE.native_labels[label_index]
        return NetFlowV2IndexedRecord(
            sample_id=(
                f"{split_name}-sample-{group_index:04d}-{label_index:02d}"
            ),
            feature_key=(
                f"{split_name}-feature-{group_index:04d}-{label_index:02d}"
            ),
            group_id=f"{split_name}-group-{group_index:04d}",
            binary_label="benign" if native_label == "Benign" else "attack",
            coarse_label=NF_TON_LABEL_SPACE.native_to_coarse[native_label],
            native_label=native_label,
            source_file="NF-ToN-IoT-v2.csv",
            source_row_id=str(source_row_id),
            host_pair_hash=f"host-{source_row_id}",
            five_tuple_hash=f"tuple-{source_row_id}",
        )

    def _records(self):
        records = []
        source_row_id = 1
        for group_index in range(100):
            for label_index in range(len(NF_TON_LABEL_SPACE.native_labels)):
                records.append(
                    self._record(
                        "train",
                        group_index,
                        label_index,
                        source_row_id,
                    )
                )
                source_row_id += 1
        for split_name in ("val", "test"):
            for label_index in range(len(NF_TON_LABEL_SPACE.native_labels)):
                records.append(
                    self._record(
                        split_name,
                        label_index,
                        label_index,
                        source_row_id,
                    )
                )
                source_row_id += 1
        return tuple(records)

    def _write_source_split(self):
        splits = {
            split_name: [
                record.sample_id
                for record in self.records
                if record.sample_id.startswith(f"{split_name}-")
            ]
            for split_name in ("train", "val", "test")
        }
        manifest = SplitManifest(
            splits=splits,
            ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42,
            schema_version=NF_TON_IOT_V2_SCHEMA.version,
            source_hash="source-hash",
            sample_groups={
                record.sample_id: record.group_id for record in self.records
            },
            group_assignments={
                record.group_id: split_name
                for split_name in ("train", "val", "test")
                for record in self.records
                if record.sample_id.startswith(f"{split_name}-")
            },
        )
        config = {"audit_config": str(self.root / "audit.yaml")}
        write_json_snapshot(self.source / "config.snapshot.json", config)
        write_json_snapshot(self.source / "data_manifest.json", {"source": "test"})
        (self.source / "selected_records.jsonl").write_text(
            "".join(
                canonical_json(asdict(record)) + "\n"
                for record in sorted(
                    self.records,
                    key=lambda value: value.sample_id,
                )
            ),
            encoding="utf-8",
        )
        (self.source / "split_manifest.json").write_text(
            manifest.to_json() + "\n",
            encoding="utf-8",
        )
        summary = {
            "dataset_id": NF_TON_IOT_V2_SCHEMA.dataset_id,
            "schema_version": NF_TON_IOT_V2_SCHEMA.version,
            "schema_hash": NF_TON_IOT_V2_SCHEMA.schema_hash,
            "label_space_hash": NF_TON_LABEL_SPACE.label_space_hash,
            "config_hash": stable_hash(config, SPLIT_CONFIG_NAMESPACE),
            "source_hash": manifest.source_hash,
            "seed": manifest.seed,
            "ratios": dict(manifest.ratios),
            "selected_records_sha256": _hash_file(
                self.source / "selected_records.jsonl",
                "sha256",
            ),
            "split_manifest_sha256": _hash_file(
                self.source / "split_manifest.json",
                "sha256",
            ),
            "source_manifest_sha256": _hash_file(
                self.source / "data_manifest.json",
                "sha256",
            ),
            "audit_config_hash": "audit-config-hash",
            "raw_csv_sha1": "raw-csv-sha1",
        }
        write_json_snapshot(self.source / "split_summary.json", summary)
        return manifest

    def test_derives_group_safe_one_percent_and_preserves_eval(self):
        output = self.root / "derived"
        result = publish_nf_ton_few_shot_split(
            self.source,
            output,
            train_percent=1,
            seed=42,
            repo_dir=self.root,
            data_root=self.root,
        )

        derived = SplitManifest.from_json(
            (output / "split_manifest.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (output / "split_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["target_train_rows"], 10)
        self.assertEqual(result["selected_train_rows"], 10)
        self.assertEqual(result["selected_train_groups"], 1)
        self.assertEqual(
            derived.splits["val"],
            self.manifest.splits["val"],
        )
        self.assertEqual(
            derived.splits["test"],
            self.manifest.splits["test"],
        )
        train_groups = {
            derived.sample_groups[sample_id]
            for sample_id in derived.splits["train"]
        }
        eval_groups = {
            derived.sample_groups[sample_id]
            for split_name in ("val", "test")
            for sample_id in derived.splits[split_name]
        }
        self.assertFalse(train_groups & eval_groups)
        self.assertEqual(
            set(summary["native_labels"]),
            set(NF_TON_LABEL_SPACE.native_labels),
        )
        self.assertFalse((output / ".incomplete").exists())

    def test_cli_accepts_few_shot_derivation_arguments(self):
        args = build_parser().parse_args(
            [
                "derive-nf-ton-few-shot",
                "--source-split-dir",
                str(self.source),
                "--train-percent",
                "1",
                "--output",
                str(self.root / "derived"),
            ]
        )
        self.assertEqual(args.command, "derive-nf-ton-few-shot")
        self.assertEqual(args.train_percent, 1)
        self.assertEqual(args.seed, 42)


if __name__ == "__main__":
    unittest.main()
