import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from flowfm.data import formal_dataset as formal_data
from flowfm.data.formal_dataset import (
    LEGACY_NUMERIC_SCALING_POLICY,
    NUMERIC_SCALING_POLICY,
    FormalTrainOnlyPreprocessor,
    fit_formal_preprocessor,
    load_formal_preprocessor,
    save_formal_preprocessor,
)
from flowfm.data.formal_cohort import get_formal_dataset_spec


class FormalDatasetTests(unittest.TestCase):
    def _row(self):
        spec = get_formal_dataset_spec("nf_bot")
        return {
            feature.name: ("6" if feature.categorical else "1")
            for feature in spec.schema.model_features
        }

    def test_generic_preprocessor_is_train_fitted_and_dataset_bound(self):
        preprocessor = fit_formal_preprocessor(
            "nf_bot", (self._row(), self._row())
        )

        self.assertEqual(preprocessor.training_row_count, 2)
        self.assertEqual(preprocessor.dataset_id, "NF-BoT-IoT-v2")
        self.assertEqual(len(preprocessor.preprocessor_hash), 64)
        encoded = preprocessor.transform(self._row())
        self.assertEqual(len(encoded.features), 41)
        self.assertFalse(any(encoded.missing_mask))

    def test_numeric_transform_clips_unseen_values_to_train_range(self):
        low = self._row()
        high = self._row()
        low["SRC_TO_DST_SECOND_BYTES"] = "0"
        high["SRC_TO_DST_SECOND_BYTES"] = "1"
        preprocessor = fit_formal_preprocessor("nf_bot", (low, high))
        above = self._row()
        below = self._row()
        above["SRC_TO_DST_SECOND_BYTES"] = "1e1000"
        below["SRC_TO_DST_SECOND_BYTES"] = "-1e1000"
        feature_index = tuple(
            feature.name
            for feature in get_formal_dataset_spec("nf_bot").schema.model_features
        ).index("SRC_TO_DST_SECOND_BYTES")

        above_value = preprocessor.transform(above).features[feature_index]
        below_value = preprocessor.transform(below).features[feature_index]

        self.assertEqual(preprocessor.format_version, 2)
        self.assertEqual(
            preprocessor.numeric_scaling_policy, NUMERIC_SCALING_POLICY
        )
        self.assertEqual(above_value, 1.0)
        self.assertEqual(below_value, 0.0)
        self.assertTrue(math.isfinite(above_value))

    def test_v1_preprocessor_remains_loadable_with_original_hash(self):
        current = fit_formal_preprocessor("nf_bot", (self._row(), self._row()))
        legacy = FormalTrainOnlyPreprocessor(
            dataset_key=current.dataset_key,
            dataset_id=current.dataset_id,
            schema_version=current.schema_version,
            schema_hash=current.schema_hash,
            training_row_count=current.training_row_count,
            numeric_statistics=current.numeric_statistics,
            categorical_vocabularies=current.categorical_vocabularies,
            format_version=1,
            numeric_scaling_policy=LEGACY_NUMERIC_SCALING_POLICY,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessor.json"
            save_formal_preprocessor(path, legacy)
            loaded = load_formal_preprocessor(path)

        self.assertEqual(loaded.format_version, 1)
        self.assertEqual(loaded.preprocessor_hash, legacy.preprocessor_hash)
        self.assertNotIn("numeric_scaling_policy", loaded.to_dict())

    def test_sealed_reader_accepts_dataset_specific_restart_status(self):
        spec = get_formal_dataset_spec("nf_unsw_restart")
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            cohort = data_root / "cohort"
            cohort.mkdir()
            records_path = cohort / "sealed_test_records.jsonl"
            manifest_path = cohort / "sealed_test_manifest.json"
            features_path = cohort / "sealed_test_features.parquet"
            records_path.write_text("{}\n", encoding="ascii")
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": spec.dataset_id,
                        "split": "test",
                        "split_seed": spec.split_seed,
                        "sample_ids": ["sample-1"],
                    }
                ),
                encoding="utf-8",
            )
            features_path.write_bytes(b"test parquet placeholder")

            def sha256(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            (cohort / "sealed_test.json").write_text(
                json.dumps(
                    {
                        "status": spec.seal_status,
                        "selection_access_allowed": False,
                        "opened_for_evaluation": False,
                        "dataset_id": spec.dataset_id,
                        "sealed_test_records_sha256": sha256(records_path),
                        "sealed_test_manifest_sha256": sha256(manifest_path),
                        "sealed_test_features_sha256": sha256(features_path),
                    }
                ),
                encoding="utf-8",
            )
            record = SimpleNamespace(sample_id="sample-1", group_id="group-1")
            with mock.patch.object(
                formal_data, "_load_records", return_value=(record,)
            ), mock.patch.object(
                formal_data,
                "_feature_rows",
                return_value=iter((("sample-1", {"feature": 1}),)),
            ):
                rows = formal_data.open_formal_sealed_test_rows(
                    "nf_unsw_restart", cohort, data_root=data_root
                )

        self.assertEqual(rows.sample_order, ("sample-1",))
        self.assertEqual(rows.group_ids, ("group-1",))


if __name__ == "__main__":
    unittest.main()
