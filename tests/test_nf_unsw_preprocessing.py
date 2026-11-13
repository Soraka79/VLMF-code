import json
import os
import tempfile
import unittest
from pathlib import Path

from flowfm.data.nf_unsw_preprocessing import (
    MISSING_CATEGORY_ID,
    UNKNOWN_CATEGORY_ID,
    fit_nf_unsw_preprocessor,
    load_nf_unsw_preprocessor,
    save_nf_unsw_preprocessor,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


DATA_TMP_ROOT = Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp"))


def raw_features(*, in_bytes, protocol, out_bytes="0"):
    row = {name: "0" for name in NF_UNSW_NB15_V2_SCHEMA.model_feature_names}
    row.update(
        {
            "IN_BYTES": in_bytes,
            "OUT_BYTES": out_bytes,
            "PROTOCOL": protocol,
            "Label": "ignored",
            "Attack": "ignored",
        }
    )
    return row


class NFUNSWTrainOnlyPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.train_rows = (
            raw_features(in_bytes="10", protocol="6", out_bytes=""),
            raw_features(in_bytes="20", protocol="17", out_bytes=""),
        )

    def test_fit_uses_only_training_rows_and_transform_does_not_refit(self):
        preprocessor = fit_nf_unsw_preprocessor(self.train_rows)
        before = preprocessor.to_dict()
        before_hash = preprocessor.preprocessor_hash

        encoded = preprocessor.transform(
            raw_features(in_bytes="1000", protocol="999", out_bytes="7")
        )

        self.assertEqual(preprocessor.to_dict(), before)
        self.assertEqual(preprocessor.preprocessor_hash, before_hash)
        in_bytes_index = encoded.feature_ids.index("IN_BYTES")
        protocol_index = encoded.feature_ids.index("PROTOCOL")
        out_bytes_index = encoded.feature_ids.index("OUT_BYTES")
        self.assertAlmostEqual(encoded.features[in_bytes_index], 99.0)
        self.assertEqual(encoded.features[protocol_index], UNKNOWN_CATEGORY_ID)
        self.assertEqual(encoded.category_tokens[protocol_index], "n:999")
        self.assertFalse(encoded.missing_mask[protocol_index])
        self.assertEqual(encoded.features[out_bytes_index], 0.0)
        self.assertIsNone(encoded.category_tokens[out_bytes_index])
        self.assertFalse(encoded.missing_mask[out_bytes_index])
        self.assertEqual(len(encoded.category_tokens), len(encoded.feature_ids))

        stats = preprocessor.numeric_statistics["IN_BYTES"]
        self.assertEqual(stats["observed_count"], 2)
        self.assertEqual(stats["missing_count"], 0)
        self.assertEqual(stats["min"], "10")
        self.assertEqual(stats["max"], "20")
        self.assertEqual(stats["mean"], "15")
        self.assertNotIn("n:999", preprocessor.categorical_vocabularies["PROTOCOL"])

    def test_missing_and_unknown_categories_have_fixed_distinct_ids(self):
        preprocessor = fit_nf_unsw_preprocessor(self.train_rows)
        known = preprocessor.transform(raw_features(in_bytes="15", protocol="6"))
        unknown = preprocessor.transform(raw_features(in_bytes="15", protocol="255"))
        missing = preprocessor.transform(raw_features(in_bytes="", protocol=""))
        protocol_index = known.feature_ids.index("PROTOCOL")
        in_bytes_index = known.feature_ids.index("IN_BYTES")

        self.assertEqual(MISSING_CATEGORY_ID, 0)
        self.assertEqual(UNKNOWN_CATEGORY_ID, 1)
        self.assertGreaterEqual(known.features[protocol_index], 2)
        self.assertEqual(known.category_tokens[protocol_index], "n:6")
        self.assertEqual(unknown.features[protocol_index], UNKNOWN_CATEGORY_ID)
        self.assertEqual(unknown.category_tokens[protocol_index], "n:255")
        self.assertEqual(missing.features[protocol_index], MISSING_CATEGORY_ID)
        self.assertIsNone(missing.category_tokens[protocol_index])
        self.assertTrue(missing.missing_mask[protocol_index])
        self.assertEqual(missing.features[in_bytes_index], 0.0)
        self.assertIsNone(missing.category_tokens[in_bytes_index])
        self.assertTrue(missing.missing_mask[in_bytes_index])
    def test_unseen_valid_port_preserves_canonical_token_for_stable_bucketing(self):
        preprocessor = fit_nf_unsw_preprocessor(self.train_rows)
        row = raw_features(in_bytes="15", protocol="6")
        row["L4_SRC_PORT"] = "54321"

        encoded = preprocessor.transform(row)

        port_index = encoded.feature_ids.index("L4_SRC_PORT")
        self.assertEqual(encoded.features[port_index], UNKNOWN_CATEGORY_ID)
        self.assertEqual(encoded.category_tokens[port_index], "n:54321")
        self.assertFalse(encoded.missing_mask[port_index])


    def test_fit_and_hash_are_independent_of_training_input_order(self):
        first = fit_nf_unsw_preprocessor(self.train_rows)
        reversed_fit = fit_nf_unsw_preprocessor(tuple(reversed(self.train_rows)))

        self.assertEqual(first.to_dict(), reversed_fit.to_dict())
        self.assertEqual(first.preprocessor_hash, reversed_fit.preprocessor_hash)
        self.assertEqual(
            first.transform(raw_features(in_bytes="15", protocol="6")),
            reversed_fit.transform(raw_features(in_bytes="15", protocol="6")),
        )

    def test_save_and_load_preserve_statistics_and_verify_hash(self):
        DATA_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="flowfm-preprocessor-",
            dir=DATA_TMP_ROOT,
        ) as directory:
            path = Path(directory) / "preprocessor.json"
            fitted = fit_nf_unsw_preprocessor(self.train_rows)

            save_nf_unsw_preprocessor(path, fitted)
            loaded = load_nf_unsw_preprocessor(path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["preprocessor_hash"], fitted.preprocessor_hash)
            self.assertEqual(loaded.to_dict(), fitted.to_dict())
            self.assertEqual(loaded.preprocessor_hash, fitted.preprocessor_hash)

            payload["preprocessor_hash"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preprocessor_hash"):
                load_nf_unsw_preprocessor(path)

    def test_rejects_empty_training_data(self):
        with self.assertRaisesRegex(ValueError, "training"):
            fit_nf_unsw_preprocessor(())


if __name__ == "__main__":
    unittest.main()
