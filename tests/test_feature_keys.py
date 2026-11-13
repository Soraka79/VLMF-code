import math
import unittest
from dataclasses import replace
from decimal import Decimal

from flowfm.data.feature_keys import (
    canonicalize_value,
    compute_feature_key,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


def base_record():
    record = {
        name: str(index + 1)
        for index, name in enumerate(NF_UNSW_NB15_V2_SCHEMA.model_feature_names)
    }
    record.update(
        {
            "IPV4_SRC_ADDR": "59.166.0.1",
            "IPV4_DST_ADDR": "149.171.126.1",
            "Label": "0",
            "Attack": "Benign",
            "source_file": "raw.csv",
            "source_row_id": "2",
        }
    )
    return record


class SchemaTests(unittest.TestCase):
    def test_schema_separates_raw_model_and_trace_features(self):
        schema = NF_UNSW_NB15_V2_SCHEMA
        self.assertEqual(schema.dataset_id, "NF-UNSW-NB15-v2")
        self.assertEqual(len(schema.raw_feature_names), 43)
        self.assertEqual(len(schema.model_feature_names), 41)
        self.assertEqual(
            schema.trace_feature_names,
            ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"),
        )
        self.assertNotIn("IPV4_SRC_ADDR", schema.model_feature_names)
        self.assertNotIn("IPV4_DST_ADDR", schema.model_feature_names)
        self.assertEqual(len(schema.categorical_mask), 41)
        self.assertTrue(any(schema.categorical_mask))
        self.assertTrue(any(not value for value in schema.categorical_mask))

    def test_schema_hash_is_stable_and_versioned(self):
        schema = NF_UNSW_NB15_V2_SCHEMA
        self.assertEqual(len(schema.schema_hash), 64)
        changed = replace(schema, version="nf-unsw-v2-model-v2")
        self.assertNotEqual(schema.schema_hash, changed.schema_hash)


class CanonicalValueTests(unittest.TestCase):
    def test_numeric_spellings_and_signed_zero_are_equivalent(self):
        equivalents = ("1", "1.0", 1, 1.0, Decimal("1.000"))
        tokens = {canonicalize_value(value, "numeric") for value in equivalents}
        self.assertEqual(tokens, {"n:1"})
        self.assertEqual(canonicalize_value("-0.000", "numeric"), "n:0")

    def test_missing_spellings_are_equivalent(self):
        self.assertEqual(canonicalize_value(None, "numeric"), "m:")
        self.assertEqual(canonicalize_value("", "numeric"), "m:")

    def test_non_finite_and_invalid_numeric_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, "nan", "abc", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    canonicalize_value(value, "numeric")

    def test_string_values_are_length_framed(self):
        self.assertEqual(canonicalize_value("ab", "string"), "s:2:ab")
        self.assertNotEqual(
            canonicalize_value("1", "string"),
            canonicalize_value("1", "numeric"),
        )


class FeatureKeyTests(unittest.TestCase):
    def test_labels_source_metadata_and_mapping_order_do_not_affect_key(self):
        first = base_record()
        second = dict(reversed(list(first.items())))
        second.update(
            {
                "Label": "1",
                "Attack": "Exploits",
                "source_file": "another.csv",
                "source_row_id": "999",
                "IPV4_SRC_ADDR": "10.0.0.1",
                "IPV4_DST_ADDR": "10.0.0.2",
            }
        )
        self.assertEqual(
            compute_feature_key(first, NF_UNSW_NB15_V2_SCHEMA),
            compute_feature_key(second, NF_UNSW_NB15_V2_SCHEMA),
        )

    def test_numeric_equivalence_and_feature_change_behavior(self):
        first = base_record()
        second = dict(first)
        feature = NF_UNSW_NB15_V2_SCHEMA.model_feature_names[0]
        first[feature] = "1"
        second[feature] = "1.0000"
        self.assertEqual(
            compute_feature_key(first, NF_UNSW_NB15_V2_SCHEMA),
            compute_feature_key(second, NF_UNSW_NB15_V2_SCHEMA),
        )
        second[feature] = "2"
        self.assertNotEqual(
            compute_feature_key(first, NF_UNSW_NB15_V2_SCHEMA),
            compute_feature_key(second, NF_UNSW_NB15_V2_SCHEMA),
        )

    def test_schema_version_separates_keys(self):
        record = base_record()
        changed = replace(NF_UNSW_NB15_V2_SCHEMA, version="other-version")
        self.assertNotEqual(
            compute_feature_key(record, NF_UNSW_NB15_V2_SCHEMA),
            compute_feature_key(record, changed),
        )


if __name__ == "__main__":
    unittest.main()
