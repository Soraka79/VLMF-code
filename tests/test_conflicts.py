import json
import unittest

from flowfm.data.conflicts import analyze_conflicts
from flowfm.data.splits import deterministic_group_split


def row(
    sample_id,
    feature_key,
    binary_label,
    coarse_label,
    native_label,
    *,
    host_pair_hash="host-pair-a",
    five_tuple_hash="five-tuple-a",
):
    return {
        "sample_id": sample_id,
        "feature_key": feature_key,
        "binary_label": binary_label,
        "coarse_label": coarse_label,
        "native_label": native_label,
        "source_file": "raw.csv",
        "source_row_id": sample_id.removeprefix("s"),
        "host_pair_hash": host_pair_hash,
        "five_tuple_hash": five_tuple_hash,
        "IPV4_SRC_ADDR": "192.0.2.10",
        "IPV4_DST_ADDR": "198.51.100.20",
    }


def synthetic_rows():
    return [
        row("s1", "k1", "0", "benign", "Benign"),
        row("s2", "k1", "0", "benign", "Benign"),
        row("s3", "k2", "1", "exploit", "Exploits"),
        row("s4", "k2", "1", "exploit", "Exploits"),
        row("s5", "k2", "1", "exploit", "Fuzzers"),
        row("s6", "k3", "0", "benign", "Benign"),
        row("s7", "k3", "1", "exploit", "Exploits"),
        row("s8", "k4", "1", "scan", "Reconnaissance"),
    ]


class ConflictAggregationTests(unittest.TestCase):
    def test_summary_counts_and_group_distribution_are_exact(self):
        audit = analyze_conflicts(synthetic_rows())

        self.assertEqual(audit.summary["total_rows"], 8)
        self.assertEqual(audit.summary["unique_feature_keys"], 4)
        self.assertEqual(audit.summary["duplicate_excess_rows"], 4)
        self.assertEqual(audit.summary["conflict_feature_keys"], 2)
        self.assertEqual(audit.summary["conflict_rows"], 5)
        self.assertEqual(audit.summary["conflict_row_ratio"], 0.625)
        self.assertEqual(
            audit.summary["group_size_distribution"],
            {"1": 1, "2": 2, "3": 1},
        )

    def test_conflict_groups_preserve_traceability_and_hashed_metadata(self):
        audit = analyze_conflicts(synthetic_rows())
        group = audit.group_by_key("k2")

        self.assertTrue(group.conflict_flag)
        self.assertEqual(group.group_size, 3)
        self.assertEqual(group.conflict_label_count, 2)
        self.assertEqual(group.conflict_fields, ("native_label",))
        self.assertEqual(group.sample_ids, ("s3", "s4", "s5"))
        self.assertEqual(
            group.source_refs,
            (
                ("raw.csv", "3"),
                ("raw.csv", "4"),
                ("raw.csv", "5"),
            ),
        )
        self.assertEqual(group.host_pair_hashes, ("host-pair-a",))
        self.assertEqual(group.five_tuple_hashes, ("five-tuple-a",))

        serialized = json.dumps(audit.to_dict(), sort_keys=True)
        self.assertNotIn("192.0.2.10", serialized)
        self.assertNotIn("198.51.100.20", serialized)

    def test_weighted_deterministic_classifier_upper_bounds_are_reported(self):
        audit = analyze_conflicts(synthetic_rows())
        bounds = audit.summary["deterministic_classifier_upper_bounds"]

        self.assertEqual(
            bounds["binary_label"],
            {
                "majority_correct_rows": 7,
                "irreducible_error_rows": 1,
                "accuracy_upper_bound": 0.875,
                "conflict_only_majority_correct_rows": 1,
                "conflict_only_rows": 2,
                "conflict_only_accuracy_upper_bound": 0.5,
            },
        )
        self.assertEqual(bounds["coarse_label"]["accuracy_upper_bound"], 0.875)
        self.assertEqual(bounds["native_label"]["accuracy_upper_bound"], 0.75)
        self.assertEqual(
            bounds["native_label"]["conflict_only_accuracy_upper_bound"],
            0.6,
        )

    def test_label_pair_counts_distinguish_groups_and_cross_label_pairs(self):
        audit = analyze_conflicts(synthetic_rows())
        pairs = audit.summary["label_pair_counts"]

        self.assertEqual(
            pairs["binary_label"]["0 | 1"],
            {"group_count": 1, "cross_label_row_pairs": 1},
        )
        self.assertEqual(
            pairs["native_label"]["Exploits | Fuzzers"],
            {"group_count": 1, "cross_label_row_pairs": 2},
        )
        self.assertEqual(
            pairs["native_label"]["Benign | Exploits"],
            {"group_count": 1, "cross_label_row_pairs": 1},
        )
        self.assertEqual(audit.summary["benign_attack_conflict_groups"], 1)
        self.assertEqual(audit.summary["attack_family_conflict_groups"], 1)


class LogicalViewTests(unittest.TestCase):
    def test_every_conflicting_group_is_wholly_quarantined(self):
        audit = analyze_conflicts(synthetic_rows())

        self.assertEqual(
            audit.view_manifest("raw_full")["sample_ids"],
            tuple(f"s{index}" for index in range(1, 9)),
        )
        self.assertEqual(
            audit.view_manifest("conflict_free_core")["sample_ids"],
            ("s1", "s2", "s8"),
        )
        self.assertEqual(
            audit.view_manifest("conflict_quarantine")["sample_ids"],
            ("s3", "s4", "s5", "s6", "s7"),
        )

        core_keys = set(audit.view_manifest("conflict_free_core")["feature_keys"])
        quarantine_keys = set(
            audit.view_manifest("conflict_quarantine")["feature_keys"]
        )
        self.assertTrue(core_keys.isdisjoint(quarantine_keys))
        self.assertEqual(quarantine_keys, {"k2", "k3"})
        for feature_key in core_keys:
            self.assertFalse(audit.group_by_key(feature_key).conflict_flag)

    def test_raw_grouped_split_records_never_split_a_feature_key(self):
        audit = analyze_conflicts(synthetic_rows())
        split = deterministic_group_split(
            audit.split_records("raw_full"),
            ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            seed=7,
            schema_version="schema-v1",
            source_hash="source-hash",
        )

        for group in audit.groups:
            assigned = {
                split.group_assignments[split.sample_groups[sample_id]]
                for sample_id in group.sample_ids
            }
            self.assertEqual(len(assigned), 1)

    def test_input_order_does_not_change_audit_or_views(self):
        forward = analyze_conflicts(synthetic_rows()).to_dict()
        reverse = analyze_conflicts(reversed(synthetic_rows())).to_dict()
        self.assertEqual(forward, reverse)


class ValidationTests(unittest.TestCase):
    def test_duplicate_sample_identity_is_rejected(self):
        records = synthetic_rows()
        duplicate = dict(records[0])
        duplicate["feature_key"] = "other-key"
        records.append(duplicate)

        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            analyze_conflicts(records)

    def test_missing_required_field_is_rejected(self):
        record_without_source = synthetic_rows()[0]
        record_without_source.pop("source_row_id")

        with self.assertRaisesRegex(ValueError, "source_row_id"):
            analyze_conflicts([record_without_source])


if __name__ == "__main__":
    unittest.main()
