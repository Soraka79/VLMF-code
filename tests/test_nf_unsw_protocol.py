import json
import unittest
from dataclasses import asdict

from flowfm.data.feature_keys import compute_feature_key
from flowfm.data.nf_unsw_protocol import (
    NFUNSWIndexedRecord,
    index_nf_unsw_rows,
    select_nf_unsw_development_records,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


COARSE_LABEL_MAP = {
    "Benign": "benign",
    "Analysis": "exploit",
    "Backdoor": "backdoor",
    "DoS": "dos",
    "Exploits": "exploit",
    "Fuzzers": "exploit",
    "Generic": "exploit",
    "Reconnaissance": "scan",
    "Shellcode": "exploit",
    "Worms": "malware",
}


def raw_row(
    *,
    source="10.0.0.1",
    destination="10.0.0.2",
    source_port="1000",
    destination_port="80",
    protocol="6",
    in_bytes="10",
    binary="1",
    native="Exploits",
):
    row = {name: "0" for name in NF_UNSW_NB15_V2_SCHEMA.raw_feature_names}
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


def indexed_record(sample_id, group_id, native_label):
    return NFUNSWIndexedRecord(
        sample_id=sample_id,
        feature_key=f"feature-{sample_id}",
        group_id=group_id,
        binary_label="benign" if native_label == "Benign" else "attack",
        coarse_label=COARSE_LABEL_MAP[native_label],
        native_label=native_label,
        source_file="synthetic.csv",
        source_row_id=sample_id,
        host_pair_hash=f"host-{sample_id}",
        five_tuple_hash=f"tuple-{sample_id}",
    )


class NFUNSWIndexTests(unittest.TestCase):
    def index(self, rows, conflict_feature_keys=()):
        return index_nf_unsw_rows(
            rows,
            conflict_feature_keys=set(conflict_feature_keys),
            source_file_id="synthetic.csv",
            coarse_label_map=COARSE_LABEL_MAP,
            sensitive_hash_namespace="test-sensitive-v1",
        )

    def test_feature_key_and_five_tuple_form_transitive_leakage_groups(self):
        first = raw_row(source="10.0.0.1", destination="10.0.0.2", in_bytes="10")
        same_features = raw_row(
            source="10.0.0.3",
            destination="10.0.0.4",
            in_bytes="10",
        )
        same_tuple_as_second = raw_row(
            source="10.0.0.3",
            destination="10.0.0.4",
            in_bytes="99",
        )
        unrelated = raw_row(
            source="10.0.0.8",
            destination="10.0.0.9",
            source_port="2000",
            in_bytes="77",
        )

        indexed = self.index(
            [
                ("1", first),
                ("2", same_features),
                ("3", same_tuple_as_second),
                ("4", unrelated),
            ]
        )
        by_row = {record.source_row_id: record for record in indexed}

        self.assertEqual(by_row["1"].feature_key, by_row["2"].feature_key)
        self.assertNotEqual(by_row["2"].feature_key, by_row["3"].feature_key)
        self.assertEqual(by_row["2"].five_tuple_hash, by_row["3"].five_tuple_hash)
        self.assertEqual(by_row["1"].group_id, by_row["2"].group_id)
        self.assertEqual(by_row["2"].group_id, by_row["3"].group_id)
        self.assertNotEqual(by_row["3"].group_id, by_row["4"].group_id)

    def test_conflict_feature_key_is_excluded_as_an_entire_group(self):
        conflict = raw_row(source="10.1.0.1", destination="10.1.0.2")
        same_features = raw_row(source="10.1.0.3", destination="10.1.0.4")
        retained = raw_row(
            source="10.1.0.1",
            destination="10.1.0.2",
            in_bytes="999",
        )
        conflict_key = compute_feature_key(conflict, NF_UNSW_NB15_V2_SCHEMA)

        indexed = self.index(
            [("10", conflict), ("11", same_features), ("12", retained)],
            conflict_feature_keys={conflict_key},
        )

        self.assertEqual([record.source_row_id for record in indexed], ["12"])

    def test_labels_are_normalized_and_raw_addresses_are_not_published(self):
        benign = raw_row(
            source="192.0.2.10",
            destination="198.51.100.20",
            binary="0",
            native="Benign",
        )
        record = self.index([("7", benign)])[0]
        payload = asdict(record)
        serialized = json.dumps(payload, sort_keys=True)

        self.assertEqual(record.binary_label, "benign")
        self.assertEqual(record.coarse_label, "benign")
        self.assertEqual(record.native_label, "Benign")
        self.assertNotIn("192.0.2.10", serialized)
        self.assertNotIn("198.51.100.20", serialized)
        self.assertRegex(record.host_pair_hash, r"^[0-9a-f]{64}$")
        self.assertRegex(record.five_tuple_hash, r"^[0-9a-f]{64}$")

    def test_fixed_source_ids_make_output_independent_of_input_order(self):
        rows = [
            ("21", raw_row(source="10.2.0.1", destination="10.2.0.2")),
            (
                "22",
                raw_row(
                    source="10.2.0.3",
                    destination="10.2.0.4",
                    in_bytes="20",
                    native="DoS",
                ),
            ),
        ]

        first = self.index(rows)
        repeated = self.index(list(reversed(rows)))

        self.assertEqual(first, repeated)

    def test_rejects_duplicate_source_reference_and_unknown_label(self):
        row = raw_row()
        with self.assertRaisesRegex(ValueError, "duplicate source reference"):
            self.index([("1", row), ("1", dict(row))])

        unknown = raw_row(native="UnknownAttack")
        with self.assertRaisesRegex(ValueError, "coarse label"):
            self.index([("2", unknown)])


class NFUNSWDevelopmentSelectionTests(unittest.TestCase):
    def test_selection_respects_cap_and_keeps_leakage_groups_whole(self):
        records = (
            indexed_record("a1", "group-a", "Benign"),
            indexed_record("a2", "group-a", "Benign"),
            indexed_record("a3", "group-a", "Benign"),
            indexed_record("b1", "group-b", "DoS"),
            indexed_record("b2", "group-b", "DoS"),
            indexed_record("c1", "group-c", "Exploits"),
            indexed_record("c2", "group-c", "Exploits"),
        )

        selected = select_nf_unsw_development_records(
            records,
            max_samples=5,
            seed=17,
        )

        self.assertLessEqual(len(selected), 5)
        selected_ids = {record.sample_id for record in selected}
        for group_id in {record.group_id for record in records}:
            group_ids = {
                record.sample_id
                for record in records
                if record.group_id == group_id
            }
            self.assertIn(len(selected_ids & group_ids), {0, len(group_ids)})

    def test_selection_prioritizes_groups_containing_rare_native_labels(self):
        records = (
            indexed_record("common-1", "group-common-1", "Benign"),
            indexed_record("common-2", "group-common-2", "Benign"),
            indexed_record("rare-1", "group-rare", "Worms"),
        )

        selected = select_nf_unsw_development_records(
            records,
            max_samples=1,
            seed=3,
        )

        self.assertEqual([record.sample_id for record in selected], ["rare-1"])

    def test_selection_is_order_independent_and_seed_breaks_group_ties(self):
        records = (
            indexed_record("a", "group-a", "Benign"),
            indexed_record("b", "group-b", "Benign"),
        )

        seed_one = select_nf_unsw_development_records(
            records,
            max_samples=1,
            seed=1,
        )
        seed_one_reversed = select_nf_unsw_development_records(
            tuple(reversed(records)),
            max_samples=1,
            seed=1,
        )
        seed_zero = select_nf_unsw_development_records(
            records,
            max_samples=1,
            seed=0,
        )

        self.assertEqual(seed_one, seed_one_reversed)
        self.assertEqual([record.group_id for record in seed_one], ["group-a"])
        self.assertEqual([record.group_id for record in seed_zero], ["group-b"])

    def test_selection_rejects_a_group_larger_than_the_cap(self):
        records = tuple(
            indexed_record(f"large-{index}", "group-large", "Benign")
            for index in range(3)
        )

        with self.assertRaisesRegex(ValueError, "group-large.*max_samples"):
            select_nf_unsw_development_records(
                records,
                max_samples=2,
                seed=0,
            )

    def test_selection_requires_a_positive_non_boolean_integer_cap(self):
        records = (indexed_record("one", "group-one", "Benign"),)

        for invalid in (True, False, 0, -1, 1.5, "1"):
            with self.subTest(max_samples=invalid):
                with self.assertRaisesRegex(ValueError, "max_samples"):
                    select_nf_unsw_development_records(
                        records,
                        max_samples=invalid,
                        seed=0,
                    )


if __name__ == "__main__":
    unittest.main()
