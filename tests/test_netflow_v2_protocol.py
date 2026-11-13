import json
import unittest
from dataclasses import replace

from flowfm.data.labels import NF_TON_LABEL_SPACE
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.netflow_v2_protocol import (
    NetFlowV2IndexedRecord,
    index_netflow_v2_rows,
    select_netflow_v2_development_records,
    select_netflow_v2_proportional_development_records,
)


LEAKAGE_NAMESPACE = "flowfm-nf-ton-iot-v2-leakage-group-v1"
ORDER_NAMESPACE = "flowfm.nf-ton-iot-v2-development-group-order.v1"
SENSITIVE_NAMESPACE = "flowfm-nf-ton-iot-v2-sensitive-v1"


class NetFlowV2ProtocolTests(unittest.TestCase):
    def _base_row(self):
        row = {
            name: str(index + 1)
            for index, name in enumerate(
                NF_TON_IOT_V2_SCHEMA.model_feature_names
            )
        }
        row.update(
            {
                "IPV4_SRC_ADDR": "192.0.2.10",
                "IPV4_DST_ADDR": "198.51.100.20",
                "Label": "1",
                "Attack": "password",
            }
        )
        return row

    def _rows(self):
        first = self._base_row()
        second = dict(first)
        second.update(
            {
                "IPV4_SRC_ADDR": "203.0.113.11",
                "IPV4_DST_ADDR": "203.0.113.12",
                "Attack": "injection",
            }
        )
        third = dict(second)
        third["IN_BYTES"] = "999"
        third["Attack"] = "Benign"
        third["Label"] = "0"
        return (("1", first), ("2", second), ("3", third))

    def _index(self, rows=None):
        return index_netflow_v2_rows(
            self._rows() if rows is None else rows,
            schema=NF_TON_IOT_V2_SCHEMA,
            label_space=NF_TON_LABEL_SPACE,
            source_file_id="NF-ToN-IoT-v2.csv",
            conflict_feature_keys=set(),
            sensitive_hash_namespace=SENSITIVE_NAMESPACE,
            leakage_group_namespace=LEAKAGE_NAMESPACE,
        )

    def test_indexing_is_schema_label_and_input_order_stable(self):
        indexed = self._index()
        reversed_indexed = self._index(tuple(reversed(self._rows())))
        self.assertEqual(indexed, reversed_indexed)
        self.assertEqual(len(indexed), 3)
        self.assertEqual(len({record.sample_id for record in indexed}), 3)
        self.assertEqual(len({record.group_id for record in indexed}), 1)
        self.assertEqual(indexed[0].source_file, "NF-ToN-IoT-v2.csv")
        self.assertEqual(
            {record.native_label for record in indexed},
            {"Benign", "injection", "password"},
        )
        self.assertEqual(
            {record.coarse_label for record in indexed},
            {"benign", "brute_force", "exploit"},
        )
        self.assertEqual(
            {record.binary_label for record in indexed},
            {"attack", "benign"},
        )
        feature_keys = [record.feature_key for record in indexed]
        self.assertEqual(len(set(feature_keys)), 2)
        serialized = json.dumps(
            [record.__dict__ for record in indexed],
            sort_keys=True,
        )
        for value in (
            "192.0.2.10",
            "198.51.100.20",
            "203.0.113.11",
            "203.0.113.12",
        ):
            self.assertNotIn(value, serialized)

    def test_indexing_rejects_unknown_native_and_duplicate_source(self):
        unknown = self._base_row()
        unknown["Attack"] = "unknown"
        with self.assertRaisesRegex(ValueError, "native label"):
            self._index((("1", unknown),))

        first = self._base_row()
        second = dict(first)
        second["Attack"] = "injection"
        with self.assertRaisesRegex(ValueError, "duplicate source reference"):
            self._index((("1", first), ("1", second)))

    def test_indexing_rejects_schema_and_label_space_mismatch(self):
        from flowfm.data.labels import NF_UNSW_LABEL_SPACE

        with self.assertRaisesRegex(ValueError, "label space does not match"):
            index_netflow_v2_rows(
                self._rows(),
                schema=NF_TON_IOT_V2_SCHEMA,
                label_space=NF_UNSW_LABEL_SPACE,
                source_file_id="NF-ToN-IoT-v2.csv",
                conflict_feature_keys=set(),
                sensitive_hash_namespace=SENSITIVE_NAMESPACE,
                leakage_group_namespace=LEAKAGE_NAMESPACE,
            )

    def test_sampling_keeps_whole_groups_and_is_order_invariant(self):
        template = NetFlowV2IndexedRecord(
            sample_id="sample-a1",
            feature_key="feature-a",
            group_id="group-a",
            binary_label="attack",
            coarse_label="malware",
            native_label="ransomware",
            source_file="NF-ToN-IoT-v2.csv",
            source_row_id="1",
            host_pair_hash="host-a",
            five_tuple_hash="tuple-a",
        )
        records = (
            template,
            replace(
                template,
                sample_id="sample-a2",
                source_row_id="2",
            ),
            replace(
                template,
                sample_id="sample-b1",
                feature_key="feature-b",
                group_id="group-b",
                native_label="scanning",
                coarse_label="scan",
                source_row_id="3",
            ),
        )
        selected = select_netflow_v2_development_records(
            records,
            max_samples=2,
            seed=42,
            order_namespace=ORDER_NAMESPACE,
        )
        selected_reversed = select_netflow_v2_development_records(
            reversed(records),
            max_samples=2,
            seed=42,
            order_namespace=ORDER_NAMESPACE,
        )
        self.assertEqual(selected, selected_reversed)
        selected_groups = {record.group_id for record in selected}
        for group_id in selected_groups:
            self.assertEqual(
                sum(record.group_id == group_id for record in selected),
                sum(record.group_id == group_id for record in records),
            )

        with self.assertRaisesRegex(ValueError, "exceeds max_samples"):
            select_netflow_v2_development_records(
                records[:2],
                max_samples=1,
                seed=42,
                order_namespace=ORDER_NAMESPACE,
            )

    def test_sampling_covers_required_native_labels_when_feasible(self):
        template = NetFlowV2IndexedRecord(
            sample_id="sample-scan",
            feature_key="feature-scan",
            group_id="group-scan",
            binary_label="attack",
            coarse_label="scan",
            native_label="scanning",
            source_file="NF-ToN-IoT-v2.csv",
            source_row_id="1",
            host_pair_hash="host-scan",
            five_tuple_hash="tuple-scan",
        )
        records = (
            template,
            replace(
                template,
                sample_id="sample-ransom-a",
                feature_key="feature-ransom-a",
                group_id="group-ransom-a",
                native_label="ransomware",
                coarse_label="malware",
                source_row_id="2",
            ),
            replace(
                template,
                sample_id="sample-ransom-b",
                feature_key="feature-ransom-b",
                group_id="group-ransom-b",
                native_label="ransomware",
                coarse_label="malware",
                source_row_id="3",
            ),
            replace(
                template,
                sample_id="sample-benign-a",
                feature_key="feature-benign-a",
                group_id="group-benign-a",
                native_label="Benign",
                coarse_label="benign",
                binary_label="benign",
                source_row_id="4",
            ),
            replace(
                template,
                sample_id="sample-benign-b",
                feature_key="feature-benign-b",
                group_id="group-benign-b",
                native_label="Benign",
                coarse_label="benign",
                binary_label="benign",
                source_row_id="5",
            ),
            replace(
                template,
                sample_id="sample-benign-c",
                feature_key="feature-benign-c",
                group_id="group-benign-c",
                native_label="Benign",
                coarse_label="benign",
                binary_label="benign",
                source_row_id="6",
            ),
        )
        selected = select_netflow_v2_development_records(
            records,
            max_samples=3,
            seed=42,
            order_namespace=ORDER_NAMESPACE,
            required_native_labels=("Benign", "ransomware", "scanning"),
        )
        self.assertEqual(
            {record.native_label for record in selected},
            {"Benign", "ransomware", "scanning"},
        )
        self.assertEqual(len(selected), 3)


    def test_proportional_sampling_preserves_native_rows_not_group_frequency(self):
        template = NetFlowV2IndexedRecord(
            sample_id="benign-single",
            feature_key="feature-benign-single",
            group_id="group-benign-single",
            binary_label="benign",
            coarse_label="benign",
            native_label="Benign",
            source_file="synthetic.csv",
            source_row_id="1",
            host_pair_hash="host-benign-single",
            five_tuple_hash="tuple-benign-single",
        )
        records = [template]
        source_row = 2
        for group_id, size in (("group-benign-five", 5), ("group-benign-six", 6)):
            for index in range(size):
                records.append(
                    replace(
                        template,
                        sample_id=f"{group_id}-{index}",
                        feature_key=f"feature-{group_id}-{index}",
                        group_id=group_id,
                        source_row_id=str(source_row),
                    )
                )
                source_row += 1
        for index in range(8):
            records.append(
                replace(
                    template,
                    sample_id=f"attack-{index}",
                    feature_key=f"feature-attack-{index}",
                    group_id=f"group-attack-{index}",
                    binary_label="attack",
                    coarse_label="scan",
                    native_label="scanning",
                    source_row_id=str(source_row),
                )
            )
            source_row += 1

        selected = select_netflow_v2_proportional_development_records(
            records,
            max_samples=10,
            seed=42,
            order_namespace="flowfm.test.proportional.v1",
            required_native_labels=("Benign", "scanning"),
        )
        selected_reversed = select_netflow_v2_proportional_development_records(
            reversed(records),
            max_samples=10,
            seed=42,
            order_namespace="flowfm.test.proportional.v1",
            required_native_labels=("Benign", "scanning"),
        )
        self.assertEqual(selected, selected_reversed)
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            {label: sum(record.native_label == label for record in selected)
             for label in ("Benign", "scanning")},
            {"Benign": 6, "scanning": 4},
        )
        for group_id in {record.group_id for record in selected}:
            self.assertEqual(
                sum(record.group_id == group_id for record in selected),
                sum(record.group_id == group_id for record in records),
            )


if __name__ == "__main__":
    unittest.main()
