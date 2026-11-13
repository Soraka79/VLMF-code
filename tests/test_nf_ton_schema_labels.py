import unittest

from flowfm.data.contracts import FlowSample


class NFTONSchemaTests(unittest.TestCase):
    def test_nf_ton_reuses_ordered_netflow_features_with_distinct_identity(self):
        from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
        from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA

        self.assertEqual(NF_TON_IOT_V2_SCHEMA.dataset_id, "NF-ToN-IoT-v2")
        self.assertEqual(
            NF_TON_IOT_V2_SCHEMA.version,
            "nf-ton-iot-v2-model-41-v1",
        )
        self.assertEqual(len(NF_TON_IOT_V2_SCHEMA.raw_features), 43)
        self.assertEqual(len(NF_TON_IOT_V2_SCHEMA.model_features), 41)
        self.assertEqual(
            NF_TON_IOT_V2_SCHEMA.trace_feature_names,
            ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"),
        )
        self.assertEqual(
            NF_TON_IOT_V2_SCHEMA.raw_features,
            NF_UNSW_NB15_V2_SCHEMA.raw_features,
        )
        self.assertEqual(
            NF_TON_IOT_V2_SCHEMA.model_features,
            NF_UNSW_NB15_V2_SCHEMA.model_features,
        )
        self.assertNotEqual(
            NF_TON_IOT_V2_SCHEMA.schema_hash,
            NF_UNSW_NB15_V2_SCHEMA.schema_hash,
        )

    def test_schema_registry_requires_exact_dataset_and_version(self):
        from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
        from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
        from flowfm.data.schemas.registry import get_dataset_schema

        self.assertIs(
            get_dataset_schema(
                NF_TON_IOT_V2_SCHEMA.dataset_id,
                NF_TON_IOT_V2_SCHEMA.version,
            ),
            NF_TON_IOT_V2_SCHEMA,
        )
        self.assertIs(
            get_dataset_schema(
                NF_UNSW_NB15_V2_SCHEMA.dataset_id,
                NF_UNSW_NB15_V2_SCHEMA.version,
            ),
            NF_UNSW_NB15_V2_SCHEMA,
        )
        with self.assertRaisesRegex(ValueError, "unregistered dataset Schema"):
            get_dataset_schema("unknown", "v1")
        with self.assertRaisesRegex(ValueError, "unregistered dataset Schema"):
            get_dataset_schema(
                NF_TON_IOT_V2_SCHEMA.dataset_id,
                NF_UNSW_NB15_V2_SCHEMA.version,
            )


class NFTONLabelSpaceTests(unittest.TestCase):
    def _sample(self, *, native, coarse, binary="attack"):
        return FlowSample(
            features=(1.0,),
            feature_ids=("IN_BYTES",),
            categorical_mask=(False,),
            missing_mask=(False,),
            category_tokens=(None,),
            binary_label=binary,
            coarse_label=coarse,
            native_label=native,
            dataset_id="NF-ToN-IoT-v2",
            sample_id=f"sample-{native}-{coarse}",
            feature_key=f"feature-{native}-{coarse}",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="NF-ToN-IoT-v2.csv",
            source_row_id="1",
        )

    def test_core_datasets_share_one_eight_class_coarse_order(self):
        from flowfm.data.labels import (
            NF_TON_LABEL_SPACE,
            NF_UNSW_LABEL_SPACE,
            SHARED_BINARY_LABELS,
            SHARED_COARSE_LABELS,
        )

        expected = (
            "benign",
            "dos",
            "scan",
            "brute_force",
            "exploit",
            "botnet",
            "backdoor",
            "malware",
        )
        self.assertEqual(SHARED_BINARY_LABELS, ("benign", "attack"))
        self.assertEqual(SHARED_COARSE_LABELS, expected)
        self.assertEqual(NF_UNSW_LABEL_SPACE.coarse_labels, expected)
        self.assertEqual(NF_TON_LABEL_SPACE.coarse_labels, expected)
        self.assertEqual(
            NF_TON_LABEL_SPACE.native_to_coarse,
            {
                "Benign": "benign",
                "backdoor": "backdoor",
                "ddos": "dos",
                "dos": "dos",
                "injection": "exploit",
                "mitm": "exploit",
                "password": "brute_force",
                "ransomware": "malware",
                "scanning": "scan",
                "xss": "exploit",
            },
        )

    def test_nf_ton_hierarchy_encodes_and_rejects_unknown_or_mismatch(self):
        from flowfm.data.labels import NF_TON_LABEL_SPACE, encode_supervised_targets

        password = self._sample(native="password", coarse="brute_force")
        targets = encode_supervised_targets((password,), NF_TON_LABEL_SPACE)
        self.assertEqual(targets.binary.tolist(), [1])
        self.assertEqual(
            targets.coarse.tolist(),
            [NF_TON_LABEL_SPACE.coarse_labels.index("brute_force")],
        )
        self.assertEqual(
            targets.native.tolist(),
            [NF_TON_LABEL_SPACE.native_labels.index("password")],
        )
        with self.assertRaisesRegex(ValueError, "hierarchy is inconsistent"):
            encode_supervised_targets(
                (self._sample(native="password", coarse="exploit"),),
                NF_TON_LABEL_SPACE,
            )
        with self.assertRaisesRegex(ValueError, "unknown native label"):
            encode_supervised_targets(
                (self._sample(native="unknown", coarse="exploit"),),
                NF_TON_LABEL_SPACE,
            )


if __name__ == "__main__":
    unittest.main()
