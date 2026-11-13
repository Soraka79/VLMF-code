import unittest

from flowfm.data.contracts import FlowSample


class NFCSESchemaTests(unittest.TestCase):
    def test_nf_cse_reuses_ordered_netflow_features_with_distinct_identity(self):
        from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
            NF_CSE_CIC_IDS2018_V2_SCHEMA,
        )
        from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA

        self.assertEqual(
            NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
            "NF-CSE-CIC-IDS2018-v2",
        )
        self.assertEqual(
            NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
            "nf-cse-cic-ids2018-v2-model-41-v1",
        )
        self.assertEqual(len(NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_features), 43)
        self.assertEqual(len(NF_CSE_CIC_IDS2018_V2_SCHEMA.model_features), 41)
        self.assertEqual(
            NF_CSE_CIC_IDS2018_V2_SCHEMA.trace_feature_names,
            ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"),
        )
        self.assertEqual(
            NF_CSE_CIC_IDS2018_V2_SCHEMA.raw_features,
            NF_UNSW_NB15_V2_SCHEMA.raw_features,
        )
        self.assertNotEqual(
            NF_CSE_CIC_IDS2018_V2_SCHEMA.schema_hash,
            NF_UNSW_NB15_V2_SCHEMA.schema_hash,
        )

    def test_schema_registry_resolves_nf_cse_exactly(self):
        from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
            NF_CSE_CIC_IDS2018_V2_SCHEMA,
        )
        from flowfm.data.schemas.registry import get_dataset_schema

        self.assertIs(
            get_dataset_schema(
                NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
                NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
            ),
            NF_CSE_CIC_IDS2018_V2_SCHEMA,
        )


class NFCSELabelSpaceTests(unittest.TestCase):
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
            dataset_id="NF-CSE-CIC-IDS2018-v2",
            sample_id=f"sample-{native}",
            feature_key=f"feature-{native}",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="NF-CSE-CIC-IDS2018-v2.csv",
            source_row_id="1",
        )

    def test_nf_cse_uses_versioned_native_labels_and_shared_coarse_order(self):
        from flowfm.data.labels import (
            NF_CSE_LABEL_SPACE,
            SHARED_COARSE_LABELS,
        )

        self.assertEqual(NF_CSE_LABEL_SPACE.coarse_labels, SHARED_COARSE_LABELS)
        self.assertEqual(len(NF_CSE_LABEL_SPACE.native_labels), 15)
        self.assertEqual(
            NF_CSE_LABEL_SPACE.native_to_coarse,
            {
                "Benign": "benign",
                "Bot": "botnet",
                "Brute Force -Web": "brute_force",
                "Brute Force -XSS": "brute_force",
                "DDOS attack-HOIC": "dos",
                "DDOS attack-LOIC-UDP": "dos",
                "DDoS attacks-LOIC-HTTP": "dos",
                "DoS attacks-GoldenEye": "dos",
                "DoS attacks-Hulk": "dos",
                "DoS attacks-SlowHTTPTest": "dos",
                "DoS attacks-Slowloris": "dos",
                "FTP-BruteForce": "brute_force",
                "Infilteration": "exploit",
                "SQL Injection": "exploit",
                "SSH-Bruteforce": "brute_force",
            },
        )

    def test_filtered_shared_coarse_task_is_versioned_and_ordered(self):
        from flowfm.data.labels import NF_CSE_SHARED_COARSE_LABEL_SPACE

        self.assertEqual(
            NF_CSE_SHARED_COARSE_LABEL_SPACE.coarse_labels,
            ("benign", "dos", "brute_force", "exploit", "botnet"),
        )
        self.assertEqual(
            NF_CSE_SHARED_COARSE_LABEL_SPACE.dataset_id,
            "NF-CSE-CIC-IDS2018-v2",
        )
        self.assertEqual(
            NF_CSE_SHARED_COARSE_LABEL_SPACE.mapping_version,
            "nf-cse-cic-ids2018-v2-shared-coarse-five-class-v1",
        )

    def test_nf_cse_hierarchy_encodes_and_rejects_mismatch(self):
        from flowfm.data.labels import NF_CSE_LABEL_SPACE, encode_supervised_targets

        sample = self._sample(native="Bot", coarse="botnet")
        targets = encode_supervised_targets((sample,), NF_CSE_LABEL_SPACE)
        self.assertEqual(targets.binary.tolist(), [1])
        self.assertEqual(
            targets.coarse.tolist(),
            [NF_CSE_LABEL_SPACE.coarse_labels.index("botnet")],
        )
        self.assertEqual(
            targets.native.tolist(),
            [NF_CSE_LABEL_SPACE.native_labels.index("Bot")],
        )
        with self.assertRaisesRegex(ValueError, "hierarchy is inconsistent"):
            encode_supervised_targets(
                (self._sample(native="Bot", coarse="exploit"),),
                NF_CSE_LABEL_SPACE,
            )


if __name__ == "__main__":
    unittest.main()
