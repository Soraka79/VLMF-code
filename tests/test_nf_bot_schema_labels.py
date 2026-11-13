import unittest


class NFBOTSchemaTests(unittest.TestCase):
    def test_nf_bot_schema_has_stable_dataset_identity_and_shared_features(self):
        from flowfm.data.schemas.nf_bot_iot_v2 import NF_BOT_IOT_V2_SCHEMA
        from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA
        from flowfm.data.schemas.registry import get_dataset_schema

        self.assertEqual(NF_BOT_IOT_V2_SCHEMA.dataset_id, "NF-BoT-IoT-v2")
        self.assertEqual(
            NF_BOT_IOT_V2_SCHEMA.version,
            "nf-bot-iot-v2-model-41-v1",
        )
        self.assertEqual(len(NF_BOT_IOT_V2_SCHEMA.raw_features), 43)
        self.assertEqual(len(NF_BOT_IOT_V2_SCHEMA.model_features), 41)
        self.assertEqual(
            NF_BOT_IOT_V2_SCHEMA.raw_features,
            NF_UNSW_NB15_V2_SCHEMA.raw_features,
        )
        self.assertEqual(
            NF_BOT_IOT_V2_SCHEMA.model_features,
            NF_UNSW_NB15_V2_SCHEMA.model_features,
        )
        self.assertNotEqual(
            NF_BOT_IOT_V2_SCHEMA.schema_hash,
            NF_UNSW_NB15_V2_SCHEMA.schema_hash,
        )
        self.assertIs(
            get_dataset_schema(
                "NF-BoT-IoT-v2", "nf-bot-iot-v2-model-41-v1"
            ),
            NF_BOT_IOT_V2_SCHEMA,
        )


class NFBOTLabelSpaceTests(unittest.TestCase):
    def test_nf_bot_native_labels_map_to_shared_coarse_space(self):
        from flowfm.data.labels import NF_BOT_LABEL_SPACE, SHARED_COARSE_LABELS

        self.assertEqual(NF_BOT_LABEL_SPACE.coarse_labels, SHARED_COARSE_LABELS)
        self.assertEqual(
            NF_BOT_LABEL_SPACE.native_labels,
            ("Benign", "DDoS", "DoS", "Reconnaissance", "Theft"),
        )
        self.assertEqual(
            dict(NF_BOT_LABEL_SPACE.native_to_coarse),
            {
                "Benign": "benign",
                "DDoS": "dos",
                "DoS": "dos",
                "Reconnaissance": "scan",
                "Theft": "malware",
            },
        )

    def test_predeclared_coarse_task_filters_native_labels_by_parent(self):
        from flowfm.data.labels import (
            NF_BOT_LABEL_SPACE,
            coarse_task_label_space,
        )

        task = coarse_task_label_space(
            NF_BOT_LABEL_SPACE,
            ("benign", "dos", "scan"),
            mapping_version="test-three-class-v1",
        )

        self.assertEqual(task.coarse_labels, ("benign", "dos", "scan"))
        self.assertNotIn("Theft", task.native_labels)


if __name__ == "__main__":
    unittest.main()
