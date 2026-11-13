import unittest

from flowfm.data import SCHEMAFLOW_MAIN_REGISTRY as EXPORTED_MAIN_REGISTRY
from flowfm.data.schemaflow_registry import (
    MISSING_CATEGORY_ID,
    PORT_BUCKET_POLICY_VERSION,
    SCHEMAFLOW_MAIN_REGISTRY,
    UNKNOWN_CATEGORY_ID,
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    build_port_vocabulary,
    fit_categorical_vocabulary,
)


class CanonicalFeatureRegistryTests(unittest.TestCase):
    def test_main_registry_uses_explicit_stable_ids_and_excludes_dns_query_id(self):
        registry = SCHEMAFLOW_MAIN_REGISTRY

        self.assertEqual(registry.version, "schemaflow-main-features-v1")
        self.assertEqual(registry.by_name["L4_SRC_PORT"].feature_id, 1)
        self.assertEqual(registry.by_name["L4_DST_PORT"].feature_id, 2)
        self.assertEqual(registry.by_name["PROTOCOL"].feature_id, 3)
        self.assertEqual(registry.by_name["DNS_QUERY_ID"].feature_id, 38)
        self.assertFalse(registry.by_name["DNS_QUERY_ID"].model_visible)
        self.assertEqual(registry.by_name["DNS_QUERY_ID"].value_policy, "audit_only")
        self.assertNotIn(
            "DNS_QUERY_ID", tuple(spec.name for spec in registry.model_features)
        )
        self.assertEqual(registry.by_id[39].name, "DNS_QUERY_TYPE")
        self.assertEqual(len(registry.model_features), 40)
        self.assertIs(EXPORTED_MAIN_REGISTRY, registry)

    def test_registry_hash_and_lookup_do_not_depend_on_input_sequence_order(self):
        features = (
            CanonicalFeatureSpec(7, "BETA", "numeric", "continuous"),
            CanonicalFeatureSpec(3, "ALPHA", "categorical", "vocabulary"),
        )

        forward = CanonicalFeatureRegistry("test-registry-v1", features)
        reversed_registry = CanonicalFeatureRegistry(
            "test-registry-v1", tuple(reversed(features))
        )

        self.assertEqual(forward.registry_hash, reversed_registry.registry_hash)
        self.assertEqual(tuple(spec.feature_id for spec in forward.features), (3, 7))
        self.assertEqual(forward.by_name["ALPHA"].feature_id, 3)

    def test_registry_rejects_duplicate_ids_names_and_invalid_policies(self):
        with self.assertRaisesRegex(ValueError, "duplicate feature_id"):
            CanonicalFeatureRegistry(
                "test-v1",
                (
                    CanonicalFeatureSpec(1, "A", "numeric", "continuous"),
                    CanonicalFeatureSpec(1, "B", "numeric", "continuous"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "duplicate feature name"):
            CanonicalFeatureRegistry(
                "test-v1",
                (
                    CanonicalFeatureSpec(1, "A", "numeric", "continuous"),
                    CanonicalFeatureSpec(2, "A", "numeric", "continuous"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "categorical"):
            CanonicalFeatureSpec(1, "A", "numeric", "vocabulary")


class CategoricalVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.spec = CanonicalFeatureSpec(
            3, "PROTOCOL", "categorical", "vocabulary"
        )

    def test_reserved_ids_and_deterministic_frequency_cap(self):
        vocabulary = fit_categorical_vocabulary(
            self.spec,
            {"n:c": 2, "n:a": 10, "n:d": 1, "n:b": 2},
            training_row_count=15,
            training_sample_ids_hash="train-hash",
            max_known_categories=2,
        )

        self.assertEqual(MISSING_CATEGORY_ID, 0)
        self.assertEqual(UNKNOWN_CATEGORY_ID, 1)
        self.assertEqual(vocabulary.id_to_token[:2], ("<MISSING>", "<UNKNOWN>"))
        self.assertEqual(vocabulary.id_to_token[2:], ("n:a", "n:b"))
        self.assertEqual(vocabulary.encode(None), 0)
        self.assertEqual(vocabulary.encode("m:"), 0)
        self.assertEqual(vocabulary.encode("n:a"), 2)
        self.assertEqual(vocabulary.encode("n:b"), 3)
        self.assertEqual(vocabulary.encode("n:c"), 1)
        self.assertEqual(vocabulary.known_frequencies, (10, 2))
        self.assertEqual(vocabulary.overflow_unique_count, 2)
        self.assertEqual(vocabulary.overflow_frequency, 3)
        self.assertEqual(vocabulary.table_size, 4)

        same = fit_categorical_vocabulary(
            self.spec,
            {"n:b": 2, "n:d": 1, "n:a": 10, "n:c": 2},
            training_row_count=15,
            training_sample_ids_hash="train-hash",
            max_known_categories=2,
        )
        self.assertEqual(vocabulary.vocabulary_hash, same.vocabulary_hash)

    def test_vocabulary_fit_rejects_missing_tokens_bad_counts_and_non_train_scope(self):
        with self.assertRaisesRegex(ValueError, "missing token"):
            fit_categorical_vocabulary(
                self.spec,
                {"m:": 1},
                training_row_count=1,
                training_sample_ids_hash="train-hash",
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fit_categorical_vocabulary(
                self.spec,
                {"n:6": 0},
                training_row_count=1,
                training_sample_ids_hash="train-hash",
            )
        with self.assertRaisesRegex(ValueError, "fit_scope"):
            fit_categorical_vocabulary(
                self.spec,
                {"n:6": 1},
                training_row_count=1,
                training_sample_ids_hash="train-hash",
                fit_scope="all_splits",
            )

    def test_port_policy_boundaries_are_stable_and_shared(self):
        source_spec = CanonicalFeatureSpec(
            1, "L4_SRC_PORT", "categorical", PORT_BUCKET_POLICY_VERSION
        )
        destination_spec = CanonicalFeatureSpec(
            2, "L4_DST_PORT", "categorical", PORT_BUCKET_POLICY_VERSION
        )
        source = build_port_vocabulary(
            source_spec,
            {"n:0": 2, "n:1024": 3, "n:65535": 1},
            training_row_count=6,
            training_sample_ids_hash="train-hash",
        )
        destination = build_port_vocabulary(
            destination_spec,
            {"n:0": 2, "n:1024": 3, "n:65535": 1},
            training_row_count=6,
            training_sample_ids_hash="train-hash",
        )

        self.assertEqual(source.table_size, 1278)
        self.assertEqual(source.id_to_token, destination.id_to_token)
        self.assertEqual(source.encode("n:0"), 2)
        self.assertEqual(source.encode("n:1023"), 1025)
        self.assertEqual(source.encode("n:1024"), 1026)
        self.assertEqual(source.encode("n:1279"), 1026)
        self.assertEqual(source.encode("n:49151"), 1213)
        self.assertEqual(source.encode("n:49152"), 1214)
        self.assertEqual(source.encode("n:65535"), 1277)
        self.assertEqual(source.encode(None), 0)
        for token in ("n:-1", "n:65536", "n:1.5", "not-a-port"):
            self.assertEqual(source.encode(token), 1)
        self.assertEqual(source.known_frequencies[0], 2)
        self.assertEqual(source.known_frequencies[1024], 3)
        self.assertEqual(source.known_frequencies[-1], 1)


class TokenizerMetadataTests(unittest.TestCase):
    def test_metadata_round_trip_preserves_schema_vocab_and_manifest_identity(self):
        spec = CanonicalFeatureSpec(3, "PROTOCOL", "categorical", "vocabulary")
        vocabulary = fit_categorical_vocabulary(
            spec,
            {"n:6": 8, "n:17": 5},
            training_row_count=13,
            training_sample_ids_hash="training-members-sha256",
        )
        metadata = TokenizerMetadata(
            registry_version="test-registry-v1",
            registry_hash="registry-sha256",
            schema_version="schema-v1",
            schema_hash="schema-sha256",
            categorical_vocabularies={"PROTOCOL": vocabulary},
            data_manifest={"dataset_id": "fixture", "sha256": "data-sha256"},
            split_manifest_sha256="split-sha256",
            preprocessor_hash="preprocessor-sha256",
            training_sample_ids_hash="training-members-sha256",
        )

        restored = TokenizerMetadata.from_dict(metadata.to_dict())

        self.assertEqual(restored, metadata)
        self.assertEqual(restored.metadata_hash, metadata.metadata_hash)
        self.assertEqual(restored.to_dict()["reserved_category_ids"], {
            "missing_padding": 0,
            "unknown": 1,
        })
        self.assertEqual(
            restored.categorical_vocabularies["PROTOCOL"].table_size, 4
        )


if __name__ == "__main__":
    unittest.main()
