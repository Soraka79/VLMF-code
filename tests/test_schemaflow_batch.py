import unittest

import torch

from flowfm.data import SchemaFlowBatch as ExportedSchemaFlowBatch
from flowfm.data.contracts import FlowSample
from flowfm.data.schemaflow_batch import (
    CATEGORICAL_FEATURE_TYPE_ID,
    NUMERIC_FEATURE_TYPE_ID,
    SchemaFlowBatch,
    collate_schemaflow_samples,
)
from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)


def make_sample(
    *,
    sample_id,
    feature_ids,
    features,
    categorical_mask,
    missing_mask,
    category_tokens,
):
    return FlowSample(
        features=features,
        feature_ids=feature_ids,
        categorical_mask=categorical_mask,
        missing_mask=missing_mask,
        category_tokens=category_tokens,
        binary_label="attack",
        coarse_label="dos",
        native_label="DoS",
        dataset_id="fixture",
        sample_id=sample_id,
        feature_key=f"feature-{sample_id}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id,
    )


class SchemaFlowBatchTests(unittest.TestCase):
    def setUp(self):
        self.registry = CanonicalFeatureRegistry(
            "fixture-registry-v1",
            (
                CanonicalFeatureSpec(
                    7, "BYTES", "numeric", "continuous"
                ),
                CanonicalFeatureSpec(
                    3, "PROTOCOL", "categorical", "vocabulary"
                ),
                CanonicalFeatureSpec(
                    9,
                    "DNS_QUERY_ID",
                    "categorical",
                    "audit_only",
                    model_visible=False,
                ),
            ),
        )
        protocol = fit_categorical_vocabulary(
            self.registry.by_name["PROTOCOL"],
            {"n:6": 2},
            training_row_count=2,
            training_sample_ids_hash="training-hash",
        )
        self.metadata = TokenizerMetadata(
            registry_version=self.registry.version,
            registry_hash=self.registry.registry_hash,
            schema_version="fixture-schema-v1",
            schema_hash="fixture-schema-hash",
            categorical_vocabularies={
                "PROTOCOL": protocol,
            },
            data_manifest={
                "dataset_id": "fixture",
                "sha256": "data-hash",
            },
            split_manifest_sha256="split-hash",
            preprocessor_hash="preprocessor-hash",
            training_sample_ids_hash="training-hash",
        )

    def test_collate_maps_by_canonical_id_not_local_order_and_omits_audit(self):
        first = make_sample(
            sample_id="1",
            feature_ids=(
                "BYTES",
                "DNS_QUERY_ID",
                "PROTOCOL",
            ),
            features=(1.5, 2, 2),
            categorical_mask=(False, True, True),
            missing_mask=(False, False, False),
            category_tokens=(None, "n:42", "n:6"),
        )
        second = make_sample(
            sample_id="2",
            feature_ids=(
                "PROTOCOL",
                "BYTES",
                "DNS_QUERY_ID",
            ),
            features=(2, 1.5, 2),
            categorical_mask=(True, False, True),
            missing_mask=(False, False, False),
            category_tokens=("n:6", None, "n:43"),
        )

        batch = collate_schemaflow_samples(
            (first, second),
            self.metadata,
            registry=self.registry,
        )

        self.assertIsInstance(batch, SchemaFlowBatch)
        self.assertIs(ExportedSchemaFlowBatch, SchemaFlowBatch)
        self.assertEqual(batch.feature_ids.tolist(), [[3, 7], [3, 7]])
        self.assertEqual(
            batch.feature_type_ids.tolist(),
            [
                [
                    CATEGORICAL_FEATURE_TYPE_ID,
                    NUMERIC_FEATURE_TYPE_ID,
                ],
                [
                    CATEGORICAL_FEATURE_TYPE_ID,
                    NUMERIC_FEATURE_TYPE_ID,
                ],
            ],
        )
        self.assertEqual(batch.category_ids.tolist(), [[2, 0], [2, 0]])
        self.assertEqual(
            batch.numeric_values.tolist(),
            [[0.0, 1.5], [0.0, 1.5]],
        )
        self.assertTrue(batch.schema_present_mask.all().item())
        self.assertFalse(batch.missing_mask.any().item())
        self.assertEqual(batch.sample_ids, ("1", "2"))
        self.assertEqual(batch.dataset_ids, ("fixture", "fixture"))
        self.assertNotIn(9, batch.feature_ids.tolist()[0])

    def test_unknown_missing_and_schema_absence_are_distinct_and_safe(self):
        missing_numeric = make_sample(
            sample_id="1",
            feature_ids=("PROTOCOL", "BYTES"),
            features=(1, float("nan")),
            categorical_mask=(True, False),
            missing_mask=(False, True),
            category_tokens=("n:999", None),
        )
        absent_numeric = make_sample(
            sample_id="2",
            feature_ids=("PROTOCOL",),
            features=(2,),
            categorical_mask=(True,),
            missing_mask=(False,),
            category_tokens=("n:6",),
        )

        batch = collate_schemaflow_samples(
            (missing_numeric, absent_numeric),
            self.metadata,
            registry=self.registry,
        )

        self.assertEqual(batch.category_ids[:, 0].tolist(), [1, 2])
        self.assertEqual(batch.numeric_values[:, 1].tolist(), [0.0, 0.0])
        self.assertEqual(batch.missing_mask[:, 1].tolist(), [True, True])
        self.assertEqual(
            batch.schema_present_mask[:, 1].tolist(),
            [True, False],
        )
        self.assertTrue(torch.isfinite(batch.numeric_values).all().item())

    def test_rejects_observed_nan_invalid_category_type_and_unknown_feature(self):
        invalid_samples = (
            (
                make_sample(
                    sample_id="nan",
                    feature_ids=("BYTES",),
                    features=(float("nan"),),
                    categorical_mask=(False,),
                    missing_mask=(False,),
                    category_tokens=(None,),
                ),
                "finite",
            ),
            (
                make_sample(
                    sample_id="category",
                    feature_ids=("PROTOCOL",),
                    features=(-1,),
                    categorical_mask=(True,),
                    missing_mask=(False,),
                    category_tokens=("n:6",),
                ),
                "category ID",
            ),
            (
                make_sample(
                    sample_id="type",
                    feature_ids=("PROTOCOL",),
                    features=(2,),
                    categorical_mask=(False,),
                    missing_mask=(False,),
                    category_tokens=(None,),
                ),
                "type mismatch",
            ),
            (
                make_sample(
                    sample_id="unknown",
                    feature_ids=("RAW_IP",),
                    features=(1.0,),
                    categorical_mask=(False,),
                    missing_mask=(False,),
                    category_tokens=(None,),
                ),
                "unregistered feature",
            ),
        )

        for sample, message in invalid_samples:
            with self.subTest(sample=sample.sample_id):
                with self.assertRaisesRegex(ValueError, message):
                    collate_schemaflow_samples(
                        (sample,),
                        self.metadata,
                        registry=self.registry,
                    )

    def test_batch_to_and_synchronized_permutation_keep_tensors_aligned(self):
        sample = make_sample(
            sample_id="1",
            feature_ids=("PROTOCOL", "BYTES"),
            features=(2, 3.5),
            categorical_mask=(True, False),
            missing_mask=(False, False),
            category_tokens=("n:6", None),
        )
        batch = collate_schemaflow_samples(
            (sample,),
            self.metadata,
            registry=self.registry,
        )

        permuted = batch.permute_features(torch.tensor([1, 0]))
        restored = permuted.permute_features(torch.tensor([1, 0]))

        for field_name in (
            "feature_ids",
            "numeric_values",
            "category_ids",
            "feature_type_ids",
            "missing_mask",
            "schema_present_mask",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(restored, field_name),
                    getattr(batch, field_name),
                )
            )
        self.assertEqual(
            permuted.feature_ids.tolist(),
            [[7, 3]],
        )
        moved = batch.to("cpu")
        self.assertEqual(moved.sample_ids, batch.sample_ids)
        self.assertEqual(moved.numeric_values.device.type, "cpu")

        with self.assertRaisesRegex(ValueError, "permutation"):
            batch.permute_features(torch.tensor([0, 0]))


if __name__ == "__main__":
    unittest.main()
