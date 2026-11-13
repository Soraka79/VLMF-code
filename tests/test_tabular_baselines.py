import unittest

from flowfm.data.contracts import FlowSample
from flowfm.training.baselines import (
    TabularFeatureEncoder,
    inverse_frequency_sample_weights,
)


def _sample(sample_id, feature_ids, features, categorical_mask, binary="benign"):
    missing = tuple(False for _ in feature_ids)
    tokens = tuple(
        str(value) if is_cat else None
        for value, is_cat in zip(features, categorical_mask)
    )
    return FlowSample(
        features=features,
        feature_ids=feature_ids,
        categorical_mask=categorical_mask,
        missing_mask=missing,
        category_tokens=tokens,
        binary_label=binary,
        coarse_label="benign" if binary == "benign" else "DoS",
        native_label="benign" if binary == "benign" else "DoS",
        dataset_id="synthetic",
        sample_id=sample_id,
        feature_key=f"key-{sample_id}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="synthetic.csv",
        source_row_id=sample_id,
    )


class TabularBaselineEncoderTests(unittest.TestCase):
    def test_fit_is_train_only_and_unknown_maps_to_reserved_unknown(self):
        train = (
            _sample("1", ("B", "A"), (0.2, 2), (False, True)),
            _sample("2", ("B", "A"), (0.8, 0), (False, True)),
        )
        validation = (
            _sample("3", ("A", "B"), (99, 0.5), (True, False)),
        )

        encoder = TabularFeatureEncoder.fit(train)
        train_matrix = encoder.transform(train)
        validation_matrix = encoder.transform(validation)

        self.assertEqual(train_matrix.shape[1], validation_matrix.shape[1])
        self.assertEqual(encoder.feature_names, ("A", "B"))
        self.assertEqual(encoder.categorical_feature_names, ("A",))
        self.assertEqual(encoder.unknown_category_id, 1)
        self.assertEqual(validation_matrix.shape[0], 1)
        self.assertAlmostEqual(float(validation_matrix.sum()), 1.5)

    def test_feature_order_permutation_is_invariant(self):
        first = _sample("1", ("B", "A"), (0.25, 2), (False, True))
        permuted = _sample("2", ("A", "B"), (2, 0.25), (True, False))
        encoder = TabularFeatureEncoder.fit((first,))

        left = encoder.transform((first,))
        right = encoder.transform((permuted,))

        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.tolist(), right.tolist())

    def test_empty_fit_and_schema_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            TabularFeatureEncoder.fit(())

        train = (_sample("1", ("A",), (1,), (True,)),)
        encoder = TabularFeatureEncoder.fit(train)
        bad = (_sample("2", ("A",), (0.2,), (False,)),)
        with self.assertRaises(ValueError):
            encoder.transform(bad)

    def test_canonical_vocabulary_caps_high_cardinality(self):
        class Vocabulary:
            table_size = 4

            def encode(self, token):
                return 3 if token == "2" else 1

        train = (
            _sample("1", ("A",), (2,), (True,)),
        )
        encoder = TabularFeatureEncoder.fit(
            train,
            category_encoders={"A": Vocabulary()},
        )
        matrix = encoder.transform(train)
        self.assertEqual(matrix.shape, (1, 5))
        self.assertEqual(len(encoder.categorical_values["A"]), 4)


    def test_inverse_frequency_weights_balance_observed_classes(self):
        weights = inverse_frequency_sample_weights([0, 0, 0, 1])
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))
        with self.assertRaises(ValueError):
            inverse_frequency_sample_weights([])
        with self.assertRaises(ValueError):
            inverse_frequency_sample_weights([0, -1])


if __name__ == "__main__":
    unittest.main()
