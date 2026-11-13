import unittest

from flowfm.data.contracts import FlowSample
from flowfm.data.missingness import (
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_deterministic_field_missingness,
    apply_structured_field_group_missingness,
)
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
    NF_CSE_CIC_IDS2018_V2_SCHEMA,
)
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY


def _sample(sample_id="sample-1"):
    return FlowSample(
        features=(1.5, 3, 2.5, 7, 4.5),
        feature_ids=("a", "b", "c", "d", "e"),
        categorical_mask=(False, True, False, True, False),
        missing_mask=(False, False, False, False, False),
        category_tokens=(None, "n:3", None, "n:7", None),
        binary_label="attack",
        coarse_label="dos",
        native_label="DoS",
        dataset_id="fixture",
        sample_id=sample_id,
        feature_key="feature-" + sample_id,
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id,
    )


def _netflow_sample(sample_id="netflow-1"):
    schema = NF_CSE_CIC_IDS2018_V2_SCHEMA
    categorical = schema.categorical_mask
    return FlowSample(
        features=tuple(1 for _ in schema.model_features),
        feature_ids=schema.model_feature_names,
        categorical_mask=categorical,
        missing_mask=tuple(False for _ in schema.model_features),
        category_tokens=tuple(
            "n:1" if is_categorical else None for is_categorical in categorical
        ),
        binary_label="attack",
        coarse_label="dos",
        native_label="DoS",
        dataset_id="fixture",
        sample_id=sample_id,
        feature_key="feature-" + sample_id,
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id,
    )


class DeterministicMissingnessTests(unittest.TestCase):
    def test_hides_exact_floor_count_and_preserves_sample_identity(self):
        original = _sample()
        (perturbed,), report = apply_deterministic_field_missingness(
            (original,), fraction=0.4, seed=17
        )

        selected = [
            index for index, missing in enumerate(perturbed.missing_mask) if missing
        ]
        self.assertEqual(len(selected), 2)
        self.assertEqual(report["hidden_fields_per_sample"], 2)
        self.assertEqual(report["total_selected_fields"], 2)
        self.assertFalse(any(original.missing_mask))
        self.assertEqual(perturbed.sample_id, original.sample_id)
        self.assertEqual(perturbed.native_label, original.native_label)
        for index in selected:
            self.assertEqual(perturbed.features[index], 0)
            self.assertIsNone(perturbed.category_tokens[index])

    def test_selection_is_stable_across_sample_order_and_projection(self):
        first = _sample("sample-1")
        second = _sample("sample-2")
        forward, forward_report = apply_deterministic_field_missingness(
            (first, second),
            fraction=0.5,
            seed=42,
            eligible_feature_ids=("a", "b", "c", "d"),
        )
        reverse, reverse_report = apply_deterministic_field_missingness(
            (second, first),
            fraction=0.5,
            seed=42,
            eligible_feature_ids=("a", "b", "c", "d"),
        )

        self.assertEqual(
            forward_report["selected_assignments_sha256"],
            reverse_report["selected_assignments_sha256"],
        )
        self.assertEqual(forward[0].missing_mask[4], False)
        self.assertEqual(reverse[1].missing_mask, forward[0].missing_mask)

    def test_rejects_invalid_fraction_and_missing_eligible_feature(self):
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            apply_deterministic_field_missingness(
                (_sample(),), fraction=1.1, seed=1
            )
        with self.assertRaisesRegex(ValueError, "absent from sample"):
            apply_deterministic_field_missingness(
                (_sample(),),
                fraction=0.3,
                seed=1,
                eligible_feature_ids=("a", "missing"),
            )

    def test_structured_groups_are_a_frozen_partition_and_hide_whole_group(self):
        flattened = tuple(
            feature
            for fields in STRUCTURED_MISSINGNESS_GROUPS.values()
            for feature in fields
        )
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            set(flattened),
            {spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.model_features},
        )
        self.assertNotIn("DNS_QUERY_ID", flattened)
        original = _netflow_sample()
        (perturbed,), report = apply_structured_field_group_missingness(
            (original,), group_name="tcp_state_retransmission"
        )
        selected = set(
            STRUCTURED_MISSINGNESS_GROUPS["tcp_state_retransmission"]
        )
        for name, missing in zip(perturbed.feature_ids, perturbed.missing_mask):
            self.assertEqual(missing, name in selected)
        self.assertEqual(report["hidden_fields_per_sample"], len(selected))
        self.assertEqual(report["training_data_perturbed"], False)
        self.assertEqual(perturbed.sample_id, original.sample_id)
        _, reversed_report = apply_structured_field_group_missingness(
            (_netflow_sample("netflow-2"), original),
            group_name="tcp_state_retransmission",
        )
        _, forward_report = apply_structured_field_group_missingness(
            (original, _netflow_sample("netflow-2")),
            group_name="tcp_state_retransmission",
        )
        self.assertEqual(
            forward_report["selected_assignments_sha256"],
            reversed_report["selected_assignments_sha256"],
        )

    def test_structured_missingness_rejects_post_hoc_group_names(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            apply_structured_field_group_missingness(
                (_netflow_sample(),), group_name="chosen_after_test"
            )


if __name__ == "__main__":
    unittest.main()
