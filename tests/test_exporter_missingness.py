import unittest

import numpy as np

from flowfm.data.contracts import FlowSample
from flowfm.data.exporter_missingness import (
    COMPOUND_PROFILES,
    EXPORTER_PROFILES,
    MODEL_FIELDS,
    ConditionResolver,
    ConditionRule,
    apply_intermittent_missingness,
    apply_profile_missingness,
    build_supplementary_resolver,
)
from flowfm.data.missingness import (
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_deterministic_field_missingness,
    apply_structured_field_group_missingness,
)
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import (
    NF_CSE_CIC_IDS2018_V2_SCHEMA,
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


def _missing(sample):
    return {
        name for name, missing in zip(sample.feature_ids, sample.missing_mask) if missing
    }


class ProfileDefinitionTests(unittest.TestCase):
    def test_profile_sizes_are_frozen(self):
        self.assertEqual(len(MODEL_FIELDS), 40)
        self.assertEqual(len(EXPORTER_PROFILES["nf_v1_basic_template"]), 30)
        self.assertEqual(len(EXPORTER_PROFILES["dpi_disabled"]), 4)
        self.assertEqual(len(EXPORTER_PROFILES["client_to_server_only"]), 11)
        self.assertEqual(len(COMPOUND_PROFILES["compound_tcp_state_packet_length"]), 20)
        self.assertEqual(len(COMPOUND_PROFILES["compound_transport_application"]), 9)
        self.assertEqual(len(COMPOUND_PROFILES["compound_dpi_client_to_server"]), 13)

    def test_dpi_profile_keeps_header_derived_icmp_fields(self):
        self.assertNotIn("ICMP_TYPE", EXPORTER_PROFILES["dpi_disabled"])
        self.assertNotIn("ICMP_IPV4_TYPE", EXPORTER_PROFILES["dpi_disabled"])


class ProfileApplicationTests(unittest.TestCase):
    def test_profile_hides_declared_fields_only(self):
        (perturbed,), report = apply_profile_missingness(
            (_netflow_sample(),), profile="client_to_server_only"
        )
        self.assertEqual(_missing(perturbed), set(EXPORTER_PROFILES["client_to_server_only"]))
        self.assertEqual(report["samples_with_hidden_fields"], 1)
        self.assertFalse(report["training_data_perturbed"])
        for name, value, token in zip(
            perturbed.feature_ids, perturbed.features, perturbed.category_tokens
        ):
            if name in EXPORTER_PROFILES["client_to_server_only"]:
                self.assertEqual(value, 0)
                self.assertIsNone(token)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_profile_missingness((_netflow_sample(),), profile="transport_protocol")

    def test_intermittent_profile_is_deterministic_and_partial(self):
        samples = tuple(_netflow_sample(f"s-{index}") for index in range(2000))
        first, report = apply_intermittent_missingness(
            samples, profile="intermittent_application_metadata", seed=7
        )
        second, _ = apply_intermittent_missingness(
            samples, profile="intermittent_application_metadata", seed=7
        )
        self.assertEqual(
            [sample.missing_mask for sample in first],
            [sample.missing_mask for sample in second],
        )
        hit = report["samples_with_hidden_fields"]
        self.assertGreater(hit, 300)
        self.assertLess(hit, 500)
        group = set(STRUCTURED_MISSINGNESS_GROUPS["application_protocol_metadata"])
        for sample in first:
            self.assertIn(_missing(sample), (set(), group))


class ResolverTests(unittest.TestCase):
    def test_resolver_reproduces_frozen_generator_labels(self):
        resolver = build_supplementary_resolver()
        samples = tuple(_netflow_sample(f"s-{index}") for index in range(50))
        cases = [("clean", samples)]
        for name, fraction in (("missing_30pct", 0.3), ("missing_50pct", 0.5)):
            masked, _ = apply_deterministic_field_missingness(
                samples, fraction=fraction, seed=20261018, eligible_feature_ids=MODEL_FIELDS
            )
            cases.append((name, masked))
        for group in STRUCTURED_MISSINGNESS_GROUPS:
            masked, _ = apply_structured_field_group_missingness(samples, group_name=group)
            cases.append((f"structured__{group}", masked))
        for expected, masked in cases:
            for sample in masked:
                self.assertEqual(resolver.resolve_sample(sample), expected)

    def test_unregistered_profiles_fall_back_and_registered_profiles_resolve(self):
        base = build_supplementary_resolver()
        registered = build_supplementary_resolver(tuple(EXPORTER_PROFILES))
        for profile, fields in EXPORTER_PROFILES.items():
            self.assertIsNone(base.resolve(fields))
            self.assertEqual(registered.resolve(fields), f"profile__{profile}")

    def test_compound_profiles_fall_back_or_collide_with_count_rule(self):
        resolver = build_supplementary_resolver(tuple(EXPORTER_PROFILES))
        self.assertEqual(
            resolver.resolve(COMPOUND_PROFILES["compound_tcp_state_packet_length"]),
            "missing_50pct",
        )
        self.assertIsNone(resolver.resolve(COMPOUND_PROFILES["compound_transport_application"]))
        self.assertIsNone(resolver.resolve(COMPOUND_PROFILES["compound_dpi_client_to_server"]))

    def test_non_model_fields_are_ignored(self):
        resolver = build_supplementary_resolver()
        self.assertEqual(resolver.resolve({"DNS_QUERY_ID"}), "clean")

    def test_matrix_resolution_matches_record_resolution(self):
        resolver = build_supplementary_resolver(("dpi_disabled",))
        index = {name: position for position, name in enumerate(MODEL_FIELDS)}
        rows = []
        for fields in (
            (),
            EXPORTER_PROFILES["dpi_disabled"],
            COMPOUND_PROFILES["compound_transport_application"],
            STRUCTURED_MISSINGNESS_GROUPS["transport_protocol"],
            (),
        ):
            row = np.zeros(len(MODEL_FIELDS), dtype=bool)
            row[[index[name] for name in fields]] = True
            rows.append(row)
        resolved = resolver.resolve_matrix(np.stack(rows))
        self.assertEqual(
            resolved.tolist(),
            ["clean", "profile__dpi_disabled", "", "structured__transport_protocol", "clean"],
        )

    def test_ambiguous_rules_are_rejected(self):
        fields = frozenset(STRUCTURED_MISSINGNESS_GROUPS["transport_protocol"])
        with self.assertRaises(ValueError):
            ConditionResolver(
                [ConditionRule("a", missing_fields=fields), ConditionRule("b", missing_fields=fields)]
            )
        with self.assertRaises(ValueError):
            ConditionResolver(
                [ConditionRule("a", missing_count=12), ConditionRule("b", missing_count=12)]
            )
        with self.assertRaises(ValueError):
            ConditionRule("a", missing_fields=frozenset(), missing_count=3)


if __name__ == "__main__":
    unittest.main()
