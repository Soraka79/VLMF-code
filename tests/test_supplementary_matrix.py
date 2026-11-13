import unittest

from flowfm.data.contracts import FlowSample
from flowfm.data.exporter_missingness import EXPORTER_PROFILES, build_supplementary_resolver
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
from flowfm.training.a0_mr_protocol import FORMAL_CONDITIONS
from flowfm.training.supplementary_protocol import TEST_CONDITIONS, missing_mask_matrix
from scripts.run_formal_neural_model import build_parser as neural_parser
from scripts.run_supplementary_matrix import (
    _active_cohorts,
    _iter_conditions,
    _settings,
    build_parser,
)


def _samples(count=6):
    schema = NF_CSE_CIC_IDS2018_V2_SCHEMA
    categorical = schema.categorical_mask
    return tuple(
        FlowSample(
            features=tuple(1 for _ in schema.model_features),
            feature_ids=schema.model_feature_names,
            categorical_mask=categorical,
            missing_mask=tuple(False for _ in schema.model_features),
            category_tokens=tuple("n:1" if flag else None for flag in categorical),
            binary_label="attack",
            coarse_label="dos",
            native_label="DoS",
            dataset_id="fixture",
            sample_id=f"s-{index}",
            feature_key=f"k-{index}",
            conflict_flag=False,
            conflict_group_size=1,
            conflict_label_count=1,
            source_file="fixture.csv",
            source_row_id=f"r-{index}",
        )
        for index in range(count)
    )


class SupplementaryRunnerTests(unittest.TestCase):
    def test_design_is_three_cohorts_per_dataset_and_one_model_seed(self):
        self.assertEqual(len(_active_cohorts()), 9)
        self.assertEqual([s["train_percent"] for s in _settings()], [1, 5, 10])
        self.assertEqual({s["model_seed"] for s in _settings()}, {17})

    def test_condition_iterator_follows_the_frozen_test_order(self):
        names = [name for name, _, _ in _iter_conditions(_samples(), 20261018, TEST_CONDITIONS)]
        self.assertEqual(names, list(TEST_CONDITIONS))

    def test_resolver_reproduces_the_eight_paper_conditions(self):
        resolvers = (
            build_supplementary_resolver(),
            build_supplementary_resolver(tuple(EXPORTER_PROFILES)),
        )
        for name, values, _ in _iter_conditions(_samples(), 20261018, FORMAL_CONDITIONS):
            mask, order = missing_mask_matrix(values)
            for resolver in resolvers:
                self.assertEqual(set(resolver.resolve_matrix(mask, order).tolist()), {name})

    def test_parsers_accept_revision_cohorts_only_where_intended(self):
        self.assertEqual(build_parser().parse_args(["plan", "--dataset", "nf_ton_rev1"]).dataset, "nf_ton_rev1")
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["plan", "--dataset", "nf_ton"])
        args = neural_parser().parse_args([
            "--dataset", "nf_ton_rev1", "--cohort-dir", "/x", "--view-dir", "/x",
            "--matrix-root", "/x", "--model-seed", "17", "--train-percent", "1",
            "--model", "schemaflow_pretrained",
        ])
        self.assertEqual(args.dataset, "nf_ton_rev1")


if __name__ == "__main__":
    unittest.main()
