import unittest

from flowfm.data.formal_cohort import (
    FORMAL_DATASETS,
    FORMAL_REVISION_COHORT_PROTOCOL_VERSION,
    HISTORY_POLICY_DISCLOSED_RESTART,
    HISTORY_POLICY_EXCLUDE,
    REVISION_ACTIVE_COHORT_COUNT,
    REVISION_COHORT_SEEDS,
    get_formal_dataset_spec,
    revision_cohort_key,
)


class RevisionCohortSpecTests(unittest.TestCase):
    def test_revision_specs_change_only_seed_name_and_policy(self):
        for base_key in ("nf_bot", "nf_cse", "nf_ton"):
            base = get_formal_dataset_spec(base_key)
            names = set()
            for index, seed in enumerate(REVISION_COHORT_SEEDS, start=1):
                spec = get_formal_dataset_spec(revision_cohort_key(base_key, index))
                self.assertEqual(spec.split_seed, seed)
                self.assertEqual(spec.protocol_version, FORMAL_REVISION_COHORT_PROTOCOL_VERSION)
                self.assertEqual(spec.historical_exposure_policy, HISTORY_POLICY_DISCLOSED_RESTART)
                for field in (
                    "dataset_id",
                    "task_coarse_labels",
                    "task_mapping_version",
                    "audit_config_relative",
                    "audit_report_relative",
                    "leakage_group_namespace",
                    "rare_coarse_labels",
                    "max_samples",
                ):
                    self.assertEqual(getattr(spec, field), getattr(base, field))
                names.add(spec.cohort_name)
            self.assertEqual(len(names), len(REVISION_COHORT_SEEDS))

    def test_original_formal_specs_are_unchanged(self):
        self.assertEqual(get_formal_dataset_spec("nf_bot").split_seed, 20260727)
        self.assertEqual(get_formal_dataset_spec("nf_cse").split_seed, 20260731)
        self.assertEqual(get_formal_dataset_spec("nf_ton").split_seed, 20260729)
        for key in ("nf_bot", "nf_cse", "nf_ton", "nf_unsw"):
            self.assertEqual(
                get_formal_dataset_spec(key).historical_exposure_policy,
                HISTORY_POLICY_EXCLUDE,
            )

    def test_seed_reserve_is_declared(self):
        self.assertEqual(len(REVISION_COHORT_SEEDS), 5)
        self.assertEqual(REVISION_ACTIVE_COHORT_COUNT, 3)
        self.assertEqual(len([key for key in FORMAL_DATASETS if "_rev" in key]), 15)


if __name__ == "__main__":
    unittest.main()
