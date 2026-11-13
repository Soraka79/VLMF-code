import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from flowfm.data.formal_cohort import (
    FORMAL_DATASETS,
    HISTORY_POLICY_DISCLOSED_RESTART,
    _normalize_split_reference,
    get_formal_dataset_spec,
    validate_formal_overlap_contract,
)
from flowfm.data.formal_dataset import _skip_groups_exceeding_budget


class FormalCohortProtocolTests(unittest.TestCase):
    def test_split_seeds_and_predeclared_label_views_are_frozen(self):
        self.assertEqual(FORMAL_DATASETS["nf_bot"].split_seed, 20260727)
        self.assertEqual(FORMAL_DATASETS["nf_cse"].split_seed, 20260731)
        self.assertEqual(FORMAL_DATASETS["nf_unsw"].split_seed, 20260728)
        self.assertEqual(
            FORMAL_DATASETS["nf_unsw_restart"].split_seed, 20260801
        )
        self.assertEqual(FORMAL_DATASETS["nf_ton"].split_seed, 20260729)
        self.assertEqual(
            FORMAL_DATASETS["nf_bot"].task_coarse_labels,
            ("benign", "dos", "scan", "malware"),
        )
        self.assertEqual(
            FORMAL_DATASETS["nf_cse"].task_coarse_labels,
            ("benign", "dos", "brute_force", "exploit", "botnet"),
        )
        self.assertEqual(
            FORMAL_DATASETS["nf_ton"].task_coarse_labels,
            ("benign", "dos", "scan", "brute_force", "exploit"),
        )
        self.assertEqual(
            FORMAL_DATASETS["nf_bot"].rare_coarse_labels,
            ("benign", "malware"),
        )
        self.assertEqual(
            FORMAL_DATASETS["nf_cse"].rare_coarse_labels,
            ("brute_force", "exploit", "botnet"),
        )
        restart = FORMAL_DATASETS["nf_unsw_restart"]
        self.assertEqual(
            restart.historical_exposure_policy,
            HISTORY_POLICY_DISCLOSED_RESTART,
        )
        self.assertFalse(restart.requires_zero_historical_overlap)
        self.assertEqual(
            restart.feasibility_status,
            "eligible_for_disclosed_full_source_restart",
        )
        self.assertTrue(
            FORMAL_DATASETS["nf_unsw"].requires_zero_historical_overlap
        )

    def test_restart_overlap_contract_discloses_history_and_requires_isolation(self):
        restart = FORMAL_DATASETS["nf_unsw_restart"]
        overlap = {
            "historical_exposure_policy": HISTORY_POLICY_DISCLOSED_RESTART,
            "historical_overlap_expected": True,
            "historically_independent": False,
            "verified_zero_historical_overlap": False,
            "verified_zero_internal_overlap": True,
        }

        validate_formal_overlap_contract(restart, overlap)

        overlap["verified_zero_internal_overlap"] = False
        with self.assertRaisesRegex(ValueError, "internal overlap"):
            validate_formal_overlap_contract(restart, overlap)

    def test_frozen_unsw_still_requires_zero_historical_overlap(self):
        frozen = FORMAL_DATASETS["nf_unsw"]

        with self.assertRaisesRegex(ValueError, "historical overlap"):
            validate_formal_overlap_contract(
                frozen,
                {
                    "verified_zero_historical_overlap": False,
                    "verified_zero_internal_overlap": True,
                },
            )

    def test_task_label_spaces_filter_native_classes_before_training(self):
        ton = get_formal_dataset_spec("nf_ton").task_label_space

        self.assertNotIn("backdoor", ton.native_labels)
        self.assertNotIn("ransomware", ton.native_labels)
        self.assertEqual(
            ton.coarse_labels,
            ("benign", "dos", "scan", "brute_force", "exploit"),
        )

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset must be one of"):
            get_formal_dataset_spec("unknown")

    def test_provenance_file_references_normalize_to_their_split_directory(self):
        with TemporaryDirectory() as directory:
            split_root = Path(directory) / "data" / "splits" / "NF-CSE"
            known = split_root / "known-split"

            self.assertEqual(
                _normalize_split_reference(str(known), split_root),
                str(known.resolve()),
            )
            self.assertEqual(
                _normalize_split_reference(
                    str(known / "sealed_test.json"), split_root
                ),
                str(known.resolve()),
            )
            self.assertEqual(
                _normalize_split_reference(
                    str(split_root / "unknown-split" / "manifest.json"),
                    split_root,
                ),
                str((split_root / "unknown-split").resolve()),
            )

    def test_few_shot_budget_skips_an_oversized_group_without_splitting_it(self):
        records = tuple(
            SimpleNamespace(group_id=group_id, sample_id=f"{group_id}-{index}")
            for group_id, size in (("oversized", 5), ("small-a", 2), ("small-b", 1))
            for index in range(size)
        )

        eligible, audit = _skip_groups_exceeding_budget(records, 3)

        self.assertEqual(
            {record.group_id for record in eligible}, {"small-a", "small-b"}
        )
        self.assertEqual(audit["skipped_group_count"], 1)
        self.assertEqual(audit["skipped_row_count"], 5)
        self.assertEqual(audit["largest_skipped_group_rows"], 5)
        self.assertEqual(audit["eligible_source_train_rows"], 3)

    def test_few_shot_budget_rejects_when_no_whole_group_fits(self):
        records = tuple(
            SimpleNamespace(group_id="oversized") for _ in range(4)
        )

        with self.assertRaisesRegex(ValueError, "no whole leakage group fits"):
            _skip_groups_exceeding_budget(records, 3)


if __name__ == "__main__":
    unittest.main()
