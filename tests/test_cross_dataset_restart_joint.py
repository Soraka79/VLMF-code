import unittest

from scripts.summarize_cross_dataset_restart_joint import (
    _DEPLOYMENT_ROOT,
    _INDEPENDENT_DATASETS,
    _RESTART_CLAIM_SCOPE,
    _RESTART_DATASET,
    _RESTART_HISTORY_POLICY,
    deployment_rare_zero_failures,
    verify_restart_disclosure,
    verify_restart_test_access_recovery,
)
from scripts.run_formal_deployment_baseline_v2 import (
    _CONFIG as V2_CONFIG,
    _DEVELOPMENT_SELECTION,
)


def valid_disclosure_payloads():
    plan = {
        "dataset": "nf_unsw_restart",
        "dataset_id": "NF-UNSW-NB15-v2",
        "claim_scope": _RESTART_CLAIM_SCOPE,
        "historical_exposure_policy": _RESTART_HISTORY_POLICY,
        "historically_independent": False,
    }
    feasibility = {
        "status": "eligible_for_disclosed_full_source_restart",
        "claim_scope": _RESTART_CLAIM_SCOPE,
        "historically_independent_confirmatory_target": False,
        "historical_overlap_expected": True,
    }
    overlap = {
        "verified_zero_internal_overlap": True,
        "verified_zero_historical_overlap": False,
        "internal_split_overlap_counts": {
            "sample_id": {
                "train__val": 0,
                "train__test": 0,
                "val__test": 0,
            }
        },
        "history_identity_overlap_counts": {"sample_id": 37526},
    }
    return plan, feasibility, overlap


class CrossDatasetRestartJointTests(unittest.TestCase):
    def test_restart_recovery_requires_transparent_access_accounting(self):
        receipt = {
            "sealed_test_claim_count": 1,
            "sealed_artifact_hash_read_attempt_count": 3,
            "sealed_artifact_hash_read_counting_unit": (
                "one_complete_records_manifest_features_sha256_pass"
            ),
            "pre_materialization_failure_count": 1,
            "diagnostic_hash_verification_count": 1,
            "successful_test_materialization_count": 1,
            "pre_materialization_recovery": {
                "path": "/tmp/recovery.json",
                "sha256": "0" * 64,
                "bytes": 1,
            },
        }

        result = verify_restart_test_access_recovery(receipt)

        self.assertEqual(result["sealed_artifact_hash_read_attempt_count"], 3)
        self.assertEqual(result["pre_materialization_failure_count"], 1)

    def test_restart_recovery_rejects_hidden_failed_attempt(self):
        receipt = {
            "sealed_test_claim_count": 1,
            "sealed_artifact_hash_read_attempt_count": 2,
            "sealed_artifact_hash_read_counting_unit": (
                "one_complete_records_manifest_features_sha256_pass"
            ),
            "pre_materialization_failure_count": 0,
            "diagnostic_hash_verification_count": 1,
            "successful_test_materialization_count": 1,
            "pre_materialization_recovery": {
                "path": "/tmp/recovery.json",
                "sha256": "0" * 64,
                "bytes": 1,
            },
        }

        with self.assertRaisesRegex(RuntimeError, "accounting"):
            verify_restart_test_access_recovery(receipt)

    def test_deployment_zero_recall_alert_preserves_support(self):
        class_counts = {
            "dos": 299,
            "scan": 909,
            "backdoor": 28,
            "malware": 13,
        }
        conditions = {}
        for condition in ("clean", "missing_30pct", "missing_50pct"):
            conditions[condition] = {
                "template_tree": {
                    "class_counts": class_counts,
                    "per_class_recall": {
                        "dos": 0.3,
                        "scan": 0.7,
                        "backdoor": 0.0,
                        "malware": 0.0,
                    },
                }
            }

        failures = deployment_rare_zero_failures(
            {"conditions": conditions}, "pct-01-seed-17"
        )

        self.assertEqual(len(failures), 6)
        self.assertEqual({row["label"] for row in failures}, {"backdoor", "malware"})
        self.assertEqual(
            {row["support"] for row in failures if row["label"] == "malware"},
            {13},
        )

    def test_v2_deployment_profile_is_validation_selected_and_all_feature(self):
        self.assertEqual(
            _DEPLOYMENT_ROOT.name,
            "nf-unsw-restart-template-tree-deployment-v2",
        )
        self.assertIsNone(V2_CONFIG.max_features)
        self.assertEqual(V2_CONFIG.max_depth, 8)
        self.assertEqual(
            _DEVELOPMENT_SELECTION["selected_profile"], "all_d8_l128"
        )
        self.assertFalse(
            _DEVELOPMENT_SELECTION["sealed_test_files_read"]
        )

    def test_restart_is_excluded_from_independent_datasets(self):
        self.assertNotIn(_RESTART_DATASET, _INDEPENDENT_DATASETS)
        self.assertEqual(len(_INDEPENDENT_DATASETS), 3)

    def test_disclosure_requires_internal_disjointness_and_historical_reuse(self):
        result = verify_restart_disclosure(*valid_disclosure_payloads())

        self.assertFalse(result["historically_independent"])
        self.assertFalse(result["eligible_for_primary_cross_dataset_inference"])
        self.assertTrue(result["verified_zero_internal_overlap"])
        self.assertFalse(result["verified_zero_historical_overlap"])

    def test_disclosure_rejects_independent_claim(self):
        plan, feasibility, overlap = valid_disclosure_payloads()
        plan["historically_independent"] = True

        with self.assertRaisesRegex(RuntimeError, "frozen disclosure"):
            verify_restart_disclosure(plan, feasibility, overlap)

    def test_disclosure_rejects_internal_overlap(self):
        plan, feasibility, overlap = valid_disclosure_payloads()
        overlap["internal_split_overlap_counts"]["sample_id"][
            "train__test"
        ] = 1

        with self.assertRaisesRegex(RuntimeError, "internal split overlap"):
            verify_restart_disclosure(plan, feasibility, overlap)

    def test_disclosure_rejects_missing_historical_overlap_evidence(self):
        plan, feasibility, overlap = valid_disclosure_payloads()
        overlap["history_identity_overlap_counts"]["sample_id"] = 0

        with self.assertRaisesRegex(RuntimeError, "historical reuse"):
            verify_restart_disclosure(plan, feasibility, overlap)


if __name__ == "__main__":
    unittest.main()
