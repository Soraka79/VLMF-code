import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flowfm.training.safe_fusion import fit_scalar_temperature
from scripts import run_cross_dataset_formal_matrix as formal


class CrossDatasetFormalMatrixTests(unittest.TestCase):
    def _recovery_fixture(self, directory):
        root = Path(directory) / "matrix"
        cohort = Path(directory) / "cohort"
        root.mkdir()
        cohort.mkdir()
        seal_path = cohort / "sealed_test.json"
        seal = {
            "status": "sealed_before_any_restart_model_selection",
            "sealed_test_records_sha256": "1" * 64,
            "sealed_test_manifest_sha256": "2" * 64,
            "sealed_test_features_sha256": "3" * 64,
        }
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        git = {"commit": "e95f718", "dirty": False}
        plan = {
            "protocol_version": formal._PROTOCOL_VERSION,
            "dataset": "nf_unsw_restart",
            "cohort_dir": str(cohort.resolve()),
            "git": git,
            "artifacts": {"cohort_seal": formal._artifact(seal_path)},
        }
        (root / "matrix_plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        lock_path = root / "all_validation_selections.lock.json"
        global_lock = {"setting_count": 9}
        lock_path.write_text(json.dumps(global_lock), encoding="utf-8")
        claim_path = root / "sealed_test_access.claim.json"
        claim = {
            "format_version": 1,
            "protocol_version": formal._PROTOCOL_VERSION,
            "dataset": "nf_unsw_restart",
            "status": "single_test_open_claimed",
            "global_selection_lock_sha256": formal._sha256_file(lock_path),
            "setting_count": 9,
            "sealed_test_open_budget": 1,
            "bootstrap_resamples": 2000,
            "git": git,
        }
        claim_path.write_text(json.dumps(claim), encoding="utf-8")
        pipeline_path = Path(directory) / "pipeline.log"
        pipeline_path.write_text(
            "ValueError: sealed formal test lock is invalid\n", encoding="utf-8"
        )
        incident_path = Path(directory) / "incident.json"
        incident = {
            "format_version": 1,
            "protocol_version": formal._RECOVERY_PROTOCOL_VERSION,
            "status": "authorized_for_single_explicit_recovery",
            "dataset": "nf_unsw_restart",
            "dataset_id": "NF-UNSW-NB15-v2",
            "historically_independent": False,
            "eligible_for_primary_cross_dataset_inference": False,
            "access_accounting_before_recovery": {
                "diagnostic_hash_verification_count": 1,
                "pre_materialization_failure_count": 1,
                "sealed_artifact_hash_read_attempt_count": 2,
                "sealed_artifact_hash_read_counting_unit": (
                    "one_complete_records_manifest_features_sha256_pass"
                ),
                "sealed_test_claim_count": 1,
                "successful_test_materialization_count": 0,
            },
            "authorized_recovery": {
                "additional_sealed_artifact_hash_read_attempt_count": 1,
                "automatic_retry_allowed": False,
                "maximum_successful_test_materialization_count": 1,
                "recovery_scope": (
                    "resume_original_claim_after_pre_materialization_contract_failure"
                ),
            },
            "failure": {
                "exception": "ValueError: sealed formal test lock is invalid",
                "expected_dataset_seal_status": (
                    "sealed_before_any_restart_model_selection"
                ),
                "failed_evaluator_git": git,
                "labels_or_statistics_observed": False,
                "manifest_parsed": False,
                "observed_dataset_seal_status": (
                    "sealed_before_any_restart_model_selection"
                ),
                "predictions_generated": False,
                "reader_hard_coded_status": (
                    "sealed_before_any_model_selection"
                ),
                "records_materialized": False,
                "stage": (
                    "sealed_artifact_hash_verification_before_manifest_parse"
                ),
            },
            "artifacts": {
                "original_claim": formal._artifact(claim_path),
                "global_selection_lock": formal._artifact(lock_path),
                "cohort_seal": formal._artifact(seal_path),
                "pipeline_log": formal._artifact(pipeline_path),
            },
            "sealed_artifacts": formal._expected_recovery_sealed_artifacts(
                plan, seal
            ),
        }
        incident_path.write_text(json.dumps(incident), encoding="utf-8")
        return {
            "root": root,
            "lock_path": lock_path,
            "global_lock": global_lock,
            "claim_path": claim_path,
            "incident_path": incident_path,
            "incident": incident,
        }

    def test_calibration_summary_converts_numpy_scalars_to_json_types(self):
        result = formal._calibration_summary(
            np.float64(0.4),
            np.float64(0.3),
            np.float64(0.2),
            np.float64(0.1),
            np.float64(0.05),
            np.float64(0.05),
        )

        self.assertTrue(result["mean_nll_improved"])
        self.assertTrue(result["mean_brier_improved"])
        self.assertTrue(result["mean_ece_not_worse"])
        json.dumps(result, allow_nan=False)

    def test_matrix_separates_data_split_and_model_seeds(self):
        settings = formal._settings()

        self.assertEqual(len(settings), 9)
        self.assertEqual({row["model_seed"] for row in settings}, {17, 42, 20260725})
        self.assertEqual({row["train_percent"] for row in settings}, {1, 5, 10})
        self.assertEqual(formal._budget("nf_bot")["nominal_jobs"]["total"], 37)

    def test_datasets_use_isolated_versioned_training_views(self):
        self.assertIn(
            "formal_views_v1/nf_bot", str(formal._view_path("nf_bot", 1))
        )
        self.assertIn(
            "formal_views_v1/nf_cse", str(formal._view_path("nf_cse", 1))
        )
        self.assertIn(
            "formal_views_v2/nf_ton", str(formal._view_path("nf_ton", 1))
        )
        self.assertIn(
            "formal_views_restart_v1/nf_unsw_restart",
            str(formal._view_path("nf_unsw_restart", 1)),
        )
        self.assertIn(
            "cross-dataset-formal-restart-matrix-v1/nf_unsw_restart",
            str(formal._default_matrix_root("nf_unsw_restart")),
        )

        with self.assertRaisesRegex(ValueError, "not configured"):
            formal._view_path("nf_unsw", 1)

    def test_scalar_temperature_locks_cannot_change_argmax(self):
        probabilities = np.asarray(
            [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=np.float64
        )
        targets = np.asarray([0, 1, 1], dtype=np.int64)
        lock = fit_scalar_temperature(probabilities, targets)
        by_model = {
            model: {condition: probabilities for condition in formal.FORMAL_CONDITIONS}
            for model in formal._ALL_MODELS
        }
        locks = {
            model: {condition: lock for condition in formal.FORMAL_CONDITIONS}
            for model in formal._ALL_MODELS
        }

        calibrated = formal._apply_temperature_locks(by_model, locks)

        for by_condition in calibrated.values():
            for values in by_condition.values():
                np.testing.assert_array_equal(
                    values.argmax(axis=1), probabilities.argmax(axis=1)
                )

    def test_rare_collapse_keeps_frozen_mean_recall_drop_rule(self):
        metrics = {"xgboost": {}, "a0_mr": {}}
        for condition in formal.FORMAL_CONDITIONS:
            metrics["xgboost"][condition] = {
                "per_class_recall": {"rare": 0.5}
            }
            metrics["a0_mr"][condition] = {
                "per_class_recall": {
                    "rare": 0.47 if condition in formal.RANDOM_CONDITIONS else 0.5
                }
            }

        result = formal._rare_analysis(
            SimpleNamespace(rare_coarse_labels=("rare",)), metrics
        )

        self.assertTrue(result["random_mean_recall_drop_failure"])
        self.assertFalse(result["no_rare_class_recall_collapse"])
        self.assertEqual(
            result["collapse_failures"][0]["type"],
            "rare_random_mean_recall_drop",
        )

    def test_explicit_recovery_validates_pre_materialization_incident(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._recovery_fixture(directory)

            claim, incident = formal._validate_claimed_test_recovery(
                "nf_unsw_restart",
                fixture["root"],
                global_lock_path=fixture["lock_path"],
                global_lock=fixture["global_lock"],
                claim_path=fixture["claim_path"],
                incident_path=fixture["incident_path"],
                bootstrap_resamples=2000,
            )

        self.assertEqual(claim["status"], "single_test_open_claimed")
        self.assertFalse(incident["failure"]["manifest_parsed"])

    def test_explicit_recovery_rejects_observed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._recovery_fixture(directory)
            fixture["incident"]["failure"]["manifest_parsed"] = True
            fixture["incident_path"].write_text(
                json.dumps(fixture["incident"]), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "incident contract"):
                formal._validate_claimed_test_recovery(
                    "nf_unsw_restart",
                    fixture["root"],
                    global_lock_path=fixture["lock_path"],
                    global_lock=fixture["global_lock"],
                    claim_path=fixture["claim_path"],
                    incident_path=fixture["incident_path"],
                    bootstrap_resamples=2000,
                )

    def test_evaluate_parser_requires_explicit_recovery_options(self):
        arguments = formal.build_parser().parse_args(
            [
                "evaluate",
                "--dataset",
                "nf_unsw_restart",
                "--resume-claimed-test",
                "--recovery-incident",
                "incident.json",
            ]
        )

        self.assertTrue(arguments.resume_claimed_test)
        self.assertEqual(arguments.recovery_incident, Path("incident.json"))


if __name__ == "__main__":
    unittest.main()
