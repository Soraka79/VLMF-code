import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.lock_learned_ensemble_protocol import (
    C_GRID,
    EXPERIMENT_NAME,
    META_SEED,
    protocol_payload,
    run_lock,
)


class LearnedEnsembleProtocolTests(unittest.TestCase):
    def test_protocol_freezes_group_aware_inner_validation(self):
        protocol = protocol_payload()
        meta = protocol["meta_validation"]

        self.assertEqual(protocol["status"], "protocol_locked_execution_not_started")
        self.assertEqual(protocol["scope"]["setting_count"], 27)
        self.assertFalse(protocol["scope"]["base_models_retrained"])
        self.assertEqual(meta["unit"], "validation group_id")
        self.assertEqual(meta["splitter"], "StratifiedGroupKFold")
        self.assertEqual(meta["folds"], 5)
        self.assertEqual(meta["seed"], META_SEED)
        self.assertIn("all three conditions", meta["group_exclusivity"])
        self.assertEqual(
            tuple(protocol["comparators"]["logistic_stacking"]["C_grid"]),
            C_GRID,
        )
        self.assertEqual(
            tuple(protocol["comparators"]["learned_gate"]["C_grid"]), C_GRID
        )
        self.assertTrue(
            protocol["model_lock"]["test_access_enabled_only_after_all_54_model_locks"]
        )
        self.assertFalse(
            protocol["future_test_evaluation"]["bootstrap"]["p_value_inferred"]
        )

    def test_gate_is_low_capacity_and_uses_no_raw_telemetry(self):
        gate = protocol_payload()["comparators"]["learned_gate"]

        self.assertEqual(gate["tree_weight"], "sigmoid(w^T z + b)")
        self.assertEqual(gate["pool"], "linear")
        self.assertFalse(gate["raw_telemetry_features_used"])
        self.assertIn(
            "three-way declared-condition one-hot", gate["features"]
        )
        self.assertIn(
            "Jensen-Shannon divergence between branch probabilities",
            gate["features"],
        )

    def test_lock_writes_hash_and_records_zero_data_reads(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / EXPERIMENT_NAME
            result = run_lock(output, repo_root)

            protocol_path = output / "protocol.json"
            sidecar = (output / "protocol.sha256").read_text(
                encoding="ascii"
            ).split()
            observed = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(result["protocol_sha256"], observed)
            self.assertEqual(sidecar, [observed, "protocol.json"])
            self.assertEqual(audit["phase"], "protocol_locked_only")
            self.assertEqual(audit["assertions"]["validation_prediction_files_read"], 0)
            self.assertEqual(audit["assertions"]["sealed_test_files_read"], 0)
            self.assertEqual(audit["assertions"]["meta_models_fitted"], 0)
            self.assertFalse(audit["assertions"]["execution_started"])
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                run_lock(output, repo_root)


if __name__ == "__main__":
    unittest.main()
