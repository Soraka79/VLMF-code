import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_nf_cse_formal_matrix import (
    _BASE_MODELS,
    _FORMAL_CONDITIONS,
    _STRUCTURED_CONDITIONS,
    _budget_payload,
    _formal_conditions,
    _lock_all_selections,
    _record_training_step,
    _save_state,
    _select_formal_policies,
    _settings,
    _sha256_file,
    _write_formal_prediction_lock,
)


def _outputs(row_count=50):
    targets = np.arange(row_count, dtype=np.int64) % 5
    probabilities = np.full((row_count, 5), 0.05, dtype=np.float64)
    probabilities[np.arange(row_count), targets] = 0.8
    return {
        condition: {
            **{model: probabilities.copy() for model in _BASE_MODELS},
            "targets": targets.copy(),
            "sample_order_sha256": "a" * 64,
        }
        for condition in _FORMAL_CONDITIONS
    }


class NFCSEFormalMatrixTests(unittest.TestCase):
    def test_matrix_and_budget_are_predeclared(self):
        settings = _settings()
        self.assertEqual(len(settings), 9)
        self.assertEqual(
            {(value["seed"], value["train_percent"]) for value in settings},
            {
                (17, 1),
                (17, 5),
                (17, 10),
                (42, 1),
                (42, 5),
                (42, 10),
                (20260725, 1),
                (20260725, 5),
                (20260725, 10),
            },
        )
        budget = _budget_payload()
        self.assertEqual(budget["nominal_jobs"]["total"], 45)
        self.assertEqual(budget["reused_completed_jobs"]["total"], 2)
        self.assertEqual(budget["new_jobs"], 43)
        self.assertEqual(budget["test_conditions_per_setting"], 8)
        self.assertTrue(budget["gpu_over_two_hours_is_not_a_stop_condition"])

    def test_formal_conditions_include_random_and_frozen_structured_groups(self):
        from tests.test_missingness import _netflow_sample

        samples = (_netflow_sample("one"), _netflow_sample("two"))
        conditioned, reports = _formal_conditions(samples, 17)
        self.assertEqual(tuple(conditioned), _FORMAL_CONDITIONS)
        self.assertEqual(set(reports), set(_FORMAL_CONDITIONS))
        self.assertEqual(len(_STRUCTURED_CONDITIONS), 5)
        for condition in _STRUCTURED_CONDITIONS:
            self.assertEqual(
                reports[condition]["algorithm"],
                "fixed-netflow-v2-collection-field-groups-v1",
            )
            self.assertEqual(
                tuple(sample.sample_id for sample in conditioned[condition]),
                ("one", "two"),
            )

    def test_structured_policies_follow_frozen_random_routing_fallback(self):
        selection = _select_formal_policies(_outputs())
        self.assertFalse(selection["random_condition_selection"]["routing_gate"]["passed"])
        self.assertEqual(
            set(selection["locked_policy"]["policies"]),
            set(_FORMAL_CONDITIONS),
        )
        for condition in _STRUCTURED_CONDITIONS:
            self.assertEqual(
                selection["locked_policy"]["policies"][condition],
                {
                    "method": "linear",
                    "alpha": 0.55,
                    "selected_on": "frozen_fallback",
                },
            )
        self.assertFalse(selection["test_samples_materialized"])
        self.assertFalse(selection["test_metrics_used"])

    def test_prediction_archive_locks_all_models_before_metrics(self):
        outputs = _outputs(row_count=10)
        groups = tuple(f"group-{index // 2}" for index in range(10))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reloaded, locked_groups, lock = _write_formal_prediction_lock(
                root, outputs, groups
            )
            self.assertEqual(lock["status"], "locked_before_metric_computation")
            self.assertEqual(lock["group_count"], 5)
            self.assertEqual(tuple(reloaded), _FORMAL_CONDITIONS)
            self.assertEqual(locked_groups, groups)
            with np.load(root / "predictions.npz", allow_pickle=False) as archive:
                self.assertIn("ft_transformer__clean", archive.files)
                self.assertIn(
                    "schemaflow_pretrained__structured__transport_protocol",
                    archive.files,
                )
            changed = _outputs(row_count=10)
            changed[_FORMAL_CONDITIONS[-1]]["targets"][0] = 4
            with self.assertRaisesRegex(RuntimeError, "targets differ"):
                _write_formal_prediction_lock(root / "changed", changed, groups)

    def test_training_state_records_native_run_config_layouts(self):
        setting = _settings()[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, config_name in (
                ("schemaflow_scratch", "config.snapshot.json"),
                ("schemaflow_pretrained", "config.snapshot.json"),
                ("ft_transformer", "config.json"),
            ):
                run_dir = root / "runs" / name
                checkpoint = run_dir / "checkpoints" / "epoch.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(name.encode("ascii"))
                (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
                (run_dir / config_name).write_text("{}\n", encoding="utf-8")
                state = {
                    "format_version": 1,
                    "protocol_version": "nf-cse-a0-mr-formal-matrix-v2",
                    **setting,
                    "steps": {},
                }
                _record_training_step(
                    root,
                    setting,
                    state,
                    name,
                    {"checkpoint": str(checkpoint), "run_dir": str(run_dir)},
                    1.0,
                    reused=name != "ft_transformer",
                )
                saved = json.loads(
                    (root / "states" / f"{setting['setting_id']}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    Path(saved["steps"][name]["artifacts"]["config"]["path"]).name,
                    config_name,
                )

    def test_global_barrier_requires_all_nine_selections_before_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for setting in _settings():
                run_dir = root / "selections" / setting["setting_id"]
                run_dir.mkdir(parents=True)
                selection_path = run_dir / "selection.json"
                config_path = run_dir / "config.json"
                selection_path.write_text("{}\n", encoding="utf-8")
                config_path.write_text("{}\n", encoding="utf-8")
                state = {
                    "format_version": 1,
                    "protocol_version": "nf-cse-a0-mr-formal-matrix-v2",
                    **setting,
                    "steps": {
                        "validation_selection": {
                            "status": "completed",
                            "run_dir": str(run_dir),
                            "artifacts": {
                                "selection": {
                                    "path": str(selection_path),
                                    "sha256": _sha256_file(selection_path),
                                    "bytes": selection_path.stat().st_size,
                                },
                                "config": {
                                    "path": str(config_path),
                                    "sha256": _sha256_file(config_path),
                                    "bytes": config_path.stat().st_size,
                                },
                            },
                        }
                    },
                }
                _save_state(root, setting, state)
            first_dir = root / "selections" / _settings()[0]["setting_id"]
            claim = first_dir / "formal_test_access.claim.json"
            claim.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "before the global"):
                _lock_all_selections(root)
            claim.unlink()
            lock = _lock_all_selections(root)
            self.assertEqual(lock["setting_count"], 9)
            self.assertEqual(
                lock["status"],
                "all_nine_validation_selections_locked_before_any_formal_test",
            )
            claim.write_text(json.dumps({"after": "lock"}), encoding="utf-8")
            self.assertEqual(_lock_all_selections(root), lock)


if __name__ == "__main__":
    unittest.main()
