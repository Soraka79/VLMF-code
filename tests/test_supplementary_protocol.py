import unittest

import numpy as np

from flowfm.data.exporter_missingness import EXPORTER_PROFILES
from flowfm.training.a0_mr_protocol import (
    FIXED_A0_ALPHA_XGBOOST,
    FORMAL_CONDITIONS,
    frozen_protocol_hash,
)
from flowfm.training.probability_fusion import pool_probabilities
from flowfm.training.safe_fusion import apply_scalar_temperature
from flowfm.training.supplementary_protocol import (
    COMPOUND_CONDITIONS,
    INTERMITTENT_CONDITIONS,
    PROFILE_CONDITIONS,
    TEST_CONDITIONS,
    VALIDATION_CONDITIONS,
    dispatch_probabilities,
    frozen_supplementary_hash,
    frozen_supplementary_payload,
    select_registered_profiles,
)


def _probabilities(rows, classes, seed):
    values = np.random.default_rng(seed).random((rows, classes)) + 0.05
    return values / values.sum(axis=1, keepdims=True)


def _profile_outputs(rows=300, classes=3):
    targets = np.arange(rows) % classes
    outputs = {}
    for offset, condition in enumerate(PROFILE_CONDITIONS):
        tree = _probabilities(rows, classes, offset)
        tree[np.arange(rows), targets] += 0.6
        outputs[condition] = {
            "xgboost": tree / tree.sum(axis=1, keepdims=True),
            "schemaflow_pretrained": _probabilities(rows, classes, offset + 10),
            "targets": targets,
        }
    return outputs


def _selection(passed, candidate="missing_rate_conditioned_linear"):
    return {
        "protocol_hash": frozen_protocol_hash(),
        "routing_gate": {"passed": passed, "candidate_name": candidate},
        "locked_policy": {
            "policies": {
                name: {"method": "linear", "alpha": 0.8, "selected_on": name}
                for name in FORMAL_CONDITIONS
            }
        },
    }


class ConditionSetTests(unittest.TestCase):
    def test_condition_sets_extend_the_frozen_eight(self):
        self.assertEqual(VALIDATION_CONDITIONS[:8], FORMAL_CONDITIONS)
        self.assertEqual(TEST_CONDITIONS[: len(VALIDATION_CONDITIONS)], VALIDATION_CONDITIONS)
        self.assertEqual(len(PROFILE_CONDITIONS), len(EXPORTER_PROFILES))
        self.assertEqual(len(INTERMITTENT_CONDITIONS), 1)
        self.assertEqual(len(COMPOUND_CONDITIONS), 3)
        self.assertEqual(len(TEST_CONDITIONS), 15)
        self.assertEqual(len(set(TEST_CONDITIONS)), 15)

    def test_protocol_hash_is_stable_and_records_the_base_hash(self):
        self.assertEqual(frozen_supplementary_hash(), frozen_supplementary_hash())
        payload = frozen_supplementary_payload()
        self.assertEqual(payload["base_a0_mr_protocol_hash"], frozen_protocol_hash())
        self.assertEqual(payload["test_conditions"], list(TEST_CONDITIONS))
        self.assertIn("post_evaluation_analyses", payload)


class ProfileSelectionTests(unittest.TestCase):
    def test_failed_gate_registers_fixed_fusion_for_profiles(self):
        result = select_registered_profiles(_profile_outputs(), _selection(False), ("a", "b", "c"))
        for name in PROFILE_CONDITIONS:
            policy = result["registered_policies"][name]
            self.assertEqual((policy["method"], policy["alpha"]), ("linear", FIXED_A0_ALPHA_XGBOOST))
            self.assertEqual(policy["selected_on"], "frozen_fallback")
        self.assertEqual(result["protocol_hash"], frozen_supplementary_hash())

    def test_passed_gate_selects_profiles_within_the_admitted_family(self):
        result = select_registered_profiles(_profile_outputs(), _selection(True), ("a", "b", "c"))
        self.assertEqual(result["admitted_methods"], ["linear"])
        for name in PROFILE_CONDITIONS:
            policy = result["registered_policies"][name]
            self.assertEqual(policy["method"], "linear")
            self.assertEqual(policy["selected_on"], name)
        for name in FORMAL_CONDITIONS:
            self.assertEqual(result["registered_policies"][name]["alpha"], 0.8)

    def test_changed_base_protocol_is_rejected(self):
        selection = _selection(True)
        selection["protocol_hash"] = "changed"
        with self.assertRaises(ValueError):
            select_registered_profiles(_profile_outputs(), selection, ("a", "b", "c"))


class DispatchTests(unittest.TestCase):
    def test_records_follow_their_resolved_policy_and_fallback(self):
        tree, neural = _probabilities(6, 3, 1), _probabilities(6, 3, 2)
        names = np.asarray(["clean", "clean", "", "missing_30pct", "", "clean"], dtype=object)
        policies = {
            "clean": {"method": "log", "alpha": 0.7},
            "missing_30pct": {"method": "linear", "alpha": 0.2},
        }
        pooled, report = dispatch_probabilities(
            tree, neural, names, policies, temperatures={"clean": 2.0, "missing_30pct": 1.0}
        )
        clean = [0, 1, 5]
        expected = apply_scalar_temperature(
            pool_probabilities(tree[clean], neural[clean], alpha=0.7, method="log"), 2.0
        )
        np.testing.assert_allclose(pooled[clean], expected)
        fallback = [2, 4]
        np.testing.assert_allclose(
            pooled[fallback],
            pool_probabilities(tree[fallback], neural[fallback], alpha=FIXED_A0_ALPHA_XGBOOST),
        )
        np.testing.assert_allclose(
            pooled[[3]], pool_probabilities(tree[[3]], neural[[3]], alpha=0.2)
        )
        self.assertEqual(report["resolved_counts"], {"clean": 3, "fallback": 2, "missing_30pct": 1})

    def test_resolved_condition_without_policy_is_rejected(self):
        tree, neural = _probabilities(2, 3, 1), _probabilities(2, 3, 2)
        with self.assertRaises(ValueError):
            dispatch_probabilities(tree, neural, np.asarray(["clean", "x"], dtype=object),
                                   {"clean": {"method": "linear", "alpha": 0.5}})


if __name__ == "__main__":
    unittest.main()
