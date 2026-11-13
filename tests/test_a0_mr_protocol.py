import unittest

import numpy as np

from flowfm.training.a0_mr_protocol import (
    FORMAL_CONDITIONS,
    fixed_a0_policies,
    frozen_protocol_hash,
    frozen_protocol_payload,
    select_a0_mr_validation,
)


class A0MRProtocolTests(unittest.TestCase):
    def _outputs(self):
        targets = np.tile(np.arange(4, dtype=np.int64), 5)
        perfect = np.eye(4, dtype=np.float64)[targets]
        return {
            condition: {
                "xgboost": perfect,
                "schemaflow_pretrained": perfect,
                "targets": targets,
            }
            for condition in FORMAL_CONDITIONS
        }

    def test_frozen_payload_is_complete_and_hash_is_stable(self):
        payload = frozen_protocol_payload()

        self.assertEqual(payload["fixed_a0"]["alpha_xgboost"], 0.55)
        self.assertEqual(payload["pooling_methods"], ["linear", "log"])
        self.assertEqual(len(payload["alpha_grid"]), 21)
        self.assertEqual(len(frozen_protocol_hash()), 64)

    def test_no_validation_gain_falls_back_to_fixed_a0(self):
        result = select_a0_mr_validation(
            self._outputs(), ("benign", "dos", "scan", "malware")
        )

        self.assertFalse(result["routing_gate"]["passed"])
        expected = fixed_a0_policies()
        for condition, policy in result["locked_policy"]["policies"].items():
            self.assertEqual(policy["method"], expected[condition]["method"])
            self.assertEqual(policy["alpha"], expected[condition]["alpha"])
        self.assertFalse(result["test_samples_materialized"])
        self.assertFalse(result["test_metrics_used"])

    def test_condition_order_is_part_of_the_protocol(self):
        outputs = self._outputs()
        reordered = dict(reversed(tuple(outputs.items())))

        with self.assertRaisesRegex(ValueError, "frozen condition order"):
            select_a0_mr_validation(
                reordered, ("benign", "dos", "scan", "malware")
            )


if __name__ == "__main__":
    unittest.main()
