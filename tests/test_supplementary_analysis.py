import unittest

import numpy as np
from sklearn.metrics import f1_score

from flowfm.training.supplementary_analysis import (
    macro_f1,
    policy_agreement,
    stratified_group_halves,
    stratified_group_subsample,
    validation_controls,
    wilson_interval,
)
from flowfm.training.vlmf_strengthening import ALPHA_GRID, MAIN_CONDITIONS


def _groups(rng, count=600, classes=4):
    sizes = rng.integers(1, 6, size=count)
    group_ids = np.repeat([f"g{i}" for i in range(count)], sizes)
    labels = np.repeat(rng.integers(0, classes, size=count), sizes)
    labels[:3] = classes - 1
    return group_ids, labels


class WilsonTests(unittest.TestCase):
    def test_known_values_and_edges(self):
        low, high = wilson_interval(13, 13)
        self.assertAlmostEqual(low, 0.7719, places=3)
        self.assertEqual(high, 1.0)
        low, high = wilson_interval(0, 13)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2281, places=3)
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=4)
        self.assertAlmostEqual(high, 0.7634, places=4)
        self.assertIsNone(wilson_interval(0, 0))
        with self.assertRaises(ValueError):
            wilson_interval(5, 3)


class MacroF1Tests(unittest.TestCase):
    def test_matches_sklearn_including_absent_classes(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            probabilities = rng.random((500, 5))
            targets = rng.integers(0, 4, size=500)
            expected = f1_score(targets, probabilities.argmax(1), labels=range(5),
                                average="macro", zero_division=0)
            self.assertAlmostEqual(macro_f1(probabilities, targets, 5), expected, places=12)


class SamplingTests(unittest.TestCase):
    def test_subsample_keeps_whole_groups_and_every_class(self):
        rng = np.random.default_rng(1)
        group_ids, labels = _groups(rng)
        index = stratified_group_subsample(group_ids, labels, 100, np.random.default_rng(2))
        chosen = set(group_ids[index])
        self.assertEqual(set(np.flatnonzero(np.isin(group_ids, list(chosen)))), set(index))
        self.assertEqual(set(labels[index]), set(labels))
        self.assertLess(abs(index.size - 100), 60)

    def test_halves_are_group_disjoint_and_cover_all_rows(self):
        rng = np.random.default_rng(3)
        group_ids, labels = _groups(rng)
        first, second = stratified_group_halves(group_ids, labels, np.random.default_rng(4))
        self.assertEqual(first.size + second.size, labels.size)
        self.assertFalse(set(group_ids[first]) & set(group_ids[second]))
        self.assertLess(abs(first.size - second.size), labels.size * 0.1)


class ControlTests(unittest.TestCase):
    def test_controls_follow_the_frozen_tie_order(self):
        search = {}
        for offset, name in enumerate(MAIN_CONDITIONS):
            candidates = []
            for method in ("linear", "log"):
                for alpha in ALPHA_GRID:
                    score = 0.5 - abs(alpha - (0.3 + 0.2 * offset)) + (0.01 if method == "log" else 0)
                    candidates.append({"method": method, "alpha": alpha, "macro_f1": score,
                                       "mcc": score, "balanced_accuracy": score})
            search[name] = {"methods": ["linear", "log"], "alpha_grid": list(ALPHA_GRID),
                            "candidates": candidates}
        controls = validation_controls({"random_condition_selection": {"linear_or_log": search}})
        always = controls["always_conditioned_linear_or_log"]
        self.assertEqual([always[n]["alpha"] for n in MAIN_CONDITIONS], [0.3, 0.5, 0.7])
        self.assertEqual({always[n]["method"] for n in MAIN_CONDITIONS}, {"log"})
        self.assertEqual(controls["global_validation_fusion"]["clean"],
                         {"method": "log", "alpha": 0.5})

    def test_policy_agreement(self):
        first = {"a": {"method": "linear", "alpha": 0.5}, "b": {"method": "log", "alpha": 0.7}}
        second = {"a": {"method": "linear", "alpha": 0.5}, "b": {"method": "log", "alpha": 0.75}}
        self.assertEqual(policy_agreement(first, second, ("a", "b")), 0.5)


if __name__ == "__main__":
    unittest.main()
