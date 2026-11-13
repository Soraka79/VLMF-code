import unittest

import numpy as np

from flowfm.training.template_tree import (
    TemplateTreeClassifier,
    TemplateTreeConfig,
)


class TemplateTreeClassifierTests(unittest.TestCase):
    def test_components_are_finite_normalized_and_class_aligned(self):
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.9, 1.0],
                [1.0, 0.9],
                [0.4, 0.8],
                [0.5, 0.9],
            ],
            dtype=np.float32,
        )
        targets = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        model = TemplateTreeClassifier(
            class_count=3,
            random_state=17,
            config=TemplateTreeConfig(min_samples_leaf=1),
        ).fit(features, targets)

        components = model.predict_proba_components(features)

        self.assertEqual(
            set(components), {"template_only", "shallow_tree", "template_tree"}
        )
        for probabilities in components.values():
            self.assertEqual(probabilities.shape, (6, 3))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue((probabilities >= 0).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertLessEqual(model.model_summary()["tree_depth"], 8)

    def test_fit_requires_every_declared_class(self):
        model = TemplateTreeClassifier(class_count=3, random_state=42)

        with self.assertRaisesRegex(ValueError, "every declared class"):
            model.fit(
                np.asarray([[0.0], [1.0]], dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
            )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "template_weight"):
            TemplateTreeConfig(template_weight=1.1)


if __name__ == "__main__":
    unittest.main()
