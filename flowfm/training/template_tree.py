"""Fixed template matching plus a shallow decision tree for CPU deployment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class TemplateTreeConfig:
    max_depth: int = 8
    max_leaf_nodes: int = 128
    min_samples_leaf: int = 2
    max_features: str | None = "sqrt"
    template_temperature: float = 0.05
    template_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_leaf_nodes", "min_samples_leaf"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_features not in (None, "sqrt", "log2"):
            raise ValueError("max_features must be None, sqrt, or log2")
        if not np.isfinite(self.template_temperature) or self.template_temperature <= 0:
            raise ValueError("template_temperature must be positive and finite")
        if (
            not np.isfinite(self.template_weight)
            or self.template_weight < 0
            or self.template_weight > 1
        ):
            raise ValueError("template_weight must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "template_temperature": self.template_temperature,
            "template_weight": self.template_weight,
        }


class TemplateTreeClassifier:
    """Combine class-centroid templates with a bounded decision tree."""

    def __init__(
        self,
        *,
        class_count: int,
        random_state: int,
        config: TemplateTreeConfig | None = None,
    ) -> None:
        if type(class_count) is not int or class_count < 2:
            raise ValueError("class_count must be an integer of at least two")
        if type(random_state) is not int:
            raise TypeError("random_state must be an integer")
        self.class_count = class_count
        self.random_state = random_state
        self.config = config or TemplateTreeConfig()
        self.templates_: np.ndarray | None = None
        self.tree_: DecisionTreeClassifier | None = None

    @staticmethod
    def _matrix(values, *, expected_columns=None) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("features must be a non-empty rank-2 matrix")
        if expected_columns is not None and matrix.shape[1] != expected_columns:
            raise ValueError("feature dimension differs from fitted templates")
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite")
        return matrix

    def fit(self, features, targets) -> "TemplateTreeClassifier":
        matrix = self._matrix(features)
        labels = np.asarray(targets)
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("targets must align with feature rows")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("targets must contain integer class IDs")
        labels = labels.astype(np.int64, copy=False)
        if np.any(labels < 0) or np.any(labels >= self.class_count):
            raise ValueError("target class ID is outside the declared label space")
        if set(labels.tolist()) != set(range(self.class_count)):
            raise ValueError("every declared class requires a training template")

        templates = np.empty(
            (self.class_count, matrix.shape[1]), dtype=np.float32
        )
        for class_id in range(self.class_count):
            templates[class_id] = matrix[labels == class_id].mean(
                axis=0, dtype=np.float64
            )
        tree = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            class_weight="balanced",
            random_state=self.random_state,
        )
        tree.fit(matrix, labels)
        if not np.array_equal(
            tree.classes_, np.arange(self.class_count, dtype=np.int64)
        ):
            raise RuntimeError("decision tree did not preserve the class order")
        self.templates_ = templates
        self.tree_ = tree
        return self

    def _require_fitted(self) -> tuple[np.ndarray, DecisionTreeClassifier]:
        if self.templates_ is None or self.tree_ is None:
            raise RuntimeError("template-tree classifier is not fitted")
        return self.templates_, self.tree_

    def _template_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        templates, _ = self._require_fitted()
        distances = np.empty((len(matrix), self.class_count), dtype=np.float64)
        for class_id, template in enumerate(templates):
            difference = matrix - template
            distances[:, class_id] = np.mean(
                difference * difference, axis=1, dtype=np.float64
            )
        logits = -distances / self.config.template_temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict_proba_components(self, features) -> dict[str, np.ndarray]:
        templates, tree = self._require_fitted()
        matrix = self._matrix(features, expected_columns=templates.shape[1])
        template = self._template_probabilities(matrix)
        shallow_tree = np.asarray(tree.predict_proba(matrix), dtype=np.float64)
        weight = self.config.template_weight
        hybrid = weight * template + (1.0 - weight) * shallow_tree
        hybrid /= hybrid.sum(axis=1, keepdims=True)
        return {
            "template_only": template,
            "shallow_tree": shallow_tree,
            "template_tree": hybrid,
        }

    def predict_proba(self, features) -> np.ndarray:
        return self.predict_proba_components(features)["template_tree"]

    def model_summary(self) -> dict[str, object]:
        templates, tree = self._require_fitted()
        return {
            "format_version": 1,
            "class_count": self.class_count,
            "input_dimension": int(templates.shape[1]),
            "template_shape": list(templates.shape),
            "tree_depth": int(tree.get_depth()),
            "tree_leaves": int(tree.get_n_leaves()),
            "config": self.config.to_dict(),
        }


__all__ = ["TemplateTreeClassifier", "TemplateTreeConfig"]
