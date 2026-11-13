"""Shared sample contract for flow-based datasets."""

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class FlowSample:
    features: Sequence[Any]
    feature_ids: Sequence[str]
    categorical_mask: Sequence[bool]
    missing_mask: Sequence[bool]
    category_tokens: Sequence[str | None]
    binary_label: Any
    coarse_label: Any
    native_label: Any
    dataset_id: str
    sample_id: str
    feature_key: str
    conflict_flag: bool
    conflict_group_size: int
    conflict_label_count: int
    source_file: str
    source_row_id: str

    def __post_init__(self) -> None:
        aligned_lengths = {
            len(self.features),
            len(self.feature_ids),
            len(self.categorical_mask),
            len(self.missing_mask),
            len(self.category_tokens),
        }
        if len(aligned_lengths) != 1:
            raise ValueError("feature-aligned sequences must have equal length")
        for categorical, missing, token in zip(
            self.categorical_mask,
            self.missing_mask,
            self.category_tokens,
        ):
            if type(categorical) is not bool or type(missing) is not bool:
                raise ValueError("feature masks must contain booleans")
            if categorical:
                if missing:
                    if token is not None:
                        raise ValueError(
                            "missing categorical category_tokens must be None"
                        )
                elif not isinstance(token, str) or not token or token == "m:":
                    raise ValueError(
                        "observed categorical category_tokens "
                        "must be canonical strings"
                    )
            elif token is not None:
                raise ValueError("numeric category_tokens must be None")
        for field_name in (
            "dataset_id",
            "sample_id",
            "feature_key",
            "source_file",
            "source_row_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if type(self.conflict_flag) is not bool:
            raise ValueError("conflict_flag must be a boolean")
        for field_name in ("conflict_group_size", "conflict_label_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.conflict_label_count > self.conflict_group_size:
            raise ValueError(
                "conflict_label_count cannot exceed conflict_group_size"
            )
        expected_conflict = self.conflict_label_count > 1
        if self.conflict_flag != expected_conflict:
            raise ValueError(
                "conflict_flag must match whether conflict_label_count exceeds one"
            )
