#!/usr/bin/env python3
"""Run the validation-selected all-feature template-tree CPU baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowfm.training.template_tree import TemplateTreeConfig
from scripts import run_formal_deployment_baseline as base


_PROTOCOL_VERSION = "formal-template-tree-deployment-v2"
_OUTPUT_ROOT = (
    base._PROJECT_ROOT
    / "results"
    / "nf-unsw-restart-template-tree-deployment-v2"
)
_CONFIG = TemplateTreeConfig(
    max_depth=8,
    max_leaf_nodes=128,
    min_samples_leaf=2,
    max_features=None,
    template_temperature=0.05,
    template_weight=0.5,
)
_DEVELOPMENT_SELECTION = {
    "status": "frozen_after_bounded_validation_diagnostic",
    "selection_scope": "train_01pct_seed17_clean_validation_only",
    "selection_metric": "shallow_tree_clean_validation_macro_f1",
    "candidate_count": 4,
    "tie_break": [
        "higher_macro_f1",
        "shallower_tree",
        "fewer_maximum_leaves",
        "lexicographic_profile_id",
    ],
    "selected_profile": "all_d8_l128",
    "candidates": [
        {
            "profile_id": "sqrt_d8_l128",
            "max_features": "sqrt",
            "max_depth": 8,
            "max_leaf_nodes": 128,
            "macro_f1": 0.192592703996344,
            "fit_seconds": 0.10392006114125252,
        },
        {
            "profile_id": "all_d8_l128",
            "max_features": None,
            "max_depth": 8,
            "max_leaf_nodes": 128,
            "macro_f1": 0.460951984639066,
            "fit_seconds": 0.43652286008000374,
        },
        {
            "profile_id": "all_d12_l256",
            "max_features": None,
            "max_depth": 12,
            "max_leaf_nodes": 256,
            "macro_f1": 0.4543307741955617,
            "fit_seconds": 0.422652754932642,
        },
        {
            "profile_id": "log2_d12_l256",
            "max_features": "log2",
            "max_depth": 12,
            "max_leaf_nodes": 256,
            "macro_f1": 0.035864009891070046,
            "fit_seconds": 0.10396483540534973,
        },
    ],
    "candidate_results_use": "configuration_only_not_test_inference",
    "sealed_test_files_read": False,
    "test_samples_materialized": False,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the NF-UNSW all-feature template-tree CPU baseline"
    )
    parser.add_argument("--cohort-dir", type=Path, default=base._COHORT)
    parser.add_argument("--view-root", type=Path, default=base._VIEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT)
    return parser


def run(arguments):
    return base.run(
        arguments,
        config=_CONFIG,
        protocol_version=_PROTOCOL_VERSION,
        development_selection=_DEVELOPMENT_SELECTION,
        repo_root=Path(__file__).resolve().parents[1],
    )


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
