#!/usr/bin/env python3
"""Audit history and publish cross-dataset formal cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowfm.data.formal_cohort import (
    FORMAL_DATASETS,
    audit_formal_dataset,
    publish_formal_cohort,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_REPO_ROOT = _PROJECT_ROOT / "repository"
_AUDIT_ROOT = _PROJECT_ROOT / "results" / "cross-dataset-formal-audits-v1"
_RESTART_AUDIT_ROOT = (
    _PROJECT_ROOT / "results" / "cross-dataset-formal-restart-audits-v1"
)
_REVISION_AUDIT_ROOT = (
    _PROJECT_ROOT / "results" / "cross-dataset-formal-revision-audits-v1"
)


def _audit_path(dataset):
    if "_rev" in dataset:
        return _REVISION_AUDIT_ROOT / dataset
    base = _RESTART_AUDIT_ROOT if dataset == "nf_unsw_restart" else _AUDIT_ROOT
    return base / dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit or build an untouched formal NetFlow-v2 cohort"
    )
    parser.add_argument("phase", choices=("audit", "build"))
    parser.add_argument(
        "--dataset",
        choices=tuple(FORMAL_DATASETS),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    return parser


def run(arguments):
    if arguments.phase == "audit":
        output = arguments.output or _audit_path(arguments.dataset)
        return audit_formal_dataset(
            arguments.dataset,
            output,
            repo_dir=_REPO_ROOT,
            project_root=_PROJECT_ROOT,
            data_root=_DATA_ROOT,
        )
    if arguments.audit_dir is None:
        arguments.audit_dir = _audit_path(arguments.dataset)
    if arguments.output is None:
        from flowfm.data.formal_cohort import get_formal_dataset_spec

        spec = get_formal_dataset_spec(arguments.dataset)
        arguments.output = (
            _PROJECT_ROOT / "data" / "splits" / spec.dataset_id / spec.cohort_name
        )
    return publish_formal_cohort(
        arguments.dataset,
        arguments.output,
        audit_dir=arguments.audit_dir,
        repo_dir=_REPO_ROOT,
        project_root=_PROJECT_ROOT,
        data_root=_DATA_ROOT,
    )


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
