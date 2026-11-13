#!/usr/bin/env python3
"""Run the validation-only CPU template-tree deployment baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np

from flowfm.data.formal_cohort import get_formal_dataset_spec
from flowfm.data.formal_dataset import (
    TRAIN_PERCENTS,
    load_formal_selection_samples,
)
from flowfm.data.missingness import (
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_deterministic_field_missingness,
    apply_structured_field_group_missingness,
)
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.training.a0_mr_protocol import FORMAL_CONDITIONS
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.baselines import TabularFeatureEncoder
from flowfm.training.neural_baselines import project_model_visible_samples
from flowfm.training.template_tree import (
    TemplateTreeClassifier,
    TemplateTreeConfig,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_REPO_ROOT = _PROJECT_ROOT / "repository"
_COHORT = (
    _PROJECT_ROOT
    / "data"
    / "splits"
    / "NF-UNSW-NB15-v2"
    / "formal-restart-1m-seed20260801-shared-coarse-v1"
)
_VIEW_ROOT = (
    _PROJECT_ROOT / "data" / "formal_views_restart_v1" / "nf_unsw_restart"
)
_OUTPUT_ROOT = (
    _PROJECT_ROOT
    / "results"
    / "nf-unsw-restart-template-tree-deployment-v1"
)
_DATASET_KEY = "nf_unsw_restart"
_MODEL_SEEDS = (17, 42, 20260725)
_COMPONENTS = ("template_only", "shallow_tree", "template_tree")
_CONFIG = TemplateTreeConfig()
_PROTOCOL_VERSION = "formal-template-tree-deployment-v1"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path):
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise RuntimeError(f"deployment baseline artifact is missing: {value}")
    return {
        "path": str(value),
        "bytes": value.stat().st_size,
        "sha256": _sha256_file(value),
    }


def _git_state(repo_root=_REPO_ROOT):
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("deployment baseline requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": False}


def _targets(samples, label_space):
    to_id = {label: index for index, label in enumerate(label_space.coarse_labels)}
    return np.asarray(
        [to_id[sample.coarse_label] for sample in samples], dtype=np.int64
    )


def _condition(samples, name, seed):
    if name == "clean":
        return apply_deterministic_field_missingness(
            samples,
            fraction=0.0,
            seed=seed,
            eligible_feature_ids=tuple(
                feature.name
                for feature in SCHEMAFLOW_MAIN_REGISTRY.model_features
            ),
        )
    if name in {"missing_30pct", "missing_50pct"}:
        fraction = 0.3 if name == "missing_30pct" else 0.5
        return apply_deterministic_field_missingness(
            samples,
            fraction=fraction,
            seed=seed,
            eligible_feature_ids=tuple(
                feature.name
                for feature in SCHEMAFLOW_MAIN_REGISTRY.model_features
            ),
        )
    prefix = "structured__"
    if not name.startswith(prefix):
        raise ValueError(f"unknown formal condition: {name}")
    group_name = name.removeprefix(prefix)
    if group_name not in STRUCTURED_MISSINGNESS_GROUPS:
        raise ValueError(f"unknown structured missingness group: {group_name}")
    return apply_structured_field_group_missingness(
        samples, group_name=group_name
    )


def _batched_components(model, matrix, *, batch_size=8192):
    blocks = {name: [] for name in _COMPONENTS}
    for offset in range(0, len(matrix), batch_size):
        current = model.predict_proba_components(
            matrix[offset : offset + batch_size]
        )
        for name in _COMPONENTS:
            blocks[name].append(current[name])
    return {
        name: np.concatenate(values, axis=0)
        for name, values in blocks.items()
    }


def _distribution(values):
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary distribution values must be finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the NF-UNSW restart CPU template-tree baseline"
    )
    parser.add_argument("--cohort-dir", type=Path, default=_COHORT)
    parser.add_argument("--view-root", type=Path, default=_VIEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT)
    return parser


def run(
    arguments,
    *,
    config=_CONFIG,
    protocol_version=_PROTOCOL_VERSION,
    development_selection=None,
    repo_root=_REPO_ROOT,
):
    cohort = validate_output_path(
        arguments.cohort_dir,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    view_root = validate_output_path(
        arguments.view_root,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    output = validate_output_path(
        arguments.output_root,
        enforce_autodl_data_disk=True,
        data_root=_DATA_ROOT,
    )
    if output.exists():
        raise RuntimeError("deployment baseline output already exists")
    spec = get_formal_dataset_spec(_DATASET_KEY)
    git = _git_state(repo_root)
    output.mkdir(parents=True)
    marker = output / ".incomplete"
    marker.write_text("deployment baseline incomplete\n", encoding="ascii")
    config_payload = {
        "format_version": 1,
        "protocol_version": protocol_version,
        "dataset_key": _DATASET_KEY,
        "dataset_id": spec.dataset_id,
        "cohort_dir": str(cohort),
        "view_root": str(view_root),
        "train_percents": list(TRAIN_PERCENTS),
        "model_seeds": list(_MODEL_SEEDS),
        "conditions": list(FORMAL_CONDITIONS),
        "components": list(_COMPONENTS),
        "model_config": config.to_dict(),
        "execution_device": "cpu",
        "selection_scope": ["train", "validation"],
        "sealed_test_files_read": False,
        "test_samples_materialized": False,
        "historical_exposure_policy": spec.historical_exposure_policy,
        "historically_independent": False,
        "git": git,
    }
    if development_selection is not None:
        config_payload["development_selection"] = development_selection
    write_json_snapshot(output / "config.json", config_payload)

    settings = []
    for percent in TRAIN_PERCENTS:
        view = view_root / f"train-{percent:02d}pct"
        samples = load_formal_selection_samples(
            _DATASET_KEY, cohort, view, data_root=_DATA_ROOT
        )
        train = project_model_visible_samples(samples.splits["train"])
        validation = samples.splits["val"]
        encoder = TabularFeatureEncoder.fit(
            train,
            category_encoders=samples.tokenizer_metadata.categorical_vocabularies,
        )
        train_started = time.perf_counter()
        train_matrix = encoder.transform(train)
        encoding_seconds = time.perf_counter() - train_started
        train_targets = _targets(train, samples.label_space)
        models = {}
        run_dirs = {}
        for seed in _MODEL_SEEDS:
            run_dir = output / f"pct-{percent:02d}-seed-{seed}"
            run_dir.mkdir()
            fit_started = time.perf_counter()
            model = TemplateTreeClassifier(
                class_count=len(samples.label_space.coarse_labels),
                random_state=seed,
                config=config,
            ).fit(train_matrix, train_targets)
            fit_seconds = time.perf_counter() - fit_started
            model_path = run_dir / "template_tree.joblib"
            joblib.dump(model, model_path, compress=3)
            write_json_snapshot(run_dir / "model_summary.json", model.model_summary())
            write_json_snapshot(run_dir / "encoder.json", encoder.to_dict())
            write_json_snapshot(
                run_dir / "tokenizer_metadata.json",
                samples.tokenizer_metadata.to_dict(),
            )
            write_json_snapshot(
                run_dir / "label_space.json", samples.label_space.to_dict()
            )
            models[seed] = {
                "model": model,
                "fit_seconds": fit_seconds,
                "model_artifact": _artifact(model_path),
                "metrics": {},
                "inference_seconds": {},
            }
            run_dirs[seed] = run_dir

        del train_matrix
        validation_targets = _targets(validation, samples.label_space)
        condition_reports = {}
        for condition in FORMAL_CONDITIONS:
            condition_samples, missingness = _condition(
                validation, condition, spec.split_seed
            )
            condition_reports[condition] = missingness
            matrix_started = time.perf_counter()
            condition_matrix = encoder.transform(
                project_model_visible_samples(condition_samples)
            )
            matrix_seconds = time.perf_counter() - matrix_started
            for seed, record in models.items():
                inference_started = time.perf_counter()
                probabilities = _batched_components(
                    record["model"], condition_matrix
                )
                inference_seconds = time.perf_counter() - inference_started
                record["inference_seconds"][condition] = {
                    "matrix_encoding": matrix_seconds,
                    "model_probabilities": inference_seconds,
                }
                record["metrics"][condition] = {
                    name: metric_payload(
                        values,
                        validation_targets,
                        samples.label_space.coarse_labels,
                    )
                    for name, values in probabilities.items()
                }
            del condition_matrix, condition_samples

        for seed, record in models.items():
            run_dir = run_dirs[seed]
            metrics_path = write_json_snapshot(
                run_dir / "validation_metrics.json",
                {
                    "format_version": 1,
                    "train_percent": percent,
                    "model_seed": seed,
                    "components": list(_COMPONENTS),
                    "conditions": record["metrics"],
                    "evaluation_scope": "validation_only",
                    "sealed_test_files_read": False,
                },
            )
            efficiency_path = write_json_snapshot(
                run_dir / "efficiency.json",
                {
                    "execution_device": "cpu",
                    "train_matrix_encoding_seconds": encoding_seconds,
                    "fit_seconds": record["fit_seconds"],
                    "validation_seconds": record["inference_seconds"],
                    "model_artifact_bytes": record["model_artifact"]["bytes"],
                    "peak_system_memory_bytes": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    ),
                },
            )
            settings.append(
                {
                    "train_percent": percent,
                    "model_seed": seed,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "run_dir": str(run_dir),
                    "model": record["model_artifact"],
                    "metrics": _artifact(metrics_path),
                    "efficiency": _artifact(efficiency_path),
                    "metrics_payload": record["metrics"],
                    "fit_seconds": record["fit_seconds"],
                }
            )
        write_json_snapshot(
            output / f"pct-{percent:02d}-missingness.json", condition_reports
        )

    aggregates = {}
    for component in _COMPONENTS:
        aggregates[component] = {}
        for condition in FORMAL_CONDITIONS:
            aggregates[component][condition] = _distribution(
                setting["metrics_payload"][condition][component]["macro_f1"]
                for setting in settings
            )
    summary_settings = []
    for setting in settings:
        summary_settings.append(
            {
                key: value
                for key, value in setting.items()
                if key not in {"metrics_payload"}
            }
        )
    summary = {
        "format_version": 1,
        "protocol_version": protocol_version,
        "status": "all_nine_validation_only_cpu_settings_completed",
        "dataset_key": _DATASET_KEY,
        "dataset_id": spec.dataset_id,
        "setting_count": len(settings),
        "train_percents": list(TRAIN_PERCENTS),
        "model_seeds": list(_MODEL_SEEDS),
        "conditions": list(FORMAL_CONDITIONS),
        "components": list(_COMPONENTS),
        "aggregates": aggregates,
        "settings": summary_settings,
        "sealed_test_files_read": False,
        "test_samples_materialized": False,
        "historically_independent": False,
        "git": git,
    }
    if development_selection is not None:
        summary["development_selection"] = development_selection
    write_json_snapshot(output / "summary.json", summary)
    marker.unlink()
    return summary


def main(argv=None):
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
