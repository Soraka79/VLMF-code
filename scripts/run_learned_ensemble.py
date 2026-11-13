#!/usr/bin/env python3
"""Lock and evaluate the frozen logistic-stacking and learned-gate comparators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

from flowfm.training.learned_ensemble import (
    CONDITIONS,
    C_GRID,
    META_SEED,
    classification_metrics,
    dataset_stratified_paired_setting_bootstrap,
    fit_learned_gate,
    fit_logistic_stacker,
    fold_assignment_sha256,
    fold_summary,
    make_group_folds,
    predict_locked_gate,
    predict_locked_stacker,
)
from scripts.run_global_validation_fusion import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    DIRECTION_TOLERANCE,
    REPRODUCTION_TOLERANCE,
    _artifact_entry,
    _git_state,
    _load_evaluation_inputs,
    _read_json,
    _reproduce_vlmf,
    _sha256_file,
    _verify_manifest_unchanged,
    _write_json,
    _write_sha256_sidecar,
    expected_settings,
)


EXPERIMENT_NAME = "learned_ensemble_v1"
PROTOCOL_EXPERIMENT_NAME = "learned_ensemble_protocol_v1"
FROZEN_PROTOCOL_SHA256 = (
    "32c758fa25d9efaeca8d7c545beda202baa0d645b9f911e126575976179a5ef5"
)
FORMAT_VERSION = 1
MODEL_METHODS = ("logistic_stacking", "learned_gate")
FINAL_TOP_LEVEL_OUTPUTS = (
    "README.md",
    "analysis_lock.json",
    "analysis_lock.sha256",
    "audit.json",
    "cross_dataset_summary.json",
    "input_manifest.json",
    "input_manifest.validation.json",
    "model_locks.json",
    "model_locks.sha256",
    "per_setting_metrics.json",
    "protocol.json",
    "protocol.sha256",
    "tidy_metrics.csv",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LearnedAccessAudit:
    """Enforce the frozen protocol, model-lock, and sealed-test boundary."""

    def __init__(self, events: Sequence[Mapping[str, object]] | None = None):
        self.events = [dict(row) for row in (events or ())]
        self.protocol_locked = any(
            row.get("event") == "frozen_protocol_locked" for row in self.events
        )
        self.models_locked = any(
            row.get("event") == "all_meta_models_locked" for row in self.events
        )
        self.test_access_enabled = any(
            row.get("event") == "sealed_test_access_enabled"
            for row in self.events
        )

    def _record(self, event: str, **details) -> None:
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "timestamp_utc": _utc_now(),
                "event": event,
                **details,
            }
        )

    def lock_protocol(self, path: Path, sha256: str, source_path: Path) -> None:
        if self.events or self.protocol_locked:
            raise RuntimeError("the frozen protocol must be the first audit event")
        if sha256 != FROZEN_PROTOCOL_SHA256:
            raise RuntimeError("the frozen protocol hash changed")
        self.protocol_locked = True
        self._record(
            "frozen_protocol_locked",
            path=str(path),
            source_path=str(source_path),
            sha256=sha256,
        )

    def validation_read(self, path: Path, role: str) -> None:
        if not self.protocol_locked or self.models_locked:
            raise RuntimeError("validation reads require only the protocol lock")
        if "sealed_test" in Path(path).parts:
            raise RuntimeError("sealed-test paths are forbidden during model locking")
        self._record("validation_input_read", path=str(path), role=role)

    def record_model_fit(self, dataset: str, setting_id: str) -> None:
        if not self.protocol_locked or self.models_locked:
            raise RuntimeError("meta-model fitting crossed the lock boundary")
        self._record(
            "setting_meta_models_fitted",
            dataset=dataset,
            setting_id=setting_id,
            model_count=2,
        )

    def lock_models(self, path: Path, sha256: str, model_count: int) -> None:
        if not self.protocol_locked or self.models_locked or model_count != 54:
            raise RuntimeError("all 54 meta-models are required for the model lock")
        self.models_locked = True
        self._record(
            "all_meta_models_locked",
            path=str(path),
            sha256=sha256,
            model_count=model_count,
        )

    def enable_test_access(
        self, protocol_sha256: str, analysis_lock_sha256: str, model_count: int
    ) -> None:
        if (
            not self.protocol_locked
            or not self.models_locked
            or self.test_access_enabled
            or model_count != 54
        ):
            raise RuntimeError("sealed-test access requires all 54 model locks")
        if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
            raise RuntimeError("the frozen protocol hash changed")
        self.test_access_enabled = True
        self._record(
            "sealed_test_access_enabled",
            verified_protocol_sha256=protocol_sha256,
            verified_analysis_lock_sha256=analysis_lock_sha256,
            verified_model_count=model_count,
        )

    def test_read(self, path: Path, role: str) -> None:
        if not self.test_access_enabled:
            raise RuntimeError("sealed-test input read attempted before all model locks")
        self._record("sealed_test_input_read", path=str(path), role=role)

    def record(self, event: str, **details) -> None:
        self._record(event, **details)


def _parse_hash_manifest(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[1] in rows:
            raise RuntimeError(f"invalid SHA-256 manifest: {path}")
        rows[fields[1]] = fields[0]
    if not rows:
        raise RuntimeError(f"empty SHA-256 manifest: {path}")
    return rows


def _verify_frozen_protocol_bundle(project_root: Path) -> tuple[Path, list[dict]]:
    source_dir = project_root / "results" / PROTOCOL_EXPERIMENT_NAME
    protocol_path = source_dir / "protocol.json"
    protocol_sidecar = source_dir / "protocol.sha256"
    audit_path = source_dir / "audit.json"
    output_hashes_path = source_dir / "output_hashes.sha256"
    for path in (protocol_path, protocol_sidecar, audit_path, output_hashes_path):
        if not path.is_file():
            raise RuntimeError(f"frozen protocol artifact is missing: {path}")
    hashes = _parse_hash_manifest(output_hashes_path)
    for name in ("protocol.json", "protocol.sha256", "audit.json"):
        if hashes.get(name) != _sha256_file(source_dir / name):
            raise RuntimeError(f"frozen protocol bundle hash mismatch: {name}")
    sidecar_fields = protocol_sidecar.read_text(encoding="ascii").split()
    protocol_sha = _sha256_file(protocol_path)
    if (
        sidecar_fields != [protocol_sha, "protocol.json"]
        or protocol_sha != FROZEN_PROTOCOL_SHA256
    ):
        raise RuntimeError("frozen protocol SHA-256 mismatch")
    protocol = _read_json(protocol_path)
    source_audit = _read_json(audit_path)
    if (
        protocol.get("experiment_name") != PROTOCOL_EXPERIMENT_NAME
        or protocol.get("status") != "protocol_locked_execution_not_started"
        or protocol.get("model_lock", {}).get(
            "test_access_enabled_only_after_all_54_model_locks"
        )
        is not True
        or source_audit.get("phase") != "protocol_locked_only"
        or source_audit.get("assertions", {}).get("meta_models_fitted") != 0
        or source_audit.get("assertions", {}).get("sealed_test_files_read") != 0
    ):
        raise RuntimeError("frozen protocol authority bundle is invalid")
    entries = [
        _artifact_entry(path, role=f"frozen_protocol_{path.name}")
        for path in (protocol_path, protocol_sidecar, audit_path, output_hashes_path)
    ]
    return protocol_path, entries


def _validation_prediction_artifact(registry_row: Mapping[str, object]):
    try:
        return registry_row["locked_steps"]["validation_selection"]["artifacts"][
            "validation_predictions"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("validation prediction lock is missing") from error


def _load_validation_archive(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    probabilities = {}
    with np.load(path, allow_pickle=False) as archive:
        if "group_ids" not in archive.files:
            raise RuntimeError(f"validation group_ids missing: {path}")
        groups = np.asarray(archive["group_ids"])
        targets = None
        class_count = None
        for condition in CONDITIONS:
            keys = {
                "targets": f"{condition}__targets",
                "xgboost": f"{condition}__xgboost",
                "schemaflow": f"{condition}__schemaflow_pretrained",
            }
            if any(name not in archive.files for name in keys.values()):
                raise RuntimeError(f"validation fields missing for {condition}: {path}")
            observed_targets = np.asarray(archive[keys["targets"]], dtype=np.int64)
            tree = np.asarray(archive[keys["xgboost"]], dtype=np.float64)
            neural = np.asarray(archive[keys["schemaflow"]], dtype=np.float64)
            if targets is None:
                targets = observed_targets
            elif not np.array_equal(targets, observed_targets):
                raise RuntimeError("validation targets changed across conditions")
            if groups.shape != targets.shape or tree.shape != neural.shape:
                raise RuntimeError("validation archive arrays do not align")
            class_count = class_count or tree.shape[1]
            if tree.shape != (targets.size, class_count):
                raise RuntimeError("validation class count changed across conditions")
            classification_metrics(tree, targets, class_count)
            classification_metrics(neural, targets, class_count)
            probabilities[condition] = {
                "xgboost": tree,
                "schemaflow": neural,
            }
    return groups, targets, probabilities


def _model_artifact(path: Path, output_dir: Path) -> dict[str, object]:
    resolved = path.resolve()
    models_root = (output_dir / "models").resolve()
    if models_root not in resolved.parents or not path.is_file():
        raise RuntimeError("model artifact escaped the versioned output directory")
    return {
        "relative_path": str(resolved.relative_to(output_dir.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_model(
    output_dir: Path,
    dataset_key: str,
    setting_id: str,
    method: str,
    model: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, object]:
    path = output_dir / "models" / dataset_key / setting_id / f"{method}.model.json"
    _write_json(
        path,
        {
            **dict(model),
            "experiment_name": EXPERIMENT_NAME,
            "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "identity": dict(identity),
        },
    )
    return _model_artifact(path, output_dir)


def _software_state() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _lock_audit(
    audit: LearnedAccessAudit,
    *,
    model_locks_sha256: str,
    analysis_lock_sha256: str,
    git: Mapping[str, object],
) -> dict[str, object]:
    lock_sequence = next(
        row["sequence"]
        for row in audit.events
        if row["event"] == "all_meta_models_locked"
    )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "phase": "models_locked",
        "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "model_locks_sha256": model_locks_sha256,
        "analysis_lock_sha256": analysis_lock_sha256,
        "events": audit.events,
        "assertions": {
            "frozen_protocol_is_first_event": audit.events[0]["event"]
            == "frozen_protocol_locked",
            "protocol_locked_before_validation_reads": all(
                row["sequence"] > audit.events[0]["sequence"]
                for row in audit.events
                if row["event"] == "validation_input_read"
            ),
            "all_meta_models_lock_sequence": lock_sequence,
            "sealed_test_files_read_before_all_model_locks": 0,
            "sealed_test_access_enabled": False,
            "setting_count": 27,
            "model_count": 54,
            "logistic_stacker_count": 27,
            "learned_gate_count": 27,
            "base_models_retrained": False,
        },
        "software": _software_state(),
        "git": dict(git),
    }


def run_lock(project_root: Path, output_dir: Path, repo_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    if project_root not in output_dir.parents:
        raise RuntimeError("output directory must be inside the project root")
    matrix_roots = {
        name: (project_root / spec["matrix_relative_path"]).resolve()
        for name, spec in DATASETS.items()
    }
    if any(
        output_dir == root or output_dir in root.parents or root in output_dir.parents
        for root in matrix_roots.values()
    ):
        raise RuntimeError("output directory overlaps a sealed source matrix")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / ".lock.incomplete").write_text(
        "model lock incomplete\n", encoding="ascii"
    )

    source_protocol_path, manifest_entries = _verify_frozen_protocol_bundle(
        project_root
    )
    protocol_path = output_dir / "protocol.json"
    shutil.copyfile(source_protocol_path, protocol_path)
    _, protocol_sha = _write_sha256_sidecar(protocol_path)
    if protocol_sha != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("copied protocol is not byte-identical to the frozen lock")
    audit = LearnedAccessAudit()
    audit.lock_protocol(protocol_path, protocol_sha, source_protocol_path)

    settings = []
    for dataset, spec in DATASETS.items():
        matrix_root = matrix_roots[dataset]
        registry_path = matrix_root / "all_validation_selections.lock.json"
        audit.validation_read(registry_path, "validation_selection_registry")
        registry_sha = _sha256_file(registry_path)
        registry_payload = _read_json(registry_path)
        manifest_entries.append(
            _artifact_entry(
                registry_path,
                role="validation_selection_registry",
                dataset=dataset,
                observed_sha256=registry_sha,
            )
        )
        if (
            registry_payload.get("status")
            != "all_nine_validation_selections_locked_before_test"
            or registry_payload.get("dataset") != spec["matrix_key"]
            or registry_payload.get("setting_count") != 9
        ):
            raise RuntimeError(f"invalid validation registry for {dataset}")
        registry = {
            row["setting_id"]: row for row in registry_payload.get("settings", ())
        }
        if set(registry) != {row["setting_id"] for row in expected_settings()}:
            raise RuntimeError(f"validation settings incomplete for {dataset}")

        for setting in expected_settings():
            setting_id = setting["setting_id"]
            registry_row = registry[setting_id]
            if (
                registry_row.get("train_percent") != setting["train_percent"]
                or registry_row.get("model_seed") != setting["model_seed"]
            ):
                raise RuntimeError(f"setting identity mismatch for {dataset}/{setting_id}")
            validation_path = (
                matrix_root / "selections" / setting_id / "validation_predictions.npz"
            )
            artifact = _validation_prediction_artifact(registry_row)
            if Path(artifact["path"]).resolve() != validation_path:
                raise RuntimeError(
                    f"validation prediction path mismatch for {dataset}/{setting_id}"
                )
            audit.validation_read(
                validation_path, "locked_validation_prediction_archive"
            )
            validation_sha = _sha256_file(validation_path)
            if (
                validation_sha != artifact["sha256"]
                or validation_path.stat().st_size != artifact["bytes"]
            ):
                raise RuntimeError(
                    f"validation prediction hash mismatch for {dataset}/{setting_id}"
                )
            groups, targets, probabilities = _load_validation_archive(validation_path)
            folds = make_group_folds(targets, groups, folds=5, seed=META_SEED)
            stacker = fit_logistic_stacker(
                probabilities, targets, folds, C_grid=C_GRID
            )
            gate = fit_learned_gate(probabilities, targets, folds, C_grid=C_GRID)
            identity = {
                "dataset": dataset,
                "dataset_id": spec["dataset_id"],
                **setting,
                "validation_prediction_sha256": validation_sha,
                "fold_assignment_sha256": fold_assignment_sha256(folds),
            }
            model_rows = {}
            for method, fitted in (
                ("logistic_stacking", stacker),
                ("learned_gate", gate),
            ):
                model_artifact = _write_model(
                    output_dir,
                    spec["matrix_key"],
                    setting_id,
                    method,
                    fitted["model"],
                    identity,
                )
                model_rows[method] = {
                    "selected_C": fitted["selection"]["selected_C"],
                    "selection_trace": fitted["selection"],
                    "oof_candidates": fitted["oof_candidates"],
                    "feature_schema": fitted["model"]["feature_names"],
                    "preprocessing": fitted["model"]["preprocessing"],
                    "refit_coefficients": fitted["model"]["estimator"][
                        "coefficients"
                    ],
                    "refit_intercepts": fitted["model"]["estimator"]["intercepts"],
                    "locked_inference_max_absolute_error": fitted[
                        "locked_inference_max_absolute_error"
                    ],
                    "model_artifact": model_artifact,
                }
                if method == "learned_gate":
                    model_rows[method]["target_counts"] = fitted["target_counts"]
            audit.record_model_fit(dataset, setting_id)
            settings.append(
                {
                    "dataset": dataset,
                    "dataset_id": spec["dataset_id"],
                    **setting,
                    "validation_input": {
                        "path": str(validation_path),
                        "bytes": validation_path.stat().st_size,
                        "sha256": validation_sha,
                    },
                    "sample_count": int(targets.size),
                    "group_count": int(np.unique(groups).size),
                    "class_count": int(
                        probabilities[CONDITIONS[0]]["xgboost"].shape[1]
                    ),
                    "condition_order": list(CONDITIONS),
                    "fold_assignment": {
                        "splitter": "StratifiedGroupKFold",
                        "folds": 5,
                        "shuffle": True,
                        "seed": META_SEED,
                        "sha256": fold_assignment_sha256(folds),
                        "rows": fold_summary(folds, groups),
                        "group_exclusive": True,
                    },
                    "models": model_rows,
                }
            )
            manifest_entries.append(
                _artifact_entry(
                    validation_path,
                    role="locked_validation_prediction_archive",
                    dataset=dataset,
                    setting_id=setting_id,
                    observed_sha256=validation_sha,
                    expected_sha256=artifact["sha256"],
                )
            )
            print(
                json.dumps(
                    {
                        "phase": "lock",
                        "dataset": dataset,
                        "setting_id": setting_id,
                        "completed_settings": len(settings),
                        "stacker_C": stacker["selection"]["selected_C"],
                        "gate_C": gate["selection"]["selected_C"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if len(settings) != 27:
        raise RuntimeError("exactly 27 setting locks are required")
    git = _git_state(repo_root)
    model_locks_path = _write_json(
        output_dir / "model_locks.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "frozen_protocol_experiment": PROTOCOL_EXPERIMENT_NAME,
            "frozen_protocol_sha256": protocol_sha,
            "selection_scope": "validation_only_group_aware_five_fold",
            "test_files_read": False,
            "setting_count": len(settings),
            "model_count": len(settings) * len(MODEL_METHODS),
            "models_per_setting": list(MODEL_METHODS),
            "C_grid": list(C_GRID),
            "meta_seed": META_SEED,
            "software": _software_state(),
            "git": git,
            "settings": settings,
        },
    )
    _, model_locks_sha = _write_sha256_sidecar(model_locks_path)
    analysis_lock_path = _write_json(
        output_dir / "analysis_lock.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "frozen_protocol_sha256": protocol_sha,
            "model_locks_sha256": model_locks_sha,
            "setting_count": 27,
            "model_count": 54,
            "all_model_artifact_hashes_verified": True,
            "test_files_read": False,
            "locked_before_test": True,
        },
    )
    _, analysis_lock_sha = _write_sha256_sidecar(analysis_lock_path)
    audit.lock_models(analysis_lock_path, analysis_lock_sha, 54)
    _write_json(
        output_dir / "input_manifest.validation.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "phase": "validation_model_lock",
            "entry_count": len(manifest_entries),
            "all_hashes_verified": True,
            "entries": sorted(manifest_entries, key=lambda row: row["path"]),
        },
    )
    _write_json(
        output_dir / "audit.json",
        _lock_audit(
            audit,
            model_locks_sha256=model_locks_sha,
            analysis_lock_sha256=analysis_lock_sha,
            git=git,
        ),
    )
    (output_dir / ".lock.incomplete").unlink()
    return {
        "status": "all_learned_ensemble_models_locked",
        "output_dir": str(output_dir),
        "frozen_protocol_sha256": protocol_sha,
        "model_locks_sha256": model_locks_sha,
        "analysis_lock_sha256": analysis_lock_sha,
        "setting_count": 27,
        "model_count": 54,
        "test_files_read": False,
        "git": git,
    }


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    fields = sidecar.read_text(encoding="ascii").split()
    observed = _sha256_file(path)
    if fields != [observed, path.name]:
        raise RuntimeError(f"SHA-256 sidecar mismatch for {path.name}")
    return observed


def _verify_model_bundle(
    output_dir: Path, payload: Mapping[str, object]
) -> int:
    settings = payload.get("settings", ())
    if len(settings) != 27 or payload.get("model_count") != 54:
        raise RuntimeError("model lock count is invalid")
    identities = set()
    model_count = 0
    for row in settings:
        identity = (row.get("dataset"), row.get("setting_id"))
        if identity in identities or set(row.get("models", {})) != set(MODEL_METHODS):
            raise RuntimeError("model lock identities are incomplete or duplicated")
        identities.add(identity)
        for method in MODEL_METHODS:
            artifact = row["models"][method]["model_artifact"]
            path = (output_dir / artifact["relative_path"]).resolve()
            if (output_dir / "models").resolve() not in path.parents:
                raise RuntimeError("locked model path escaped the output directory")
            if (
                not path.is_file()
                or path.stat().st_size != artifact["bytes"]
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise RuntimeError(f"locked model artifact changed: {path}")
            model = _read_json(path)
            if (
                model.get("method") != method
                or model.get("frozen_protocol_sha256") != FROZEN_PROTOCOL_SHA256
                or model.get("identity", {}).get("dataset") != row["dataset"]
                or model.get("identity", {}).get("setting_id") != row["setting_id"]
                or model.get("identity", {}).get("fold_assignment_sha256")
                != row["fold_assignment"]["sha256"]
                or model.get("selected_C") != row["models"][method]["selected_C"]
                or model.get("feature_names")
                != row["models"][method]["feature_schema"]
                or model.get("preprocessing")
                != row["models"][method]["preprocessing"]
                or model.get("estimator", {}).get("coefficients")
                != row["models"][method]["refit_coefficients"]
                or model.get("estimator", {}).get("intercepts")
                != row["models"][method]["refit_intercepts"]
            ):
                raise RuntimeError("locked model identity changed")
            model_count += 1
    expected_identities = {
        (dataset, setting["setting_id"])
        for dataset in DATASETS
        for setting in expected_settings()
    }
    if identities != expected_identities:
        raise RuntimeError("the 27 declared setting identities are not all locked")
    if model_count != 54:
        raise RuntimeError("exactly 54 serialized meta-models are required")
    return model_count


def _read_locked_model(
    output_dir: Path, model_row: Mapping[str, object]
) -> dict[str, object]:
    artifact = model_row["model_artifact"]
    path = (output_dir / artifact["relative_path"]).resolve()
    if (
        path.stat().st_size != artifact["bytes"]
        or _sha256_file(path) != artifact["sha256"]
    ):
        raise RuntimeError("locked model changed during evaluation")
    return _read_json(path)


def _evaluate_models(
    inputs: Sequence[Mapping[str, object]],
    reproduced: Mapping[tuple[str, str], Mapping[str, object]],
    output_dir: Path,
    audit: LearnedAccessAudit,
) -> tuple[list[dict], list[dict], float]:
    tidy = []
    settings = []
    maximum_reproduction_error = 0.0
    for source in inputs:
        dataset = source["dataset"]
        setting_id = source["setting_id"]
        key = (dataset, setting_id)
        model_rows = source["policy_row"]["models"]
        stacker_model = _read_locked_model(
            output_dir, model_rows["logistic_stacking"]
        )
        gate_model = _read_locked_model(output_dir, model_rows["learned_gate"])
        path = source["prediction_path"]
        audit.test_read(path, "sealed_branch_probabilities_and_labels")
        condition_rows = {}
        with np.load(path, allow_pickle=False) as archive:
            targets = np.asarray(archive["targets"], dtype=np.int64)
            class_count = len(source["label_names"])
            if (
                stacker_model.get("class_count") != class_count
                or gate_model.get("class_count") != class_count
            ):
                raise RuntimeError(
                    f"locked class count changed for {dataset}/{setting_id}"
                )
            for condition_index, condition in enumerate(CONDITIONS):
                tree_key = f"uncalibrated__xgboost__{condition}"
                neural_key = f"uncalibrated__schemaflow_pretrained__{condition}"
                if tree_key not in archive.files or neural_key not in archive.files:
                    raise RuntimeError(
                        f"sealed branch fields missing for {dataset}/{setting_id}/{condition}"
                    )
                tree = np.asarray(archive[tree_key], dtype=np.float64)
                neural = np.asarray(archive[neural_key], dtype=np.float64)
                stacker_probabilities = predict_locked_stacker(
                    stacker_model, tree, neural
                )
                gate_probabilities = predict_locked_gate(
                    gate_model,
                    tree,
                    neural,
                    condition_index=condition_index,
                )
                metrics = {
                    "VLMF": reproduced[key][condition]["metrics"],
                    "logistic_stacking": classification_metrics(
                        stacker_probabilities, targets, class_count
                    ),
                    "learned_gate": classification_metrics(
                        gate_probabilities, targets, class_count
                    ),
                }
                error = reproduced[key][condition]["absolute_reproduction_error"]
                maximum_reproduction_error = max(maximum_reproduction_error, error)
                condition_rows[condition] = metrics
                for method, values in metrics.items():
                    tidy.append(
                        {
                            "dataset": dataset,
                            "dataset_id": source["dataset_id"],
                            "train_percent": source["train_percent"],
                            "model_seed": source["model_seed"],
                            "setting_id": setting_id,
                            "condition": condition,
                            "method": method,
                            **values,
                        }
                    )
        settings.append(
            {
                "dataset": dataset,
                "dataset_id": source["dataset_id"],
                "train_percent": source["train_percent"],
                "model_seed": source["model_seed"],
                "setting_id": setting_id,
                "validation_selection": {
                    method: {
                        "selected_C": model_rows[method]["selected_C"],
                        "selection_trace": model_rows[method]["selection_trace"],
                        "model_artifact": model_rows[method]["model_artifact"],
                    }
                    for method in MODEL_METHODS
                },
                "test_conditions": condition_rows,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "evaluate",
                    "dataset": dataset,
                    "setting_id": setting_id,
                    "completed_settings": len(settings),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tidy, settings, maximum_reproduction_error


def _direction_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive": sum(value > DIRECTION_TOLERANCE for value in values),
        "zero": sum(abs(value) <= DIRECTION_TOLERANCE for value in values),
        "negative": sum(value < -DIRECTION_TOLERANCE for value in values),
    }


def _build_summary(
    tidy: Sequence[Mapping[str, object]],
    model_locks: Mapping[str, object],
    maximum_reproduction_error: float,
) -> dict[str, object]:
    methods = ("VLMF", *MODEL_METHODS)
    absolute = {}
    for method in methods:
        rows = [row for row in tidy if row["method"] == method]
        by_dataset = {}
        for dataset in DATASETS:
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            by_condition = {
                condition: float(
                    np.mean(
                        [
                            row["macro_f1"]
                            for row in dataset_rows
                            if row["condition"] == condition
                        ]
                    )
                )
                for condition in CONDITIONS
            }
            by_dataset[dataset] = {
                "condition_macro_f1": by_condition,
                "three_condition_macro_f1": float(
                    np.mean(list(by_condition.values()))
                ),
            }
        absolute[method] = {
            "datasets": by_dataset,
            "equal_dataset_macro_f1": float(
                np.mean(
                    [row["three_condition_macro_f1"] for row in by_dataset.values()]
                )
            ),
        }

    comparisons = {}
    for comparator in MODEL_METHODS:
        deltas_by_dataset = {}
        setting_rows = []
        for dataset in DATASETS:
            deltas = []
            for setting in expected_settings():
                selected = [
                    row
                    for row in tidy
                    if row["dataset"] == dataset
                    and row["setting_id"] == setting["setting_id"]
                ]
                vlmf = [
                    row["macro_f1"] for row in selected if row["method"] == "VLMF"
                ]
                learned = [
                    row["macro_f1"]
                    for row in selected
                    if row["method"] == comparator
                ]
                if len(vlmf) != 3 or len(learned) != 3:
                    raise RuntimeError("setting aggregation inputs are incomplete")
                delta = float(np.mean(vlmf) - np.mean(learned))
                deltas.append(delta)
                setting_rows.append(
                    {
                        "dataset": dataset,
                        **setting,
                        "VLMF_minus_comparator_macro_f1": delta,
                    }
                )
            deltas_by_dataset[dataset] = deltas
        bootstrap = dataset_stratified_paired_setting_bootstrap(
            deltas_by_dataset,
            datasets=tuple(DATASETS),
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        all_deltas = [
            row["VLMF_minus_comparator_macro_f1"] for row in setting_rows
        ]
        comparisons[f"VLMF_minus_{comparator}"] = {
            "difference_semantics": "VLMF_macro_f1_minus_comparator_macro_f1",
            "dataset_means": bootstrap["observed_by_dataset"],
            "equal_dataset_mean": bootstrap["observed_equal_dataset_mean"],
            "setting_direction_counts": _direction_counts(all_deltas),
            "setting_count": 27,
            "setting_rows": setting_rows,
            "bootstrap": bootstrap,
        }

    C_counts = {
        method: {
            str(C): sum(
                row["models"][method]["selected_C"] == C
                for row in model_locks["settings"]
            )
            for C in C_GRID
        }
        for method in MODEL_METHODS
    }
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "scope": {
            "datasets": list(DATASETS),
            "conditions": list(CONDITIONS),
            "settings": 27,
            "condition_cells": 81,
            "base_models_retrained": False,
        },
        "absolute_macro_f1": absolute,
        "primary_comparisons": comparisons,
        "selected_C_counts": C_counts,
        "VLMF_reproduction": {
            "condition_cells": 81,
            "tolerance": REPRODUCTION_TOLERANCE,
            "maximum_absolute_macro_f1_error": maximum_reproduction_error,
            "passed": maximum_reproduction_error <= REPRODUCTION_TOLERANCE,
        },
        "reporting": {
            "all_results_reported_regardless_of_direction": True,
            "p_value_inferred": False,
            "test_results_changed_protocol_or_candidate_grid": False,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _readme(summary: Mapping[str, object]) -> str:
    stack = summary["primary_comparisons"]["VLMF_minus_logistic_stacking"]
    gate = summary["primary_comparisons"]["VLMF_minus_learned_gate"]
    return f"""# Learned Ensemble v1

This independently versioned analysis executes the byte-locked
`learned_ensemble_protocol_v1`. It fits no base model and uses only the three
declared clean/random-loss conditions. All 54 validation-selected meta-models
were serialized and hashed before any sealed-test input was read.

Frozen protocol SHA-256: `{FROZEN_PROTOCOL_SHA256}`

## Exact reproduction

From `/root/autodl-tmp/Res-AE-Attn/repository`:

```bash
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.run_learned_ensemble lock \\
  --project-root /root/autodl-tmp/Res-AE-Attn \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/{EXPERIMENT_NAME}

env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.run_learned_ensemble evaluate \\
  --project-root /root/autodl-tmp/Res-AE-Attn \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/{EXPERIMENT_NAME}

sha256sum -c /root/autodl-tmp/Res-AE-Attn/results/{EXPERIMENT_NAME}/output_hashes.sha256
```

The lock command refuses an existing output directory. The evaluate command
refuses existing result files and cannot enable test access unless all 54 model
artifacts, the model registry, the frozen protocol, and their hashes verify.

## Reporting rule

Each condition is weighted equally within a setting, the nine settings are
weighted equally within each dataset, and the three datasets receive equal
weight. Confidence intervals use the predeclared dataset-stratified paired-
setting bootstrap with 20,000 resamples and seed 20260730. No p-value is inferred.

VLMF minus logistic stacking: `{stack['equal_dataset_mean']}` macro-F1,
95% CI `[{stack['bootstrap']['ci95_lower']}, {stack['bootstrap']['ci95_upper']}]`.

VLMF minus learned gate: `{gate['equal_dataset_mean']}` macro-F1,
95% CI `[{gate['bootstrap']['ci95_lower']}, {gate['bootstrap']['ci95_upper']}]`.
"""


def _write_output_hashes(output_dir: Path) -> Path:
    paths = [output_dir / name for name in FINAL_TOP_LEVEL_OUTPUTS]
    paths.extend(sorted((output_dir / "models").rglob("*.model.json")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"required outputs are missing: {missing}")
    rows = [
        f"{_sha256_file(path)}  {path.relative_to(output_dir)}" for path in paths
    ]
    destination = output_dir / "output_hashes.sha256"
    destination.write_text("\n".join(rows) + "\n", encoding="ascii")
    return destination


def run_evaluate(
    project_root: Path, output_dir: Path, repo_root: Path
) -> dict[str, object]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    if not output_dir.is_dir() or (output_dir / ".lock.incomplete").exists():
        raise RuntimeError("completed model lock output is required")
    forbidden = [
        name
        for name in (
            "README.md",
            "cross_dataset_summary.json",
            "input_manifest.json",
            "per_setting_metrics.json",
            "tidy_metrics.csv",
            "output_hashes.sha256",
            ".evaluate.incomplete",
        )
        if (output_dir / name).exists()
    ]
    if forbidden:
        raise RuntimeError(f"refusing to overwrite evaluation outputs: {forbidden}")

    protocol_path = output_dir / "protocol.json"
    model_locks_path = output_dir / "model_locks.json"
    analysis_lock_path = output_dir / "analysis_lock.json"
    protocol_sha = _verify_sidecar(protocol_path)
    model_locks_sha = _verify_sidecar(model_locks_path)
    analysis_lock_sha = _verify_sidecar(analysis_lock_path)
    if protocol_sha != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("the frozen protocol changed before evaluation")
    model_locks = _read_json(model_locks_path)
    analysis_lock = _read_json(analysis_lock_path)
    model_count = _verify_model_bundle(output_dir, model_locks)
    if (
        model_locks.get("frozen_protocol_sha256") != protocol_sha
        or model_locks.get("test_files_read") is not False
        or analysis_lock.get("frozen_protocol_sha256") != protocol_sha
        or analysis_lock.get("model_locks_sha256") != model_locks_sha
        or analysis_lock.get("model_count") != 54
        or analysis_lock.get("test_files_read") is not False
    ):
        raise RuntimeError("model analysis lock bundle is invalid")
    lock_audit = _read_json(output_dir / "audit.json")
    if (
        lock_audit.get("phase") != "models_locked"
        or lock_audit.get("analysis_lock_sha256") != analysis_lock_sha
        or lock_audit.get("assertions", {}).get(
            "sealed_test_files_read_before_all_model_locks"
        )
        != 0
    ):
        raise RuntimeError("model lock audit is invalid")
    validation_manifest = _read_json(output_dir / "input_manifest.validation.json")
    validation_entries = validation_manifest["entries"]
    _verify_manifest_unchanged(validation_entries)
    audit = LearnedAccessAudit(lock_audit["events"])
    audit.record(
        "validation_inputs_reverified_before_test_access",
        entry_count=len(validation_entries),
        unchanged=True,
    )
    audit.enable_test_access(protocol_sha, analysis_lock_sha, model_count)
    _write_json(
        output_dir / "audit.json",
        {
            **lock_audit,
            "phase": "evaluation_unlocked",
            "events": audit.events,
            "assertions": {
                **lock_audit["assertions"],
                "sealed_test_access_enabled": True,
            },
        },
    )
    (output_dir / ".evaluate.incomplete").write_text(
        "evaluation incomplete\n", encoding="ascii"
    )

    inputs, test_entries = _load_evaluation_inputs(
        project_root, model_locks["settings"], audit
    )
    reproduced = _reproduce_vlmf(inputs, audit)
    tidy, settings, maximum_error = _evaluate_models(
        inputs, reproduced, output_dir, audit
    )
    summary = _build_summary(tidy, model_locks, maximum_error)
    if not summary["VLMF_reproduction"]["passed"]:
        raise RuntimeError("VLMF reproduction failed before result generation")

    manifest_entries = sorted(
        [*validation_entries, *test_entries], key=lambda row: row["path"]
    )
    _verify_manifest_unchanged(manifest_entries)
    audit.record(
        "all_input_hashes_reverified_after_evaluation",
        entry_count=len(manifest_entries),
        validation_prediction_archive_count=sum(
            row["role"] == "locked_validation_prediction_archive"
            for row in manifest_entries
        ),
        sealed_prediction_archive_count=sum(
            row["role"] == "sealed_prediction_archive"
            for row in manifest_entries
        ),
        unchanged=True,
    )
    _write_csv(output_dir / "tidy_metrics.csv", tidy)
    _write_json(
        output_dir / "per_setting_metrics.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "setting_count": len(settings),
            "settings": settings,
        },
    )
    _write_json(output_dir / "cross_dataset_summary.json", summary)
    _write_json(
        output_dir / "input_manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "entry_count": len(manifest_entries),
            "all_hashes_verified": True,
            "all_inputs_unchanged_after_evaluation": True,
            "entries": manifest_entries,
        },
    )
    (output_dir / "README.md").write_text(_readme(summary), encoding="utf-8")

    lock_sequence = next(
        row["sequence"]
        for row in audit.events
        if row["event"] == "all_meta_models_locked"
    )
    first_test_sequence = min(
        row["sequence"]
        for row in audit.events
        if row["event"] == "sealed_test_input_read"
    )
    final_audit = {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "phase": "evaluated",
        "frozen_protocol_sha256": protocol_sha,
        "model_locks_sha256": model_locks_sha,
        "analysis_lock_sha256": analysis_lock_sha,
        "events": audit.events,
        "assertions": {
            **lock_audit["assertions"],
            "sealed_test_access_enabled": True,
            "first_sealed_test_read_sequence": first_test_sequence,
            "all_sealed_test_reads_after_all_model_locks": first_test_sequence
            > lock_sequence,
            "all_54_model_artifacts_verified_before_test": True,
            "VLMF_macro_f1_reproduced_before_new_results": True,
            "VLMF_reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "VLMF_maximum_absolute_reproduction_error": maximum_error,
            "setting_count": 27,
            "condition_cell_count": 81,
            "model_count": 54,
            "all_input_hashes_verified": True,
            "all_inputs_unchanged_after_evaluation": True,
            "source_matrix_files_written": False,
            "output_directory_disjoint_from_source_matrices": True,
            "base_models_retrained": False,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "p_value_inferred": False,
            "all_results_reported_regardless_of_direction": True,
            "test_results_changed_protocol_or_candidate_grid": False,
        },
        "software": _software_state(),
        "git": _git_state(repo_root),
    }
    _write_json(output_dir / "audit.json", final_audit)
    (output_dir / ".evaluate.incomplete").unlink()
    output_hashes_path = _write_output_hashes(output_dir)
    return {
        "status": "learned_ensemble_completed",
        "output_dir": str(output_dir),
        "frozen_protocol_sha256": protocol_sha,
        "model_locks_sha256": model_locks_sha,
        "analysis_lock_sha256": analysis_lock_sha,
        "output_hashes_sha256": _sha256_file(output_hashes_path),
        "cross_dataset_summary": summary,
        "git": _git_state(repo_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the frozen validation-controlled learned ensembles"
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("lock", "evaluate"):
        child = subparsers.add_parser(phase)
        child.add_argument(
            "--project-root",
            type=Path,
            default=Path("/root/autodl-tmp/Res-AE-Attn"),
        )
        child.add_argument(
            "--output-dir",
            type=Path,
            default=Path("/root/autodl-tmp/Res-AE-Attn/results")
            / EXPERIMENT_NAME,
        )
        child.add_argument(
            "--repo-root",
            type=Path,
            default=Path("/root/autodl-tmp/Res-AE-Attn/repository"),
        )
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.phase == "lock":
        result = run_lock(
            arguments.project_root, arguments.output_dir, arguments.repo_root
        )
    else:
        result = run_evaluate(
            arguments.project_root, arguments.output_dir, arguments.repo_root
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
