#!/usr/bin/env python3
"""Run the revision supplementary matrix on resampling cohorts.

One cohort per invocation: train-only views at 1/5/10%, one model seed, the
XGBoost, LightGBM and target-excluded pretrained SchemaFlow endpoints, frozen
A0-MR selection on validation, registered exporter profiles, and one sealed test
open with per-record mask resolution. Model fitting and inference reuse the
formal matrix runner; every phase must run at the commit recorded in plan.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from flowfm.data.exporter_missingness import (
    EXPORTER_PROFILES,
    apply_intermittent_missingness,
    apply_profile_missingness,
    build_supplementary_resolver,
)
from flowfm.data.formal_cohort import (
    REVISION_ACTIVE_COHORT_COUNT,
    get_formal_dataset_spec,
    revision_cohort_key,
    validate_formal_overlap_contract,
)
from flowfm.data.formal_dataset import (
    TRAIN_PERCENTS,
    load_formal_selection_samples,
    materialize_formal_test_samples,
    open_formal_sealed_test_rows,
    publish_formal_training_view,
)
from flowfm.data.missingness import (
    STRUCTURED_MISSINGNESS_GROUPS,
    apply_deterministic_field_missingness,
    apply_structured_field_group_missingness,
)
from flowfm.data.schemaflow_registry import SCHEMAFLOW_MAIN_REGISTRY
from flowfm.experiment.metadata import validate_output_path, write_json_snapshot
from flowfm.training.a0_mr_protocol import (
    FORMAL_CONDITIONS,
    fixed_a0_policies,
    frozen_protocol_hash,
    policy_probabilities,
    select_a0_mr_validation,
)
from flowfm.training.baseline_evaluation import metric_payload
from flowfm.training.baselines import TabularFeatureEncoder
from flowfm.training.confirmatory_statistics import confirmatory_metric_payload
from flowfm.training.probability_fusion import pool_probabilities
from flowfm.training.safe_fusion import apply_scalar_temperature, fit_scalar_temperature
from flowfm.training.supplementary_protocol import (
    ENDPOINTS,
    TEST_CONDITIONS,
    VALIDATION_CONDITIONS,
    dispatch_probabilities,
    frozen_supplementary_hash,
    frozen_supplementary_payload,
    missing_mask_matrix,
    select_registered_profiles,
)
from scripts.run_cross_dataset_formal_matrix import (
    _LightGBMBoosterAdapter,
    _coarse_targets,
    _fit_lightgbm,
    _fit_xgboost,
    _open_claim,
    _restore_schemaflow,
    _run_logged,
    _schemaflow_probabilities,
    _tree_probabilities,
)
from scripts.run_nf_cse_a0_mr_confirmation import (
    _encoder_from_lock,
    _model_visible_samples,
    _sha256_file,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_REPO_ROOT = _PROJECT_ROOT / "repository"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_MATRIX_BASE = _RESULTS_ROOT / "cross-dataset-supplementary-matrix-v3"
_VIEW_BASE = _PROJECT_ROOT / "data" / "supplementary_views_v3"
_SOURCE_CHECKPOINT = (
    _RESULTS_ROOT
    / "schemaflow-nf-unsw-pretrain-smoke-20261017T120707789167Z-ecf5bdc6-7de8cdfb4f42"
    / "checkpoints"
    / "epoch-0001.pt"
)
_SOURCE_CHECKPOINT_SHA256 = (
    "b60d4097ae3557c8c29d99ae85bb0754fadd9a5934409d03fffbeec085bc7373"
)
_MODEL_SEED = 17
_RUN_VERSION = "cross-dataset-supplementary-matrix-v3"
# Inference only; training keeps its batch of 64. Batch 1024 matched batch 64 within
# 4e-6 in probability with no argmax change on 20,000 validation records.
_INFERENCE_BATCH_SIZE = 1024
_DRY_RUN_RECORDS = 2000
# Expected endpoint hyperparameters; n_jobs depends on the host and is not compared.
_XGBOOST_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}
_LIGHTGBM_PARAMETERS = {
    "n_estimators": 300,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "deterministic": True,
}


def _settings():
    return tuple(
        {
            "setting_id": f"pct-{percent:02d}-seed-{_MODEL_SEED}",
            "train_percent": percent,
            "model_seed": _MODEL_SEED,
        }
        for percent in TRAIN_PERCENTS
    )


def _active_cohorts():
    return tuple(
        revision_cohort_key(base, index)
        for base in ("nf_bot", "nf_cse", "nf_ton")
        for index in range(1, REVISION_ACTIVE_COHORT_COUNT + 1)
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    return write_json_snapshot(path, payload)


def _artifact(path):
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise RuntimeError(f"required artifact is missing: {value}")
    return {"path": str(value), "sha256": _sha256_file(value), "bytes": value.stat().st_size}


def _cohort_path(spec):
    return _PROJECT_ROOT / "data" / "splits" / spec.dataset_id / spec.cohort_name


def _matrix_root(key):
    return validate_output_path(
        _MATRIX_BASE / key, enforce_autodl_data_disk=True, data_root=_DATA_ROOT
    )


def _view_path(key, percent):
    return _VIEW_BASE / key / f"train-{percent:02d}pct"


def _state_path(root, setting):
    return root / "states" / f"{setting['setting_id']}.json"


def _load_state(root, setting):
    path = _state_path(root, setting)
    if not path.exists():
        return {"format_version": 1, "run_version": _RUN_VERSION, **setting, "steps": {}}
    payload = _read_json(path)
    if any(payload.get(name) != value for name, value in setting.items()):
        raise RuntimeError("supplementary state belongs to another setting")
    return payload


def _save_state(root, setting, state):
    path = _state_path(root, setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, state)


def _completed_step(state, name):
    step = state["steps"].get(name)
    if not isinstance(step, dict) or step.get("status") != "completed":
        return None
    for artifact in step.get("artifacts", {}).values():
        path = Path(artifact["path"])
        if not path.is_file() or _sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"completed artifact changed: {path}")
    return step


def _git_state():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("supplementary matrix requires a clean repository checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"commit": commit, "branch": branch, "dirty": False}


def _require_plan_commit(root):
    plan = _read_json(root / "plan.json")
    current = _git_state()
    if current["commit"] != plan["git"]["commit"]:
        raise RuntimeError(
            "code changed since plan.json; every phase must run at the planned commit"
        )
    return plan


def _plan(key):
    root = _matrix_root(key)
    spec = get_formal_dataset_spec(key)
    cohort = _cohort_path(spec)
    validate_formal_overlap_contract(spec, _read_json(cohort / "overlap_audit.json"))
    if _sha256_file(_SOURCE_CHECKPOINT) != _SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint changed")
    payload = {
        "format_version": 1,
        "run_version": _RUN_VERSION,
        "dataset_key": key,
        "dataset_id": spec.dataset_id,
        "split_seed": spec.split_seed,
        "cohort_dir": str(cohort),
        "cohort_seal_sha256": _sha256_file(cohort / "sealed_test.json"),
        "matrix_root": str(root),
        "settings": list(_settings()),
        "schemaflow_inference_batch_size": _INFERENCE_BATCH_SIZE,
        "source_checkpoint": _artifact(_SOURCE_CHECKPOINT),
        "a0_mr_protocol_hash": frozen_protocol_hash(),
        "supplementary_protocol_hash": frozen_supplementary_hash(),
        "git": _git_state(),
    }
    path = root / "plan.json"
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise RuntimeError("supplementary plan changed; start a new matrix root")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "protocol.json", frozen_supplementary_payload())
    _write_json(path, payload)
    return payload


def _prepare(key, root):
    plan = _require_plan_commit(root)
    for percent in TRAIN_PERCENTS:
        path = _view_path(key, percent)
        if (path / ".incomplete").exists():
            raise RuntimeError(f"training view is incomplete: {path}")
        if not path.exists():
            publish_formal_training_view(
                key, plan["cohort_dir"], path, train_percent=percent, data_root=_DATA_ROOT
            )
        summary = _read_json(path / "training_view_summary.json")
        if (
            summary.get("train_percent") != percent
            or summary.get("sealed_test_files_read") is not False
            or summary.get("test_labels_or_statistics_present") is not False
        ):
            raise RuntimeError("training view violates sealed scope")
    return {"status": "views_prepared"}


def _train(key, root):
    plan = _require_plan_commit(root)
    for setting in _settings():
        state = _load_state(root, setting)
        if _completed_step(state, "schemaflow_pretrained") is not None:
            continue
        result, wall = _run_logged(
            (
                sys.executable, "-m", "scripts.run_formal_neural_model",
                "--dataset", key,
                "--cohort-dir", plan["cohort_dir"],
                "--view-dir", _view_path(key, setting["train_percent"]),
                "--matrix-root", root,
                "--source-checkpoint", _SOURCE_CHECKPOINT,
                "--model-seed", setting["model_seed"],
                "--train-percent", setting["train_percent"],
                "--model", "schemaflow_pretrained",
            ),
            root / "logs" / f"{setting['setting_id']}-schemaflow_pretrained.log",
        )
        run_dir = Path(result["run_dir"])
        provenance = _read_json(run_dir / "formal_provenance.json")
        if provenance.get("git_commit") != plan["git"]["commit"]:
            raise RuntimeError("neural run recorded a different commit than the plan")
        state["steps"]["schemaflow_pretrained"] = {
            "status": "completed",
            "wall_seconds": wall,
            "run_dir": str(run_dir),
            "artifacts": {
                "checkpoint": _artifact(Path(result["checkpoint"])),
                "config": _artifact(run_dir / "config.snapshot.json"),
                "metrics": _artifact(run_dir / "metrics.json"),
                "formal_provenance": _artifact(run_dir / "formal_provenance.json"),
            },
        }
        _save_state(root, setting, state)
    return {"status": "neural_training_completed"}


def _iter_conditions(samples, seed, names):
    """Yield (condition, perturbed samples, report) one condition at a time."""

    for name in names:
        if name == "clean" or name.startswith("missing_"):
            fraction = {"clean": 0.0, "missing_30pct": 0.3, "missing_50pct": 0.5}[name]
            values, report = apply_deterministic_field_missingness(
                samples,
                fraction=fraction,
                seed=seed,
                eligible_feature_ids=tuple(
                    spec.name for spec in SCHEMAFLOW_MAIN_REGISTRY.model_features
                ),
            )
        elif name.startswith("structured__"):
            values, report = apply_structured_field_group_missingness(
                samples, group_name=name.removeprefix("structured__")
            )
        elif name.startswith("profile__"):
            values, report = apply_profile_missingness(
                samples, profile=name.removeprefix("profile__")
            )
        elif name.startswith("intermittent__"):
            values, report = apply_intermittent_missingness(
                samples, profile=name.removeprefix("intermittent__"), seed=seed
            )
        elif name.startswith("compound__"):
            values, report = apply_profile_missingness(
                samples, profile="compound_" + name.removeprefix("compound__")
            )
        else:
            raise ValueError(f"undeclared condition: {name}")
        yield name, values, report


def _endpoint_outputs(samples, *, encoder, xgboost, lightgbm, pretrained, tokenizer,
                      label_space, device, batch_size, timings):
    class_count = len(label_space.coarse_labels)
    started = time.perf_counter()
    matrix = encoder.transform(_model_visible_samples(samples))
    xgb = _tree_probabilities(xgboost, matrix, class_count)
    timings["xgboost"] += time.perf_counter() - started
    started = time.perf_counter()
    lgb = _tree_probabilities(lightgbm, matrix, class_count)
    timings["lightgbm"] += time.perf_counter() - started
    del matrix
    neural, elapsed = _schemaflow_probabilities(
        pretrained, samples, tokenizer, device, batch_size
    )
    timings["schemaflow_pretrained"] += elapsed
    return {
        "xgboost": xgb,
        "lightgbm": lgb,
        "schemaflow_pretrained": neural,
        "targets": _coarse_targets(samples, label_space),
    }


def _check_parameters(actual, expected, name):
    changed = {key: actual.get(key) for key, value in expected.items() if actual.get(key) != value}
    if changed:
        raise RuntimeError(f"{name} parameters differ from the protocol: {changed}")


def _fit_policy_temperatures(probabilities, outputs, label_names):
    locks, before, after = {}, {}, {}
    for condition, values in probabilities.items():
        target = outputs[condition]["targets"]
        lock = fit_scalar_temperature(values, target)
        calibrated = apply_scalar_temperature(values, lock["temperature"])
        if not np.array_equal(calibrated.argmax(axis=1), values.argmax(axis=1)):
            raise RuntimeError("scalar temperature changed argmax predictions")
        locks[condition] = lock
        before[condition] = metric_payload(values, target, label_names)
        after[condition] = metric_payload(calibrated, target, label_names)
    return locks, before, after


def _select_setting(key, root, setting):
    plan = _require_plan_commit(root)
    state = _load_state(root, setting)
    if _completed_step(state, "validation_selection") is not None:
        return state["steps"]["validation_selection"]
    neural = _completed_step(state, "schemaflow_pretrained")
    if neural is None:
        raise RuntimeError("SchemaFlow must finish before validation selection")
    run_dir = root / "selections" / setting["setting_id"]
    if run_dir.exists():
        # Validation-only output; a crashed attempt is kept for inspection.
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.failed-{int(time.time())}"))
    spec = get_formal_dataset_spec(key)
    samples = load_formal_selection_samples(
        key, plan["cohort_dir"], _view_path(key, setting["train_percent"]), data_root=_DATA_ROOT
    )
    label_names = samples.label_space.coarse_labels
    train = _model_visible_samples(samples.splits["train"])
    train_targets = _coarse_targets(train, samples.label_space)
    encoder = TabularFeatureEncoder.fit(
        train, category_encoders=samples.tokenizer_metadata.categorical_vocabularies
    )
    train_matrix = encoder.transform(train)
    xgboost, xgb_parameters, xgb_seconds = _fit_xgboost(
        train_matrix, train_targets, len(label_names), setting["model_seed"]
    )
    lightgbm, lgb_parameters, lgb_seconds = _fit_lightgbm(
        train_matrix, train_targets, len(label_names), setting["model_seed"]
    )
    _check_parameters(xgb_parameters, _XGBOOST_PARAMETERS, "XGBoost")
    _check_parameters(lgb_parameters, _LIGHTGBM_PARAMETERS, "LightGBM")
    del train_matrix, train
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    pretrained, _, pretrained_payload = _restore_schemaflow(
        neural, samples, device
    )
    timings = dict.fromkeys(ENDPOINTS, 0.0)
    outputs, missingness = {}, {}
    for condition, values, report in _iter_conditions(
        samples.splits["val"], spec.split_seed, VALIDATION_CONDITIONS
    ):
        outputs[condition] = _endpoint_outputs(
            values,
            encoder=encoder, xgboost=xgboost, lightgbm=lightgbm, pretrained=pretrained,
            tokenizer=samples.tokenizer_metadata, label_space=samples.label_space,
            device=device, batch_size=_INFERENCE_BATCH_SIZE,
            timings=timings,
        )
        missingness[condition] = report
    formal = {name: outputs[name] for name in FORMAL_CONDITIONS}
    selection = select_a0_mr_validation(formal, label_names)
    registered = select_registered_profiles(outputs, selection, label_names)
    validation_probabilities = {
        "fixed_a0": policy_probabilities(formal, fixed_a0_policies()),
        "a0_mr": policy_probabilities(formal, selection["locked_policy"]["policies"]),
        "a0_mr_registered": policy_probabilities(outputs, registered["registered_policies"]),
    }
    temperatures, metrics_before, metrics_after = {}, {}, {}
    for policy, probabilities in validation_probabilities.items():
        temperatures[policy], metrics_before[policy], metrics_after[policy] = (
            _fit_policy_temperatures(probabilities, outputs, label_names)
        )
    run_dir.mkdir(parents=True)
    marker = run_dir / ".incomplete"
    marker.write_text("supplementary validation selection incomplete\n", encoding="ascii")
    xgboost.save_model(str(run_dir / "xgboost_model.json"))
    lightgbm.booster_.save_model(str(run_dir / "lightgbm_model.txt"))
    _write_json(run_dir / "encoder.json", encoder.to_dict())
    _write_json(run_dir / "tokenizer_metadata.json", samples.tokenizer_metadata.to_dict())
    _write_json(run_dir / "label_space.json", samples.label_space.to_dict())
    _write_json(run_dir / "selection.json", selection)
    _write_json(run_dir / "registered_selection.json", registered)
    _write_json(run_dir / "temperature_lock.json", temperatures)
    _write_json(
        run_dir / "validation_metrics.json",
        {"format_version": 1, "uncalibrated": metrics_before, "temperature_calibrated": metrics_after},
    )
    _write_json(run_dir / "missingness.json", missingness)
    arrays = {
        "group_ids": np.asarray(samples.groups["val"], dtype="U64"),
        "sample_ids": np.asarray([sample.sample_id for sample in samples.splits["val"]], dtype="U64"),
        "targets": np.asarray(outputs["clean"]["targets"], dtype="<i8"),
    }
    for condition, values in outputs.items():
        for endpoint in ENDPOINTS:
            arrays[f"{condition}__{endpoint}"] = np.asarray(values[endpoint], dtype="<f8")
    np.savez_compressed(run_dir / "validation_predictions.npz", **arrays)
    _write_json(
        run_dir / "config.json",
        {
            "format_version": 1,
            "run_version": _RUN_VERSION,
            "dataset_key": key,
            **setting,
            "selection_scope": ["train", "validation"],
            "sealed_test_files_read": False,
            "test_samples_materialized": False,
            "a0_mr_protocol_hash": frozen_protocol_hash(),
            "supplementary_protocol_hash": frozen_supplementary_hash(),
            "missingness_seed": spec.split_seed,
            "xgboost_parameters": xgb_parameters,
            "lightgbm_parameters": lgb_parameters,
            "pretrained_initialization": pretrained_payload["extra_state"].get("initialization"),
            "git": _git_state(),
        },
    )
    _write_json(
        run_dir / "efficiency.json",
        {
            "xgboost_fit_seconds": xgb_seconds,
            "lightgbm_fit_seconds": lgb_seconds,
            "validation_inference_seconds": timings,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_system_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
    )
    marker.unlink()
    state["steps"]["validation_selection"] = {
        "status": "completed",
        "run_dir": str(run_dir),
        "artifacts": {
            name: _artifact(run_dir / filename)
            for name, filename in {
                "selection": "selection.json",
                "registered_selection": "registered_selection.json",
                "temperature_lock": "temperature_lock.json",
                "config": "config.json",
                "xgboost_model": "xgboost_model.json",
                "lightgbm_model": "lightgbm_model.txt",
                "encoder": "encoder.json",
                "tokenizer": "tokenizer_metadata.json",
                "label_space": "label_space.json",
                "validation_predictions": "validation_predictions.npz",
            }.items()
        },
    }
    _save_state(root, setting, state)
    return state["steps"]["validation_selection"]


def _lock(key, root):
    plan = _require_plan_commit(root)
    rows = []
    for setting in _settings():
        state = _load_state(root, setting)
        steps = {
            name: _completed_step(state, name)
            for name in ("schemaflow_pretrained", "validation_selection")
        }
        if any(value is None for value in steps.values()):
            raise RuntimeError(f"setting is not complete: {setting['setting_id']}")
        rows.append(
            {**setting, "steps": {name: {"run_dir": step["run_dir"], "artifacts": step["artifacts"]}
                                  for name, step in steps.items()}}
        )
    payload = {
        "format_version": 1,
        "run_version": _RUN_VERSION,
        "dataset_key": key,
        "status": "all_validation_selections_locked_before_test",
        "a0_mr_protocol_hash": frozen_protocol_hash(),
        "supplementary_protocol_hash": frozen_supplementary_hash(),
        "git": plan["git"],
        "settings": rows,
    }
    path = root / "all_validation_selections.lock.json"
    if path.exists():
        if _read_json(path) != payload:
            raise RuntimeError("global validation lock changed")
        return payload
    _write_json(path, payload)
    return payload


def _verify_lock(key, root):
    path = root / "all_validation_selections.lock.json"
    payload = _read_json(path)
    if (
        payload.get("dataset_key") != key
        or payload.get("a0_mr_protocol_hash") != frozen_protocol_hash()
        or payload.get("supplementary_protocol_hash") != frozen_supplementary_hash()
        or len(payload.get("settings", [])) != len(_settings())
    ):
        raise RuntimeError("global validation lock is invalid")
    for row in payload["settings"]:
        setting = {name: row[name] for name in ("setting_id", "train_percent", "model_seed")}
        state = _load_state(root, setting)
        for name, locked in row["steps"].items():
            current = _completed_step(state, name)
            if current is None or current["artifacts"] != locked["artifacts"]:
                raise RuntimeError("a locked artifact changed before the sealed test")
    return path, payload


def _evaluate_core(key, root, setting, selection_samples, eval_samples, group_ids, output_dir):
    """Score every test condition and policy on ``eval_samples``."""

    spec = get_formal_dataset_spec(key)
    state = _load_state(root, setting)
    neural = _completed_step(state, "schemaflow_pretrained")
    chosen = _completed_step(state, "validation_selection")
    selection_dir = Path(chosen["run_dir"])
    selection = _read_json(selection_dir / "selection.json")
    registered = _read_json(selection_dir / "registered_selection.json")
    temperatures = _read_json(selection_dir / "temperature_lock.json")
    if (
        selection.get("protocol_hash") != frozen_protocol_hash()
        or registered.get("protocol_hash") != frozen_supplementary_hash()
    ):
        raise RuntimeError("selection protocol changed before the sealed test")
    label_names = selection_samples.label_space.coarse_labels
    tokenizer = selection_samples.tokenizer_metadata
    encoder = _encoder_from_lock(_read_json(selection_dir / "encoder.json"), tokenizer)
    xgboost = XGBClassifier()
    xgboost.load_model(str(selection_dir / "xgboost_model.json"))
    lightgbm = _LightGBMBoosterAdapter(selection_dir / "lightgbm_model.txt", len(label_names))
    device = torch.device("cuda")
    pretrained, _, _ = _restore_schemaflow(neural, selection_samples, device)
    resolvers = {
        "a0_mr": build_supplementary_resolver(),
        "a0_mr_registered": build_supplementary_resolver(tuple(EXPORTER_PROFILES)),
    }
    policies = {
        "a0_mr": selection["locked_policy"]["policies"],
        "a0_mr_registered": registered["registered_policies"],
    }
    timings = dict.fromkeys(ENDPOINTS, 0.0)
    arrays = {
        "group_ids": np.asarray(group_ids, dtype="U64"),
        "sample_ids": np.asarray([sample.sample_id for sample in eval_samples], dtype="U64"),
    }
    metrics, routing, agreement, missingness = {}, {}, {}, {}
    for condition, values, report in _iter_conditions(eval_samples, spec.split_seed, TEST_CONDITIONS):
        outputs = _endpoint_outputs(
            values,
            encoder=encoder, xgboost=xgboost, lightgbm=lightgbm, pretrained=pretrained,
            tokenizer=tokenizer, label_space=selection_samples.label_space, device=device,
            batch_size=_INFERENCE_BATCH_SIZE, timings=timings,
        )
        arrays.setdefault("targets", np.asarray(outputs["targets"], dtype="<i8"))
        mask, field_order = missing_mask_matrix(values)
        scored = {name: outputs[name] for name in ENDPOINTS}
        scored["fixed_a0"] = pool_probabilities(
            outputs["xgboost"], outputs["schemaflow_pretrained"],
            alpha=fixed_a0_policies(("clean",))["clean"]["alpha"], method="linear",
        )
        routing[condition] = {}
        for policy, resolver in resolvers.items():
            resolved = resolver.resolve_matrix(mask, field_order)
            if condition in FORMAL_CONDITIONS:
                agreement.setdefault(policy, {})[condition] = float(np.mean(resolved == condition))
            scored[policy], routing[condition][policy] = dispatch_probabilities(
                outputs["xgboost"], outputs["schemaflow_pretrained"], resolved, policies[policy]
            )
            scored[f"{policy}__calibrated"], _ = dispatch_probabilities(
                outputs["xgboost"], outputs["schemaflow_pretrained"], resolved, policies[policy],
                temperatures={name: lock["temperature"] for name, lock in temperatures[policy].items()},
            )
        metrics[condition] = {
            name: confirmatory_metric_payload(values_, outputs["targets"], label_names)
            for name, values_ in scored.items()
        }
        for name, values_ in scored.items():
            arrays[f"{condition}__{name}"] = np.asarray(values_, dtype="<f8")
        missingness[condition] = report
        del values, outputs, scored, mask
    result = {
        "format_version": 1,
        "run_version": _RUN_VERSION,
        "dataset_key": key,
        "setting_id": setting["setting_id"],
        "record_count": len(eval_samples),
        "test_conditions": list(TEST_CONDITIONS),
        "metrics": metrics,
        "routing": routing,
        "resolver_label_agreement_original_conditions": agreement,
        "inference_seconds": timings,
    }
    if output_dir is None:
        return result
    output_dir.mkdir(parents=True, exist_ok=False)
    marker = output_dir / ".incomplete"
    marker.write_text("sealed setting output incomplete\n", encoding="ascii")
    np.savez_compressed(output_dir / "predictions.npz", **arrays)
    result["prediction_archive"] = _artifact(output_dir / "predictions.npz")
    _write_json(output_dir / "missingness.json", missingness)
    _write_json(output_dir / "metrics.json", result)
    marker.unlink()
    return result


def _dry_run(key, root):
    """Exercise the full evaluation path on validation records before any test claim."""

    plan = _require_plan_commit(root)
    rows = []
    for setting in _settings():
        samples = load_formal_selection_samples(
            key, plan["cohort_dir"], _view_path(key, setting["train_percent"]), data_root=_DATA_ROOT
        )
        validation = samples.splits["val"][:_DRY_RUN_RECORDS]
        result = _evaluate_core(
            key, root, setting, samples, validation,
            samples.groups["val"][:_DRY_RUN_RECORDS], None,
        )
        agreement = result["resolver_label_agreement_original_conditions"]
        if any(value != 1.0 for values in agreement.values() for value in values.values()):
            raise RuntimeError(f"resolver disagrees with generator labels: {agreement}")
        rows.append({"setting_id": setting["setting_id"], "records": result["record_count"]})
        del samples
    return rows


def _evaluate(key, root):
    plan = _require_plan_commit(root)
    lock_path, _ = _verify_lock(key, root)
    completed = root / "sealed_test_access.completed.json"
    if completed.exists():
        return _read_json(completed)
    claim_path = root / "sealed_test_access.claim.json"
    if claim_path.exists():
        raise RuntimeError("sealed test was claimed but not completed; automatic rerun is forbidden")
    dry_run = _dry_run(key, root)
    claim = {
        "format_version": 1,
        "run_version": _RUN_VERSION,
        "dataset_key": key,
        "status": "single_test_open_claimed",
        "global_selection_lock_sha256": _sha256_file(lock_path),
        "supplementary_protocol_hash": frozen_supplementary_hash(),
        "validation_dry_run": dry_run,
        "sealed_test_open_budget": 1,
        "git": plan["git"],
    }
    _open_claim(claim_path, claim)
    started = time.perf_counter()
    spec = get_formal_dataset_spec(key)
    sealed = open_formal_sealed_test_rows(key, _cohort_path(spec), data_root=_DATA_ROOT)
    results = []
    for setting in _settings():
        samples = load_formal_selection_samples(
            key, plan["cohort_dir"], _view_path(key, setting["train_percent"]), data_root=_DATA_ROOT
        )
        test = materialize_formal_test_samples(sealed, samples.preprocessor)
        result = _evaluate_core(
            key, root, setting, samples, test, sealed.group_ids,
            root / "sealed_test" / setting["setting_id"],
        )
        results.append({"setting_id": setting["setting_id"], "prediction_archive": result["prediction_archive"]})
        del samples, test
    receipt = {
        **claim,
        "status": "single_test_open_completed",
        "sealed_test_open_count": 1,
        "sealed_artifact_hashes": {
            "seal": sealed.seal_sha256,
            "records": sealed.records_sha256,
            "manifest": sealed.manifest_sha256,
            "features": sealed.features_sha256,
        },
        "sample_count": len(sealed.sample_order),
        "setting_results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(completed, receipt)
    return receipt


def build_parser():
    parser = argparse.ArgumentParser(description="Revision supplementary matrix for one cohort")
    parser.add_argument(
        "phase", choices=("plan", "prepare", "train", "select", "lock", "evaluate", "all")
    )
    parser.add_argument("--dataset", choices=_active_cohorts(), required=True)
    return parser


def run(arguments):
    key = arguments.dataset
    if arguments.phase == "plan":
        return _plan(key)
    root = _matrix_root(key)
    if arguments.phase == "all":
        _plan(key)
    phases = {
        "prepare": [_prepare],
        "train": [_train],
        "select": [lambda k, r: [_select_setting(k, r, s) for s in _settings()]],
        "lock": [_lock],
        "evaluate": [_evaluate],
        "all": [
            _prepare,
            _train,
            lambda k, r: [_select_setting(k, r, s) for s in _settings()],
            _lock,
            _evaluate,
        ],
    }[arguments.phase]
    result = None
    for phase in phases:
        result = phase(key, root)
    return {"phase": arguments.phase, "dataset": key, "status": "completed", "result_type": type(result).__name__}


def main(argv=None):
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
