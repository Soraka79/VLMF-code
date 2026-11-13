#!/usr/bin/env python3
"""Validation-only A0-MR-Safe-v1 development on frozen NF-CSE views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from xgboost import XGBClassifier

from flowfm.config import load_config
from flowfm.data.labels import NF_CSE_LABEL_SPACE
from flowfm.data.nf_cse_confirmatory import SUPPORTED_COARSE_LABELS
from flowfm.data.splits import SplitManifest
from flowfm.training.confirmatory_statistics import (
    NF_CSE_RARE_COARSE_LABELS,
    confirmatory_metric_payload,
)
from flowfm.training.probability_fusion import (
    default_alpha_grid,
    pool_probabilities,
    probability_sha256,
)
from flowfm.training.safe_fusion import (
    MAX_RARE_RECALL_DROP,
    MIN_RARE_VALIDATION_SUPPORT,
    SAFE_PROTOCOL_VERSION,
    safe_route_and_calibrate,
)
from scripts.run_nf_cse_a0_mr_confirmation import (
    _coarse_targets,
    _encoder_from_lock,
    _load_selection_inputs,
    _model_visible_samples,
    _restore_model,
    _schemaflow_probabilities,
    _sha256_file,
    _xgboost_probabilities,
)
from scripts.run_nf_cse_formal_matrix import (
    _FORMAL_CONDITIONS,
    _formal_conditions,
    _select_formal_policies,
)


_DATA_ROOT = Path("/root/autodl-tmp")
_PROJECT_ROOT = _DATA_ROOT / "Res-AE-Attn"
_RESULTS_ROOT = _PROJECT_ROOT / "results"
_DEFAULT_ROOT = _RESULTS_ROOT / "nf-cse-a0-mr-safe-v1-seed42-pilot"
_SOURCE_CHECKPOINT = (
    _RESULTS_ROOT
    / "schemaflow-nf-unsw-pretrain-smoke-20260722T054417924340Z-"
    "9965a490-7de8cdfb4f42/checkpoints/epoch-0001.pt"
)
_BASE_SPLIT = (
    _PROJECT_ROOT
    / "data/splits/NF-CSE-CIC-IDS2018-v2/"
    "confirmatory-unused-1m-seed20260725-shared-coarse-v1"
)
_CURRENT_SELECTIONS = {
    1: (
        _RESULTS_ROOT
        / "nf-cse-formal-selection-seed-42-pct-01-"
        "20260725T182223764599Z-cd314e1615b4-b105ab16"
    ),
    5: (
        _RESULTS_ROOT
        / "nf-cse-formal-selection-seed-42-pct-05-"
        "20260725T185352547450Z-cd314e1615b4-1e50e6cd"
    ),
}
_TRAIN_PERCENTS = (1, 5)
_MODEL_SEED = 42
_MISSING_SEED = 42
_CONSISTENCY_WEIGHTS = (0.05, 0.10, 0.20)
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 42
_EARLY_STOPPING = {
    "early_stopping_patience": 2,
    "early_stopping_interval": 10,
    "early_stopping_min_delta": 0.0001,
}
_CURRENT_FORMAL_FACTS = {
    "scope": "previously_published_nf_cse_formal_test_aggregate",
    "source_accessed_by_safe_development": False,
    "macro_f1": {
        "clean": 0.87079,
        "missing_30pct": 0.75214,
        "missing_50pct": 0.60384,
    },
    "tcp_mean_delta_vs_xgboost": -0.01060,
    "missing_50_exploit_recall": 0.15518,
    "retention": {"missing_30pct": 0.86390, "missing_50pct": 0.69362},
}
_CANDIDATES = (
    {
        "id": "v0",
        "variant": "V0",
        "classification_loss": "weighted_cross_entropy",
        "class_balance": "inverse_frequency",
        "consistency_weight": 0.0,
        "change_count": 0,
        "train": False,
    },
    {
        "id": "v1-balanced-softmax",
        "variant": "V1",
        "classification_loss": "balanced_softmax",
        "class_balance": "none",
        "consistency_weight": 0.0,
        "change_count": 1,
        "train": True,
    },
    *tuple(
        {
            "id": f"v2-consistency-{str(weight).replace('.', 'p')}",
            "variant": "V2",
            "classification_loss": "weighted_cross_entropy",
            "class_balance": "inverse_frequency",
            "consistency_weight": weight,
            "change_count": 1,
            "train": True,
        }
        for weight in _CONSISTENCY_WEIGHTS
    ),
    *tuple(
        {
            "id": f"v3-balanced-consistency-{str(weight).replace('.', 'p')}",
            "variant": "V3",
            "classification_loss": "balanced_softmax",
            "class_balance": "none",
            "consistency_weight": weight,
            "change_count": 2,
            "train": True,
        }
        for weight in _CONSISTENCY_WEIGHTS
    ),
)
_CANDIDATE_BY_ID = {candidate["id"]: candidate for candidate in _CANDIDATES}
_RANDOM_CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
_TCP_CONDITION = "structured__tcp_state_retransmission"


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("plan", "train", "evaluate", "run"), nargs="?", default="run"
    )
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=_BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"required artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _git_state(repo_dir: Path):
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise RuntimeError("A0-MR-Safe execution requires a clean Git worktree")
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": False, "status": []}


def _validated_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_relative_to(_RESULTS_ROOT) or root == _RESULTS_ROOT:
        raise SystemExit(f"output-root must be below {_RESULTS_ROOT}")
    return root


def _forbidden_reason(path_value) -> str | None:
    if not isinstance(path_value, (str, bytes, os.PathLike)):
        return None
    try:
        path = Path(os.fsdecode(path_value)).expanduser().resolve()
    except (OSError, ValueError):
        return None
    base_forbidden = {
        _BASE_SPLIT / "sealed_test.json",
        _BASE_SPLIT / "split_manifest.json",
        _BASE_SPLIT / "selected_records.jsonl",
        _BASE_SPLIT / "split_summary.json",
    }
    if path in base_forbidden:
        return "sealed NF-CSE base split artifact"
    value = str(path)
    old_formal = (
        "nf-cse-formal-matrix-v2" in path.parts
        or "nf-cse-a0-mr-confirmation" in value
    )
    if old_formal and path.name == "metrics.json":
        return "existing NF-CSE formal metrics"
    if old_formal and "predictions" in path.parts:
        return "existing NF-CSE formal prediction archive"
    if old_formal and path.name == "formal_test_access.claim.json":
        return "existing NF-CSE formal test access claim"
    return None


def _install_read_guard() -> None:
    def audit(event, arguments):
        if event != "open" or not arguments:
            return
        reason = _forbidden_reason(arguments[0])
        if reason is not None:
            raise PermissionError(
                f"A0-MR-Safe development blocked access to {reason}: {arguments[0]}"
            )

    sys.addaudithook(audit)


def _candidate_config(base_config, candidate, train_percent: int, root: Path):
    payload = copy.deepcopy(base_config.to_dict())
    payload["experiment"]["name"] = (
        f"schemaflow-nf-cse-a0-mr-safe-seed42-pct-{train_percent:02d}-"
        f"{candidate['id']}"
    )
    payload["experiment"]["output_root"] = str(_RESULTS_ROOT)
    payload["experiment"]["seed"] = _MODEL_SEED
    payload["runtime"]["device"] = "cuda"
    payload["runtime"]["execution"] = "train"
    training = payload["training"]
    training["epochs"] = 50
    training["class_balance"] = candidate["class_balance"]
    training["classification_loss"] = candidate["classification_loss"]
    training["consistency_weight"] = candidate["consistency_weight"]
    training["consistency_missing_seed"] = _MISSING_SEED
    training["evaluation_scope"] = "validation_only"
    training.update(_EARLY_STOPPING)
    path = root / "configs" / f"pct-{train_percent:02d}-{candidate['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )
    loaded = load_config(path, expected_workflow="finetune")
    if loaded.to_dict() != payload:
        raise RuntimeError("generated Safe-v1 training config did not round-trip")
    return path


def _validate_current_selection(train_percent: int):
    directory = _CURRENT_SELECTIONS[train_percent]
    config_path = directory / "config.json"
    selection_path = directory / "selection.json"
    config = _read_json(config_path)
    selection = _read_json(selection_path)
    expected_flags = {
        "test_manifest_members_read": False,
        "sealed_test_file_content_read": False,
        "test_samples_materialized": False,
        "test_metrics_used": False,
    }
    for name, expected in expected_flags.items():
        source = selection if name == "test_metrics_used" else config
        if source.get(name) is not expected:
            raise RuntimeError(f"current selection violates sealed scope: {name}")
    if config.get("selection_scope") != ["train", "validation"]:
        raise RuntimeError("current baseline is not validation-only")
    if tuple(config.get("conditions", ())) != _FORMAL_CONDITIONS:
        raise RuntimeError("current baseline condition protocol changed")
    random_search = selection.get("random_condition_selection", {}).get(
        "search_details", {}
    )
    clean_linear = random_search.get("linear", {}).get("clean", {})
    expected_grid = list(default_alpha_grid(0.05))
    if (
        clean_linear.get("alpha_grid") != expected_grid
        or selection.get("structured_extension_rule", {}).get("alpha_grid")
        != expected_grid
    ):
        raise RuntimeError("current A0-MR alpha grid changed")
    expected_tie_break = [
        "mcc",
        "balanced_accuracy",
        "linear_pooling",
        "higher_xgboost_alpha",
    ]
    if (
        clean_linear.get("tie_break") != expected_tie_break
        or selection.get("structured_extension_rule", {}).get("tie_break")
        != expected_tie_break
    ):
        raise RuntimeError("current A0-MR tie-break changed")
    policies = selection.get("locked_policy", {}).get("policies", {})
    if set(policies) != set(_FORMAL_CONDITIONS):
        raise RuntimeError("current A0-MR does not lock all Pilot conditions")
    return directory, config, selection


def _initialize(root: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    git = _git_state(repo_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "protocol.lock.json"
    settings = []
    for train_percent in _TRAIN_PERCENTS:
        directory, config, selection = _validate_current_selection(train_percent)
        pretrained_config = Path(config["pretrained_config_path"]).resolve()
        base_config = load_config(pretrained_config, expected_workflow="finetune")
        if base_config.training.get("evaluation_scope") != "validation_only":
            raise RuntimeError("baseline training config is not validation-only")
        split_dir = Path(base_config.data["split_dir"]).resolve()
        summary_path = split_dir / "selection_summary.json"
        summary = _read_json(summary_path)
        if (
            summary.get("selection_scope") != "train_validation_only"
            or summary.get("materialized_splits") != ["train", "val"]
            or summary.get("test_labels_present") is not False
            or summary.get("test_statistics_present") is not False
        ):
            raise RuntimeError("few-shot selection view exposes test information")
        generated = {
            candidate["id"]: str(
                _candidate_config(base_config, candidate, train_percent, root)
            )
            for candidate in _CANDIDATES
            if candidate["train"]
        }
        settings.append(
            {
                "train_percent": train_percent,
                "train_rows": summary["split_counts"]["train"]["rows"],
                "validation_rows": summary["split_counts"]["val"]["rows"],
                "train_groups": summary["split_counts"]["train"]["groups"],
                "validation_groups": summary["split_counts"]["val"]["groups"],
                "current_selection_dir": str(directory),
                "current_selection_config": _artifact(directory / "config.json"),
                "current_selection": _artifact(directory / "selection.json"),
                "current_xgboost": _artifact(directory / "xgboost_model.json"),
                "current_encoder": _artifact(directory / "encoder.json"),
                "current_pretrained_config": _artifact(pretrained_config),
                "current_pretrained_checkpoint": _artifact(
                    Path(config["pretrained_checkpoint_path"])
                ),
                "selection_manifest": _artifact(split_dir / "selection_manifest.json"),
                "selection_records": _artifact(split_dir / "selection_records.jsonl"),
                "selection_summary": _artifact(summary_path),
                "generated_configs": generated,
                "locked_policies": selection["locked_policy"]["policies"],
            }
        )
    lock = {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "status": "locked_before_safe_pilot_results",
        "git": git,
        "model_seed": _MODEL_SEED,
        "data_split_seed_semantics": "reuse_frozen_seed42_group_safe_views",
        "missing_seed": _MISSING_SEED,
        "train_percents": list(_TRAIN_PERCENTS),
        "conditions": list(_FORMAL_CONDITIONS),
        "variants": list(_CANDIDATES),
        "consistency_weight_grid": list(_CONSISTENCY_WEIGHTS),
        "alpha_grid": list(default_alpha_grid(0.05)),
        "pooling_methods": ["linear", "log"],
        "fusion_tie_break": [
            "mcc",
            "balanced_accuracy",
            "linear_pooling",
            "higher_xgboost_alpha",
        ],
        "candidate_tie_break": [
            "higher_mean_random_macro_f1",
            "higher_min_random_macro_f1",
            "higher_mean_all_condition_macro_f1",
            "higher_mean_all_condition_mcc",
            "lower_consistency_weight",
            "fewer_objective_changes",
            "earlier_predeclared_candidate_order",
        ],
        "early_stopping": _EARLY_STOPPING,
        "bootstrap": {
            "resamples": _BOOTSTRAP_RESAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "cluster": "leakage_group",
        },
        "safe_route": {
            "bootstrap_ci95_lower_must_exceed_zero": True,
            "minimum_rare_validation_support": MIN_RARE_VALIDATION_SUPPORT,
            "maximum_rare_recall_drop": MAX_RARE_RECALL_DROP,
            "rare_class_collapse": "xgboost_recall_at_least_0p10_and_candidate_zero",
            "fallback": "xgboost",
        },
        "calibration": {
            "type": "scalar_temperature",
            "fit_scope": "validation_only_after_route_lock",
            "objective": "nll",
            "used_for_macro_f1_selection": False,
        },
        "continuation_thresholds": {
            "mean_random_macro_f1_gain": 0.005,
            "minimum_random_condition_delta": -0.002,
            "tcp_not_below_xgboost": True,
            "maximum_rare_recall_drop": 0.02,
            "missing_30_retention_gain_or_absolute": [0.02, 0.90],
            "missing_50_retention_gain": 0.02,
            "calibrated_nll_strictly_improves": True,
            "calibrated_ece_not_worse": True,
            "no_rare_class_collapse": True,
        },
        "settings": settings,
        "source_pretrain_checkpoint": _artifact(_SOURCE_CHECKPOINT),
        "current_formal_facts": _CURRENT_FORMAL_FACTS,
        "forbidden_access": {
            "base_split_files": [
                str(_BASE_SPLIT / name)
                for name in (
                    "sealed_test.json",
                    "split_manifest.json",
                    "selected_records.jsonl",
                    "split_summary.json",
                )
            ],
            "existing_formal_metrics_json": True,
            "existing_formal_prediction_archives": True,
            "training_subprocess_environment_guard": (
                "FLOWFM_NF_CSE_SAFE_DEVELOPMENT=1"
            ),
            "test_probabilities_generated": False,
        },
        "paper_draft": {
            "allowed": False,
            "until": "all_four_dataset_formal_experiments_complete",
        },
    }
    if lock_path.exists():
        existing = _read_json(lock_path)
        if existing != lock:
            raise RuntimeError("existing Safe-v1 protocol lock differs from code/provenance")
    else:
        _write_json(lock_path, lock)
        (root / "protocol.lock.sha256").write_text(
            _sha256_file(lock_path) + "  protocol.lock.json\n", encoding="ascii"
        )
    budget = {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "recorded_before_gpu_execution": True,
        "new_training_runs": sum(candidate["train"] for candidate in _CANDIDATES)
        * len(_TRAIN_PERCENTS),
        "maximum_epochs_per_run": 50,
        "training_rows_by_percent": {
            str(setting["train_percent"]): setting["train_rows"] for setting in settings
        },
        "validation_rows_per_condition": settings[0]["validation_rows"],
        "conditions": len(_FORMAL_CONDITIONS),
        "candidate_validation_schemaflow_record_evaluations": (
            len(_CANDIDATES)
            * len(_TRAIN_PERCENTS)
            * len(_FORMAL_CONDITIONS)
            * settings[0]["validation_rows"]
        ),
        "estimated_gpu_wall_hours_range": [6.0, 14.0],
        "estimated_peak_gpu_memory_gib": 8.0,
        "estimated_result_storage_gib": 6.0,
        "gpu_over_two_hours_is_not_a_stop_condition": True,
    }
    _write_json(root / "budget.json", budget)
    (root / "PAPER_DRAFT_BLOCKED.txt").write_text(
        "Paper drafting is blocked until formal experiments finish on all four datasets.\n",
        encoding="utf-8",
    )
    return {"status": "safe_pilot_planned", "root": str(root), "budget": budget}


def _state_path(root: Path, train_percent: int, candidate_id: str):
    return root / "state" / f"pct-{train_percent:02d}-{candidate_id}.json"


def _load_lock(root: Path):
    path = root / "protocol.lock.json"
    if not path.is_file():
        raise RuntimeError("Safe-v1 protocol is not initialized")
    expected = (root / "protocol.lock.sha256").read_text(encoding="ascii").split()[0]
    if _sha256_file(path) != expected:
        raise RuntimeError("Safe-v1 protocol lock changed")
    lock = _read_json(path)
    if lock.get("git", {}).get("commit") != _git_state(
        Path(__file__).resolve().parents[1]
    )["commit"]:
        raise RuntimeError("Safe-v1 protocol lock belongs to another Git commit")
    return lock


def _run_logged(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["FLOWFM_NF_CSE_SAFE_DEVELOPMENT"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        tuple(str(value) for value in command),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"Safe-v1 subprocess failed ({completed.returncode}); see {log_path}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Safe-v1 subprocess returned no JSON: {log_path}")
    try:
        result = json.loads(lines[-1])
    except ValueError as exc:
        raise RuntimeError(f"Safe-v1 subprocess result is not JSON: {log_path}") from exc
    return result, time.perf_counter() - started


def _completed_training(root: Path, train_percent: int, candidate_id: str):
    path = _state_path(root, train_percent, candidate_id)
    if not path.is_file():
        return None
    state = _read_json(path)
    if state.get("status") != "completed":
        return None
    for artifact in state.get("artifacts", {}).values():
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_file() or _sha256_file(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"completed Safe-v1 training artifact changed: {artifact_path}")
    return state


def _train(root: Path):
    lock = _load_lock(root)
    settings = {value["train_percent"]: value for value in lock["settings"]}
    completed = 0
    for train_percent in _TRAIN_PERCENTS:
        for candidate in _CANDIDATES:
            if not candidate["train"]:
                continue
            existing = _completed_training(root, train_percent, candidate["id"])
            if existing is not None:
                completed += 1
                continue
            config_path = Path(
                settings[train_percent]["generated_configs"][candidate["id"]]
            )
            result, wall = _run_logged(
                (
                    sys.executable,
                    "-m",
                    "flowfm",
                    "finetune",
                    "--config",
                    config_path,
                    "--initialize-from",
                    _SOURCE_CHECKPOINT,
                    "--allow-cross-dataset-init",
                ),
                root / "logs" / f"pct-{train_percent:02d}-{candidate['id']}.log",
            )
            run_dir = Path(result["run_dir"]).resolve()
            checkpoint = Path(result["checkpoint"]).resolve()
            metrics = _read_json(run_dir / "metrics.json")
            if metrics.get("evaluation_scope") != "validation_only" or "test" in metrics:
                raise RuntimeError("Safe-v1 training checkpoint contains test evaluation")
            state = {
                "format_version": 1,
                "protocol_version": SAFE_PROTOCOL_VERSION,
                "status": "completed",
                "train_percent": train_percent,
                "candidate": candidate,
                "wall_seconds": wall,
                "run_dir": str(run_dir),
                "selected_epoch": metrics["early_stopping"]["selected_epoch"],
                "last_trained_epoch": metrics["early_stopping"]["last_trained_epoch"],
                "artifacts": {
                    "checkpoint": _artifact(checkpoint),
                    "config": _artifact(config_path),
                    "metrics": _artifact(run_dir / "metrics.json"),
                    "training_objective": _artifact(
                        run_dir / "training_objective.json"
                    ),
                },
                "test_probabilities_generated": False,
            }
            if candidate["consistency_weight"] > 0:
                state["artifacts"]["paired_consistency"] = _artifact(
                    run_dir / "paired_consistency.json"
                )
            _write_json(_state_path(root, train_percent, candidate["id"]), state)
            completed += 1
    return {"status": "safe_pilot_training_completed", "runs": completed}


def _metrics(probabilities, targets):
    return confirmatory_metric_payload(
        probabilities, targets, tuple(SUPPORTED_COARSE_LABELS)
    )


def _probability_hashes(probabilities):
    return {
        condition: probability_sha256(values)
        for condition, values in probabilities.items()
    }


def _restore_candidate(root, train_percent, candidate, tokenizer, device):
    selection_config = _read_json(_CURRENT_SELECTIONS[train_percent] / "config.json")
    if candidate["id"] == "v0":
        config_path = Path(selection_config["pretrained_config_path"])
        checkpoint = Path(selection_config["pretrained_checkpoint_path"])
    else:
        state = _completed_training(root, train_percent, candidate["id"])
        if state is None:
            raise RuntimeError(f"Safe-v1 candidate is not trained: {candidate['id']}")
        config_path = Path(state["artifacts"]["config"]["path"])
        checkpoint = Path(state["artifacts"]["checkpoint"]["path"])
    config = load_config(config_path, expected_workflow="finetune")
    model, payload, adaptation = _restore_model(config, checkpoint, tokenizer, device)
    if payload["extra_state"]["metrics"].get("evaluation_scope") != "validation_only":
        raise RuntimeError("candidate checkpoint is not validation-only")
    if "test" in payload["extra_state"]["metrics"]:
        raise RuntimeError("candidate checkpoint exposes test metrics")
    if candidate["id"] != "v0":
        objective = payload["extra_state"].get("training_objective", {})
        if (
            objective.get("classification_loss") != candidate["classification_loss"]
            or objective.get("class_balance") != candidate["class_balance"]
            or not np.isclose(
                objective.get("paired_consistency", {}).get("weight", -1),
                candidate["consistency_weight"],
                atol=0,
                rtol=0,
            )
        ):
            raise RuntimeError("candidate checkpoint objective provenance changed")
    return model, config, checkpoint, payload, adaptation


def _setting_inputs(train_percent: int):
    selection_dir, selection_config, current_selection = _validate_current_selection(
        train_percent
    )
    scratch_config = Path(selection_config["scratch_config_path"])
    pretrained_config = Path(selection_config["pretrained_config_path"])
    inputs = _load_selection_inputs(scratch_config, pretrained_config)
    manifest = SplitManifest.from_json(
        inputs["split_manifest_path"].read_text(encoding="utf-8")
    )
    validation = inputs["dataset"].splits["val"]
    groups = tuple(manifest.sample_groups[sample.sample_id] for sample in validation)
    if len(groups) != len(validation) or len(set(groups)) < 2:
        raise RuntimeError("validation leakage groups are unavailable")
    conditions, missingness = _formal_conditions(validation, _MISSING_SEED)
    return (
        selection_dir,
        selection_config,
        current_selection,
        inputs,
        groups,
        conditions,
        missingness,
    )


def _common_outputs(train_percent: int, device):
    (
        selection_dir,
        selection_config,
        current_selection,
        inputs,
        groups,
        conditions,
        missingness,
    ) = _setting_inputs(train_percent)
    encoder = _encoder_from_lock(
        _read_json(selection_dir / "encoder.json"), inputs["tokenizer"]
    )
    xgboost = XGBClassifier()
    xgboost.load_model(str(selection_dir / "xgboost_model.json"))
    current_model, current_config, current_checkpoint, _, _ = _restore_candidate(
        Path("."), train_percent, _CANDIDATES[0], inputs["tokenizer"], device
    )
    outputs = {}
    current_neural = {}
    xgboost_probabilities = {}
    for condition, samples in conditions.items():
        targets = _coarse_targets(samples)
        projected = _model_visible_samples(samples)
        xgb = _xgboost_probabilities(xgboost, encoder.transform(projected))
        neural, _ = _schemaflow_probabilities(
            current_model,
            samples,
            inputs["tokenizer"],
            device,
            current_config.training["batch_size"],
        )
        xgboost_probabilities[condition] = xgb
        current_neural[condition] = neural
        outputs[condition] = {
            "xgboost": xgb,
            "schemaflow_pretrained": neural,
            "targets": targets,
        }
    del current_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    policies = current_selection["locked_policy"]["policies"]
    current_a0_mr = {
        condition: pool_probabilities(
            outputs[condition]["xgboost"],
            outputs[condition]["schemaflow_pretrained"],
            alpha=policies[condition]["alpha"],
            method=policies[condition]["method"],
        )
        for condition in _FORMAL_CONDITIONS
    }
    return {
        "selection_dir": selection_dir,
        "selection_config": selection_config,
        "current_selection": current_selection,
        "inputs": inputs,
        "groups": groups,
        "conditions": conditions,
        "missingness": missingness,
        "outputs": outputs,
        "xgboost": xgboost_probabilities,
        "current_neural": current_neural,
        "current_a0_mr": current_a0_mr,
        "current_checkpoint": current_checkpoint,
    }


def _evaluate_candidate(root, train_percent, candidate, common, device, resamples):
    if candidate["id"] == "v0":
        neural = common["current_neural"]
        checkpoint = common["current_checkpoint"]
    else:
        model, config, checkpoint, _, _ = _restore_candidate(
            root, train_percent, candidate, common["inputs"]["tokenizer"], device
        )
        neural = {}
        for condition, samples in common["conditions"].items():
            neural[condition], _ = _schemaflow_probabilities(
                model,
                samples,
                common["inputs"]["tokenizer"],
                device,
                config.training["batch_size"],
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    outputs = {
        condition: {
            "xgboost": common["xgboost"][condition],
            "schemaflow_pretrained": neural[condition],
            "targets": common["outputs"][condition]["targets"],
        }
        for condition in _FORMAL_CONDITIONS
    }
    selection = _select_formal_policies(outputs)
    if candidate["id"] == "v0" and selection["locked_policy"]["policies"] != (
        common["current_selection"]["locked_policy"]["policies"]
    ):
        raise RuntimeError("recomputed V0 A0-MR policy differs from frozen baseline")
    policies = selection["locked_policy"]["policies"]
    fused = {
        condition: pool_probabilities(
            common["xgboost"][condition],
            neural[condition],
            alpha=policies[condition]["alpha"],
            method=policies[condition]["method"],
        )
        for condition in _FORMAL_CONDITIONS
    }
    route, routed, calibrated = safe_route_and_calibrate(
        outputs,
        fused,
        common["xgboost"],
        common["groups"],
        tuple(SUPPORTED_COARSE_LABELS),
        rare_labels=NF_CSE_RARE_COARSE_LABELS,
        bootstrap_resamples=resamples,
        bootstrap_seed=_BOOTSTRAP_SEED,
    )
    result = {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "selection_scope": "validation_only",
        "train_percent": train_percent,
        "candidate": candidate,
        "checkpoint": _artifact(checkpoint),
        "a0_mr_selection": selection,
        "safe_route_and_calibration": route,
        "schemaflow_metrics": {
            condition: _metrics(neural[condition], outputs[condition]["targets"])
            for condition in _FORMAL_CONDITIONS
        },
        "prediction_hashes": {
            "schemaflow": _probability_hashes(neural),
            "a0_mr_candidate": _probability_hashes(fused),
            "safe_routed": _probability_hashes(routed),
            "safe_calibrated": _probability_hashes(calibrated),
        },
        "test_manifest_read": False,
        "test_labels_read": False,
        "test_probabilities_generated": False,
    }
    return result


def _candidate_score(results):
    random_metrics = [
        result["safe_route_and_calibration"]["routed_metrics"][condition]
        for result in results
        for condition in _RANDOM_CONDITIONS
    ]
    all_metrics = [
        metric
        for result in results
        for metric in result["safe_route_and_calibration"]["routed_metrics"].values()
    ]
    candidate = results[0]["candidate"]
    index = next(
        position
        for position, value in enumerate(_CANDIDATES)
        if value["id"] == candidate["id"]
    )
    values = {
        "mean_random_macro_f1": float(
            np.mean([metric["macro_f1"] for metric in random_metrics])
        ),
        "minimum_random_macro_f1": float(
            min(metric["macro_f1"] for metric in random_metrics)
        ),
        "mean_all_condition_macro_f1": float(
            np.mean([metric["macro_f1"] for metric in all_metrics])
        ),
        "mean_all_condition_mcc": float(
            np.mean([metric["mcc"] for metric in all_metrics])
        ),
        "consistency_weight": candidate["consistency_weight"],
        "change_count": candidate["change_count"],
        "predeclared_index": index,
    }
    score = (
        values["mean_random_macro_f1"],
        values["minimum_random_macro_f1"],
        values["mean_all_condition_macro_f1"],
        values["mean_all_condition_mcc"],
        -values["consistency_weight"],
        -values["change_count"],
        -index,
    )
    return score, values


def _aggregate_gate(results, current_metrics):
    random_deltas = []
    by_setting_deltas = {}
    for result, current in zip(results, current_metrics, strict=True):
        train_percent = result["train_percent"]
        safe = result["safe_route_and_calibration"]["routed_metrics"]
        values = {
            condition: float(
                safe[condition]["macro_f1"] - current[condition]["macro_f1"]
            )
            for condition in _RANDOM_CONDITIONS
        }
        by_setting_deltas[str(train_percent)] = values
        random_deltas.extend(values.values())
    current_condition_means = {
        condition: float(np.mean([value[condition]["macro_f1"] for value in current_metrics]))
        for condition in _RANDOM_CONDITIONS
    }
    safe_condition_means = {
        condition: float(
            np.mean(
                [
                    result["safe_route_and_calibration"]["routed_metrics"][condition][
                        "macro_f1"
                    ]
                    for result in results
                ]
            )
        )
        for condition in _RANDOM_CONDITIONS
    }
    current_retention = {
        condition: current_condition_means[condition] / current_condition_means["clean"]
        for condition in _RANDOM_CONDITIONS[1:]
    }
    safe_retention = {
        condition: safe_condition_means[condition] / safe_condition_means["clean"]
        for condition in _RANDOM_CONDITIONS[1:]
    }
    rare_failures = []
    rare_zero_out = []
    tcp_deltas = {}
    before_nll = []
    after_nll = []
    before_ece = []
    after_ece = []
    for result in results:
        train_percent = result["train_percent"]
        payload = result["safe_route_and_calibration"]
        safe = payload["routed_metrics"]
        calibrated = payload["calibrated_metrics"]
        xgb = payload["xgboost_metrics"]
        tcp_deltas[str(train_percent)] = float(
            safe[_TCP_CONDITION]["macro_f1"] - xgb[_TCP_CONDITION]["macro_f1"]
        )
        for condition in _FORMAL_CONDITIONS:
            before_nll.append(safe[condition]["nll"])
            after_nll.append(calibrated[condition]["nll"])
            before_ece.append(safe[condition]["ece"])
            after_ece.append(calibrated[condition]["ece"])
            for label in NF_CSE_RARE_COARSE_LABELS:
                if safe[condition]["per_class_support"].get(label, 0) < (
                    MIN_RARE_VALIDATION_SUPPORT
                ):
                    continue
                delta = float(
                    safe[condition]["per_class_recall"][label]
                    - xgb[condition]["per_class_recall"][label]
                )
                if delta < -MAX_RARE_RECALL_DROP:
                    rare_failures.append(
                        {
                            "train_percent": train_percent,
                            "condition": condition,
                            "label": label,
                            "recall_delta": delta,
                        }
                    )
                if (
                    xgb[condition]["per_class_recall"][label] >= 0.10
                    and safe[condition]["per_class_recall"][label] <= 0.0
                ):
                    rare_zero_out.append(
                        {
                            "train_percent": train_percent,
                            "condition": condition,
                            "label": label,
                        }
                    )
    calibration = {
        "mean_nll_before": float(np.mean(before_nll)),
        "mean_nll_after": float(np.mean(after_nll)),
        "mean_ece_before": float(np.mean(before_ece)),
        "mean_ece_after": float(np.mean(after_ece)),
    }
    checks = {
        "mean_random_macro_f1_gain_at_least_0p005": float(np.mean(random_deltas))
        >= 0.005,
        "no_random_condition_drop_below_minus_0p002": min(random_deltas) >= -0.002,
        "tcp_not_below_xgboost": min(tcp_deltas.values()) >= -1e-12,
        "rare_recall_guard": not rare_failures,
        "missing_30_retention_improved_or_at_least_0p90": (
            safe_retention["missing_30pct"] >= 0.90
            or safe_retention["missing_30pct"]
            - current_retention["missing_30pct"]
            >= 0.02
        ),
        "missing_50_retention_improved_at_least_0p02": (
            safe_retention["missing_50pct"] - current_retention["missing_50pct"]
            >= 0.02
        ),
        "calibration_mean_nll_improved": calibration["mean_nll_after"]
        < calibration["mean_nll_before"] - 1e-12,
        "calibration_mean_ece_not_worse": calibration["mean_ece_after"]
        <= calibration["mean_ece_before"] + 1e-12,
        "no_rare_class_collapse": not rare_zero_out,
    }
    return {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "predeclared": True,
        "aggregation": "equal_weight_across_1pct_5pct_settings_and_conditions",
        "random_deltas_by_setting": by_setting_deltas,
        "mean_random_macro_f1_gain": float(np.mean(random_deltas)),
        "minimum_random_condition_delta": float(min(random_deltas)),
        "current_condition_mean_macro_f1": current_condition_means,
        "safe_condition_mean_macro_f1": safe_condition_means,
        "current_retention": current_retention,
        "safe_retention": safe_retention,
        "tcp_macro_f1_deltas_vs_xgboost": tcp_deltas,
        "rare_recall_failures": rare_failures,
        "rare_zero_out_failures": rare_zero_out,
        "calibration": calibration,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _markdown_report(report):
    gate = report["seed42_gate"]
    selected = report["selected_candidate"]
    status = "PASS" if gate["passed"] else "FAIL"
    rows = [
        "# A0-MR-Safe-v1 NF-CSE seed=42 Pilot",
        "",
        f"Status: **{status}**",
        "",
        "This report is validation-only. No NF-CSE sealed-test manifest, labels, "
        "predictions, or formal metrics.json were read.",
        "",
        f"Selected candidate: `{selected['id']}` ({selected['variant']})",
        f"Mean random macro-F1 gain vs current A0-MR: "
        f"{gate['mean_random_macro_f1_gain']:.6f}",
        f"30% retention: {gate['safe_retention']['missing_30pct']:.6f}",
        f"50% retention: {gate['safe_retention']['missing_50pct']:.6f}",
        "",
        "## Gate",
        "",
    ]
    rows.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in gate["checks"].items()
    )
    rows.extend(
        [
            "",
            "## Consequence",
            "",
            (
                "The seed=42 gate passed; the frozen candidate may proceed to the "
                "three-seed stability stage."
                if gate["passed"]
                else "The seed=42 gate failed. Other model seeds and NF-BoT confirmation "
                "are blocked; current A0-MR remains the downstream main framework."
            ),
            "",
            "Paper drafting remains blocked until all four dataset experiments finish.",
        ]
    )
    return "\n".join(rows) + "\n"


def _evaluate(root: Path, resamples: int):
    _load_lock(root)
    if resamples != _BOOTSTRAP_RESAMPLES:
        raise SystemExit(
            f"bootstrap-resamples is frozen at {_BOOTSTRAP_RESAMPLES}"
        )
    if torch.cuda.is_available() is False:
        raise RuntimeError("Safe-v1 evaluation requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    summaries = {candidate["id"]: [] for candidate in _CANDIDATES}
    current_metrics = []
    setting_reports = []
    for train_percent in _TRAIN_PERCENTS:
        common = _common_outputs(train_percent, device)
        current = {
            condition: _metrics(
                common["current_a0_mr"][condition],
                common["outputs"][condition]["targets"],
            )
            for condition in _FORMAL_CONDITIONS
        }
        current_metrics.append(current)
        base_report = {
            "format_version": 1,
            "train_percent": train_percent,
            "current_a0_mr_metrics": current,
            "xgboost_metrics": {
                condition: _metrics(
                    common["xgboost"][condition],
                    common["outputs"][condition]["targets"],
                )
                for condition in _FORMAL_CONDITIONS
            },
            "current_prediction_hashes": {
                "xgboost": _probability_hashes(common["xgboost"]),
                "schemaflow_v0": _probability_hashes(common["current_neural"]),
                "a0_mr": _probability_hashes(common["current_a0_mr"]),
            },
            "missingness": common["missingness"],
            "validation_rows": len(common["groups"]),
            "validation_groups": len(set(common["groups"])),
        }
        _write_json(
            root / "evaluation" / f"pct-{train_percent:02d}" / "baseline.json",
            base_report,
        )
        for candidate in _CANDIDATES:
            result = _evaluate_candidate(
                root, train_percent, candidate, common, device, resamples
            )
            summaries[candidate["id"]].append(result)
            _write_json(
                root
                / "evaluation"
                / f"pct-{train_percent:02d}"
                / f"{candidate['id']}.json",
                result,
            )
        setting_reports.append(base_report)
        del common
        torch.cuda.empty_cache()
    candidate_scores = {}
    score_values = {}
    for candidate in _CANDIDATES:
        score, values = _candidate_score(summaries[candidate["id"]])
        candidate_scores[candidate["id"]] = score
        score_values[candidate["id"]] = values
    selected_id = max(candidate_scores, key=candidate_scores.__getitem__)
    selected = _CANDIDATE_BY_ID[selected_id]
    selected_results = summaries[selected_id]
    gate = _aggregate_gate(selected_results, current_metrics)
    component = {}
    v0_mean = score_values["v0"]["mean_random_macro_f1"]
    for variant in ("V1", "V2", "V3"):
        members = [
            candidate
            for candidate in _CANDIDATES
            if candidate["variant"] == variant
        ]
        best = max(
            members,
            key=lambda candidate: candidate_scores[candidate["id"]],
        )
        component[variant] = {
            "best_candidate": best,
            "mean_random_macro_f1": score_values[best["id"]][
                "mean_random_macro_f1"
            ],
            "gain_vs_v0_safe_route": (
                score_values[best["id"]]["mean_random_macro_f1"] - v0_mean
            ),
            "seed42_gate_if_selected": _aggregate_gate(
                summaries[best["id"]], current_metrics
            ),
        }
    report = {
        "format_version": 1,
        "protocol_version": SAFE_PROTOCOL_VERSION,
        "status": "seed42_passed" if gate["passed"] else "seed42_failed_stop",
        "selection_scope": "NF-CSE_train_validation_only",
        "selected_candidate": selected,
        "candidate_scores": score_values,
        "component_analysis": component,
        "seed42_gate": gate,
        "settings": setting_reports,
        "selected_setting_results": selected_results,
        "efficiency": {
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_system_memory_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "sealed_test_audit": {
            "guard_installed": True,
            "base_test_manifest_read": False,
            "base_sealed_test_read": False,
            "existing_formal_metrics_json_read": False,
            "existing_prediction_archive_read": False,
            "test_labels_read": False,
            "test_probabilities_generated": False,
        },
        "continuation": {
            "run_model_seeds_17_42_20260725": gate["passed"],
            "run_nf_bot_confirmation": False,
            "reason": (
                "seed42_gate_passed_three_seed_stage_required_next"
                if gate["passed"]
                else "seed42_gate_failed_protocol_stops_extension"
            ),
        },
        "paper_draft_allowed": False,
    }
    report_path = root / "seed42_pilot_report.json"
    _write_json(report_path, report)
    (root / "seed42_pilot_report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    if gate["passed"]:
        _write_json(
            root / "THREE_SEED_REQUIRED.json",
            {
                "status": "required_before_nf_bot",
                "selected_candidate": selected,
                "model_seeds": [17, 42, 20260725],
                "test_access_allowed": False,
            },
        )
    else:
        _write_json(
            root / "STOP_AFTER_SEED42.json",
            {
                "status": "stopped_by_predeclared_seed42_gate",
                "failed_checks": [
                    name for name, passed in gate["checks"].items() if not passed
                ],
                "other_model_seeds_started": False,
                "nf_bot_started": False,
                "current_a0_mr_retained": True,
            },
        )
    return {
        "status": report["status"],
        "report": str(report_path),
        "selected_candidate": selected_id,
        "passed": gate["passed"],
    }


def main():
    args = _args()
    root = _validated_root(args.output_root)
    _install_read_guard()
    planned = _initialize(root)
    if args.phase == "plan":
        result = planned
    elif args.phase == "train":
        result = _train(root)
    elif args.phase == "evaluate":
        result = _evaluate(root, args.bootstrap_resamples)
    else:
        _train(root)
        result = _evaluate(root, args.bootstrap_resamples)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
