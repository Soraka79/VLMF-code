#!/usr/bin/env python3
"""Lock and evaluate a globally tuned validation-only fusion baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score

from flowfm.experiment.metadata import write_json_snapshot
from flowfm.training.probability_fusion import (
    default_alpha_grid,
    pool_probabilities,
)


EXPERIMENT_NAME = "global_validation_fusion_v1"
PROTOCOL_VERSION = "global-validation-fusion-v1"
FORMAT_VERSION = 1
CONDITIONS = ("clean", "missing_30pct", "missing_50pct")
METHODS = ("linear", "log")
ALPHA_GRID = default_alpha_grid(0.05)
TRAIN_PERCENTS = (1, 5, 10)
MODEL_SEEDS = (17, 42, 20260725)
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260730
REPRODUCTION_TOLERANCE = 1e-12
DIRECTION_TOLERANCE = 1e-12
TIE_BREAK = (
    "three_condition_mean_macro_f1",
    "three_condition_mean_mcc",
    "three_condition_mean_balanced_accuracy",
    "linear_before_log",
    "higher_xgboost_alpha",
)
DATASETS = MappingProxyType(
    {
        "NF-BoT": {
            "dataset_id": "NF-BoT-IoT-v2",
            "matrix_key": "nf_bot",
            "matrix_relative_path": "results/cross-dataset-formal-matrix-v1/nf_bot",
        },
        "NF-CSE": {
            "dataset_id": "NF-CSE-CIC-IDS2018-v2",
            "matrix_key": "nf_cse",
            "matrix_relative_path": "results/cross-dataset-formal-matrix-v1/nf_cse",
        },
        "NF-ToN": {
            "dataset_id": "NF-ToN-IoT-v2",
            "matrix_key": "nf_ton",
            "matrix_relative_path": "results/cross-dataset-formal-matrix-v3/nf_ton",
        },
    }
)
FINAL_OUTPUT_NAMES = (
    "README.md",
    "audit.json",
    "cross_dataset_summary.json",
    "global_validation_policies.lock.json",
    "global_validation_policies.lock.sha256",
    "input_manifest.json",
    "per_setting_metrics.json",
    "protocol.json",
    "protocol.sha256",
    "tidy_metrics.csv",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> Path:
    return write_json_snapshot(path, payload)


def _write_sha256_sidecar(path: Path) -> tuple[Path, str]:
    digest = _sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return sidecar, digest


def _verify_sha256_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"invalid SHA-256 sidecar for {path.name}")
    observed = _sha256_file(path)
    if fields[0] != observed:
        raise RuntimeError(f"SHA-256 sidecar mismatch for {path.name}")
    return observed


def _git_state(repo_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def _setting_id(train_percent: int, model_seed: int) -> str:
    return f"pct-{train_percent:02d}-seed-{model_seed}"


def expected_settings() -> tuple[dict[str, int | str], ...]:
    return tuple(
        {
            "setting_id": _setting_id(percent, seed),
            "train_percent": percent,
            "model_seed": seed,
        }
        for percent in TRAIN_PERCENTS
        for seed in MODEL_SEEDS
    )


def protocol_payload() -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "test whether VLMF condition indexing outperforms ordinary global "
            "validation tuning"
        ),
        "scope": {
            "datasets": [
                {
                    "name": name,
                    "dataset_id": spec["dataset_id"],
                    "matrix_relative_path": spec["matrix_relative_path"],
                }
                for name, spec in DATASETS.items()
            ],
            "train_percents": list(TRAIN_PERCENTS),
            "model_seeds": list(MODEL_SEEDS),
            "setting_count": 27,
            "conditions": list(CONDITIONS),
            "base_models_retrained": False,
            "excluded_analyses": [
                "stacking",
                "learned_gate",
                "dynamic_ensemble",
                "branch_complementarity",
                "structured_missingness",
                "calibration",
                "threshold_sensitivity",
            ],
        },
        "selection": {
            "scope": "one_global_policy_per_dataset_train_fraction_model_seed",
            "source": "selection.json:random_condition_selection.linear_or_log",
            "validation_conditions": list(CONDITIONS),
            "condition_weighting": "equal",
            "methods": list(METHODS),
            "alpha_grid": list(ALPHA_GRID),
            "alpha_semantics": "xgboost_probability_weight",
            "objective": "three_condition_mean_macro_f1",
            "tie_break": list(TIE_BREAK),
            "float_comparison": "exact_binary64_lexicographic",
            "vlmf_admission_gate_used": False,
            "fixed_alpha_fallback_used": False,
            "same_policy_applied_to_all_test_conditions": True,
        },
        "lock_boundary": {
            "protocol_written_before_validation_input_reads": True,
            "policy_lock_written_before_any_sealed_test_read": True,
            "test_labels_used_only_after_policy_lock": True,
            "test_metrics_used_only_for_post_lock_vlmf_reproduction": True,
        },
        "evaluation": {
            "prediction_source": "sealed predictions.npz",
            "probability_calibration": "uncalibrated",
            "global_pool_implementation": (
                "flowfm.training.probability_fusion.pool_probabilities"
            ),
            "comparison_probability_key": "uncalibrated__a0_mr__{condition}",
            "vlmf_macro_f1_reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "abort_before_global_result_generation_on_reproduction_failure": True,
        },
        "aggregation": {
            "condition_cell": (
                "dataset_x_train_fraction_x_model_seed_x_condition"
            ),
            "setting_mean": "equal_mean_over_clean_missing30_missing50",
            "dataset_mean": "equal_mean_over_nine_setting_means",
            "cross_dataset_mean": "equal_mean_over_three_dataset_means",
            "direction_zero_tolerance": DIRECTION_TOLERANCE,
            "bootstrap": {
                "method": "dataset_stratified_paired_setting_bootstrap",
                "within_dataset_units": 9,
                "within_dataset_sampling": "with_replacement",
                "dataset_weighting": "equal",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile_95pct",
                "p_value_inferred": False,
            },
        },
        "reporting": {
            "report_all_results_regardless_of_direction": True,
            "baseline_definition_may_change_after_test": False,
        },
    }


class AccessAudit:
    """Record and enforce the validation-lock/sealed-test access boundary."""

    def __init__(self, events: Sequence[Mapping[str, object]] | None = None):
        self.events = [dict(event) for event in (events or ())]
        self.protocol_locked = any(
            event.get("event") == "protocol_locked" for event in self.events
        )
        self.policies_locked = any(
            event.get("event") == "policies_locked" for event in self.events
        )
        self.test_access_enabled = any(
            event.get("event") == "sealed_test_access_enabled"
            for event in self.events
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

    def lock_protocol(self, path: Path, sha256: str) -> None:
        if self.protocol_locked or self.events:
            raise RuntimeError("protocol must be the first audited event")
        self.protocol_locked = True
        self._record("protocol_locked", path=str(path), sha256=sha256)

    def validation_read(self, path: Path, role: str) -> None:
        if not self.protocol_locked or self.policies_locked:
            raise RuntimeError("validation reads require only the protocol lock")
        if "sealed_test" in Path(path).parts:
            raise RuntimeError("sealed-test paths are forbidden during lock")
        self._record("validation_input_read", path=str(path), role=role)

    def lock_policies(self, path: Path, sha256: str) -> None:
        if not self.protocol_locked or self.policies_locked:
            raise RuntimeError("policy lock boundary is invalid")
        self.policies_locked = True
        self._record("policies_locked", path=str(path), sha256=sha256)

    def enable_test_access(self, protocol_sha256: str, policies_sha256: str) -> None:
        if not self.protocol_locked or not self.policies_locked:
            raise RuntimeError("test access requires protocol and policy locks")
        if self.test_access_enabled:
            raise RuntimeError("sealed-test access was already enabled")
        self.test_access_enabled = True
        self._record(
            "sealed_test_access_enabled",
            verified_protocol_sha256=protocol_sha256,
            verified_policies_sha256=policies_sha256,
        )

    def test_read(self, path: Path, role: str) -> None:
        if not self.test_access_enabled:
            raise RuntimeError("sealed-test input read attempted before policy lock")
        self._record("sealed_test_input_read", path=str(path), role=role)

    def record(self, event: str, **details) -> None:
        self._record(event, **details)


def _artifact_entry(
    path: Path,
    *,
    role: str,
    dataset: str | None = None,
    setting_id: str | None = None,
    observed_sha256: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"required input is missing: {path}")
    observed = observed_sha256 or _sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"locked input hash mismatch: {path}")
    return {
        "path": str(path),
        "role": role,
        "dataset": dataset,
        "setting_id": setting_id,
        "bytes": path.stat().st_size,
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "hash_verified": expected_sha256 is None or observed == expected_sha256,
    }


def _candidate_key(method: str, alpha: float) -> tuple[str, float]:
    return method, round(float(alpha), 12)


def _validate_condition_candidates(
    condition_searches: Mapping[str, object],
    *,
    methods: Sequence[str] = METHODS,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> dict[str, dict[tuple[str, float], dict[str, float | str]]]:
    if tuple(condition_searches) != CONDITIONS:
        raise RuntimeError("validation conditions are incomplete or out of order")
    expected_keys = {
        _candidate_key(method, alpha)
        for method in methods
        for alpha in alpha_grid
    }
    result = {}
    for condition in CONDITIONS:
        search = condition_searches[condition]
        if tuple(search.get("methods", ())) != tuple(methods):
            raise RuntimeError(f"{condition} pooling methods changed")
        if tuple(search.get("alpha_grid", ())) != tuple(alpha_grid):
            raise RuntimeError(f"{condition} alpha grid changed")
        candidates = {}
        for row in search.get("candidates", ()):
            key = _candidate_key(row.get("method"), row.get("alpha"))
            if key in candidates:
                raise RuntimeError(f"duplicate candidate for {condition}: {key}")
            metrics = {}
            for metric in ("macro_f1", "mcc", "balanced_accuracy"):
                value = float(row.get(metric))
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite {metric} for {condition}: {key}")
                metrics[metric] = value
            candidates[key] = {
                "method": key[0],
                "alpha": key[1],
                **metrics,
            }
        if set(candidates) != expected_keys:
            missing = sorted(expected_keys - set(candidates))
            extra = sorted(set(candidates) - expected_keys)
            raise RuntimeError(
                f"candidate pool mismatch for {condition}; missing={missing}, extra={extra}"
            )
        result[condition] = candidates
    return result


def _ranking_key(candidate: Mapping[str, object]) -> tuple[float, ...]:
    method_preference = 1.0 if candidate["method"] == "linear" else 0.0
    return (
        float(candidate["mean_macro_f1"]),
        float(candidate["mean_mcc"]),
        float(candidate["mean_balanced_accuracy"]),
        method_preference,
        float(candidate["alpha"]),
    )


def _deciding_criterion(first: Mapping[str, object], second: Mapping[str, object]) -> str:
    first_key = _ranking_key(first)
    second_key = _ranking_key(second)
    for name, left, right in zip(TIE_BREAK, first_key, second_key):
        if left != right:
            return name
    raise RuntimeError("top two global candidates have identical ranking keys")


def select_global_policy(
    condition_searches: Mapping[str, object],
    *,
    methods: Sequence[str] = METHODS,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> dict[str, object]:
    """Select one policy by an equal mean over the three validation conditions."""

    validated = _validate_condition_candidates(
        condition_searches,
        methods=methods,
        alpha_grid=alpha_grid,
    )
    candidates = []
    for method in methods:
        for alpha in alpha_grid:
            key = _candidate_key(method, alpha)
            by_condition = {
                condition: validated[condition][key] for condition in CONDITIONS
            }
            candidates.append(
                {
                    "method": method,
                    "alpha": float(alpha),
                    "mean_macro_f1": float(
                        np.mean(
                            [
                                by_condition[condition]["macro_f1"]
                                for condition in CONDITIONS
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "mean_mcc": float(
                        np.mean(
                            [
                                by_condition[condition]["mcc"]
                                for condition in CONDITIONS
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "mean_balanced_accuracy": float(
                        np.mean(
                            [
                                by_condition[condition]["balanced_accuracy"]
                                for condition in CONDITIONS
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "condition_metrics": {
                        condition: {
                            metric: float(by_condition[condition][metric])
                            for metric in (
                                "macro_f1",
                                "mcc",
                                "balanced_accuracy",
                            )
                        }
                        for condition in CONDITIONS
                    },
                }
            )
    ranked = sorted(candidates, key=_ranking_key, reverse=True)
    ranking = [
        {
            "rank": index,
            "method": row["method"],
            "alpha": row["alpha"],
            "mean_macro_f1": row["mean_macro_f1"],
            "mean_mcc": row["mean_mcc"],
            "mean_balanced_accuracy": row["mean_balanced_accuracy"],
        }
        for index, row in enumerate(ranked, start=1)
    ]
    selected = ranked[0]
    runner_up = ranked[1]
    return {
        "policy": {
            "method": selected["method"],
            "alpha": selected["alpha"],
            "alpha_semantics": "xgboost_probability_weight",
            "applies_to_test_conditions": list(CONDITIONS),
        },
        "validation_objective": {
            "mean_macro_f1": selected["mean_macro_f1"],
            "mean_mcc": selected["mean_mcc"],
            "mean_balanced_accuracy": selected["mean_balanced_accuracy"],
            "condition_metrics": selected["condition_metrics"],
        },
        "selection_basis": {
            "candidate_count": len(ranked),
            "ranking": list(TIE_BREAK),
            "deciding_criterion_vs_runner_up": _deciding_criterion(
                selected, runner_up
            ),
            "runner_up": ranking[1],
            "complete_candidate_ranking": ranking,
        },
    }


def _lock_audit_payload(
    audit: AccessAudit,
    *,
    protocol_sha256: str,
    policies_sha256: str,
    output_dir: Path,
    git: Mapping[str, object],
) -> dict[str, object]:
    policy_sequence = next(
        event["sequence"]
        for event in audit.events
        if event["event"] == "policies_locked"
    )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "phase": "locked",
        "protocol_sha256": protocol_sha256,
        "policies_sha256": policies_sha256,
        "git": dict(git),
        "events": audit.events,
        "assertions": {
            "protocol_is_first_event": audit.events[0]["event"]
            == "protocol_locked",
            "protocol_locked_before_validation_reads": all(
                event["sequence"] > audit.events[0]["sequence"]
                for event in audit.events
                if event["event"] == "validation_input_read"
            ),
            "policy_lock_sequence": policy_sequence,
            "sealed_test_files_read_before_policy_lock": 0,
            "sealed_test_access_enabled": False,
            "output_directory": str(output_dir),
        },
    }


def run_lock(project_root: Path, output_dir: Path, repo_root: Path) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    if project_root not in output_dir.parents:
        raise RuntimeError("output directory must be inside the project root")
    matrix_roots = {
        dataset: (project_root / spec["matrix_relative_path"]).resolve()
        for dataset, spec in DATASETS.items()
    }
    if any(
        root == output_dir
        or root in output_dir.parents
        or output_dir in root.parents
        for root in matrix_roots.values()
    ):
        raise RuntimeError("output directory overlaps a sealed source matrix")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / ".lock.incomplete").write_text("lock incomplete\n", encoding="ascii")

    audit = AccessAudit()
    protocol_path = _write_json(output_dir / "protocol.json", protocol_payload())
    _, protocol_sha256 = _write_sha256_sidecar(protocol_path)
    audit.lock_protocol(protocol_path, protocol_sha256)

    validation_entries = []
    policies = []
    source_protocol_hashes = set()
    for dataset, spec in DATASETS.items():
        matrix_root = matrix_roots[dataset]
        validation_lock_path = matrix_root / "all_validation_selections.lock.json"
        audit.validation_read(validation_lock_path, "validation_selection_registry")
        validation_lock = _read_json(validation_lock_path)
        validation_entries.append(
            _artifact_entry(
                validation_lock_path,
                role="validation_selection_registry",
                dataset=dataset,
            )
        )
        if (
            validation_lock.get("status")
            != "all_nine_validation_selections_locked_before_test"
            or validation_lock.get("dataset") != spec["matrix_key"]
            or validation_lock.get("setting_count") != 9
        ):
            raise RuntimeError(f"invalid validation registry for {dataset}")
        registry = {
            row["setting_id"]: row for row in validation_lock.get("settings", ())
        }
        expected_ids = {row["setting_id"] for row in expected_settings()}
        if set(registry) != expected_ids:
            raise RuntimeError(f"validation settings are incomplete for {dataset}")

        for setting in expected_settings():
            setting_id = setting["setting_id"]
            registry_row = registry[setting_id]
            if (
                registry_row.get("train_percent") != setting["train_percent"]
                or registry_row.get("model_seed") != setting["model_seed"]
            ):
                raise RuntimeError(f"setting identity mismatch for {dataset}/{setting_id}")
            selection_path = matrix_root / "selections" / setting_id / "selection.json"
            if Path(registry_row.get("selection_path", "")).resolve() != selection_path:
                raise RuntimeError(f"selection path mismatch for {dataset}/{setting_id}")
            audit.validation_read(selection_path, "validation_selection")
            selection_sha256 = _sha256_file(selection_path)
            if selection_sha256 != registry_row.get("selection_sha256"):
                raise RuntimeError(f"selection hash mismatch for {dataset}/{setting_id}")
            selection = _read_json(selection_path)
            if (
                selection.get("selection_scope") != "validation_only"
                or selection.get("test_samples_materialized") is not False
                or selection.get("test_metrics_used") is not False
            ):
                raise RuntimeError(f"selection boundary invalid for {dataset}/{setting_id}")
            source_protocol_hashes.add(selection.get("protocol_hash"))
            searches = selection.get("random_condition_selection", {}).get(
                "linear_or_log"
            )
            selected = select_global_policy(searches)
            policies.append(
                {
                    "dataset": dataset,
                    "dataset_id": spec["dataset_id"],
                    **setting,
                    "selection_input": {
                        "path": str(selection_path),
                        "bytes": selection_path.stat().st_size,
                        "sha256": selection_sha256,
                    },
                    **selected,
                }
            )
            validation_entries.append(
                _artifact_entry(
                    selection_path,
                    role="validation_selection",
                    dataset=dataset,
                    setting_id=setting_id,
                    observed_sha256=selection_sha256,
                    expected_sha256=registry_row["selection_sha256"],
                )
            )

    if len(source_protocol_hashes) != 1 or None in source_protocol_hashes:
        raise RuntimeError("source VLMF protocol hashes are inconsistent")
    if len(policies) != 27:
        raise RuntimeError("exactly 27 global policies are required")
    git = _git_state(repo_root)
    policies_path = _write_json(
        output_dir / "global_validation_policies.lock.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256,
            "source_vlmf_protocol_sha256": next(iter(source_protocol_hashes)),
            "selection_scope": "validation_only",
            "test_files_read": False,
            "setting_count": len(policies),
            "one_policy_per_setting": True,
            "same_policy_for_all_three_test_conditions": True,
            "git": git,
            "policies": policies,
        },
    )
    _, policies_sha256 = _write_sha256_sidecar(policies_path)
    audit.lock_policies(policies_path, policies_sha256)
    _write_json(
        output_dir / "input_manifest.validation.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "phase": "validation_lock",
            "entry_count": len(validation_entries),
            "entries": sorted(validation_entries, key=lambda row: row["path"]),
        },
    )
    _write_json(
        output_dir / "audit.json",
        _lock_audit_payload(
            audit,
            protocol_sha256=protocol_sha256,
            policies_sha256=policies_sha256,
            output_dir=output_dir,
            git=git,
        ),
    )
    (output_dir / ".lock.incomplete").unlink()
    return {
        "status": "global_validation_policies_locked",
        "output_dir": str(output_dir),
        "protocol_sha256": protocol_sha256,
        "policies_sha256": policies_sha256,
        "setting_count": len(policies),
        "test_files_read": False,
        "git": git,
    }


def core_metrics(probabilities, targets, class_count: int) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or values.shape != (target_values.size, class_count):
        raise RuntimeError("probability and target shapes do not match")
    if not np.isfinite(values).all() or not np.allclose(
        values.sum(axis=1), 1.0, atol=1e-6, rtol=0
    ):
        raise RuntimeError("invalid probability matrix")
    labels = np.arange(class_count, dtype=np.int64)
    predictions = values.argmax(axis=1)
    recalls = recall_score(
        target_values,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                target_values,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(target_values, predictions)),
        "balanced_accuracy": float(recalls.mean()),
    }


def evaluate_global_policy(
    probabilities_by_condition: Mapping[str, Mapping[str, np.ndarray]],
    targets,
    class_count: int,
    policy: Mapping[str, object],
) -> dict[str, object]:
    if tuple(probabilities_by_condition) != CONDITIONS:
        raise RuntimeError("test conditions are incomplete or out of order")
    result = {}
    for condition in CONDITIONS:
        values = probabilities_by_condition[condition]
        pooled = pool_probabilities(
            values["xgboost"],
            values["schemaflow_pretrained"],
            method=policy["method"],
            alpha=policy["alpha"],
        )
        result[condition] = {
            "applied_policy": {
                "method": policy["method"],
                "alpha": float(policy["alpha"]),
            },
            "metrics": core_metrics(pooled, targets, class_count),
        }
    return result


def dataset_stratified_paired_setting_bootstrap(
    deltas_by_dataset: Mapping[str, Sequence[float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    pairing_unit: str = (
        "same_setting_three_condition_mean_vlmf_minus_global_validation_fusion"
    ),
) -> dict[str, object]:
    if type(resamples) is not int or resamples < 2_000:
        raise ValueError("bootstrap requires at least 2,000 resamples")
    if set(deltas_by_dataset) != set(DATASETS):
        raise ValueError("bootstrap requires exactly the three declared datasets")
    arrays = {}
    for dataset in DATASETS:
        values = np.asarray(deltas_by_dataset[dataset], dtype=np.float64)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("each dataset requires nine finite setting deltas")
        arrays[dataset] = values
    observed_by_dataset = {
        dataset: float(values.mean()) for dataset, values in arrays.items()
    }
    observed = float(np.mean(list(observed_by_dataset.values()), dtype=np.float64))
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        strata = [
            float(values[rng.integers(0, 9, size=9)].mean())
            for values in arrays.values()
        ]
        samples[index] = float(np.mean(strata, dtype=np.float64))
    return {
        "format_version": FORMAT_VERSION,
        "method": "dataset_stratified_paired_setting_bootstrap",
        "pairing_unit": pairing_unit,
        "within_dataset_units": 9,
        "within_dataset_sampling": "with_replacement",
        "dataset_weighting": "equal",
        "resamples": resamples,
        "seed": seed,
        "observed_by_dataset": observed_by_dataset,
        "observed_equal_dataset_mean": observed,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "p_value_inferred": False,
    }


def _verify_summary_artifact(record, expected_path: Path, description: str) -> str:
    if Path(record.get("path", "")).resolve() != expected_path.resolve():
        raise RuntimeError(f"{description} path changed")
    if expected_path.stat().st_size != record.get("bytes"):
        raise RuntimeError(f"{description} size changed")
    observed = _sha256_file(expected_path)
    if observed != record.get("sha256"):
        raise RuntimeError(f"{description} hash changed")
    return observed


def _load_evaluation_inputs(
    project_root: Path,
    policies: Sequence[Mapping[str, object]],
    audit: AccessAudit,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inputs = []
    manifest_entries = []
    policies_by_dataset = {
        dataset: {
            row["setting_id"]: row
            for row in policies
            if row["dataset"] == dataset
        }
        for dataset in DATASETS
    }
    for dataset, spec in DATASETS.items():
        matrix_root = project_root / spec["matrix_relative_path"]
        summary_path = matrix_root / "formal_matrix_summary.json"
        audit.test_read(summary_path, "sealed_formal_matrix_summary")
        summary_sha = _sha256_file(summary_path)
        summary = _read_json(summary_path)
        manifest_entries.append(
            _artifact_entry(
                summary_path,
                role="sealed_formal_matrix_summary",
                dataset=dataset,
                observed_sha256=summary_sha,
            )
        )
        if (
            summary.get("status") != "formal_matrix_completed"
            or summary.get("dataset") != spec["matrix_key"]
            or summary.get("setting_count") != 9
        ):
            raise RuntimeError(f"formal matrix summary invalid for {dataset}")
        result_artifacts = {
            row["setting_id"]: row for row in summary.get("result_artifacts", ())
        }
        if set(result_artifacts) != set(policies_by_dataset[dataset]):
            raise RuntimeError(f"sealed settings are incomplete for {dataset}")
        for setting in expected_settings():
            setting_id = setting["setting_id"]
            policy_row = policies_by_dataset[dataset][setting_id]
            sealed_dir = matrix_root / "sealed_test" / setting_id
            artifacts = result_artifacts[setting_id]
            prediction_lock_path = sealed_dir / "prediction_lock.json"
            metrics_path = sealed_dir / "metrics.json"
            predictions_path = sealed_dir / "predictions.npz"
            audit.test_read(prediction_lock_path, "sealed_prediction_lock")
            prediction_lock_sha = _verify_summary_artifact(
                artifacts["prediction_lock"],
                prediction_lock_path,
                f"{dataset}/{setting_id} prediction lock",
            )
            prediction_lock = _read_json(prediction_lock_path)
            manifest_entries.append(
                _artifact_entry(
                    prediction_lock_path,
                    role="sealed_prediction_lock",
                    dataset=dataset,
                    setting_id=setting_id,
                    observed_sha256=prediction_lock_sha,
                    expected_sha256=artifacts["prediction_lock"]["sha256"],
                )
            )
            audit.test_read(metrics_path, "sealed_reference_metrics")
            metrics_sha = _verify_summary_artifact(
                artifacts["metrics"],
                metrics_path,
                f"{dataset}/{setting_id} metrics",
            )
            metrics = _read_json(metrics_path)
            manifest_entries.append(
                _artifact_entry(
                    metrics_path,
                    role="sealed_reference_metrics",
                    dataset=dataset,
                    setting_id=setting_id,
                    observed_sha256=metrics_sha,
                    expected_sha256=artifacts["metrics"]["sha256"],
                )
            )
            audit.test_read(predictions_path, "sealed_prediction_archive_hash")
            predictions_sha = _sha256_file(predictions_path)
            expected_predictions_sha = prediction_lock.get(
                "prediction_archive_sha256"
            )
            if (
                predictions_sha != expected_predictions_sha
                or predictions_path.stat().st_size
                != prediction_lock.get("prediction_archive_bytes")
            ):
                raise RuntimeError(
                    f"sealed prediction archive changed for {dataset}/{setting_id}"
                )
            manifest_entries.append(
                _artifact_entry(
                    predictions_path,
                    role="sealed_prediction_archive",
                    dataset=dataset,
                    setting_id=setting_id,
                    observed_sha256=predictions_sha,
                    expected_sha256=expected_predictions_sha,
                )
            )
            if (
                metrics.get("dataset") != spec["matrix_key"]
                or metrics.get("setting_id") != setting_id
                or metrics.get("train_percent") != setting["train_percent"]
                or metrics.get("model_seed") != setting["model_seed"]
            ):
                raise RuntimeError(f"sealed metric identity changed for {dataset}/{setting_id}")
            inputs.append(
                {
                    "dataset": dataset,
                    "dataset_id": spec["dataset_id"],
                    **setting,
                    "policy_row": policy_row,
                    "prediction_path": predictions_path,
                    "reference_metrics": metrics,
                    "label_names": tuple(metrics["label_names"]),
                }
            )
    return inputs, manifest_entries


def _reproduce_vlmf(
    inputs: Sequence[Mapping[str, object]], audit: AccessAudit
) -> dict[tuple[str, str], dict[str, object]]:
    reproduced = {}
    for row in inputs:
        path = row["prediction_path"]
        audit.test_read(path, "sealed_uncalibrated_vlmf_probabilities_and_labels")
        with np.load(path, allow_pickle=False) as archive:
            targets = np.asarray(archive["targets"], dtype=np.int64)
            class_count = len(row["label_names"])
            by_condition = {}
            for condition in CONDITIONS:
                key = f"uncalibrated__a0_mr__{condition}"
                if key not in archive.files:
                    raise RuntimeError(f"missing VLMF probability key: {key}")
                metrics = core_metrics(archive[key], targets, class_count)
                reference = row["reference_metrics"]["uncalibrated"]["a0_mr"][
                    condition
                ]["macro_f1"]
                error = abs(metrics["macro_f1"] - float(reference))
                if error > REPRODUCTION_TOLERANCE:
                    raise RuntimeError(
                        "VLMF macro-F1 reproduction failed for "
                        f"{row['dataset']}/{row['setting_id']}/{condition}: {error}"
                    )
                by_condition[condition] = {
                    "metrics": metrics,
                    "reference_macro_f1": float(reference),
                    "absolute_reproduction_error": error,
                }
        reproduced[(row["dataset"], row["setting_id"])] = by_condition
    audit.record(
        "vlmf_macro_f1_reproduction_completed",
        setting_count=len(inputs),
        condition_cell_count=len(inputs) * len(CONDITIONS),
        tolerance=REPRODUCTION_TOLERANCE,
        passed=True,
    )
    return reproduced


def _evaluate_global(
    inputs: Sequence[Mapping[str, object]], audit: AccessAudit
) -> dict[tuple[str, str], dict[str, object]]:
    evaluated = {}
    for row in inputs:
        path = row["prediction_path"]
        audit.test_read(
            path,
            "sealed_uncalibrated_branch_probabilities_and_labels",
        )
        with np.load(path, allow_pickle=False) as archive:
            targets = np.asarray(archive["targets"], dtype=np.int64)
            probabilities = {}
            for condition in CONDITIONS:
                xgb_key = f"uncalibrated__xgboost__{condition}"
                neural_key = f"uncalibrated__schemaflow_pretrained__{condition}"
                if xgb_key not in archive.files or neural_key not in archive.files:
                    raise RuntimeError(
                        f"missing branch probabilities for {row['dataset']}/{row['setting_id']}"
                    )
                probabilities[condition] = {
                    "xgboost": archive[xgb_key],
                    "schemaflow_pretrained": archive[neural_key],
                }
            result = evaluate_global_policy(
                probabilities,
                targets,
                len(row["label_names"]),
                row["policy_row"]["policy"],
            )
        applied = {
            (values["applied_policy"]["method"], values["applied_policy"]["alpha"])
            for values in result.values()
        }
        if len(applied) != 1:
            raise RuntimeError("a setting used more than one global test policy")
        evaluated[(row["dataset"], row["setting_id"])] = result
    audit.record(
        "global_validation_fusion_evaluation_completed",
        setting_count=len(inputs),
        condition_cell_count=len(inputs) * len(CONDITIONS),
        same_policy_per_setting_verified=True,
    )
    return evaluated


def _build_metrics(
    inputs: Sequence[Mapping[str, object]],
    reproduced: Mapping[tuple[str, str], Mapping[str, object]],
    global_results: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    tidy_rows = []
    setting_rows = []
    for row in inputs:
        key = (row["dataset"], row["setting_id"])
        conditions = {}
        for condition in CONDITIONS:
            vlmf = reproduced[key][condition]
            global_value = global_results[key][condition]
            delta = (
                vlmf["metrics"]["macro_f1"]
                - global_value["metrics"]["macro_f1"]
            )
            conditions[condition] = {
                "vlmf": vlmf,
                "global_validation_fusion": global_value,
                "vlmf_minus_global_macro_f1": delta,
            }
            tidy_rows.append(
                {
                    "experiment_name": EXPERIMENT_NAME,
                    "dataset": row["dataset"],
                    "dataset_id": row["dataset_id"],
                    "setting_id": row["setting_id"],
                    "train_percent": row["train_percent"],
                    "model_seed": row["model_seed"],
                    "condition": condition,
                    "global_pool_method": row["policy_row"]["policy"]["method"],
                    "global_xgboost_alpha": row["policy_row"]["policy"]["alpha"],
                    "vlmf_macro_f1": vlmf["metrics"]["macro_f1"],
                    "global_validation_fusion_macro_f1": global_value["metrics"][
                        "macro_f1"
                    ],
                    "vlmf_minus_global_macro_f1": delta,
                    "vlmf_mcc": vlmf["metrics"]["mcc"],
                    "global_validation_fusion_mcc": global_value["metrics"]["mcc"],
                    "vlmf_balanced_accuracy": vlmf["metrics"][
                        "balanced_accuracy"
                    ],
                    "global_validation_fusion_balanced_accuracy": global_value[
                        "metrics"
                    ]["balanced_accuracy"],
                    "vlmf_reference_macro_f1": vlmf["reference_macro_f1"],
                    "vlmf_reproduction_absolute_error": vlmf[
                        "absolute_reproduction_error"
                    ],
                }
            )
        vlmf_mean = float(
            np.mean(
                [conditions[name]["vlmf"]["metrics"]["macro_f1"] for name in CONDITIONS]
            )
        )
        global_mean = float(
            np.mean(
                [
                    conditions[name]["global_validation_fusion"]["metrics"][
                        "macro_f1"
                    ]
                    for name in CONDITIONS
                ]
            )
        )
        setting_rows.append(
            {
                "dataset": row["dataset"],
                "dataset_id": row["dataset_id"],
                "setting_id": row["setting_id"],
                "train_percent": row["train_percent"],
                "model_seed": row["model_seed"],
                "global_policy": row["policy_row"]["policy"],
                "validation_objective": row["policy_row"]["validation_objective"],
                "selection_basis": {
                    "candidate_count": row["policy_row"]["selection_basis"][
                        "candidate_count"
                    ],
                    "ranking": row["policy_row"]["selection_basis"]["ranking"],
                    "deciding_criterion_vs_runner_up": row["policy_row"][
                        "selection_basis"
                    ]["deciding_criterion_vs_runner_up"],
                    "runner_up": row["policy_row"]["selection_basis"]["runner_up"],
                },
                "conditions": conditions,
                "three_condition_mean": {
                    "vlmf_macro_f1": vlmf_mean,
                    "global_validation_fusion_macro_f1": global_mean,
                    "vlmf_minus_global_macro_f1": vlmf_mean - global_mean,
                },
            }
        )

    dataset_condition_rows = []
    dataset_rows = []
    deltas_by_dataset = {}
    for dataset in DATASETS:
        dataset_settings = [row for row in setting_rows if row["dataset"] == dataset]
        setting_deltas = [
            row["three_condition_mean"]["vlmf_minus_global_macro_f1"]
            for row in dataset_settings
        ]
        deltas_by_dataset[dataset] = setting_deltas
        for condition in CONDITIONS:
            cells = [
                row
                for row in tidy_rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            vlmf_mean = float(np.mean([row["vlmf_macro_f1"] for row in cells]))
            global_mean = float(
                np.mean([row["global_validation_fusion_macro_f1"] for row in cells])
            )
            dataset_condition_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "setting_count": len(cells),
                    "vlmf_mean_macro_f1": vlmf_mean,
                    "global_validation_fusion_mean_macro_f1": global_mean,
                    "vlmf_minus_global_mean_macro_f1": vlmf_mean - global_mean,
                }
            )
        dataset_rows.append(
            {
                "dataset": dataset,
                "setting_count": len(dataset_settings),
                "three_condition_mean_vlmf_minus_global_macro_f1": float(
                    np.mean(setting_deltas)
                ),
            }
        )

    all_setting_deltas = [
        row["three_condition_mean"]["vlmf_minus_global_macro_f1"]
        for row in setting_rows
    ]
    direction_counts = {
        "positive": sum(value > DIRECTION_TOLERANCE for value in all_setting_deltas),
        "zero": sum(abs(value) <= DIRECTION_TOLERANCE for value in all_setting_deltas),
        "negative": sum(value < -DIRECTION_TOLERANCE for value in all_setting_deltas),
    }
    bootstrap = dataset_stratified_paired_setting_bootstrap(deltas_by_dataset)
    policy_counts = Counter(
        (
            row["global_policy"]["method"],
            float(row["global_policy"]["alpha"]),
        )
        for row in setting_rows
    )
    cross_summary = {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "setting_count": len(setting_rows),
        "condition_cell_count": len(tidy_rows),
        "dataset_condition_macro_f1": dataset_condition_rows,
        "dataset_three_condition_mean_differences": dataset_rows,
        "setting_three_condition_mean_differences": [
            {
                "dataset": row["dataset"],
                "setting_id": row["setting_id"],
                "train_percent": row["train_percent"],
                "model_seed": row["model_seed"],
                **row["three_condition_mean"],
            }
            for row in setting_rows
        ],
        "setting_direction_counts": {
            **direction_counts,
            "zero_tolerance": DIRECTION_TOLERANCE,
        },
        "three_dataset_equal_weight_mean_vlmf_minus_global_macro_f1": bootstrap[
            "observed_equal_dataset_mean"
        ],
        "dataset_stratified_paired_setting_bootstrap": bootstrap,
        "global_policy_frequencies": [
            {"method": method, "alpha": alpha, "setting_count": count}
            for (method, alpha), count in sorted(policy_counts.items())
        ],
        "vlmf_reproduction": {
            "passed": True,
            "condition_cell_count": len(tidy_rows),
            "maximum_absolute_macro_f1_error": max(
                row["vlmf_reproduction_absolute_error"] for row in tidy_rows
            ),
            "tolerance": REPRODUCTION_TOLERANCE,
        },
        "p_value_inferred": False,
    }
    per_setting = {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "setting_count": len(setting_rows),
        "settings": setting_rows,
    }
    return tidy_rows, per_setting, cross_summary


def _write_tidy_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("tidy metrics cannot be empty")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _readme_text(summary: Mapping[str, object]) -> str:
    bootstrap = summary["dataset_stratified_paired_setting_bootstrap"]
    directions = summary["setting_direction_counts"]
    dataset_lines = "\n".join(
        "- {dataset}: {three_condition_mean_vlmf_minus_global_macro_f1:+.8f}".format(
            **row
        )
        for row in summary["dataset_three_condition_mean_differences"]
    )
    return f"""# Global Validation Fusion v1

This directory is the immutable output of the validation-selected global fusion
baseline. One `linear` or `log` pool and one XGBoost weight are selected per
dataset/train-fraction/model-seed setting by equal validation macro-F1 over
`clean`, `missing_30pct`, and `missing_50pct`. The same locked policy is applied
to all three sealed-test conditions.

## Reproduction

From `/root/autodl-tmp/Res-AE-Attn/repository`:

```bash
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.run_global_validation_fusion lock \\
  --project-root /root/autodl-tmp/Res-AE-Attn \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/global_validation_fusion_v1
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.run_global_validation_fusion evaluate \\
  --project-root /root/autodl-tmp/Res-AE-Attn \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/global_validation_fusion_v1
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m unittest \\
  tests.test_global_validation_fusion -v
```

The `lock` command refuses an existing output directory and reads validation
selection files only. The `evaluate` command verifies the protocol and policy
SHA-256 sidecars before reading any sealed-test artifact.

## Result Definition

Differences are `VLMF - global validation fusion` macro-F1. Each setting is an
equal mean over the three main conditions. Each dataset is an equal mean over
nine settings. The cross-dataset estimate equally weights the three datasets.

{dataset_lines}

- Three-dataset equal mean: {summary['three_dataset_equal_weight_mean_vlmf_minus_global_macro_f1']:+.8f}
- 95% bootstrap CI: [{bootstrap['ci95_lower']:+.8f}, {bootstrap['ci95_upper']:+.8f}]
- Setting directions (positive/zero/negative): {directions['positive']}/{directions['zero']}/{directions['negative']}
- Bootstrap: {bootstrap['resamples']:,} resamples, seed {bootstrap['seed']}; no p-value inferred

`input_manifest.json` records every consumed path, byte size, and SHA-256.
`audit.json` records the access order and input immutability verification.
"""


def _verify_manifest_unchanged(entries: Sequence[Mapping[str, object]]) -> None:
    for entry in entries:
        path = Path(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha256_file(path) != entry["sha256"]
        ):
            raise RuntimeError(f"input changed during evaluation: {path}")


def run_evaluate(
    project_root: Path, output_dir: Path, repo_root: Path
) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if not output_dir.is_dir() or (output_dir / ".lock.incomplete").exists():
        raise RuntimeError("completed lock output is required")
    forbidden_existing = [
        name
        for name in (
            "tidy_metrics.csv",
            "per_setting_metrics.json",
            "cross_dataset_summary.json",
            "input_manifest.json",
            "README.md",
            "output_hashes.sha256",
            ".evaluate.incomplete",
        )
        if (output_dir / name).exists()
    ]
    if forbidden_existing:
        raise RuntimeError(f"refusing to overwrite evaluation outputs: {forbidden_existing}")

    protocol_path = output_dir / "protocol.json"
    policies_path = output_dir / "global_validation_policies.lock.json"
    protocol_sha256 = _verify_sha256_sidecar(protocol_path)
    policies_sha256 = _verify_sha256_sidecar(policies_path)
    if _read_json(protocol_path) != protocol_payload():
        raise RuntimeError("protocol payload does not match the implemented protocol")
    policies_payload = _read_json(policies_path)
    if (
        policies_payload.get("protocol_sha256") != protocol_sha256
        or policies_payload.get("setting_count") != 27
        or len(policies_payload.get("policies", ())) != 27
        or policies_payload.get("test_files_read") is not False
    ):
        raise RuntimeError("global policy lock is invalid")
    lock_audit = _read_json(output_dir / "audit.json")
    if (
        lock_audit.get("phase") != "locked"
        or lock_audit.get("assertions", {}).get(
            "sealed_test_files_read_before_policy_lock"
        )
        != 0
    ):
        raise RuntimeError("lock audit is invalid")
    audit = AccessAudit(lock_audit["events"])
    audit.enable_test_access(protocol_sha256, policies_sha256)
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

    validation_manifest = _read_json(output_dir / "input_manifest.validation.json")
    validation_entries = validation_manifest["entries"]
    _verify_manifest_unchanged(validation_entries)
    inputs, test_entries = _load_evaluation_inputs(
        project_root, policies_payload["policies"], audit
    )
    reproduced = _reproduce_vlmf(inputs, audit)
    global_results = _evaluate_global(inputs, audit)
    tidy_rows, per_setting, cross_summary = _build_metrics(
        inputs, reproduced, global_results
    )
    manifest_entries = sorted(
        [*validation_entries, *test_entries], key=lambda row: row["path"]
    )
    _verify_manifest_unchanged(manifest_entries)
    audit.record(
        "all_input_hashes_reverified_after_evaluation",
        entry_count=len(manifest_entries),
        sealed_prediction_archive_count=sum(
            row["role"] == "sealed_prediction_archive" for row in manifest_entries
        ),
        unchanged=True,
    )

    _write_tidy_csv(output_dir / "tidy_metrics.csv", tidy_rows)
    _write_json(output_dir / "per_setting_metrics.json", per_setting)
    _write_json(output_dir / "cross_dataset_summary.json", cross_summary)
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
    (output_dir / "README.md").write_text(
        _readme_text(cross_summary), encoding="utf-8"
    )

    policy_sequence = next(
        event["sequence"]
        for event in audit.events
        if event["event"] == "policies_locked"
    )
    first_test_sequence = min(
        event["sequence"]
        for event in audit.events
        if event["event"] == "sealed_test_input_read"
    )
    final_audit = {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "phase": "evaluated",
        "protocol_sha256": protocol_sha256,
        "policies_sha256": policies_sha256,
        "events": audit.events,
        "assertions": {
            **lock_audit["assertions"],
            "sealed_test_access_enabled": True,
            "first_sealed_test_read_sequence": first_test_sequence,
            "all_sealed_test_reads_after_policy_lock": first_test_sequence
            > policy_sequence,
            "vlmf_macro_f1_reproduced": True,
            "vlmf_reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "setting_count": 27,
            "one_policy_per_setting": True,
            "same_policy_for_all_three_test_conditions": True,
            "all_input_hashes_verified": True,
            "all_inputs_unchanged_after_evaluation": True,
            "source_matrix_files_written": False,
            "output_directory_disjoint_from_source_matrices": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "p_value_inferred": False,
        },
        "git": _git_state(repo_root),
    }
    _write_json(output_dir / "audit.json", final_audit)
    (output_dir / ".evaluate.incomplete").unlink()

    hashes = []
    for name in FINAL_OUTPUT_NAMES:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"required output missing: {name}")
        hashes.append(f"{_sha256_file(path)}  {name}")
    (output_dir / "output_hashes.sha256").write_text(
        "\n".join(hashes) + "\n", encoding="ascii"
    )
    return {
        "status": "global_validation_fusion_completed",
        "output_dir": str(output_dir),
        "protocol_sha256": protocol_sha256,
        "policies_sha256": policies_sha256,
        "cross_dataset_summary": cross_summary,
        "git": _git_state(repo_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lock and evaluate the global validation fusion baseline"
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
            default=(
                Path("/root/autodl-tmp/Res-AE-Attn/results") / EXPERIMENT_NAME
            ),
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
