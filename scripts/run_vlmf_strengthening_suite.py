#!/usr/bin/env python3
"""Lock and evaluate the validation-controlled VLMF strengthening suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from flowfm.training.probability_fusion import pool_probabilities
from flowfm.training.vlmf_strengthening import (
    ALPHA_GRID,
    GLOBAL_RANKING,
    MAIN_CONDITIONS,
    POOL_METHODS,
    RANKING,
    candidate_key,
    candidate_ranking_key,
    complementarity_payload,
    global_ranking_key,
    rank_condition_candidates,
    rank_global_candidates,
    route_stability_record,
    select_branch_policies,
    select_threshold_variant,
    threshold_variants,
    validate_condition_searches,
)
from scripts.run_global_validation_fusion import (
    AccessAudit,
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
    _verify_sha256_sidecar,
    _write_json,
    _write_sha256_sidecar,
    core_metrics,
    dataset_stratified_paired_setting_bootstrap,
    expected_settings,
)


EXPERIMENT_NAME = "vlmf_strengthening_suite_v1"
PROTOCOL_VERSION = "vlmf-strengthening-suite-v1"
FORMAT_VERSION = 1
GLOBAL_EXPERIMENT_NAME = "global_validation_fusion_v1"
VALIDATION_REPRODUCTION_TOLERANCE = 1e-12
FINAL_OUTPUT_NAMES = (
    "README.md",
    "analysis_lock.json",
    "analysis_lock.sha256",
    "audit.json",
    "branch_selection_summary.json",
    "branch_tidy_metrics.csv",
    "complementarity_metrics.json",
    "cross_dataset_summary.json",
    "input_manifest.json",
    "per_setting_metrics.json",
    "policies.lock.json",
    "policies.lock.sha256",
    "protocol.json",
    "protocol.sha256",
    "route_stability.json",
    "threshold_sensitivity_metrics.csv",
    "threshold_sensitivity_summary.json",
    "validation_diagnostics.lock.json",
    "validation_diagnostics.lock.sha256",
)


def protocol_payload() -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "complete the predeclared simple-alternative, complementarity, route-"
            "stability, and threshold-sensitivity evidence without retraining"
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
            "train_percents": [1, 5, 10],
            "model_seeds": [17, 42, 20260725],
            "setting_count": 27,
            "conditions": list(MAIN_CONDITIONS),
            "base_models_retrained": False,
            "probability_calibration": "uncalibrated",
            "structured_missingness_included": False,
            "learned_models_included": False,
        },
        "lock_boundary": {
            "audited_lock_run_writes_protocol_before_its_validation_reads": True,
            "prelock_feasibility_inspection": {
                "scope": "one NF-BoT validation setting",
                "purpose": (
                    "confirm archive fields, endpoint metric reproduction, and "
                    "canonical-route implementability"
                ),
                "sealed_test_files_read": 0,
                "test_metrics_read": 0,
                "candidate_grid_or_reporting_rule_changed_after_inspection": False,
            },
            "protocol_frozen_before_any_sealed_test_read": True,
            "all_branch_and_threshold_policies_locked_before_test_reads": True,
            "validation_diagnostics_locked_before_test_reads": True,
            "test_labels_used_only_after_analysis_lock": True,
            "VLMF_reproduced_before_new_test_results_are_generated": True,
            "test_results_may_change_protocol_or_policies": False,
        },
        "branch_selection": {
            "candidates": ["xgboost", "schemaflow_pretrained"],
            "global": (
                "one branch per dataset/train-fraction/model-seed setting, "
                "selected by the equal mean over three validation conditions"
            ),
            "conditioned": "one branch per setting and declared condition",
            "objective": "macro_f1",
            "tie_break": [
                "mcc",
                "balanced_accuracy",
                "xgboost_on_exact_tie",
            ],
        },
        "complementarity": {
            "splits": ["validation", "sealed_test_after_lock"],
            "unit": "paired sample",
            "reported_partition": [
                "both_correct",
                "xgboost_only_correct",
                "schemaflow_only_correct",
                "both_wrong_same_prediction",
                "both_wrong_different_prediction",
            ],
            "reported_rates": [
                "prediction_disagreement_rate",
                "oracle_branch_accuracy",
                "error_jaccard",
                "error_overlap_coefficient",
            ],
            "aggregation": "equal setting means; never pooled across datasets",
            "semantic_field_specialization_claim_allowed": False,
        },
        "route_stability": {
            "test_oracle_role": "post_lock_diagnostic_only",
            "candidate_space": {
                "methods": list(POOL_METHODS),
                "alpha_grid": list(ALPHA_GRID),
                "alpha_semantics": "xgboost_probability_weight",
            },
            "candidate_tie_break": list(RANKING),
            "condition_level": [
                "unconstrained_validation_best_vs_test_oracle",
                "deployed_VLMF_route_vs_test_oracle",
                "conditioned_branch_selection_vs_test_oracle_branch",
            ],
            "setting_level": [
                "global_validation_fusion_vs_global_test_oracle",
                "global_branch_selection_vs_global_test_oracle_branch",
            ],
            "reported": [
                "exact_match",
                "pool_match",
                "absolute_alpha_difference",
                "test_rank",
                "test_macro_f1_regret",
            ],
            "oracle_used_to_redefine_or_select_method": False,
        },
        "threshold_sensitivity": {
            "reference": "canonical VLMF admission gate and fixed-alpha fallback",
            "design": "predeclared one-factor-at-a-time plus two simplifications",
            "variants": [dict(row) for row in threshold_variants()],
            "selection_source": (
                "validation selection.json complete linear/log candidate records"
            ),
            "canonical_route_must_match_existing_locked_VLMF": True,
            "all_variants_reported_regardless_of_direction": True,
        },
        "evaluation": {
            "input": "existing SHA-256-locked predictions.npz archives",
            "pool_implementation": (
                "flowfm.training.probability_fusion.pool_probabilities"
            ),
            "VLMF_probability_key": "uncalibrated__a0_mr__{condition}",
            "VLMF_reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "abort_before_new_results_on_reproduction_failure": True,
        },
        "aggregation": {
            "condition_cell": (
                "dataset_x_train_fraction_x_model_seed_x_condition"
            ),
            "setting_mean": "equal mean over three conditions",
            "dataset_mean": "equal mean over nine setting means",
            "cross_dataset_mean": "equal mean over three datasets",
            "direction_zero_tolerance": DIRECTION_TOLERANCE,
            "bootstrap": {
                "method": "dataset_stratified_paired_setting_bootstrap",
                "within_dataset_units": 9,
                "dataset_weighting": "equal",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile_95pct",
                "p_value_inferred": False,
            },
        },
        "explicit_exclusions": [
            "base_model_retraining",
            "logistic_stacking",
            "learned_gate",
            "dynamic_ensemble_training",
            "structured_missingness",
            "calibration",
            "paper_modification",
        ],
    }


def _policy(method: object, alpha: object) -> dict[str, object]:
    key = candidate_key(method, alpha)
    return {"method": key[0], "alpha": key[1]}


def _same_policy(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return _policy(left["method"], left["alpha"]) == _policy(
        right["method"], right["alpha"]
    )


def _global_lock_inputs(project_root: Path, audit: AccessAudit):
    directory = project_root / "results" / GLOBAL_EXPERIMENT_NAME
    protocol_path = directory / "protocol.json"
    policies_path = directory / "global_validation_policies.lock.json"
    for path, role in (
        (protocol_path, "prior_global_protocol"),
        (protocol_path.with_suffix(".sha256"), "prior_global_protocol_sidecar"),
        (policies_path, "prior_global_validation_policy_lock"),
        (policies_path.with_suffix(".sha256"), "prior_global_policy_sidecar"),
    ):
        audit.validation_read(path, role)
    global_protocol_sha = _verify_sha256_sidecar(protocol_path)
    global_policies_sha = _verify_sha256_sidecar(policies_path)
    payload = _read_json(policies_path)
    if (
        payload.get("experiment_name") != GLOBAL_EXPERIMENT_NAME
        or payload.get("setting_count") != 27
        or len(payload.get("policies", ())) != 27
        or payload.get("test_files_read") is not False
    ):
        raise RuntimeError("prior global validation policy lock is invalid")
    entries = [
        _artifact_entry(protocol_path, role="prior_global_protocol"),
        _artifact_entry(
            protocol_path.with_suffix(".sha256"),
            role="prior_global_protocol_sidecar",
        ),
        _artifact_entry(policies_path, role="prior_global_validation_policy_lock"),
        _artifact_entry(
            policies_path.with_suffix(".sha256"),
            role="prior_global_policy_sidecar",
        ),
    ]
    return payload, global_protocol_sha, global_policies_sha, entries


def _validation_prediction_artifact(registry_row: Mapping[str, object]):
    try:
        return registry_row["locked_steps"]["validation_selection"]["artifacts"][
            "validation_predictions"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("validation prediction lock is missing") from error


def _validation_diagnostics(
    path: Path,
    searches: Mapping[str, object],
    class_count: int,
) -> dict[str, object]:
    validated = validate_condition_searches(searches)
    conditions = {}
    with np.load(path, allow_pickle=False) as archive:
        if "group_ids" not in archive.files:
            raise RuntimeError(f"validation group_ids missing: {path}")
        sample_count = int(np.asarray(archive["group_ids"]).shape[0])
        reference_targets = None
        for condition in MAIN_CONDITIONS:
            keys = {
                "targets": f"{condition}__targets",
                "xgboost": f"{condition}__xgboost",
                "schemaflow": f"{condition}__schemaflow_pretrained",
            }
            if any(key not in archive.files for key in keys.values()):
                raise RuntimeError(f"validation fields missing for {condition}: {path}")
            targets = np.asarray(archive[keys["targets"]], dtype=np.int64)
            tree = np.asarray(archive[keys["xgboost"]], dtype=np.float64)
            neural = np.asarray(archive[keys["schemaflow"]], dtype=np.float64)
            if targets.shape != (sample_count,):
                raise RuntimeError("validation targets do not align with group_ids")
            if reference_targets is None:
                reference_targets = targets.copy()
            elif not np.array_equal(reference_targets, targets):
                raise RuntimeError("validation targets changed across conditions")
            tree_metrics = core_metrics(tree, targets, class_count)
            neural_metrics = core_metrics(neural, targets, class_count)
            endpoint_metrics = {
                "xgboost": validated[condition][("linear", 1.0)],
                "schemaflow_pretrained": validated[condition][("linear", 0.0)],
            }
            for name, observed in (
                ("xgboost", tree_metrics),
                ("schemaflow_pretrained", neural_metrics),
            ):
                for metric in ("macro_f1", "mcc", "balanced_accuracy"):
                    error = abs(observed[metric] - endpoint_metrics[name][metric])
                    if error > VALIDATION_REPRODUCTION_TOLERANCE:
                        raise RuntimeError(
                            f"validation endpoint reproduction failed: {condition}/{name}/{metric}"
                        )
            conditions[condition] = {
                "branch_metrics": {
                    "xgboost": tree_metrics,
                    "schemaflow_pretrained": neural_metrics,
                },
                "complementarity": complementarity_payload(tree, neural, targets),
            }
    return {
        "sample_count": sample_count,
        "group_ids_present": True,
        "targets_identical_across_conditions": True,
        "endpoint_candidate_metrics_reproduced": True,
        "conditions": conditions,
    }


def _lock_audit(
    audit: AccessAudit,
    *,
    protocol_sha256: str,
    analysis_lock_sha256: str,
    git: Mapping[str, object],
) -> dict[str, object]:
    lock_sequence = next(
        row["sequence"] for row in audit.events if row["event"] == "policies_locked"
    )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "phase": "locked",
        "protocol_sha256": protocol_sha256,
        "analysis_lock_sha256": analysis_lock_sha256,
        "events": audit.events,
        "assertions": {
            "protocol_is_first_event": audit.events[0]["event"] == "protocol_locked",
            "protocol_locked_before_validation_reads": all(
                row["sequence"] > audit.events[0]["sequence"]
                for row in audit.events
                if row["event"] == "validation_input_read"
            ),
            "analysis_lock_sequence": lock_sequence,
            "sealed_test_files_read_before_analysis_lock": 0,
            "sealed_test_access_enabled": False,
            "setting_count": 27,
            "branch_policies_locked": True,
            "threshold_variant_policies_locked": True,
            "validation_diagnostics_locked": True,
        },
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
        "lock incomplete\n", encoding="ascii"
    )

    audit = AccessAudit()
    protocol_path = _write_json(output_dir / "protocol.json", protocol_payload())
    _, protocol_sha256 = _write_sha256_sidecar(protocol_path)
    audit.lock_protocol(protocol_path, protocol_sha256)
    (
        global_payload,
        global_protocol_sha,
        global_policies_sha,
        manifest_entries,
    ) = _global_lock_inputs(project_root, audit)
    global_policies = {
        (row["dataset"], row["setting_id"]): row
        for row in global_payload["policies"]
    }

    policy_rows = []
    validation_rows = []
    source_protocol_hashes = set()
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
            selection_path = matrix_root / "selections" / setting_id / "selection.json"
            validation_path = (
                matrix_root
                / "selections"
                / setting_id
                / "validation_predictions.npz"
            )
            if Path(registry_row["selection_path"]).resolve() != selection_path:
                raise RuntimeError(f"selection path mismatch for {dataset}/{setting_id}")
            audit.validation_read(selection_path, "validation_selection")
            selection_sha = _sha256_file(selection_path)
            if selection_sha != registry_row["selection_sha256"]:
                raise RuntimeError(f"selection hash mismatch for {dataset}/{setting_id}")
            selection = _read_json(selection_path)
            if (
                selection.get("selection_scope") != "validation_only"
                or selection.get("test_samples_materialized") is not False
                or selection.get("test_metrics_used") is not False
            ):
                raise RuntimeError(f"selection boundary invalid for {dataset}/{setting_id}")
            source_protocol_hashes.add(selection.get("protocol_hash"))
            searches = selection["random_condition_selection"]["linear_or_log"]
            validate_condition_searches(searches)
            global_row = global_policies[(dataset, setting_id)]
            global_policy = _policy(
                global_row["policy"]["method"], global_row["policy"]["alpha"]
            )
            branch = select_branch_policies(searches)
            fixed_metrics = {
                name: selection["fixed_a0"]["validation_metrics"][name]
                for name in MAIN_CONDITIONS
            }
            variants = {
                variant["name"]: select_threshold_variant(
                    searches, fixed_metrics, global_policy, variant
                )
                for variant in threshold_variants()
            }
            deployed = {
                name: _policy(
                    selection["locked_policy"]["policies"][name]["method"],
                    selection["locked_policy"]["policies"][name]["alpha"],
                )
                for name in MAIN_CONDITIONS
            }
            canonical = variants["canonical"]["locked_policies"]
            if any(
                not _same_policy(canonical[name], deployed[name])
                for name in MAIN_CONDITIONS
            ):
                raise RuntimeError(
                    f"canonical threshold route mismatch for {dataset}/{setting_id}"
                )

            validation_artifact = _validation_prediction_artifact(registry_row)
            if Path(validation_artifact["path"]).resolve() != validation_path:
                raise RuntimeError(
                    f"validation prediction path mismatch for {dataset}/{setting_id}"
                )
            audit.validation_read(
                validation_path, "locked_validation_prediction_archive"
            )
            validation_sha = _sha256_file(validation_path)
            if (
                validation_sha != validation_artifact["sha256"]
                or validation_path.stat().st_size != validation_artifact["bytes"]
            ):
                raise RuntimeError(
                    f"validation prediction hash mismatch for {dataset}/{setting_id}"
                )
            diagnostics = _validation_diagnostics(
                validation_path,
                searches,
                len(
                    selection["fixed_a0"]["validation_metrics"]["clean"][
                        "label_names"
                    ]
                ),
            )
            validation_best = {
                name: _policy(
                    searches[name]["selected"]["method"],
                    searches[name]["selected"]["alpha"],
                )
                for name in MAIN_CONDITIONS
            }
            policy_rows.append(
                {
                    "dataset": dataset,
                    "dataset_id": spec["dataset_id"],
                    **setting,
                    "selection_input": {
                        "path": str(selection_path),
                        "bytes": selection_path.stat().st_size,
                        "sha256": selection_sha,
                    },
                    "global_validation_fusion": global_policy,
                    "branch_selection": branch,
                    "VLMF_deployed_policies": deployed,
                    "unconstrained_validation_best": validation_best,
                    "threshold_variants": variants,
                }
            )
            validation_rows.append(
                {
                    "dataset": dataset,
                    "dataset_id": spec["dataset_id"],
                    **setting,
                    "validation_prediction_input": {
                        "path": str(validation_path),
                        "bytes": validation_path.stat().st_size,
                        "sha256": validation_sha,
                    },
                    **diagnostics,
                }
            )
            manifest_entries.extend(
                (
                    _artifact_entry(
                        selection_path,
                        role="validation_selection",
                        dataset=dataset,
                        setting_id=setting_id,
                        observed_sha256=selection_sha,
                        expected_sha256=registry_row["selection_sha256"],
                    ),
                    _artifact_entry(
                        validation_path,
                        role="locked_validation_prediction_archive",
                        dataset=dataset,
                        setting_id=setting_id,
                        observed_sha256=validation_sha,
                        expected_sha256=validation_artifact["sha256"],
                    ),
                )
            )

    if len(policy_rows) != 27 or len(validation_rows) != 27:
        raise RuntimeError("exactly 27 setting locks are required")
    if len(source_protocol_hashes) != 1 or None in source_protocol_hashes:
        raise RuntimeError("source VLMF protocol hashes are inconsistent")
    git = _git_state(repo_root)
    policies_path = _write_json(
        output_dir / "policies.lock.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "protocol_sha256": protocol_sha256,
            "source_VLMF_protocol_sha256": next(iter(source_protocol_hashes)),
            "prior_global_protocol_sha256": global_protocol_sha,
            "prior_global_policies_sha256": global_policies_sha,
            "selection_scope": "validation_only",
            "test_files_read": False,
            "setting_count": 27,
            "threshold_variant_count": len(threshold_variants()),
            "git": git,
            "settings": policy_rows,
        },
    )
    _, policies_sha = _write_sha256_sidecar(policies_path)
    diagnostics_path = _write_json(
        output_dir / "validation_diagnostics.lock.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "protocol_sha256": protocol_sha256,
            "selection_scope": "validation_only",
            "test_files_read": False,
            "setting_count": 27,
            "settings": validation_rows,
        },
    )
    _, diagnostics_sha = _write_sha256_sidecar(diagnostics_path)
    analysis_lock_path = _write_json(
        output_dir / "analysis_lock.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "protocol_sha256": protocol_sha256,
            "policies_sha256": policies_sha,
            "validation_diagnostics_sha256": diagnostics_sha,
            "prior_global_policies_sha256": global_policies_sha,
            "setting_count": 27,
            "test_files_read": False,
            "locked_before_test": True,
        },
    )
    _, analysis_lock_sha = _write_sha256_sidecar(analysis_lock_path)
    audit.lock_policies(analysis_lock_path, analysis_lock_sha)
    _write_json(
        output_dir / "input_manifest.validation.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "phase": "validation_lock",
            "entry_count": len(manifest_entries),
            "entries": sorted(manifest_entries, key=lambda row: row["path"]),
        },
    )
    _write_json(
        output_dir / "audit.json",
        _lock_audit(
            audit,
            protocol_sha256=protocol_sha256,
            analysis_lock_sha256=analysis_lock_sha,
            git=git,
        ),
    )
    (output_dir / ".lock.incomplete").unlink()
    return {
        "status": "vlmf_strengthening_suite_locked",
        "output_dir": str(output_dir),
        "protocol_sha256": protocol_sha256,
        "policies_sha256": policies_sha,
        "validation_diagnostics_sha256": diagnostics_sha,
        "analysis_lock_sha256": analysis_lock_sha,
        "setting_count": 27,
        "test_files_read": False,
        "git": git,
    }


def _candidate_test_grid(tree, neural, targets, class_count: int):
    rows = []
    indexed = {}
    for method in POOL_METHODS:
        for alpha in ALPHA_GRID:
            metrics = core_metrics(
                pool_probabilities(tree, neural, alpha=alpha, method=method),
                targets,
                class_count,
            )
            row = {"method": method, "alpha": alpha, **metrics}
            rows.append(row)
            indexed[(method, alpha)] = row
    ranked = sorted(rows, key=candidate_ranking_key, reverse=True)
    return ranked, indexed


def _ranked_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [{"rank": index, **dict(row)} for index, row in enumerate(rows, start=1)]


def _global_route_stability(
    validation_policy: Mapping[str, object],
    test_ranking: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validation_key = candidate_key(
        validation_policy["method"], validation_policy["alpha"]
    )
    indexed = {
        candidate_key(row["method"], row["alpha"]): (index, row)
        for index, row in enumerate(test_ranking, start=1)
    }
    if validation_key not in indexed:
        raise RuntimeError("global validation policy is absent from test grid")
    oracle = test_ranking[0]
    oracle_key = candidate_key(oracle["method"], oracle["alpha"])
    rank, selected = indexed[validation_key]
    return {
        "test_oracle": dict(oracle),
        "validation_policy": {
            "method": validation_key[0],
            "alpha": validation_key[1],
        },
        "exact_match": validation_key == oracle_key,
        "pool_match": validation_key[0] == oracle_key[0],
        "alpha_absolute_difference": abs(validation_key[1] - oracle_key[1]),
        "validation_policy_test_rank": rank,
        "validation_policy_test_macro_f1_regret": float(
            oracle["mean_macro_f1"] - selected["mean_macro_f1"]
        ),
        "complete_test_candidate_ranking": _ranked_rows(test_ranking),
    }


def _branch_oracle(
    xgboost_metrics: Mapping[str, object],
    schemaflow_metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [
        {
            "branch": "xgboost",
            "method": "linear",
            "alpha": 1.0,
            **dict(xgboost_metrics),
        },
        {
            "branch": "schemaflow_pretrained",
            "method": "linear",
            "alpha": 0.0,
            **dict(schemaflow_metrics),
        },
    ]
    return sorted(rows, key=candidate_ranking_key, reverse=True)


def _evaluate_suite(
    inputs: Sequence[Mapping[str, object]],
    reproduced: Mapping[tuple[str, str], Mapping[str, object]],
    validation_diagnostics: Mapping[tuple[str, str], Mapping[str, object]],
    audit: AccessAudit,
):
    setting_rows = []
    branch_tidy = []
    threshold_tidy = []
    route_condition_rows = []
    route_global_rows = []
    complementarity_test_rows = []
    maximum_canonical_reproduction_error = 0.0
    for source in inputs:
        dataset = source["dataset"]
        setting_id = source["setting_id"]
        key = (dataset, setting_id)
        policy_row = source["policy_row"]
        path = source["prediction_path"]
        audit.test_read(path, "sealed_branch_probabilities_and_labels")
        conditions = {}
        condition_candidate_maps = {}
        condition_branch_metrics = {}
        with np.load(path, allow_pickle=False) as archive:
            targets = np.asarray(archive["targets"], dtype=np.int64)
            class_count = len(source["label_names"])
            for condition in MAIN_CONDITIONS:
                tree_key = f"uncalibrated__xgboost__{condition}"
                neural_key = f"uncalibrated__schemaflow_pretrained__{condition}"
                if tree_key not in archive.files or neural_key not in archive.files:
                    raise RuntimeError(
                        f"sealed branch fields missing for {dataset}/{setting_id}/{condition}"
                    )
                tree = np.asarray(archive[tree_key], dtype=np.float64)
                neural = np.asarray(archive[neural_key], dtype=np.float64)
                tree_metrics = core_metrics(tree, targets, class_count)
                neural_metrics = core_metrics(neural, targets, class_count)
                branch_ranking = _branch_oracle(tree_metrics, neural_metrics)
                branch_oracle = branch_ranking[0]

                global_branch_policy = policy_row["branch_selection"]["global"]
                conditioned_branch_policy = policy_row["branch_selection"][
                    "conditioned"
                ][condition]
                global_branch_metrics = (
                    tree_metrics
                    if float(global_branch_policy["alpha"]) == 1.0
                    else neural_metrics
                )
                conditioned_branch_metrics = (
                    tree_metrics
                    if float(conditioned_branch_policy["alpha"]) == 1.0
                    else neural_metrics
                )
                global_policy = policy_row["global_validation_fusion"]
                global_fusion_metrics = core_metrics(
                    pool_probabilities(
                        tree,
                        neural,
                        alpha=float(global_policy["alpha"]),
                        method=str(global_policy["method"]),
                    ),
                    targets,
                    class_count,
                )

                variant_metrics = {}
                for variant in threshold_variants():
                    name = variant["name"]
                    locked_policy = policy_row["threshold_variants"][name][
                        "locked_policies"
                    ][condition]
                    variant_metrics[name] = core_metrics(
                        pool_probabilities(
                            tree,
                            neural,
                            alpha=float(locked_policy["alpha"]),
                            method=str(locked_policy["method"]),
                        ),
                        targets,
                        class_count,
                    )
                canonical_metrics = variant_metrics["canonical"]
                reproduced_metrics = reproduced[key][condition]["metrics"]
                canonical_error = abs(
                    canonical_metrics["macro_f1"]
                    - reproduced_metrics["macro_f1"]
                )
                maximum_canonical_reproduction_error = max(
                    maximum_canonical_reproduction_error, canonical_error
                )
                if canonical_error > REPRODUCTION_TOLERANCE:
                    raise RuntimeError(
                        "canonical route failed to reproduce VLMF for "
                        f"{dataset}/{setting_id}/{condition}: {canonical_error}"
                    )

                test_ranking, test_candidates = _candidate_test_grid(
                    tree, neural, targets, class_count
                )
                condition_candidate_maps[condition] = test_candidates
                route = route_stability_record(
                    policy_row["unconstrained_validation_best"][condition],
                    test_ranking,
                    deployed_policy=policy_row["VLMF_deployed_policies"][condition],
                )
                route["complete_test_candidate_ranking"] = _ranked_rows(test_ranking)
                route["conditioned_branch_selection"] = {
                    "validation_selected_branch": conditioned_branch_policy["branch"],
                    "test_oracle_branch": branch_oracle["branch"],
                    "exact_match": conditioned_branch_policy["branch"]
                    == branch_oracle["branch"],
                    "test_macro_f1_regret": float(
                        branch_oracle["macro_f1"]
                        - conditioned_branch_metrics["macro_f1"]
                    ),
                }
                route_condition_rows.append(
                    {
                        "dataset": dataset,
                        "setting_id": setting_id,
                        "train_percent": source["train_percent"],
                        "model_seed": source["model_seed"],
                        "condition": condition,
                        **route,
                    }
                )

                complementarity = complementarity_payload(tree, neural, targets)
                complementarity_test_rows.append(
                    {
                        "dataset": dataset,
                        "setting_id": setting_id,
                        "train_percent": source["train_percent"],
                        "model_seed": source["model_seed"],
                        "condition": condition,
                        "split": "sealed_test",
                        **complementarity,
                    }
                )
                condition_branch_metrics[condition] = {
                    "xgboost": tree_metrics,
                    "schemaflow_pretrained": neural_metrics,
                    "global_branch_selection": global_branch_metrics,
                    "conditioned_branch_selection": conditioned_branch_metrics,
                }
                conditions[condition] = {
                    "xgboost": tree_metrics,
                    "schemaflow_pretrained": neural_metrics,
                    "global_branch_selection": {
                        "policy": {
                            "branch": global_branch_policy["branch"],
                            "method": "linear",
                            "alpha": global_branch_policy["alpha"],
                        },
                        "metrics": global_branch_metrics,
                    },
                    "conditioned_branch_selection": {
                        "policy": {
                            "branch": conditioned_branch_policy["branch"],
                            "method": "linear",
                            "alpha": conditioned_branch_policy["alpha"],
                        },
                        "metrics": conditioned_branch_metrics,
                    },
                    "global_validation_fusion": {
                        "policy": global_policy,
                        "metrics": global_fusion_metrics,
                    },
                    "VLMF": {
                        "policy": policy_row["VLMF_deployed_policies"][condition],
                        "metrics": canonical_metrics,
                        "archive_reproduction_error": canonical_error,
                    },
                    "threshold_variants": variant_metrics,
                    "complementarity": complementarity,
                    "route_stability": route,
                }
                branch_tidy.append(
                    {
                        "experiment_name": EXPERIMENT_NAME,
                        "dataset": dataset,
                        "dataset_id": source["dataset_id"],
                        "setting_id": setting_id,
                        "train_percent": source["train_percent"],
                        "model_seed": source["model_seed"],
                        "condition": condition,
                        "xgboost_macro_f1": tree_metrics["macro_f1"],
                        "schemaflow_pretrained_macro_f1": neural_metrics["macro_f1"],
                        "global_branch_selection": global_branch_policy["branch"],
                        "global_branch_macro_f1": global_branch_metrics["macro_f1"],
                        "conditioned_branch_selection": conditioned_branch_policy[
                            "branch"
                        ],
                        "conditioned_branch_macro_f1": conditioned_branch_metrics[
                            "macro_f1"
                        ],
                        "global_validation_fusion_macro_f1": global_fusion_metrics[
                            "macro_f1"
                        ],
                        "VLMF_macro_f1": canonical_metrics["macro_f1"],
                        "VLMF_minus_global_branch_macro_f1": canonical_metrics[
                            "macro_f1"
                        ]
                        - global_branch_metrics["macro_f1"],
                        "VLMF_minus_conditioned_branch_macro_f1": canonical_metrics[
                            "macro_f1"
                        ]
                        - conditioned_branch_metrics["macro_f1"],
                        "conditioned_minus_global_branch_macro_f1": conditioned_branch_metrics[
                            "macro_f1"
                        ]
                        - global_branch_metrics["macro_f1"],
                    }
                )
                for variant in threshold_variants():
                    name = variant["name"]
                    threshold_tidy.append(
                        {
                            "experiment_name": EXPERIMENT_NAME,
                            "dataset": dataset,
                            "dataset_id": source["dataset_id"],
                            "setting_id": setting_id,
                            "train_percent": source["train_percent"],
                            "model_seed": source["model_seed"],
                            "condition": condition,
                            "variant": name,
                            "gate_passed": policy_row["threshold_variants"][name][
                                "gate_passed"
                            ],
                            "pool_method": policy_row["threshold_variants"][name][
                                "locked_policies"
                            ][condition]["method"],
                            "xgboost_alpha": policy_row["threshold_variants"][name][
                                "locked_policies"
                            ][condition]["alpha"],
                            "macro_f1": variant_metrics[name]["macro_f1"],
                            "canonical_VLMF_macro_f1": canonical_metrics["macro_f1"],
                            "canonical_minus_variant_macro_f1": canonical_metrics[
                                "macro_f1"
                            ]
                            - variant_metrics[name]["macro_f1"],
                        }
                    )

        global_test_ranking = rank_global_candidates(condition_candidate_maps)
        global_stability = _global_route_stability(
            policy_row["global_validation_fusion"], global_test_ranking
        )
        global_branch_rows = []
        for branch, alpha in (("schemaflow_pretrained", 0.0), ("xgboost", 1.0)):
            metrics = [condition_branch_metrics[name][branch] for name in MAIN_CONDITIONS]
            global_branch_rows.append(
                {
                    "branch": branch,
                    "alpha": alpha,
                    "mean_macro_f1": float(
                        np.mean([row["macro_f1"] for row in metrics])
                    ),
                    "mean_mcc": float(np.mean([row["mcc"] for row in metrics])),
                    "mean_balanced_accuracy": float(
                        np.mean([row["balanced_accuracy"] for row in metrics])
                    ),
                }
            )
        global_branch_rows.sort(
            key=lambda row: (
                row["mean_macro_f1"],
                row["mean_mcc"],
                row["mean_balanced_accuracy"],
                row["alpha"],
            ),
            reverse=True,
        )
        global_branch_selected = policy_row["branch_selection"]["global"]["branch"]
        selected_global_branch_row = next(
            row for row in global_branch_rows if row["branch"] == global_branch_selected
        )
        global_branch_stability = {
            "validation_selected_branch": global_branch_selected,
            "test_oracle_branch": global_branch_rows[0]["branch"],
            "exact_match": global_branch_selected == global_branch_rows[0]["branch"],
            "test_macro_f1_regret": float(
                global_branch_rows[0]["mean_macro_f1"]
                - selected_global_branch_row["mean_macro_f1"]
            ),
            "test_ranking": _ranked_rows(global_branch_rows),
        }
        route_global_rows.append(
            {
                "dataset": dataset,
                "setting_id": setting_id,
                "train_percent": source["train_percent"],
                "model_seed": source["model_seed"],
                "global_validation_fusion": global_stability,
                "global_branch_selection": global_branch_stability,
            }
        )
        setting_rows.append(
            {
                "dataset": dataset,
                "dataset_id": source["dataset_id"],
                "setting_id": setting_id,
                "train_percent": source["train_percent"],
                "model_seed": source["model_seed"],
                "locked_branch_selection": policy_row["branch_selection"],
                "locked_threshold_variants": policy_row["threshold_variants"],
                "conditions": conditions,
                "global_route_stability": global_stability,
                "global_branch_stability": global_branch_stability,
                "validation_diagnostics_sha256": validation_diagnostics[key][
                    "validation_prediction_input"
                ]["sha256"],
            }
        )
    audit.record(
        "strengthening_suite_test_evaluation_completed",
        setting_count=len(setting_rows),
        condition_cell_count=len(branch_tidy),
        threshold_variant_count=len(threshold_variants()),
        canonical_VLMF_maximum_reproduction_error=(
            maximum_canonical_reproduction_error
        ),
    )
    return {
        "settings": setting_rows,
        "branch_tidy": branch_tidy,
        "threshold_tidy": threshold_tidy,
        "route_condition_rows": route_condition_rows,
        "route_global_rows": route_global_rows,
        "complementarity_test_rows": complementarity_test_rows,
        "maximum_canonical_reproduction_error": (
            maximum_canonical_reproduction_error
        ),
    }


def _direction_counts(values: Sequence[float]) -> dict[str, object]:
    return {
        "positive": sum(value > DIRECTION_TOLERANCE for value in values),
        "zero": sum(abs(value) <= DIRECTION_TOLERANCE for value in values),
        "negative": sum(value < -DIRECTION_TOLERANCE for value in values),
        "zero_tolerance": DIRECTION_TOLERANCE,
    }


def _setting_means(
    rows: Sequence[Mapping[str, object]], value_name: str
) -> list[dict[str, object]]:
    grouped = {}
    for row in rows:
        key = (
            row["dataset"],
            row["setting_id"],
            row["train_percent"],
            row["model_seed"],
        )
        grouped.setdefault(key, []).append(float(row[value_name]))
    result = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(MAIN_CONDITIONS):
            raise RuntimeError(f"setting aggregation is incomplete: {key}")
        result.append(
            {
                "dataset": key[0],
                "setting_id": key[1],
                "train_percent": key[2],
                "model_seed": key[3],
                "three_condition_mean": float(np.mean(values)),
            }
        )
    return result


def _paired_comparison_summary(
    rows: Sequence[Mapping[str, object]],
    value_name: str,
) -> dict[str, object]:
    settings = _setting_means(rows, value_name)
    by_dataset = {
        dataset: [
            row["three_condition_mean"]
            for row in settings
            if row["dataset"] == dataset
        ]
        for dataset in DATASETS
    }
    if any(len(values) != 9 for values in by_dataset.values()):
        raise RuntimeError(f"paired comparison settings are incomplete: {value_name}")
    bootstrap = dataset_stratified_paired_setting_bootstrap(
        by_dataset,
        pairing_unit=f"same_setting_three_condition_mean_{value_name}",
    )
    return {
        "difference_column": value_name,
        "dataset_means": {
            dataset: float(np.mean(values)) for dataset, values in by_dataset.items()
        },
        "three_dataset_equal_weight_mean": bootstrap[
            "observed_equal_dataset_mean"
        ],
        "setting_direction_counts": _direction_counts(
            [row["three_condition_mean"] for row in settings]
        ),
        "dataset_stratified_paired_setting_bootstrap": bootstrap,
        "setting_means": settings,
        "p_value_inferred": False,
    }


def _branch_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    score_columns = (
        "xgboost_macro_f1",
        "schemaflow_pretrained_macro_f1",
        "global_branch_macro_f1",
        "conditioned_branch_macro_f1",
        "global_validation_fusion_macro_f1",
        "VLMF_macro_f1",
    )
    dataset_condition = []
    for dataset in DATASETS:
        for condition in MAIN_CONDITIONS:
            cells = [
                row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            if len(cells) != 9:
                raise RuntimeError(
                    f"branch condition cells incomplete: {dataset}/{condition}"
                )
            dataset_condition.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "setting_count": len(cells),
                    **{
                        name: float(np.mean([float(row[name]) for row in cells]))
                        for name in score_columns
                    },
                }
            )
    comparisons = {
        "VLMF_minus_global_branch": _paired_comparison_summary(
            rows, "VLMF_minus_global_branch_macro_f1"
        ),
        "VLMF_minus_conditioned_branch": _paired_comparison_summary(
            rows, "VLMF_minus_conditioned_branch_macro_f1"
        ),
        "conditioned_minus_global_branch": _paired_comparison_summary(
            rows, "conditioned_minus_global_branch_macro_f1"
        ),
    }
    global_counts = Counter(row["global_branch_selection"] for row in rows)
    conditioned_counts = Counter(
        row["conditioned_branch_selection"] for row in rows
    )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "setting_count": 27,
        "condition_cell_count": len(rows),
        "dataset_condition_macro_f1": dataset_condition,
        "comparisons": comparisons,
        "selection_frequencies": {
            "global_branch_condition_cells": dict(sorted(global_counts.items())),
            "conditioned_branch_condition_cells": dict(
                sorted(conditioned_counts.items())
            ),
        },
        "p_value_inferred": False,
    }


def _threshold_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    variants = []
    for specification in threshold_variants():
        name = specification["name"]
        variant_rows = [row for row in rows if row["variant"] == name]
        if len(variant_rows) != 81:
            raise RuntimeError(f"threshold variant cells incomplete: {name}")
        score_settings = _setting_means(variant_rows, "macro_f1")
        comparison = _paired_comparison_summary(
            variant_rows, "canonical_minus_variant_macro_f1"
        )
        dataset_scores = {
            dataset: float(
                np.mean(
                    [
                        row["three_condition_mean"]
                        for row in score_settings
                        if row["dataset"] == dataset
                    ]
                )
            )
            for dataset in DATASETS
        }
        gate_passed_settings = {
            dataset: sum(
                bool(row["gate_passed"])
                for row in variant_rows
                if row["dataset"] == dataset
                and row["condition"] == MAIN_CONDITIONS[0]
            )
            for dataset in DATASETS
        }
        variants.append(
            {
                "name": name,
                "specification": dict(specification),
                "dataset_macro_f1": dataset_scores,
                "three_dataset_equal_weight_macro_f1": float(
                    np.mean(list(dataset_scores.values()))
                ),
                "canonical_minus_variant": comparison,
                "gate_passed_setting_count_by_dataset": gate_passed_settings,
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "reference_variant": "canonical",
        "variant_count": len(variants),
        "variants": variants,
        "all_variants_reported": True,
        "p_value_inferred": False,
    }


def _mean_nullable(values: Sequence[object]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return float(np.mean(retained)) if retained else None


def _complementarity_summary(
    validation_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = []
    for validation in validation_rows:
        for condition in MAIN_CONDITIONS:
            rows.append(
                {
                    "dataset": validation["dataset"],
                    "setting_id": validation["setting_id"],
                    "train_percent": validation["train_percent"],
                    "model_seed": validation["model_seed"],
                    "condition": condition,
                    "split": "validation",
                    **validation["conditions"][condition]["complementarity"],
                }
            )
    rows.extend(test_rows)
    rate_names = (
        "prediction_disagreement_rate",
        "both_correct_rate",
        "xgboost_only_correct_rate",
        "schemaflow_only_correct_rate",
        "both_wrong_same_prediction_rate",
        "both_wrong_different_prediction_rate",
        "oracle_branch_accuracy",
        "error_jaccard",
        "error_overlap_coefficient",
    )
    aggregates = []
    for split in ("validation", "sealed_test"):
        for dataset in DATASETS:
            for condition in MAIN_CONDITIONS:
                cells = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["dataset"] == dataset
                    and row["condition"] == condition
                ]
                if len(cells) != 9:
                    raise RuntimeError(
                        f"complementarity cells incomplete: {split}/{dataset}/{condition}"
                    )
                aggregates.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "condition": condition,
                        "setting_count": 9,
                        **{
                            name: _mean_nullable(
                                [cell["rates"].get(name) for cell in cells]
                            )
                            for name in rate_names
                        },
                    }
                )
    shifts = []
    for dataset in DATASETS:
        for condition in MAIN_CONDITIONS:
            validation = next(
                row
                for row in aggregates
                if row["split"] == "validation"
                and row["dataset"] == dataset
                and row["condition"] == condition
            )
            test = next(
                row
                for row in aggregates
                if row["split"] == "sealed_test"
                and row["dataset"] == dataset
                and row["condition"] == condition
            )
            shifts.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    **{
                        f"test_minus_validation_{name}": (
                            None
                            if validation[name] is None or test[name] is None
                            else test[name] - validation[name]
                        )
                        for name in rate_names
                    },
                }
            )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "aggregation": "equal mean over nine settings",
        "sample_level_rows": rows,
        "dataset_condition_means": aggregates,
        "test_minus_validation": shifts,
        "semantic_field_specialization_claimed": False,
    }


def _route_stability_summary(
    condition_rows: Sequence[Mapping[str, object]],
    global_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def aggregate_condition(cells):
        return {
            "cell_count": len(cells),
            "validation_exact_match_count": sum(
                row["validation_exact_match"] for row in cells
            ),
            "validation_pool_match_count": sum(
                row["validation_pool_match"] for row in cells
            ),
            "mean_validation_alpha_absolute_difference": float(
                np.mean(
                    [row["validation_alpha_absolute_difference"] for row in cells]
                )
            ),
            "mean_validation_policy_test_rank": float(
                np.mean([row["validation_policy_test_rank"] for row in cells])
            ),
            "mean_validation_policy_test_macro_f1_regret": float(
                np.mean(
                    [
                        row["validation_policy_test_macro_f1_regret"]
                        for row in cells
                    ]
                )
            ),
            "deployed_exact_match_count": sum(
                row["deployed_exact_match"] for row in cells
            ),
            "deployed_pool_match_count": sum(
                row["deployed_pool_match"] for row in cells
            ),
            "mean_deployed_alpha_absolute_difference": float(
                np.mean([row["deployed_alpha_absolute_difference"] for row in cells])
            ),
            "mean_deployed_policy_test_rank": float(
                np.mean([row["deployed_policy_test_rank"] for row in cells])
            ),
            "mean_deployed_policy_test_macro_f1_regret": float(
                np.mean(
                    [row["deployed_policy_test_macro_f1_regret"] for row in cells]
                )
            ),
            "conditioned_branch_exact_match_count": sum(
                row["conditioned_branch_selection"]["exact_match"] for row in cells
            ),
            "mean_conditioned_branch_test_macro_f1_regret": float(
                np.mean(
                    [
                        row["conditioned_branch_selection"][
                            "test_macro_f1_regret"
                        ]
                        for row in cells
                    ]
                )
            ),
        }

    condition_aggregates = []
    for dataset in DATASETS:
        for condition in MAIN_CONDITIONS:
            cells = [
                row
                for row in condition_rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            if len(cells) != 9:
                raise RuntimeError(
                    f"route stability cells incomplete: {dataset}/{condition}"
                )
            condition_aggregates.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    **aggregate_condition(cells),
                }
            )

    global_aggregates = []
    for dataset in DATASETS:
        cells = [row for row in global_rows if row["dataset"] == dataset]
        if len(cells) != 9:
            raise RuntimeError(f"global stability settings incomplete: {dataset}")
        fusion = [row["global_validation_fusion"] for row in cells]
        branches = [row["global_branch_selection"] for row in cells]
        global_aggregates.append(
            {
                "dataset": dataset,
                "setting_count": 9,
                "global_validation_fusion_exact_match_count": sum(
                    row["exact_match"] for row in fusion
                ),
                "global_validation_fusion_pool_match_count": sum(
                    row["pool_match"] for row in fusion
                ),
                "mean_global_validation_alpha_absolute_difference": float(
                    np.mean([row["alpha_absolute_difference"] for row in fusion])
                ),
                "mean_global_validation_policy_test_rank": float(
                    np.mean([row["validation_policy_test_rank"] for row in fusion])
                ),
                "mean_global_validation_test_macro_f1_regret": float(
                    np.mean(
                        [
                            row["validation_policy_test_macro_f1_regret"]
                            for row in fusion
                        ]
                    )
                ),
                "global_branch_exact_match_count": sum(
                    row["exact_match"] for row in branches
                ),
                "mean_global_branch_test_macro_f1_regret": float(
                    np.mean([row["test_macro_f1_regret"] for row in branches])
                ),
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "test_oracle_role": "post_lock_diagnostic_only",
        "condition_cell_count": len(condition_rows),
        "global_setting_count": len(global_rows),
        "condition_level": {
            "per_setting": list(condition_rows),
            "dataset_condition_aggregates": condition_aggregates,
            "overall": aggregate_condition(condition_rows),
        },
        "global_setting_level": {
            "per_setting": list(global_rows),
            "dataset_aggregates": global_aggregates,
        },
        "test_oracle_used_for_selection": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _cross_summary(
    branch: Mapping[str, object],
    threshold: Mapping[str, object],
    complementarity: Mapping[str, object],
    stability: Mapping[str, object],
    maximum_reproduction_error: float,
) -> dict[str, object]:
    sensitivity_differences = [
        row["canonical_minus_variant"]["three_dataset_equal_weight_mean"]
        for row in threshold["variants"]
        if row["name"] != "canonical"
    ]
    test_complementarity = [
        row
        for row in complementarity["dataset_condition_means"]
        if row["split"] == "sealed_test"
    ]
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "setting_count": 27,
        "condition_cell_count": 81,
        "branch_comparisons": {
            name: {
                "three_dataset_equal_weight_mean": row[
                    "three_dataset_equal_weight_mean"
                ],
                "ci95_lower": row[
                    "dataset_stratified_paired_setting_bootstrap"
                ]["ci95_lower"],
                "ci95_upper": row[
                    "dataset_stratified_paired_setting_bootstrap"
                ]["ci95_upper"],
                "setting_direction_counts": row["setting_direction_counts"],
            }
            for name, row in branch["comparisons"].items()
        },
        "threshold_sensitivity": {
            "variant_count": threshold["variant_count"],
            "minimum_canonical_minus_variant_macro_f1": min(
                sensitivity_differences
            ),
            "maximum_canonical_minus_variant_macro_f1": max(
                sensitivity_differences
            ),
        },
        "sealed_test_complementarity_equal_dataset_condition_mean": {
            name: _mean_nullable([row[name] for row in test_complementarity])
            for name in (
                "prediction_disagreement_rate",
                "xgboost_only_correct_rate",
                "schemaflow_only_correct_rate",
                "oracle_branch_accuracy",
                "error_jaccard",
            )
        },
        "route_stability_overall": stability["condition_level"]["overall"],
        "VLMF_reproduction": {
            "passed": True,
            "condition_cell_count": 81,
            "maximum_absolute_macro_f1_error": maximum_reproduction_error,
            "tolerance": REPRODUCTION_TOLERANCE,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "p_value_inferred": False,
        },
        "all_results_reported_regardless_of_direction": True,
    }


def _readme_text(summary: Mapping[str, object]) -> str:
    branch_lines = []
    for name, row in summary["branch_comparisons"].items():
        directions = row["setting_direction_counts"]
        branch_lines.append(
            "- {name}: {mean:+.8f}, 95% CI [{lower:+.8f}, {upper:+.8f}], "
            "setting +/-/0 = {positive}/{negative}/{zero}".format(
                name=name,
                mean=row["three_dataset_equal_weight_mean"],
                lower=row["ci95_lower"],
                upper=row["ci95_upper"],
                positive=directions["positive"],
                negative=directions["negative"],
                zero=directions["zero"],
            )
        )
    branch_text = "\n".join(branch_lines)
    threshold = summary["threshold_sensitivity"]
    return f"""# VLMF Strengthening Suite v1

This immutable directory contains the validation-locked simple branch
comparators, validation/test prediction complementarity, post-lock route
stability diagnostics, and the complete predeclared threshold-sensitivity set.
No base model or learned ensemble was trained.

## Reproduction

From `/root/autodl-tmp/Res-AE-Attn/repository`:

```bash
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.run_vlmf_strengthening_suite lock \\
  --project-root /root/autodl-tmp/Res-AE-Attn \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/{EXPERIMENT_NAME}
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m scripts.run_vlmf_strengthening_suite evaluate \\
  --project-root /root/autodl-tmp/Res-AE-Attn \\
  --output-dir /root/autodl-tmp/Res-AE-Attn/results/{EXPERIMENT_NAME}
env -u OMP_NUM_THREADS /root/autodl-tmp/envs/schemaflow/bin/python \\
  -m unittest tests.test_vlmf_strengthening tests.test_learned_ensemble_protocol -v
```

The `lock` phase reads only validation-side archives and pre-existing
validation-only locks. The `evaluate` phase verifies every lock and input hash,
reproduces all 81 VLMF macro-F1 cells, and only then generates new test results.

## Primary Paired Results

Differences use the order encoded in each name. Conditions are averaged within
each setting, nine settings within each dataset, and datasets receive equal
weight.

{branch_text}

Across the {threshold['variant_count']} frozen threshold variants, canonical
VLMF minus variant ranges from
`{threshold['minimum_canonical_minus_variant_macro_f1']:+.8f}` to
`{threshold['maximum_canonical_minus_variant_macro_f1']:+.8f}` macro-F1.

All bootstrap intervals use 20,000 dataset-stratified paired-setting resamples
with seed 20260730. No p-value is inferred. Test-oracle routes in
`route_stability.json` are diagnostics only and never participate in selection.
`input_manifest.json` and `audit.json` record hashes and access order.
"""


def run_evaluate(
    project_root: Path, output_dir: Path, repo_root: Path
) -> dict[str, object]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    if not output_dir.is_dir() or (output_dir / ".lock.incomplete").exists():
        raise RuntimeError("completed lock output is required")
    forbidden = [
        name
        for name in (
            "branch_tidy_metrics.csv",
            "threshold_sensitivity_metrics.csv",
            "per_setting_metrics.json",
            "branch_selection_summary.json",
            "threshold_sensitivity_summary.json",
            "complementarity_metrics.json",
            "route_stability.json",
            "cross_dataset_summary.json",
            "input_manifest.json",
            "README.md",
            "output_hashes.sha256",
            ".evaluate.incomplete",
        )
        if (output_dir / name).exists()
    ]
    if forbidden:
        raise RuntimeError(f"refusing to overwrite evaluation outputs: {forbidden}")

    protocol_path = output_dir / "protocol.json"
    policies_path = output_dir / "policies.lock.json"
    diagnostics_path = output_dir / "validation_diagnostics.lock.json"
    analysis_lock_path = output_dir / "analysis_lock.json"
    protocol_sha = _verify_sha256_sidecar(protocol_path)
    policies_sha = _verify_sha256_sidecar(policies_path)
    diagnostics_sha = _verify_sha256_sidecar(diagnostics_path)
    analysis_lock_sha = _verify_sha256_sidecar(analysis_lock_path)
    if _read_json(protocol_path) != protocol_payload():
        raise RuntimeError("protocol payload does not match implemented protocol")
    policies_payload = _read_json(policies_path)
    diagnostics_payload = _read_json(diagnostics_path)
    analysis_lock = _read_json(analysis_lock_path)
    if (
        policies_payload.get("protocol_sha256") != protocol_sha
        or policies_payload.get("setting_count") != 27
        or len(policies_payload.get("settings", ())) != 27
        or policies_payload.get("test_files_read") is not False
        or diagnostics_payload.get("protocol_sha256") != protocol_sha
        or diagnostics_payload.get("setting_count") != 27
        or len(diagnostics_payload.get("settings", ())) != 27
        or diagnostics_payload.get("test_files_read") is not False
        or analysis_lock.get("protocol_sha256") != protocol_sha
        or analysis_lock.get("policies_sha256") != policies_sha
        or analysis_lock.get("validation_diagnostics_sha256") != diagnostics_sha
        or analysis_lock.get("test_files_read") is not False
    ):
        raise RuntimeError("analysis lock bundle is invalid")
    lock_audit = _read_json(output_dir / "audit.json")
    if (
        lock_audit.get("phase") != "locked"
        or lock_audit.get("analysis_lock_sha256") != analysis_lock_sha
        or lock_audit.get("assertions", {}).get(
            "sealed_test_files_read_before_analysis_lock"
        )
        != 0
    ):
        raise RuntimeError("lock audit is invalid")
    audit = AccessAudit(lock_audit["events"])
    audit.enable_test_access(protocol_sha, analysis_lock_sha)
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
        project_root, policies_payload["settings"], audit
    )
    reproduced = _reproduce_vlmf(inputs, audit)
    validation_by_setting = {
        (row["dataset"], row["setting_id"]): row
        for row in diagnostics_payload["settings"]
    }
    evaluated = _evaluate_suite(
        inputs, reproduced, validation_by_setting, audit
    )
    branch_summary = _branch_summary(evaluated["branch_tidy"])
    threshold_summary = _threshold_summary(evaluated["threshold_tidy"])
    complementarity = _complementarity_summary(
        diagnostics_payload["settings"], evaluated["complementarity_test_rows"]
    )
    stability = _route_stability_summary(
        evaluated["route_condition_rows"], evaluated["route_global_rows"]
    )
    cross_summary = _cross_summary(
        branch_summary,
        threshold_summary,
        complementarity,
        stability,
        evaluated["maximum_canonical_reproduction_error"],
    )

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
            row["role"] == "sealed_prediction_archive" for row in manifest_entries
        ),
        unchanged=True,
    )
    _write_csv(output_dir / "branch_tidy_metrics.csv", evaluated["branch_tidy"])
    _write_csv(
        output_dir / "threshold_sensitivity_metrics.csv",
        evaluated["threshold_tidy"],
    )
    _write_json(
        output_dir / "per_setting_metrics.json",
        {
            "format_version": FORMAT_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "setting_count": len(evaluated["settings"]),
            "settings": evaluated["settings"],
        },
    )
    _write_json(output_dir / "branch_selection_summary.json", branch_summary)
    _write_json(
        output_dir / "threshold_sensitivity_summary.json", threshold_summary
    )
    _write_json(output_dir / "complementarity_metrics.json", complementarity)
    _write_json(output_dir / "route_stability.json", stability)
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

    lock_sequence = next(
        row["sequence"] for row in audit.events if row["event"] == "policies_locked"
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
        "protocol_sha256": protocol_sha,
        "policies_sha256": policies_sha,
        "validation_diagnostics_sha256": diagnostics_sha,
        "analysis_lock_sha256": analysis_lock_sha,
        "events": audit.events,
        "assertions": {
            **lock_audit["assertions"],
            "sealed_test_access_enabled": True,
            "first_sealed_test_read_sequence": first_test_sequence,
            "all_sealed_test_reads_after_analysis_lock": first_test_sequence
            > lock_sequence,
            "VLMF_macro_f1_reproduced_before_new_results": True,
            "VLMF_reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "setting_count": 27,
            "condition_cell_count": 81,
            "threshold_variant_count": len(threshold_variants()),
            "all_input_hashes_verified": True,
            "all_inputs_unchanged_after_evaluation": True,
            "source_matrix_files_written": False,
            "output_directory_disjoint_from_source_matrices": True,
            "base_models_retrained": False,
            "learned_models_fitted": False,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "p_value_inferred": False,
            "all_results_reported_regardless_of_direction": True,
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
        "status": "vlmf_strengthening_suite_completed",
        "output_dir": str(output_dir),
        "protocol_sha256": protocol_sha,
        "analysis_lock_sha256": analysis_lock_sha,
        "cross_dataset_summary": cross_summary,
        "git": _git_state(repo_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
