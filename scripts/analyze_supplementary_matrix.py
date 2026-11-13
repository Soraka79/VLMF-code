#!/usr/bin/env python3
"""Run the predeclared post-evaluation analyses of the revision supplementary matrix.

Reads only the post-lock archives under results/cross-dataset-supplementary-matrix-v3
and refuses to start unless every cohort has its sealed-test completion receipt.
Policies are always chosen from validation records; test labels only score policies
that were already chosen. Outputs go to results/supplementary-analysis-v1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from flowfm.data.exporter_missingness import (
    COMPOUND_PROFILES,
    EXPORTER_PROFILES,
    INTERMITTENT_PROFILES,
    MODEL_FIELDS,
    build_supplementary_resolver,
)
from flowfm.data.formal_cohort import get_formal_dataset_spec
from flowfm.data.missingness import STRUCTURED_MISSINGNESS_GROUPS
from flowfm.training.a0_mr_protocol import (
    FORMAL_CONDITIONS,
    RANDOM_CONDITIONS,
    STRUCTURED_CONDITIONS,
    select_a0_mr_validation,
)
from flowfm.training.confirmatory_statistics import paired_cluster_bootstrap_macro_f1
from flowfm.training.learned_ensemble import dataset_stratified_paired_setting_bootstrap
from flowfm.training.probability_fusion import pool_probabilities
from flowfm.training.supplementary_analysis import (
    endpoint_outputs,
    fixed_policies,
    macro_f1,
    policy_agreement,
    policy_macro_f1,
    selection_metrics_only,
    stacking_scores,
    stratified_group_halves,
    stratified_group_subsample,
    validation_controls,
    wilson_interval,
)
from flowfm.training.supplementary_protocol import (
    COMPOUND_CONDITIONS,
    INTERMITTENT_CONDITIONS,
    POST_EVALUATION_ANALYSES,
    PROFILE_CONDITIONS,
    TEST_CONDITIONS,
    dispatch_probabilities,
    frozen_supplementary_hash,
)
from flowfm.training.vlmf_strengthening import MAIN_CONDITIONS


_PROJECT = Path("/root/autodl-tmp/Res-AE-Attn")
_REPO = _PROJECT / "repository"
MATRIX = _PROJECT / "results" / "cross-dataset-supplementary-matrix-v3"
OUTPUT = _PROJECT / "results" / "supplementary-analysis-v1"
COHORTS = tuple(f"{base}_rev{i}" for base in ("nf_bot", "nf_cse", "nf_ton") for i in (1, 2, 3))
SETTINGS = ("pct-01-seed-17", "pct-05-seed-17", "pct-10-seed-17")
DATASETS = {"nf_bot": "NF-BoT", "nf_cse": "NF-CSE", "nf_ton": "NF-ToN"}
NEW_CONDITIONS = PROFILE_CONDITIONS + INTERMITTENT_CONDITIONS + COMPOUND_CONDITIONS
_PLAN = POST_EVALUATION_ANALYSES
VAL_SIZE_SEED = _PLAN["validation_size_curve"]["seed"]
VAL_SIZE_REPEATS = _PLAN["validation_size_curve"]["repeats"]
VAL_SIZES = tuple(v for v in _PLAN["validation_size_curve"]["validation_group_subsample_rows"] if v != "all")
SPLIT_SEED = _PLAN["split_half_validation"]["seed"]
SPLIT_HALF_REPEATS = _PLAN["split_half_validation"]["repeats"]
SPLIT_CHUNK = 10
CLUSTER_SEED = _PLAN["test_group_cluster_bootstrap"]["seed"]
CLUSTER_RESAMPLES = _PLAN["test_group_cluster_bootstrap"]["resamples"]
# Setting-level summaries reuse the paper's frozen bootstrap contract.
PAPER_BOOTSTRAP_SEED = 20260730
PAPER_BOOTSTRAP_RESAMPLES = 20_000
RARE_RECALL_SCREEN = -0.02


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cohort_settings():
    return tuple((c, s) for c in COHORTS for s in SETTINGS)


def _dataset(cohort):
    return cohort.rsplit("_rev", 1)[0]


def _selection_dir(cohort, setting):
    return MATRIX / cohort / "selections" / setting


def _sealed_dir(cohort, setting):
    return MATRIX / cohort / "sealed_test" / setting


def _require_receipts():
    receipts = {}
    for cohort in COHORTS:
        receipt = _read_json(MATRIX / cohort / "sealed_test_access.completed.json")
        if receipt.get("status") != "single_test_open_completed" or receipt.get("sealed_test_open_count") != 1:
            raise RuntimeError(f"{cohort} has no valid sealed-test completion receipt")
        receipts[cohort] = receipt
    return receipts


def _git_state():
    run = lambda *a: subprocess.run(["git", *a], cwd=_REPO, check=True, capture_output=True, text=True).stdout.strip()
    if run("status", "--porcelain"):
        raise RuntimeError("analysis requires a clean repository checkout")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current")}


def _rng(base, *parts):
    return np.random.default_rng(np.random.SeedSequence([base, *parts]))


def _labels(cohort, setting):
    return tuple(_read_json(_selection_dir(cohort, setting) / "label_space.json")["coarse_labels"])


def _load(path, keys):
    with np.load(path) as archive:
        return {key: archive[key] for key in keys}


def _cached(section, name, fn):
    path = OUTPUT / "tasks" / section / f"{name}.json"
    if path.exists():
        return _read_json(path)
    result = fn()
    _write_json(path, result)
    return result


def _primary(cohort, setting):
    metrics = _read_json(_sealed_dir(cohort, setting) / "metrics.json")
    selection = _read_json(_selection_dir(cohort, setting) / "selection.json")
    registered = _read_json(_selection_dir(cohort, setting) / "registered_selection.json")
    keep = ("macro_f1", "mcc", "balanced_accuracy", "nll", "brier", "ece", "per_class_recall", "per_class_support")
    return {
        "cohort": cohort,
        "setting": setting,
        "metrics": {
            condition: {model: {k: values[k] for k in keep} for model, values in models.items()}
            for condition, models in metrics["metrics"].items()
        },
        "routing": metrics["routing"],
        "resolver_label_agreement": metrics["resolver_label_agreement_original_conditions"],
        "gate_passed": selection["routing_gate"]["passed"],
        "candidate_name": selection["routing_gate"]["candidate_name"],
        "registered_gate_passed": registered["random_gate_passed"],
        "test_inference_seconds": metrics["inference_seconds"],
    }


def _controls(cohort, setting):
    selection = _read_json(_selection_dir(cohort, setting) / "selection.json")
    labels = _labels(cohort, setting)
    sealed = _sealed_dir(cohort, setting) / "predictions.npz"
    keys = ["targets"] + [f"{c}__{m}" for c in FORMAL_CONDITIONS for m in ("xgboost", "schemaflow_pretrained", "a0_mr")]
    test = _load(sealed, keys)
    controls = validation_controls(selection)
    outputs = endpoint_outputs(test, MAIN_CONDITIONS)
    scores = {name: policy_macro_f1(outputs, policies, len(labels)) for name, policies in controls.items()}
    rebuilt = endpoint_outputs(test, FORMAL_CONDITIONS)
    locked = selection["locked_policy"]["policies"]
    pool_error = max(
        float(np.max(np.abs(
            pool_probabilities(rebuilt[c]["xgboost"], rebuilt[c]["schemaflow_pretrained"],
                               alpha=locked[c]["alpha"], method=locked[c]["method"]) - test[f"{c}__a0_mr"]
        )))
        for c in FORMAL_CONDITIONS
    )
    stored = _read_json(_sealed_dir(cohort, setting) / "metrics.json")["metrics"]
    f1_error = 0.0
    with np.load(sealed) as archive:
        targets = archive["targets"]
        for condition, models in stored.items():
            for model, values in models.items():
                computed = macro_f1(archive[f"{condition}__{model}"], targets, len(labels))
                f1_error = max(f1_error, abs(computed - values["macro_f1"]))
    return {
        "cohort": cohort,
        "setting": setting,
        "policies": controls,
        "test_macro_f1": scores,
        "reproduction": {"a0_mr_pool_max_abs_error": pool_error, "stored_macro_f1_max_abs_error": f1_error},
    }


def _stack(cohort, setting):
    labels = _labels(cohort, setting)
    keys_val = ["targets", "group_ids"] + [f"{c}__{m}" for c in MAIN_CONDITIONS for m in ("xgboost", "schemaflow_pretrained")]
    keys_test = ["targets"] + [f"{c}__{m}" for c in MAIN_CONDITIONS for m in ("xgboost", "schemaflow_pretrained")]
    validation = _load(_selection_dir(cohort, setting) / "validation_predictions.npz", keys_val)
    test = _load(_sealed_dir(cohort, setting) / "predictions.npz", keys_test)
    try:
        result = stacking_scores(validation, test, len(labels))
        return {"cohort": cohort, "setting": setting, "status": "completed", **result}
    except RuntimeError as error:
        return {"cohort": cohort, "setting": setting, "status": "failed", "error": str(error)}


def _validation_keys():
    return ["targets", "group_ids"] + [f"{c}__{m}" for c in FORMAL_CONDITIONS for m in ("xgboost", "schemaflow_pretrained")]


def _main_test(cohort, setting):
    keys = ["targets"] + [f"{c}__{m}" for c in MAIN_CONDITIONS for m in ("xgboost", "schemaflow_pretrained")]
    return endpoint_outputs(_load(_sealed_dir(cohort, setting) / "predictions.npz", keys), MAIN_CONDITIONS)


def _mean(scores):
    return float(np.mean(list(scores.values())))


def _val_size(cohort, setting, ci, si):
    labels = _labels(cohort, setting)
    k = len(labels)
    validation = _load(_selection_dir(cohort, setting) / "validation_predictions.npz", _validation_keys())
    test = _main_test(cohort, setting)
    stored = _read_json(_selection_dir(cohort, setting) / "selection.json")["locked_policy"]["policies"]
    fixed_test = _mean(policy_macro_f1(test, fixed_policies(MAIN_CONDITIONS), k))
    full_test = _mean(policy_macro_f1(test, stored, k))
    rows = []
    for size in VAL_SIZES:
        for repeat in range(VAL_SIZE_REPEATS):
            index = stratified_group_subsample(
                validation["group_ids"], validation["targets"], size, _rng(VAL_SIZE_SEED, ci, si, size, repeat)
            )
            with selection_metrics_only():
                chosen = select_a0_mr_validation(endpoint_outputs(validation, FORMAL_CONDITIONS, index), labels)
            policies = chosen["locked_policy"]["policies"]
            rows.append({
                "size": size,
                "repeat": repeat,
                "rows": int(index.size),
                "test_macro_f1": _mean(policy_macro_f1(test, policies, k)),
                "agreement_with_full": policy_agreement(policies, stored),
                "gate_passed": bool(chosen["routing_gate"]["passed"]),
            })
    return {"cohort": cohort, "setting": setting, "fixed_test": fixed_test, "full_test": full_test, "repeats": rows}


def _split_half(cohort, setting, ci, si, start, stop):
    labels = _labels(cohort, setting)
    k = len(labels)
    validation = _load(_selection_dir(cohort, setting) / "validation_predictions.npz", _validation_keys())
    test = _main_test(cohort, setting)
    stored = _read_json(_selection_dir(cohort, setting) / "selection.json")["locked_policy"]["policies"]
    fixed = fixed_policies(MAIN_CONDITIONS)
    full_test = _mean(policy_macro_f1(test, stored, k))
    fixed_test = _mean(policy_macro_f1(test, fixed, k))
    main = lambda outputs: {c: outputs[c] for c in MAIN_CONDITIONS}
    rows = []
    for repeat in range(start, stop):
        a, b = stratified_group_halves(validation["group_ids"], validation["targets"], _rng(SPLIT_SEED, ci, si, repeat))
        out_a = endpoint_outputs(validation, FORMAL_CONDITIONS, a)
        out_b = endpoint_outputs(validation, FORMAL_CONDITIONS, b)
        with selection_metrics_only():
            sel_a = select_a0_mr_validation(out_a, labels)
            sel_b = select_a0_mr_validation(out_b, labels)
        pa, pb = sel_a["locked_policy"]["policies"], sel_b["locked_policy"]["policies"]
        gain = lambda outputs, policies: _mean(policy_macro_f1(main(outputs), policies, k)) - _mean(
            policy_macro_f1(main(outputs), fixed, k))
        optimism = 0.5 * ((gain(out_a, pa) - gain(out_b, pa)) + (gain(out_b, pb) - gain(out_a, pb)))
        test_a, test_b = _mean(policy_macro_f1(test, pa, k)), _mean(policy_macro_f1(test, pb, k))
        rows.append({
            "repeat": repeat,
            "half_rows": [int(a.size), int(b.size)],
            "policy_agreement": policy_agreement(pa, pb),
            "policy_agreement_random": policy_agreement(pa, pb, RANDOM_CONDITIONS),
            "gate_agreement": sel_a["routing_gate"]["passed"] == sel_b["routing_gate"]["passed"],
            "optimism": optimism,
            "test_minus_full": 0.5 * (test_a + test_b) - full_test,
            "test_minus_fixed": 0.5 * (test_a + test_b) - fixed_test,
        })
    return {"cohort": cohort, "setting": setting, "repeats": rows}


def _test_outputs(test, conditions):
    return {c: {"targets": test["targets"]} for c in conditions}


def _bootstrap(cohort, setting):
    labels = _labels(cohort, setting)
    plans = [
        ("a0_mr", "xgboost", {"random": RANDOM_CONDITIONS, "structured": STRUCTURED_CONDITIONS}),
        ("a0_mr", "lightgbm", {"random": RANDOM_CONDITIONS, "structured": STRUCTURED_CONDITIONS}),
        ("a0_mr", "fixed_a0", {"random": RANDOM_CONDITIONS}),
        ("a0_mr_registered", "a0_mr", {"profiles": PROFILE_CONDITIONS}),
        ("a0_mr_registered", "xgboost", {"new": NEW_CONDITIONS}),
        ("a0_mr_registered", "lightgbm", {"new": NEW_CONDITIONS}),
    ]
    keys = {"targets"} | {f"{c}__{m}" for first, second, groups in plans for conds in groups.values()
                          for c in conds for m in (first, second)}
    sealed = _sealed_dir(cohort, setting)
    test = _load(sealed / "predictions.npz", sorted(keys))
    with np.load(sealed / "predictions.npz") as archive:
        group_ids = archive["group_ids"]
    result = {}
    for first, second, groups in plans:
        for label, conds in groups.items():
            summary = paired_cluster_bootstrap_macro_f1(
                _test_outputs(test, conds),
                {c: test[f"{c}__{first}"] for c in conds},
                {c: test[f"{c}__{second}"] for c in conds},
                group_ids,
                class_count=len(labels),
                resamples=CLUSTER_RESAMPLES,
                seed=CLUSTER_SEED,
            )
            result[f"{first}_vs_{second}__{label}"] = summary["summaries"]
    return {"cohort": cohort, "setting": setting, "cluster_count": int(np.unique(group_ids).size), "comparisons": result}


def _rare(cohort, setting):
    rare = tuple(get_formal_dataset_spec(cohort).rare_coarse_labels)
    metrics = _read_json(_sealed_dir(cohort, setting) / "metrics.json")["metrics"]
    rows = []
    for condition in TEST_CONDITIONS:
        for model in ("xgboost", "lightgbm", "a0_mr", "a0_mr_registered"):
            values = metrics[condition][model]
            for label in rare:
                support = int(values["per_class_support"][label])
                recall = float(values["per_class_recall"][label])
                successes = int(round(recall * support))
                interval = wilson_interval(successes, support)
                rows.append({
                    "condition": condition, "model": model, "label": label, "support": support,
                    "successes": successes, "recall": recall,
                    "wilson_low": None if interval is None else interval[0],
                    "wilson_high": None if interval is None else interval[1],
                })
    screen = None
    if rare:
        deltas = [
            metrics[c]["a0_mr"]["per_class_recall"][label] - metrics[c]["xgboost"]["per_class_recall"][label]
            for c in MAIN_CONDITIONS for label in rare
        ]
        screen = {"mean_recall_delta_vs_xgboost": float(np.mean(deltas)),
                  "flagged": bool(np.mean(deltas) < RARE_RECALL_SCREEN)}
    return {"cohort": cohort, "setting": setting, "rare_labels": list(rare), "rows": rows, "screen": screen}


def _masks(condition, count, rng):
    columns = {name: i for i, name in enumerate(MODEL_FIELDS)}
    mask = np.zeros((count, len(MODEL_FIELDS)), dtype=bool)

    def hide(fields, rows=None):
        cols = [columns[f] for f in fields]
        if rows is None:
            mask[:, cols] = True
        else:
            mask[np.ix_(np.flatnonzero(rows), cols)] = True

    if condition.startswith("missing_"):
        k = {"missing_30pct": 12, "missing_50pct": 20}[condition]
        order = np.argsort(rng.random((count, len(MODEL_FIELDS))), axis=1)[:, :k]
        np.put_along_axis(mask, order, True, axis=1)
    elif condition.startswith("structured__"):
        hide(STRUCTURED_MISSINGNESS_GROUPS[condition.removeprefix("structured__")])
    elif condition.startswith("profile__"):
        hide(EXPORTER_PROFILES[condition.removeprefix("profile__")])
    elif condition.startswith("intermittent__"):
        spec = INTERMITTENT_PROFILES[condition.removeprefix("intermittent__")]
        hide(STRUCTURED_MISSINGNESS_GROUPS[spec["group"]], rng.random(count) < spec["fraction"])
    elif condition.startswith("compound__"):
        hide(COMPOUND_PROFILES["compound_" + condition.removeprefix("compound__")])
    return mask


def _latency(cohort, setting):
    selection_dir = _selection_dir(cohort, setting)
    registered = _read_json(selection_dir / "registered_selection.json")["registered_policies"]
    locks = _read_json(selection_dir / "temperature_lock.json")["a0_mr_registered"]
    temperatures = {name: lock["temperature"] for name, lock in locks.items()}
    test = _load(_sealed_dir(cohort, setting) / "predictions.npz",
                 ["targets", "clean__xgboost", "clean__schemaflow_pretrained"])
    count = test["targets"].size
    resolver = build_supplementary_resolver(tuple(EXPORTER_PROFILES))
    rows = {}
    for condition in TEST_CONDITIONS:
        mask = _masks(condition, count, _rng(CLUSTER_SEED, COHORTS.index(cohort), SETTINGS.index(setting)))
        resolve_s, dispatch_s = [], []
        for _ in range(3):
            started = time.perf_counter()
            names = resolver.resolve_matrix(mask, MODEL_FIELDS)
            resolve_s.append(time.perf_counter() - started)
            started = time.perf_counter()
            dispatch_probabilities(test["clean__xgboost"], test["clean__schemaflow_pretrained"], names,
                                   registered, temperatures=temperatures)
            dispatch_s.append(time.perf_counter() - started)
        rows[condition] = {
            "records": int(count),
            "resolve_us_per_record": 1e6 * float(np.median(resolve_s)) / count,
            "dispatch_us_per_record": 1e6 * float(np.median(dispatch_s)) / count,
        }
    registry_bytes = len(json.dumps({"policies": registered, "temperatures": temperatures}, sort_keys=True).encode())
    return {"cohort": cohort, "setting": setting, "conditions": rows, "registry_json_bytes": registry_bytes,
            "scope": "CPU only; endpoint inference seconds are reported separately in the sealed-test metrics"}


def _task_results(section):
    rows = {}
    for path in sorted((OUTPUT / "tasks" / section).glob("*.json")):
        payload = _read_json(path)
        rows.setdefault((payload["cohort"], payload["setting"]), []).append(payload)
    return rows


def _by_dataset(keys):
    out = {}
    for cohort, setting in keys:
        out.setdefault(_dataset(cohort), []).append((cohort, setting))
    return out


def _spread(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)), "n": len(values)}


def _paper_bootstrap(deltas):
    if any(len(v) != 9 for v in deltas.values()) or set(deltas) != set(DATASETS):
        return None
    return dataset_stratified_paired_setting_bootstrap(
        {DATASETS[d]: v for d, v in deltas.items()}, datasets=list(DATASETS.values()),
        resamples=PAPER_BOOTSTRAP_RESAMPLES, seed=PAPER_BOOTSTRAP_SEED,
    )


def _summarize():
    primary = {k: v[0] for k, v in _task_results("primary").items()}
    controls = {k: v[0] for k, v in _task_results("controls").items()}
    stacking = {k: v[0] for k, v in _task_results("stacking").items()}
    groups = _by_dataset(sorted(primary, key=lambda k: (COHORTS.index(k[0]), SETTINGS.index(k[1]))))
    f1 = lambda key, c, m: primary[key]["metrics"][c][m]["macro_f1"]
    models = ("a0_mr", "a0_mr_registered", "xgboost", "lightgbm", "fixed_a0", "schemaflow_pretrained")
    table = {d: {c: {m: _spread([f1(k, c, m) for k in keys]) for m in models} for c in TEST_CONDITIONS}
             for d, keys in groups.items()}

    def three(key, model):
        if model in ("global_validation_fusion", "always_conditioned_linear_or_log"):
            return _mean(controls[key]["test_macro_f1"][model]) if key in controls else None
        if model == "logistic_stacking":
            row = stacking.get(key)
            return _mean(row["test_macro_f1"]) if row and row["status"] == "completed" else None
        return float(np.mean([f1(key, c, model) for c in MAIN_CONDITIONS]))

    gains = {}
    for comparator in ("xgboost", "lightgbm", "fixed_a0", "schemaflow_pretrained", "global_validation_fusion",
                       "always_conditioned_linear_or_log", "logistic_stacking"):
        deltas = {}
        for d, keys in groups.items():
            values = [(three(k, "a0_mr"), three(k, comparator)) for k in keys]
            if any(b is None for _, b in values):
                deltas = None
                break
            deltas[d] = [a - b for a, b in values]
        if deltas is None:
            gains[comparator] = None
            continue
        flat = [x for v in deltas.values() for x in v]
        gains[comparator] = {
            "by_dataset": {DATASETS[d]: float(np.mean(v)) for d, v in deltas.items()},
            "bootstrap": _paper_bootstrap(deltas),
            "directions_pos_tie_neg": [sum(x > 1e-12 for x in flat), sum(abs(x) <= 1e-12 for x in flat),
                                       sum(x < -1e-12 for x in flat)],
        }
    structured = {DATASETS[d]: {c: {
        "a0_mr_minus_xgboost": table[d][c]["a0_mr"]["mean"] - table[d][c]["xgboost"]["mean"],
        "a0_mr_minus_lightgbm": table[d][c]["a0_mr"]["mean"] - table[d][c]["lightgbm"]["mean"],
    } for c in STRUCTURED_CONDITIONS} for d in groups}
    new = {DATASETS[d]: {c: {
        "a0_mr": table[d][c]["a0_mr"]["mean"],
        "a0_mr_registered": table[d][c]["a0_mr_registered"]["mean"],
        "xgboost": table[d][c]["xgboost"]["mean"],
        "lightgbm": table[d][c]["lightgbm"]["mean"],
        "registration_gain": table[d][c]["a0_mr_registered"]["mean"] - table[d][c]["a0_mr"]["mean"],
        "fallback_fraction_a0_mr": float(np.mean([primary[k]["routing"][c]["a0_mr"]["fallback_fraction"] for k in keys])),
        "fallback_fraction_registered": float(np.mean(
            [primary[k]["routing"][c]["a0_mr_registered"]["fallback_fraction"] for k in keys])),
    } for c in NEW_CONDITIONS} for d, keys in groups.items()}
    calibration = {DATASETS[d]: {metric: {
        name: float(np.mean([primary[k]["metrics"][c][name][metric] for k in keys for c in FORMAL_CONDITIONS]))
        for name in ("a0_mr", "a0_mr__calibrated")} for metric in ("nll", "brier", "ece")}
        for d, keys in groups.items()}
    return primary, controls, stacking, groups, table, gains, structured, new, calibration


def _summarize_rest(groups):
    val = _task_results("val_size")
    split = _task_results("split_half")
    boot = _task_results("bootstrap")
    rare = _task_results("rare")
    latency = _task_results("latency")
    val_size = {}
    for d, keys in groups.items():
        val_size[DATASETS[d]] = {}
        for size in VAL_SIZES:
            regret, over_fixed, gate, agree = [], [], [], []
            for key in keys:
                task = val[key][0]
                rows = [r for r in task["repeats"] if r["size"] == size]
                regret.append(np.mean([r["test_macro_f1"] for r in rows]) - task["full_test"])
                over_fixed.append(np.mean([r["test_macro_f1"] for r in rows]) - task["fixed_test"])
                gate.append(np.mean([r["gate_passed"] for r in rows]))
                agree.append(np.mean([r["agreement_with_full"] for r in rows]))
            val_size[DATASETS[d]][str(size)] = {
                "test_minus_full_validation": float(np.mean(regret)),
                "test_minus_fixed": float(np.mean(over_fixed)),
                "gate_pass_rate": float(np.mean(gate)),
                "policy_agreement_with_full": float(np.mean(agree)),
            }
    split_half = {}
    for d, keys in groups.items():
        rows = [r for key in keys for task in split[key] for r in task["repeats"]]
        split_half[DATASETS[d]] = {
            "repeats_per_setting": len(rows) // len(keys),
            **{name: float(np.mean([r[name] for r in rows])) for name in (
                "policy_agreement", "policy_agreement_random", "gate_agreement", "optimism",
                "test_minus_full", "test_minus_fixed")},
        }
    signs = {}
    for key, tasks in boot.items():
        for name, summaries in tasks[0]["comparisons"].items():
            row = summaries["mean_across_conditions"]
            bucket = signs.setdefault(name, {}).setdefault(DATASETS[_dataset(key[0])], Counter())
            bucket["above_zero" if row["ci95_lower"] > 0 else "below_zero" if row["ci95_upper"] < 0 else "spans_zero"] += 1
    rare_screen = {}
    for d, keys in groups.items():
        screens = [rare[k][0]["screen"] for k in keys if rare[k][0]["screen"] is not None]
        rare_screen[DATASETS[d]] = None if not screens else {
            "settings": len(screens),
            "flagged": sum(s["flagged"] for s in screens),
            "worst_mean_recall_delta": float(min(s["mean_recall_delta_vs_xgboost"] for s in screens)),
        }
    timing = [c for tasks in latency.values() for c in tasks[0]["conditions"].values()]
    latency_summary = {
        "median_resolve_us_per_record": float(np.median([r["resolve_us_per_record"] for r in timing])),
        "median_dispatch_us_per_record": float(np.median([r["dispatch_us_per_record"] for r in timing])),
        "max_registry_json_bytes": max(t[0]["registry_json_bytes"] for t in latency.values()),
    }
    signs = {name: {d: dict(c) for d, c in value.items()} for name, value in signs.items()}
    return val_size, split_half, signs, rare, rare_screen, latency_summary


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return str(path), digest.hexdigest()


def summarize(scratch=False):
    missing = [(section, key) for section, _, key in _tasks() if not (OUTPUT / "tasks" / section / f"{key}.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} analysis tasks are missing, e.g. {missing[:3]}")
    primary, controls, stacking, groups, table, gains, structured, new, calibration = _summarize()
    val_size, split_half, signs, rare, rare_screen, latency = _summarize_rest(groups)
    summary = {
        "format_version": 1,
        "matrix": str(MATRIX),
        "settings_per_dataset": {DATASETS[d]: len(k) for d, k in groups.items()},
        "macro_f1_table": {DATASETS[d]: v for d, v in table.items()},
        "a0_mr_three_condition_gains": gains,
        "structured_comparisons": structured,
        "new_conditions": new,
        "calibration": calibration,
        "validation_size_curve": val_size,
        "split_half_validation": split_half,
        "cluster_bootstrap_sign_counts": signs,
        "rare_class_screen": rare_screen,
        "resolver_latency": latency,
        "stacking_failures": [list(k) for k, v in stacking.items() if v["status"] != "completed"],
        "gate_pass_counts": {DATASETS[d]: sum(primary[k]["gate_passed"] for k in keys) for d, keys in groups.items()},
        "resolver_agreement_all_one": all(
            value == 1.0 for p in primary.values() for by in p["resolver_label_agreement"].values() for value in by.values()),
        "reproduction_max_abs_error": {
            name: max(c["reproduction"][name] for c in controls.values())
            for name in ("a0_mr_pool_max_abs_error", "stored_macro_f1_max_abs_error")},
    }
    _write_json(OUTPUT / "summary.json", summary)
    rows = [{"dataset": DATASETS[d], "condition": c, "model": m, **v}
            for d, by in table.items() for c, models in by.items() for m, v in models.items()]
    _write_csv(OUTPUT / "macro_f1_by_condition.csv", rows)
    rows = []
    for comparator, row in gains.items():
        if row is None:
            continue
        boot = row["bootstrap"] or {}
        rows.append({"comparator": comparator, **{f"gain_{k}": v for k, v in row["by_dataset"].items()},
                     "equal_dataset_gain": boot.get("observed_equal_dataset_mean"),
                     "ci95_lower": boot.get("ci95_lower"), "ci95_upper": boot.get("ci95_upper"),
                     "pos_tie_neg": "/".join(map(str, row["directions_pos_tie_neg"]))})
    _write_csv(OUTPUT / "a0_mr_gains.csv", rows)
    rows = [{"cohort": k[0], "setting": k[1], **r} for k, v in sorted(rare.items()) for r in v[0]["rows"]]
    _write_csv(OUTPUT / "rare_class_recall.csv", rows)
    if not scratch:
        _provenance()
    return summary


def _provenance():
    receipts = _require_receipts()
    files = [p for c in COHORTS for s in SETTINGS for p in (
        _sealed_dir(c, s) / "predictions.npz", _selection_dir(c, s) / "validation_predictions.npz")]
    with multiprocessing.Pool(8) as pool:
        hashes = dict(pool.map(_hash, files))
    _write_json(OUTPUT / "provenance.json", {
        "analysis_git": _git_state(),
        "matrix_commits": sorted({r["git"]["commit"] for r in receipts.values()}),
        "supplementary_protocol_hash": frozen_supplementary_hash(),
        "receipt_sha256": {c: _hash(MATRIX / c / "sealed_test_access.completed.json")[1] for c in COHORTS},
        "archive_sha256": hashes,
        "seeds": {"validation_size": VAL_SIZE_SEED, "split_half": SPLIT_SEED, "cluster_bootstrap": CLUSTER_SEED,
                  "setting_bootstrap": PAPER_BOOTSTRAP_SEED, "derivation": "SeedSequence([seed, cohort, setting, ...])"},
        "selection_metrics": "selection_metrics_only reproduces all 27 stored selections exactly",
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })


def _tasks():
    tasks = []
    for ci, (cohort, setting) in enumerate(_cohort_settings()):
        c, s = COHORTS.index(cohort), SETTINGS.index(setting)
        key = f"{cohort}__{setting}"
        for section, fn in (("primary", _primary), ("controls", _controls), ("stacking", _stack),
                            ("bootstrap", _bootstrap), ("rare", _rare), ("latency", _latency)):
            tasks.append((section, (fn.__name__, cohort, setting), key))
        tasks.append(("val_size", ("_val_size", cohort, setting, c, s), key))
        for start in range(0, SPLIT_HALF_REPEATS, SPLIT_CHUNK):
            stop = min(start + SPLIT_CHUNK, SPLIT_HALF_REPEATS)
            tasks.append(("split_half", ("_split_half", cohort, setting, c, s, start, stop), f"{key}__{start:02d}"))
    order = {"split_half": 0, "val_size": 1, "stacking": 2, "bootstrap": 3}
    return sorted(tasks, key=lambda t: order.get(t[0], 4))


def _run_task(task):
    section, (name, *arguments), key = task
    started = time.time()
    _cached(section, key, lambda: globals()[name](*arguments))
    return section, key, round(time.time() - started)


def main(argv=None):
    global OUTPUT, VAL_SIZE_REPEATS, SPLIT_HALF_REPEATS, COHORTS, SETTINGS
    parser = argparse.ArgumentParser(description="Revision supplementary post-evaluation analyses")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--scratch", type=Path, help="smoke run: one setting, few repeats, no git check")
    args = parser.parse_args(argv)
    _require_receipts()
    if args.scratch:
        OUTPUT, VAL_SIZE_REPEATS, SPLIT_HALF_REPEATS = args.scratch, 1, 2
        COHORTS, SETTINGS = ("nf_ton_rev1", "nf_bot_rev1", "nf_cse_rev1"), ("pct-01-seed-17",)
    else:
        _git_state()
    tasks = [t for t in _tasks() if not (OUTPUT / "tasks" / t[0] / f"{t[2]}.json").exists()]
    print(json.dumps({"pending_tasks": len(tasks), "output": str(OUTPUT)}), flush=True)
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for section, key, seconds in pool.imap_unordered(_run_task, tasks):
            print(f"{time.strftime('%H:%M:%S')} done {section} {key} {seconds}s", flush=True)
    summary = summarize(scratch=bool(args.scratch))
    print(json.dumps({"status": "completed", "gains": {k: (v or {}).get("by_dataset") for k, v in
                                                       summary["a0_mr_three_condition_gains"].items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
