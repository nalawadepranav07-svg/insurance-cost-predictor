"""Training orchestration: CV comparison -> selection -> hold-out evaluation -> persistence.

Protocol (see docs/architecture):
  1. Hold out a stratified test set that is NOT used for model selection or tuning.
  2. On the training set, compare every candidate on the *same* shuffled 5-fold splits.
     Tuned candidates use nested CV (RandomizedSearchCV inside each outer training fold).
  3. Select the model with the lowest mean CV RMSE.
  4. Fit the selected pipeline (re-tuning on the full training set if it is a tuned
     candidate), then evaluate ONCE on the hold-out set.
  5. Persist the fitted pipeline + metadata. The shipped artifact is fitted on the training
     split only, so the reported hold-out metrics describe exactly the shipped model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_validate

from src.data.loader import load_raw, sha256_of, split_xy, train_test_holdout
from src.evaluation import metrics as M
from src.evaluation import plots
from src.models import candidates as C
from src.utils.config import feature_columns, resolve
from src.utils.reproducibility import environment_versions, set_global_seed

log = logging.getLogger("coverwise.train")

SCORING = {
    "mae": "neg_mean_absolute_error",       # sklearn negates errors so "higher is better";
    "rmse": "neg_root_mean_squared_error",  # we flip the sign back before reporting.
    "r2": "r2",
}


def make_outer_cv(cfg: dict[str, Any]) -> KFold:
    """The single 5-fold splitter shared by every candidate."""
    t = cfg["training"]
    return KFold(n_splits=t["cv_folds"], shuffle=True, random_state=t["random_seed"])


def cross_validate_model(estimator, X: pd.DataFrame, y: pd.Series, cv: KFold) -> dict[str, Any]:
    """Cross-validate a full pipeline; returns fold-level values plus mean/std per metric."""
    start = time.perf_counter()
    res = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
    out: dict[str, Any] = {"wall_time_s": round(time.perf_counter() - start, 2)}
    for key in SCORING:
        vals = res[f"test_{key}"]
        vals = vals if key == "r2" else -vals  # undo sklearn's sign convention for errors
        out[key] = {
            "folds": [float(v) for v in vals],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
        }
    out["fit_time_per_fold_s"] = float(np.mean(res["fit_time"]))
    return out


def compare_models(cfg: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> list[dict]:
    """Run identical-fold CV for every baseline and (if enabled) tuned candidate."""
    cv = make_outer_cv(cfg)
    baselines = C.build_baselines(cfg)
    results = []
    for name in C.BASELINES:
        log.info("CV: %s", name)
        r = cross_validate_model(baselines[name], X_train, y_train, cv)
        results.append({"model": name, "tuned": False, "cv": r})
    if cfg["training"]["tuning"]["enabled"]:
        for name, base in ((C.RF_TUNED, C.RF), (C.GB_TUNED, C.GB)):
            log.info("Nested CV: %s (this is the slow step)", name)
            r = cross_validate_model(C.build_tuned(cfg, base), X_train, y_train, cv)
            results.append({"model": name, "tuned": True, "base": base, "cv": r})
    return results


def select_best(results: list[dict]) -> dict:
    """Lowest mean CV RMSE wins. RMSE is in USD and penalises large misses, which matter
    most for a heavy-tailed cost target. Training scores are never consulted."""
    return min(results, key=lambda r: r["cv"]["rmse"]["mean"])


def _fit_final(cfg: dict[str, Any], choice: dict, X_train, y_train):
    """Fit the chosen candidate on the whole training set; returns (pipeline, best_params)."""
    if choice["tuned"]:
        search = C.build_tuned(cfg, choice["base"])
        search.fit(X_train, y_train)
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        return search.best_estimator_, params
    pipe = C.build_baselines(cfg)[choice["model"]]
    pipe.fit(X_train, y_train)
    return pipe, {}


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the full workflow and write all artifacts and reports. Returns the metadata."""
    seed = cfg["training"]["random_seed"]
    set_global_seed(seed)
    df = load_raw(cfg)
    X, y = split_xy(df, cfg)
    assert cfg["dataset"]["target"] not in X.columns  # leakage guard
    X_train, X_test, y_train, y_test = train_test_holdout(X, y, cfg)
    log.info("Train rows: %d | Hold-out rows: %d", len(X_train), len(X_test))

    results = compare_models(cfg, X_train, y_train)
    best = select_best(results)
    log.info("Selected (lowest mean CV RMSE): %s", best["model"])

    pipeline, best_params = _fit_final(cfg, best, X_train, y_train)
    pred = pipeline.predict(X_test)
    holdout = M.regression_metrics(y_test, pred)
    holdout_ci = M.bootstrap_ci(y_test, pred, seed=seed)

    # Permutation importance on RAW features (works for any pipeline, no one-hot confusion).
    pi = permutation_importance(
        pipeline, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=30, random_state=seed,
    )
    importance = sorted(
        ({"feature": f, "mean": float(m), "std": float(s)}
         for f, m, s in zip(X_test.columns, pi.importances_mean, pi.importances_std)),
        key=lambda d: d["mean"], reverse=True,
    )

    # ---- write reports -------------------------------------------------------------
    reports = resolve(cfg, cfg["paths"]["reports_dir"])
    figs = resolve(cfg, cfg["paths"]["figures_dir"])
    cv_dir, mc_dir, ea_dir = reports / "cross_validation", reports / "model_comparison", reports / "error_analysis"
    for d in (cv_dir, mc_dir, ea_dir, figs):
        d.mkdir(parents=True, exist_ok=True)

    outer = make_outer_cv(cfg)
    folds = [{"fold": i + 1, "train_n": int(len(tr)), "test_n": int(len(te)),
              "test_row_positions": [int(v) for v in te]}
             for i, (tr, te) in enumerate(outer.split(X_train))]
    protocol = {
        "splitter": "KFold", "n_splits": cfg["training"]["cv_folds"], "shuffle": True,
        "random_state": seed, "scoring": "MAE, RMSE (USD), R² on each outer test fold",
        "tuning": ("RandomizedSearchCV nested inside each outer training fold: "
                   f"n_iter={cfg['training']['tuning']['n_iter']}, "
                   f"inner folds={cfg['training']['tuning']['inner_cv_folds']}, scoring=RMSE"),
        "preprocessing": "fitted inside each training fold via sklearn Pipeline",
    }
    (cv_dir / "cv_results.json").write_text(json.dumps(
        {"protocol": protocol, "folds": folds, "models": results}, indent=2, default=_jsonable))

    comparison = [
        {"model": r["model"], "tuned": r["tuned"], "cv": {k: r["cv"][k] for k in ("mae", "rmse", "r2")},
         "fit_time_per_fold_s": r["cv"]["fit_time_per_fold_s"], "selected": r["model"] == best["model"]}
        for r in results
    ]
    (mc_dir / "model_comparison.json").write_text(json.dumps(
        {"protocol": protocol, "models": comparison}, indent=2, default=_jsonable))
    rows = [{
        "model": c["model"],
        **{f"cv_{k}_mean": c["cv"][k]["mean"] for k in ("mae", "rmse", "r2")},
        **{f"cv_{k}_std": c["cv"][k]["std"] for k in ("mae", "rmse", "r2")},
        "selected": c["selected"],
    } for c in comparison]
    pd.DataFrame(rows).to_csv(mc_dir / "model_comparison.csv", index=False)

    sub = M.subgroup_errors(X_test, y_test, pred)
    sub.to_csv(ea_dir / "subgroup_errors.csv", index=False)

    plots.model_comparison_plot(comparison, figs / "model_comparison_cv.png")
    plots.predicted_vs_actual(y_test, pred, figs / "holdout_predicted_vs_actual.png",
                              f"{best['model']}: hold-out predictions")
    plots.residual_plots(y_test, pred, figs / "holdout_residuals.png")
    plots.importance_plot([d["feature"] for d in importance], [d["mean"] for d in importance],
                          [d["std"] for d in importance], figs / "permutation_importance.png")

    # ---- persist artifact + metadata ------------------------------------------------
    model_path = resolve(cfg, cfg["paths"]["model_path"])
    meta_path = resolve(cfg, cfg["paths"]["metadata_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    artifact_hash = sha256_of(model_path)
    metadata = {
        "model_id": artifact_hash[:12],
        "model_name": best["model"],
        "best_params": best_params,
        "selection_criterion": "lowest mean 5-fold CV RMSE on the training split",
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_sha256": artifact_hash,
        "dataset_sha256": sha256_of(resolve(cfg, cfg["dataset"]["raw_path"])),
        "dataset_rows": int(len(df)), "train_rows": int(len(X_train)), "holdout_rows": int(len(X_test)),
        "target": cfg["dataset"]["target"], "target_unit": cfg["dataset"]["target_unit"],
        "features": feature_columns(cfg),
        "schema": cfg["schema"],
        "cv_protocol": protocol,
        "cv_metrics": {c["model"]: c["cv"] for c in comparison},
        "holdout_metrics": holdout,
        "holdout_metrics_95ci": holdout_ci,
        "feature_importance": {"method": "permutation importance on hold-out set (RMSE increase, USD)",
                               "values": importance},
        "versions": environment_versions(),
        "random_seed": seed,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=_jsonable))
    return metadata
