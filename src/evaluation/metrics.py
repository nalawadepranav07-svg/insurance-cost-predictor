"""Metrics, bootstrap uncertainty and subgroup error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MIN_GROUP_SIZE = 30  # below this, subgroup metrics are flagged as unreliable


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE and RMSE (USD, lower is better) and R² (higher is better; can be negative)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap confidence intervals for the hold-out metrics."""
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    draws = {"mae": [], "rmse": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        m = regression_metrics(yt[idx], yp[idx])
        for k in draws:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in draws.items()}


def add_bands(X: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable age / BMI bands for subgroup analysis (analysis only, not a model input)."""
    out = X.copy()
    out["age_band"] = pd.cut(out["age"], [17, 29, 44, 64], labels=["18-29", "30-44", "45-64"])
    out["bmi_band"] = pd.cut(
        out["bmi"], [0, 18.5, 25, 30, 100], right=False,
        labels=["<18.5", "18.5-24.9", "25-29.9", "30+"],
    )
    return out


def subgroup_errors(X: pd.DataFrame, y_true, y_pred,
                    group_cols=("sex", "smoker", "region", "age_band", "bmi_band")) -> pd.DataFrame:
    """MAE / RMSE / R² / mean signed error per subgroup, with sample counts and a small-n flag."""
    data = add_bands(X).reset_index(drop=True)
    data["y"] = np.asarray(y_true)
    data["pred"] = np.asarray(y_pred)
    rows = []
    for col in group_cols:
        for level, g in data.groupby(col, observed=True):
            n = len(g)
            m = regression_metrics(g["y"], g["pred"]) if n >= 2 else {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
            rows.append({
                "group": col, "level": str(level), "n": n,
                "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                "mean_error": float((g["pred"] - g["y"]).mean()),  # >0 = over-prediction
                "small_sample": n < MIN_GROUP_SIZE,
            })
    return pd.DataFrame(rows)
