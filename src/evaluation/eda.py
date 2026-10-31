"""Exploratory data analysis helpers shared by the CLI script and the notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.plots import BLUE, NAVY, _save


def iqr_outliers(s: pd.Series) -> dict[str, float]:
    """Count values outside 1.5×IQR fences (investigated, NOT removed)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {"lower_fence": float(lo), "upper_fence": float(hi), "n_outliers": int(((s < lo) | (s > hi)).sum())}


def summarize(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Numeric facts about the dataset computed from the real data."""
    num = df.select_dtypes("number")
    return {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "describe": num.describe().round(3).to_dict(),
        "target_skewness": float(df[target].skew()),
        "log_target_skewness": float(np.log1p(df[target]).skew()),
        "target_outliers_iqr": iqr_outliers(df[target]),
        "bmi_outliers_iqr": iqr_outliers(df["bmi"]),
        "correlation_with_target": num.corr()[target].drop(target).round(3).to_dict(),
        "mean_charges_by": {
            c: df.groupby(c)[target].agg(["count", "mean", "median"]).round(2).to_dict("index")
            for c in ("smoker", "sex", "region", "children")
        },
        "categorical_balance": {c: df[c].value_counts(normalize=True).round(3).to_dict()
                                for c in ("sex", "smoker", "region")},
    }


def make_figures(df: pd.DataFrame, target: str, out_dir: Path) -> list[Path]:
    """Create labelled EDA figures; returns the saved paths."""
    sns.set_theme(style="whitegrid")
    paths = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[target], bins=40, color=BLUE, ax=ax[0])
    ax[0].set(xlabel="Charges (USD)", title=f"Charges are right-skewed (skew = {df[target].skew():.2f})")
    sns.histplot(np.log1p(df[target]), bins=40, color=NAVY, ax=ax[1])
    ax[1].set(xlabel="log(1 + charges)", title=f"After log transform (skew = {np.log1p(df[target]).skew():.2f})")
    paths.append(_save(fig, out_dir / "eda_target_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="age", y=target, hue="smoker", alpha=0.7, ax=ax,
                    palette={"no": BLUE, "yes": "#D97706"})
    ax.set(xlabel="Age (years)", ylabel="Charges (USD)", title="Age vs charges, coloured by smoker status")
    paths.append(_save(fig, out_dir / "eda_age_vs_charges.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="bmi", y=target, hue="smoker", alpha=0.7, ax=ax,
                    palette={"no": BLUE, "yes": "#D97706"})
    ax.axvline(30, ls="--", color="grey")
    ax.set(xlabel="BMI (kg/m²)", ylabel="Charges (USD)", title="BMI vs charges (dashed line: BMI 30)")
    paths.append(_save(fig, out_dir / "eda_bmi_vs_charges.png"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["smoker", "region", "sex"]):
        sns.boxplot(data=df, x=col, y=target, color=BLUE, ax=ax)
        ax.set(xlabel=col.capitalize(), ylabel="Charges (USD)", title=f"Charges by {col}")
    paths.append(_save(fig, out_dir / "eda_categorical_vs_charges.png"))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Pearson correlation (numeric columns)")
    paths.append(_save(fig, out_dir / "eda_correlation.png"))
    return paths
