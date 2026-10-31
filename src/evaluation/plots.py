"""Matplotlib figures for reports (saved as PNG, never shown interactively)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, BLUE, GREY, AMBER = "#0B1F44", "#2F5BEA", "#94A3B8", "#D97706"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def model_comparison_plot(comparison: list[dict], path: Path) -> Path:
    """Mean ± std of CV RMSE and MAE per model."""
    names = [r["model"] for r in comparison]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, label in zip(axes, ["rmse", "mae"], ["RMSE (USD)", "MAE (USD)"]):
        means = [r["cv"][key]["mean"] for r in comparison]
        stds = [r["cv"][key]["std"] for r in comparison]
        ax.barh(names, means, xerr=stds, color=BLUE, ecolor=NAVY, capsize=3)
        ax.set_xlabel(f"5-fold CV {label} (mean ± std; lower is better)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def predicted_vs_actual(y_true, y_pred, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.65, color=BLUE)
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lim, lim, "--", color=AMBER, label="Perfect prediction")
    ax.set(xlabel="Actual charges (USD)", ylabel="Predicted charges (USD)", title=title)
    ax.legend()
    return _save(fig, path)


def residual_plots(y_true, y_pred, path: Path) -> Path:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_pred, resid, s=18, alpha=0.65, color=BLUE)
    axes[0].axhline(0, color=AMBER, ls="--")
    axes[0].set(xlabel="Predicted charges (USD)", ylabel="Residual: actual − predicted (USD)",
                title="Residuals vs predicted (hold-out)")
    axes[1].hist(resid, bins=30, color=BLUE, edgecolor="white")
    axes[1].set(xlabel="Residual (USD)", ylabel="Count", title="Residual distribution (hold-out)")
    return _save(fig, path)


def importance_plot(names: list[str], means: list[float], stds: list[float], path: Path) -> Path:
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order],
            color=BLUE, ecolor=NAVY, capsize=3)
    ax.set_xlabel("Increase in hold-out RMSE when feature is shuffled (USD)")
    ax.set_title("Permutation importance (predictive association, not causation)")
    return _save(fig, path)
