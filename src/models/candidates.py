"""Candidate regression models, each wrapped in a full preprocessing + estimator Pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor

# Names are used as keys in every report, so keep them stable.
LINEAR = "Linear Regression"
LINEAR_LOG = "Linear Regression (log target)"
RF = "Random Forest"
GB = "Gradient Boosting"
RF_TUNED = "Random Forest (tuned)"
GB_TUNED = "Gradient Boosting (tuned)"

BASELINES = [LINEAR, LINEAR_LOG, RF, GB]
TUNED = [RF_TUNED, GB_TUNED]

# Deliberately modest search spaces (see docs/decisions).
RF_SPACE = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [3, 5, 8, 12, None],
    "model__min_samples_leaf": [1, 2, 4, 8, 16],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features": [1.0, 0.7, 0.5],
}
GB_SPACE = {
    "model__n_estimators": [50, 100, 200, 300, 400],
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [1, 3, 5, 10, 20],
    "model__subsample": [0.7, 0.85, 1.0],
}


def _pipeline(cfg: dict[str, Any], estimator: Any, scale: bool) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(cfg, scale_numeric=scale)), ("model", estimator)])


def build_baselines(cfg: dict[str, Any]) -> dict[str, Pipeline]:
    """Untuned candidates. All share the same preprocessing design."""
    seed = cfg["training"]["random_seed"]
    return {
        LINEAR: _pipeline(cfg, LinearRegression(), scale=True),
        # Charges are right-skewed: fit on log1p(charges), predict back on the dollar scale.
        LINEAR_LOG: _pipeline(
            cfg,
            TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1),
            scale=True,
        ),
        RF: _pipeline(cfg, RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1), scale=False),
        GB: _pipeline(cfg, GradientBoostingRegressor(random_state=seed), scale=False),
    }


def build_tuned(cfg: dict[str, Any], base: str, n_jobs: int = -1) -> RandomizedSearchCV:
    """RandomizedSearchCV wrapper around a tree-based baseline.

    Used inside the outer CV loop (=> nested CV) and once more on the full training set
    to choose the parameters of the shipped model.
    """
    t = cfg["training"]
    tune = t["tuning"]
    pipe = build_baselines(cfg)[base]
    space = RF_SPACE if base == RF else GB_SPACE
    inner = KFold(n_splits=tune["inner_cv_folds"], shuffle=True, random_state=t["random_seed"] + 1)
    return RandomizedSearchCV(
        pipe,
        param_distributions=space,
        n_iter=tune["n_iter"],
        cv=inner,
        scoring="neg_root_mean_squared_error",
        random_state=t["random_seed"],
        n_jobs=n_jobs,
        refit=True,
    )
