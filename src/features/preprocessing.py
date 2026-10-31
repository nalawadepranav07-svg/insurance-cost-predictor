"""Leak-safe preprocessing.

Everything learned from data (medians, scaling statistics, category lists) lives inside a
scikit-learn ``Pipeline`` so it is fitted only on training data of each CV fold, and the
*same fitted object* is reused at inference time.
"""
from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cfg: dict[str, Any], scale_numeric: bool) -> ColumnTransformer:
    """Create the ColumnTransformer.

    * Numeric: median imputation (robust to skew). Standardisation only when
      ``scale_numeric`` is True (needed for linear models; tree models are scale-invariant).
    * Categorical: most-frequent imputation, then one-hot encoding. ``handle_unknown='ignore'``
      so an unseen category yields all-zero columns instead of crashing (the prediction layer
      still rejects unknown categories explicitly).
    """
    schema = cfg["schema"]
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), schema["numeric_features"]),
            ("cat", Pipeline(cat_steps), schema["categorical_features"]),
        ],
        remainder="drop",  # anything else (e.g. the target) is dropped, never passed through
        verbose_feature_names_out=False,
    )
