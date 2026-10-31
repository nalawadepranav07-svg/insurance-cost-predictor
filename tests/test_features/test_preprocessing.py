import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from src.data.loader import split_xy
from src.features.preprocessing import build_preprocessor
from src.utils.config import feature_columns


def test_feature_columns_exclude_target(cfg):
    assert "charges" not in feature_columns(cfg)
    assert len(feature_columns(cfg)) == 6


def test_columns_routed_correctly(cfg, synthetic_df):
    X, _ = split_xy(synthetic_df, cfg)
    pre = build_preprocessor(cfg, scale_numeric=True).fit(X)
    names = list(pre.get_feature_names_out())
    assert {"age", "bmi", "children"} <= set(names)
    assert any(n.startswith("region_") for n in names)
    assert "charges" not in names
    assert pre.transform(X).shape == (len(X), len(names))


def test_unseen_category_does_not_crash(cfg, synthetic_df):
    X, _ = split_xy(synthetic_df, cfg)
    pre = build_preprocessor(cfg, scale_numeric=False).fit(X)
    odd = X.iloc[:2].copy()
    odd["region"] = "atlantis"
    out = pre.transform(odd)
    assert out.shape[1] == pre.transform(X.iloc[:2]).shape[1]
    assert np.isfinite(out).all()


def test_missing_values_are_imputed(cfg, synthetic_df):
    X, _ = split_xy(synthetic_df, cfg)
    X = X.copy()
    X.loc[0, "bmi"] = np.nan
    X.loc[1, "smoker"] = np.nan
    out = build_preprocessor(cfg, scale_numeric=True).fit_transform(X)
    assert np.isfinite(out).all()


def test_transforms_learn_only_from_training_fold(cfg, synthetic_df):
    """Leakage check: scaler statistics must equal the TRAIN fold's, not the full data's."""
    X, _ = split_xy(synthetic_df, cfg)
    for train_idx, _test_idx in KFold(5, shuffle=True, random_state=0).split(X):
        pre = clone(build_preprocessor(cfg, scale_numeric=True)).fit(X.iloc[train_idx])
        scaler = pre.named_transformers_["num"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[train_idx][["age", "bmi", "children"]].mean().values)
        assert not np.allclose(scaler.mean_, X[["age", "bmi", "children"]].mean().values)


def test_column_order_is_aligned_by_name(cfg, synthetic_df):
    X, _ = split_xy(synthetic_df, cfg)
    pre = build_preprocessor(cfg, scale_numeric=True).fit(X)
    shuffled = X[list(reversed(X.columns))]
    np.testing.assert_allclose(pre.transform(X), pre.transform(shuffled))


def test_target_column_in_input_is_dropped(cfg, synthetic_df):
    pre = build_preprocessor(cfg, scale_numeric=True)
    pre.fit(synthetic_df.drop(columns=["charges"]))
    out = pre.transform(synthetic_df)  # includes charges; remainder='drop'
    assert out.shape[1] == len(pre.get_feature_names_out())
