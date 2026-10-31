"""Shared fixtures. Training in tests uses a tiny SYNTHETIC dataset and temp directories."""
from __future__ import annotations

import copy

import pytest

from src.utils.config import load_config
from tests.fixtures.synthetic import make_synthetic


@pytest.fixture()
def synthetic_df():
    return make_synthetic()


@pytest.fixture()
def cfg(tmp_path, synthetic_df):
    """Real schema/config, but every path points into a temp dir and tuning is tiny."""
    c = copy.deepcopy(load_config())
    csv = tmp_path / "insurance.csv"
    synthetic_df.to_csv(csv, index=False)
    c["dataset"]["raw_path"] = str(csv)
    c["dataset"]["expected_sha256"] = None
    c["paths"]["model_path"] = str(tmp_path / "model.joblib")
    c["paths"]["metadata_path"] = str(tmp_path / "meta.json")
    c["paths"]["reports_dir"] = str(tmp_path / "reports")
    c["paths"]["figures_dir"] = str(tmp_path / "reports" / "figures")
    c["training"]["tuning"].update(n_iter=2, inner_cv_folds=2)
    return c


@pytest.fixture()
def trained_cfg(cfg):
    """Config for which a model has really been trained (tuning disabled for speed)."""
    from src.models.trainer import run_training
    cfg["training"]["tuning"]["enabled"] = False
    run_training(cfg)
    return cfg


VALID_INPUT = {"age": 35, "sex": "female", "bmi": 27.5, "children": 1, "smoker": "no", "region": "northwest"}
