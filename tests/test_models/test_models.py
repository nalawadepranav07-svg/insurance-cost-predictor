import json

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.data.loader import load_raw, split_xy, train_test_holdout
from src.models import candidates as C
from src.models.trainer import (cross_validate_model, make_outer_cv, run_training, select_best)
from src.utils.config import resolve


@pytest.fixture()
def split(cfg, synthetic_df):
    X, y = split_xy(synthetic_df, cfg)
    return train_test_holdout(X, y, cfg)


@pytest.mark.parametrize("name", C.BASELINES)
def test_each_candidate_fits_and_predicts(cfg, split, name):
    X_train, X_test, y_train, _ = split
    pipe = C.build_baselines(cfg)[name]
    assert isinstance(pipe, Pipeline) and list(pipe.named_steps) == ["preprocess", "model"]
    pred = pipe.fit(X_train, y_train).predict(X_test)
    assert pred.shape == (len(X_test),) and np.isfinite(pred).all()


def test_at_least_three_required_models_present(cfg):
    names = set(C.build_baselines(cfg))
    assert {C.LINEAR, C.RF, C.GB} <= names


def test_seeded_training_is_repeatable(cfg, split):
    X_train, X_test, y_train, _ = split
    p1 = C.build_baselines(cfg)[C.RF].fit(X_train, y_train).predict(X_test)
    p2 = C.build_baselines(cfg)[C.RF].fit(X_train, y_train).predict(X_test)
    np.testing.assert_allclose(p1, p2)


def test_all_models_use_identical_folds(cfg, split):
    X_train = split[0]
    a = [te.tolist() for _, te in make_outer_cv(cfg).split(X_train)]
    b = [te.tolist() for _, te in make_outer_cv(cfg).split(X_train)]
    assert a == b and len(a) == 5


def test_cv_result_structure_and_sign_convention(cfg, split):
    X_train, _, y_train, _ = split
    r = cross_validate_model(C.build_baselines(cfg)[C.LINEAR], X_train, y_train, make_outer_cv(cfg))
    for key in ("mae", "rmse", "r2"):
        assert len(r[key]["folds"]) == 5
    assert r["mae"]["mean"] > 0 and r["rmse"]["mean"] >= r["mae"]["mean"]  # errors reported positive


def test_selection_uses_cv_rmse_only():
    results = [
        {"model": "A", "cv": {"rmse": {"mean": 5.0}}},
        {"model": "B", "cv": {"rmse": {"mean": 3.0}}, "train_score": -999},
    ]
    assert select_best(results)["model"] == "B"


def test_tuned_candidate_builds_search(cfg):
    s = C.build_tuned(cfg, C.GB)
    assert s.n_iter == 2 and s.cv.n_splits == 2


def test_end_to_end_training_writes_artifacts(cfg):
    cfg["training"]["tuning"]["enabled"] = True  # exercise nested CV with tiny n_iter
    meta = run_training(cfg)
    assert resolve(cfg, cfg["paths"]["model_path"]).exists()
    saved = json.loads(resolve(cfg, cfg["paths"]["metadata_path"]).read_text())
    assert saved["model_id"] == meta["model_id"]
    assert {"mae", "rmse", "r2"} <= set(saved["holdout_metrics"])
    assert set(saved["features"]) == {"age", "sex", "bmi", "children", "smoker", "region"}
    assert "charges" not in saved["features"]
    assert len(saved["cv_metrics"]) == 6  # 4 baselines + 2 tuned
    reports = resolve(cfg, cfg["paths"]["reports_dir"])
    assert (reports / "model_comparison" / "model_comparison.json").exists()
    assert (reports / "cross_validation" / "cv_results.json").exists()


def test_artifact_round_trip_gives_same_predictions(trained_cfg, synthetic_df):
    path = resolve(trained_cfg, trained_cfg["paths"]["model_path"])
    pipe = joblib.load(path)
    X, _ = split_xy(synthetic_df, trained_cfg)
    np.testing.assert_allclose(pipe.predict(X), joblib.load(path).predict(X))
    assert isinstance(pipe, Pipeline)  # preprocessing is bundled with the estimator


def test_cv_metrics_in_metadata_are_not_training_scores(trained_cfg):
    meta = json.loads(resolve(trained_cfg, trained_cfg["paths"]["metadata_path"]).read_text())
    assert "holdout_metrics" in meta and "cv_metrics" in meta
    assert meta["cv_protocol"]["n_splits"] == 5
