import hashlib
import json

import pandas as pd
import pytest

from src.data.loader import (DatasetError, load_raw, sha256_of, split_xy, train_test_holdout,
                             validate_dataframe)
from src.utils.config import PROJECT_ROOT, load_config, resolve


def test_valid_synthetic_loads(cfg):
    df = load_raw(cfg)
    assert len(df) == 300 and "charges" in df.columns


def test_missing_file_gives_actionable_error(cfg, tmp_path):
    cfg["dataset"]["raw_path"] = str(tmp_path / "nope.csv")
    with pytest.raises(DatasetError, match="download_data"):
        load_raw(cfg)


def test_empty_file(cfg, tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    cfg["dataset"]["raw_path"] = str(p)
    with pytest.raises(DatasetError):
        load_raw(cfg)


def test_missing_required_column(cfg, synthetic_df):
    with pytest.raises(DatasetError, match="missing required columns"):
        validate_dataframe(synthetic_df.drop(columns=["bmi"]), cfg)


def test_non_numeric_column_flagged(cfg, synthetic_df):
    bad = synthetic_df.copy()
    bad["age"] = bad["age"].astype(str).where(bad.index != 0, "abc")
    assert not validate_dataframe(bad, cfg).ok


def test_unexpected_category_flagged(cfg, synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "region"] = "mars"
    report = validate_dataframe(bad, cfg)
    assert any("region" in p for p in report.problems)


def test_out_of_range_and_target_validation(cfg, synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "age"] = 5
    bad.loc[1, "charges"] = -10
    problems = " ".join(validate_dataframe(bad, cfg).problems)
    assert "age" in problems and "non-positive" in problems


def test_duplicates_and_missing_are_counted_not_dropped(cfg, synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    df.loc[3, "bmi"] = None
    report = validate_dataframe(df, cfg)
    assert report.duplicate_rows == 1 and report.missing_per_column["bmi"] == 1
    assert report.rows == len(df)  # nothing removed


def test_target_never_in_features(cfg, synthetic_df):
    X, y = split_xy(synthetic_df, cfg)
    assert "charges" not in X.columns and list(X.columns) == ["age", "bmi", "children", "sex", "smoker", "region"]
    assert len(y) == len(X)


def test_split_is_deterministic_and_disjoint(cfg, synthetic_df):
    X, y = split_xy(synthetic_df, cfg)
    a = train_test_holdout(X, y, cfg)
    b = train_test_holdout(X, y, cfg)
    assert list(a[0].index) == list(b[0].index)
    assert set(a[0].index).isdisjoint(a[1].index)
    assert len(a[1]) == pytest.approx(0.2 * len(X), abs=2)


def test_raw_file_not_modified_by_loading(cfg):
    path = resolve(cfg, cfg["dataset"]["raw_path"])
    before = sha256_of(path)
    load_raw(cfg)
    assert sha256_of(path) == before


def test_checksum_mismatch_detected(cfg):
    cfg["dataset"]["expected_sha256"] = "0" * 64
    with pytest.raises(DatasetError, match="checksum"):
        load_raw(cfg)


def test_real_dataset_if_present():
    real = load_config()
    path = PROJECT_ROOT / real["dataset"]["raw_path"]
    if not path.exists():
        pytest.skip("real dataset not downloaded")
    df = load_raw(real)
    assert df.shape == (1338, 7)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == real["dataset"]["expected_sha256"]
    manifest = json.loads((PROJECT_ROOT / real["dataset"]["manifest_path"]).read_text())
    assert manifest["rows"] == 1338
