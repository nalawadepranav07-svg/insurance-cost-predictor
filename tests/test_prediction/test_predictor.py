import json
import math

import pytest

from src.prediction.predictor import (DISCLAIMER, ModelUnavailableError, Predictor, ValidationError,
                                      validate_inputs)
from src.utils.config import resolve
from tests.conftest import VALID_INPUT


# ------------------------------------------------------------------ validation
def test_valid_input_passes(cfg):
    clean = validate_inputs(VALID_INPUT, cfg)
    assert clean["age"] == 35 and isinstance(clean["age"], int) and clean["bmi"] == 27.5


def test_normalises_case_and_numeric_strings(cfg):
    clean = validate_inputs({**VALID_INPUT, "sex": " Female ", "age": "40"}, cfg)
    assert clean["sex"] == "female" and clean["age"] == 40


@pytest.mark.parametrize("field", ["age", "sex", "bmi", "children", "smoker", "region"])
def test_each_missing_field_rejected(cfg, field):
    data = {k: v for k, v in VALID_INPUT.items() if k != field}
    with pytest.raises(ValidationError) as e:
        validate_inputs(data, cfg)
    assert field in e.value.errors


@pytest.mark.parametrize("field,value", [
    ("age", 17), ("age", 65), ("age", -3), ("age", 30.5),
    ("bmi", 10), ("bmi", 80), ("bmi", float("nan")), ("bmi", float("inf")), ("bmi", "abc"),
    ("children", -1), ("children", 9), ("children", 1.5),
    ("sex", "other"), ("smoker", "maybe"), ("region", "atlantis"),
    ("age", True), ("age", None), ("region", ""),
])
def test_invalid_values_rejected(cfg, field, value):
    with pytest.raises(ValidationError) as e:
        validate_inputs({**VALID_INPUT, field: value}, cfg)
    assert field in e.value.errors


def test_target_and_unknown_fields_not_accepted(cfg):
    with pytest.raises(ValidationError) as e:
        validate_inputs({**VALID_INPUT, "charges": 12000, "income": 5}, cfg)
    assert "charges" in e.value.errors and "income" in e.value.errors


def test_all_errors_reported_together(cfg):
    with pytest.raises(ValidationError) as e:
        validate_inputs({"age": 5, "bmi": 99}, cfg)
    assert len(e.value.errors) >= 5


# ------------------------------------------------------------------ model loading
def test_missing_model_raises_clear_error(cfg):
    with pytest.raises(ModelUnavailableError, match="train_model"):
        Predictor.load(cfg)


def test_corrupt_model_detected_by_checksum(trained_cfg):
    resolve(trained_cfg, trained_cfg["paths"]["model_path"]).write_bytes(b"not a model")
    with pytest.raises(ModelUnavailableError, match="checksum"):
        Predictor.load(trained_cfg)


def test_unreadable_metadata(trained_cfg):
    resolve(trained_cfg, trained_cfg["paths"]["metadata_path"]).write_text("{broken")
    with pytest.raises(ModelUnavailableError, match="metadata"):
        Predictor.load(trained_cfg)


def test_schema_mismatch_detected(trained_cfg):
    path = resolve(trained_cfg, trained_cfg["paths"]["metadata_path"])
    meta = json.loads(path.read_text())
    meta["features"] = meta["features"][:-1]
    path.write_text(json.dumps(meta))
    with pytest.raises(ModelUnavailableError, match="schema"):
        Predictor.load(trained_cfg)


# ------------------------------------------------------------------ inference
def test_prediction_is_finite_numeric_and_labelled(trained_cfg):
    result = Predictor.load(trained_cfg).predict(VALID_INPUT)
    assert isinstance(result.estimate, float) and math.isfinite(result.estimate) and result.estimate >= 0
    assert "not an insurer quote" in result.disclaimer and result.disclaimer == DISCLAIMER
    assert "USD" in result.target_unit and result.model_id


def test_all_six_features_influence_the_saved_pipeline_inputs(trained_cfg):
    p = Predictor.load(trained_cfg)
    base = p.predict(VALID_INPUT).estimate
    assert p.predict({**VALID_INPUT, "smoker": "yes"}).estimate != base
    assert p.predict({**VALID_INPUT, "age": 60}).estimate != base
    for k, v in (("sex", "male"), ("bmi", 40.0), ("children", 4), ("region", "southeast")):
        p.predict({**VALID_INPUT, k: v})  # must not raise


def test_prediction_uses_saved_pipeline(trained_cfg):
    import pandas as pd
    p = Predictor.load(trained_cfg)
    direct = float(p.pipeline.predict(pd.DataFrame([VALID_INPUT]))[0])
    assert p.predict(VALID_INPUT).estimate == pytest.approx(max(direct, 0.0))


def test_invalid_input_never_reaches_model(trained_cfg):
    p = Predictor.load(trained_cfg)
    with pytest.raises(ValidationError):
        p.predict({**VALID_INPUT, "age": 200})
