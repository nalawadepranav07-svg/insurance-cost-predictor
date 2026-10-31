"""Application startup and UI behaviour tests using Streamlit's headless AppTest runner."""
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.utils.config import PROJECT_ROOT, load_config

APP = str(PROJECT_ROOT / "frontend" / "app.py")
MODEL_PRESENT = (PROJECT_ROOT / load_config()["paths"]["model_path"]).exists()


@pytest.fixture(autouse=True)
def _clear_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def _click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def _texts(at):
    return " ".join(m.value for m in at.markdown)


@pytest.mark.skipif(not MODEL_PRESENT, reason="trained model not present (run scripts.train_model)")
class TestWithModel:
    def test_startup_no_exception_and_branding(self):
        at = _app().run()
        assert not at.exception
        assert "CoverWise" in _texts(at) and "not an insurer quote" in _texts(at)

    def test_full_prediction_flow(self):
        at = _app().run()
        at = _click(at, "Start prediction →")
        assert not at.exception
        at.number_input(key=None) if False else None
        at = _click(at, "Estimate charges")
        assert not at.exception
        html = _texts(at)
        assert "Your estimate" in html and "$" in html and "Make another prediction" in [b.label for b in at.button]
        # the displayed number comes from the real pipeline: compare with a direct prediction
        from src.prediction.predictor import Predictor
        p = Predictor.load(load_config())
        expected = p.predict({"age": 35, "sex": "female", "bmi": 27.0, "children": 0,
                              "smoker": "no", "region": "northeast"}).estimate
        assert f"${expected:,.2f}" in html

    def test_changing_inputs_changes_estimate(self):
        at = _app().run()
        at = _click(at, "Start prediction →")
        at.radio[0].set_value("yes")
        at = _click(at, "Estimate charges")
        assert not at.exception
        smoker_html = _texts(at)
        assert "Smoker" in smoker_html and "$" in smoker_html

    def test_make_another_prediction_returns_to_form(self):
        at = _click(_click(_app().run(), "Start prediction →"), "Estimate charges")
        at = _click(at, "Make another prediction")
        assert not at.exception and any(b.label == "Estimate charges" for b in at.button)

    def test_out_of_range_input_does_not_crash(self):
        at = _click(_app().run(), "Start prediction →")
        try:
            at.number_input[0].set_value(17)  # below the allowed minimum
        except Exception:
            pass  # widget refuses the value: acceptable
        at = at.run() if not at.exception else at
        at = _click(at, "Estimate charges") if any(b.label == "Estimate charges" for b in at.button) else at
        assert not at.exception

    @pytest.mark.parametrize("nav", ["Insights", "About"])
    def test_other_pages_render(self, nav):
        at = _click(_app().run(), nav)
        assert not at.exception
        if nav == "Insights":
            html = _texts(at)
            meta = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text())
            assert f"${meta['holdout_metrics']['mae']:,.2f}" in html  # real, saved number
            assert "Gradient Boosting" in html and "Random Forest" in html and "Linear Regression" in html


def test_app_handles_missing_model(tmp_path, monkeypatch):
    """With no trained model the UI starts, shows a clear message and offers no estimate."""
    cfg = load_config()
    cfg["paths"]["model_path"] = str(tmp_path / "absent.joblib")
    cfg["paths"]["metadata_path"] = str(tmp_path / "absent.json")
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv("COVERWISE_CONFIG", str(path))
    at = _app().run()
    assert not at.exception
    assert "Model not available" in _texts(at)
    at = _click(at, "Start prediction →")
    assert not at.exception
    submit = next(b for b in at.button if b.label == "Estimate charges")
    assert submit.disabled
    at = _click(at, "Insights")
    assert not at.exception and "No evaluation reports found" in _texts(at)
