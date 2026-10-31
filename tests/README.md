# `tests/`
Run: `python -m pytest`. Unit/integration tests use a tiny **synthetic** dataset (`fixtures/synthetic.py`) and temp
directories; `test_app/` drives the Streamlit app headlessly (`streamlit.testing.v1.AppTest`) and skips model-dependent
tests if no trained model exists. Covers: data validation, leakage (scaler statistics come only from the training fold),
unseen categories, all candidates, identical CV folds, artifact round-trip, checksum/schema/corruption handling, invalid
inputs, finite numeric predictions, missing model, app startup and the predict flow.
