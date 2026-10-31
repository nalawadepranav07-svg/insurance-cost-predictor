# `frontend/` — CoverWise (Streamlit)
`streamlit run frontend/app.py`. Files: `app.py` (entry/navigation), `views.py` (Home, Predict, Result, Insights, About),
`components.py` (reusable UI pieces), `styles.py` (design tokens + CSS), `data.py` (cached model/report loading).
UI never re-implements preprocessing: it calls `src.prediction.Predictor`. If the model is missing/corrupt the app shows a
clear error and disables prediction. All metrics/importances displayed are read from saved reports.
