# CoverWise — Insurance Cost Predictor

CoverWise is an **educational machine-learning project**: it estimates historical annual medical insurance charges (USD)
from six personal attributes using the public *Medical Cost Personal Dataset*, compares regression models with 5-fold
cross-validation, and serves the selected model through a Streamlit app.

> **Educational estimate only — not an insurer quote, premium offer or medical advice.** The model reflects patterns in a
> small historical dataset and cannot predict any individual's real costs.

## Quick start

```bash
# 1. Install (Python 3.11+)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # add requirements-dev.txt for tests/notebooks

# 2. Get the data (validated + checksummed)
python -m scripts.download_data

# 3. Train, compare models, evaluate, save the best pipeline (~2 min on one CPU core)
python -m scripts.train_model --overwrite

# 4. Launch the app
streamlit run frontend/app.py

# 5. Run the tests
pip install -r requirements-dev.txt
python -m pytest
```
Faster, untuned training: `python -m scripts.train_model --no-tuning --overwrite`. Other commands: see
[`scripts/README.md`](scripts/README.md). Tested with Python 3.12, scikit-learn 1.8, pandas 3.0, Streamlit 1.66.

## Results (real outputs from `reports/`, seed 42)

### 5-fold cross-validation (training split, identical shuffled folds)

| Model | MAE (USD) | RMSE (USD) | R² | Selected |
|---|---|---|---|---|
| Linear Regression | 4,270 ± 234 | 6,208 ± 465 | 0.736 ± 0.018 |  |
| Linear Regression (log target) | 4,335 ± 376 | 8,690 ± 1,068 | 0.464 ± 0.192 |  |
| Random Forest | 2,701 ± 352 | 4,860 ± 653 | 0.838 ± 0.029 |  |
| Gradient Boosting | 2,649 ± 304 | 4,677 ± 693 | 0.850 ± 0.031 |  |
| Random Forest (tuned) | 2,603 ± 258 | 4,567 ± 666 | 0.857 ± 0.028 |  |
| Gradient Boosting (tuned) | 2,614 ± 280 | 4,527 ± 702 | 0.859 ± 0.030 | **yes** |

### Final hold-out test (268 unseen rows) — Gradient Boosting (tuned)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| MAE (USD) | 2,418 | 2,023 – 2,826 |
| RMSE (USD) | 4,166 | 3,251 – 4,970 |
| R² | 0.882 | 0.821 – 0.930 |

### Permutation importance (hold-out, RMSE increase in USD)

| Feature | Mean | Std |
|---|---|---|
| smoker | 10,750 | 547 |
| bmi | 3,330 | 352 |
| age | 2,549 | 228 |
| children | 151 | 47 |
| region | 7 | 19 |
| sex | -8 | 6 |

Selected hyper-parameters: `{"model__learning_rate": 0.07853737079666878, "model__max_depth": 3, "model__min_samples_leaf": 3, "model__n_estimators": 50, "model__subsample": 1.0}`

How to read this: CV rows are mean ± standard deviation over the same 5 shuffled folds of the 1,070-row training split
(tuned models use nested CV). The hold-out table is the final test on 268 rows never used for selection or tuning.
The top two models are within one standard deviation of each other, so Gradient Boosting (tuned) was chosen by the
pre-declared rule (lowest mean CV RMSE), not because it is proven superior. R² is not an accuracy percentage.

## App screens
Home (value proposition, CTA, model at a glance) → Predict (validated form) → Result (estimate in USD, submitted inputs,
typical error, disclaimer, "make another prediction") · Insights (hold-out metrics with CIs, CV comparison table, permutation
importance, predicted-vs-actual, subgroup errors) · About (dataset, methodology, disclaimer, limitations).
All numbers shown come from the saved model and reports. If no model exists the app says so and disables prediction.

## EDA highlights (computed in `reports/eda_summary.json`, narrated in the notebook)
- 1,338 rows × 7 columns; no missing values; 1 duplicate row (kept).
- Charges are right-skewed (skewness 1.52; -0.09 after log); 139 rows lie above the 1.5×IQR fence, 136 of them smokers — investigated, not removed.
- Mean charges: smokers ≈ $32,050 vs non-smokers ≈ $8,434. Smokers with BMI ≥ 30 have far higher charges than smokers below 30; non-smokers barely change — an interaction linear models cannot capture.
- Pearson correlation with charges: age 0.299, BMI 0.198, children 0.068.
- Region and sex differences are small relative to within-group spread. These are associations, not causes.

## Project structure
```
configs/default.json      all settings (dataset, schema, seed, CV, tuning, paths)
data/                     raw (git-ignored CSV + MANIFEST.json) / interim / processed / external
src/                      data, features, models, evaluation, prediction, utils
scripts/                  download_data, run_eda, train_model, export_report, build_notebook
frontend/                 Streamlit app (app.py, views.py, components.py, styles.py, data.py, design/)
notebooks/                01_eda_and_model_evaluation.ipynb (executed)
reports/                  RESULTS.md, EDA summary, CV + comparison + error-analysis reports, figures/
models/                   trained pipeline + metadata (generated; binary git-ignored)
tests/                    data, features, models, prediction and app tests (synthetic fixtures)
docs/                     architecture, decisions (ADRs), methodology, GitHub setup, original blueprint
```

## Methodology in brief
Stratified 80/20 split → identical shuffled 5-fold CV for Linear Regression, a log-target Linear Regression, Random Forest
and Gradient Boosting (plus tuned RF/GB via randomised search nested inside CV) → select by lowest mean CV RMSE → fit on the
training split → single hold-out evaluation with bootstrap CIs → persist the **complete fitted pipeline** with metadata.
Preprocessing lives inside the pipeline (no leakage; identical at inference). Details: [`docs/methodology.md`](docs/methodology.md),
[`docs/architecture/architecture.md`](docs/architecture/architecture.md), [`docs/decisions/decisions.md`](docs/decisions/decisions.md).

## Dataset
Kaggle "Medical Cost Personal Datasets" (`mirichoi0218/insurance`, derived from Brett Lantz, *Machine Learning with R*),
fetched from a public GitHub mirror with a pinned SHA-256. Licence: Kaggle lists ODbL/DbCL — verify before redistributing
(the CSV is therefore git-ignored). See [`data/README.md`](data/README.md).

## Testing
`python -m pytest` runs unit and integration tests on a tiny **synthetic** dataset plus headless Streamlit tests: data
validation, leakage (preprocessing statistics come from the training fold only), unseen categories, every candidate model,
identical CV folds, artifact round-trip, checksum/schema/corruption handling, invalid inputs, finite numeric predictions,
missing model, app startup and the full predict flow.

## Limitations
Small, US-centric teaching dataset with six features; wide uncertainty (hold-out CI above); large errors on high-cost cases;
some subgroups too small to judge; sex/smoker/region are used only because they exist in the data and are not causal; no
actuarial validation; not for pricing, underwriting or decisions about care. No Figma file was created (no integration was
available); the design brief was implemented directly in Streamlit — see `frontend/design/IMPLEMENTATION_NOTES.md`.

## Future improvements
Repeated CV and more candidates (e.g. HistGradientBoosting), prediction intervals (quantile or conformal), SHAP-based local
explanations, larger/more representative data, calibration by cost band, a browser-level UI test suite, CI (GitHub Actions),
and a `LICENSE`.

## Publishing to GitHub
See [`docs/github_setup.md`](docs/github_setup.md).
