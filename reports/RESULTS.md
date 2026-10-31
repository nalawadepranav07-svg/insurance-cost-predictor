# CoverWise — evaluation results

Generated from the saved reports. Model ID `6325fb2a11d6`; dataset SHA-256 `505c1cbc2e63…`; random seed 42.

## 5-fold cross-validation (training split, identical shuffled folds)

| Model | MAE (USD) | RMSE (USD) | R² | Selected |
|---|---|---|---|---|
| Linear Regression | 4,270 ± 234 | 6,208 ± 465 | 0.736 ± 0.018 |  |
| Linear Regression (log target) | 4,335 ± 376 | 8,690 ± 1,068 | 0.464 ± 0.192 |  |
| Random Forest | 2,701 ± 352 | 4,860 ± 653 | 0.838 ± 0.029 |  |
| Gradient Boosting | 2,649 ± 304 | 4,677 ± 693 | 0.850 ± 0.031 |  |
| Random Forest (tuned) | 2,603 ± 258 | 4,567 ± 666 | 0.857 ± 0.028 |  |
| Gradient Boosting (tuned) | 2,614 ± 280 | 4,527 ± 702 | 0.859 ± 0.030 | **yes** |

## Final hold-out test (268 unseen rows) — Gradient Boosting (tuned)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| MAE (USD) | 2,418 | 2,023 – 2,826 |
| RMSE (USD) | 4,166 | 3,251 – 4,970 |
| R² | 0.882 | 0.821 – 0.930 |

## Permutation importance (hold-out, RMSE increase in USD)

| Feature | Mean | Std |
|---|---|---|
| smoker | 10,750 | 547 |
| bmi | 3,330 | 352 |
| age | 2,549 | 228 |
| children | 151 | 47 |
| region | 7 | 19 |
| sex | -8 | 6 |

Selected hyper-parameters: `{"model__learning_rate": 0.07853737079666878, "model__max_depth": 3, "model__min_samples_leaf": 3, "model__n_estimators": 50, "model__subsample": 1.0}`
