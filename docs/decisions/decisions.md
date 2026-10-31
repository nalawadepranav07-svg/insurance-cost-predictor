# Architecture decision records
All dated 2026-10-31. Evidence = measured results in `reports/RESULTS.md`.

| # | Decision | Alternatives | Rationale / trade-off |
|---|---|---|---|
| 1 | Dataset: Kaggle *Medical Cost Personal Dataset*, fetched from a public GitHub mirror with a pinned SHA-256 | Kaggle API (needs credentials) | Reproducible without secrets. Licence must still be verified before redistribution, so the CSV is git-ignored. |
| 2 | Target: raw `charges` in USD | Log target as the main target | Users want dollars. Log target kept as a *candidate*; it scored worse on dollar-scale RMSE. |
| 3 | Missing data: median / most-frequent imputation inside the pipeline | Drop rows | Dataset has no missing values; imputation is defensive for inference and tests. |
| 4 | Outliers/duplicates: investigated, **not removed** (139 high-charge rows, 136 of them smokers; 1 duplicate) | IQR clipping | Removing them would hide real structure and inflate metrics. |
| 5 | `ColumnTransformer`: scaling only for linear models; one-hot with `handle_unknown="ignore"` | Scale everything | Trees are scale-invariant; fewer needless transforms. |
| 6 | CV: shuffled `KFold(5, seed 42)` shared by all models | Stratified/repeated CV | Required identical folds; repeated CV would triple cost on a 1-CPU machine. |
| 7 | Candidates: LR, LR (log target), RF, GB (+ tuned RF/GB). HistGradientBoosting not included | Add HGB | Brief required three; the log-target variant tests the skew question. |
| 8 | Tuning: `RandomizedSearchCV` (20 iterations, 3 inner folds, RMSE) **nested** in the outer CV | Tune on all data | Nested CV avoids optimistic scores. Modest search spaces keep it fast (~2 min on 1 CPU). |
| 9 | Selection: lowest mean CV RMSE; hold-out used once | Select on MAE or hold-out | RMSE is in USD and penalises large misses. The top two models were within one std, so the choice is a convention, not a proven win. |
| 10 | Shipped model is fitted on the training split only | Refit on all rows | Reported hold-out metrics then describe exactly the shipped artifact. Trade-off: 20 % of data unused. |
| 11 | Artifact: joblib + JSON metadata, SHA-256 verified before load | pickle only | joblib can execute code on load; the checksum guards against swapped files. |
| 12 | Frontend: Streamlit, multi-page flow via session state | Gradio; single page | Matches the brief's preference; the multi-page flow matches the project prompt. |
| 13 | Disclaimer wording: "Educational estimate only — not an insurer quote." shown on every result | — | Required by the blueprint. |
