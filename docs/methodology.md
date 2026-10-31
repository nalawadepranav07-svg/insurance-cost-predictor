# Methodology, limitations and fairness

## Leakage controls
`charges` is split off before anything else; `remainder="drop"` and a config check forbid it as a feature. All learned
preprocessing is inside the `Pipeline`, so each CV fold fits imputers/scalers/encoders on its training part only (tested in
`tests/test_features`). The hold-out set is never used for tuning or selection. Tuning is nested inside CV.

## Preprocessing
Numeric (`age`, `bmi`, `children`): median imputation; standardisation for linear models only. Categorical (`sex`, `smoker`,
`region`): most-frequent imputation + one-hot (binary columns collapsed). Unknown categories encode to zeros but the
prediction layer rejects them explicitly.

## Evaluation protocol
1. 80/20 stratified (by smoker) split, seed 42 → 1,070 train / 268 hold-out.
2. Shuffled 5-fold CV on the training split, identical for all candidates; report mean ± std of MAE, RMSE, R².
3. Select by lowest mean CV RMSE. 4. Evaluate the selected pipeline once on the hold-out; report a 95 % bootstrap CI.
Metrics: MAE and RMSE in USD (lower is better); R² (higher is better, can be negative, not a percentage of "accuracy").
CV metrics estimate generalisation within the training split; hold-out metrics are the final unbiased check.

## Interpreting importance
Permutation importance (RMSE increase when a feature is shuffled on the hold-out set) measures predictive reliance, not
causation. It can be slightly negative for useless features (noise).

## Limitations
- Tiny, US-centric, historical teaching dataset; six features; no health history, plan or provider information.
- Hold-out has 268 rows → wide intervals (see RESULTS.md); some subgroups have few records (flagged when n < 30).
- Heavy-tailed target: large errors for high-cost cases remain.
- Predictions for inputs near the edges of the training range are less reliable; inputs outside it are rejected.
- Not validated by actuaries; not an underwriting or pricing tool.

## Fairness considerations
`sex`, `smoker` and `region` are predictive inputs here only because they exist in the dataset. Observed associations are not
causal, and using such attributes in real insurance pricing may be restricted by law. Subgroup error tables
(`reports/error_analysis/subgroup_errors.csv`) are signals to investigate, **not** a fairness certification.
