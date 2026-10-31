# Original blueprint prompts (from the supplied ZIP)

Preserved verbatim for traceability. The folder READMEs now describe what was actually built.


---

## `README.md`

# CoverWise — Insurance Cost Predictor

**Purpose:** Build a transparent regression application using the Medical Cost Personal Dataset (commonly containing `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`). Explore and preprocess the dataset, compare at least three regression models, evaluate with MAE, RMSE and R² using 5-fold cross-validation, and deploy the selected model in a Streamlit or Gradio app.

> **Important scope correction:** This dataset predicts historical medical insurance charges from a small set of demographic/behavioral features. It does not calculate an actual insurer's quote, establish medical need, or determine an individual's true future healthcare cost. The UI and report must label output as an educational estimate from a dataset, not a binding insurance premium.

## Repository status
This is a code-free implementation blueprint. It contains no Python application code, trained model, or frontend source. Every directory has a README with an actionable implementation prompt.

## Recommended stack
- Python 3.11+, pandas, NumPy, scikit-learn, matplotlib.
- Streamlit for a fast, single-page demo unless there is a concrete reason to choose Gradio.
- scikit-learn `Pipeline` and `ColumnTransformer` to prevent preprocessing leakage.
- Candidate models: Linear Regression (baseline), Random Forest Regressor, Gradient Boosting Regressor. Optionally compare HistGradientBoosting as a fourth candidate.
- Metrics: MAE, RMSE, R²; use identical shuffled 5-fold splits with a fixed random seed.
- Model persistence: joblib with environment/model metadata; load only trusted local artifacts.

## Correct ML methodology
1. Inspect schema, types, duplicates, missingness, ranges, target distribution and categorical balance.
2. Split features/target; keep `charges` out of preprocessing and never use it as a feature.
3. Put numeric imputation/scaling and categorical imputation/one-hot encoding inside a `ColumnTransformer` pipeline.
4. Use cross-validation on the full pipeline so every fold fits encoders/scalers only on its training split.
5. Compare all models on the same folds; report mean and standard deviation of MAE, RMSE and R².
6. Select based on validation metrics and error behavior, not training score. Keep a final holdout set only if enough data and a clear protocol are available; do not repeatedly tune against it.
7. Inspect residuals and subgroup error patterns. Discuss limitations and potential fairness concerns, especially around `sex`, `smoker`, and `region`.
8. Save the complete fitted pipeline, not just the estimator, and record feature schema/version and package versions.

## Recommended project structure
| Folder | Responsibility |
|---|---|
| `data/` | Raw source dataset (not committed if licence/privacy uncertain), interim/cleaned data, documentation |
| `notebooks/` | EDA and model experiments; promote final logic to `src/` |
| `src/data/` | Loading, schema checks, validation and splits |
| `src/features/` | Column definitions and leak-safe preprocessing |
| `src/models/` | Candidate estimators, training and persistence |
| `src/evaluation/` | CV metrics, residual and subgroup analysis |
| `src/prediction/` | Input validation and inference interface |
| `src/utils/` | Config, logging and reproducibility helpers |
| `frontend/` | Prediction app, explanation and UI design brief |
| `tests/` | Data, feature, model and prediction tests |
| `configs/` | Reproducible experiment settings; no secrets |
| `docs/` | Methodology, architecture, limitations and decisions |
| `reports/` | CV, model comparison, residual and error analyses |
| `models/` | Locally generated trusted model artefacts; exclude from git unless explicitly required |
| `scripts/` | Explicit training/evaluation commands |

## Definition of done
- Dataset schema and provenance documented; target leakage prevented.
- EDA covers numeric distributions, categorical balance, target skew, missing values, outliers and correlations without overclaiming causality.
- Preprocessing is fitted within CV folds through a scikit-learn pipeline.
- At least three models are compared on the same 5 folds and reported with mean ± standard deviation for MAE, RMSE and R².
- Residual/error analysis explains where predictions are unreliable; include subgroup checks and ethical limitations.
- Best pipeline is persisted with schema/version metadata and used by the app.
- UI validates inputs, shows a predicted charge in dataset currency/units, gives a plain-language disclaimer, and does not imply insurer approval or quote validity.
- Reproducible setup, training, evaluation and app instructions are documented.

## Business angle (optional, clearly separated from the assignment)
A credible extension is a B2B analytics dashboard for brokers or health-benefit teams to explore portfolio-level cost drivers using aggregated, authorized data. Do not pitch this student dataset model as production underwriting. A real commercial product would require actuarial validation, larger representative datasets, data rights, privacy/security controls, fairness review, model governance, and regulatory/legal review. Validate willingness to pay with interviews before building a SaaS wrapper.


---

## `configs/README.md`

# Experiment configuration

**Build prompt:** Define reproducible settings for data path, schema, target name/units, CV split count (5), shuffle and seed, candidate models, parameter grids, scoring metrics, output paths, and selected model. Store examples only; no credentials. Validate settings before training and serialize the exact configuration with every report and model artifact.


---

## `data/README.md`

# Data lifecycle

**Build prompt:** Document the dataset source, licence, collection context, row count, schema, target units/currency and any known sampling limitations. Set up separate raw/interim/processed/external zones. Do not silently modify the raw file. Validate expected columns and types, duplicates, missing values, impossible ranges and target positivity. Make data loading deterministic and report a clear error when the expected dataset is absent. Never include personal or identifiable health data. Add provenance and a data dictionary before analysis.


---

## `data/external/README.md`

# External data

**Build prompt:** This directory is reserved for supplementary datasets only if their purpose, licence, provenance and compatibility are documented. Do not mix external datasets into the primary benchmark without a predeclared evaluation plan. If no authorized external dataset is used, keep this folder empty.


---

## `data/interim/README.md`

# Interim data

**Build prompt:** Define reproducible intermediate data products for validated rows, duplicate review, and split manifests. Keep transformations traceable back to raw records. Document dropped/changed rows and reasons. Avoid saving derived sensitive information or embedding target-derived features. Prefer generating intermediate data on demand unless materialization materially improves reproducibility.


---

## `data/processed/README.md`

# Processed data

**Build prompt:** Store only clearly versioned, validated datasets or split manifests created by the pipeline. Ensure all learned preprocessing (imputation, scaling, encoding) is fit inside model pipelines and CV folds, not pre-applied to the entire dataset. Include schema/version and source checksum metadata. Never commit processed data if licence or privacy does not permit.


---

## `data/raw/README.md`

# Raw dataset

**Build prompt:** Store the original Medical Cost Personal Dataset here only when its source and licence permit. Keep it immutable and exclude it from version control by default if redistribution rights are unclear. Record exact source URL/name, retrieval date, licence, checksum, row count, and schema in a separate manifest. Do not put modified files in this folder. Never replace missing source data with synthetic values without labelling them.


---

## `docs/README.md`

# Project documentation

**Build prompt:** Document setup, dataset provenance, schema, ML methodology, leakage controls, evaluation protocol, model limitations, target units, UI disclaimer, and reproduction instructions. Explain why historical charges are not equivalent to an actual insurer quote. Do not present sex/smoking/region as causal factors; describe observed predictive associations only.


---

## `docs/architecture/README.md`

# Architecture

**Build prompt:** Document a reproducible pipeline: raw dataset → validation → exploratory audit → split → fold-local preprocessing → estimator → cross-validation metrics → selected full pipeline artifact → input validation → prediction. Include a flow diagram and explain how the exact same preprocessing is reused at inference. Identify artifact versioning and failure behavior.


---

## `docs/decisions/README.md`

# Architecture decision records

**Build prompt:** Record choices for dataset source/licence, target units, missing/outlier handling, column transformer, scaling, CV split strategy, candidate estimators, tuning strategy, model selection, artifact format, frontend framework, and disclaimer wording. Include context, alternatives, selected decision, trade-offs, evidence and date.


---

## `frontend/README.md`

# Frontend — CoverWise

**Build prompt:** Build an accessible Streamlit app (preferred for this assignment) following `design/README.md`. Create a single polished experience with a clearly labelled educational cost estimate, validated input controls, prediction action, result card with target units, model explanation and limitations, and a compact model performance section populated from actual CV reports. Handle missing/corrupt model artifacts gracefully. Never imply a guaranteed insurer quote or financial/medical advice. Do not hard-code fake model metrics. Make the app responsive within Streamlit constraints and separate UI from prediction logic. No code is included in this starter.


---

## `frontend/assets/README.md`

# Frontend assets

**Build prompt:** Add approved logo/icon assets only. Prefer a simple wordmark and consistent line icons; avoid stock medical imagery, insurer logos, trust badges or decorative illustrations implying certification. Document licence and source for any external asset. Ensure SVGs have accessible names when meaningful.


---

## `frontend/design/README.md`

# Figma UI design brief — CoverWise

## Figma handoff status
This is a Figma-ready design specification, not an editable Figma canvas. The connected Figma account currently has a **View** seat on its team; canvas creation/editing requires an editor seat or another editable destination file.

## Design objective
Design **CoverWise**, an educational insurance-cost prediction dashboard. It should feel reliable, clear, data-literate and approachable—not like an insurer's quote checkout page. The most important UX task is making the estimate and its limitations unmissable.

## Visual direction
- Reference desktop canvas: 1440px; adapt to tablet/mobile.
- Modern financial/analytics product aesthetic: deep navy/ink, clean white and cool light-grey surfaces, one restrained cobalt/blue accent, muted green only for positive/validated states, amber for uncertainty/disclaimer.
- Clear numerical typography for currency values and metrics; accessible body text.
- 8px spacing system, 12–16px radius, subtle borders, restrained shadows, charts with labelled axes and no decorative effects.
- Avoid insurance-company branding, fake trust seals, medical imagery, and “guaranteed savings” language.

## Required frames
1. **Landing / predictor:** product title, brief explanation, compact disclaimer, left input form, right preview/result placeholder.
2. **Input form focus:** age, sex (match dataset category labels exactly), BMI, number of children, smoker status, region. Each field has helper text, bounds and validation state. Keep all fields visible without overwhelming the user.
3. **Prediction result:** prominent estimated charges with currency/unit, “educational estimate—not an insurer quote” disclaimer, model version and short interpretation of selected inputs.
4. **Model performance:** comparison table for Linear Regression, Random Forest and Gradient Boosting; columns MAE, RMSE, R², reported as mean ± standard deviation from 5-fold CV. Use placeholders labelled “populate from experiment” until actual values exist—never fabricate results.
5. **Model explanation / limitations:** feature contribution/importance explanation only if supported by the selected model and method; otherwise show a transparent “what the model uses” feature summary. Include dataset and fairness limitations.
6. **Validation/error states:** invalid BMI/age, missing model, prediction service failure, unsupported category, and empty/untrained state.
7. **About methodology:** dataset fields, preprocessing overview, 5-fold CV, metrics definitions, and disclaimer.

## Main screen layout
- Header: CoverWise wordmark, “Medical cost estimator” subtitle, Methodology nav link.
- Main content grid: left panel “Your details” (form); right panel “Estimate” (initially neutral state).
- Result panel: large currency value; concise explanatory sentence; input summary; model label; prominent disclaimer.
- Below fold: “How the model was evaluated” metrics row and model-comparison table, with expandable methodology and limitations.

## Component inventory
Form field, select, segmented control, number stepper, primary button, estimate card, metric card, comparison table, disclaimer banner, info tooltip, validation message, feature summary row, model status pill, error/empty state, chart legend.

## Interaction rules
- Inputs reflect the dataset schema and accepted categories, not arbitrary free text.
- Validate on submit and give field-level errors; do not auto-correct silently.
- Do not show an estimate until a trained artifact has loaded successfully.
- Display target units/currency explicitly and consistently with the dataset source.
- Explain that the prediction is an educational benchmark estimate, not an official premium or quote.
- Model comparison values come from actual reports and show fold variation, not just one cherry-picked score.
- Any feature importance display must name its method and avoid causal claims.

## Figma file organization
Pages: `00 Foundations`, `01 Components`, `02 Predictor`, `03 Results & Methodology`, `04 Error States`, `05 Responsive`.
Use Auto Layout, local design tokens, reusable form components and variants. Name layers semantically (`Form/Field/Age`, `Result/EstimateCard`, `Metrics/ModelComparison`).

## Quality checklist
Can a user understand the estimate is not a real quote? Are input bounds visible? Are metrics labelled with units and CV variation? Are results empty until a valid model is loaded? Does the design avoid implying causation or guaranteed savings? Is mobile form entry usable?

## Prompt to give Figma AI / designer
“Create an editable, componentized web app UI for CoverWise, an educational medical insurance cost predictor built from a public tabular dataset. The app accepts age, sex, BMI, number of children, smoker status, and region and returns a model-estimated historical charge. Design a polished 1440px desktop predictor page with an input form on the left and a result card on the right, plus frames for validation, prediction results, model comparison, methodology, limitations and mobile responsive layout. Use a trustworthy modern analytics aesthetic: navy/ink text, white and cool-grey surfaces, restrained cobalt accent, subtle green for validated state and amber for disclaimer/uncertainty. Keep numerical values legible, forms uncluttered, labels and units explicit, axes labelled, and spacing consistent. Use Auto Layout and reusable variants. Make the disclaimer ‘Educational estimate only — not an insurer quote’ prominent. Do not add fake metrics, fake savings, insurer logos, trust badges or causal claims.”


---

## `models/README.md`

# Local model artifacts

**Build prompt:** Store only trusted artifacts created by the project's training workflow (e.g., a serialized full preprocessing+estimator pipeline and JSON metadata). Exclude model binaries from version control by default; document how to reproduce them. Validate metadata before loading. Never load arbitrary uploaded pickle/joblib files. Record the dataset checksum, feature schema, target units, CV metrics, library versions, training timestamp and artifact hash. Do not commit artifacts trained from private/unauthorized data.


---

## `notebooks/README.md`

# Analysis notebooks

**Build prompt:** Create notebooks for (1) data quality and schema audit, (2) exploratory analysis and target skew, (3) baseline regression, (4) model comparison, and (5) residual/subgroup error analysis. Use a fixed seed, explain each plot, and avoid causal language. Keep notebooks exploratory; move stable reusable logic into `src/`. Do not commit huge outputs, local absolute paths, or private data. Record experiment settings and findings in reports.


---

## `reports/README.md`

# Reports

**Build prompt:** Store reproducible EDA, comparison, CV, residual and limitations reports. Tie each report to a data checksum, configuration, random seed, package versions and model artifact ID. Do not populate sample results until experiments have run. Clearly separate findings from recommendations.


---

## `reports/cross_validation/README.md`

# Cross-validation records

**Build prompt:** Record the exact 5-fold splitter settings, seed, fold assignments or reproducible splitter configuration, per-fold metrics, and preprocessing pipeline behavior. Ensure all preprocessing is trained only within each training fold. If hyperparameter tuning is performed, explain whether nested CV or a separate validation protocol was used.


---

## `reports/error_analysis/README.md`

# Error and fairness analysis

**Build prompt:** Examine residuals, large absolute errors, systematic under/over-prediction, heteroscedasticity, influential points and subgroup MAE/RMSE/R² where sample sizes permit. Include sample counts and uncertainty; warn when groups are small. Treat demographic attributes carefully and explain that benchmark disparities are not a fairness certification. List practical limits and data changes needed before any real-world use.


---

## `reports/model_comparison/README.md`

# Model comparison

**Build prompt:** Compare Linear Regression, Random Forest and Gradient Boosting on the same 5-fold CV splits. Report mean ± standard deviation for MAE, RMSE and R², fold-level metrics, training/inference time if measured, and model complexity. Explain metric trade-offs. Select the winner based on validation evidence and error analysis, not on a single score. Leave values blank until real results exist.


---

## `scripts/README.md`

# Training and evaluation commands

**Build prompt:** Add CLI commands to validate data, run EDA summaries, train/compare models, evaluate the saved pipeline, and launch the app. Commands must have help text, explicit inputs/outputs, deterministic configuration, meaningful exit codes, and safe overwrite behavior. They should call package modules rather than duplicate notebooks. Do not silently train on an undocumented file.


---

## `src/README.md`

# Source package

**Build prompt:** Implement reusable, typed modules organized by data, features, models, evaluation, prediction and utilities. Keep training and inference paths consistent by persisting the complete scikit-learn pipeline. Centralize schema and configuration. Avoid data leakage, hidden state, notebook-only logic and duplicate preprocessing implementations. Add tests before integrating the frontend.


---

## `src/data/README.md`

# Data loading and validation

**Build prompt:** Implement dataset loading with explicit path/config, schema validation, type normalization, duplicate and missingness reports, sensible range checks, target validation, and deterministic train/test or CV split helpers. Preserve raw data and emit a concise audit report. Never drop rows or clip outliers without measuring and documenting the impact. Do not inspect `charges` to create predictors.


---

## `src/evaluation/README.md`

# Model evaluation

**Build prompt:** Evaluate each candidate on identical shuffled 5-fold splits. Report fold-level and mean ± standard deviation for MAE, RMSE, and R². Use a scorer setup whose metric direction is correct (lower MAE/RMSE is better; higher R² is better). Explain R² can be negative and is not a percentage of accuracy. Add residual plots, predicted-vs-actual plots, error distribution, and subgroup error analysis by available categories with sample counts and uncertainty. Treat subgroup disparities as signals requiring investigation, not proof of discrimination or causation. Never report training metrics as generalization performance.


---

## `src/features/README.md`

# Feature engineering and preprocessing

**Build prompt:** Define numeric and categorical feature columns from the dataset schema. Build a scikit-learn `ColumnTransformer` inside a `Pipeline`: numeric imputation (and scaling only where useful), categorical imputation and `OneHotEncoder(handle_unknown='ignore')`. Fit all preprocessing inside each CV fold. Ensure inference accepts the original raw feature columns and uses the exact same fitted pipeline. Do not encode the target or fit transformations before cross-validation. Document why every transformation exists.


---

## `src/models/README.md`

# Model training and persistence

**Build prompt:** Compare at least LinearRegression, RandomForestRegressor and GradientBoostingRegressor using consistent folds and one full preprocessing pipeline per estimator. Use documented, modest hyperparameter grids if tuning is justified; keep CV nested or use a separate validation strategy when tuning to avoid optimistic estimates. Record random seeds and versions. Persist the full selected pipeline (preprocessing + estimator) with metadata: feature schema, target units, dataset checksum, training timestamp, CV protocol, metrics, package versions and model ID. Do not claim the best algorithm before measured results exist.


---

## `src/prediction/README.md`

# Inference and input validation

**Build prompt:** Define a stable prediction interface using the original feature schema: age, sex, BMI, number of children, smoking status, and region (confirm exact column names against the dataset). Validate bounds, types, categories and missing fields; reject nonsensical values with clear messages. Load only a trusted local model artifact and verify its schema/model metadata before predicting. Return prediction, model version, target unit and an explicit educational-estimate disclaimer. Never claim the output is an insurer quote, premium offer, medical estimate, or guarantee.


---

## `src/utils/README.md`

# Utilities

**Build prompt:** Implement configuration, fixed random-seed utilities, structured logging, metric formatting, path helpers, and environment/version reporting. Do not log user-entered personal details. Redact file paths where appropriate. Keep feature logic in `src/features/` and metrics in `src/evaluation/`.


---

## `tests/README.md`

# Test strategy

**Build prompt:** Add unit and integration tests for data validation, preprocessing leakage prevention, unknown categories, pipeline fit/predict, metric calculation, artifact metadata, and frontend input constraints. Use tiny synthetic fixtures for deterministic tests. Verify that no preprocessing is fit on validation folds. Test impossible age/BMI/children inputs and unknown categorical values. Do not use live personal data in tests.


---

## `tests/fixtures/README.md`

# Synthetic test fixtures

**Build prompt:** Create tiny synthetic datasets with representative numeric/categorical values, missing values, unknown category cases, duplicate rows, and skewed target values. Label all records synthetic. Fixtures should test software behavior, not approximate real population statistics. Include explicit expected schema and units.


---

## `tests/test_data/README.md`

# Data tests

**Build prompt:** Test required columns, extra columns policy, numeric parsing, invalid types, missing values, duplicate counts, impossible ranges, target validation, empty files, and deterministic splits. Ensure validation errors are actionable and source data remains unchanged.


---

## `tests/test_features/README.md`

# Feature pipeline tests

**Build prompt:** Verify numeric and categorical columns are routed correctly; one-hot encoding handles unseen categories; fitted transforms are learned only from training folds; output dimensions stay consistent; and inference accepts raw features in any column order only when the interface explicitly aligns by name. Check that target `charges` is excluded from feature columns.


---

## `tests/test_models/README.md`

# Model tests

**Build prompt:** Test each candidate model can fit/predict through the complete pipeline, random seeds produce repeatable results within expected tolerance, and model comparison uses identical splits. Test artifact save/load round-trip and metadata/schema mismatch handling. Avoid asserting a model wins based on a fixed metric from a small sample.


---

## `tests/test_prediction/README.md`

# Prediction tests

**Build prompt:** Test valid inputs, missing fields, out-of-range age/BMI/children, invalid smoking/sex/region categories, model artifact missing/corrupt, schema mismatch, and prediction output units/disclaimer. Assert no hidden target fields are accepted as predictors.


---

## `frontend/design/FIGMA_PROMPT.md`

# Copy-ready Figma prompt — CoverWise

Design an editable, componentized desktop-first web app called CoverWise: an educational medical insurance cost predictor using age, sex, BMI, children, smoker status and region to estimate historical charges from a tabular dataset. It is not an actual insurer quote or premium offer.

Build a 1440px predictor dashboard with a clean header, left “Your details” input panel and right “Estimate” panel. Inputs must match the dataset's actual categories and include sensible validation/helper text. Before prediction, show a neutral placeholder state. After prediction, show a prominent estimated charge with explicit currency/target unit, a compact summary of submitted inputs, model version, and highly visible text: “Educational estimate only — not an insurer quote.” Add sections for model performance, comparison of Linear Regression / Random Forest / Gradient Boosting using MAE, RMSE and R² from 5-fold CV (mean ± standard deviation; mark all values as placeholders until supplied), methodology, dataset limitations, and fairness considerations. Include error states for invalid input, unavailable model artifact, and prediction failure.

Visual direction: trustworthy modern analytics tool, navy/ink text, white and cool-grey surfaces, restrained cobalt accents, muted green for valid input, amber for uncertainty/disclaimer. Strong numerical hierarchy, labelled axes, generous whitespace, 8px spacing system, 12–16px radii, subtle borders and restrained shadows. Do not mimic an insurer checkout page.

Use Auto Layout and semantic component variants for form fields, selects, number steppers, buttons, result cards, metric cards, model comparison table, disclaimer banner, tooltips and validation states. Pages: Foundations, Components, Predictor, Results & Methodology, Error States, Responsive. Design mobile layout with a stacked form then result card. Avoid fake metrics, fake savings, insurer logos, trust seals, medical imagery, and causal claims. Keep all content editable.
