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
