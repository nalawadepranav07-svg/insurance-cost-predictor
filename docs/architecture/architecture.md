# Architecture

```mermaid
flowchart LR
  A[download_data<br/>raw CSV + SHA-256 manifest] --> B[validate schema, ranges, target]
  B --> C[stratified 80/20 split<br/>seed 42]
  C --> D[Train split]
  C --> H[Hold-out split<br/>untouched]
  D --> E[Same shuffled 5-fold CV<br/>Pipeline = ColumnTransformer + model<br/>preprocessing fitted per fold]
  E --> F[Select lowest mean CV RMSE]
  F --> G[Fit chosen pipeline on full train split<br/>re-tune if tuned candidate]
  G --> H2[Evaluate ONCE on hold-out]
  H --> H2
  G --> I[(coverwise_model.joblib<br/>+ model_metadata.json)]
  I --> J[Predictor: verify SHA-256 + schema,<br/>validate inputs, predict]
  J --> K[Streamlit UI]
```

**Same preprocessing at inference.** The persisted object is the complete fitted `Pipeline`
(`preprocess` → `model`). The UI/`Predictor` passes a one-row DataFrame with the six raw columns; no encoding, scaling or
imputation is re-implemented outside the pipeline.

**Artifacts & versioning.** `model_metadata.json` stores model ID (first 12 hex chars of the artifact SHA-256), dataset
SHA-256, schema, protocol, metrics, library versions, timestamp and seed.

**Failure behaviour.** Missing/corrupt/tampered model, unreadable metadata or schema mismatch → `ModelUnavailableError`
(UI shows the message and disables prediction). Invalid inputs → `ValidationError` with per-field messages (model never
called). Non-finite model output → `PredictionError`. Missing dataset → `DatasetError` with the fix command.
