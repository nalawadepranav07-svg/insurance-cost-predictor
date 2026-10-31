# `src/features`
`build_preprocessor(cfg, scale_numeric)` returns the `ColumnTransformer` used by every candidate.
Numeric: median imputation (+ standardisation for linear models only). Categorical: most-frequent imputation +
`OneHotEncoder(handle_unknown="ignore", drop="if_binary")`. `remainder="drop"` guarantees no stray column (e.g. the target)
is passed through. It is always used *inside* a `Pipeline`, so it is fitted per CV fold.
