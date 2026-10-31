# `src/prediction`
`validate_inputs` rejects missing fields, wrong types, non-finite/out-of-range numbers, unknown categories and any extra
field (including `charges`). `Predictor.load` verifies the artifact's SHA-256 and the feature schema **before** unpickling,
and raises `ModelUnavailableError` with a fix hint otherwise. `Predictor.predict` returns estimate, model ID, target unit
and the educational disclaimer. Only load artifacts you trained yourself: joblib files can execute code when loaded.
