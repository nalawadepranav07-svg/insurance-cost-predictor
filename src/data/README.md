# `src/data`
`loader.py` — `download_dataset` (fixed HTTPS URL, writes `data/raw/MANIFEST.json`), `load_raw` (checksum + schema +
range validation, never mutates the file), `validate_dataframe`, `split_xy` (target is never in `X`), `train_test_holdout`
(seeded, stratified by `smoker`). Missing data raises `DatasetError` with the fix; nothing is ever synthesised.
