# `data/`
Zones: `raw/` (immutable download + `MANIFEST.json`), `interim/`, `processed/`, `external/` (unused; empty by design — no
external datasets are mixed in). Raw CSV is git-ignored; recreate it with `python -m scripts.download_data`.

**Source:** Kaggle "Medical Cost Personal Datasets" (`mirichoi0218/insurance`), derived from Brett Lantz, *Machine Learning
with R* (Packt). Downloaded from a public GitHub mirror (URL in `configs/default.json`, SHA-256 pinned).
**Licence:** Kaggle lists ODbL / DbCL v1.0 — verify before redistributing. **Rows:** 1,338. **Target:** `charges`, USD.

| Column | Type | Meaning |
|---|---|---|
| age | int (18–64) | Age of primary beneficiary |
| sex | female / male | Category as recorded in the dataset |
| bmi | float (15.96–53.13) | Body-mass index, kg/m² |
| children | int (0–5) | Number of dependants covered |
| smoker | no / yes | Smoking status |
| region | northeast / northwest / southeast / southwest | US residential area |
| charges | float | Historical individual medical costs billed (USD) — the **target** |
