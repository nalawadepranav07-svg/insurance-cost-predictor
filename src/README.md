# `src/` — source package
Typed, reusable modules. Training and inference share one persisted scikit-learn `Pipeline`.
| Package | Contents |
|---|---|
| `data/` | `loader.py`: download, checksum, schema/range validation, load, deterministic split |
| `features/` | `preprocessing.py`: leak-safe `ColumnTransformer` (imputation, scaling, one-hot) |
| `models/` | `candidates.py` (models + tuning spaces), `trainer.py` (CV → selection → hold-out → persistence) |
| `evaluation/` | `metrics.py` (MAE/RMSE/R², bootstrap CI, subgroup errors), `plots.py`, `eda.py` |
| `prediction/` | `predictor.py`: input validation, checksum/schema-verified model loading, inference |
| `utils/` | `config.py`, `reproducibility.py`, `formatting.py` |
