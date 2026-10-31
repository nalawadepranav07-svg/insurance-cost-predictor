# `scripts/`
| Command | Purpose |
|---|---|
| `python -m scripts.download_data [--force]` | Download + validate the dataset, write manifest |
| `python -m scripts.run_eda` | EDA statistics + figures |
| `python -m scripts.train_model [--no-tuning] [--overwrite]` | CV comparison, selection, hold-out evaluation, save model |
| `python -m scripts.export_report` | Render `reports/RESULTS.md` from saved JSON |
| `python -m scripts.build_notebook` | Regenerate the EDA notebook (execute with `jupyter nbconvert`) |
Exit codes: 0 success, 1 error (message on stderr), 2 refused to overwrite an existing model.
