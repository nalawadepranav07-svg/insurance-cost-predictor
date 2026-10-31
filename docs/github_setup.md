# GitHub repository setup
```bash
cd insurance-cost-predictor
git init -b main
git add .
git status            # confirm: no .venv, no data/raw/insurance.csv, no models/*.joblib
git commit -m "CoverWise: insurance cost predictor (initial commit)"
# create an EMPTY repo on github.com first (no README/licence), then:
git remote add origin https://github.com/<your-username>/insurance-cost-predictor.git
git push -u origin main
```
With the GitHub CLI: `gh repo create insurance-cost-predictor --public --source=. --push`.

**Before publishing:** add a `LICENSE` of your choice (none is included); the dataset has its own licence (see `data/README.md`).

**Deploying (e.g. Streamlit Community Cloud):** the model binary and CSV are git-ignored. Either (a) run
`python -m scripts.download_data && python -m scripts.train_model --overwrite` in the deploy environment's build step, or
(b) deliberately commit `models/coverwise_model.joblib` + `models/model_metadata.json` (≈ 90 KB; remove the `models/*.joblib`
line from `.gitignore`). Main file path: `frontend/app.py`.
