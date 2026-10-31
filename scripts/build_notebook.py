"""Generate notebooks/01_eda_and_model_evaluation.ipynb (then execute it with nbconvert).

The notebook only READS saved reports for model results; training logic lives in src/.
Usage: python -m scripts.build_notebook
"""
from __future__ import annotations

import nbformat as nbf

from src.utils.config import PROJECT_ROOT

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# CoverWise — EDA and model evaluation\n\n"
       "Exploratory analysis of the Medical Cost Personal Dataset plus a read-only review of the saved "
       "model-comparison reports. Stable logic lives in `src/`; this notebook only explores and explains.\n\n"
       "> Educational project. Associations shown here are **not** causal."),
    code("import json, sys\nfrom pathlib import Path\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
         "from src.utils.config import load_config\nfrom src.data.loader import load_raw, validate_dataframe\n"
         "from src.evaluation.eda import iqr_outliers\n"
         "sns.set_theme(style='whitegrid'); SEED = 42\n"
         "cfg = load_config(); df = load_raw(cfg)\ndf.head()"),
    md("## 1. Data quality and schema audit"),
    code("report = validate_dataframe(df, cfg)\n"
         "print('Shape:', df.shape)\nprint('Validation problems:', report.problems or 'none')\n"
         "print('Duplicate rows:', report.duplicate_rows)\nprint('Missing per column:', report.missing_per_column)\n"
         "df.dtypes"),
    md("**Finding.** The file has 1,338 rows and 7 columns, no missing values and one exact duplicate row. "
       "The duplicate is *kept*: with only six coarse attributes, two different people can legitimately share all "
       "values, and dropping it would change the data without evidence it is an error."),
    code("df.describe().round(2)"),
    code("df[['sex','smoker','region']].apply(lambda s: s.value_counts(normalize=True).round(3)).T"),
    md("## 2. Target distribution and skew"),
    code("t = cfg['dataset']['target']\n"
         "print('skew(charges)      =', round(df[t].skew(), 3))\n"
         "print('skew(log1p charges)=', round(np.log1p(df[t]).skew(), 3))\n"
         "print('IQR outliers in charges:', iqr_outliers(df[t]))\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
         "sns.histplot(df[t], bins=40, ax=ax[0]); ax[0].set(xlabel='Charges (USD)', title='Raw charges')\n"
         "sns.histplot(np.log1p(df[t]), bins=40, ax=ax[1]); ax[1].set(xlabel='log(1 + charges)', title='Log scale')\nplt.show()"),
    md("**Finding.** Charges are strongly right-skewed (skewness ≈ 1.52) and roughly symmetric after a log transform "
       "(≈ −0.09). 139 rows (about 10%) fall above the 1.5×IQR fence. These are **investigated, not removed**: "
       "136 of the 139 are smokers, i.e. real structure rather than data errors. A log-target linear model was "
       "included as a candidate to test whether the skew needs special treatment (see section 5)."),
    md("## 3. Relationships with charges"),
    code("fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))\n"
         "sns.scatterplot(data=df, x='age', y=t, hue='smoker', alpha=.7, ax=ax[0]); ax[0].set(xlabel='Age (years)', ylabel='Charges (USD)', title='Age vs charges')\n"
         "sns.scatterplot(data=df, x='bmi', y=t, hue='smoker', alpha=.7, ax=ax[1]); ax[1].axvline(30, ls='--', c='grey'); ax[1].set(xlabel='BMI (kg/m²)', ylabel='Charges (USD)', title='BMI vs charges')\nplt.show()"),
    code("fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
         "for ax, c in zip(axes, ['smoker','region','sex']):\n    sns.boxplot(data=df, x=c, y=t, ax=ax); ax.set(ylabel='Charges (USD)', title=f'Charges by {c}')\nplt.show()\n"
         "df.groupby('smoker')[t].agg(['count','mean','median']).round(0)"),
    code("df.groupby('region')[t].agg(['count','mean','median']).round(0)"),
    md("**Findings.**\n"
       "- **Smoking** is the dominant separator: mean charges ≈ $32,050 for smokers vs ≈ $8,434 for non-smokers.\n"
       "- **Age** shows several roughly parallel bands rising with age; of the 162 rows above $30,000, about 94% are smokers.\n"
       "- **BMI** matters mainly for smokers: mean charges are ≈ $21.4k for smokers with BMI < 30 vs ≈ $41.6k with BMI ≥ 30, "
       "while non-smokers move only from ≈ $8.0k to ≈ $8.8k. That *interaction* is something a plain linear model cannot "
       "represent, which motivates tree-based models.\n"
       "- **Region** differences in mean charges are modest (≈ $12.3k–$14.7k) relative to the spread within each region; "
       "**sex** differs little (medians ≈ $9.4k for both; the male mean is somewhat higher, ≈ $14.0k vs ≈ $12.6k, driven by the tail).\n"
       "These are descriptive associations in one small dataset, not causal effects."),
    md("## 4. Correlation"),
    code("corr = df.select_dtypes('number').corr()\nsns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues'); plt.title('Pearson correlation'); plt.show()"),
    md("**Finding.** Linear correlation with charges is weak-to-moderate for age (≈ 0.30) and BMI (≈ 0.20) and very "
       "weak for children (≈ 0.07). Pearson correlation misses the smoker interaction above, so it understates how "
       "predictive these variables are *jointly*."),
    md("## 5. Preprocessing and model comparison (read from saved reports)\n\n"
       "Preprocessing (median/most-frequent imputation, one-hot encoding, scaling for linear models only) lives inside "
       "scikit-learn `Pipeline`s, so it is fitted inside each CV fold. Nothing is recomputed here."),
    code("mc = json.loads((ROOT/'reports/model_comparison/model_comparison.json').read_text())\n"
         "rows = [{'Model': m['model'], **{f\"{k.upper()} mean\": round(m['cv'][k]['mean'], 3 if k=='r2' else 0) for k in ('mae','rmse','r2')},\n"
         "         **{f\"{k.upper()} std\": round(m['cv'][k]['std'], 3 if k=='r2' else 0) for k in ('mae','rmse','r2')}, 'Selected': m['selected']} for m in mc['models']]\n"
         "pd.DataFrame(rows)"),
    code("from IPython.display import Image\nImage(str(ROOT/'reports/figures/model_comparison_cv.png'))"),
    md("**Interpretation.** On identical shuffled 5-fold splits the tree-based models clearly beat the linear models. "
       "The log-target linear model is *worse* than the plain linear model on dollar-scale RMSE — transforming the "
       "target fixed the distribution's shape but did not improve dollar-scale error; a plausible (untested) reason is that "
       "errors on the log scale are amplified when converted back for high-cost cases, and the missing smoker × BMI "
       "interaction is likely the larger limitation of linear models. The tuned boosting and tuned random-forest models "
       "are within one fold-to-fold standard deviation of each other, so the choice between them is not clear-cut; "
       "gradient boosting was selected by the pre-declared rule (lowest mean CV RMSE)."),
    md("## 6. Final hold-out test, residuals and subgroup errors"),
    code("meta = json.loads((ROOT/'models/model_metadata.json').read_text())\n"
         "print(meta['model_name'], meta['best_params'])\nprint('Hold-out:', {k: round(v, 3) for k, v in meta['holdout_metrics'].items()})\n"
         "print('95% bootstrap CI:', {k: [round(x, 3) for x in v] for k, v in meta['holdout_metrics_95ci'].items()})"),
    code("Image(str(ROOT/'reports/figures/holdout_residuals.png'))"),
    code("Image(str(ROOT/'reports/figures/holdout_predicted_vs_actual.png'))"),
    code("pd.read_csv(ROOT/'reports/error_analysis/subgroup_errors.csv').round(1)"),
    md("**Notes.** Hold-out metrics come from 268 rows never used for selection or tuning. Subgroup rows with fewer than "
       "30 records (e.g. BMI < 18.5) are flagged and should not be over-interpreted. Subgroup differences are signals to "
       "investigate, not a fairness certification."),
    code("Image(str(ROOT/'reports/figures/permutation_importance.png'))"),
    md("**Feature importance.** Permutation importance on the hold-out set ranks smoker status first, then BMI and age; "
       "children, region and sex contribute little to predictive accuracy. This describes what the model uses, not what "
       "causes costs."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
out = PROJECT_ROOT / "notebooks" / "01_eda_and_model_evaluation.ipynb"
nbf.write(nb, out)
print("Wrote", out.name)
