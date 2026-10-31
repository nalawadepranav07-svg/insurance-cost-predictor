"""Render reports/RESULTS.md from the saved evaluation JSON (no numbers are typed by hand).

Usage: python -m scripts.export_report
"""
from __future__ import annotations

import json
import sys

from src.utils.config import load_config, resolve


def build_markdown(meta: dict, comparison: list[dict]) -> str:
    lines = ["# CoverWise — evaluation results", "",
             f"Generated from the saved reports. Model ID `{meta['model_id']}`; dataset SHA-256 "
             f"`{meta['dataset_sha256'][:12]}…`; random seed {meta['random_seed']}.", "",
             "## 5-fold cross-validation (training split, identical shuffled folds)", "",
             "| Model | MAE (USD) | RMSE (USD) | R² | Selected |", "|---|---|---|---|---|"]
    for r in comparison:
        c = r["cv"]
        lines.append(f"| {r['model']} | {c['mae']['mean']:,.0f} ± {c['mae']['std']:,.0f} | "
                     f"{c['rmse']['mean']:,.0f} ± {c['rmse']['std']:,.0f} | "
                     f"{c['r2']['mean']:.3f} ± {c['r2']['std']:.3f} | {'**yes**' if r['selected'] else ''} |")
    h, ci = meta["holdout_metrics"], meta["holdout_metrics_95ci"]
    lines += ["", f"## Final hold-out test ({meta['holdout_rows']} unseen rows) — {meta['model_name']}", "",
              "| Metric | Value | 95% bootstrap CI |", "|---|---|---|",
              f"| MAE (USD) | {h['mae']:,.0f} | {ci['mae'][0]:,.0f} – {ci['mae'][1]:,.0f} |",
              f"| RMSE (USD) | {h['rmse']:,.0f} | {ci['rmse'][0]:,.0f} – {ci['rmse'][1]:,.0f} |",
              f"| R² | {h['r2']:.3f} | {ci['r2'][0]:.3f} – {ci['r2'][1]:.3f} |", "",
              "## Permutation importance (hold-out, RMSE increase in USD)", "",
              "| Feature | Mean | Std |", "|---|---|---|"]
    for d in meta["feature_importance"]["values"]:
        lines.append(f"| {d['feature']} | {d['mean']:,.0f} | {d['std']:,.0f} |")
    lines += ["", "Selected hyper-parameters: `" + json.dumps(meta["best_params"]) + "`", ""]
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    try:
        meta = json.loads(resolve(cfg, cfg["paths"]["metadata_path"]).read_text())
        comparison = json.loads(resolve(cfg, "reports/model_comparison/model_comparison.json").read_text())["models"]
    except (OSError, json.JSONDecodeError):
        print("ERROR: run `python -m scripts.train_model --overwrite` first.", file=sys.stderr)
        return 1
    out = resolve(cfg, "reports/RESULTS.md")
    out.write_text(build_markdown(meta, comparison))
    print(f"Wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
