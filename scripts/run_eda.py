"""Compute EDA statistics and figures from the raw dataset.

Usage: python -m scripts.run_eda
Writes reports/eda_summary.json and reports/figures/eda_*.png
"""
from __future__ import annotations

import json
import sys

from src.data.loader import DatasetError, load_raw
from src.evaluation.eda import make_figures, summarize
from src.utils.config import load_config, resolve


def main() -> int:
    cfg = load_config()
    try:
        df = load_raw(cfg)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    target = cfg["dataset"]["target"]
    summary = summarize(df, target)
    reports = resolve(cfg, cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    paths = make_figures(df, target, resolve(cfg, cfg["paths"]["figures_dir"]))
    print(f"Wrote eda_summary.json and {len(paths)} figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
