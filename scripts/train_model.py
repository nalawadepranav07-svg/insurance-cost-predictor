"""Compare candidate models with 5-fold CV, evaluate on the hold-out set, save the best pipeline.

Usage: python -m scripts.train_model [--config configs/default.json] [--no-tuning] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src.data.loader import DatasetError
from src.models.trainer import run_training
from src.utils.config import ConfigError, load_config, resolve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to a JSON config (default: configs/default.json).")
    ap.add_argument("--no-tuning", action="store_true", help="Skip nested-CV hyperparameter tuning (faster).")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing model artifact.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        cfg = load_config(args.config)
        if args.no_tuning:
            cfg["training"]["tuning"]["enabled"] = False
        if resolve(cfg, cfg["paths"]["model_path"]).exists() and not args.overwrite:
            print("A trained model already exists. Re-run with --overwrite to replace it.", file=sys.stderr)
            return 2
        meta = run_training(cfg)
    except (ConfigError, DatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: meta[k] for k in ("model_id", "model_name", "best_params", "holdout_metrics")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
