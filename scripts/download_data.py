"""Download the Medical Cost Personal Dataset and write its provenance manifest.

Usage: python -m scripts.download_data [--force]
"""
from __future__ import annotations

import argparse
import sys

from src.data.loader import DatasetError, download_dataset, load_raw
from src.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = ap.parse_args(argv)
    cfg = load_config()
    try:
        path = download_dataset(cfg, force=args.force)
        df = load_raw(cfg)  # validates schema + checksum
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {path.name} ({len(df)} rows) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
