"""Dataset acquisition, loading and validation.

The raw file is treated as immutable: it is downloaded once, checksummed and never edited.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import feature_columns, resolve


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, corrupt or fails validation."""


def sha256_of(path: Path) -> str:
    """SHA-256 checksum of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_dataset(cfg: dict[str, Any], force: bool = False) -> Path:
    """Download the raw CSV (once) and write a provenance manifest.

    Raises ``DatasetError`` with a clear, actionable message on failure. Nothing is
    ever synthesised as a fallback.
    """
    ds = cfg["dataset"]
    dest = resolve(cfg, ds["raw_path"])
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(ds["url"], timeout=30) as resp:  # noqa: S310 (fixed https URL)
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DatasetError(
            f"Could not download the dataset ({exc}). Download 'insurance.csv' manually "
            f"from Kaggle (mirichoi0218/insurance) or {ds['url']} and place it at "
            f"{ds['raw_path']}."
        ) from exc
    dest.write_bytes(data)
    _write_manifest(cfg, dest)
    return dest


def _write_manifest(cfg: dict[str, Any], path: Path) -> None:
    ds = cfg["dataset"]
    df = pd.read_csv(path)
    manifest = {
        "name": ds["name"],
        "source_url": ds["url"],
        "original_source": "Kaggle: 'Medical Cost Personal Datasets' (mirichoi0218/insurance), "
        "derived from Brett Lantz, 'Machine Learning with R' (Packt)",
        "licence": "Kaggle lists the dataset under the Open Database Licence (ODbL) with "
        "contents under DbCL v1.0; verify terms before redistributing.",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_of(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_unit": ds["target_unit"],
    }
    resolve(cfg, ds["manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")


@dataclass
class AuditReport:
    """Result of validating the raw dataset."""

    rows: int
    columns: list[str]
    missing_per_column: dict[str, int]
    duplicate_rows: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_dataframe(df: pd.DataFrame, cfg: dict[str, Any]) -> AuditReport:
    """Check schema, types, categories, ranges and target validity (no mutation)."""
    schema, target = cfg["schema"], cfg["dataset"]["target"]
    required = feature_columns(cfg) + [target]
    problems: list[str] = []

    if df.empty:
        raise DatasetError("Dataset is empty.")
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise DatasetError(f"Dataset is missing required columns: {absent}")

    for col in schema["numeric_features"] + [target]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Column '{col}' is not numeric.")
    for col, allowed in schema["categories"].items():
        unexpected = set(df[col].dropna().unique()) - set(allowed)
        if unexpected:
            problems.append(f"Column '{col}' has unexpected categories: {sorted(unexpected)}")
    if pd.api.types.is_numeric_dtype(df[target]):
        if (df[target].dropna() <= 0).any():
            problems.append("Target contains non-positive values.")
        if df[target].isna().any():
            problems.append("Target contains missing values.")
    for col, (lo, hi) in schema["bounds"].items():
        if pd.api.types.is_numeric_dtype(df[col]):
            out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if out:
                problems.append(f"{out} rows have '{col}' outside [{lo}, {hi}].")

    return AuditReport(
        rows=len(df),
        columns=list(df.columns),
        missing_per_column={c: int(v) for c, v in df.isna().sum().items()},
        duplicate_rows=int(df.duplicated().sum()),
        problems=problems,
    )


def load_raw(cfg: dict[str, Any], validate: bool = True) -> pd.DataFrame:
    """Load the raw CSV and (optionally) validate it. The file is never modified."""
    path = resolve(cfg, cfg["dataset"]["raw_path"])
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at '{cfg['dataset']['raw_path']}'. "
            "Run: python -m scripts.download_data"
        )
    expected = cfg["dataset"].get("expected_sha256")
    if expected and sha256_of(path) != expected:
        raise DatasetError("Dataset checksum does not match configs/default.json.")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Dataset could not be parsed: {exc}") from exc
    if validate:
        report = validate_dataframe(df, cfg)
        if not report.ok:
            raise DatasetError("Dataset validation failed: " + "; ".join(report.problems))
    return df


def split_xy(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw features from the target. The target is never part of ``X``."""
    X = df[feature_columns(cfg)].copy()
    y = df[cfg["dataset"]["target"]].copy()
    return X, y


def train_test_holdout(
    X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic hold-out split, stratified on smoker status (the dominant cost driver)."""
    t = cfg["training"]
    return train_test_split(
        X, y,
        test_size=t["test_size"],
        random_state=t["random_seed"],
        stratify=X["smoker"] if "smoker" in X else None,
    )
