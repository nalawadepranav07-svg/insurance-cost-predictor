"""Configuration loading. All tunable settings live in ``configs/default.json``."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"

_REQUIRED_SECTIONS = ("dataset", "schema", "training", "paths")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the experiment configuration.

    The path may be passed explicitly, set via the ``COVERWISE_CONFIG`` environment variable,
    or defaults to ``configs/default.json``.
    """
    cfg_path = Path(path or os.environ.get("COVERWISE_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path.name}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {exc}") from exc

    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Config is missing sections: {missing}")
    if cfg["training"].get("cv_folds") != 5:
        raise ConfigError("This project requires cv_folds == 5.")
    schema = cfg["schema"]
    features = schema["numeric_features"] + schema["categorical_features"]
    if cfg["dataset"]["target"] in features:
        raise ConfigError("Target column must not be listed as a feature (leakage).")
    return cfg


def resolve(cfg: dict[str, Any], relative: str) -> Path:
    """Resolve a project-relative path from the config (never absolute)."""
    return PROJECT_ROOT / relative


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Ordered list of raw model input columns."""
    s = cfg["schema"]
    return list(s["numeric_features"]) + list(s["categorical_features"])
