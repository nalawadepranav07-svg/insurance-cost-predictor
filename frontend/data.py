"""Loading of the trained model and saved reports for the UI (cached)."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.prediction.predictor import ModelUnavailableError, Predictor
from src.utils.config import ConfigError, load_config, resolve


@st.cache_resource(show_spinner=False)
def get_config() -> dict[str, Any]:
    return load_config()


@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> tuple[Predictor | None, str | None]:
    """Return (predictor, None) or (None, user-facing error message). Never raises."""
    try:
        return Predictor.load(get_config()), None
    except (ModelUnavailableError, ConfigError) as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - the UI must not crash on startup
        return None, f"Unexpected error while loading the model ({type(exc).__name__})."


@st.cache_data(show_spinner=False)
def load_report_json(relative_path: str) -> dict | None:
    path = resolve(get_config(), relative_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_subgroups() -> pd.DataFrame | None:
    path = resolve(get_config(), "reports/error_analysis/subgroup_errors.csv")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
