"""Input validation and inference.

The predictor loads ONLY the trusted local artifact produced by ``scripts/train_model.py``,
verifies its checksum and schema, and feeds raw feature values to the *same fitted pipeline*
that was trained (no preprocessing is re-implemented here or in the UI).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.data.loader import sha256_of
from src.utils.config import feature_columns, resolve

DISCLAIMER = (
    "Educational estimate only — not an insurer quote, premium offer, medical advice or "
    "guarantee. It reflects patterns in a historical dataset."
)


class ValidationError(ValueError):
    """User input failed validation. ``errors`` maps field name -> message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


class ModelUnavailableError(RuntimeError):
    """The trained artifact is missing, corrupt, tampered with or schema-incompatible."""


class PredictionError(RuntimeError):
    """The model failed to produce a valid numeric prediction."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_inputs(raw: Mapping[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise user input; raises ``ValidationError`` with per-field messages.

    Rejects missing fields, wrong types, non-finite or out-of-range numbers, unknown
    categories, and any unexpected field (including the target ``charges``).
    """
    schema = cfg["schema"]
    allowed = set(feature_columns(cfg))
    errors: dict[str, str] = {}

    for extra in sorted(set(raw) - allowed):
        errors[extra] = "Unexpected field. Only the six model inputs are accepted."

    clean: dict[str, Any] = {}
    for col in schema["numeric_features"]:
        if col not in raw or raw[col] is None or raw[col] == "":
            errors[col] = "This field is required."
            continue
        value = raw[col]
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                errors[col] = "Must be a number."
                continue
        if not _is_number(value) or not math.isfinite(value):
            errors[col] = "Must be a finite number."
            continue
        lo, hi = schema["bounds"][col]
        if not lo <= value <= hi:
            errors[col] = f"Must be between {lo} and {hi}."
            continue
        if col in ("age", "children"):
            if float(value) != int(value):
                errors[col] = "Must be a whole number."
                continue
            value = int(value)
        else:
            value = float(value)
        clean[col] = value

    for col in schema["categorical_features"]:
        options = schema["categories"][col]
        if col not in raw or raw[col] is None or raw[col] == "":
            errors[col] = "This field is required."
            continue
        value = str(raw[col]).strip().lower()
        if value not in options:
            errors[col] = f"Must be one of: {', '.join(options)}."
            continue
        clean[col] = value

    if errors:
        raise ValidationError(errors)
    return clean


@dataclass(frozen=True)
class PredictionResult:
    estimate: float
    inputs: dict[str, Any]
    model_id: str
    model_name: str
    target_unit: str
    disclaimer: str = DISCLAIMER


class Predictor:
    """Wraps the trusted fitted pipeline plus its metadata."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any], cfg: dict[str, Any]):
        self.pipeline, self.metadata, self.cfg = pipeline, metadata, cfg

    @classmethod
    def load(cls, cfg: dict[str, Any]) -> "Predictor":
        """Load artifact + metadata; raises ``ModelUnavailableError`` with a helpful message."""
        model_path: Path = resolve(cfg, cfg["paths"]["model_path"])
        meta_path: Path = resolve(cfg, cfg["paths"]["metadata_path"])
        hint = "Train it first with: python -m scripts.train_model --overwrite"
        if not model_path.exists() or not meta_path.exists():
            raise ModelUnavailableError(f"No trained model found. {hint}")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ModelUnavailableError(f"Model metadata is unreadable ({exc}). {hint}") from exc

        # Integrity check BEFORE unpickling anything (joblib files can execute code on load).
        if sha256_of(model_path) != metadata.get("artifact_sha256"):
            raise ModelUnavailableError(
                f"Model file does not match its recorded checksum (corrupt or modified). {hint}")
        if metadata.get("features") != feature_columns(cfg) or metadata.get("schema") != cfg["schema"]:
            raise ModelUnavailableError(f"Model schema does not match the current configuration. {hint}")
        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - any unpickling failure means "unusable"
            raise ModelUnavailableError(f"Model file could not be loaded ({type(exc).__name__}). {hint}") from exc
        return cls(pipeline, metadata, cfg)

    def predict(self, raw_inputs: Mapping[str, Any]) -> PredictionResult:
        """Validate inputs, then predict with the saved pipeline."""
        clean = validate_inputs(raw_inputs, self.cfg)
        frame = pd.DataFrame([clean], columns=feature_columns(self.cfg))
        try:
            value = float(self.pipeline.predict(frame)[0])
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Prediction failed ({type(exc).__name__}).") from exc
        if not math.isfinite(value):
            raise PredictionError("The model returned a non-finite value.")
        return PredictionResult(
            estimate=max(value, 0.0),  # charges cannot be negative
            inputs=clean,
            model_id=self.metadata["model_id"],
            model_name=self.metadata["model_name"],
            target_unit=self.metadata["target_unit"],
        )
