"""SYNTHETIC test data. Every record is artificial and exists only to test software behaviour;
it does not approximate real population statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the real schema (units: USD-like charges)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "sex": rng.choice(["female", "male"], n),
        "bmi": np.round(np.clip(rng.normal(30, 5, n), 16, 52), 2),
        "children": rng.integers(0, 6, n),
        "smoker": rng.choice(["no", "yes"], n, p=[0.8, 0.2]),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    noise = rng.normal(0, 800, n)
    df["charges"] = np.round(
        2000 + 250 * df["age"] + 150 * df["children"]
        + (df["smoker"] == "yes") * (15000 + 400 * (df["bmi"] - 25).clip(0)) + noise, 2
    ).clip(lower=1000)
    return df
