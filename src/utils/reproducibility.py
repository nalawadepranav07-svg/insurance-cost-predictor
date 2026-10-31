"""Seeding and environment/version reporting."""
from __future__ import annotations

import platform
import random
import sys

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs (scikit-learn estimators get seeds explicitly)."""
    random.seed(seed)
    np.random.seed(seed)


def environment_versions() -> dict[str, str]:
    """Package versions recorded in model metadata."""
    import joblib
    import pandas
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
