"""Small display-formatting helpers."""
from __future__ import annotations


def format_usd(value: float) -> str:
    """Format a number as US dollars, e.g. ``$12,345.67``."""
    return f"${value:,.2f}"


def mean_std(mean: float, std: float, digits: int = 2) -> str:
    """Format ``mean ± std``."""
    return f"{mean:,.{digits}f} ± {std:,.{digits}f}"
