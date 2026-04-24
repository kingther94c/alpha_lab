"""Cross-sectional and time-series feature transforms.

Kept intentionally minimal — extend with rolling variants, group-wise ops,
and neutralization as real research demands them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(x: pd.Series | pd.DataFrame, axis: int = 0) -> pd.Series | pd.DataFrame:
    """Standardize to mean 0 / std 1 along *axis*.

    For a time-series Series: ``axis=0`` (over time).
    For a cross-sectional DataFrame (rows=date, cols=asset): ``axis=1``.
    """
    mean = x.mean(axis=axis)
    std = x.std(axis=axis).replace(0, np.nan)
    return (x.sub(mean, axis=1 - axis) if axis == 1 else x - mean) / std


def winsorize(x: pd.Series | pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.Series | pd.DataFrame:
    """Clip values to the [lower, upper] quantile range."""
    lo = x.quantile(lower)
    hi = x.quantile(upper)
    return x.clip(lower=lo, upper=hi, axis=1) if isinstance(x, pd.DataFrame) else x.clip(lower=lo, upper=hi)


def cross_sectional_rank(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Rank each row across columns (typical cross-sectional factor prep).

    Parameters
    ----------
    df : DataFrame with index=date, columns=asset.
    pct : return percentile ranks in [0, 1] (True) or integer ranks (False).
    """
    return df.rank(axis=1, pct=pct)
