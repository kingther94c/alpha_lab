"""Cross-sectional and time-series feature transforms.

Kept intentionally minimal — extend with rolling variants, group-wise ops,
and neutralization as real research demands them.

Includes :class:`Standardizer`, a leak-safe fit-on-train / transform-on-val
normalizer for ML pipelines.
"""

from __future__ import annotations

from typing import Literal

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


class Standardizer:
    """Leak-safe normalizer for ML pipelines.

    Fit cross-sectional or pooled mean / std on the training set, then apply
    the same parameters when transforming validation / test sets. NEVER refit
    on val/test.

    Parameters
    ----------
    mode : ``"per_column"`` (default) computes a separate mean/std for each
        column (typical for time-series features per asset). ``"pooled"``
        pools across all values into a single scalar mean/std (useful when
        feature scale is uniform across assets).
    winsorize_bounds : optional ``(lower_q, upper_q)`` to clip before fitting.
        Bounds are computed on the training set only and stored.
    """

    def __init__(
        self,
        *,
        mode: Literal["per_column", "pooled"] = "per_column",
        winsorize_bounds: tuple[float, float] | None = None,
    ):
        if mode not in ("per_column", "pooled"):
            raise ValueError(f"mode must be 'per_column' or 'pooled', got {mode!r}")
        self.mode = mode
        self.winsorize_bounds = winsorize_bounds
        self._mean: pd.Series | float | None = None
        self._std: pd.Series | float | None = None
        self._winsor_lo: pd.Series | float | None = None
        self._winsor_hi: pd.Series | float | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        x = df
        if self.winsorize_bounds is not None:
            lo_q, hi_q = self.winsorize_bounds
            self._winsor_lo = x.quantile(lo_q)
            self._winsor_hi = x.quantile(hi_q)
            x = x.clip(lower=self._winsor_lo, upper=self._winsor_hi, axis=1)
        if self.mode == "per_column":
            self._mean = x.mean()
            std = x.std()
            self._std = std.replace(0, np.nan)
        else:  # pooled
            # pandas 3.0 removed the dropna kwarg from stack(); manually drop after.
            flat = x.stack().dropna()
            self._mean = float(flat.mean())
            std_val = float(flat.std())
            self._std = std_val if std_val != 0 else float("nan")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Standardizer must be .fit() before .transform()")
        x = df
        if self.winsorize_bounds is not None:
            x = x.clip(lower=self._winsor_lo, upper=self._winsor_hi, axis=1)
        return (x - self._mean) / self._std

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
