"""Factor analytics: IC, rank IC, quantile buckets.

Starter set. TODO: cross-sectional IC time series, quantile return curves,
turnover, information decay, neutralization.
"""

from __future__ import annotations

import pandas as pd


def ic(factor: pd.Series, forward_returns: pd.Series) -> float:
    """Pearson correlation between factor and forward returns. NaNs dropped pairwise."""
    aligned = pd.concat([factor, forward_returns], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def rank_ic(factor: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman (rank) correlation — more robust to outliers than ``ic``."""
    aligned = pd.concat([factor, forward_returns], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def quantile_buckets(factor: pd.Series | pd.DataFrame, n: int = 5) -> pd.Series | pd.DataFrame:
    """Assign each value to one of *n* quantile buckets (1..n).

    For a cross-sectional DataFrame (rows=date, cols=asset), ranks within each row.
    """
    if isinstance(factor, pd.DataFrame):
        return factor.apply(lambda row: pd.qcut(row, n, labels=False, duplicates="drop") + 1, axis=1)
    return pd.qcut(factor, n, labels=False, duplicates="drop") + 1
