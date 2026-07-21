"""Risk analytics: covariance, portfolio vol, marginal/risk contributions.

Starter set. TODO: shrinkage cov, factor-model decomposition, stress scenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cov_matrix(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Sample covariance matrix, annualized by *periods*."""
    return returns.cov() * periods


def portfolio_vol(weights: pd.Series | np.ndarray, cov: pd.DataFrame) -> float:
    """Annualized portfolio volatility given weights and covariance."""
    if isinstance(weights, pd.Series) and isinstance(cov, pd.DataFrame):
        missing = cov.index.difference(weights.index)
        if len(missing):
            raise ValueError(f"weights missing covariance labels: {missing.tolist()}")
        w = weights.reindex(cov.index).to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    cov_arr = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)
    return float(np.sqrt(w @ cov_arr @ w))


def geometric_portfolio_vol(
    weights: pd.Series | np.ndarray,
    cov_short: pd.DataFrame | np.ndarray,
    cov_long: pd.DataFrame | np.ndarray,
) -> float:
    """Geometric mean of portfolio volatility from two covariance windows."""
    short_vol = portfolio_vol(weights, cov_short)
    long_vol = portfolio_vol(weights, cov_long)
    return float(np.sqrt(short_vol * long_vol))


def geometric_realized_vol(
    returns: pd.Series,
    *,
    short_window: int = 21,
    long_window: int = 63,
    periods: int = 252,
) -> float:
    """Geometric mean of short- and long-window realized volatility."""
    if short_window < 2 or long_window < short_window:
        raise ValueError("windows must satisfy 2 <= short_window <= long_window")
    clean = returns.dropna()
    if len(clean) < long_window:
        return float("nan")
    short_vol = float(clean.tail(short_window).std() * np.sqrt(periods))
    long_vol = float(clean.tail(long_window).std() * np.sqrt(periods))
    return float(np.sqrt(short_vol * long_vol))


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Per-asset risk contribution to total portfolio vol. Sums to portfolio_vol."""
    w = weights.reindex(cov.index).fillna(0).values.astype(float)
    cov_arr = cov.values
    total = float(np.sqrt(w @ cov_arr @ w))
    if total == 0:
        return pd.Series(0.0, index=cov.index)
    mrc = cov_arr @ w / total          # marginal risk contribution
    rc = w * mrc                        # component risk contribution
    return pd.Series(rc, index=cov.index)


def cvar(returns: pd.Series, q: float = 0.05) -> float:
    """Conditional VaR: mean return in the worst ``q`` fraction of observations."""
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")

    r = returns.dropna()
    if r.empty:
        return float("nan")

    cutoff = r.quantile(q)
    return float(r[r <= cutoff].mean())
