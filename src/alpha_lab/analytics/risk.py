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
    w = np.asarray(weights, dtype=float)
    cov_arr = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)
    return float(np.sqrt(w @ cov_arr @ w))


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
