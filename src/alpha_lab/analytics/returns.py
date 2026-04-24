"""Return math. Works on Series or DataFrame of prices/returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.Series | pd.DataFrame


def simple_returns(prices: Frame) -> Frame:
    """Arithmetic returns: p_t / p_{t-1} - 1."""
    return prices.pct_change()


def log_returns(prices: Frame) -> Frame:
    """Continuously-compounded returns: ln(p_t / p_{t-1})."""
    return np.log(prices / prices.shift(1))


def cumulative_returns(returns: Frame) -> Frame:
    """Compound simple returns into a wealth index starting at 1."""
    return (1 + returns.fillna(0)).cumprod()


def drawdown(returns: Frame) -> Frame:
    """Drawdown series from simple returns (peak-to-trough, as a negative fraction)."""
    wealth = cumulative_returns(returns)
    peak = wealth.cummax()
    return wealth / peak - 1


def annualized_vol(returns: Frame, periods: int = 252) -> Frame | float:
    """sqrt(periods) * std. Default assumes daily returns."""
    return returns.std() * np.sqrt(periods)


def sharpe(returns: Frame, rf: float = 0.0, periods: int = 252) -> Frame | float:
    """Naive Sharpe: (mean(excess) / std(excess)) * sqrt(periods).

    ``rf`` is a per-period risk-free rate (same freq as ``returns``).
    """
    excess = returns - rf
    mu = excess.mean()
    sd = excess.std()
    return (mu / sd) * np.sqrt(periods)
