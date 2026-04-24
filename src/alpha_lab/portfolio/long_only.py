"""Long-only ETF portfolio construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha_lab.data.calendars import rebalance_dates


def equal_weight_weights(columns: pd.Index | list[str], *, name: object | None = None) -> pd.Series:
    """Return fully-invested equal weights for the supplied asset names."""
    index = pd.Index(columns)
    if index.empty:
        return pd.Series(dtype=float, name=name)
    return pd.Series(1.0 / len(index), index=index, name=name)


def rolling_equal_weight_weights(prices: pd.DataFrame, *, rebalance: str = "ME") -> pd.DataFrame:
    """Build long-only equal weights on each rebalance date."""
    rows = [equal_weight_weights(prices.columns, name=date) for date in rebalance_dates(prices.index, freq=rebalance)]
    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    return pd.DataFrame(rows).reindex(columns=prices.columns)


def inverse_volatility_weights(
    asset_returns: pd.DataFrame,
    *,
    min_periods: int = 20,
) -> pd.Series:
    """Allocate inversely to trailing realized volatility."""
    clean = asset_returns.dropna(how="any")
    if len(clean) < min_periods:
        return pd.Series(dtype=float)

    vol = clean.std()
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
    inv_vol = inv_vol[inv_vol > 0]
    if inv_vol.empty:
        return equal_weight_weights(asset_returns.columns, name=asset_returns.index[-1])
    return (inv_vol / inv_vol.sum()).reindex(asset_returns.columns, fill_value=0.0).rename(asset_returns.index[-1])


def rolling_inverse_volatility_weights(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 63,
    rebalance: str = "ME",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build rolling long-only inverse-volatility weights."""
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")

    min_periods = min_periods or lookback_days
    returns = prices.pct_change()
    rows = []
    for date in rebalance_dates(prices.index, freq=rebalance):
        window = returns.loc[:date].tail(lookback_days)
        weights = inverse_volatility_weights(window, min_periods=min_periods)
        if not weights.empty:
            rows.append(weights)

    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    return pd.DataFrame(rows).reindex(columns=prices.columns).fillna(0.0)


def momentum_weights(
    momentum: pd.Series,
    *,
    top_n: int,
    vol: pd.Series | None = None,
) -> pd.Series:
    """Allocate to the top momentum assets, equal-weighted or inverse-vol weighted."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    scores = momentum.dropna().sort_values(ascending=False)
    winners = scores.head(top_n).index
    if winners.empty:
        return pd.Series(dtype=float)

    if vol is None:
        weights = equal_weight_weights(winners)
    else:
        risk = vol.reindex(winners).replace(0.0, np.nan).dropna()
        if risk.empty:
            weights = equal_weight_weights(winners)
        else:
            inv_vol = 1.0 / risk
            weights = inv_vol / inv_vol.sum()

    return weights.reindex(momentum.index, fill_value=0.0)


def rolling_momentum_weights(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 252,
    skip_days: int = 21,
    top_n: int = 3,
    rebalance: str = "ME",
    vol_adjust: bool = False,
    vol_lookback_days: int = 63,
) -> pd.DataFrame:
    """Build long-only top-momentum weights from trailing prices."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if skip_days < 0:
        raise ValueError("skip_days must be >= 0")
    if vol_lookback_days < 2:
        raise ValueError("vol_lookback_days must be >= 2")

    momentum = prices.shift(skip_days) / prices.shift(lookback_days + skip_days) - 1
    returns = prices.pct_change()

    rows = []
    for date in rebalance_dates(prices.index, freq=rebalance):
        vol = None
        if vol_adjust:
            vol = returns.loc[:date].tail(vol_lookback_days).std()
        weights = momentum_weights(momentum.loc[date], top_n=top_n, vol=vol)
        if not weights.empty:
            weights.name = date
            rows.append(weights)

    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    return pd.DataFrame(rows).reindex(columns=prices.columns).fillna(0.0)


def mean_variance_weights(
    asset_returns: pd.DataFrame,
    *,
    risk_aversion: float = 10.0,
    cov_ridge: float = 1e-8,
    min_periods: int = 20,
) -> pd.Series:
    """Optimize long-only weights using trailing mean-variance utility."""
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0")

    clean = asset_returns.dropna(how="any")
    if len(clean) < min_periods:
        return pd.Series(dtype=float)

    mu = clean.mean().to_numpy()
    cov = clean.cov().to_numpy() + np.eye(clean.shape[1]) * cov_ridge
    n_assets = clean.shape[1]
    initial = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        port_mean = weights @ mu
        port_var = weights @ cov @ weights
        return -(port_mean - 0.5 * risk_aversion * port_var)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=({"type": "eq", "fun": lambda weights: weights.sum() - 1.0},),
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")

    weights = pd.Series(result.x, index=clean.columns, name=asset_returns.index[-1])
    weights[weights.abs() < 1e-10] = 0.0
    return weights / weights.sum()


def rolling_mean_variance_weights(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 63,
    rebalance: str = "ME",
    risk_aversion: float = 10.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build rolling long-only mean-variance weights."""
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")

    min_periods = min_periods or lookback_days
    returns = prices.pct_change()
    rows = []
    for date in rebalance_dates(prices.index, freq=rebalance):
        window = returns.loc[:date].tail(lookback_days)
        weights = mean_variance_weights(
            window,
            risk_aversion=risk_aversion,
            min_periods=min_periods,
        )
        if not weights.empty:
            rows.append(weights)

    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    return pd.DataFrame(rows).reindex(columns=prices.columns).fillna(0.0)
