"""Long-only active mean-variance portfolio construction."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha_lab.data.calendars import rebalance_dates


def active_mean_variance_weights(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    risk_aversion: float = 10.0,
    cov_ridge: float = 1e-8,
) -> pd.Series:
    """Optimize long-only weights against a benchmark active-return objective."""
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0")

    aligned = asset_returns.join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    if aligned.empty:
        raise ValueError("asset and benchmark returns have no overlapping observations")

    assets = aligned.drop(columns="__benchmark__")
    active = assets.sub(aligned["__benchmark__"], axis=0)
    mu = active.mean().to_numpy()
    cov = active.cov().to_numpy() + np.eye(active.shape[1]) * cov_ridge

    n_assets = active.shape[1]
    initial = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        active_mean = weights @ mu
        active_var = weights @ cov @ weights
        return -(active_mean - 0.5 * risk_aversion * active_var)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=({"type": "eq", "fun": lambda weights: weights.sum() - 1.0},),
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")

    weights = pd.Series(result.x, index=active.columns, name=asset_returns.index[-1])
    weights[weights.abs() < 1e-10] = 0.0
    return weights / weights.sum()


def rolling_active_mean_variance_weights(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    *,
    lookback_days: int = 63,
    rebalance: str = "W-FRI",
    risk_aversion: float = 10.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build weekly rolling long-only weights from trailing active returns."""
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")

    min_periods = min_periods or lookback_days
    asset_returns = prices.pct_change()
    benchmark_returns = benchmark_prices.pct_change()

    common_index = prices.index.intersection(benchmark_prices.index)
    asset_returns = asset_returns.reindex(common_index)
    benchmark_returns = benchmark_returns.reindex(common_index)

    rows = []
    dates = []
    for date in rebalance_dates(common_index, freq=rebalance):
        asset_window = asset_returns.loc[:date].tail(lookback_days)
        benchmark_window = benchmark_returns.loc[:date].tail(lookback_days)
        enough_data = asset_window.dropna(how="any").shape[0] >= min_periods
        if not enough_data:
            continue
        rows.append(
            active_mean_variance_weights(
                asset_window,
                benchmark_window,
                risk_aversion=risk_aversion,
            )
        )
        dates.append(date)

    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).reindex(columns=prices.columns)
