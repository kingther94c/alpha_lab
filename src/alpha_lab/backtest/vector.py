"""Vectorized signal-backtest engine.

Simple model: target weights are set on rebalance dates and held constant
until the next rebalance. Signals are lagged by 1 period (act on today's
signal starting next period) to avoid look-ahead. Costs apply on turnover.

Assumes ``signals`` are already interpreted as target weights. Upstream code
is responsible for ranking / normalizing raw factor values into weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.analytics.returns import cumulative_returns
from alpha_lab.data.calendars import rebalance_dates


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    weights: pd.DataFrame       # lagged weights actually held each period
    asset_returns: pd.DataFrame
    returns: pd.Series          # net portfolio returns (after costs)
    gross_returns: pd.Series
    turnover: pd.Series         # per-period one-way turnover (half-sum of |Δw|)
    costs: pd.Series            # per-period cost drag as a fraction

    @property
    def equity(self) -> pd.Series:
        """Wealth index starting at 1 from net returns."""
        return cumulative_returns(self.returns)


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    rebalance: str | None = None,
    costs_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """Run a vectorized signal backtest.

    Parameters
    ----------
    signals : DataFrame of target weights (index=date, cols=asset). Rows need
        not sum to 1 — interpret as dollar exposure per unit capital.
    prices : wide DataFrame of prices (same columns as ``signals``).
    rebalance : pandas offset alias (``"ME"``, ``"QE"``, ``"W-FRI"``, ...).
        ``None`` rebalances every date in ``prices.index``.
    costs_bps : one-way commission, in basis points of notional traded.
    slippage_bps : one-way slippage, in basis points of notional traded.

    Returns
    -------
    BacktestResult
    """
    signals = signals.reindex(columns=prices.columns).reindex(prices.index).ffill()

    if rebalance is None:
        reb = prices.index
    else:
        reb = rebalance_dates(prices.index, freq=rebalance)

    # Target weights: refresh only on rebalance dates, ffill between.
    tgt = signals.loc[signals.index.isin(reb)].reindex(prices.index).ffill().fillna(0.0)

    # Lag by 1 so today's signal is held starting next period.
    held = tgt.shift(1).fillna(0.0)

    asset_rets = prices.pct_change().fillna(0.0)
    gross = (held * asset_rets).sum(axis=1)

    # Turnover: half-sum of absolute weight change vs. previous day's held weights.
    # Between rebalances held is constant, so this is only nonzero the day after
    # each rebalance (when new target kicks in).
    prev = held.shift(1).fillna(0.0)
    turn = (held - prev).abs().sum(axis=1) / 2.0

    cost = turn * (costs_bps + slippage_bps) / 10_000.0
    net = gross - cost

    return BacktestResult(
        weights=held,
        asset_returns=asset_rets,
        returns=net,
        gross_returns=gross,
        turnover=turn,
        costs=cost,
    )
