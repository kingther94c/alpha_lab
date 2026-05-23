"""Vectorized signal-backtest engine.

Simple model: target weights are set on rebalance dates and held constant
until the next rebalance. Signals are lagged by 1 period (act on today's
signal starting next period) to avoid look-ahead. Costs apply on turnover.

Assumes ``signals`` are already interpreted as target weights. Upstream code
is responsible for ranking / normalizing raw factor values into weights.

Funding (perp-only)
-------------------
Pass a ``funding`` DataFrame indexed by funding timestamp (e.g. 8h cadence)
with one column per symbol matching ``prices``. Each funding event is
charged at the bar in ``prices.index`` that contains it: long pays a positive
rate, short receives. The funding cost is reported separately and netted off
in ``returns``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from alpha_lab.analytics.returns import cumulative_returns
from alpha_lab.data.calendars import rebalance_dates


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    weights: pd.DataFrame       # lagged weights actually held each period
    asset_returns: pd.DataFrame
    returns: pd.Series          # net portfolio returns (after costs + funding)
    gross_returns: pd.Series
    turnover: pd.Series         # per-period one-way turnover (half-sum of |Δw|)
    costs: pd.Series            # commission + slippage drag, per period (fraction)
    funding_costs: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64")
    )
    meta: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        """Wealth index starting at 1 from net returns."""
        return cumulative_returns(self.returns)


def _bucket_funding_to_bars(
    funding: pd.DataFrame,
    prices_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map funding-event rows onto the bars in ``prices_index`` that contain them.

    Each funding timestamp is floored to the largest ``prices_index`` entry
    that is ``<=`` the funding timestamp. Multiple funding events landing in
    the same bar are summed (extremely rare in practice). Funding events
    that pre-date the first bar are dropped. Returns a DataFrame with the
    same columns as ``funding`` and index ``prices_index``; bars with no
    funding event have NaN — callers should treat those as 0.
    """
    if funding is None or funding.empty:
        cols = funding.columns if funding is not None else []
        return pd.DataFrame(index=prices_index, columns=cols, dtype="float64")
    # searchsorted preserves tz of prices_index, unlike Series.asof().values
    positions = prices_index.searchsorted(funding.index, side="right") - 1
    valid_mask = positions >= 0
    if not valid_mask.any():
        return pd.DataFrame(index=prices_index, columns=funding.columns, dtype="float64")
    valid_positions = positions[valid_mask]
    funding_in = funding[valid_mask]
    bar_keys = prices_index[valid_positions]
    grouped = funding_in.groupby(bar_keys).sum(min_count=1)
    return grouped.reindex(prices_index)


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    rebalance: str | None = None,
    costs_bps: float = 1.0,
    slippage_bps: float | dict[str, float] = 2.0,
    funding: pd.DataFrame | None = None,
    funding_basis: Literal["held"] = "held",
    bars_per_year: int = 252,
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
        Either a scalar (applied uniformly across symbols) or a ``dict``
        mapping symbol → bps. Symbols absent from the dict get 0 bps.
    funding : optional wide DataFrame of funding rates (decimal fraction;
        e.g. 0.0001 = 1 bp). Index = funding timestamps (irregular, e.g.
        8h cadence). Columns = subset of ``prices.columns``.
    funding_basis : how to charge funding. ``"held"`` (default) = current
        held weight × rate; positive rate is a cost for longs and credit
        for shorts.
    bars_per_year : annualization base for downstream metric callers; stored
        in ``BacktestResult.meta``. Use 252 (daily), 365 * 24 * 12 (5min),
        365 * 24 * 60 (1min), etc.

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
    delta = (held - prev).abs()
    turn = delta.sum(axis=1) / 2.0
    if isinstance(slippage_bps, dict):
        # Per-symbol slippage: charge each symbol's one-way turnover at its own
        # (commission + slip) bps; sum across symbols for the bar-level cost.
        slip_vec = pd.Series(
            {c: float(slippage_bps.get(c, 0.0)) for c in held.columns}
        )
        per_symbol_bps = slip_vec + costs_bps
        cost = (delta / 2.0 * per_symbol_bps).sum(axis=1) / 10_000.0
    else:
        cost = turn * (costs_bps + slippage_bps) / 10_000.0

    # Funding (perp). Long pays positive rate; short receives. Cost per bar =
    # sum over symbols of held_weight * funding_rate at the bar containing the
    # funding event. NaN treated as 0.
    if funding is not None and not funding.empty:
        common_cols = [c for c in held.columns if c in funding.columns]
        f_aligned = _bucket_funding_to_bars(funding[common_cols], prices.index)
        funding_cost = (held[common_cols] * f_aligned.fillna(0.0)).sum(axis=1)
    else:
        funding_cost = pd.Series(0.0, index=prices.index)

    net = gross - cost - funding_cost

    meta = {
        "bars_per_year": bars_per_year,
        "funding_basis": funding_basis,
        "has_funding": funding is not None and not funding.empty,
    }

    return BacktestResult(
        weights=held,
        asset_returns=asset_rets,
        returns=net,
        gross_returns=gross,
        turnover=turn,
        costs=cost,
        funding_costs=funding_cost,
        meta=meta,
    )
