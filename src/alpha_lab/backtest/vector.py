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

import numpy as np
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


@dataclass
class DriftBacktestResult:
    """Outputs from the long-only, drift-aware close-to-close engine."""

    weights: pd.DataFrame
    pre_trade_weights: pd.DataFrame
    target_weights: pd.DataFrame
    asset_returns: pd.DataFrame
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    traded_notional: pd.Series
    costs: pd.Series
    cash_returns: pd.Series
    trade_flags: pd.Series
    decision_to_trade: pd.Series
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


def run_drift_backtest(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    trading_bps: float = 5.0,
    execution_delay_bars: int = 1,
    rebalance_threshold: float = 0.0,
    cash_returns: pd.Series | None = None,
    cash_column: str = "CASH",
    bars_per_year: int = 252,
) -> DriftBacktestResult:
    """Backtest sparse long-only targets with next-close execution and drift.

    A target formed at close ``t`` trades at close ``t + execution_delay_bars``.
    The new position therefore first earns the following close-to-close return.
    Between trades, asset shares are unchanged and weights drift with prices.
    Trading costs are charged on actual non-cash buy plus sell notional.
    """
    if prices.empty:
        raise ValueError("prices must not be empty")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices index must be sorted and unique")
    numeric_prices = prices.astype(float)
    if not np.isfinite(numeric_prices.to_numpy()).all() or (numeric_prices <= 0.0).any().any():
        raise ValueError("prices must be finite and strictly positive")
    if execution_delay_bars < 1:
        raise ValueError("execution_delay_bars must be >= 1")
    if trading_bps < 0:
        raise ValueError("trading_bps must be >= 0")
    if rebalance_threshold < 0:
        raise ValueError("rebalance_threshold must be >= 0")
    if cash_column in prices.columns:
        raise ValueError(f"prices must not contain reserved cash column {cash_column!r}")

    asset_columns = prices.columns
    unknown = sorted(set(target_weights.columns) - set(asset_columns) - {cash_column})
    if unknown:
        raise ValueError(f"target weights reference missing price columns: {unknown}")
    if not target_weights.index.isin(prices.index).all():
        raise ValueError("every target date must be an actual price-bar date")

    targets = target_weights.reindex(columns=asset_columns, fill_value=0.0).astype(float)
    if targets.isna().any().any():
        raise ValueError("target weights must not contain NaN")
    if (targets < -1e-12).any().any():
        raise ValueError("target weights must be long-only")

    if cash_column in target_weights.columns:
        cash_target = target_weights[cash_column].astype(float)
    else:
        cash_target = 1.0 - targets.sum(axis=1)
    full_targets = targets.copy()
    full_targets[cash_column] = cash_target
    sums = full_targets.sum(axis=1)
    if (full_targets < -1e-10).any().any() or not np.allclose(sums, 1.0, atol=1e-10):
        raise ValueError("target weights must be non-negative and sum to 1 including cash")
    full_targets[full_targets.abs() < 1e-12] = 0.0

    cash = (
        pd.Series(0.0, index=prices.index, name=cash_column)
        if cash_returns is None
        else cash_returns.reindex(prices.index).fillna(0.0).astype(float).rename(cash_column)
    )
    if (cash <= -1.0).any():
        raise ValueError("cash returns must be greater than -100%")

    execution_targets: dict[pd.Timestamp, pd.Series] = {}
    decision_to_trade: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date, target in full_targets.iterrows():
        decision_pos = prices.index.get_loc(decision_date)
        trade_pos = decision_pos + execution_delay_bars
        if trade_pos >= len(prices.index):
            continue
        trade_date = prices.index[trade_pos]
        execution_targets[trade_date] = target
        decision_to_trade[decision_date] = trade_date

    n_assets = len(asset_columns)
    shares = np.zeros(n_assets, dtype=float)
    cash_value = 1.0
    previous_value = 1.0
    cost_rate = trading_bps / 10_000.0
    rows: list[dict[str, object]] = []

    for date, price_row in numeric_prices.iterrows():
        price_values = price_row.to_numpy()
        asset_values = shares * price_values
        cash_value *= 1.0 + float(cash.loc[date])
        pre_value = float(asset_values.sum() + cash_value)
        pre_weights = np.append(asset_values, cash_value) / pre_value

        scheduled = execution_targets.get(date)
        traded_fraction = 0.0
        one_way_turnover = 0.0
        cost_value = 0.0
        traded = False
        scheduled_values = np.full(n_assets + 1, np.nan)

        if scheduled is not None:
            scheduled_values = scheduled.reindex([*asset_columns, cash_column]).to_numpy(dtype=float)
            deviation = float(np.max(np.abs(scheduled_values - pre_weights)))
            if deviation > rebalance_threshold + 1e-12:
                target_assets = scheduled_values[:n_assets]
                cost_value = _solve_trade_cost(
                    pre_value,
                    asset_values,
                    target_assets,
                    cost_rate,
                )
                post_value = pre_value - cost_value
                desired_asset_values = target_assets * post_value
                traded_fraction = float(np.abs(desired_asset_values - asset_values).sum() / pre_value)
                desired_full_values = scheduled_values * post_value
                one_way_turnover = float(
                    np.abs(desired_full_values - np.append(asset_values, cash_value)).sum()
                    / (2.0 * pre_value)
                )
                shares = desired_asset_values / price_values
                cash_value = float(scheduled_values[-1] * post_value)
                traded = True
            else:
                post_value = pre_value
        else:
            post_value = pre_value

        post_asset_values = shares * price_values
        post_weights = np.append(post_asset_values, cash_value) / post_value
        rows.append(
            {
                "date": date,
                "return": post_value / previous_value - 1.0,
                "gross_return": pre_value / previous_value - 1.0,
                "turnover": one_way_turnover,
                "traded_notional": traded_fraction,
                "cost": cost_value / previous_value,
                "trade_flag": traded,
                "pre_weights": pre_weights,
                "post_weights": post_weights,
                "target_weights": scheduled_values,
            }
        )
        previous_value = post_value

    result = pd.DataFrame(rows).set_index("date")
    all_columns = [*asset_columns, cash_column]
    pre_frame = pd.DataFrame(result.pop("pre_weights").tolist(), index=prices.index, columns=all_columns)
    post_frame = pd.DataFrame(result.pop("post_weights").tolist(), index=prices.index, columns=all_columns)
    target_frame = pd.DataFrame(result.pop("target_weights").tolist(), index=prices.index, columns=all_columns)

    return DriftBacktestResult(
        weights=post_frame,
        pre_trade_weights=pre_frame,
        target_weights=target_frame,
        asset_returns=numeric_prices.pct_change().fillna(0.0),
        returns=result["return"].rename("returns"),
        gross_returns=result["gross_return"].rename("gross_returns"),
        turnover=result["turnover"].rename("turnover"),
        traded_notional=result["traded_notional"].rename("traded_notional"),
        costs=result["cost"].rename("costs"),
        cash_returns=cash,
        trade_flags=result["trade_flag"].astype(bool).rename("trade_flags"),
        decision_to_trade=pd.Series(decision_to_trade, name="trade_date"),
        meta={
            "bars_per_year": bars_per_year,
            "execution_delay_bars": execution_delay_bars,
            "trading_bps": trading_bps,
            "rebalance_threshold": rebalance_threshold,
            "cash_column": cash_column,
        },
    )


def _solve_trade_cost(
    portfolio_value: float,
    current_asset_values: np.ndarray,
    target_asset_weights: np.ndarray,
    cost_rate: float,
) -> float:
    """Solve cost = rate * actual post-cost non-cash traded notional."""
    if cost_rate == 0.0:
        return 0.0
    cost = cost_rate * float(
        np.abs(target_asset_weights * portfolio_value - current_asset_values).sum()
    )
    for _ in range(20):
        updated = cost_rate * float(
            np.abs(target_asset_weights * (portfolio_value - cost) - current_asset_values).sum()
        )
        if abs(updated - cost) <= 1e-14 * max(portfolio_value, 1.0):
            return updated
        cost = updated
    return cost
