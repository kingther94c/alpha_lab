"""Synthetic protective-collar research benchmark.

This module models a Cboe-CLL-like overlay when an executable historical option
chain is unavailable.  It is deliberately labelled synthetic: VIX supplies an
ATM volatility proxy, configurable wing/skew buffers approximate strike IV, and
entry haircuts approximate bid/ask execution.  It is not an order router.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from alpha_lab.backtest.put_write import black_scholes_put


@dataclass(frozen=True)
class SyntheticCollarConfig:
    """Frozen assumptions for a long-underlying, long-put, short-call collar."""

    put_otm: float = 0.05
    call_otm: float = 0.10
    put_iv_buffer: float = 0.04
    call_iv_haircut: float = 0.02
    realized_vol_buffer: float = 0.02
    minimum_call_iv: float = 0.08
    put_ask_markup: float = 0.10
    call_bid_haircut: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.put_otm < 1.0:
            raise ValueError("put_otm must be in (0, 1)")
        if self.call_otm <= 0.0:
            raise ValueError("call_otm must be positive")
        if min(self.put_iv_buffer, self.call_iv_haircut, self.realized_vol_buffer) < 0.0:
            raise ValueError("volatility buffers must be non-negative")
        if self.minimum_call_iv <= 0.0:
            raise ValueError("minimum_call_iv must be positive")
        if not 0.0 <= self.put_ask_markup < 1.0:
            raise ValueError("put_ask_markup must be in [0, 1)")
        if not 0.0 <= self.call_bid_haircut < 1.0:
            raise ValueError("call_bid_haircut must be in [0, 1)")


@dataclass(frozen=True)
class SyntheticCollarResult:
    """Daily synthetic collar returns, equity, and marked components."""

    returns: pd.Series
    equity: pd.Series
    diagnostics: pd.DataFrame
    config: SyntheticCollarConfig


@dataclass(frozen=True)
class SyntheticOptionOverlayConfig:
    """Frozen strike, IV-skew, and execution assumptions for a SPY overlay."""

    long_put_otm: float = 0.05
    short_put_otm: float | None = None
    call_otm: float | None = None
    long_put_iv_buffer: float = 0.04
    short_put_iv_buffer: float = 0.08
    call_iv_haircut: float = 0.02
    realized_vol_buffer: float = 0.02
    minimum_call_iv: float = 0.08
    long_option_ask_markup: float = 0.10
    short_option_bid_haircut: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.long_put_otm < 1.0:
            raise ValueError("long_put_otm must be in (0, 1)")
        if self.short_put_otm is not None and not self.long_put_otm < self.short_put_otm < 1.0:
            raise ValueError("short_put_otm must be deeper OTM than long_put_otm")
        if self.call_otm is not None and self.call_otm <= 0.0:
            raise ValueError("call_otm must be positive")
        buffers = [
            self.long_put_iv_buffer,
            self.short_put_iv_buffer,
            self.call_iv_haircut,
            self.realized_vol_buffer,
        ]
        if min(buffers) < 0.0:
            raise ValueError("volatility buffers must be non-negative")
        if self.minimum_call_iv <= 0.0:
            raise ValueError("minimum_call_iv must be positive")
        if not 0.0 <= self.long_option_ask_markup < 1.0:
            raise ValueError("long_option_ask_markup must be in [0, 1)")
        if not 0.0 <= self.short_option_bid_haircut < 1.0:
            raise ValueError("short_option_bid_haircut must be in [0, 1)")


@dataclass(frozen=True)
class SyntheticOptionOverlayResult:
    """Daily returns, equity, and marked components for a base-plus-option overlay."""

    returns: pd.Series
    equity: pd.Series
    diagnostics: pd.DataFrame
    config: SyntheticOptionOverlayConfig


@dataclass
class _OptionPosition:
    strike: float
    expiry: pd.Timestamp
    contracts: float


def black_scholes_call(
    spot: float,
    strike: float,
    years_to_expiry: float,
    annual_rate: float,
    annual_vol: float,
    *,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return European call value and spot delta."""
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if annual_vol <= 0.0:
        raise ValueError("annual_vol must be positive")
    if years_to_expiry <= 0.0:
        intrinsic = max(spot - strike, 0.0)
        if spot > strike:
            delta = 1.0
        elif spot < strike:
            delta = 0.0
        else:
            delta = 0.5
        return intrinsic, delta

    sigma_sqrt_t = annual_vol * sqrt(years_to_expiry)
    d1 = (
        log(spot / strike) + (annual_rate - dividend_yield + 0.5 * annual_vol**2) * years_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    price = spot * exp(-dividend_yield * years_to_expiry) * norm.cdf(d1) - strike * exp(
        -annual_rate * years_to_expiry
    ) * norm.cdf(d2)
    delta = exp(-dividend_yield * years_to_expiry) * norm.cdf(d1)
    return float(price), float(delta)


def third_friday_roll_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each calendar third Friday to the last available bar on or before it."""
    if index.empty:
        return index
    normalized = pd.DatetimeIndex(index).tz_localize(None).normalize()
    start = normalized.min().to_period("M")
    end = normalized.max().to_period("M")
    mapped: list[pd.Timestamp] = []
    for period in pd.period_range(start, end, freq="M"):
        first = period.start_time
        first_friday_offset = (4 - first.weekday()) % 7
        third_friday = first + pd.Timedelta(days=first_friday_offset + 14)
        month_bars = normalized[
            (normalized.to_period("M") == period) & (normalized <= third_friday)
        ]
        if len(month_bars):
            mapped.append(month_bars[-1])
    return pd.DatetimeIndex(mapped).drop_duplicates()


def _next_calendar_third_friday(date: pd.Timestamp, *, quarterly: bool) -> pd.Timestamp:
    period = date.to_period("M") + 1
    while quarterly and period.month not in {3, 6, 9, 12}:
        period += 1
    first = period.start_time
    first_friday_offset = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=first_friday_offset + 14)


def _mark_put(
    position: _OptionPosition | None,
    *,
    spot: float,
    date: pd.Timestamp,
    annual_rate: float,
    annual_vol: float,
) -> float:
    if position is None:
        return 0.0
    years = max((position.expiry - date).days / 365.25, 0.0)
    value, _ = black_scholes_put(
        spot,
        position.strike,
        years,
        annual_rate,
        annual_vol,
    )
    return value * position.contracts


def _mark_call(
    position: _OptionPosition | None,
    *,
    spot: float,
    date: pd.Timestamp,
    annual_rate: float,
    annual_vol: float,
) -> float:
    if position is None:
        return 0.0
    years = max((position.expiry - date).days / 365.25, 0.0)
    value, _ = black_scholes_call(
        spot,
        position.strike,
        years,
        annual_rate,
        annual_vol,
    )
    return value * position.contracts


def run_synthetic_collar(
    adjusted_spy: pd.Series,
    option_spot: pd.Series,
    vix: pd.Series,
    cash_returns: pd.Series,
    annual_rates: pd.Series,
    *,
    config: SyntheticCollarConfig = SyntheticCollarConfig(),  # noqa: B008
    initial_equity: float = 1.0,
) -> SyntheticCollarResult:
    """Run a CLL-like synthetic collar with monthly calls and quarterly puts.

    The strategy starts on the first quarterly third-Friday roll for which all
    inputs and a 21-session realised-volatility estimate exist.  Option positions
    are marked daily.  Rolls use the same close but earn no pre-roll same-bar P&L;
    the only immediate change is the conservative entry spread haircut.
    """
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    frame = pd.concat(
        [
            adjusted_spy.rename("adjusted_spy"),
            option_spot.rename("option_spot"),
            vix.rename("vix"),
            cash_returns.rename("cash_return"),
            annual_rates.rename("annual_rate"),
        ],
        axis=1,
        sort=False,
    ).sort_index()
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("inputs must have a sorted, unique index")
    if (frame[["adjusted_spy", "option_spot"]].dropna() <= 0.0).any().any():
        raise ValueError("SPY prices must be positive")
    if (frame["vix"].dropna() <= 0.0).any():
        raise ValueError("VIX must be positive")

    spy_returns = frame["adjusted_spy"].pct_change()
    realized_vol = spy_returns.rolling(21, min_periods=21).std() * np.sqrt(252)
    base_iv = pd.concat(
        [
            (frame["vix"] / 100.0).rename("vix"),
            (realized_vol + config.realized_vol_buffer).rename("rv"),
        ],
        axis=1,
    ).max(axis=1)
    put_iv = base_iv + config.put_iv_buffer
    call_iv = (base_iv - config.call_iv_haircut).clip(lower=config.minimum_call_iv)
    rolls = third_friday_roll_dates(frame.index)
    quarterly_rolls = rolls[rolls.month.isin([3, 6, 9, 12])]
    eligible_start = frame.dropna().index.min()
    starts = quarterly_rolls[
        (quarterly_rolls >= eligible_start) & put_iv.reindex(quarterly_rolls).notna()
    ]
    if starts.empty:
        raise ValueError("no quarterly roll has sufficient collar inputs")
    start = starts[0]
    frame = frame.loc[start:].ffill().dropna()
    spy_returns = frame["adjusted_spy"].pct_change().fillna(0.0)
    put_iv = put_iv.reindex(frame.index).ffill()
    call_iv = call_iv.reindex(frame.index).ffill()
    roll_set = set(rolls.intersection(frame.index))
    quarterly_set = set(quarterly_rolls.intersection(frame.index))

    stock_value = float(initial_equity)
    cash = 0.0
    put_position: _OptionPosition | None = None
    call_position: _OptionPosition | None = None
    put_value = 0.0
    call_value = 0.0
    previous_equity = float(initial_equity)
    rows: list[dict[str, float | bool]] = []

    for position, (date, row) in enumerate(frame.iterrows()):
        spot = float(row["option_spot"])
        rate = float(row["annual_rate"])
        if position:
            stock_value *= 1.0 + float(spy_returns.loc[date])
            cash *= 1.0 + float(row["cash_return"])
        put_value = _mark_put(
            put_position,
            spot=spot,
            date=date,
            annual_rate=rate,
            annual_vol=float(put_iv.loc[date]),
        )
        call_value = _mark_call(
            call_position,
            spot=spot,
            date=date,
            annual_rate=rate,
            annual_vol=float(call_iv.loc[date]),
        )

        call_roll = date in roll_set
        put_roll = date in quarterly_set
        if call_roll:
            cash -= call_value
            call_position = None
            call_value = 0.0
            if put_roll:
                cash += put_value
                put_position = None
                put_value = 0.0

            equity_before_new = stock_value + cash + put_value
            target_stock_value = equity_before_new
            cash += stock_value - target_stock_value
            stock_value = target_stock_value

            if put_roll:
                put_expiry = _next_calendar_third_friday(date, quarterly=True)
                put_strike = spot * (1.0 - config.put_otm)
                put_contracts = target_stock_value / spot
                put_position = _OptionPosition(put_strike, put_expiry, put_contracts)
                put_value = _mark_put(
                    put_position,
                    spot=spot,
                    date=date,
                    annual_rate=rate,
                    annual_vol=float(put_iv.loc[date]),
                )
                cash -= put_value * (1.0 + config.put_ask_markup)

            call_expiry = _next_calendar_third_friday(date, quarterly=False)
            call_strike = spot * (1.0 + config.call_otm)
            call_contracts = target_stock_value / spot
            call_position = _OptionPosition(call_strike, call_expiry, call_contracts)
            call_value = _mark_call(
                call_position,
                spot=spot,
                date=date,
                annual_rate=rate,
                annual_vol=float(call_iv.loc[date]),
            )
            cash += call_value * (1.0 - config.call_bid_haircut)

        equity = stock_value + cash + put_value - call_value
        rows.append(
            {
                "equity": equity,
                "return": equity / previous_equity - 1.0,
                "stock_value": stock_value,
                "cash": cash,
                "put_value": put_value,
                "call_value": call_value,
                "put_iv": float(put_iv.loc[date]),
                "call_iv": float(call_iv.loc[date]),
                "call_roll": call_roll,
                "put_roll": put_roll,
            }
        )
        previous_equity = equity

    diagnostics = pd.DataFrame(rows, index=frame.index)
    return SyntheticCollarResult(
        returns=diagnostics["return"].rename("synthetic_collar_return"),
        equity=diagnostics["equity"].rename("synthetic_collar_equity"),
        diagnostics=diagnostics,
        config=config,
    )


def run_synthetic_option_overlay(
    base_returns: pd.Series,
    adjusted_spy: pd.Series,
    option_spot: pd.Series,
    vix: pd.Series,
    cash_returns: pd.Series,
    annual_rates: pd.Series,
    put_ratio: pd.Series | float,
    call_ratio: pd.Series | float = 0.0,
    *,
    config: SyntheticOptionOverlayConfig = SyntheticOptionOverlayConfig(),  # noqa: B008
    initial_equity: float = 1.0,
) -> SyntheticOptionOverlayResult:
    """Attach a quarterly SPY put structure and optional monthly call to any base return stream.

    Ratios are fractions of current strategy NAV.  A ratio series is observed at the roll close;
    callers should lag state-dependent ratios before passing them.  The base sleeve is reset to
    100% of NAV whenever an option leg rolls, while option market value is financed through the
    overlay cash account.  Positive and negative overlay cash earn or pay the supplied cash rate.
    """
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    put_ratio_series = _overlay_ratio_series(put_ratio, base_returns.index, "put_ratio")
    call_ratio_series = _overlay_ratio_series(call_ratio, base_returns.index, "call_ratio")
    frame = pd.concat(
        [
            base_returns.rename("base_return"),
            adjusted_spy.rename("adjusted_spy"),
            option_spot.rename("option_spot"),
            vix.rename("vix"),
            cash_returns.rename("cash_return"),
            annual_rates.rename("annual_rate"),
            put_ratio_series,
            call_ratio_series,
        ],
        axis=1,
        sort=False,
    ).sort_index()
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("inputs must have a sorted, unique index")
    if (frame["base_return"].dropna() <= -1.0).any():
        raise ValueError("base returns must be greater than -100%")
    if (frame[["adjusted_spy", "option_spot"]].dropna() <= 0.0).any().any():
        raise ValueError("SPY prices must be positive")
    if (frame["vix"].dropna() <= 0.0).any():
        raise ValueError("VIX must be positive")

    realized_vol = frame["adjusted_spy"].pct_change().rolling(21, min_periods=21).std() * np.sqrt(
        252
    )
    base_iv = pd.concat(
        [
            (frame["vix"] / 100.0).rename("vix"),
            (realized_vol + config.realized_vol_buffer).rename("rv"),
        ],
        axis=1,
    ).max(axis=1)
    long_put_iv = base_iv + config.long_put_iv_buffer
    short_put_iv = base_iv + config.short_put_iv_buffer
    call_iv = (base_iv - config.call_iv_haircut).clip(lower=config.minimum_call_iv)
    rolls = third_friday_roll_dates(frame.index)
    quarterly_rolls = rolls[rolls.month.isin([3, 6, 9, 12])]
    eligible_start = frame.dropna().index.min()
    starts = quarterly_rolls[
        (quarterly_rolls >= eligible_start) & long_put_iv.reindex(quarterly_rolls).notna()
    ]
    if starts.empty:
        raise ValueError("no quarterly roll has sufficient overlay inputs")
    start = starts[0]
    frame = frame.loc[start:].ffill().dropna()
    long_put_iv = long_put_iv.reindex(frame.index).ffill()
    short_put_iv = short_put_iv.reindex(frame.index).ffill()
    call_iv = call_iv.reindex(frame.index).ffill()
    roll_set = set(rolls.intersection(frame.index))
    quarterly_set = set(quarterly_rolls.intersection(frame.index))

    base_value = float(initial_equity)
    overlay_cash = 0.0
    long_put: _OptionPosition | None = None
    short_put: _OptionPosition | None = None
    short_call: _OptionPosition | None = None
    previous_equity = float(initial_equity)
    rows: list[dict[str, float | bool]] = []

    for position, (date, row) in enumerate(frame.iterrows()):
        spot = float(row["option_spot"])
        rate = float(row["annual_rate"])
        if position:
            base_value *= 1.0 + float(row["base_return"])
            overlay_cash *= 1.0 + float(row["cash_return"])

        long_put_value = _mark_put(
            long_put,
            spot=spot,
            date=date,
            annual_rate=rate,
            annual_vol=float(long_put_iv.loc[date]),
        )
        short_put_value = _mark_put(
            short_put,
            spot=spot,
            date=date,
            annual_rate=rate,
            annual_vol=float(short_put_iv.loc[date]),
        )
        short_call_value = _mark_call(
            short_call,
            spot=spot,
            date=date,
            annual_rate=rate,
            annual_vol=float(call_iv.loc[date]),
        )

        put_roll = date in quarterly_set
        call_roll = config.call_otm is not None and date in roll_set
        entry_drag = 0.0
        if put_roll or call_roll:
            if call_roll:
                overlay_cash -= short_call_value
                short_call = None
                short_call_value = 0.0
            if put_roll:
                overlay_cash += long_put_value - short_put_value
                long_put = None
                short_put = None
                long_put_value = 0.0
                short_put_value = 0.0

            equity_before_new = (
                base_value + overlay_cash + long_put_value - short_put_value - short_call_value
            )
            overlay_cash += base_value - equity_before_new
            base_value = equity_before_new

            if put_roll and float(row["put_ratio"]) > 0.0:
                put_expiry = _next_calendar_third_friday(date, quarterly=True)
                contracts = equity_before_new * float(row["put_ratio"]) / spot
                long_put = _OptionPosition(
                    strike=spot * (1.0 - config.long_put_otm),
                    expiry=put_expiry,
                    contracts=contracts,
                )
                long_put_value = _mark_put(
                    long_put,
                    spot=spot,
                    date=date,
                    annual_rate=rate,
                    annual_vol=float(long_put_iv.loc[date]),
                )
                overlay_cash -= long_put_value * (1.0 + config.long_option_ask_markup)
                if config.short_put_otm is not None:
                    short_put = _OptionPosition(
                        strike=spot * (1.0 - config.short_put_otm),
                        expiry=put_expiry,
                        contracts=contracts,
                    )
                    short_put_value = _mark_put(
                        short_put,
                        spot=spot,
                        date=date,
                        annual_rate=rate,
                        annual_vol=float(short_put_iv.loc[date]),
                    )
                    overlay_cash += short_put_value * (1.0 - config.short_option_bid_haircut)

            if call_roll and float(row["call_ratio"]) > 0.0:
                call_expiry = _next_calendar_third_friday(date, quarterly=False)
                short_call = _OptionPosition(
                    strike=spot * (1.0 + float(config.call_otm)),
                    expiry=call_expiry,
                    contracts=equity_before_new * float(row["call_ratio"]) / spot,
                )
                short_call_value = _mark_call(
                    short_call,
                    spot=spot,
                    date=date,
                    annual_rate=rate,
                    annual_vol=float(call_iv.loc[date]),
                )
                overlay_cash += short_call_value * (1.0 - config.short_option_bid_haircut)

            equity_after_new = (
                base_value + overlay_cash + long_put_value - short_put_value - short_call_value
            )
            entry_drag = equity_after_new - equity_before_new

        equity = base_value + overlay_cash + long_put_value - short_put_value - short_call_value
        rows.append(
            {
                "equity": equity,
                "return": equity / previous_equity - 1.0,
                "base_value": base_value,
                "overlay_cash": overlay_cash,
                "long_put_value": long_put_value,
                "short_put_value": short_put_value,
                "short_call_value": short_call_value,
                "overlay_value": (
                    overlay_cash + long_put_value - short_put_value - short_call_value
                ),
                "entry_drag": entry_drag,
                "put_ratio": float(row["put_ratio"]),
                "call_ratio": float(row["call_ratio"]),
                "long_put_iv": float(long_put_iv.loc[date]),
                "short_put_iv": float(short_put_iv.loc[date]),
                "call_iv": float(call_iv.loc[date]),
                "put_roll": put_roll,
                "call_roll": call_roll,
            }
        )
        previous_equity = equity

    diagnostics = pd.DataFrame(rows, index=frame.index)
    return SyntheticOptionOverlayResult(
        returns=diagnostics["return"].rename("synthetic_option_overlay_return"),
        equity=diagnostics["equity"].rename("synthetic_option_overlay_equity"),
        diagnostics=diagnostics,
        config=config,
    )


def _overlay_ratio_series(
    ratio: pd.Series | float,
    index: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    """Align and validate a non-levered option-notional ratio."""
    if isinstance(ratio, pd.Series):
        series = ratio.rename(name).reindex(index)
    else:
        series = pd.Series(float(ratio), index=index, name=name)
    values = series.dropna()
    tolerance = 1e-10
    if ((values < -tolerance) | (values > 1.0 + tolerance)).any():
        raise ValueError(f"{name} must be between 0 and 1")
    return series.clip(lower=0.0, upper=1.0)
