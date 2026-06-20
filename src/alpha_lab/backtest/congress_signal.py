"""Congressional-flow signals (research-plan Angles A & C) as ETF target weights.

Pure signal construction — no I/O. Inputs: a normalized PTR long frame (from
:mod:`alpha_lab.data.loaders.congress`), a ticker→GICS-sector map (from
:mod:`alpha_lab.data.congress_universe`), and a trading-day index. Outputs: leak-safe
weight panels for :func:`alpha_lab.backtest.vector.run_backtest`.

Point-in-time rule
------------------
Every trade is attributed to its **filing_date** (the public date), then bucketed onto
the first trading day ``>=`` filing_date — i.e. the first day the disclosure is
actionable at the close. All rolling windows are trailing, so any value at day *t* uses
only disclosures visible by *t*. (The backtester additionally lags weights one day.)

Angle A (core) — sector net-flow tilt
    Net signed log-mid $ flow per GICS sector over a rolling window → trailing
    time-series z-score (de-means each sector's own history) → long the top-N /
    short the bottom-N sector ETFs (dollar-neutral, which strips out market beta —
    the plan's key to beating a naive NANC-style long-only copy).

Angle C — aggregate / party macro tilt
    Aggregate congressional net flow as a risk-appetite proxy → z-score → growth-vs-
    small tilt (long QQQ / short IWM when Congress is net-accumulating, and the
    reverse when net-distributing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.backtest.sector_momentum import top_bottom_view_weights
from alpha_lab.data.congress_universe import gics_sectors, sector_etf_map

UNMAPPED = "Unknown"


# --------------------------------------------------------------------------------------
# Shared: bucket calendar-dated flow onto the trading-day grid (next-trading-day rule)
# --------------------------------------------------------------------------------------
def bucket_onto_grid(daily_calendar: pd.DataFrame | pd.Series, trading_index: pd.DatetimeIndex):
    """Re-bin flow keyed by calendar (filing) date onto a trading-day index.

    All flow with ``filing_date`` in ``(prev_trading_day, t]`` lands on trading day
    ``t`` — the first session on which it is actionable. Implemented as
    cumulative-sum → as-of sample at the grid → first difference, so weekend/holiday
    filings roll forward to the next session. Days before the first filing are 0.
    """
    cal = daily_calendar.sort_index()
    cum = cal.cumsum()
    asof = cum.reindex(cum.index.union(trading_index)).ffill().reindex(trading_index).fillna(0.0)
    on_grid = asof.diff()
    on_grid.iloc[0] = asof.iloc[0]
    return on_grid


# --------------------------------------------------------------------------------------
# Angle A — sector net-flow tilt
# --------------------------------------------------------------------------------------
def sector_net_flow(
    trades: pd.DataFrame,
    sector_of: pd.Series,
    trading_index: pd.DatetimeIndex,
    *,
    window: int = 63,
    value: str = "amount_logmid",
) -> pd.DataFrame:
    """Rolling net signed $ flow per GICS sector on the trading-day grid.

    Parameters
    ----------
    trades : normalized PTR long frame (needs ``filing_date``, ``ticker`` and
        ``value``; ``amount_logmid`` is already sign×magnitude, so summing it nets
        buys against sells).
    sector_of : Series ticker → GICS sector (``"Unknown"`` dropped).
    trading_index : the price/backtest DatetimeIndex.
    window : rolling lookback in **trading days** (≈63 ≈ 3 months, matched to the
        45-day disclosure lag + persistence the plan targets).
    value : flow column to sum (default ``amount_logmid``).

    Returns
    -------
    Wide DataFrame, index=trading_index, columns=the 11 GICS sectors.
    """
    sectors = gics_sectors()
    df = trades.dropna(subset=["filing_date"]).copy()
    df["sector"] = df["ticker"].map(sector_of)
    df = df[df["sector"].isin(sectors)]

    daily = (
        df.assign(d=df["filing_date"].dt.normalize())
        .groupby(["d", "sector"])[value]
        .sum()
        .unstack("sector")
        .sort_index()
        .reindex(columns=sectors)
        .fillna(0.0)
    )
    if daily.empty:
        return pd.DataFrame(0.0, index=trading_index, columns=sectors)
    on_grid = bucket_onto_grid(daily, trading_index)
    return on_grid.rolling(window, min_periods=1).sum()


def sector_flow_zscore(
    net_flow: pd.DataFrame, *, z_window: int = 252, min_periods: int = 63
) -> pd.DataFrame:
    """Trailing time-series z-score of each sector's net flow (rolling, leak-safe).

    De-means each sector against its own recent history so the cross-section
    compares *abnormal* accumulation, not structurally higher-volume sectors.
    """
    mean = net_flow.rolling(z_window, min_periods=min_periods).mean()
    std = net_flow.rolling(z_window, min_periods=min_periods).std()
    return (net_flow - mean) / std.replace(0.0, np.nan)


def sector_tilt_weights(
    zscore: pd.DataFrame,
    *,
    top_n: int = 3,
    bottom_n: int = 3,
    long_gross: float = 1.0,
    short_gross: float = 1.0,
) -> pd.DataFrame:
    """Long top-N / short bottom-N sectors, returned in **sector-ETF** columns.

    Dollar-neutral when ``long_gross == short_gross`` — isolating sector selection
    from market beta. Columns are renamed from GICS sector → representative ETF
    (XLK, XLE, …) via :func:`sector_etf_map`.
    """
    view = top_bottom_view_weights(
        zscore, top_n=top_n, bottom_n=bottom_n, long_gross=long_gross, short_gross=short_gross
    )
    return view.rename(columns=sector_etf_map())


# --------------------------------------------------------------------------------------
# Angle C — aggregate / party macro tilt
# --------------------------------------------------------------------------------------
def aggregate_net_flow(
    trades: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    window: int = 63,
    by: str | None = None,
    value: str = "amount_logmid",
) -> pd.DataFrame | pd.Series:
    """Rolling aggregate net signed $ flow across all members (optionally split ``by``).

    With ``by=None`` returns a Series (total Congress risk-appetite proxy); with
    ``by="party"`` returns a wide DataFrame (one column per party).
    """
    df = trades.dropna(subset=["filing_date"]).copy()
    df["d"] = df["filing_date"].dt.normalize()
    if by is None:
        daily = df.groupby("d")[value].sum().sort_index()
        return bucket_onto_grid(daily, trading_index).rolling(window, min_periods=1).sum()
    daily = df.groupby(["d", by])[value].sum().unstack(by).sort_index().fillna(0.0)
    return bucket_onto_grid(daily, trading_index).rolling(window, min_periods=1).sum()


def risk_on_weights(
    agg_flow: pd.Series,
    *,
    z_window: int = 252,
    min_periods: int = 63,
    long: str = "QQQ",
    short: str = "IWM",
    gross: float = 0.5,
) -> pd.DataFrame:
    """Growth-vs-small tilt from the aggregate-flow z-score (dollar-neutral).

    Risk-on (z>0, Congress net-accumulating) → long ``long`` / short ``short``;
    risk-off flips the sign. Magnitude is fixed (``±gross`` each leg) — a clean,
    falsifiable expression rather than a tuned scaler.
    """
    mean = agg_flow.rolling(z_window, min_periods=min_periods).mean()
    std = agg_flow.rolling(z_window, min_periods=min_periods).std()
    z = (agg_flow - mean) / std.replace(0.0, np.nan)
    side = z.apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
    return pd.DataFrame({long: gross * side, short: -gross * side}).fillna(0.0)
