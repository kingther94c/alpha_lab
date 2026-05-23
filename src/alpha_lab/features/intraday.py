"""Intraday feature builders.

Conventions
-----------
- LEAK-SAFE by construction: the value at row ``t`` depends only on data at
  rows ``<= t``. **No ``.shift(-k)`` appears anywhere in this module.**
  Forward shifts belong in labels / target construction (see
  :func:`alpha_lab.data.align.forward_returns`), never in features.
- Input shapes:
  - Single-symbol features take Series (close, high, low, volume).
  - Cross-asset features take a wide DataFrame of close prices.
  - Index must be a sorted ``DatetimeIndex``.
- Output: same index as input; Series for single-symbol features,
  DataFrame for cross-asset / multi-component features (MACD).

All windows are in BARS, not wall-clock — caller decides whether to feed 1m,
5m, or coarser data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- Returns & volatility --------------------------------------------------

def log_return(close: pd.Series, window: int = 1) -> pd.Series:
    """``log(close[t] / close[t - window])``. First ``window`` rows are NaN."""
    return np.log(close / close.shift(window))


def realized_vol_close(close: pd.Series, window: int = 60) -> pd.Series:
    """Rolling stdev of 1-bar log returns over ``window`` bars."""
    r = np.log(close / close.shift(1))
    return r.rolling(window).std()


def realized_vol_parkinson(
    high: pd.Series, low: pd.Series, window: int = 60,
) -> pd.Series:
    """Parkinson high-low realized vol over ``window`` bars."""
    log_hl = np.log(high / low)
    return np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2)))


def realized_vol_garman_klass(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Garman–Klass OHLC realized vol estimator over ``window`` bars."""
    hl = 0.5 * np.log(high / low) ** 2
    co = (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    return np.sqrt((hl - co).rolling(window).mean())


# --- Volume / liquidity ----------------------------------------------------

def volume_zscore(volume: pd.Series, window: int = 240) -> pd.Series:
    """Z-score of ``volume[t]`` vs rolling mean / std over ``window`` bars."""
    m = volume.rolling(window).mean()
    s = volume.rolling(window).std()
    return (volume - m) / s


def rolling_taker_imbalance(
    taker_buy_base: pd.Series, volume: pd.Series, window: int = 60,
) -> pd.Series:
    """Rolling mean of (taker_buy_base / volume - 0.5), an aggressive-buy ratio
    centered at 0. Values > 0 mean taker buying is dominant."""
    ratio = taker_buy_base / volume - 0.5
    return ratio.rolling(window).mean()


# --- Trend / mean reversion ------------------------------------------------

def ma_slope(close: pd.Series, window: int = 60) -> pd.Series:
    """First difference of the rolling mean — proxy for MA slope."""
    ma = close.rolling(window).mean()
    return ma - ma.shift(1)


def distance_from_ma(close: pd.Series, window: int = 60) -> pd.Series:
    """Pct distance of ``close`` from its rolling MA: ``(close - MA) / MA``."""
    ma = close.rolling(window).mean()
    return (close - ma) / ma


def breakout_distance(close: pd.Series, window: int = 240) -> pd.Series:
    """Position of ``close[t]`` within the rolling ``[t-window+1, t]`` range:
    1.0 = at the rolling high, 0.0 = at the rolling low, 0.5 = midpoint.

    Returns 0.5 when the rolling range collapses to a point.
    """
    def _pos(x: np.ndarray) -> float:
        lo, hi = x.min(), x.max()
        return 0.5 if hi == lo else (x[-1] - lo) / (hi - lo)
    return close.rolling(window).apply(_pos, raw=True)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14,
) -> pd.Series:
    """Average True Range over ``window`` bars (Wilder smoothing replaced by SMA)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI on log-style price deltas over ``window`` bars."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, histogram. Columns: ``macd, signal, hist``."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


def bollinger_pct_b(
    close: pd.Series, window: int = 20, n_std: float = 2.0,
) -> pd.Series:
    """Bollinger %B: where ``close`` sits between the bands ([0, 1] inside)."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    band = upper - lower
    return (close - lower) / band.replace(0, np.nan)


def donchian_position(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20,
) -> pd.Series:
    """Donchian channel position of ``close`` within the rolling high-low."""
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    band = upper - lower
    return (close - lower) / band.replace(0, np.nan)


# --- Calendar / regime dummies ---------------------------------------------

def time_of_day_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Hours since UTC midnight as a float in ``[0, 24)``."""
    return pd.Series(
        index.hour + index.minute / 60.0 + index.second / 3600.0,
        index=index, name="hour_of_day",
    )


def day_of_week(index: pd.DatetimeIndex) -> pd.Series:
    """``Monday=0 .. Sunday=6``."""
    return pd.Series(index.dayofweek, index=index, name="day_of_week")


# --- Cross-asset features --------------------------------------------------

def relative_strength(panel: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    """Per-asset cumulative log return over ``window`` bars, demeaned across
    the panel. > 0 means the asset is outperforming the cross-sectional mean.
    """
    r = np.log(panel / panel.shift(1))
    cum = r.rolling(window).sum()
    return cum.sub(cum.mean(axis=1), axis=0)


def spread_zscore(a: pd.Series, b: pd.Series, window: int = 240) -> pd.Series:
    """Z-score of ``log(a / b)`` over a rolling ``window``."""
    spread = np.log(a / b)
    m = spread.rolling(window).mean()
    s = spread.rolling(window).std()
    return (spread - m) / s


def rolling_beta_residual(
    y: pd.Series, x: pd.Series, window: int = 240,
) -> pd.DataFrame:
    """Rolling OLS beta of ``y`` on ``x`` over ``window`` bars (log returns).

    Returns a DataFrame with columns ``beta`` (rolling slope) and ``residual``
    (latest-bar residual of the rolling regression). Indexed like ``y``.
    """
    ry = np.log(y / y.shift(1))
    rx = np.log(x / x.shift(1))
    cov = ry.rolling(window).cov(rx)
    var_x = rx.rolling(window).var()
    beta = cov / var_x.replace(0, np.nan)
    mean_y = ry.rolling(window).mean()
    mean_x = rx.rolling(window).mean()
    residual = (ry - mean_y) - beta * (rx - mean_x)
    return pd.DataFrame({"beta": beta, "residual": residual})


# --- Funding-rate features (perp only) -------------------------------------

def funding_zscore(funding: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Z-score of recent funding rates over ``window`` events (NOT bars).

    Input: wide funding panel indexed by funding timestamp (8h cadence on
    USD-M perp). Output: same shape, z-scored per symbol.
    """
    m = funding.rolling(window).mean()
    s = funding.rolling(window).std()
    return (funding - m) / s


def funding_cumulative(funding: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Trailing ``window``-event sum of funding rates per symbol.

    Useful as a "carry" proxy — large positive sum means longs have been
    paying for a while.
    """
    return funding.rolling(window).sum()
