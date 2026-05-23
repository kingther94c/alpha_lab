"""Tests for src/alpha_lab/features/intraday.py.

Focus: LEAK-SAFETY. Every feature must satisfy the truncation invariant —
its value at row t depends only on input rows <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.features import intraday as f


# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.1, n)
    volume = rng.uniform(50, 500, n)
    taker_buy_base = volume * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "taker_buy_base": taker_buy_base},
        index=idx,
    )


@pytest.fixture
def two_asset_close(ohlcv) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = len(ohlcv)
    a = ohlcv["close"].rename("A")
    b = pd.Series(
        100 + np.cumsum(rng.normal(0, 0.05, n)),
        index=ohlcv.index, name="B",
    )
    return pd.concat([a, b], axis=1)


# --- Truncation-invariant (leak-safety) check ------------------------------

def _truncation_safe(feature_fn, *args_at_t, t: int = 200):
    """Helper: verify feature_fn produces the same value at row t whether
    we feed it the full series or the series truncated at t inclusive.
    """
    full = feature_fn(*args_at_t)
    truncated_args = [a.iloc[: t + 1] if isinstance(a, pd.Series) else a for a in args_at_t]
    truncated = feature_fn(*truncated_args)
    if isinstance(full, pd.DataFrame):
        for c in full.columns:
            v_full = full[c].iloc[t]
            v_trunc = truncated[c].iloc[t]
            if pd.isna(v_full) and pd.isna(v_trunc):
                continue
            assert v_full == pytest.approx(v_trunc, rel=1e-9, abs=1e-12), (
                f"Leak: {feature_fn.__name__}[{c}] differs at t={t} "
                f"(full={v_full}, truncated={v_trunc})"
            )
    else:
        v_full = full.iloc[t]
        v_trunc = truncated.iloc[t]
        if pd.isna(v_full) and pd.isna(v_trunc):
            return
        assert v_full == pytest.approx(v_trunc, rel=1e-9, abs=1e-12), (
            f"Leak: {feature_fn.__name__} differs at t={t} "
            f"(full={v_full}, truncated={v_trunc})"
        )


def test_leak_safety_returns_and_vol(ohlcv):
    c = ohlcv["close"]
    h, l, o = ohlcv["high"], ohlcv["low"], ohlcv["open"]
    _truncation_safe(f.log_return, c)
    _truncation_safe(f.realized_vol_close, c)
    _truncation_safe(f.realized_vol_parkinson, h, l)
    _truncation_safe(f.realized_vol_garman_klass, o, h, l, c)


def test_leak_safety_volume(ohlcv):
    _truncation_safe(f.volume_zscore, ohlcv["volume"])
    _truncation_safe(f.rolling_taker_imbalance, ohlcv["taker_buy_base"], ohlcv["volume"])


def test_leak_safety_trend_mr(ohlcv):
    c = ohlcv["close"]
    h, l = ohlcv["high"], ohlcv["low"]
    _truncation_safe(f.ma_slope, c)
    _truncation_safe(f.distance_from_ma, c)
    _truncation_safe(f.breakout_distance, c)
    _truncation_safe(f.atr, h, l, c)
    _truncation_safe(f.rsi, c)
    _truncation_safe(f.macd, c)
    _truncation_safe(f.bollinger_pct_b, c)
    _truncation_safe(f.donchian_position, h, l, c)


def test_leak_safety_cross_asset(two_asset_close):
    a, b = two_asset_close["A"], two_asset_close["B"]
    _truncation_safe(f.spread_zscore, a, b)
    _truncation_safe(f.rolling_beta_residual, a, b)
    # relative_strength takes a DataFrame; check at one column
    full = f.relative_strength(two_asset_close)
    truncated = f.relative_strength(two_asset_close.iloc[:201])
    for col in full.columns:
        v_full = full[col].iloc[200]
        v_trunc = truncated[col].iloc[200]
        if pd.isna(v_full) and pd.isna(v_trunc):
            continue
        assert v_full == pytest.approx(v_trunc, rel=1e-9, abs=1e-12)


# --- Specific value-level sanity checks -----------------------------------

def test_breakout_distance_at_rolling_high(ohlcv):
    """At a bar that is the rolling-window high, breakout_distance should be 1.0."""
    c = ohlcv["close"].copy()
    # Force a known high at index 50 (window=20)
    c.iloc[30:50] = 100.0
    c.iloc[50] = 200.0
    bd = f.breakout_distance(c, window=20)
    assert bd.iloc[50] == pytest.approx(1.0)


def test_distance_from_ma_zero_when_flat(ohlcv):
    """A constant series has 0 distance from its MA."""
    c = pd.Series(50.0, index=ohlcv.index)
    out = f.distance_from_ma(c, window=20)
    # After warmup, all values are 0
    assert out.iloc[100:].abs().max() < 1e-12


def test_rsi_bounds(ohlcv):
    """RSI must lie in [0, 100]."""
    r = f.rsi(ohlcv["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_components_align(ohlcv):
    mdf = f.macd(ohlcv["close"])
    assert list(mdf.columns) == ["macd", "signal", "hist"]
    # hist == macd - signal exactly (modulo NaN)
    diff = (mdf["hist"] - (mdf["macd"] - mdf["signal"])).dropna()
    assert diff.abs().max() < 1e-12


def test_time_of_day_and_dow():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01 00:30", tz="UTC"),
         pd.Timestamp("2024-01-01 12:45", tz="UTC"),
         pd.Timestamp("2024-01-07 23:00", tz="UTC")]  # Sunday
    )
    h = f.time_of_day_hours(idx)
    assert h.iloc[0] == pytest.approx(0.5)
    assert h.iloc[1] == pytest.approx(12.75)
    assert h.iloc[2] == pytest.approx(23.0)
    d = f.day_of_week(idx)
    assert d.iloc[2] == 6  # Sunday


def test_relative_strength_demeaned(two_asset_close):
    rs = f.relative_strength(two_asset_close, window=50).dropna()
    # Each row sums to 0 (demeaned across columns)
    assert rs.sum(axis=1).abs().max() < 1e-10


# NOTE: leak-safety is verified by the truncation-invariant tests above.
# A naive source-scan for ".shift(-" would also flag the module docstring
# describing the rule, so we rely on the behavioral invariant instead.
