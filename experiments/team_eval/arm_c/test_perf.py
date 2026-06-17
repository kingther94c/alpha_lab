"""Tests for the equity performance summary (arm_c).

The pure core (``equity_perf``) carries the math, so most assertions hit it directly with
analytically-known curves. One test drives the full path through a real ``Store`` using the
prescribed fixture pattern: ``Store("t", path=tmp_path / "bot.db")`` + ``append_equity``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from perf import PerfSummary, bot_perf, equity_perf

from quant_bot_manager.core.store import Store


def _daily_ts(n: int, start: str = "2024-01-01") -> pd.Series:
    """n consecutive daily timestamps as strings (how the runner records marks)."""
    d0 = dt.date.fromisoformat(start)
    return pd.Series([(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)])


def test_constant_growth_curve():
    """A curve compounding at a fixed *daily* rate has zero vol / zero drawdown, and its CAGR
    is that daily rate annualized over a 365.25-day year (the function scales by real elapsed
    calendar time, so consecutive daily marks => (1+g)^365.25 - 1)."""
    g = 0.0005  # per-day growth
    n = 253  # 252 daily steps -> a 252-calendar-day span
    eq = 100.0 * (1.0 + g) ** np.arange(n)
    ts = _daily_ts(n)

    s = equity_perf(eq, ts=ts)

    assert isinstance(s, PerfSummary)
    assert s.n_obs == n
    assert s.ann_return == pytest.approx((1 + g) ** 365.25 - 1, rel=1e-6)
    assert s.ann_vol == pytest.approx(0.0, abs=1e-9)
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_flat_curve_has_undefined_sharpe():
    """A perfectly flat curve has zero variance; Sharpe is NaN (not +/-inf or a huge number)."""
    s = equity_perf([100.0] * 20)
    assert s.ann_return == pytest.approx(0.0, abs=1e-12)
    assert s.ann_vol == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(s.sharpe)
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_known_drawdown():
    """Equity 100 -> 120 -> 90 -> 110: peak 120, trough 90, so 120 -> 90 is -30,
    i.e. -30/120 = -0.25 (a 25% drawdown)."""
    s = equity_perf([100.0, 120.0, 90.0, 110.0])
    assert s.max_drawdown == pytest.approx(-30.0 / 120.0)  # 120 -> 90 == -30/120 == -0.25
    assert s.n_obs == 4


def test_vol_and_sharpe_match_house_convention():
    """ann_vol and Sharpe equal the std/mean * sqrt(periods) formulas on the period returns."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=300)
    eq = 100.0 * np.cumprod(1.0 + rets)
    realized = pd.Series(eq).pct_change().dropna()

    s = equity_perf(eq, periods=252)

    assert s.ann_vol == pytest.approx(realized.std(ddof=1) * np.sqrt(252))
    expected_sharpe = realized.mean() / realized.std(ddof=1) * np.sqrt(252)
    assert s.sharpe == pytest.approx(expected_sharpe)


def test_rf_lowers_sharpe():
    """A positive per-period risk-free rate reduces Sharpe (excess-return hurdle)."""
    rng = np.random.default_rng(1)
    eq = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, size=300))
    assert equity_perf(eq, rf=0.0005).sharpe < equity_perf(eq, rf=0.0).sharpe


@pytest.mark.parametrize("equity", [[], [100.0], [100.0, np.nan]])
def test_degenerate_inputs_dont_raise(equity):
    """Too-few / NaN points yield NaN metrics rather than an exception."""
    s = equity_perf(equity)
    assert np.isnan(s.ann_return) and np.isnan(s.max_drawdown)


def test_non_positive_start_is_safe():
    """A zero/negative starting equity can't produce a real growth ratio -> NaN, no crash."""
    s = equity_perf([0.0, 100.0, 110.0])
    assert np.isnan(s.ann_return)


def test_two_point_curve_is_internally_consistent():
    """One return (n==2) is too few to annualize a horizon or estimate a std. ALL annualized
    metrics must be NaN together — never a finite ann_return (the old bug extrapolated a single
    +10% mark to ~2.7e10) sitting next to NaN risk. Drawdown stays defined for the curve."""
    s = equity_perf([100.0, 110.0])
    assert s.n_obs == 2
    assert np.isnan(s.ann_return)
    assert np.isnan(s.ann_vol)
    assert np.isnan(s.sharpe)
    # An all-up 2-point curve never drew down.
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_negative_equity_drawdown_clamped():
    """A leveraged paper book can cross zero. Drawdown is a fraction bounded in [-1, 0]: you
    can't lose more than the running peak. The naive eq/peak-1 gives -1.1 here; clamp to -1.
    The growth-ratio / pct_change metrics are undefined across a sign flip, so they're NaN."""
    s = equity_perf([100.0, 50.0, -10.0, 20.0])
    assert s.max_drawdown == pytest.approx(-1.0)
    assert s.max_drawdown >= -1.0
    assert np.isnan(s.ann_return)
    assert np.isnan(s.ann_vol)
    assert np.isnan(s.sharpe)


def test_zero_touch_equity_drawdown_clamped():
    """Equity touching exactly zero is also a full -100% drawdown, not worse, and the
    return-based metrics are undefined (a return relative to a zero mark blows up)."""
    s = equity_perf([100.0, 50.0, 0.0, 20.0, 30.0])
    assert s.max_drawdown == pytest.approx(-1.0)
    assert np.isnan(s.ann_return)


def test_interior_nan_is_not_bridged():
    """An interior missing mark must NOT be bridged into one fabricated jumbo return. With a
    single hole and only 3 raw points, the lone surviving return is too few to annualize, so
    everything is NaN — not the old ~7e20 ann_return from gluing 100 and 121 together."""
    s = equity_perf([100.0, np.nan, 121.0])
    assert s.n_obs == 2  # valid marks only
    assert np.isnan(s.ann_return)
    assert np.isnan(s.ann_vol)
    assert np.isnan(s.sharpe)


def test_interior_nan_excludes_only_the_straddling_return():
    """A longer curve with one interior hole keeps the genuine adjacent returns and drops only
    the return that would straddle the gap. Compare against the same curve with the hole's
    neighbours treated as non-adjacent: vol/Sharpe come from the real steps, not the bridge."""
    eq_hole = [100.0, 101.0, np.nan, 103.0, 104.0, 105.0]
    s = equity_perf(eq_hole)
    # Returns that actually exist: 100->101, 103->104, 104->105. The 101->103 jump is dropped.
    expected = pd.Series([101 / 100, 104 / 103, 105 / 104]) - 1.0
    assert s.n_obs == 5  # five valid marks
    assert s.ann_vol == pytest.approx(expected.std(ddof=1) * np.sqrt(252))


def test_unsorted_timestamps_give_same_horizon_as_sorted():
    """_years must sort timestamps before differencing, so an out-of-order ts series yields the
    real elapsed span (a positive horizon) instead of silently collapsing to even-spacing."""
    eq = [100.0, 101.0, 102.0, 103.0]
    sorted_ts = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    shuffled_ts = pd.Series(["2024-01-04", "2024-01-01", "2024-01-02", "2024-01-03"])
    assert equity_perf(eq, ts=shuffled_ts).ann_return == pytest.approx(
        equity_perf(eq, ts=sorted_ts).ann_return
    )


def test_large_balance_tiny_drift_keeps_real_sharpe():
    """The flat-curve guard floors std in returns-space (a small absolute value), not scaled by
    the equity LEVEL. A genuine tiny drift on a large balance must still produce a finite Sharpe,
    not get masked to NaN by a level-scaled threshold."""
    rng = np.random.default_rng(7)
    base = 1e8 * np.cumprod(1.0 + rng.normal(1e-7, 1e-6, size=300))
    s = equity_perf(base)
    assert np.isfinite(s.sharpe)
    assert s.ann_vol > 0


def test_bot_perf_reads_store(tmp_path):
    """End-to-end through a real Store using the prescribed fixture pattern."""
    store = Store("t", path=tmp_path / "bot.db")
    g = 0.001
    ts = _daily_ts(60)
    for i, t in enumerate(ts):
        total = 100.0 * (1.0 + g) ** i
        store.append_equity(t, total, total * 0.5, total * 0.5)

    s = bot_perf(store)

    assert isinstance(s, PerfSummary)
    assert s.n_obs == 60
    assert s.ann_return > 0  # monotone-up curve
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert s.ann_vol == pytest.approx(0.0, abs=1e-9)


def test_bot_perf_uses_store_timestamps(tmp_path):
    """Proves the ts column is actually wired through bot_perf, not ignored in favour of the
    even-spacing fallback. Marks span ~10 calendar years (far from 3 evenly-spaced days), so a
    fixed +21% total growth annualizes to a small CAGR via real time, but to a huge one under
    even spacing. Asserting the small value pins the timestamp path."""
    store = Store("t", path=tmp_path / "bot.db")
    marks = [("2010-01-01", 100.0), ("2015-01-01", 110.0), ("2020-01-01", 121.0)]
    for t, total in marks:
        store.append_equity(t, total, total * 0.5, total * 0.5)

    s = bot_perf(store)

    # 21% over ~10 years ~= 1.9%/yr; even-spacing (3 daily steps) would give (1.21)**(252/2)-1.
    assert s.ann_return == pytest.approx(1.21 ** (1.0 / 10.0) - 1.0, rel=2e-3)
    assert s.ann_return < 0.05  # nowhere near the even-spacing blow-up


def test_bot_perf_empty_store(tmp_path):
    """A brand-new bot with no marks returns NaN metrics, not an error."""
    store = Store("t", path=tmp_path / "bot.db")
    s = bot_perf(store)
    assert s.n_obs == 0
    assert np.isnan(s.ann_return)
