"""Tests for arm_b/perf.py — pure core + store-backed wrapper.

Coverage focuses on the paths reviewers flagged:
  - annualization at real 15-min cadence (inference branch, previously untested)
  - NaN equity marks in the store (desync bug)
  - non-positive / sub-zero de-fauceted equity
  - rf hurdle in Sharpe
  - vol floor preventing dust Sharpe explosion
  - drawdown agreement with risk.drawdown (kill-switch definition)
  - two-equity-point edge case (single return, ddof=1 would blow up)
"""
from __future__ import annotations

import datetime as dt
import math

import pytest
from perf import _RF_ANNUAL, PerfSummary, bot_perf, perf_metrics

from quant_bot_manager.core import risk
from quant_bot_manager.core.store import Store

# ---------------------------------------------------------------------------
# perf_metrics — pure core
# ---------------------------------------------------------------------------


def test_empty_and_single_point_return_nan():
    # Empty: risk.drawdown([]) returns 0.0, so max_drawdown is 0.0 (not NaN).
    s = perf_metrics([])
    assert all(math.isnan(v) for v in (s.ann_return, s.ann_vol, s.sharpe))
    assert s.max_drawdown == pytest.approx(0.0)   # no peak -> no drawdown

    # Single point: same rule — no returns to compute, max_drawdown is 0.0.
    s1 = perf_metrics([100.0])
    assert math.isnan(s1.ann_return)
    assert math.isnan(s1.ann_vol)
    assert math.isnan(s1.sharpe)
    assert s1.max_drawdown == pytest.approx(0.0)


def test_monotonic_up_no_drawdown():
    """Strictly rising equity must have max_drawdown == 0."""
    equity = [100.0, 105.0, 110.0, 115.0, 120.0]
    s = perf_metrics(equity, periods_per_year=4.0, rf=0.0)
    assert s.max_drawdown == pytest.approx(0.0)
    assert s.ann_return > 0
    assert s.ann_vol >= 0


def test_known_annualised_return():
    """Two points 1 year apart (periods_per_year=1, rf=0): ann_return == simple return."""
    s = perf_metrics([100.0, 200.0], periods_per_year=1.0, rf=0.0)
    assert s.ann_return == pytest.approx(1.0)


def test_known_max_drawdown():
    """Peak 120, trough 90 -> max_drawdown = 90/120 - 1 = -0.25."""
    equity = [100.0, 120.0, 90.0, 110.0]
    s = perf_metrics(equity, periods_per_year=252.0)
    assert s.max_drawdown == pytest.approx(-0.25)


def test_flat_equity_zero_vol_zero_sharpe():
    """Constant equity -> vol_eps floor -> ann_vol=0.0, sharpe=0.0 (not NaN, not giant)."""
    equity = [100.0, 100.0, 100.0, 100.0]
    s = perf_metrics(equity, periods_per_year=252.0)
    assert s.ann_vol == pytest.approx(0.0)
    assert s.ann_return == pytest.approx(0.0)
    assert s.sharpe == pytest.approx(0.0)   # flat -> 0, not NaN or huge
    assert s.max_drawdown == pytest.approx(0.0)


def test_dust_flat_equity_no_sharpe_explosion():
    """Near-flat equity with sub-penny dust must NOT produce a giant Sharpe (vol floor)."""
    rng = __import__("numpy").random.default_rng(42)
    equity = [100.0 + rng.normal(0, 1e-10) for _ in range(200)]
    s = perf_metrics(equity, periods_per_year=365.0)
    # vol floor should kick in: Sharpe must be 0.0, not ~1e10
    assert s.sharpe == pytest.approx(0.0)
    assert s.ann_vol == pytest.approx(0.0)


def test_negative_ann_return():
    """Equity that halves over one period (periods_per_year=1, rf=0): ann_return == -0.5."""
    s = perf_metrics([100.0, 50.0], periods_per_year=1.0, rf=0.0)
    assert s.ann_return == pytest.approx(-0.5)


def test_sharpe_sign_matches_excess_return():
    """Positive excess return -> positive Sharpe; negative -> negative."""
    # Use rf=0 so sign is driven by direction of returns only.
    pos = perf_metrics([100.0, 102.0, 104.0], periods_per_year=252.0, rf=0.0)
    assert pos.sharpe > 0

    neg = perf_metrics([100.0, 98.0, 96.0], periods_per_year=252.0, rf=0.0)
    assert neg.sharpe < 0


def test_sharpe_arithmetic_not_geometric():
    """Sharpe is arithmetic mean(excess)/std(excess)*sqrt(ppy), NOT geometric/vol.

    These diverge when returns are volatile (geometric is always smaller for
    non-zero variance). We verify the arithmetic formula by computing it by hand.
    """
    import numpy as np

    equity = [100.0, 110.0, 95.0, 120.0]
    ppy = 1.0
    rf = 0.0
    rets = np.array([110 / 100 - 1, 95 / 110 - 1, 120 / 95 - 1])
    expected_sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(ppy))

    s = perf_metrics(equity, periods_per_year=ppy, rf=rf)
    assert s.sharpe == pytest.approx(expected_sharpe, rel=1e-6)


def test_rf_hurdle_lowers_sharpe():
    """A positive rf reduces Sharpe relative to rf=0."""
    equity = [100.0, 101.0, 102.0, 103.0, 104.0]
    s0 = perf_metrics(equity, periods_per_year=252.0, rf=0.0)
    s4 = perf_metrics(equity, periods_per_year=252.0, rf=0.04)
    assert s0.sharpe > s4.sharpe


def test_non_positive_equity_ann_return_nan():
    """De-fauceted equity at or below zero -> ann_return is NaN (CAGR undefined)."""
    # Path goes through zero
    s = perf_metrics([100.0, 0.0], periods_per_year=1.0)
    assert math.isnan(s.ann_return)

    # Path goes negative
    s2 = perf_metrics([100.0, -50.0], periods_per_year=1.0)
    assert math.isnan(s2.ann_return)


def test_non_positive_equity_drawdown_matches_risk_module():
    """max_drawdown must agree with risk.drawdown for any equity path."""
    equity = [100.0, 120.0, 30.0, 80.0]   # includes a large drawdown
    s = perf_metrics(equity, periods_per_year=252.0)
    assert s.max_drawdown == pytest.approx(risk.drawdown(equity))


def test_drawdown_peak_guard_near_zero():
    """Equity collapsing toward zero must not blow up max_drawdown (peak>0 guard)."""
    equity = [100.0, 50.0, 1e-9, 2e-9]
    s = perf_metrics(equity, periods_per_year=365.0)
    # Should be very close to -1.0 (near-total loss) and finite
    assert math.isfinite(s.max_drawdown)
    assert s.max_drawdown <= 0.0


def test_two_points_no_vol_nan_sharpe():
    """Two equity points -> one return -> ddof=1 std is undefined; vol/Sharpe are NaN."""
    s = perf_metrics([100.0, 110.0], periods_per_year=1.0)
    assert s.ann_return == pytest.approx(0.1)  # ann_return IS valid
    assert math.isnan(s.ann_vol)
    assert math.isnan(s.sharpe)


def test_frozen_dataclass():
    """PerfSummary must be immutable."""
    s = perf_metrics([100.0, 110.0, 120.0], periods_per_year=1.0)
    with pytest.raises(AttributeError):
        s.sharpe = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# bot_perf — store-backed wrapper
# ---------------------------------------------------------------------------


def test_bot_perf_returns_none_on_empty_store(tmp_path):
    store = Store("t", path=tmp_path / "bot.db")
    assert bot_perf(store) is None


def test_bot_perf_returns_none_on_single_mark(tmp_path):
    store = Store("t", path=tmp_path / "bot.db")
    store.append_equity("2026-01-01T00:00:00+00:00", 10_000.0, 6_000.0, 4_000.0)
    assert bot_perf(store) is None


def test_bot_perf_basic_roundtrip(tmp_path):
    """Three equity marks (so >=2 returns) yield a valid PerfSummary with finite metrics."""
    store = Store("t", path=tmp_path / "bot.db")
    store.append_equity("2026-01-01T00:00:00+00:00", 10_000.0, 6_000.0, 4_000.0)
    store.append_equity("2026-02-01T00:00:00+00:00", 10_300.0, 6_180.0, 4_120.0)
    store.append_equity("2026-03-01T00:00:00+00:00", 10_600.0, 6_360.0, 4_240.0)
    result = bot_perf(store)
    assert result is not None
    assert isinstance(result, PerfSummary)
    assert result.ann_return > 0      # grew
    assert result.max_drawdown == pytest.approx(0.0)   # monotonic up


def test_bot_perf_explicit_periods_per_year(tmp_path):
    """Explicit periods_per_year bypasses timestamp inference."""
    store = Store("t", path=tmp_path / "bot.db")
    store.append_equity("2026-01-01T00:00:00+00:00", 100.0, 0.0, 0.0)
    store.append_equity("2026-07-01T00:00:00+00:00", 200.0, 0.0, 0.0)
    result = bot_perf(store, periods_per_year=1.0, rf=0.0)
    assert result is not None
    assert result.ann_return == pytest.approx(1.0)     # doubles in 1 period = 1 year


def test_bot_perf_uses_strategy_equity_not_raw(tmp_path):
    """bot_perf must use de-fauceted equity so a faucet top-up doesn't distort metrics."""
    store = Store("t", path=tmp_path / "bot.db")
    store.set_faucet_offset(90_000.0)
    # raw totals 100k / 105k / 110k; strategy equity 10k / 15k / 20k (offset 90k)
    store.append_equity("2026-01-01T00:00:00+00:00", 100_000.0, 0.0, 0.0)
    store.append_equity("2026-05-01T00:00:00+00:00", 105_000.0, 0.0, 0.0)
    store.append_equity("2026-09-01T00:00:00+00:00", 110_000.0, 0.0, 0.0)
    result = bot_perf(store, periods_per_year=1.0, rf=0.0)
    assert result is not None
    # strategy equity: [10_000, 15_000, 20_000]; 10->20 doubles over 2 periods; ppy=1
    # ann_return = (20000/10000)^(1/2) - 1 = sqrt(2) - 1 ≈ 0.4142
    assert result.ann_return == pytest.approx(2.0 ** 0.5 - 1, rel=1e-4)


def test_bot_perf_drawdown_from_store(tmp_path):
    """Store equity path with a drawdown propagates correctly through bot_perf."""
    store = Store("t", path=tmp_path / "bot.db")
    for ts, total in [
        ("2026-01-01T00:00:00+00:00", 10_000.0),
        ("2026-02-01T00:00:00+00:00", 12_000.0),
        ("2026-03-01T00:00:00+00:00",  9_000.0),
        ("2026-04-01T00:00:00+00:00", 11_000.0),
    ]:
        store.append_equity(ts, total, 0.0, 0.0)
    result = bot_perf(store)
    assert result is not None
    # peak 12k -> 9k: drawdown = 9/12 - 1 = -0.25
    assert result.max_drawdown == pytest.approx(-0.25)


def test_bot_perf_inference_correct_for_15min_cadence(tmp_path):
    """Cadence inference must use median gap and match actual 15-min spacing.

    Regular 15-min marks legitimately produce ppy~35,064 (not a bug).  The key
    correctness check is that the inferred ppy matches the actual mark cadence —
    not that large numbers are suppressed.
    """
    from perf import _infer_ppy

    store = Store("t", path=tmp_path / "bot.db")
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    n = 20
    for i in range(n):
        ts = (base + dt.timedelta(minutes=15 * i)).isoformat()
        store.append_equity(ts, 10_000.0 + float(i), 0.0, 0.0)

    df = store.read_equity_df()
    df = df.dropna(subset=["total_equity"])
    ppy = _infer_ppy(df["ts"].tolist())
    expected_ppy = 365.25 * 24 * 60 / 15   # ~35,064
    assert ppy == pytest.approx(expected_ppy, rel=0.01)


def test_bot_perf_median_gap_robust_to_restart_gap(tmp_path):
    """REGRESSION: median-gap inference must be robust to large restart gaps.

    Mean-based inference (n/elapsed) inflates ppy when most marks are 15 min apart
    but one gap spans a multi-day restart — the mean sees the large gap and returns
    a ppy MUCH lower than the true cadence, making the ann_return wrong.  The median
    is unaffected by a single large outlier gap.
    """
    from perf import _infer_ppy

    # 19 marks at 15-min spacing, then a 7-day gap (simulating a restart), then 5 more.
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    timestamps = []
    for i in range(19):
        timestamps.append(base + dt.timedelta(minutes=15 * i))
    # 7-day restart gap
    restart = base + dt.timedelta(days=7)
    for i in range(5):
        timestamps.append(restart + dt.timedelta(minutes=15 * i))

    ppy_median = _infer_ppy(timestamps)
    expected_ppy = 365.25 * 24 * 60 / 15   # ~35,064

    # Mean-based (n/elapsed) would be: 23 / ((7days + ~4.5h) / 365.25) ≈ 1,160 — badly wrong
    total_secs = (timestamps[-1] - timestamps[0]).total_seconds()
    ppy_mean = (len(timestamps) - 1) / (total_secs / (365.25 * 86_400))

    # Median correctly identifies 15-min cadence; mean is ~30x off
    assert ppy_median == pytest.approx(expected_ppy, rel=0.01), (
        f"median inference wrong: {ppy_median:.0f} vs expected {expected_ppy:.0f}"
    )
    assert ppy_mean < expected_ppy / 10, (
        f"mean inference should be badly wrong here but got {ppy_mean:.0f}"
    )


def test_bot_perf_nan_marks_aligned_with_equity(tmp_path):
    """NaN total rows must be dropped BEFORE ppy inference, keeping timestamps and equity aligned.

    Old bug: ppy used len(full_df)-1 but equity was computed from filtered rows -> 2x overstatement.
    """
    store = Store("t", path=tmp_path / "bot.db")
    # 3 rows; middle total is None (failed mark); real equity is Jan 1 -> Jul 1 = ~0.5yr
    store.append_equity("2026-01-01T00:00:00+00:00", 10_000.0, 0.0, 0.0)
    store.append_equity("2026-04-01T00:00:00+00:00", None, 0.0, 0.0)    # failed mark
    store.append_equity("2026-07-01T00:00:00+00:00", 11_000.0, 0.0, 0.0)
    result = bot_perf(store)
    assert result is not None
    # Only two valid marks (Jan and Jul), one return of +10% over ~0.5yr.
    # ann_return should be ~+21% (1.10^(1/0.5) - 1), NOT ~47% (the 2x overstatement).
    # Allow ±10 pp tolerance for the exact elapsed-time computation.
    assert result.ann_return == pytest.approx(0.21, abs=0.10)
    # Definitely should not be near 0.47 (the old bug's output).
    assert result.ann_return < 0.40, f"ann_return suspiciously high: {result.ann_return:.3f}"


def test_bot_perf_rf_default_is_nonzero(tmp_path):
    """Default rf must be _RF_ANNUAL (0.04), not 0, matching the runner's RF_ANNUAL hurdle."""
    assert _RF_ANNUAL == pytest.approx(0.04)
    store = Store("t", path=tmp_path / "bot.db")
    for i, total in enumerate([10_000.0, 10_100.0, 10_200.0, 10_300.0]):
        ts = f"2026-0{i+1}-01T00:00:00+00:00"
        store.append_equity(ts, total, 0.0, 0.0)
    result_default = bot_perf(store)              # rf=0.04 by default
    result_zero_rf = bot_perf(store, rf=0.0)
    assert result_default is not None
    assert result_zero_rf is not None
    # Positive returns -> rf=0 Sharpe > rf=0.04 Sharpe
    assert result_zero_rf.sharpe > result_default.sharpe
