"""Cross-check: vectorized engine vs Backtrader oracle.

Skipped if backtrader is not importable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.bt_engine import has_backtrader, run_backtrader_check
from alpha_lab.backtest.vector import run_backtest


pytestmark = pytest.mark.skipif(
    not has_backtrader(),
    reason="backtrader not installed",
)


def _synthetic_ohlcv(n: int = 240, seed: int = 7) -> pd.DataFrame:
    """Hourly OHLCV with mild drift + noise. UTC tz-aware index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-06-01", periods=n, freq="1h", tz="UTC")
    drift = 0.0001  # ~+1 bp / bar
    close = 50_000 * np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n))
    volume = rng.uniform(50, 500, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_bt_matches_vector_always_long():
    ohlcv = _synthetic_ohlcv(n=240)
    sig = pd.Series(1.0, index=ohlcv.index)  # always long
    # vector engine: signal as single-column wide frame
    sig_wide = sig.to_frame("X")
    prices_wide = ohlcv[["close"]].rename(columns={"close": "X"})
    vec = run_backtest(sig_wide, prices_wide, costs_bps=0, slippage_bps=0)
    bt = run_backtrader_check(sig, ohlcv, commission_bps=0, slippage_bps=0)
    # The two engines may have different first-bar semantics (vector returns 0
    # at the first bar because pct_change is NaN; BT records cash before first
    # trade). Skip first bar in comparison.
    diff = (vec.returns.iloc[1:] - bt.iloc[1:]).abs().max()
    assert diff < 1e-3, f"max abs diff = {diff}"


def test_bt_matches_vector_long_only_directional_agreement():
    """Long/flat strategy with zero costs: cumulative returns from both engines
    should agree in DIRECTION (both up or both down).

    Per-bar exact match is not expected: Backtrader records broker value at
    start of next() (post previous-bar fill), while the vectorized engine
    multiplies pct_change by the already-lagged held weight. The two timings
    differ by one bar at each transition. We only assert the engines move
    the same way overall and that the gap is small in absolute terms.
    """
    ohlcv = _synthetic_ohlcv(n=240)
    sig = pd.Series(np.where((np.arange(240) // 48) % 2 == 0, 1.0, 0.0), index=ohlcv.index)
    sig_wide = sig.to_frame("X")
    prices_wide = ohlcv[["close"]].rename(columns={"close": "X"})
    vec = run_backtest(sig_wide, prices_wide, costs_bps=0, slippage_bps=0)
    bt = run_backtrader_check(sig, ohlcv, commission_bps=0, slippage_bps=0)
    vec_total = float((1 + vec.returns).prod())
    bt_total = float((1 + bt).prod())
    # Same side of 1.0 (both losses or both gains), and gap < 2%
    assert (vec_total - 1) * (bt_total - 1) > 0 or abs(vec_total - bt_total) < 1e-3, (
        f"Engines disagree on direction: vec={vec_total:.6f} bt={bt_total:.6f}"
    )
    assert abs(vec_total - bt_total) < 0.02, (
        f"Cumulative return gap too wide: vec={vec_total:.6f} bt={bt_total:.6f}"
    )


def test_bt_costs_are_a_drag():
    """Sanity: under positive turnover, with-cost run produces a worse outcome
    than zero-cost run. This is a sanity check, not a numerical match —
    BT's cost model (commission × fill notional, slippage on fill price) differs
    from vector.py's flat bps × turnover, so total drags can diverge."""
    ohlcv = _synthetic_ohlcv(n=240)
    sig = pd.Series(np.where((np.arange(240) // 24) % 2 == 0, 1.0, 0.0), index=ohlcv.index)
    bt_free = run_backtrader_check(sig, ohlcv, commission_bps=0, slippage_bps=0)
    bt_costs = run_backtrader_check(sig, ohlcv, commission_bps=2.0, slippage_bps=3.0)
    assert (1 + bt_costs).prod() < (1 + bt_free).prod()


def test_bt_engine_smoke_flat_zero_returns():
    """A flat (zero-weight) strategy should produce zero returns in BT."""
    ohlcv = _synthetic_ohlcv(n=120)
    sig = pd.Series(0.0, index=ohlcv.index)
    bt = run_backtrader_check(sig, ohlcv, commission_bps=0, slippage_bps=0)
    assert bt.abs().max() < 1e-9
