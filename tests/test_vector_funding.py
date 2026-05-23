"""Tests for the funding-cost extension of run_backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.vector import _bucket_funding_to_bars, run_backtest


@pytest.fixture
def simple_prices() -> pd.DataFrame:
    """3 days, two symbols, constant +1% / +0.5% daily returns."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    btc = pd.Series([100.0, 101.0, 102.01, 103.0301], index=idx)
    eth = pd.Series([10.0, 10.05, 10.10025, 10.1507513], index=idx)
    return pd.DataFrame({"BTCUSDT": btc, "ETHUSDT": eth})


def test_backward_compat_no_funding(simple_prices):
    """funding=None must produce identical numerical output to pre-funding API."""
    signals = pd.DataFrame(0.5, index=simple_prices.index, columns=simple_prices.columns)
    res = run_backtest(signals, simple_prices, costs_bps=0, slippage_bps=0)
    # all funding_costs zero, meta says no funding
    assert (res.funding_costs == 0).all()
    assert res.meta["has_funding"] is False
    # gross == net when no funding and no cost
    pd.testing.assert_series_equal(res.gross_returns, res.returns, check_names=False)


def test_funding_long_pays_positive_rate(simple_prices):
    """Long position with positive funding rate has POSITIVE funding_cost
    (which is subtracted from gross to net)."""
    signals = pd.DataFrame(
        {"BTCUSDT": [1.0, 1.0, 1.0, 1.0], "ETHUSDT": [0.0, 0.0, 0.0, 0.0]},
        index=simple_prices.index,
    )
    # Funding event at 2024-01-02 12:00 UTC (mid-day -> floors into the 2024-01-02 bar)
    funding = pd.DataFrame(
        {"BTCUSDT": [0.001], "ETHUSDT": [0.001]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 12:00", tz="UTC")]),
    )
    res = run_backtest(
        signals, simple_prices, funding=funding, costs_bps=0, slippage_bps=0,
    )
    # On 2024-01-02, held weight is 1.0 in BTCUSDT (lagged from t=0 signal), 0 in ETH.
    # funding_cost on that bar = 1.0 * 0.001 + 0 * 0.001 = 0.001
    assert res.funding_costs.loc[pd.Timestamp("2024-01-02", tz="UTC")] == pytest.approx(0.001)
    # All other bars zero
    other = res.funding_costs.drop(pd.Timestamp("2024-01-02", tz="UTC"))
    assert (other == 0).all()


def test_funding_short_receives_positive_rate(simple_prices):
    """Short position with positive funding rate has NEGATIVE funding_cost
    (a credit, which boosts net)."""
    signals = pd.DataFrame(-1.0, index=simple_prices.index, columns=simple_prices.columns)
    funding = pd.DataFrame(
        {"BTCUSDT": [0.001], "ETHUSDT": [0.001]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 00:00", tz="UTC")]),
    )
    res = run_backtest(
        signals, simple_prices, funding=funding, costs_bps=0, slippage_bps=0,
    )
    # held on 2024-01-02 = -1.0 on each symbol; cost = -1*0.001 + -1*0.001 = -0.002
    assert res.funding_costs.loc[pd.Timestamp("2024-01-02", tz="UTC")] == pytest.approx(-0.002)


def test_funding_zero_when_flat(simple_prices):
    """A flat (zero-weight) strategy pays no funding regardless of rates."""
    signals = pd.DataFrame(0.0, index=simple_prices.index, columns=simple_prices.columns)
    funding = pd.DataFrame(
        {"BTCUSDT": [0.001] * 3, "ETHUSDT": [0.001] * 3},
        index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
    )
    res = run_backtest(signals, simple_prices, funding=funding)
    assert (res.funding_costs == 0).all()


def test_funding_aligned_to_containing_bar():
    """Funding event at HH:00.001 must land in the HH:00 bar (floor), not HH+1:00."""
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="1min", tz="UTC")
    prices = pd.DataFrame({"BTCUSDT": np.linspace(100, 101, 10)}, index=idx)
    # Funding at 00:05.001 -> should bucket into 00:05 bar
    funding = pd.DataFrame(
        {"BTCUSDT": [0.0005]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:05:00.001", tz="UTC")]),
    )
    bucketed = _bucket_funding_to_bars(funding, idx)
    bar_005 = pd.Timestamp("2024-01-01 00:05", tz="UTC")
    bar_006 = pd.Timestamp("2024-01-01 00:06", tz="UTC")
    assert bucketed.loc[bar_005, "BTCUSDT"] == pytest.approx(0.0005)
    assert pd.isna(bucketed.loc[bar_006, "BTCUSDT"]) or bucketed.loc[bar_006, "BTCUSDT"] == 0.0


def test_funding_sum_matches_manual_calc():
    """Three funding events over a few bars; total funding drag matches a
    hand-computed expectation."""
    idx = pd.date_range("2024-01-01", periods=6, freq="8h", tz="UTC")
    prices = pd.DataFrame({"BTCUSDT": np.linspace(100, 106, 6)}, index=idx)
    signals = pd.DataFrame(1.0, index=idx, columns=["BTCUSDT"])
    # 3 events exactly on bars 1, 3, 5 (after the lag, held=1 on bars 1..5)
    funding = pd.DataFrame(
        {"BTCUSDT": [0.0001, 0.0002, 0.0003]},
        index=pd.DatetimeIndex([idx[1], idx[3], idx[5]]),
    )
    res = run_backtest(signals, prices, funding=funding, costs_bps=0, slippage_bps=0)
    # held=1 on bars 1, 2, 3, 4, 5. funding events are at bars 1, 3, 5.
    # Expected sum = 1*0.0001 + 1*0.0002 + 1*0.0003 = 0.0006
    assert res.funding_costs.sum() == pytest.approx(0.0006)


def test_meta_carries_bars_per_year(simple_prices):
    signals = pd.DataFrame(0.0, index=simple_prices.index, columns=simple_prices.columns)
    res = run_backtest(signals, simple_prices, bars_per_year=525_600)
    assert res.meta["bars_per_year"] == 525_600
    assert res.meta["has_funding"] is False
