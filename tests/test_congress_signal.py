"""Tests for the congressional-signal core helpers.

Covers the subtle / easy-to-break pieces (per the repo's minimal-testing rule):
amount log-midpoint, direction parsing, the point-in-time flow bucketing, signed
sector aggregation, z-score leak-safety, dollar-neutral weights, and the stats. All
synthetic — no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.backtest.congress_signal import (
    bucket_onto_grid,
    sector_flow_zscore,
    sector_net_flow,
    sector_tilt_weights,
)
from alpha_lab.data.congress_universe import gics_sectors, sector_etf_map
from alpha_lab.data.loaders.congress import _normalize_direction, amount_logmid
from alpha_lab.stats.tests import bootstrap_sharpe_ci, deflated_sharpe_ratio


# ----------------------------------------------------------------------------- loaders
def test_amount_logmid_geometric_mean():
    assert amount_logmid(1001, 15000) == np.sqrt(1001 * 15000)
    # open-ended top bucket → falls back to low
    assert amount_logmid(50_000_000, None) == 50_000_000
    assert amount_logmid(None, 15000) == 15000
    assert np.isnan(amount_logmid(None, None))


def test_normalize_direction():
    assert _normalize_direction("Purchase") == ("buy", 1)
    assert _normalize_direction("Sale (Full)") == ("sell", -1)
    assert _normalize_direction("Sale (Partial)") == ("sell", -1)
    assert _normalize_direction("Exchange") == ("exchange", 0)
    assert _normalize_direction("") == ("other", 0)


# -------------------------------------------------------------------- PIT bucketing
def test_bucket_onto_grid_rolls_weekend_to_next_session_and_conserves_flow():
    # A filing dated Saturday 2022-01-01 must land on the next trading session.
    trading_index = pd.bdate_range("2021-12-31", "2022-01-10")  # Fri 12/31, Mon 1/3, ...
    cal = pd.Series({pd.Timestamp("2022-01-01"): 10.0})
    on_grid = bucket_onto_grid(cal, trading_index)
    assert on_grid.loc["2022-01-03"] == 10.0          # rolled forward to Monday
    assert on_grid.loc["2021-12-31"] == 0.0           # nothing before the filing
    assert np.isclose(on_grid.sum(), 10.0)            # flow conserved, none lost/duplicated


def test_bucket_onto_grid_no_future_leak():
    # Flow dated AFTER the last trading day in the grid must not appear on any grid day.
    trading_index = pd.bdate_range("2022-01-03", "2022-01-07")
    cal = pd.Series({pd.Timestamp("2022-01-20"): 5.0})
    on_grid = bucket_onto_grid(cal, trading_index)
    assert np.isclose(on_grid.sum(), 0.0)


# ----------------------------------------------------------------- sector aggregation
def _trades(rows):
    return pd.DataFrame(rows, columns=["filing_date", "ticker", "amount_logmid"]).assign(
        filing_date=lambda d: pd.to_datetime(d["filing_date"]))


def test_sector_net_flow_signs_net_correctly():
    idx = pd.bdate_range("2022-01-03", "2022-03-31")
    sector_of = pd.Series({"AAA": "Technology", "BBB": "Technology", "CCC": "Energy"})
    trades = _trades([
        ("2022-01-05", "AAA", +100.0),   # buy
        ("2022-01-05", "BBB", -30.0),    # sell
        ("2022-01-05", "CCC", +40.0),
    ])
    net = sector_net_flow(trades, sector_of, idx, window=252)
    last = net.iloc[-1]
    assert np.isclose(last["Technology"], 70.0)   # +100 − 30 netted
    assert np.isclose(last["Energy"], 40.0)
    # Unknown tickers are dropped, not bucketed into a sector.
    assert set(net.columns) == set(gics_sectors())


def test_sector_flow_zscore_is_leak_safe():
    # z at time t must not change when future data is appended (trailing window only).
    idx = pd.bdate_range("2020-01-01", "2022-12-31")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx,
                         columns=["Technology", "Energy", "Financials"]).cumsum()
    cut = idx[400]
    z_full = sector_flow_zscore(panel, z_window=252, min_periods=60)
    z_short = sector_flow_zscore(panel.loc[:cut], z_window=252, min_periods=60)
    common = z_full.loc[:cut].dropna()
    pd.testing.assert_frame_equal(common, z_short.loc[common.index], check_freq=False)


def test_sector_tilt_weights_dollar_neutral_and_etf_columns():
    idx = pd.bdate_range("2022-01-03", periods=5)
    sectors = gics_sectors()
    z = pd.DataFrame(np.tile(np.arange(len(sectors), dtype=float), (len(idx), 1)),
                     index=idx, columns=sectors)
    w = sector_tilt_weights(z, top_n=3, bottom_n=3, long_gross=1.0, short_gross=1.0)
    assert np.allclose(w.sum(axis=1), 0.0)                      # dollar-neutral
    assert np.isclose(w.clip(lower=0).sum(axis=1).iloc[-1], 1.0)  # long gross = 1
    assert set(w.columns) <= set(sector_etf_map().values())     # ETF-ticker columns


# ----------------------------------------------------------------------------- stats
def test_deflated_sharpe_ratio_bounds_and_monotonicity():
    high = deflated_sharpe_ratio(2.0, n_obs=2000, n_trials=1)["dsr"]
    low = deflated_sharpe_ratio(0.2, n_obs=2000, n_trials=50)["dsr"]
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high > low                                           # strong single trial >> weak among many


def test_bootstrap_sharpe_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0006, 0.01, size=1500))          # positive-drift daily returns
    out = bootstrap_sharpe_ci(r, n_boot=400, block=21, seed=2)
    assert out["lo"] <= out["sharpe"] <= out["hi"]
    assert out["p_gt_0"] > 0.5
