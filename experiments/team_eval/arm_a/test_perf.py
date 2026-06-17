"""Tests for the bot performance summary (CORE metrics + store-backed wrapper).

The CORE (``equity_summary``) is exercised with paths whose metrics have closed forms, so the
assertions check the math independently rather than restating it. The wrapper (``summarize_bot``)
is exercised through a real ``Store`` on a tmp_path db, using the fixture pattern from the task:
``Store("t", path=tmp_path/"bot.db")`` then ``.append_equity(ts, total, fut, spot)``.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from perf import PerfSummary, equity_summary, summarize_bot
from pytest import approx

from quant_bot_manager.core.store import Store

_DAY = dt.timedelta(days=1)
_T0 = dt.datetime(2024, 1, 1)


def _daily_ts(n: int) -> list[str]:
    return [(_T0 + i * _DAY).isoformat() for i in range(n)]


def test_constant_growth_path_has_zero_vol_and_known_return():
    # +1%/period, every period -> zero return variance -> vol 0, Sharpe 0 (guarded), but a
    # well-defined annualized return. Pin ppy=365 so the CAGR expectation is exact (inference is
    # covered separately); 365 constant periods compound to exactly 1.01**365 - 1.
    n = 30
    eq = [100.0 * (1.01 ** i) for i in range(n)]
    s = equity_summary(eq, _daily_ts(n), periods_per_year=365.0)

    assert isinstance(s, PerfSummary)
    assert s.n_marks == n
    # Constant per-period return => no measurable risk. (Float dust in 1.01**i keeps std at ~1e-15,
    # which the CORE's epsilon guard collapses to exactly 0.0 so Sharpe can't blow up.)
    assert s.ann_vol == approx(0.0, abs=1e-9)
    assert s.sharpe == approx(0.0, abs=1e-6)
    assert s.max_drawdown == 0.0  # monotonically increasing
    assert s.ann_return == approx(1.01 ** 365 - 1.0, rel=1e-6)


def test_ann_return_matches_geometric_formula_with_explicit_ppy():
    # Override ppy so the expected CAGR is exact and independent of timestamp inference.
    eq = [100.0, 110.0, 90.0, 120.0]
    ppy = 252.0
    s = equity_summary(eq, _daily_ts(len(eq)), periods_per_year=ppy)

    total_growth = eq[-1] / eq[0]
    expected = total_growth ** (ppy / (len(eq) - 1)) - 1.0
    assert s.ann_return == approx(expected, rel=1e-9)


def test_vol_and_sharpe_match_independent_numpy():
    eq = [100.0, 105.0, 102.0, 108.0, 104.0, 112.0]
    ppy = 252.0
    rf = 0.0
    s = equity_summary(eq, _daily_ts(len(eq)), rf=rf, periods_per_year=ppy)

    rets = pd.Series(eq).pct_change().dropna().to_numpy()
    exp_vol = rets.std(ddof=1) * np.sqrt(ppy)
    exp_sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(ppy)

    assert s.ann_vol == approx(exp_vol, rel=1e-9)
    assert s.sharpe == approx(exp_sharpe, rel=1e-9)


def test_nonzero_rf_lowers_sharpe():
    eq = [100.0, 105.0, 102.0, 108.0, 104.0, 112.0]
    ppy = 252.0
    s0 = equity_summary(eq, _daily_ts(len(eq)), rf=0.0, periods_per_year=ppy)
    s1 = equity_summary(eq, _daily_ts(len(eq)), rf=0.05, periods_per_year=ppy)
    # Positive drift, positive hurdle -> excess mean falls, vol unchanged -> Sharpe strictly lower.
    assert s1.sharpe < s0.sharpe
    assert s1.ann_vol == approx(s0.ann_vol, rel=1e-12)


def test_max_drawdown_matches_known_path():
    # Peak 120 then trough 90 -> worst dd = 90/120 - 1 = -0.25.
    eq = [100.0, 120.0, 90.0, 110.0]
    s = equity_summary(eq, _daily_ts(len(eq)), periods_per_year=252.0)
    assert s.max_drawdown == approx(-0.25, rel=1e-12)


def test_periods_per_year_inferred_from_daily_timestamps():
    # Daily spacing -> ppy ~ 365.25; check the inference lands near a calendar year.
    eq = [100.0, 101.0, 99.0, 102.0, 103.0]
    s = equity_summary(eq, _daily_ts(len(eq)))
    # Recover ppy from the (single-period> guarded) vol: ann_vol = std * sqrt(ppy).
    rets = pd.Series(eq).pct_change().dropna()
    inferred_ppy = (s.ann_vol / rets.std(ddof=1)) ** 2
    assert inferred_ppy == approx(365.25, rel=1e-6)


def test_empty_and_single_mark_are_safe():
    empty = equity_summary([], None)
    assert empty == PerfSummary(0.0, 0.0, 0.0, 0.0, 0)

    one = equity_summary([100.0], _daily_ts(1))
    assert one.n_marks == 1
    assert one.ann_return == 0.0
    assert one.ann_vol == 0.0
    assert one.sharpe == 0.0
    assert one.max_drawdown == 0.0


def test_none_marks_are_dropped():
    # A skipped mark (None total) must not crash or count.
    eq = [100.0, None, 110.0, None, 121.0]
    s = equity_summary(eq, periods_per_year=252.0)
    assert s.n_marks == 3


def test_summarize_bot_via_store(tmp_path):
    store = Store("t", path=tmp_path / "bot.db")
    eq = [100.0, 120.0, 90.0, 110.0, 130.0]
    ts = _daily_ts(len(eq))
    for t, total in zip(ts, eq, strict=True):
        store.append_equity(t, total, total * 0.5, total * 0.5)

    summary = summarize_bot(store)
    core = equity_summary(eq, ts)

    assert summary == core
    assert summary.n_marks == len(eq)
    assert summary.max_drawdown == approx(-0.25, rel=1e-9)  # 120 -> 90


def test_summarize_bot_empty_store_is_safe(tmp_path):
    store = Store("t", path=tmp_path / "bot.db")
    summary = summarize_bot(store)
    assert summary == PerfSummary(0.0, 0.0, 0.0, 0.0, 0)
