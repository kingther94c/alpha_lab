import pandas as pd

from alpha_lab.data.calendars import rebalance_dates, trading_days


def test_trading_days_excludes_weekend():
    idx = trading_days("2024-01-01", "2024-01-10")
    assert all(d.weekday() < 5 for d in idx)
    # New Year's Day 2024 was a Monday and is a US federal holiday.
    assert pd.Timestamp("2024-01-01") not in idx


def test_rebalance_dates_monthly_picks_last_trading_day():
    idx = trading_days("2024-01-01", "2024-03-31")
    month_ends = rebalance_dates(idx, freq="ME")
    assert len(month_ends) == 3
    # Jan, Feb, Mar 2024 last trading days:
    assert month_ends[0] == pd.Timestamp("2024-01-31")
    assert month_ends[1] == pd.Timestamp("2024-02-29")
    # Mar 29 2024 is Good Friday (NYSE closed) but NOT a US federal holiday,
    # so the federal-calendar trading_days keeps it.
    assert month_ends[2] == pd.Timestamp("2024-03-29")
    # Every rebalance date is a real trading day in the input index.
    assert month_ends.isin(idx).all()


def test_rebalance_dates_quarterly():
    idx = trading_days("2024-01-01", "2024-12-31")
    q = rebalance_dates(idx, freq="QE")
    assert len(q) == 4
    assert q.isin(idx).all()


def test_rebalance_dates_daily_returns_index():
    idx = trading_days("2024-01-01", "2024-01-10")
    out = rebalance_dates(idx, freq="D")
    pd.testing.assert_index_equal(out, idx)


def test_rebalance_dates_empty_index():
    empty = pd.DatetimeIndex([])
    assert len(rebalance_dates(empty, freq="ME")) == 0
