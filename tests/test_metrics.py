import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.metrics import monthly_table, summary


def test_summary_keys_present():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 252))
    s = summary(r)
    assert set(s.keys()) >= {"CAGR", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar", "HitRate", "NPeriods"}


def test_summary_constant_positive_returns_no_drawdown():
    r = pd.Series([0.001] * 252)
    s = summary(r)
    assert s["MaxDD"] == pytest.approx(0.0)
    assert np.isnan(s["Calmar"])
    assert s["HitRate"] == pytest.approx(1.0)
    assert s["CAGR"] > 0


def test_summary_empty_returns_empty_dict():
    assert summary(pd.Series([], dtype=float)) == {}


def test_monthly_table_rows_are_years():
    idx = pd.bdate_range("2022-01-03", "2023-12-29")
    r = pd.Series(0.001, index=idx)
    tbl = monthly_table(r)
    assert set(tbl.index) == {2022, 2023}
    assert "YTD" in tbl.columns
    # YTD should approximately equal product of monthly columns for that year.
    months = [c for c in tbl.columns if c != "YTD"]
    for _, row in tbl.iterrows():
        product = (1 + row[months].dropna()).prod() - 1
        assert row["YTD"] == pytest.approx(product)
