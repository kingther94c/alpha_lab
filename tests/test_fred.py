import pandas as pd
import pytest

from alpha_lab.data.loaders import fred
from alpha_lab.data.loaders.fred import cash_total_return_index, discount_rate_to_daily_rate


class _Response:
    text = "observation_date,DTB3\n2024-01-02,5.25\n2024-01-03,.\n"

    def raise_for_status(self):
        return None


def test_load_series_accepts_current_fred_observation_date_header(monkeypatch):
    def fake_get(*args, **kwargs):
        return _Response()

    monkeypatch.setattr(fred.httpx, "get", fake_get)

    series = fred.load_series("DTB3")

    assert series.index[0] == pd.Timestamp("2024-01-02")
    assert series.loc[pd.Timestamp("2024-01-02"), "DTB3"] == pytest.approx(5.25)
    assert pd.isna(series.loc[pd.Timestamp("2024-01-03"), "DTB3"])


def test_discount_rate_to_daily_rate_converts_bank_discount_quote():
    rates = pd.Series([4.0], index=[pd.Timestamp("2024-01-02")], name="DTB3")

    daily = discount_rate_to_daily_rate(rates, maturity_days=91)

    price = 1.0 - 0.04 * 91 / 360
    expected = (1.0 / price) ** (1.0 / 91) - 1.0
    assert daily.iloc[0] == pytest.approx(expected)


def test_cash_total_return_index_accrues_calendar_day_gaps_with_today_rate():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"])
    rates = pd.Series([3.0, 4.0, 5.0], index=idx, name="DTB3")

    tr = cash_total_return_index(rates, maturity_days=91)
    daily = discount_rate_to_daily_rate(rates, maturity_days=91)

    assert tr.iloc[0] == pytest.approx(100.0)
    assert tr.iloc[1] == pytest.approx(100.0 * (1.0 + daily.iloc[1] * 1))
    assert tr.iloc[2] == pytest.approx(tr.iloc[1] * (1.0 + daily.iloc[2] * 5))
