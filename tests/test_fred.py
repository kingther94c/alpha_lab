import pandas as pd
import pytest

from alpha_lab.data.loaders import fred
from alpha_lab.data.loaders.fred import cash_total_return_index, discount_rate_to_daily_rate


class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def test_load_series_accepts_current_fred_observation_date_header(monkeypatch):
    def fake_get(*args, **kwargs):
        return _Response("observation_date,DTB3\n2024-01-02,5.25\n2024-01-03,.\n")

    monkeypatch.setattr(fred.httpx, "get", fake_get)

    series = fred.load_series("DTB3")

    assert series.index[0] == pd.Timestamp("2024-01-02")
    assert series.loc[pd.Timestamp("2024-01-02"), "DTB3"] == pytest.approx(5.25)
    assert pd.isna(series.loc[pd.Timestamp("2024-01-03"), "DTB3"])


def test_merge_fred_frames_outer_joins_on_date_index():
    """Join logic is unit-tested without any network call (the repro's core)."""
    a = fred._parse_fred_csv(
        "observation_date,BAMLH0A0HYM2\n2022-01-01,3.0\n2022-01-03,3.5\n",
        "BAMLH0A0HYM2",
    )
    b = fred._parse_fred_csv(
        "observation_date,NFCI\n2022-01-02,-0.5\n2022-01-03,-0.4\n",
        "NFCI",
    )

    df = fred._merge_fred_frames([a, b])

    assert list(df.columns) == ["BAMLH0A0HYM2", "NFCI"]
    assert list(df.index) == [pd.Timestamp(d) for d in ("2022-01-01", "2022-01-02", "2022-01-03")]
    # dates present in only one series are NaN-filled in the other
    assert df.loc[pd.Timestamp("2022-01-01"), "BAMLH0A0HYM2"] == pytest.approx(3.0)
    assert pd.isna(df.loc[pd.Timestamp("2022-01-01"), "NFCI"])
    assert pd.isna(df.loc[pd.Timestamp("2022-01-02"), "BAMLH0A0HYM2"])
    assert df.loc[pd.Timestamp("2022-01-03"), "NFCI"] == pytest.approx(-0.4)


def test_load_series_multi_fetches_each_code_and_outer_joins(monkeypatch):
    """Multiple codes must be fetched one-per-request, never as ``id=A,B``."""
    responses = {
        "BAMLH0A0HYM2": "observation_date,BAMLH0A0HYM2\n2022-01-03,3.0\n2022-02-01,4.0\n",
        "NFCI": "observation_date,NFCI\n2022-01-07,-0.5\n2022-02-04,-0.4\n",
    }
    requested_ids = []

    def fake_get(*args, **kwargs):
        code = kwargs["params"]["id"]
        requested_ids.append(code)
        return _Response(responses[code])

    monkeypatch.setattr(fred.httpx, "get", fake_get)

    df = fred.load_series(["BAMLH0A0HYM2", "NFCI"], start="2022-01-01", end="2022-03-01")

    # one request per code, each a bare id (the comma-joined form was the bug)
    assert requested_ids == ["BAMLH0A0HYM2", "NFCI"]
    assert all("," not in i for i in requested_ids)
    assert list(df.columns) == ["BAMLH0A0HYM2", "NFCI"]
    assert df.loc[pd.Timestamp("2022-01-03"), "BAMLH0A0HYM2"] == pytest.approx(3.0)
    assert df.loc[pd.Timestamp("2022-01-07"), "NFCI"] == pytest.approx(-0.5)
    # union index, NaN where a series has no observation on a given date
    assert pd.isna(df.loc[pd.Timestamp("2022-01-03"), "NFCI"])


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
