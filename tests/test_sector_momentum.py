import pandas as pd
import pytest

from alpha_lab.backtest.sector_momentum import (
    express_sector_views,
    sector_momentum_signal,
    top_bottom_view_weights,
)


def test_sector_momentum_signal_uses_monthly_skip_window():
    idx = pd.bdate_range("2023-01-02", "2024-03-29")
    prices = pd.DataFrame({"XLK": range(100, 100 + len(idx))}, index=idx)

    signal = sector_momentum_signal(prices, lookback_months=3, skip_months=1)

    monthly = prices.resample("ME").last()
    expected = monthly.shift(1) / monthly.shift(3) - 1
    pd.testing.assert_frame_equal(signal, expected)


def test_top_bottom_view_weights_is_dollar_neutral_by_default():
    signal = pd.DataFrame(
        [[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]],
        index=[pd.Timestamp("2024-01-31")],
        columns=list("ABCDEF"),
    )

    weights = top_bottom_view_weights(signal, top_n=2, bottom_n=2)

    assert weights.loc["2024-01-31", "A"] == pytest.approx(0.5)
    assert weights.loc["2024-01-31", "B"] == pytest.approx(0.5)
    assert weights.loc["2024-01-31", "E"] == pytest.approx(-0.5)
    assert weights.loc["2024-01-31", "F"] == pytest.approx(-0.5)
    assert weights.sum(axis=1).iloc[0] == pytest.approx(0.0)


def test_express_original_short_uses_signed_original_etfs():
    views = pd.DataFrame({"XLK": [0.5], "XLF": [-0.5]}, index=[pd.Timestamp("2024-01-31")])
    universe = pd.DataFrame(
        {
            "sector": ["Technology", "Financials"],
            "signal_etf": ["XLK", "XLF"],
            "long_1x_etf": ["XLK", "XLF"],
            "long_2x_etf": ["", ""],
            "long_3x_etf": ["TECL", "FAS"],
            "inverse_1x_etf": ["", "SEF"],
            "inverse_2x_etf": ["", "SKF"],
            "inverse_3x_etf": ["TECS", "FAZ"],
        }
    )

    trade_weights = express_sector_views(views, universe, mode="original_short")

    assert trade_weights.loc["2024-01-31", "XLK"] == pytest.approx(0.5)
    assert trade_weights.loc["2024-01-31", "XLF"] == pytest.approx(-0.5)


def test_express_leveraged_etf_preserves_approximate_exposure():
    views = pd.DataFrame({"XLK": [0.6], "XLF": [-0.6]}, index=[pd.Timestamp("2024-01-31")])
    universe = pd.DataFrame(
        {
            "sector": ["Technology", "Financials"],
            "signal_etf": ["XLK", "XLF"],
            "long_1x_etf": ["XLK", "XLF"],
            "long_2x_etf": ["", ""],
            "long_3x_etf": ["TECL", "FAS"],
            "inverse_1x_etf": ["", "SEF"],
            "inverse_2x_etf": ["", "SKF"],
            "inverse_3x_etf": ["TECS", "FAZ"],
        }
    )

    trade_weights = express_sector_views(views, universe, mode="leveraged_etf", leverage=3)

    assert trade_weights.loc["2024-01-31", "TECL"] == pytest.approx(0.2)
    assert trade_weights.loc["2024-01-31", "FAZ"] == pytest.approx(0.2)


def test_express_leveraged_etf_requires_available_tickers():
    views = pd.DataFrame({"XLC": [-1.0]}, index=[pd.Timestamp("2024-01-31")])
    universe = pd.DataFrame(
        {
            "sector": ["Communication Services"],
            "signal_etf": ["XLC"],
            "long_1x_etf": ["XLC"],
            "long_2x_etf": [""],
            "long_3x_etf": [""],
            "inverse_1x_etf": [""],
            "inverse_2x_etf": [""],
            "inverse_3x_etf": [""],
        }
    )

    with pytest.raises(ValueError, match="missing inverse_3x_etf"):
        express_sector_views(views, universe, mode="leveraged_etf", leverage=3)
