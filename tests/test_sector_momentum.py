import pandas as pd
import pytest

from alpha_lab.backtest.sector_momentum import (
    express_sector_views,
    multi_horizon_momentum_signal,
    sector_momentum_signal,
    top_bottom_view_weights,
    volume_confirmation_signal,
)


def test_sector_momentum_signal_uses_monthly_skip_window():
    idx = pd.bdate_range("2023-01-02", "2024-03-29")
    prices = pd.DataFrame({"XLK": range(100, 100 + len(idx))}, index=idx)

    signal = sector_momentum_signal(prices, lookback_months=3, skip_months=1)

    monthly = prices.loc[prices.groupby(prices.index.to_period("M")).tail(1).index]
    expected = monthly.shift(1) / monthly.shift(3) - 1
    pd.testing.assert_frame_equal(signal, expected)


def test_sector_momentum_signal_uses_actual_last_trading_day():
    idx = pd.bdate_range("2024-07-01", "2024-08-30")
    prices = pd.DataFrame({"XLK": range(len(idx))}, index=idx)

    signal = sector_momentum_signal(prices, lookback_months=1, skip_months=0)

    assert pd.Timestamp("2024-08-30") in signal.index
    assert pd.Timestamp("2024-08-31") not in signal.index


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


def test_top_bottom_view_weights_sparse_row_keeps_long_and_short_disjoint():
    # Only 5 valid names but top_n + bottom_n = 6, so the middle name (C) lands in
    # both the top-3 and bottom-3 rank sets. The long side must take priority; C
    # stays long and is not overwritten into a short.
    signal = pd.DataFrame(
        [[5.0, 4.0, 3.0, 2.0, 1.0]],
        index=[pd.Timestamp("2024-01-31")],
        columns=list("ABCDE"),
    )

    weights = top_bottom_view_weights(signal, top_n=3, bottom_n=3)
    row = weights.loc["2024-01-31"]

    longs = row[row > 0]
    shorts = row[row < 0]

    # No name is simultaneously long and short.
    assert set(longs.index).isdisjoint(shorts.index)
    assert list(longs.index) == ["A", "B", "C"]
    assert list(shorts.index) == ["D", "E"]

    # Long side keeps its full intended gross; each leg carries the per-name weight.
    assert longs.sum() == pytest.approx(1.0)
    assert longs.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert shorts.tolist() == pytest.approx([-1 / 3, -1 / 3])


def test_multi_horizon_momentum_signal_is_unchanged_by_future_prices():
    idx = pd.bdate_range("2023-01-02", periods=100)
    prices = pd.DataFrame(
        {"A": range(100, 200), "B": range(200, 100, -1)},
        index=idx,
        dtype=float,
    )
    changed = prices.copy()
    changed.loc[idx[-10]:, "A"] *= 10.0

    original = multi_horizon_momentum_signal(
        prices,
        lookback_days=(20, 40),
        skip_days=5,
    )
    revised = multi_horizon_momentum_signal(
        changed,
        lookback_days=(20, 40),
        skip_days=5,
    )

    pd.testing.assert_frame_equal(original.loc[:idx[-11]], revised.loc[:idx[-11]])


def test_multi_horizon_momentum_signal_uses_lookback_as_older_endpoint():
    idx = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "A": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            "B": [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        },
        index=idx,
    )

    signal = multi_horizon_momentum_signal(prices, lookback_days=(6,), skip_days=2)
    trailing = prices.shift(2) / prices.shift(6) - 1.0

    pd.testing.assert_frame_equal(signal, trailing.rank(axis=1, pct=True))


def test_multi_horizon_momentum_signal_rejects_nonpositive_measurement_window():
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="greater than skip_days"):
        multi_horizon_momentum_signal(prices, lookback_days=(2,), skip_days=2)


def test_volume_confirmation_uses_trailing_own_history():
    idx = pd.bdate_range("2024-01-02", periods=50)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    volumes = pd.DataFrame(
        {"A": range(100, 150), "B": range(150, 100, -1)},
        index=idx,
        dtype=float,
    )

    signal = volume_confirmation_signal(prices, volumes, short_window=5, long_window=20)

    assert signal.loc[idx[-1], "A"] > 0
    assert signal.loc[idx[-1], "B"] < 0
