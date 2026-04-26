import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.country_momentum import (
    country_momentum_signal,
    country_momentum_weights,
)


def test_country_momentum_signal_uses_12_minus_1_month_return():
    idx = pd.bdate_range("2022-01-03", "2024-03-29")
    prices = pd.DataFrame({"EWJ": np.arange(len(idx), dtype=float) + 100.0}, index=idx)

    signal = country_momentum_signal(prices, lookback_months=12, skip_months=1)

    monthly = prices.resample("ME").last()
    expected = monthly.shift(1) / monthly.shift(12) - 1
    pd.testing.assert_frame_equal(signal, expected)


def test_country_momentum_weights_are_dollar_neutral():
    idx = pd.bdate_range("2022-01-03", periods=420)
    cols = [f"C{i}" for i in range(12)]
    prices = pd.DataFrame(index=idx)
    for i, col in enumerate(cols):
        prices[col] = 100.0 * (1 + (i + 1) / 10_000) ** np.arange(len(idx))

    weights = country_momentum_weights(
        prices,
        top_n=3,
        bottom_n=3,
        rank_change_threshold=0,
    )

    assert not weights.empty
    assert weights.iloc[-1].clip(lower=0).sum() == pytest.approx(0.5)
    assert weights.iloc[-1].clip(upper=0).sum() == pytest.approx(-0.5)
    assert weights.iloc[-1].sum() == pytest.approx(0.0)


def test_country_momentum_rank_change_buffer_keeps_stable_names():
    idx = pd.bdate_range("2022-01-03", periods=420)
    cols = [f"C{i}" for i in range(12)]
    prices = pd.DataFrame(index=idx)
    for i, col in enumerate(cols):
        prices[col] = 100.0 * (1 + (i + 1) / 10_000) ** np.arange(len(idx))

    buffered = country_momentum_weights(
        prices,
        top_n=2,
        bottom_n=2,
        rank_change_threshold=2,
    )
    unbuffered = country_momentum_weights(
        prices,
        top_n=2,
        bottom_n=2,
        rank_change_threshold=0,
    )

    assert buffered.index.equals(unbuffered.index)
    assert buffered.abs().sum(axis=1).iloc[-1] == pytest.approx(1.0)
