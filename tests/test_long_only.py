import numpy as np
import pandas as pd
import pytest

from alpha_lab.portfolio.long_only import (
    inverse_volatility_weights,
    mean_variance_weights,
    rolling_momentum_weights,
)


def test_inverse_volatility_weights_are_long_only_and_sum_to_one():
    idx = pd.bdate_range("2024-01-02", periods=40)
    returns = pd.DataFrame(
        {
            "low_vol": [0.001, -0.001] * 20,
            "high_vol": [0.01, -0.01] * 20,
        },
        index=idx,
    )

    weights = inverse_volatility_weights(returns, min_periods=20)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights["low_vol"] > weights["high_vol"]


def test_rolling_momentum_weights_select_top_assets_without_lookahead():
    idx = pd.bdate_range("2024-01-02", periods=80)
    prices = pd.DataFrame(
        {
            "winner": 100 * (1.003 ** np.arange(len(idx))),
            "middle": 100 * (1.001 ** np.arange(len(idx))),
            "loser": 100 * (0.999 ** np.arange(len(idx))),
        },
        index=idx,
    )

    weights = rolling_momentum_weights(
        prices,
        lookback_days=20,
        skip_days=5,
        top_n=1,
        rebalance="W-FRI",
    )

    assert not weights.empty
    assert (weights >= 0.0).all().all()
    pd.testing.assert_series_equal(weights.sum(axis=1), pd.Series(1.0, index=weights.index))
    assert weights.iloc[-1]["winner"] == pytest.approx(1.0)


def test_mean_variance_weights_are_long_only_and_prefer_high_return_asset():
    idx = pd.bdate_range("2024-01-02", periods=60)
    returns = pd.DataFrame(
        {
            "high": np.full(len(idx), 0.002),
            "low": np.full(len(idx), 0.0005),
            "negative": np.full(len(idx), -0.001),
        },
        index=idx,
    )

    weights = mean_variance_weights(returns, risk_aversion=1.0, min_periods=40)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights.idxmax() == "high"
