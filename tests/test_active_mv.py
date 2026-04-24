import numpy as np
import pandas as pd
import pytest

from alpha_lab.portfolio.active_mv import (
    active_mean_variance_weights,
    rolling_active_mean_variance_weights,
)


def test_active_mean_variance_weights_are_long_only_and_sum_to_one():
    idx = pd.bdate_range("2024-01-02", periods=80)
    asset_returns = pd.DataFrame(
        {
            "A": np.full(len(idx), 0.002),
            "B": np.full(len(idx), 0.001),
            "C": np.full(len(idx), -0.001),
        },
        index=idx,
    )
    benchmark_returns = pd.Series(0.0, index=idx)

    weights = active_mean_variance_weights(asset_returns, benchmark_returns, risk_aversion=1.0)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights.idxmax() == "A"


def test_rolling_active_mean_variance_weights_uses_rebalance_dates():
    idx = pd.bdate_range("2024-01-02", periods=90)
    prices = pd.DataFrame(
        {
            "A": 100 * (1.001 ** np.arange(len(idx))),
            "B": 100 * (1.0005 ** np.arange(len(idx))),
        },
        index=idx,
    )
    benchmark = pd.Series(100 * (1.0007 ** np.arange(len(idx))), index=idx)

    weights = rolling_active_mean_variance_weights(
        prices,
        benchmark,
        lookback_days=20,
        rebalance="W-FRI",
        risk_aversion=5.0,
    )

    assert not weights.empty
    assert set(weights.columns) == {"A", "B"}
    assert (weights >= 0.0).all().all()
    pd.testing.assert_series_equal(weights.sum(axis=1), pd.Series(1.0, index=weights.index))
