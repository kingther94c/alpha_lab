import numpy as np
import pandas as pd
import pytest

from alpha_lab.analytics.risk import cvar, geometric_portfolio_vol, geometric_realized_vol


def test_cvar_averages_tail_observations():
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    assert cvar(returns, q=0.4) == pytest.approx(-0.075)


def test_cvar_rejects_invalid_quantile():
    with pytest.raises(ValueError, match="q must be between 0 and 1"):
        cvar(pd.Series([0.01]), q=1.0)


def test_geometric_portfolio_vol_uses_both_covariance_windows():
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    cov_short = pd.DataFrame(np.diag([0.04, 0.04]), index=weights.index, columns=weights.index)
    cov_long = pd.DataFrame(np.diag([0.01, 0.01]), index=weights.index, columns=weights.index)

    result = geometric_portfolio_vol(weights, cov_short, cov_long)

    expected = np.sqrt(np.sqrt(0.02) * np.sqrt(0.005))
    assert result == pytest.approx(expected)


def test_geometric_portfolio_vol_aligns_series_weights_to_covariance_labels():
    weights = pd.Series({"B": 0.25, "A": 0.75})
    cov_short = pd.DataFrame(
        np.diag([0.04, 0.01]),
        index=["A", "B"],
        columns=["A", "B"],
    )
    cov_long = cov_short.copy()

    result = geometric_portfolio_vol(weights, cov_short, cov_long)

    assert result == pytest.approx(np.sqrt(0.75**2 * 0.04 + 0.25**2 * 0.01))


def test_geometric_realized_vol_requires_long_window_history():
    returns = pd.Series([0.01, -0.01] * 20)

    assert np.isnan(geometric_realized_vol(returns, short_window=21, long_window=63))
