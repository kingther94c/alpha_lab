import numpy as np
import pandas as pd
import pytest

from alpha_lab.analytics.returns import (
    cumulative_returns,
    drawdown,
    drawdown_duration_metrics,
    log_returns,
    simple_returns,
)


def test_simple_returns_basic():
    s = pd.Series([100.0, 110.0, 99.0])
    r = simple_returns(s)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.1)
    assert r.iloc[2] == pytest.approx(-0.1)


def test_log_returns_matches_exp_identity():
    s = pd.Series([100.0, 110.0, 99.0])
    lr = log_returns(s).dropna()
    assert float(np.exp(lr.sum())) == pytest.approx(s.iloc[-1] / s.iloc[0])


def test_cumulative_returns_and_drawdown():
    r = pd.Series([0.0, 0.1, -0.1, 0.05])
    wealth = cumulative_returns(r)
    assert wealth.iloc[-1] == pytest.approx((1 + 0.1) * (1 - 0.1) * (1 + 0.05))
    dd = drawdown(r)
    assert dd.max() == pytest.approx(0.0)
    assert dd.min() < 0


def test_drawdown_counts_loss_from_initial_capital():
    r = pd.Series([-0.10, 0.05])
    dd = drawdown(r)
    assert dd.iloc[0] == pytest.approx(-0.10)
    assert dd.iloc[1] == pytest.approx(-0.055)


def test_drawdown_duration_splits_underwater_and_recovery_legs():
    wealth = pd.Series([1.0, 1.1, 1.0, 0.9, 1.0, 1.1])
    returns = wealth.pct_change().fillna(0.0)
    result = drawdown_duration_metrics(
        returns, material_threshold=0.05, recovery_target_days=2
    )

    assert result["max_underwater_days"] == 3
    assert result["max_trough_to_recovery_days"] == 2
    assert result["max_material_recovery_days"] == 2
    assert result["median_material_recovery_days"] == pytest.approx(2.0)
    assert result["share_material_recovered_within_target"] == pytest.approx(1.0)
    assert result["material_drawdown_count"] == 1
    assert result["unrecovered_material_drawdown_count"] == 0


def test_drawdown_duration_treats_open_episode_as_censored_failure():
    wealth = pd.Series([1.0, 1.1, 1.0, 0.9, 0.95])
    returns = wealth.pct_change().fillna(0.0)
    result = drawdown_duration_metrics(
        returns, material_threshold=0.05, recovery_target_days=20
    )

    assert result["max_underwater_days"] == 3
    assert result["max_material_recovery_days"] == 1
    assert result["share_material_recovered_within_target"] == pytest.approx(0.0)
    assert result["unrecovered_material_drawdown_count"] == 1


def test_drawdown_duration_excludes_immaterial_episode():
    wealth = pd.Series([1.0, 1.1, 1.09, 1.1])
    returns = wealth.pct_change().fillna(0.0)
    result = drawdown_duration_metrics(returns, material_threshold=0.05)

    assert result["max_underwater_days"] == 1
    assert result["material_drawdown_count"] == 0
    assert np.isnan(result["median_material_recovery_days"])
