import numpy as np
import pandas as pd
import pytest

from alpha_lab.analytics.returns import (
    cumulative_returns,
    drawdown,
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
