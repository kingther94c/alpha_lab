import numpy as np
import pandas as pd
import pytest

from alpha_lab.data.align import align_prices, forward_returns


def test_forward_returns_horizon_1_is_next_period():
    r = pd.Series([0.01, 0.02, -0.01, 0.03])
    fwd = forward_returns(r, horizon=1)
    # fwd[t] == r[t+1]
    assert fwd.iloc[0] == pytest.approx(0.02)
    assert fwd.iloc[1] == pytest.approx(-0.01)
    assert fwd.iloc[2] == pytest.approx(0.03)
    assert np.isnan(fwd.iloc[-1])


def test_forward_returns_horizon_h_compounds():
    r = pd.Series([0.0, 0.10, -0.10, 0.05, 0.01])
    fwd2 = forward_returns(r, horizon=2)
    # fwd[0] over next 2 periods = (1+0.10)*(1-0.10) - 1
    assert fwd2.iloc[0] == pytest.approx((1 + 0.10) * (1 - 0.10) - 1)
    # fwd[1] = (1-0.10)*(1+0.05) - 1
    assert fwd2.iloc[1] == pytest.approx((1 - 0.10) * (1 + 0.05) - 1)
    # Last two rows NaN.
    assert np.isnan(fwd2.iloc[-1])
    assert np.isnan(fwd2.iloc[-2])


def test_forward_returns_rejects_bad_horizon():
    r = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError):
        forward_returns(r, horizon=0)


def test_align_prices_ffills_onto_calendar():
    dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"A": [100.0, 101.0]}, index=dates)
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])

    aligned = align_prices(prices, calendar)

    assert aligned.loc["2024-01-03", "A"] == pytest.approx(100.0)  # ffilled
    assert aligned.loc["2024-01-05", "A"] == pytest.approx(101.0)
    assert len(aligned) == len(calendar)


def test_align_prices_none_method_leaves_gaps():
    dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"A": [100.0, 101.0]}, index=dates)
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    aligned = align_prices(prices, calendar, method="none")
    assert np.isnan(aligned.loc["2024-01-03", "A"])
