import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.collar import (
    SyntheticCollarConfig,
    SyntheticOptionOverlayConfig,
    black_scholes_call,
    run_synthetic_collar,
    run_synthetic_option_overlay,
    third_friday_roll_dates,
)
from alpha_lab.backtest.put_write import black_scholes_put


def test_black_scholes_call_put_parity() -> None:
    spot = 100.0
    strike = 105.0
    years = 45.0 / 365.25
    rate = 0.04
    vol = 0.22
    call, _ = black_scholes_call(spot, strike, years, rate, vol)
    put, _ = black_scholes_put(spot, strike, years, rate, vol)
    expected = spot - strike * np.exp(-rate * years)
    assert call - put == pytest.approx(expected)


def test_third_friday_maps_holiday_to_prior_bar() -> None:
    index = pd.bdate_range("2022-03-01", "2022-04-30")
    index = index[index != pd.Timestamp("2022-04-15")]
    rolls = third_friday_roll_dates(index)
    assert pd.Timestamp("2022-03-18") in rolls
    assert pd.Timestamp("2022-04-14") in rolls


def test_synthetic_collar_is_finite_and_rolls() -> None:
    index = pd.bdate_range("2018-01-01", "2020-12-31")
    daily_return = pd.Series(0.0002, index=index)
    adjusted = 100.0 * (1.0 + daily_return).cumprod()
    spot = adjusted.copy()
    vix = pd.Series(18.0, index=index)
    cash = pd.Series(0.01 / 252.0, index=index)
    rates = pd.Series(0.01, index=index)
    result = run_synthetic_collar(adjusted, spot, vix, cash, rates)
    assert np.isfinite(result.returns).all()
    assert (result.equity > 0.0).all()
    assert int(result.diagnostics["call_roll"].sum()) >= 24
    assert int(result.diagnostics["put_roll"].sum()) >= 8


def test_synthetic_collar_rejects_nonpositive_vix() -> None:
    index = pd.bdate_range("2019-01-01", periods=300)
    adjusted = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    with pytest.raises(ValueError, match="VIX must be positive"):
        run_synthetic_collar(
            adjusted,
            adjusted,
            pd.Series(0.0, index=index),
            pd.Series(0.0, index=index),
            pd.Series(0.02, index=index),
            config=SyntheticCollarConfig(),
        )


def test_option_overlay_zero_ratios_track_base_returns() -> None:
    index = pd.bdate_range("2019-01-01", "2020-12-31")
    base = pd.Series(0.0003, index=index)
    adjusted = 100.0 * (1.0 + pd.Series(0.0002, index=index)).cumprod()
    result = run_synthetic_option_overlay(
        base,
        adjusted,
        adjusted,
        pd.Series(18.0, index=index),
        pd.Series(0.01 / 252.0, index=index),
        pd.Series(0.01, index=index),
        put_ratio=0.0,
    )
    expected = base.reindex(result.returns.index)
    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.returns.iloc[1:].to_numpy() == pytest.approx(expected.iloc[1:].to_numpy())


def test_quarterly_put_spread_reduces_a_sudden_base_drawdown() -> None:
    index = pd.bdate_range("2019-01-01", "2019-06-30")
    adjusted = pd.Series(100.0, index=index)
    adjusted.loc["2019-04-01":] = 80.0
    base = adjusted.pct_change().fillna(0.0)
    result = run_synthetic_option_overlay(
        base,
        adjusted,
        adjusted,
        pd.Series(18.0, index=index),
        pd.Series(0.0, index=index),
        pd.Series(0.0, index=index),
        put_ratio=1.0,
        config=SyntheticOptionOverlayConfig(short_put_otm=0.15),
    )
    unhedged = (1.0 + base.reindex(result.equity.index)).cumprod()
    assert float(result.equity.min()) > float(unhedged.min())
    assert int(result.diagnostics["put_roll"].sum()) >= 2


def test_option_overlay_rejects_inverted_put_spread() -> None:
    with pytest.raises(ValueError, match="deeper OTM"):
        SyntheticOptionOverlayConfig(short_put_otm=0.03)


def test_option_overlay_tolerates_floating_point_ratio_noise() -> None:
    index = pd.bdate_range("2020-01-02", "2020-12-31")
    spot = pd.Series(100.0, index=index)
    result = run_synthetic_option_overlay(
        pd.Series(0.0, index=index),
        spot,
        spot,
        pd.Series(20.0, index=index),
        pd.Series(0.0, index=index),
        pd.Series(0.02, index=index),
        put_ratio=pd.Series(1.0 + 1e-12, index=index),
        config=SyntheticOptionOverlayConfig(short_put_otm=0.15, call_otm=None),
    )

    assert result.diagnostics["put_ratio"].max() == pytest.approx(1.0)
