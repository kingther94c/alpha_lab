import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.collar import black_scholes_call
from alpha_lab.backtest.wheel import (
    WheelConfig,
    run_wheel_backtest,
    strike_for_target_call_delta,
)


def _inputs(prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        pd.Series(0.30, index=prices.index),
        pd.Series(0.0, index=prices.index),
        pd.Series(0.02, index=prices.index),
    )


def test_target_call_strike_matches_requested_delta() -> None:
    spot = 100.0
    target = 0.25
    years = 21 / 252
    strike = strike_for_target_call_delta(
        spot,
        target,
        years,
        0.03,
        0.25,
        strike_increment=0.01,
    )
    _, delta = black_scholes_call(spot, strike, years, 0.03, 0.25)
    assert delta == pytest.approx(target, abs=0.001)


def test_wheel_assigns_put_then_calls_stock_away() -> None:
    index = pd.bdate_range("2024-01-02", periods=12)
    prices = pd.Series(100.0, index=index)
    prices.iloc[3:6] = 75.0
    prices.iloc[6:] = 120.0
    iv, cash, rates = _inputs(prices)
    result = run_wheel_backtest(
        prices,
        iv,
        cash,
        rates,
        config=WheelConfig(
            put_target_delta=0.40,
            call_target_delta=0.40,
            tenor_trading_days=2,
            entry_spread_fraction=0.0,
            commission_per_contract=0.0,
            strike_increment=0.01,
        ),
    )
    events = result.events["event"].tolist()
    assert "put_assignment" in events
    assert "call_entry" in events
    assert "call_assignment" in events
    assert np.isfinite(result.returns).all()
    assert (result.equity > 0.0).all()


def test_option_entry_has_no_same_bar_profit_at_mid() -> None:
    index = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.Series(100.0, index=index)
    iv, cash, rates = _inputs(prices)
    result = run_wheel_backtest(
        prices,
        iv,
        cash,
        rates,
        config=WheelConfig(
            tenor_trading_days=2,
            entry_spread_fraction=0.0,
            commission_per_contract=0.0,
            strike_increment=0.01,
        ),
        initial_equity=100_000.0,
    )
    first_entry = result.events[result.events["event"] == "put_entry"].index[0]
    assert result.equity.loc[first_entry] == pytest.approx(100_000.0)


def test_wheel_rejects_invalid_delta() -> None:
    with pytest.raises(ValueError, match="call_target_delta"):
        WheelConfig(call_target_delta=0.5)
