from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.backtest.put_write import (
    PutWriteConfig,
    black_scholes_put,
    build_put_write_advice,
    run_cash_secured_put_backtest,
    strike_for_target_put_delta,
)


def test_target_strike_reprices_near_requested_delta() -> None:
    strike = strike_for_target_put_delta(
        100.0,
        0.15,
        30 / 365.25,
        0.04,
        0.25,
        strike_increment=0.01,
    )
    _, delta = black_scholes_put(100.0, strike, 30 / 365.25, 0.04, 0.25)

    assert delta == pytest.approx(-0.15, abs=0.001)
    assert strike < 100.0


def test_flat_market_collects_premium_without_same_bar_gain() -> None:
    index = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.Series(100.0, index=index)
    iv = pd.Series(0.20, index=index)
    cash = pd.Series(0.0, index=index)
    rates = pd.Series(0.04, index=index)
    config = PutWriteConfig(
        tenor_trading_days=5,
        collateral_fraction=1.0,
        max_assignment_days=0,
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        strike_increment=0.01,
    )

    result = run_cash_secured_put_backtest(prices, iv, cash, rates, config=config)

    first_entry = result.events[result.events["event"] == "put_entry"].index[0]
    assert result.returns.loc[first_entry] == pytest.approx(0.0, abs=1e-12)
    assert result.equity.iloc[-1] > 1.0
    assert "expiry_otm" in set(result.events["event"])


def test_assignment_blocks_new_put_until_stock_exit() -> None:
    index = pd.bdate_range("2024-01-02", periods=18)
    prices = pd.Series(
        [100, 100, 99, 97, 94, 85, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93],
        index=index,
        dtype=float,
    )
    iv = pd.Series(0.25, index=index)
    cash = pd.Series(0.0, index=index)
    rates = pd.Series(0.04, index=index)
    config = PutWriteConfig(
        target_delta=0.25,
        tenor_trading_days=5,
        collateral_fraction=0.5,
        max_assignment_days=3,
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        stock_exit_cost_bps=0.0,
        strike_increment=0.01,
    )

    result = run_cash_secured_put_backtest(prices, iv, cash, rates, config=config)
    assignments = result.events[result.events["event"] == "assignment"]
    exits = result.events[result.events["event"] == "stock_exit"]

    assert len(assignments) >= 1
    assert len(exits) >= 1
    first_assignment = assignments.index[0]
    first_exit = exits.index[0]
    entries_between = result.events[
        (result.events["event"] == "put_entry")
        & (result.events.index > first_assignment)
        & (result.events.index <= first_exit)
    ]
    assert entries_between.empty
    assert exits.iloc[0]["reason"] == "timeout"


def test_advice_uses_whole_contracts_and_quoted_bid() -> None:
    config = PutWriteConfig(
        collateral_fraction=0.5,
        tenor_trading_days=21,
        strike_increment=1.0,
    )
    advice = build_put_write_advice(
        spot=500.0,
        annual_iv=0.25,
        annual_cash_yield=0.04,
        account_nav=200_000.0,
        config=config,
        quoted_bid=4.25,
    )

    assert advice.contracts >= 1
    assert advice.collateral <= 100_000.0
    assert advice.price_source == "quoted bid"
    assert advice.breakeven < advice.strike
