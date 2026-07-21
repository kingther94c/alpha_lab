from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.backtest.put_write import (
    PutWriteConfig,
    run_cash_secured_put_backtest,
)


def _backtest(
    prices: list[float],
    iv: list[float],
    config: PutWriteConfig,
):
    index = pd.bdate_range("2024-01-02", periods=len(prices))
    return run_cash_secured_put_backtest(
        pd.Series(prices, index=index, dtype=float),
        pd.Series(iv, index=index, dtype=float),
        pd.Series(0.0, index=index),
        pd.Series(0.04, index=index),
        config=config,
    )


def _assignment_prices() -> list[float]:
    return [100, 100, 90, 80, 110, 111, 112, 113, 114, 115, 116, 117]


def test_absolute_iv_gate_uses_previous_close() -> None:
    iv = [0.15] * 5 + [0.30] * 9
    config = PutWriteConfig(
        tenor_trading_days=2,
        max_assignment_days=0,
        entry_min_iv=0.25,
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        strike_increment=0.01,
    )

    result = _backtest([100.0] * len(iv), iv, config)
    entries = result.events[result.events["event"] == "put_entry"]
    index = result.returns.index

    assert entries.index[0] == index[6]
    assert entries.iloc[0]["signal_iv_raw"] == pytest.approx(0.30)


def test_rolling_iv_percentile_does_not_see_future_spike() -> None:
    iv = [0.18, 0.19, 0.20, 0.21, 0.22, 0.20, 0.19, 0.18, 0.80, 0.17, 0.16, 0.15]
    config = PutWriteConfig(
        tenor_trading_days=2,
        max_assignment_days=0,
        entry_min_iv_percentile=0.75,
        entry_iv_lookback=4,
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        strike_increment=0.01,
    )

    with_spike = _backtest([100.0] * len(iv), iv, config)
    without_spike = _backtest(
        [100.0] * len(iv),
        [*iv[:8], 0.18, *iv[9:]],
        config,
    )
    cutoff = with_spike.returns.index[8]
    spike_entries = with_spike.events[
        (with_spike.events["event"] == "put_entry")
        & (with_spike.events.index <= cutoff)
    ].index
    plain_entries = without_spike.events[
        (without_spike.events["event"] == "put_entry")
        & (without_spike.events.index <= cutoff)
    ].index

    assert list(spike_entries) == list(plain_entries)
    assert spike_entries[0] == with_spike.returns.index[4]


def test_timeout_only_ignores_price_recovery() -> None:
    config = PutWriteConfig(
        target_delta=0.25,
        tenor_trading_days=2,
        collateral_fraction=0.5,
        max_assignment_days=3,
        stock_exit_mode="timeout_only",
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        stock_exit_cost_bps=0.0,
        strike_increment=0.01,
    )

    result = _backtest(_assignment_prices(), [0.30] * 12, config)
    assignment = result.events[result.events["event"] == "assignment"].iloc[0]
    exit_event = result.events[result.events["event"] == "stock_exit"].iloc[0]

    assert assignment["spot"] < assignment["strike"]
    assert exit_event["reason"] == "timeout"
    assert exit_event["stock_days"] == 3


def test_iv_exit_trades_one_close_after_signal() -> None:
    config = PutWriteConfig(
        target_delta=0.25,
        tenor_trading_days=2,
        collateral_fraction=0.5,
        max_assignment_days=10,
        stock_exit_mode="iv_or_timeout",
        stock_exit_iv_max=0.20,
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        stock_exit_cost_bps=0.0,
        strike_increment=0.01,
    )
    iv = [0.30, 0.30, 0.30, 0.30, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18]

    result = _backtest(_assignment_prices(), iv, config)
    exit_event = result.events[result.events["event"] == "stock_exit"].iloc[0]
    exit_date = result.events[result.events["event"] == "stock_exit"].index[0]
    low_iv_date = result.returns.index[4]

    assert exit_event["reason"] == "iv_target"
    assert exit_event["signal_iv"] == pytest.approx(0.18)
    assert exit_date == result.returns.index[result.returns.index.get_loc(low_iv_date) + 1]


def test_stock_price_target_is_relative_to_recovery_base() -> None:
    common = dict(
        target_delta=0.25,
        tenor_trading_days=2,
        collateral_fraction=0.5,
        max_assignment_days=3,
        entry_spread_fraction=0.0,
        commission_per_contract=0.0,
        stock_exit_cost_bps=0.0,
        strike_increment=0.01,
    )
    base = _backtest(
        _assignment_prices(),
        [0.30] * 12,
        PutWriteConfig(**common),
    )
    target = _backtest(
        _assignment_prices(),
        [0.30] * 12,
        PutWriteConfig(**common, stock_target_return=0.05),
    )
    base_recovery = base.events[base.events["event"] == "assignment"].iloc[0][
        "recovery_price"
    ]
    target_recovery = target.events[target.events["event"] == "assignment"].iloc[0][
        "recovery_price"
    ]

    assert target_recovery == pytest.approx(base_recovery * 1.05)