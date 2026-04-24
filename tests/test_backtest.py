import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.vector import run_backtest


def _constant_prices(daily_ret: float, n: int = 250, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["A"]
    idx = pd.bdate_range("2024-01-02", periods=n)
    path = (1 + daily_ret) ** np.arange(n)
    return pd.DataFrame({c: path * 100 for c in cols}, index=idx)


def test_buy_and_hold_matches_asset_return_with_zero_costs():
    prices = _constant_prices(0.001, n=100)
    signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    res = run_backtest(signals, prices, costs_bps=0.0, slippage_bps=0.0)
    # After day-1 lag, the portfolio earns exactly the asset return.
    expected = prices.pct_change().fillna(0.0).iloc[:, 0]
    # Day 0 held is 0 (signal lagged), then constant 1.
    assert res.returns.iloc[0] == pytest.approx(0.0)
    pd.testing.assert_series_equal(
        res.returns.iloc[2:],
        expected.iloc[2:],
        check_names=False,
    )


def test_costs_reduce_net_returns():
    prices = _constant_prices(0.001, n=60)
    signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    gross = run_backtest(signals, prices, costs_bps=0.0, slippage_bps=0.0).returns.sum()
    net = run_backtest(signals, prices, costs_bps=5.0, slippage_bps=5.0).returns.sum()
    assert net < gross


def test_turnover_concentrated_on_rebalance_days():
    prices = _constant_prices(0.0, n=80, cols=["A", "B"])
    # Alternate 100% A and 100% B each month.
    s = pd.DataFrame(0.0, index=prices.index, columns=["A", "B"])
    s.loc[s.index.month % 2 == 0, "A"] = 1.0
    s.loc[s.index.month % 2 == 1, "B"] = 1.0
    res = run_backtest(s, prices, rebalance="ME", costs_bps=0.0, slippage_bps=0.0)
    nonzero = res.turnover[res.turnover > 1e-9]
    # Turnover only the day after each month-end rebalance.
    assert len(nonzero) <= len(res.turnover.index.to_period("M").unique())
    # First rebalance goes 0 -> 100% A (half-sum = 0.5). Subsequent flips A<->B = 1.0.
    assert nonzero.iloc[0] == pytest.approx(0.5)
    assert (nonzero.iloc[1:].round(9) == 1.0).all()


def test_signal_lag_prevents_lookahead():
    # Prices spike on day 10. If no lag, the backtest would magically capture it;
    # with lag, position is only established starting the day *after* the signal.
    n = 20
    prices = pd.DataFrame(
        {"A": [100.0] * 10 + [110.0] * (n - 10)},
        index=pd.bdate_range("2024-01-02", periods=n),
    )
    signals = pd.DataFrame(0.0, index=prices.index, columns=["A"])
    signals.iloc[9] = 1.0  # signal on spike day — return already happened
    res = run_backtest(signals, prices, costs_bps=0.0, slippage_bps=0.0)
    # No position on the spike day itself -> no capture.
    assert res.returns.iloc[9] == pytest.approx(0.0)


def test_equity_curve_property():
    prices = _constant_prices(0.001, n=30)
    signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    res = run_backtest(signals, prices, costs_bps=0.0, slippage_bps=0.0)
    eq = res.equity
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] > 1.0
