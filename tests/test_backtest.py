import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.vector import run_backtest, run_drift_backtest


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


def test_drift_backtest_trades_next_close_and_earns_following_return():
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 121.0]}, index=idx)
    targets = pd.DataFrame({"A": [1.0]}, index=[idx[0]])

    result = run_drift_backtest(targets, prices, trading_bps=0.0)

    assert result.decision_to_trade.loc[idx[0]] == idx[1]
    assert result.returns.loc[idx[1]] == pytest.approx(0.0)
    assert result.returns.loc[idx[2]] == pytest.approx(0.10)


def test_drift_backtest_weights_drift_between_rebalances():
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 100.0, 200.0, 200.0], "B": [100.0] * 4},
        index=idx,
    )
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[idx[0]])

    result = run_drift_backtest(targets, prices, trading_bps=0.0)

    assert result.pre_trade_weights.loc[idx[2], "A"] == pytest.approx(2 / 3)
    assert result.pre_trade_weights.loc[idx[2], "B"] == pytest.approx(1 / 3)


def test_drift_backtest_threshold_compares_with_drifted_weights():
    idx = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"A": [100.0, 100.0, 102.0, 102.0, 102.0], "B": [100.0] * 5}, index=idx)
    targets = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=[idx[0], idx[2]],
    )

    result = run_drift_backtest(
        targets,
        prices,
        trading_bps=0.0,
        rebalance_threshold=0.01,
    )

    assert result.trade_flags.loc[idx[1]]
    assert not result.trade_flags.loc[idx[3]]


def test_drift_backtest_charges_actual_non_cash_notional():
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"A": [100.0] * 4, "B": [100.0] * 4}, index=idx)
    targets = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        index=[idx[0], idx[1]],
    )

    result = run_drift_backtest(targets, prices, trading_bps=10.0)

    assert result.traded_notional.loc[idx[1]] == pytest.approx(1.0 / 1.001)
    assert result.traded_notional.loc[idx[2]] == pytest.approx(2.0 / 1.001, rel=1e-6)
    assert result.costs.loc[idx[2]] > result.costs.loc[idx[1]]
    assert np.allclose(result.weights.sum(axis=1), 1.0)


def test_drift_backtest_cash_earns_provided_return_before_first_trade():
    idx = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0] * 3}, index=idx)
    targets = pd.DataFrame({"A": [1.0]}, index=[idx[1]])
    cash_returns = pd.Series([0.0, 0.001, 0.001], index=idx)

    result = run_drift_backtest(
        targets,
        prices,
        trading_bps=0.0,
        cash_returns=cash_returns,
    )

    assert result.returns.loc[idx[1]] == pytest.approx(0.001)
    assert result.returns.loc[idx[2]] == pytest.approx(0.001)
    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_price", [np.nan, 0.0, -1.0, np.inf])
def test_drift_backtest_rejects_nonpositive_or_nonfinite_prices(bad_price):
    idx = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, bad_price, 101.0]}, index=idx)
    targets = pd.DataFrame({"A": [1.0]}, index=[idx[0]])

    with pytest.raises(ValueError, match="finite and strictly positive"):
        run_drift_backtest(targets, prices)


def test_drift_backtest_preserves_timezone_in_decision_to_trade():
    idx = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    prices = pd.DataFrame({"A": [100.0, 100.0, 101.0]}, index=idx)
    targets = pd.DataFrame({"A": [1.0]}, index=[idx[0]])

    result = run_drift_backtest(targets, prices, trading_bps=0.0)

    assert result.decision_to_trade.loc[idx[0]] == idx[1]
    assert result.decision_to_trade.dt.tz == idx.tz
