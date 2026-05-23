"""Backtrader cross-check oracle for the vectorized engine.

This module exists for **one job**: prove that
:func:`alpha_lab.backtest.vector.run_backtest` matches an independent
event-driven backtester on a small slice, so we can trust the vectorized
engine for the bulk of research. It is NOT a workhorse — do not import it
from notebooks.

Scope deliberately narrow:
  - Single-asset, target-weight strategy.
  - Cheat-on-close fills so the execution timing matches vector.py's
    "lag 1, return as pct_change" convention.
  - Commission (bps) + percent slippage. No funding (tested directly).

If :mod:`backtrader` is not importable the helper raises ``ImportError``;
the cross-check test is then skipped.
"""

from __future__ import annotations

import pandas as pd

try:
    import backtrader as bt  # noqa: F401
    _HAS_BT = True
except ImportError:  # pragma: no cover
    _HAS_BT = False


def has_backtrader() -> bool:
    """True iff backtrader is importable in the current env."""
    return _HAS_BT


def run_backtrader_check(
    signals: pd.Series,
    ohlcv: pd.DataFrame,
    *,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    starting_cash: float = 100_000.0,
) -> pd.Series:
    """Run a Backtrader target-weight strategy and return per-bar net returns.

    Parameters
    ----------
    signals : Series of target weights in [-1, 1] (single asset). Index must
        match ``ohlcv.index``.
    ohlcv : DataFrame with at least ``open, high, low, close, volume`` columns
        and a tz-aware UTC DatetimeIndex.
    commission_bps, slippage_bps : matched to the vectorized engine's
        ``costs_bps`` and ``slippage_bps`` for the comparison.
    starting_cash : initial broker cash (cancels out in fractional returns).

    Returns
    -------
    pandas.Series of per-bar **net** returns (after commission + slippage),
    tz-aware UTC index, same length as ``ohlcv``.
    """
    if not _HAS_BT:
        raise ImportError(
            "backtrader is not installed in this environment; "
            "install with `pip install backtrader` or skip the cross-check."
        )
    import backtrader as bt  # type: ignore

    # Align signals to ohlcv and clamp any leading NaN to 0
    s = signals.reindex(ohlcv.index).ffill().fillna(0.0)

    # Backtrader expects strict column names. Build a minimal feed frame.
    feed_df = pd.DataFrame(
        {
            "open": ohlcv["open"].astype(float).to_numpy(),
            "high": ohlcv["high"].astype(float).to_numpy(),
            "low": ohlcv["low"].astype(float).to_numpy(),
            "close": ohlcv["close"].astype(float).to_numpy(),
            "volume": ohlcv["volume"].astype(float).to_numpy(),
            "openinterest": 0.0,
        },
        index=ohlcv.index.tz_convert(None) if ohlcv.index.tz is not None else ohlcv.index,
    )

    class _TargetWeightStrategy(bt.Strategy):
        params = dict(targets=s)

        def __init__(self):
            self.bar_values: list[float] = []

        def next(self):
            # Record the broker value at each bar BEFORE acting on signal[t].
            self.bar_values.append(float(self.broker.get_value()))
            target = float(self.p.targets.iloc[len(self) - 1])
            # COC=True: order placed here fills at this bar's close, so the
            # position is in place starting next bar's open. Matches the
            # vectorized engine's "today's signal becomes tomorrow's held weight".
            self.order_target_percent(target=target)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_cash(starting_cash)
    cerebro.broker.set_coc(True)
    if commission_bps > 0:
        cerebro.broker.setcommission(commission=commission_bps / 10_000.0)
    if slippage_bps > 0:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10_000.0)
    cerebro.adddata(bt.feeds.PandasData(dataname=feed_df))
    cerebro.addstrategy(_TargetWeightStrategy)
    results = cerebro.run()
    strat = results[0]

    values = pd.Series(strat.bar_values, index=ohlcv.index[: len(strat.bar_values)])
    rets = values.pct_change().fillna(0.0)
    # Pad to full ohlcv length if Backtrader skipped any leading bars
    rets = rets.reindex(ohlcv.index).fillna(0.0)
    return rets
