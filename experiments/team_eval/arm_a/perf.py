"""Headline performance summary for a paper-trading bot (execution leg).

Two layers, kept decoupled so the math is unit-testable without a live bot or a DB:

  * ``equity_summary``  — the metric CORE: a pure function over an equity *path* (list of
    mark-to-market totals, oldest-first) plus its timestamps. Computes the four headline
    numbers — annualized return, annualized volatility, Sharpe, and max drawdown.
  * ``summarize_bot``   — a thin store-backed wrapper: pull the equity series out of a
    ``quant_bot_manager.core.store.Store`` and hand it to ``equity_summary``.

Annualization note: a bot's marks are not guaranteed daily (the runner marks once per cycle at
whatever cadence it runs), so we *infer* periods-per-year from the median spacing between
timestamps rather than assuming 252/365. With <2 marks or degenerate spacing we fall back to 365
(this leg trades crypto and day-counts on a 365-ish calendar — see ``runner.rf_hurdle``).

Drawdown reuses ``quant_bot_manager.core.risk.drawdown`` so the summary and the kill-switch agree
on exactly one definition of "worst peak-to-trough".
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_bot_manager.core.risk import drawdown

# Fallback marks-per-year when cadence can't be inferred (crypto, 365-day calendar — matches the
# runner's cost-of-cash day-count). Daily marks => 365; the inference below overrides this when
# timestamps are present.
_DEFAULT_PERIODS_PER_YEAR = 365.0


@dataclass(frozen=True)
class PerfSummary:
    """The four headline metrics for an equity path, plus the number of marks behind them.

    ``ann_return`` / ``ann_vol`` are fractions (0.12 == 12%); ``max_drawdown`` is a fraction <= 0;
    ``sharpe`` is annualized and unitless. ``n_marks`` is how many equity points were used.
    """

    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    n_marks: int


def _infer_periods_per_year(timestamps: Sequence | None, n: int) -> float:
    """Marks-per-year implied by the median gap between (sorted) timestamps.

    Falls back to ``_DEFAULT_PERIODS_PER_YEAR`` when there aren't >=2 usable timestamps or the
    spacing is degenerate (zero/non-finite) — e.g. several marks sharing one timestamp.
    """
    if timestamps is None or n < 2:
        return _DEFAULT_PERIODS_PER_YEAR
    ts = pd.to_datetime(pd.Series(list(timestamps))).sort_values()
    gaps = ts.diff().dropna().dt.total_seconds().to_numpy()
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return _DEFAULT_PERIODS_PER_YEAR
    median_gap_s = float(np.median(gaps))
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median_gap_s


def equity_summary(
    equity: Sequence[float],
    timestamps: Sequence | None = None,
    *,
    rf: float = 0.0,
    periods_per_year: float | None = None,
) -> PerfSummary:
    """Headline metrics for an equity path (oldest-first list of mark-to-market totals).

    Pure: no I/O, no Store. ``timestamps`` (one per equity point) are used only to infer the
    annualization factor; pass ``periods_per_year`` to override the inference. ``rf`` is an
    *annual* risk-free rate used as the Sharpe hurdle (de-annualized per period internally).

    With fewer than two marks there are no returns to measure, so return-derived metrics are 0.0
    and drawdown is whatever the single point implies (0.0).
    """
    eq = pd.Series([e for e in equity if e is not None], dtype="float64")
    n = int(eq.size)
    max_dd = drawdown(eq.tolist())
    if n < 2:
        return PerfSummary(ann_return=0.0, ann_vol=0.0, sharpe=0.0,
                           max_drawdown=max_dd, n_marks=n)

    ppy = periods_per_year if periods_per_year is not None else _infer_periods_per_year(timestamps, n)
    rets = eq.pct_change().dropna()

    # Annualized return: geometric (CAGR) over the realized number of periods, annualized by ppy.
    total_growth = float(eq.iloc[-1] / eq.iloc[0])
    n_periods = n - 1
    ann_return = total_growth ** (ppy / n_periods) - 1.0 if total_growth > 0 else -1.0

    # A flat (or numerically near-flat) path has no measurable risk: report vol/Sharpe as 0.0
    # rather than letting floating-point dust in the denominator blow Sharpe up to ~1e15.
    sd_ret = float(rets.std(ddof=1)) if n_periods > 1 else 0.0
    vol_eps = 1e-12 * max(1.0, abs(float(rets.mean())))
    ann_vol = float(sd_ret * np.sqrt(ppy)) if sd_ret > vol_eps else 0.0

    excess = rets - rf / ppy
    sd_exc = float(excess.std(ddof=1)) if n_periods > 1 else 0.0
    sharpe = float(excess.mean() / sd_exc * np.sqrt(ppy)) if sd_exc > vol_eps else 0.0

    return PerfSummary(
        ann_return=float(ann_return),
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=float(max_dd),
        n_marks=n,
    )


def summarize_bot(store, *, rf: float = 0.0) -> PerfSummary:
    """Performance summary for a bot, read from its per-bot ``Store``.

    Thin wrapper: reads the equity time series (``read_equity_df``) and feeds the total-equity path
    and its timestamps to ``equity_summary``. ``rf`` is the annual risk-free Sharpe hurdle.
    """
    df = store.read_equity_df()
    if df.empty:
        return equity_summary([], None, rf=rf)
    return equity_summary(
        df["total_equity"].tolist(),
        df["ts"].tolist(),
        rf=rf,
    )
