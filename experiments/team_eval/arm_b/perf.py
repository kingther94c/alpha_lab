"""Bot performance summary — headline metrics over an equity time series.

Core (pure, unit-testable):
    perf_metrics(equity, periods_per_year, rf) -> PerfSummary

Store-backed wrapper:
    bot_perf(store, periods_per_year, rf) -> PerfSummary | None

Annualization: inferred from the MEDIAN inter-mark gap (not n/elapsed) so
restart gaps and sparse stretches do not poison the factor. The default
marks-per-year falls back to 365 when fewer than two timestamps are available.

Drawdown: delegates to ``quant_bot_manager.core.risk.drawdown`` so this summary
and the kill-switch agree on exactly one definition.

Sharpe: arithmetic (mean excess return / std) * sqrt(ppy), consistent with
``alpha_lab.analytics.returns`` and with the runner's RF_ANNUAL hurdle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_bot_manager.core import risk
from quant_bot_manager.core.store import Store

# Cost-of-cash hurdle that matches runner.RF_ANNUAL — used as the default Sharpe
# risk-free rate so "beat zero" is never confused with "beat cash".
_RF_ANNUAL: float = 0.04

# Fallback marks-per-year when cadence can't be inferred (crypto 365-day calendar).
_DEFAULT_PPY: float = 365.0


@dataclass(frozen=True)
class PerfSummary:
    """Four headline metrics for a bot equity path."""
    ann_return: float    # annualised geometric return (fraction, e.g. 0.15 = 15 %); NaN if indeterminate
    ann_vol: float       # annualised volatility of EXCESS period returns (fraction); 0.0 if flat
    sharpe: float        # (mean excess return / std excess return) * sqrt(ppy); 0.0 if flat/insufficient
    max_drawdown: float  # worst peak-to-trough drawdown (<= 0; e.g. -0.20 = -20 %)


def _infer_ppy(timestamps: list) -> float:
    """Marks-per-year implied by the MEDIAN gap between consecutive timestamps.

    Falls back to ``_DEFAULT_PPY`` when there are fewer than two usable
    timestamps or the spacing is degenerate (all-zero or non-finite).
    Using the median (not mean) makes the estimate robust to restart gaps
    and sparse stretches that would otherwise inflate mean-based inference.
    """
    if len(timestamps) < 2:
        return _DEFAULT_PPY
    ts = pd.to_datetime(pd.Series(timestamps)).sort_values()
    gaps = ts.diff().dropna().dt.total_seconds().to_numpy()
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return _DEFAULT_PPY
    median_gap_s = float(np.median(gaps))
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median_gap_s


def perf_metrics(
    equity: list[float],
    periods_per_year: float = _DEFAULT_PPY,
    rf: float = _RF_ANNUAL,
) -> PerfSummary:
    """Compute headline performance metrics from a sequence of equity marks.

    ``equity`` is an ordered list of total-equity floats (oldest-first). NaN /
    None values are filtered out before any computation.

    ``periods_per_year`` converts observation frequency to annualised figures.
    ``rf`` is an annual risk-free rate deducted per period in the Sharpe
    numerator (default ``_RF_ANNUAL`` = 0.04, matching runner.RF_ANNUAL).

    Fewer than two non-NaN points: ann_return/ann_vol/sharpe are NaN,
    max_drawdown is 0.0.

    Non-positive starting equity (de-fauceted path gone to/below zero):
    ann_return is NaN; vol/Sharpe are computed on the surviving returns only.
    """
    vals = [e for e in equity if e is not None and math.isfinite(e)]
    max_dd = risk.drawdown(vals)   # reuse kill-switch definition; guards peak > 0 internally

    if len(vals) < 2:
        nan = float("nan")
        return PerfSummary(ann_return=nan, ann_vol=nan, sharpe=nan, max_drawdown=max_dd)

    arr = np.array(vals, dtype=float)
    n_periods = len(arr) - 1

    # --- annualised geometric return -----------------------------------------
    total_growth = arr[-1] / arr[0] if arr[0] > 0 else None
    if total_growth is not None and total_growth > 0:
        ann_ret = float(total_growth ** (periods_per_year / n_periods) - 1.0)
    else:
        ann_ret = float("nan")   # non-positive equity: geometric CAGR is undefined

    # --- Sharpe (arithmetic, excess-of-hurdle) --------------------------------
    # Consistent with alpha_lab.analytics.returns: mean(excess)/std(excess)*sqrt(ppy).
    # Two equity points -> one return; ddof=1 requires >=2 returns to avoid divide-by-zero.
    rets = arr[1:] / arr[:-1] - 1.0
    rf_per_period = rf / periods_per_year
    excess = rets - rf_per_period

    if n_periods < 2:
        # Single return: can't estimate vol (ddof=1 divides by zero); report valid metrics only.
        ann_vol = float("nan")
        sharpe = float("nan")
    else:
        sd_ret = float(np.std(rets, ddof=1))
        sd_exc = float(np.std(excess, ddof=1))
        # Vol floor: sub-penny numerical dust on a flat path must not manufacture a huge Sharpe.
        # Guard on annualized vol: if ann_vol < 1e-8 (0.000001%), the path is effectively flat.
        ann_vol_candidate = sd_ret * math.sqrt(periods_per_year)
        if ann_vol_candidate < 1e-8:
            ann_vol = 0.0
            sharpe = 0.0
        else:
            ann_vol = ann_vol_candidate
            sharpe = float(excess.mean() / sd_exc * math.sqrt(periods_per_year))

    return PerfSummary(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_drawdown=max_dd)


def bot_perf(
    store: Store,
    periods_per_year: float | None = None,
    rf: float = _RF_ANNUAL,
) -> PerfSummary | None:
    """Return headline metrics for a bot by reading its equity history from the store.

    Uses de-fauceted equity (``all_strategy_equity``) so demo top-ups do not
    inflate the base and hide real drawdowns.

    When ``periods_per_year`` is None the cadence is inferred from the MEDIAN
    inter-mark gap of the timestamps that correspond to non-None equity rows.
    Using the same filtered set for both the timestamps (cadence) and the
    equity (returns) keeps the annualization denominator consistent with the
    return count — a single failed mark (None total) cannot desync them.

    Returns None when fewer than two usable equity marks are available.
    """
    df = store.read_equity_df()

    # Align timestamps and equity: drop rows where total_equity is NaN/None so
    # the ppy inference and the equity path both operate on exactly the same rows.
    df = df.dropna(subset=["total_equity"])
    if len(df) < 2:
        return None

    # Apply de-faucet offset to the filtered totals (mirrors all_strategy_equity).
    offset = store.get_faucet_offset() or 0.0
    strategy_equity = (df["total_equity"] - offset).tolist()

    if periods_per_year is None:
        periods_per_year = _infer_ppy(df["ts"].tolist())

    return perf_metrics(strategy_equity, periods_per_year=periods_per_year, rf=rf)
