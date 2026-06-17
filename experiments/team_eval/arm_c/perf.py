"""Performance summary for a paper-trading bot's equity curve.

Two layers, kept decoupled so the math is unit-testable without a live bot:

- ``equity_perf`` — a **pure** function over an equity series (numbers in, dataclass out).
  No store, no I/O. This is the core; test it directly.
- ``bot_perf`` — a thin wrapper that reads a bot's recorded equity from
  ``quant_bot_manager.core.store.Store`` and hands it to ``equity_perf``.

Four headline metrics, following the repo's house conventions in
``alpha_lab.analytics.returns`` (sqrt-of-periods annualization; peak-to-trough drawdown):

- annualized return  — CAGR over the elapsed horizon (uses real timestamps when present,
  since paper marks aren't guaranteed evenly spaced).
- annualized vol     — std of per-period simple returns * sqrt(periods).
- Sharpe             — mean(excess) / std(excess) * sqrt(periods); ``rf`` is per-period.
- max drawdown       — worst peak-to-trough on the equity curve, as a negative fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Trading days/year — the repo's default annualization factor (alpha_lab.analytics.returns).
TRADING_DAYS = 252


@dataclass(frozen=True)
class PerfSummary:
    """Headline performance of an equity curve. All values are fractions (0.1 == 10%)."""

    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    n_obs: int


def _years(ts: pd.Series, n_steps: int, periods: int) -> float:
    """Elapsed years across the curve. Uses the real first/last timestamp span when given
    (paper marks can be irregular); else assumes ``n_steps`` evenly-spaced periods.

    The timestamps are sorted first, so an out-of-order series still yields a non-negative
    span rather than silently collapsing to the even-spacing fallback. If the span is still
    non-positive after sorting (e.g. every mark shares one calendar instant), we fall back to
    even spacing — the only honest choice when there is no elapsed time to measure."""
    if ts is not None and len(ts) >= 2:
        stamps = pd.to_datetime(pd.Series(list(ts))).sort_values()
        span = stamps.iloc[-1] - stamps.iloc[0]
        days = span.total_seconds() / 86_400.0
        if days > 0:
            return days / 365.25
    return n_steps / periods


def _max_drawdown(eq: pd.Series) -> float:
    """Worst peak-to-trough as a negative fraction, clamped to [-1, 0].

    A drawdown can't be worse than -100% (you can't lose more than the running peak), but the
    naive ``eq/peak - 1`` formula breaks once equity crosses zero — e.g. a peak of 100 then a
    trough of -10 gives -1.1. Clamping keeps the metric a valid fraction for the leveraged
    perp/futures case where the paper book can go non-positive."""
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    return max(dd, -1.0)


def equity_perf(
    equity: pd.Series | np.ndarray,
    ts: pd.Series | None = None,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
) -> PerfSummary:
    """Compute headline metrics from an equity (mark-to-market total) curve.

    Pure: numbers in, ``PerfSummary`` out. ``equity`` is the level series (not returns);
    ``ts`` is the matching timestamps (optional, used only to time-scale the CAGR);
    ``rf`` is a per-period risk-free rate; ``periods`` is the annualization factor.

    Degenerate inputs return NaN metrics rather than raising — a half-warm bot must not
    crash the cockpit. "Degenerate" means: fewer than 3 valid points (i.e. <2 returns, too
    few to annualize a horizon or estimate a std), a non-positive starting equity, or zero
    variance. Equity that touches zero or goes negative also yields NaN return/vol/Sharpe,
    since growth ratios and a percent-change series are undefined once the curve crosses
    zero (a leveraged perp/futures paper book can do this).

    Interior NaNs are NOT bridged: a return that straddles a missing mark is dropped, not
    fabricated across the hole. (``n_obs`` and the CAGR endpoints still use the valid marks,
    so leading/trailing NaNs remain harmless.)
    """
    raw = pd.Series(np.asarray(equity, dtype=float))
    # Returns are computed on the raw series so a return adjacent to a NaN mark is itself NaN
    # and gets dropped — we never concatenate across an interior hole into one jumbo return.
    rets = raw.pct_change().dropna()

    eq = raw.dropna()  # valid marks only: drives n_obs, the CAGR endpoints, and drawdown.
    n = int(eq.size)
    # Need >=2 returns (n>=3) to both annualize a horizon and estimate a std. Applying this to
    # ALL annualized metrics (not just vol/Sharpe) avoids a one-step curve extrapolating a
    # single +10% mark to 252 periods while risk reads NaN.
    if n < 3 or eq.iloc[0] <= 0:
        # Drawdown is still well-defined for any all-positive curve, even a 2-point one;
        # report it when we can, but never let it exceed -100% (see below).
        max_dd = _max_drawdown(eq) if n >= 2 and bool((eq > 0).all()) else np.nan
        return PerfSummary(np.nan, np.nan, np.nan, max_dd, n)

    # Max drawdown: worst peak-to-trough on the curve (negative fraction in [-1, 0]).
    max_dd = _max_drawdown(eq)

    # A curve that crosses zero makes pct_change (and the growth ratio below) meaningless:
    # the "return" across a sign flip is a nonsense large number. Treat the return-based
    # metrics as undefined in that case rather than emitting garbage; drawdown is already
    # clamped to [-1, 0] above.
    if not bool((eq > 0).all()):
        return PerfSummary(np.nan, np.nan, np.nan, max_dd, n)
    # After excluding straddle-the-hole returns we may have too few left to estimate a std.
    if rets.size < 2:
        return PerfSummary(np.nan, np.nan, np.nan, max_dd, n)

    # Annualized vol: std of per-period returns, scaled to a year.
    sd = float(rets.std(ddof=1))
    ann_vol = sd * np.sqrt(periods)

    # Sharpe: per-period excess mean / std, annualized (matches alpha_lab.analytics.returns).
    # A flat curve has only float jitter for std; treat that as zero variance -> undefined Sharpe
    # rather than reporting an absurd 1e13. The floor is in returns-space (a small absolute
    # value), not scaled by the equity level, so a genuine tiny drift on a large balance isn't
    # masked.
    excess = rets - rf
    sd_excess = float(excess.std(ddof=1))
    flat = not np.isfinite(sd_excess) or sd_excess <= 1e-12
    sharpe = np.nan if flat else float(excess.mean() / sd_excess * np.sqrt(periods))

    # Annualized return: CAGR over the elapsed horizon.
    years = _years(ts, n - 1, periods)
    total_growth = float(eq.iloc[-1] / eq.iloc[0])
    ann_return = total_growth ** (1.0 / years) - 1.0 if years > 0 else np.nan

    return PerfSummary(ann_return, ann_vol, sharpe, max_dd, n)


def bot_perf(store, rf: float = 0.0, periods: int = TRADING_DAYS) -> PerfSummary:
    """Performance summary for a bot, read from its ``Store`` equity history.

    Thin wrapper: pulls the recorded ``total`` equity series (and timestamps) and defers
    all math to :func:`equity_perf`. ``store`` is a ``quant_bot_manager.core.store.Store``.
    """
    df = store.read_equity_df()
    ts = df["ts"] if "ts" in df else None
    return equity_perf(
        df.get("total_equity", pd.Series(dtype=float)), ts=ts, rf=rf, periods=periods
    )
