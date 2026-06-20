"""Event-study cumulative abnormal returns (CAR) around dated events.

Generic and reusable: given a table of ``(date, ticker[, sign])`` events and a wide
price panel, compute the average **market-adjusted** CAR path over a window around the
event, with per-horizon standard errors and t-stats.

Used for the congress single-name validation gate (research-plan Angle D): is there
measurable drift after a member trades (``transaction_date``) and — the part that
matters for us — *after public disclosure* (``filing_date``)? Abnormal return is the
simple market model with β=1: ``r_stock - r_benchmark``. Buys and sells are oriented by
``sign`` so a positive CAR means "the disclosed direction made money".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventStudyResult:
    """Average CAR path and significance around an event set."""

    car: pd.Series       # mean CAR by relative trading day (index −pre … +post)
    se: pd.Series        # standard error of the mean at each relative day
    tstat: pd.Series     # car / se
    n_events: int        # number of usable events
    matrix: pd.DataFrame # per-event oriented CAR paths (rows=event, cols=rel day)

    def drift(self, start: int = 0, end: int | None = None) -> dict:
        """Post-event drift from rel-day ``start`` to ``end`` (default last), re-based
        to 0 at ``start``: mean, t-stat, n. Answers "is there drift *after* the event"."""
        end = self.matrix.columns.max() if end is None else end
        seg = self.matrix[end] - self.matrix[start]
        n = int(seg.notna().sum())
        mu = float(seg.mean())
        se = float(seg.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return {"from": start, "to": end, "mean_car": mu,
                "tstat": mu / se if se and se == se else float("nan"), "n": n}


def event_car(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    date_col: str = "filing_date",
    ticker_col: str = "ticker",
    sign_col: str | None = "sign",
    benchmark: str = "SPY",
    pre: int = 5,
    post: int = 20,
) -> EventStudyResult:
    """Average market-adjusted CAR around events.

    Parameters
    ----------
    events : rows with ``date_col``, ``ticker_col`` and optionally ``sign_col``
        (+1 buy / −1 sell — used to orient the CAR; absent → all treated as +1).
    prices : wide price panel covering the event tickers and ``benchmark``.
    date_col : event date column (``filing_date`` for the tradeable question,
        ``transaction_date`` for the "did information exist" question).
    benchmark : column subtracted to form abnormal returns (β=1 market model).
    pre, post : window half-widths in trading days.

    Returns
    -------
    EventStudyResult — empty (n_events=0) if nothing is usable.
    """
    rets = prices.pct_change()
    idx = rets.index
    bench = rets[benchmark] if benchmark in rets.columns else pd.Series(0.0, index=idx)
    rel = list(range(-pre, post + 1))

    paths: list[np.ndarray] = []
    for _, e in events.iterrows():
        tk = e[ticker_col]
        if tk not in rets.columns:
            continue
        d = pd.Timestamp(e[date_col])
        if pd.isna(d):
            continue
        pos = idx.searchsorted(d, side="left")  # first trading day >= event date
        if pos - pre < 0 or pos + post >= len(idx):
            continue
        sl = slice(pos - pre, pos + post + 1)
        ar = rets[tk].iloc[sl].to_numpy() - bench.iloc[sl].to_numpy()
        if np.isnan(ar).any() or len(ar) != len(rel):
            continue
        s = e[sign_col] if (sign_col and pd.notna(e.get(sign_col))) else 1.0
        paths.append(np.cumsum(ar) * (1.0 if s >= 0 else -1.0))

    if not paths:
        empty = pd.Series(dtype="float64")
        return EventStudyResult(empty, empty, empty, 0, pd.DataFrame(columns=rel))

    matrix = pd.DataFrame(paths, columns=rel)
    car = matrix.mean()
    se = matrix.std(ddof=1) / np.sqrt(len(matrix))
    return EventStudyResult(car=car, se=se, tstat=car / se, n_events=len(matrix), matrix=matrix)
