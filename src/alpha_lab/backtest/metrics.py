"""Summary statistics for a strategy return stream."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.analytics.returns import annualized_vol, drawdown, sharpe


def summary(returns: pd.Series, periods: int = 252) -> dict[str, float]:
    """Return the common performance stats for a return stream.

    Keys: CAGR, AnnVol, Sharpe, Sortino, MaxDD, Calmar, HitRate, NPeriods.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    n = len(r)
    total = float((1 + r).prod())
    cagr = total ** (periods / n) - 1
    vol = float(annualized_vol(r, periods=periods))
    shp = float(sharpe(r, periods=periods))

    downside = r[r < 0]
    sortino = float((r.mean() / downside.std()) * np.sqrt(periods)) if len(downside) > 1 else float("nan")

    dd = drawdown(r)
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    return {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": shp,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "HitRate": float((r > 0).mean()),
        "NPeriods": n,
    }


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Year × month matrix of compounded monthly returns, plus a YTD column."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()

    monthly = (1 + r).resample("ME").prod() - 1
    df = monthly.to_frame("r")
    df["year"] = df.index.year
    df["month"] = df.index.month
    table = df.pivot(index="year", columns="month", values="r")
    table.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns]

    yearly = (1 + r).resample("YE").prod() - 1
    yearly.index = yearly.index.year
    table["YTD"] = yearly.reindex(table.index)
    return table
