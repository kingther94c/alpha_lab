"""FRED macro / rates loader via the public fredgraph CSV endpoint.

Uses httpx (already a core dep) — no extra package required. Most FRED series
(rates, CPI, spreads) are freely downloadable as CSV without an API key.
"""

from __future__ import annotations

import os
from io import StringIO

import httpx
import pandas as pd

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def load_series(
    codes: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download one or more FRED series and return a wide DataFrame.

    Each code is fetched on its own request and the results are outer-joined on
    the date index. The fredgraph.csv endpoint has no working comma-joined
    multi-id mode (``id=A,B`` returns a malformed CSV), so per-series fetches are
    the only reliable way to support more than one code.

    Parameters
    ----------
    codes : FRED series id(s), e.g. ``"DGS10"`` or ``["DGS10", "DGS2"]``.
    start, end : optional ISO date strings for filtering (applied client-side).
    timeout : per-request timeout in seconds.

    Returns
    -------
    DataFrame indexed by date, one column per code. Missing observations are
    left as NaN (FRED uses ``.`` which is coerced here).
    """
    if isinstance(codes, str):
        codes = [codes]
    headers = {}
    # Respect an API key if the user has one, but don't require it.
    if key := os.environ.get("FRED_API_KEY"):
        headers["X-API-Key"] = key

    frames = []
    for code in codes:
        resp = httpx.get(_FRED_CSV, params={"id": code}, headers=headers, timeout=timeout)
        resp.raise_for_status()
        frames.append(_parse_fred_csv(resp.text, code))

    df = _merge_fred_frames(frames)

    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    return df.sort_index(axis=1)


def _parse_fred_csv(text: str, code: str) -> pd.DataFrame:
    """Parse one fredgraph.csv response into a single-column, date-indexed frame.

    The endpoint returns ``observation_date,<ID>`` (older dumps used ``DATE``).
    The lone data column is renamed to ``code`` so callers always get exactly one
    column per requested id, and ``.`` placeholders are coerced to NaN.
    """
    df = pd.read_csv(StringIO(text), na_values=["."])
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = None
    df.columns = [code]
    return df


def _merge_fred_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join per-series frames on the date index, chronologically sorted.

    Dates present in only some series are filled with NaN, preserving the wide
    one-column-per-code return contract.
    """
    merged = pd.concat(frames, axis=1, join="outer", sort=False)
    return merged.sort_index()


def discount_rate_to_daily_rate(
    discount_rate: pd.Series,
    *,
    maturity_days: int = 91,
) -> pd.Series:
    """Convert a T-bill bank discount rate into a daily investment return.

    FRED DTB3 is quoted as a bank discount annual percentage rate. This first
    infers the bill price, then converts that holding-period return into a
    daily compounded rate over ``maturity_days``.
    """
    rate = pd.to_numeric(discount_rate, errors="coerce") / 100.0
    price = 1.0 - rate * maturity_days / 360.0
    daily_rate = (1.0 / price) ** (1.0 / maturity_days) - 1.0
    return daily_rate.rename(discount_rate.name)


def cash_total_return_index(
    discount_rate: pd.Series,
    *,
    base: float = 100.0,
    maturity_days: int = 91,
) -> pd.Series:
    """Build a cash total-return index from a T-bill discount-rate series.

    Accrual is applied between observed dates as ``day_diff * today_daily_rate``
    so weekends and holidays are carried by the next available observation.
    The first valid date starts at ``base``.
    """
    daily_rate = discount_rate_to_daily_rate(discount_rate, maturity_days=maturity_days).dropna()
    if daily_rate.empty:
        return pd.Series(dtype=float, name="Cash_TR")

    day_diff = daily_rate.index.to_series().diff().dt.days.fillna(0.0)
    accrual = 1.0 + daily_rate * day_diff
    index = accrual.cumprod() * base
    index.iloc[0] = base
    return index.rename("Cash_TR")


def load_cash_total_return_index(
    start: str | None = None,
    end: str | None = None,
    *,
    code: str = "DTB3",
    base: float = 100.0,
    maturity_days: int = 91,
    timeout: float = 30.0,
) -> pd.Series:
    """Download FRED DTB3 and return a cash total-return index."""
    rates = load_series(code, start=start, end=end, timeout=timeout)[code]
    return cash_total_return_index(rates, base=base, maturity_days=maturity_days)
