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

    Parameters
    ----------
    codes : FRED series id(s), e.g. ``"DGS10"`` or ``["DGS10", "DGS2"]``.
    start, end : optional ISO date strings for filtering (applied client-side).
    timeout : request timeout in seconds.

    Returns
    -------
    DataFrame indexed by date, one column per code. Missing observations are
    left as NaN (FRED uses ``.`` which is coerced here).
    """
    if isinstance(codes, str):
        codes = [codes]
    params = {"id": ",".join(codes)}
    headers = {}
    # Respect an API key if the user has one, but don't require it.
    if key := os.environ.get("FRED_API_KEY"):
        headers["X-API-Key"] = key

    resp = httpx.get(_FRED_CSV, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(
        StringIO(resp.text),
        parse_dates=["DATE"],
        na_values=["."],
    ).set_index("DATE")
    df.index.name = None

    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    return df.sort_index(axis=1)
