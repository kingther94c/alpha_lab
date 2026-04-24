"""yfinance-backed price loader.

Free, rate-limited, fine as a default for ETFs / FX / futures proxies.
Swap in a better source later by adding another module alongside this one.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_prices(
    tickers: list[str] | str,
    start: str,
    end: str | None = None,
    *,
    field: str = "Close",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download a wide DataFrame of prices (index=date, columns=ticker).

    Parameters
    ----------
    tickers : one or more Yahoo tickers (e.g. ``"SPY"`` or ``["SPY", "IEF"]``).
    start, end : ISO date strings. ``end=None`` means "up to today".
    field : which OHLCV field to keep. Defaults to ``"Close"``; with
        ``auto_adjust=True`` (default) this is split/dividend-adjusted.
    auto_adjust : passed through to yfinance.

    Returns
    -------
    DataFrame with a DatetimeIndex and one column per ticker, sorted by ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        return pd.DataFrame(columns=sorted(tickers))

    # Single ticker returns flat columns; multi-ticker returns a MultiIndex
    # with (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[field]
    else:
        prices = raw[[field]].rename(columns={field: tickers[0]})

    return prices.sort_index(axis=1).dropna(how="all")
