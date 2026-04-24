"""Data loaders.

Stubs. Fill these in per source as needed. Prefer one small function per source
over a large multi-source dispatcher.

TODO:
- yfinance-based ETF / FX / futures-proxy prices
- FRED macro / rates pulls
- IBKR historical bars
- local CSV / parquet loader for manually-scraped data
"""

from __future__ import annotations

import pandas as pd


def load_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted close prices (index=date, columns=ticker).

    TODO: implement — first pass can use yfinance for ETFs / FX.
    """
    raise NotImplementedError("TODO: wire up a source (yfinance first)")


def load_returns(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Convenience wrapper: prices → simple returns."""
    from alpha_lab.analytics.returns import simple_returns

    prices = load_prices(tickers, start, end)
    return simple_returns(prices)
