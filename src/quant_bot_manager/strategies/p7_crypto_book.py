"""P7 multi-strategy book — the research↔execution handoff.

Wraps the research strategy ``alpha_lab.backtest.crypto_book`` into a live target-weight
function for a Bot. Market data comes from the live Binance API (ccxt, daily, leak-safe through
*yesterday's* close) since the Vision archives the backtest uses lag ~1 month. The signal logic
itself is unchanged research code — that is the whole point of the two-leg split.
"""
from __future__ import annotations

import datetime as dt

import ccxt
import pandas as pd

from alpha_lab.backtest import crypto_book as cb
from alpha_lab.data.loaders.yfinance import load_prices

PERP_CCXT = {"BTC.p": "BTC/USDT:USDT", "ETH.p": "ETH/USDT:USDT",
             "SOL.p": "SOL/USDT:USDT", "BNB.p": "BNB/USDT:USDT"}
SPOT_CCXT = {"BTC.s": "BTC/USDT", "ETH.s": "ETH/USDT"}


def build_live_bookdata(lookback_days: int = 420) -> cb.BookData:
    """A crypto_book.BookData built from live Binance API data (daily, through yesterday)."""
    spot_ex = ccxt.binance({"enableRateLimit": True})
    fut_ex = ccxt.binanceusdm({"enableRateLimit": True})

    def closes(ex, sym):
        d = ex.fetch_ohlcv(sym, "1d", limit=lookback_days)
        return pd.Series({pd.Timestamp(r[0], unit="ms", tz="UTC").normalize(): float(r[4]) for r in d})

    perp_close = pd.DataFrame({leg: closes(fut_ex, s) for leg, s in PERP_CCXT.items()})
    spot_close = pd.DataFrame({leg: closes(spot_ex, s) for leg, s in SPOT_CCXT.items()})
    grid = perp_close.index.union(spot_close.index)
    today = dt.datetime.now(dt.UTC).date().isoformat()
    grid = grid[grid < pd.Timestamp(today, tz="UTC")]            # drop today's forming bar -> leak-safe
    perp_close, spot_close = perp_close.reindex(grid).ffill(), spot_close.reindex(grid).ffill()
    prices = pd.concat([spot_close, perp_close], axis=1)

    def funding_hist(sym):
        d = fut_ex.fetch_funding_rate_history(sym, limit=1000)
        return pd.Series({pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in d})

    funding = pd.DataFrame({leg: funding_hist(s) for leg, s in PERP_CCXT.items()}).sort_index()
    df_fund = cb._daily_funding(funding, grid)

    hyg = load_prices("HYG", str(grid.min().date()), None)["HYG"]
    hyg.index = pd.DatetimeIndex(hyg.index).tz_localize("UTC")
    hyg = hyg.reindex(grid).ffill()

    naive = grid.tz_localize(None).normalize()
    rf_daily = pd.Series([cb.RF_FALLBACK.get(d.year, 0.04) / 365 for d in naive], index=grid).astype(float)
    return cb.BookData(grid=grid, perp_close=perp_close, spot_close=spot_close, funding=funding,
                       df_fund=df_fund, hyg=hyg, rf_daily=rf_daily, prices=prices,
                       rf_source="fallback", macro_source="yfinance HYG (live)")


def latest_targets(method: str = "equal_capital"):
    """Return (target_weights: Series, asof: Timestamp, last_px: dict) for execution today."""
    bd = build_live_bookdata()
    tgt = cb.latest_target_weights(bd, method=method)
    last_px = {**{lg: float(bd.spot_close[lg].iloc[-1]) for lg in SPOT_CCXT if lg in bd.spot_close},
               **{lg: float(bd.perp_close[lg].iloc[-1]) for lg in PERP_CCXT if lg in bd.perp_close}}
    return tgt, bd.grid.max(), last_px
