"""Bridge single-stock congressional trades onto a tradable sector-ETF universe.

The research plan builds the signal at the **GICS-sector** level and expresses it
through liquid sector ETFs (the 11 SPDRs), because the trader only deals in
ETFs/futures/options, never individual names. This module is that bridge:

- :func:`load_ticker_sector_map` — classify each traded stock ticker to a GICS
  sector (yfinance sector taxonomy, cached incrementally to parquet, with curated
  overrides for renamed/delisted tickers and ADRs).
- :func:`sector_etf_map` — map each GICS sector to its representative ETF, reusing
  the shared ``configs/us_sector_etf.csv`` universe.

Caveat (documented in the decision record): sector classification here is the
*current* yfinance label, not strictly point-in-time. Sector membership is far more
stable than index membership, so this is a mild assumption for a sector-flow signal;
the unmapped share is reported so its materiality is visible.
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from alpha_lab.utils.cache import read_parquet, write_parquet
from alpha_lab.utils.paths import CONFIGS_DIR, INTERIM_DIR

logger = logging.getLogger(__name__)

#: yfinance sector taxonomy → GICS sector names used in ``configs/us_sector_etf.csv``.
YF_TO_GICS = {
    "Technology": "Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
}

#: Curated overrides for tickers yfinance no longer resolves (renames / acquisitions)
#: or that are commonly mis-tagged. These always win over the live lookup.
TICKER_SECTOR_OVERRIDES = {
    "FB": "Communication Services",     # → META
    "GOOG": "Communication Services",
    "GOOGL": "Communication Services",
    "TWTR": "Communication Services",   # acquired/delisted
    "ATVI": "Communication Services",   # acquired by MSFT
    "DISCA": "Communication Services", "DISCK": "Communication Services",
    "VIAC": "Communication Services",   # → PARA
    "RTN": "Industrials",               # → RTX
    "CTXS": "Technology", "XLNX": "Technology", "MXIM": "Technology",
    "ANTM": "Health Care",              # → ELV
    "CERN": "Health Care", "ALXN": "Health Care",
    "CXO": "Energy", "WLL": "Energy", "XEC": "Energy", "NBL": "Energy",
    "MYL": "Health Care",               # → VTRS
    "CBS": "Communication Services", "TWX": "Communication Services",
    "BRK.B": "Financials", "BRK/B": "Financials",
}

_SECTOR_MAP_CACHE = INTERIM_DIR / "congress_ticker_sector_map.parquet"
_CURATED_CSV = CONFIGS_DIR / "congress_ticker_sector.csv"


@functools.lru_cache(maxsize=1)
def _curated_overrides() -> dict[str, str]:
    """Authoritative ticker→sector overrides: the curated CSV merged over the inline
    dict (CSV wins). These take precedence over yfinance and the live-lookup cache, so
    rate-limited / delisted high-flow names still classify deterministically."""
    csv_map: dict[str, str] = {}
    if _CURATED_CSV.exists():
        df = pd.read_csv(_CURATED_CSV, comment="#").dropna(subset=["ticker", "sector"])
        csv_map = {str(t).upper().strip(): str(s).strip()
                   for t, s in zip(df["ticker"], df["sector"], strict=True)}
    return {**TICKER_SECTOR_OVERRIDES, **csv_map}


def sector_etf_map() -> dict[str, str]:
    """``{GICS sector → representative signal ETF}`` from the shared universe CSV."""
    uni = pd.read_csv(CONFIGS_DIR / "us_sector_etf.csv")
    return dict(zip(uni["sector"], uni["signal_etf"], strict=True))


def gics_sectors() -> list[str]:
    """The 11 GICS sectors (order matches the universe CSV)."""
    return list(sector_etf_map().keys())


def _yf_sector(ticker: str, *, attempts: int = 2, pause: float = 0.2) -> str | None:
    """Best-effort live sector lookup for one ticker; ``None`` if unresolved.

    Retries with linear backoff because yfinance throttles bursts (a throttled
    call returns empty/raises, not a real "no sector"). A genuinely sector-less
    instrument (ETF/fund) just exhausts the attempts and returns ``None``.
    """
    import time

    import yfinance as yf

    for i in range(attempts):
        try:
            info = yf.Ticker(ticker).info
            sec = info.get("sector") or info.get("sectorKey")
            if sec:
                return sec
        except Exception:  # noqa: BLE001 — yfinance throws many shapes
            pass
        time.sleep(pause * (i + 1))
    return None


def _load_cache() -> pd.DataFrame:
    if _SECTOR_MAP_CACHE.exists():
        return read_parquet(_SECTOR_MAP_CACHE).set_index("ticker")
    return pd.DataFrame(columns=["yf_sector", "gics_sector"], index=pd.Index([], name="ticker"))


def load_ticker_sector_map(
    tickers,
    *,
    refresh: bool = False,
    retry_unknown: bool = False,
    use_yfinance: bool = True,
    max_workers: int = 6,
) -> pd.Series:
    """Map stock tickers → GICS sector (``"Unknown"`` if unresolved).

    Results are cached incrementally to ``data/interim/`` so the expensive
    yfinance pull happens once and grows as new tickers appear. Curated overrides
    always take precedence over the live lookup.

    Parameters
    ----------
    tickers : iterable of ticker strings.
    refresh : ignore the cache and re-fetch everything.
    retry_unknown : also re-fetch tickers cached as ``"Unknown"`` (these are
        usually prior throttled fetches). Run a few passes to fill coverage in.
    use_yfinance : if False, only overrides are applied (everything else
        ``"Unknown"``) — useful for fully-offline runs / tests.
    max_workers : thread pool size for the live lookups (kept modest to avoid
        yfinance throttling).

    Returns
    -------
    Series indexed by ticker, values = GICS sector name or ``"Unknown"``.
    """
    tickers = sorted({str(t).upper().strip() for t in tickers if pd.notna(t) and str(t).strip()})
    ov = _curated_overrides()
    cache = pd.DataFrame(columns=["yf_sector", "gics_sector"], index=pd.Index([], name="ticker")) \
        if refresh else _load_cache()

    cached_unknown = set(cache.index[cache["gics_sector"] == "Unknown"]) if retry_unknown else set()
    to_fetch = [
        t for t in tickers
        if t not in ov and (t not in set(cache.index) or t in cached_unknown)
    ]

    new_rows: dict[str, tuple[str | None, str]] = {}
    if use_yfinance and to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_yf_sector, t): t for t in to_fetch}
            for fut in as_completed(futs):
                t = futs[fut]
                yfs = fut.result()
                new_rows[t] = (yfs, YF_TO_GICS.get(yfs, "Unknown") if yfs else "Unknown")
        n_ok = sum(1 for _, g in new_rows.values() if g != "Unknown")
        logger.info("ticker→sector: resolved %d/%d fetched tickers", n_ok, len(to_fetch))
    elif not use_yfinance:
        new_rows = {t: (None, "Unknown") for t in to_fetch}

    if new_rows:
        add = pd.DataFrame(
            [{"ticker": t, "yf_sector": a, "gics_sector": g} for t, (a, g) in new_rows.items()]
        ).set_index("ticker")
        cache = pd.concat([cache.drop(index=[t for t in new_rows if t in cache.index]), add])
        write_parquet(cache.reset_index(), _SECTOR_MAP_CACHE)

    out = cache.reindex(tickers)["gics_sector"].fillna("Unknown")
    # Curated overrides win unconditionally (even over previously-cached live values).
    hits = [t for t in ov if t in out.index]
    if hits:
        out.loc[hits] = [ov[t] for t in hits]
    return out.rename("gics_sector")


def coverage_report(trades: pd.DataFrame, sector_of: pd.Series) -> dict:
    """Share of transactions and of |log-mid $ flow| that resolved to a sector."""
    sec = trades["ticker"].map(sector_of)
    resolved = sec.notna() & (sec != "Unknown")
    flow = trades["amount_logmid"].abs()
    return {
        "n_trades": int(len(trades)),
        "pct_trades_mapped": float(resolved.mean() * 100),
        "pct_flow_mapped": float(flow[resolved].sum() / flow.sum() * 100) if flow.sum() else float("nan"),
        "n_unmapped_tickers": int(trades.loc[~resolved, "ticker"].nunique()),
    }
