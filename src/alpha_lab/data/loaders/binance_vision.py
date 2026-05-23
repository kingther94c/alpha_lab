"""Binance Vision (data.binance.vision) public archive loader.

Downloads monthly ZIP archives of klines and (for perp only) funding rates,
parses them into wide OHLCV / funding panels, and caches the parsed result
to ``data/interim/binance/``.

Conventions
-----------
- All timestamps are tz-aware UTC.
- ``market='spot'`` reads from ``/spot/``; ``market='perp'`` reads from
  ``/futures/um/`` (USD-margined linear perpetuals).
- 1m archival cadence is the default. Resample to coarser intervals
  downstream as horizon research dictates.
- Monthly archives only on the first pass. The most recent ~1 month is
  typically not yet published as monthly — a daily-archive fallback is a
  TODO.

The returned :class:`OHLCVPanel` is a wide-DataFrame-compatible object:
``panel.close_panel()`` returns the legacy ``DatetimeIndex × symbol`` close
panel that the rest of the package accepts.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from alpha_lab.data.holdout import PMHoldout, enforce
from alpha_lab.data.intraday_calendar import expected_bars, to_pandas_freq
from alpha_lab.utils.cache import read_parquet, write_parquet
from alpha_lab.utils.paths import INTERIM_DIR, RAW_DIR, ensure_dir


BASE_URL = "https://data.binance.vision"
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

Market = Literal["spot", "perp"]


# 11-column Binance kline schema. Older files have 11 cols, newer 12 (extra "ignore");
# we always read the first 11 by position.
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote",
]
_OHLCV_OUT_COLS = [
    "open", "high", "low", "close", "volume", "quote_volume", "n_trades", "taker_buy_base",
]


def _market_path(market: Market) -> str:
    if market == "spot":
        return "spot"
    if market == "perp":
        return "futures/um"
    raise ValueError(f"Unknown market={market!r}; expected 'spot' or 'perp'.")


# --- Listing (S3 XML) -------------------------------------------------------

def available_history(
    market: Market,
    symbol: str,
    interval: str,
    *,
    kind: str = "klines",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """List monthly archives published on data.binance.vision.

    Parameters
    ----------
    market : 'spot' or 'perp'.
    symbol : e.g. 'BTCUSDT'.
    interval : e.g. '1m', '5m', '1h'. Ignored for ``kind='fundingRate'``.
    kind : 'klines' (default) or 'fundingRate' (perp only).

    Returns
    -------
    DataFrame sorted ascending by ``period`` with columns
    ``period`` (YYYY-MM), ``url``, ``size_bytes``.
    """
    if kind == "klines":
        prefix = f"data/{_market_path(market)}/monthly/klines/{symbol}/{interval}/"
        infix = f"{symbol}-{interval}-"
    elif kind == "fundingRate":
        if market != "perp":
            raise ValueError("fundingRate is only available for market='perp'.")
        prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
        infix = f"{symbol}-fundingRate-"
    else:
        raise ValueError(f"Unknown kind={kind!r}; expected 'klines' or 'fundingRate'.")

    s = session or requests
    rows: list[dict] = []
    token: str | None = None
    while True:
        params: dict = {"prefix": prefix, "list-type": "2"}
        if token:
            params["continuation-token"] = token
        resp = s.get(LIST_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for c in root.findall("s3:Contents", _NS):
            key = c.findtext("s3:Key", default="", namespaces=_NS)
            if not key.endswith(".zip") or ".CHECKSUM" in key:
                continue
            stem = key.rsplit("/", 1)[-1].removesuffix(".zip")
            period = stem.replace(infix, "")
            size = int(c.findtext("s3:Size", default="0", namespaces=_NS))
            rows.append({"period": period, "url": f"{BASE_URL}/{key}", "size_bytes": size})
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_NS) == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=_NS)
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


# --- ZIP cache + parsing ----------------------------------------------------

def _raw_root() -> Path:
    return RAW_DIR / "binance_vision"


def _interim_root() -> Path:
    return INTERIM_DIR / "binance"


def _cache_zip_path(
    market: Market, symbol: str, interval: str, period: str,
    *, kind: str = "klines",
) -> Path:
    base = _raw_root() / market / symbol
    if kind == "klines":
        return base / interval / f"{symbol}-{interval}-{period}.zip"
    if kind == "fundingRate":
        return base / "funding" / f"{symbol}-fundingRate-{period}.zip"
    raise ValueError(f"Unknown kind={kind!r}")


def _download_zip(
    url: str, dest: Path, *, refresh: bool = False,
    session: requests.Session | None = None,
) -> Path:
    if dest.exists() and not refresh:
        return dest
    ensure_dir(dest.parent)
    s = session or requests
    resp = s.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _read_csv_in_zip(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
        if csv_name is None:
            raise ValueError(f"No CSV inside {zip_path}")
        return zf.read(csv_name)


def _has_header_row(raw_bytes: bytes) -> bool:
    """True if the CSV starts with a header row (newer files) rather than
    numeric data (older files)."""
    first = raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
    if not first:
        return False
    first_field = first.split(",")[0].lstrip("-").lstrip("+")
    return not first_field.isdigit()


def parse_kline_zip(zip_path: Path) -> pd.DataFrame:
    """Parse one kline ZIP into a tz-aware (UTC) DataFrame indexed by ``open_time``.

    Returned columns: ``open, high, low, close, volume, quote_volume, n_trades,
    taker_buy_base``.
    """
    raw = _read_csv_in_zip(zip_path)
    has_header = _has_header_row(raw)
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        usecols=range(11),
        low_memory=False,
    )
    df.columns = _KLINE_COLUMNS  # normalize regardless of source naming
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
    df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").fillna(0).astype("int64")
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[_OHLCV_OUT_COLS]


def parse_funding_zip(zip_path: Path) -> pd.DataFrame:
    """Parse one funding-rate ZIP into a tz-aware (UTC) DataFrame indexed by ``calc_time``.

    Returned columns: ``funding_interval_hours, last_funding_rate``.
    """
    raw = _read_csv_in_zip(zip_path)
    has_header = _has_header_row(raw)
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, low_memory=False)
    # Some funding files have 3 cols, some have only 2; normalize.
    cols = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    df = df.iloc[:, : len(cols)].copy()
    df.columns = cols[: df.shape[1]]
    if "funding_interval_hours" not in df.columns:
        df["funding_interval_hours"] = 8
    df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce").astype("int64")
    df["funding_interval_hours"] = pd.to_numeric(
        df["funding_interval_hours"], errors="coerce",
    ).fillna(8).astype("int64")
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce").astype("float64")
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("calc_time").sort_index()
    return df[["funding_interval_hours", "last_funding_rate"]]


# --- High-level load API ----------------------------------------------------

@dataclass(frozen=True)
class OHLCVPanel:
    """A collection of per-symbol OHLCV DataFrames at a single interval.

    The legacy wide close panel is available via ``close_panel()``; the
    underlying per-symbol frames keep full OHLCV for VWAP / ATR / volume
    feature work.
    """

    frames: dict[str, pd.DataFrame]
    market: str
    interval: str

    def close_panel(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        return pd.DataFrame({sym: df["close"] for sym, df in self.frames.items()})

    def field(self, name: str) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        return pd.DataFrame({sym: df[name] for sym, df in self.frames.items()})

    @property
    def symbols(self) -> list[str]:
        return list(self.frames.keys())


def _months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Return ``YYYY-MM`` strings whose month overlaps ``[start, end)``."""
    s = start.tz_convert("UTC") if start.tz is not None else start.tz_localize("UTC")
    e = end.tz_convert("UTC") if end.tz is not None else end.tz_localize("UTC")
    s = s.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    periods: list[str] = []
    cur = s
    while cur < e:
        periods.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return periods


def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def load_klines(
    symbols: list[str] | str,
    interval: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    market: Market = "perp",
    refresh: bool = False,
    holdout: PMHoldout | None = None,
    session: requests.Session | None = None,
) -> OHLCVPanel:
    """Download / parse / cache monthly kline archives and return an :class:`OHLCVPanel`.

    Per-symbol parquet caches live at
    ``data/interim/binance/{market}_{symbol}_{interval}.parquet`` and are
    re-used between calls. The final per-symbol frame is sliced to
    ``[start, end)`` and run through :func:`alpha_lab.data.holdout.enforce`.

    A missing monthly archive (HTTP 404) is silently skipped — useful when
    the start of the requested window predates Binance's coverage.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    s_ts = _to_utc(start)
    e_ts = _to_utc(end)
    periods = _months_in_range(s_ts, e_ts)
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        cache_path = _interim_root() / f"{market}_{sym}_{interval}.parquet"
        if cache_path.exists() and not refresh:
            sym_df = read_parquet(cache_path)
            if (
                len(sym_df) > 0
                and sym_df.index.min() <= s_ts
                and sym_df.index.max() >= e_ts - pd.Timedelta(to_pandas_freq(interval))
            ):
                frames[sym] = enforce(
                    sym_df.loc[(sym_df.index >= s_ts) & (sym_df.index < e_ts)],
                    holdout=holdout,
                    context=f"binance_vision.load_klines[{sym}]",
                )
                continue
        parts: list[pd.DataFrame] = []
        for period in periods:
            url = (
                f"{BASE_URL}/data/{_market_path(market)}/monthly/klines/"
                f"{sym}/{interval}/{sym}-{interval}-{period}.zip"
            )
            dest = _cache_zip_path(market, sym, interval, period, kind="klines")
            try:
                _download_zip(url, dest, refresh=refresh, session=session)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue
                raise
            parts.append(parse_kline_zip(dest))
        if not parts:
            frames[sym] = pd.DataFrame(columns=_OHLCV_OUT_COLS)
            continue
        sym_df = pd.concat(parts).sort_index()
        sym_df = sym_df[~sym_df.index.duplicated(keep="first")]
        ensure_dir(cache_path.parent)
        write_parquet(sym_df, cache_path)
        sliced = sym_df.loc[(sym_df.index >= s_ts) & (sym_df.index < e_ts)]
        frames[sym] = enforce(
            sliced, holdout=holdout, context=f"binance_vision.load_klines[{sym}]",
        )
    return OHLCVPanel(frames=frames, market=market, interval=interval)


def load_funding(
    symbols: list[str] | str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    refresh: bool = False,
    holdout: PMHoldout | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Wide USD-M perp funding-rate panel.

    Returns a DataFrame indexed by funding timestamp (UTC), columns = symbol,
    values = ``last_funding_rate`` as a decimal fraction (e.g. 0.0001 = 0.01%).
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    s_ts = _to_utc(start)
    e_ts = _to_utc(end)
    periods = _months_in_range(s_ts, e_ts)
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        parts = []
        for period in periods:
            url = (
                f"{BASE_URL}/data/futures/um/monthly/fundingRate/"
                f"{sym}/{sym}-fundingRate-{period}.zip"
            )
            dest = _cache_zip_path("perp", sym, "", period, kind="fundingRate")
            try:
                _download_zip(url, dest, refresh=refresh, session=session)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue
                raise
            parts.append(parse_funding_zip(dest))
        if not parts:
            series[sym] = pd.Series(dtype="float64")
            continue
        sym_df = pd.concat(parts).sort_index()
        sym_df = sym_df[~sym_df.index.duplicated(keep="first")]
        series[sym] = sym_df["last_funding_rate"]
    panel = pd.DataFrame(series).sort_index()
    sliced = panel.loc[(panel.index >= s_ts) & (panel.index < e_ts)]
    return enforce(sliced, holdout=holdout, context="binance_vision.load_funding")


# --- Quality report ---------------------------------------------------------

def data_quality_report(panel: OHLCVPanel) -> pd.DataFrame:
    """Per-(symbol, year-month) quality summary for *panel*.

    Columns: ``symbol, year_month, n_bars_expected, n_bars_actual,
    n_duplicates, n_gaps, max_gap_bars, n_zero_volume_bars, first_ts,
    last_ts, ok``.

    ``ok`` is True when the month is at least 99% complete and has no
    duplicate timestamps. Tweak downstream if you want a stricter bar.
    """
    rows: list[dict] = []
    step = pd.Timedelta(to_pandas_freq(panel.interval))
    for sym, df in panel.frames.items():
        if df.empty:
            continue
        # Compute month tags from the tz-aware index without going through
        # PeriodIndex (which drops tz and warns).
        month_keys = df.index.strftime("%Y-%m")
        for m in pd.Index(month_keys).unique():
            month_df = df[month_keys == m]
            start = pd.Timestamp(f"{m}-01", tz="UTC")
            # advance one month for the exclusive end
            year, mon = int(m[:4]), int(m[5:7])
            if mon == 12:
                end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            else:
                end = pd.Timestamp(f"{year}-{mon + 1:02d}-01", tz="UTC")
            expected = expected_bars(start, end, panel.interval)
            n_exp = len(expected)
            n_act = len(month_df)
            n_dup = int(month_df.index.duplicated().sum())
            missing = expected.difference(month_df.index)
            if len(missing) > 0:
                grouped_n: list[int] = []
                cur_n = 1
                for i in range(1, len(missing)):
                    if missing[i] == missing[i - 1] + step:
                        cur_n += 1
                    else:
                        grouped_n.append(cur_n)
                        cur_n = 1
                grouped_n.append(cur_n)
                n_gaps = len(grouped_n)
                max_gap = max(grouped_n)
            else:
                n_gaps = 0
                max_gap = 0
            n_zero_vol = int((month_df["volume"] == 0).sum())
            ok = bool((n_act >= n_exp * 0.99) and (n_dup == 0))
            rows.append({
                "symbol": sym,
                "year_month": str(m),
                "n_bars_expected": n_exp,
                "n_bars_actual": n_act,
                "n_duplicates": n_dup,
                "n_gaps": n_gaps,
                "max_gap_bars": max_gap,
                "n_zero_volume_bars": n_zero_vol,
                "first_ts": str(month_df.index.min()),
                "last_ts": str(month_df.index.max()),
                "ok": ok,
            })
    return pd.DataFrame(rows)
