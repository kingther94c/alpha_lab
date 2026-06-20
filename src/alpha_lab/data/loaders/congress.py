"""US Congressional trading disclosure (STOCK Act / PTR) loader.

Under the *Stop Trading on Congressional Knowledge (STOCK) Act, 2012*, members of
Congress (and their spouses / dependents) must publicly file every securities
transaction over $1,000 within 45 days, as a **Periodic Transaction Report
(PTR)**. This module loads those disclosures into one normalized, tidy long
frame so the research leg can build sector / party positioning signals from them.

Point-in-time discipline (the cardinal rule here)
--------------------------------------------------
The **signal date is ``filing_date``** — the day a trade became *public* — never
``transaction_date``. A backtest may only act on information visible at the time;
the ~45-day disclosure lag is the whole reason fast event-copy has little edge
(see the research plan). Downstream signal code keys off ``filing_date``.

Data sources (resolved empirically — see the decision record)
-------------------------------------------------------------
Scraping raw filings end-to-end is impractical: House PTRs are scanned/handwritten
PDFs (OCR), and the Senate eFD portal is Akamai-bot-walled and session-gated. So,
exactly as the plan recommends, we use a **pre-parsed source for the backtest** and
the **official portals for ground-truth audit**:

- ``kadoa``  (default, primary) — kadoa.com's open, MIT-licensed normalized export
  of all three official portals (House Clerk + Senate eFD + OGE executive), 2012→
  present, carrying both ``transaction_date`` and ``filing_date`` plus a ``doc_url``
  back to the official PDF. One ~4 MB JSON file.
- ``senate_stockwatcher`` — Senate-only, independently parsed (different pipeline);
  used to cross-check the Senate subset.
- :func:`fetch_house_filing_index` — the official House Clerk annual XML index
  (``<YEAR>FD.zip``). This *is* "direct from the filings": it is the authoritative
  list of who filed a PTR and when (filer, date, DocID → PDF). Transaction details
  still live in the PDFs, so we use the index for a coverage / freshness **audit**
  of the pre-parsed data, not as a transaction source.

All datetimes are timezone-naive (America/New_York convention, per the repo's
research-artifact contracts). Amounts are USD; ``amount_logmid`` is the standard
log-midpoint (geometric-mean) position estimate for the disclosed range bucket.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

import httpx
import numpy as np
import pandas as pd

from alpha_lab.utils.cache import cached_parquet
from alpha_lab.utils.paths import RAW_DIR, ensure_dir

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------
# kadoa publishes a roster (filers.json) plus one file per filer holding that member's
# FULL transaction history. ``trades.json`` is only a recent-5000 view, so for a multi-
# year backtest we aggregate the per-filer files and join party/chamber/state from the
# roster (the per-filer trade rows omit those).
KADOA_BASE = (
    "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/HEAD/public/data/"
)
KADOA_FILERS_URL = KADOA_BASE + "filers.json"
KADOA_FILER_URL = KADOA_BASE + "filer/{filer_id}.json"
KADOA_TRADES_URL = KADOA_BASE + "trades.json"  # recent-5000 view only (not full history)
SENATE_SW_URL = (
    "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/"
    "HEAD/aggregate/all_transactions.json"
)
HOUSE_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
HOUSE_PTR_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"

_UA = {"User-Agent": "Mozilla/5.0 (alpha_lab research; +https://github.com/)"}

#: Canonical column order for the normalized long frame (one row per transaction).
CANONICAL_COLUMNS = [
    "filing_date",        # PIT signal date — when it became public
    "transaction_date",   # when the trade actually happened
    "member",
    "chamber",            # house | senate | executive
    "party",              # R | D | I | (NA)
    "state",
    "owner",              # self | spouse | joint | dependent | (NA)
    "ticker",
    "asset_name",
    "asset_type",         # ST = stock, etc.
    "direction",          # buy | sell | exchange | other
    "sign",               # +1 buy, -1 sell, 0 otherwise
    "amount_low",
    "amount_high",
    "amount_logmid",      # signed log-midpoint $ estimate (sign * sqrt(low*high))
    "branch",             # congress | executive
    "is_amendment",
    "source",             # kadoa | senate_stockwatcher
    "doc_url",            # link to the official filing (audit trail)
]


# --------------------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------------------
def amount_logmid(low: float | None, high: float | None) -> float:
    """Log-midpoint (geometric mean) of a disclosed amount range, in USD.

    PTRs report a *range* (e.g. ``$1,001–$15,000``), not an exact figure. The
    geometric mean ``sqrt(low*high)`` is the standard unbiased estimate on a log
    scale and avoids the upward bias of the arithmetic midpoint. Open-ended top
    buckets (``high`` missing/0) fall back to ``low``; a missing ``low`` falls
    back to ``high``. Returns ``nan`` if neither is usable.
    """
    lo = float(low) if low not in (None, "") and pd.notna(low) else np.nan
    hi = float(high) if high not in (None, "") and pd.notna(high) else np.nan
    if np.isnan(lo) and np.isnan(hi):
        return np.nan
    if np.isnan(hi) or hi <= 0:
        return lo
    if np.isnan(lo) or lo <= 0:
        return hi
    return float(np.sqrt(lo * hi))


def _normalize_direction(raw: object) -> tuple[str, int]:
    """Map a vendor transaction-type string to ``(direction, sign)``.

    ``buy`` → +1, ``sell`` → -1 (full or partial), ``exchange``/other → 0.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return "other", 0
    if "purchase" in s or s.startswith("buy") or s == "p":
        return "buy", 1
    if "sale" in s or "sell" in s or s in {"s", "sf", "sp"}:
        return "sell", -1
    if "exchange" in s:
        return "exchange", 0
    return "other", 0


def _normalize_owner(raw: object) -> str:
    """Normalize the owner field to {self, spouse, joint, dependent, (empty)}."""
    s = str(raw or "").strip().lower()
    if not s or s in {"nan", "none", "--", "self", "filer"}:
        return "self" if s in {"self", "filer"} else ""
    if "spouse" in s or s == "sp":
        return "spouse"
    if "joint" in s or s == "jt":
        return "joint"
    if "dependent" in s or "child" in s or s in {"dc", "dep"}:
        return "dependent"
    return s


def _to_naive_datetime(s: pd.Series) -> pd.Series:
    """Parse a string date Series to tz-naive datetime (mixed formats tolerated)."""
    out = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------
def _fetch_bytes(url: str, *, timeout: float = 90.0) -> bytes:
    with httpx.Client(headers=_UA, follow_redirects=True, timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _snapshot_raw(name: str, payload: bytes) -> None:
    """Persist a dated raw snapshot under ``data/raw/congress/`` for provenance.

    A fresh vendor download is a new source-of-truth artifact (one file per UTC
    day); existing snapshots are never modified.
    """
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d")
    out = ensure_dir(RAW_DIR / "congress") / f"{name}_{stamp}.json"
    if not out.exists():
        out.write_bytes(payload)


# --------------------------------------------------------------------------------------
# Normalize each source to CANONICAL_COLUMNS
# --------------------------------------------------------------------------------------
def _fetch_filer_roster() -> pd.DataFrame:
    """Fetch the kadoa filer roster (id, name, branch/chamber/party/state, counts)."""
    payload = _fetch_bytes(KADOA_FILERS_URL)
    _snapshot_raw("kadoa_filers", payload)
    return pd.DataFrame(json.loads(payload))


def _fetch_one_filer_trades(filer_id: str) -> list[dict]:
    """Fetch one filer's full trade history; return [] on any error (best-effort)."""
    try:
        payload = _fetch_bytes(KADOA_FILER_URL.format(filer_id=filer_id), timeout=60.0)
        obj = json.loads(payload)
    except Exception as exc:  # noqa: BLE001 — network/JSON; tolerate & skip one filer
        logger.warning("kadoa filer fetch failed for %s: %r", filer_id, exc)
        return []
    trades = obj.get("trades", []) if isinstance(obj, dict) else []
    for t in trades:
        t.setdefault("filer_id", filer_id)
    return trades


def _build_kadoa_trades(*, max_workers: int = 12) -> pd.DataFrame:
    """Aggregate every filer's full history into one normalized frame.

    Fetches the roster, then each ``filer/<id>.json`` concurrently, and joins
    party/chamber/state/member from the roster (per-filer trade rows omit them).
    """
    roster = _fetch_filer_roster()
    ids = roster["id"].dropna().astype(str).tolist()

    all_trades: list[dict] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed({ex.submit(_fetch_one_filer_trades, fid) for fid in ids}):
            rows = fut.result()
            if rows:
                all_trades.extend(rows)
            else:
                failures += 1
    if not all_trades:
        raise RuntimeError(
            "kadoa per-filer aggregation returned no trades — is the network reachable?"
        )
    df = pd.DataFrame(all_trades)
    logger.info(
        "kadoa: aggregated %d trades from %d/%d filers (%d empty/failed)",
        len(df), len(ids) - failures, len(ids), failures,
    )

    meta = roster.set_index("id")
    fid = df["filer_id"].astype(str)
    dir_sign = df["transaction_type"].map(_normalize_direction)
    ftype = df.get("filing_type", pd.Series("", index=df.index)).astype("string").str.lower()
    out = pd.DataFrame(
        {
            "filing_date": _to_naive_datetime(df["filing_date"]),
            "transaction_date": _to_naive_datetime(df["transaction_date"]),
            "member": fid.map(meta["full_name"]).astype("string"),
            "chamber": fid.map(meta["chamber"]).astype("string").str.lower(),
            "party": fid.map(meta["party"]).astype("string").str.upper(),
            "state": fid.map(meta["state"]).astype("string"),
            "owner": df["owner"].map(_normalize_owner).astype("string"),
            "ticker": df["ticker"].astype("string").str.upper().str.strip(),
            "asset_name": df["asset_name"].astype("string"),
            "asset_type": df["asset_type"].astype("string").str.upper(),
            "direction": [d for d, _ in dir_sign],
            "sign": [s for _, s in dir_sign],
            "amount_low": pd.to_numeric(df["amount_range_low"], errors="coerce"),
            "amount_high": pd.to_numeric(df["amount_range_high"], errors="coerce"),
            "branch": fid.map(meta["branch"]).astype("string").str.lower(),
            "is_amendment": ftype.str.contains("amend", na=False),
            "source": "kadoa",
            "doc_url": df["doc_url"].astype("string"),
        }
    )
    # Collapse exact duplicate disclosed lines (defensive; each filer file is distinct).
    out = out.drop_duplicates(
        subset=["member", "filing_date", "transaction_date", "ticker",
                "direction", "amount_low", "amount_high", "doc_url"]
    )
    return _finalize(out)


def _build_senate_sw_trades() -> pd.DataFrame:
    """Fetch the Senate Stock Watcher export (Senate-only cross-check source).

    Note: this feed carries only ``transaction_date`` (no machine filing date),
    so it is for *validation* of the Senate subset, not point-in-time signals.
    """
    payload = _fetch_bytes(SENATE_SW_URL)
    _snapshot_raw("senate_stockwatcher", payload)
    df = pd.DataFrame(json.loads(payload))
    dir_sign = df["type"].map(_normalize_direction)
    lohi = df["amount"].map(_parse_sw_amount_range)
    out = pd.DataFrame(
        {
            "filing_date": pd.NaT,  # not provided machine-readable in this feed
            "transaction_date": _to_naive_datetime(df["transaction_date"]),
            "member": df["senator"].astype("string"),
            "chamber": "senate",
            "party": pd.NA,
            "state": pd.NA,
            "owner": df["owner"].map(_normalize_owner).astype("string"),
            "ticker": df["ticker"].astype("string").str.upper().str.strip(),
            "asset_name": df.get("asset_description", pd.Series(index=df.index)).astype("string"),
            "asset_type": df.get("asset_type", pd.Series(index=df.index)).astype("string").str.upper(),
            "direction": [d for d, _ in dir_sign],
            "sign": [s for _, s in dir_sign],
            "amount_low": [lo for lo, _ in lohi],
            "amount_high": [hi for _, hi in lohi],
            "branch": "congress",
            "is_amendment": False,
            "source": "senate_stockwatcher",
            "doc_url": df.get("ptr_link", pd.Series(index=df.index)).astype("string"),
        }
    )
    return _finalize(out)


def _parse_sw_amount_range(label: object) -> tuple[float, float]:
    """Parse a Stock Watcher amount label like ``$15,001 - $50,000`` to (low, high)."""
    s = str(label or "").replace("$", "").replace(",", "").strip()
    if not s or s in {"--", "nan"}:
        return (np.nan, np.nan)
    parts = [p.strip() for p in s.split("-")]
    try:
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
        return (float(parts[0]), np.nan)
    except ValueError:
        return (np.nan, np.nan)


def _finalize(out: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns, order, and sort by the PIT signal date."""
    out["amount_logmid"] = [
        sign * amount_logmid(lo, hi) if not np.isnan(amount_logmid(lo, hi)) else np.nan
        for sign, lo, hi in zip(out["sign"], out["amount_low"], out["amount_high"], strict=True)
    ]
    out = out.reindex(columns=CANONICAL_COLUMNS)
    sort_key = out["filing_date"].fillna(out["transaction_date"])
    return out.assign(_k=sort_key).sort_values("_k").drop(columns="_k").reset_index(drop=True)


_BUILDERS = {
    "kadoa": ("congress_trades_kadoa", _build_kadoa_trades),
    "senate_stockwatcher": ("congress_trades_senate_sw", _build_senate_sw_trades),
}


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def load_congress_trades(
    start: str | None = None,
    end: str | None = None,
    *,
    source: str = "kadoa",
    refresh: bool = False,
    asset_types: tuple[str, ...] | None = ("ST",),
    chambers: tuple[str, ...] | None = ("house", "senate"),
    by: str = "filing_date",
) -> pd.DataFrame:
    """Load normalized congressional PTR transactions as a tidy long frame.

    Parameters
    ----------
    start, end : ISO date strings to clip on ``by`` (inclusive). ``None`` = open.
    source : ``"kadoa"`` (default, House+Senate+exec) or ``"senate_stockwatcher"``.
    refresh : force re-download + re-normalize (else use the parquet cache).
    asset_types : keep only these asset types (default ``("ST",)`` = single stocks,
        the tradable-via-sector-ETF universe). Pass ``None`` to keep all.
    chambers : keep only these chambers (default House + Senate = Congress). Pass
        ``None`` to keep executive-branch (OGE) filings too.
    by : which date column to clip ``start``/``end`` on — ``"filing_date"`` (the
        PIT signal date, default) or ``"transaction_date"``.

    Returns
    -------
    Long DataFrame with :data:`CANONICAL_COLUMNS`, sorted by ``filing_date``.
    """
    if source not in _BUILDERS:
        raise ValueError(f"source must be one of {sorted(_BUILDERS)}, got {source!r}")
    key, builder = _BUILDERS[source]
    df = cached_parquet(key, builder, refresh=refresh)

    if asset_types is not None:
        df = df[df["asset_type"].isin(asset_types)]
    if chambers is not None:
        df = df[df["chamber"].isin(chambers)]
    if start is not None:
        df = df[df[by] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[by] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def fetch_house_filing_index(
    years: int | list[int],
    *,
    filing_types: tuple[str, ...] | None = ("P",),
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch the **official** House Clerk annual financial-disclosure index.

    This is the authoritative, machine-readable list of filings straight from
    ``disclosures-clerk.house.gov`` — the "direct from filings" source. Each
    ``<YEAR>FD.zip`` holds a ``<YEAR>FD.xml`` index with one row per filing
    (filer name, filing type, state/district, year, filing date, DocID). Use it
    to audit coverage / freshness of the pre-parsed transaction data — the
    transaction *details* themselves are in per-DocID PDFs (not parsed here).

    Parameters
    ----------
    years : a year or list of years (e.g. ``2024`` or ``[2023, 2024, 2025]``).
    filing_types : keep only these ``FilingType`` codes (``"P"`` = PTR). ``None``
        keeps all (annual reports, new-filer, etc.).
    refresh : re-download instead of using the parquet cache.

    Returns
    -------
    DataFrame: member, filing_type, state_district, year, filing_date, doc_id, doc_url.
    """
    if isinstance(years, int):
        years = [years]

    def _build() -> pd.DataFrame:
        frames = []
        for yr in years:
            payload = _fetch_bytes(HOUSE_INDEX_URL.format(year=yr))
            zf = zipfile.ZipFile(io.BytesIO(payload))
            xml_name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
            root = ET.fromstring(zf.read(xml_name))
            recs = []
            for m in root:
                g = {c.tag: (c.text or "").strip() for c in m}
                first = " ".join(x for x in [g.get("Prefix", ""), g.get("First", "")] if x)
                last = " ".join(x for x in [g.get("Last", ""), g.get("Suffix", "")] if x)
                doc_id = g.get("DocID", "")
                recs.append(
                    {
                        "member": f"{first} {last}".strip(),
                        "filing_type": g.get("FilingType", ""),
                        "state_district": g.get("StateDst", ""),
                        "year": int(g.get("Year") or yr),
                        "filing_date": g.get("FilingDate", ""),
                        "doc_id": doc_id,
                        "doc_url": HOUSE_PTR_PDF_URL.format(year=yr, doc_id=doc_id),
                    }
                )
            frames.append(pd.DataFrame(recs))
        out = pd.concat(frames, ignore_index=True)
        out["filing_date"] = _to_naive_datetime(out["filing_date"])
        return out

    key = f"house_index_{min(years)}_{max(years)}"
    df = cached_parquet(key, _build, refresh=refresh)
    if filing_types is not None:
        df = df[df["filing_type"].isin(filing_types)]
    return df.reset_index(drop=True)


def audit_coverage(trades: pd.DataFrame, house_index: pd.DataFrame) -> dict:
    """Cross-check the pre-parsed House trades against the official PTR index.

    Matches on the House Clerk DocID embedded in each ``doc_url`` (e.g.
    ``.../ptr-pdfs/2024/20025031.pdf``). Returns a small summary dict for the
    report: how many official PTR documents are represented in the parsed data,
    and the latest filing date each source has seen.
    """
    def _doc_ids(urls: pd.Series) -> set[str]:
        return set(
            urls.dropna()
            .astype(str)
            .str.extract(r"/ptr-pdfs/\d+/(\d+)\.pdf", expand=False)
            .dropna()
        )

    house_trades = trades[trades["chamber"] == "house"]
    parsed_docs = _doc_ids(house_trades["doc_url"])
    official_docs = set(house_index["doc_id"].dropna().astype(str))
    matched = parsed_docs & official_docs
    return {
        "official_ptr_docs": len(official_docs),
        "parsed_house_docs": len(parsed_docs),
        "matched_docs": len(matched),
        "coverage_pct": (len(matched) / len(official_docs) * 100.0) if official_docs else float("nan"),
        "latest_official_filing": (
            house_index["filing_date"].max().date().isoformat()
            if not house_index.empty else None
        ),
        "latest_parsed_filing": (
            house_trades["filing_date"].max().date().isoformat()
            if not house_trades.empty else None
        ),
    }
