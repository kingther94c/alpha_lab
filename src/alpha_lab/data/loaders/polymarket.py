"""Polymarket prediction-market helpers via the public Gamma API.

No API key required. Two entry points most notebooks need:

    from alpha_lab.data.loaders.polymarket import search_markets, top_by_liquidity

``search_markets`` returns a tidy DataFrame of markets matching a query/tag.
``top_by_liquidity`` (alias ``pick_largest``) picks the single market with the
most liquidity / volume when several near-duplicates describe the same event.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import httpx
import pandas as pd

_GAMMA = "https://gamma-api.polymarket.com"

_MARKET_COLS = [
    "id",
    "slug",
    "question",
    "conditionId",
    "endDate",
    "startDate",
    "active",
    "closed",
    "volume",
    "volumeNum",
    "liquidity",
    "liquidityNum",
    "outcomes",
    "outcomePrices",
    "lastTradePrice",
    "bestBid",
    "bestAsk",
    "spread",
    "category",
    "eventSlug",
]


def _get(path: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
    resp = httpx.get(f"{_GAMMA}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _coerce_jsonish(v: Any) -> Any:
    """Polymarket sometimes returns list-valued fields as JSON strings."""
    if isinstance(v, str) and v.startswith(("[", "{")):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _markets_to_frame(raw: Iterable[dict]) -> pd.DataFrame:
    rows = list(raw)
    if not rows:
        return pd.DataFrame(columns=_MARKET_COLS)
    df = pd.DataFrame(rows)
    for col in ("outcomes", "outcomePrices"):
        if col in df.columns:
            df[col] = df[col].apply(_coerce_jsonish)
    for col in ("volumeNum", "liquidityNum", "lastTradePrice", "bestBid", "bestAsk", "spread"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("endDate", "startDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    keep = [c for c in _MARKET_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


def search_markets(
    query: str | None = None,
    *,
    tag_slug: str | None = None,
    active: bool | None = True,
    closed: bool | None = False,
    limit: int = 100,
    offset: int = 0,
    order: str = "liquidityNum",
    ascending: bool = False,
) -> pd.DataFrame:
    """Search Polymarket markets and return a tidy DataFrame.

    Parameters
    ----------
    query : substring match against the market question (server-side ``q``).
    tag_slug : e.g. ``"fed"``, ``"geopolitics"``, ``"crypto"``.
    active / closed : state filters — defaults show only live markets.
    limit, offset : pagination.
    order, ascending : server-side sort; defaults to largest liquidity first.
    """
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    if query:
        params["q"] = query
    if tag_slug:
        params["tag_slug"] = tag_slug
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()

    raw = _get("/markets", params)
    return _markets_to_frame(raw)


def search_events(
    query: str | None = None,
    *,
    tag_slug: str | None = None,
    active: bool | None = True,
    closed: bool | None = False,
    limit: int = 50,
    offset: int = 0,
    order: str = "liquidity",
    ascending: bool = False,
) -> pd.DataFrame:
    """Search Polymarket *events* (groups of related markets).

    Returns one row per event; the nested ``markets`` column holds the child
    markets. Useful when one real-world question is split across many binary
    markets (e.g. FOMC meetings).
    """
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    if query:
        params["q"] = query
    if tag_slug:
        params["tag_slug"] = tag_slug
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()

    raw = _get("/events", params)
    return pd.DataFrame(raw)


def get_market(slug_or_id: str) -> dict:
    """Fetch a single market by slug (preferred) or numeric id."""
    try:
        int(slug_or_id)
        raw = _get(f"/markets/{slug_or_id}", {})
    except ValueError:
        hits = _get("/markets", {"slug": slug_or_id, "limit": 1})
        if not hits:
            raise KeyError(f"no Polymarket market with slug {slug_or_id!r}")
        raw = hits[0]
    return raw


def top_by_liquidity(
    markets: pd.DataFrame,
    *,
    by: str = "liquidityNum",
    n: int = 1,
) -> pd.DataFrame:
    """Pick the top-*n* markets by liquidity (or volume).

    Use when several near-duplicate markets describe the same real-world event —
    keep only the deepest one so signal-to-noise stays high.
    """
    if markets.empty or by not in markets.columns:
        return markets.head(n)
    return markets.sort_values(by, ascending=False).head(n)


pick_largest = top_by_liquidity


def implied_prob(market: dict | pd.Series, outcome: str = "Yes") -> float | None:
    """Return the implied probability for a given outcome (default ``"Yes"``).

    Works with either a dict from ``get_market`` or a row from ``search_markets``.
    """
    outcomes = _coerce_jsonish(market.get("outcomes"))
    prices = _coerce_jsonish(market.get("outcomePrices"))
    if not outcomes or not prices:
        return None
    try:
        idx = [o.lower() for o in outcomes].index(outcome.lower())
    except ValueError:
        return None
    try:
        return float(prices[idx])
    except (TypeError, ValueError):
        return None


def tidy(markets: pd.DataFrame) -> pd.DataFrame:
    """Compact view: question, endDate, Yes price, liquidity, volume."""
    if markets.empty:
        return markets
    out = markets.copy()
    out["yes"] = out.apply(lambda r: implied_prob(r, "Yes"), axis=1)
    cols = ["question", "endDate", "yes", "liquidityNum", "volumeNum", "slug"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("liquidityNum", ascending=False, na_position="last")
