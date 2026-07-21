"""Daily *idea-flag* panel from congressional disclosures (not a strategy).

The standalone L/S sector book was **rejected** (see
``docs/research_decisions/2026-06-19_congressional_trading_signal.md``: net Sharpe
0.12 vs SPY 0.82, Deflated-Sharpe 0.15 — a clean, audited null). Its documented
next step was: *"wire the sector-flow z-score into the daily idea pipeline as a
low-correlation context flag, not a standalone book."* This module is that flag.

It turns the normalized PTR frame into two things the daily macro brief consumes:

1. :func:`sector_backdrop` — a standing, low-frequency context line: which GICS
   sectors Congress is *accumulating* (buy-side z-score), expressed in sector ETFs.
2. :func:`material_events` — the rare, genuinely notable single filings/clusters
   worth surfacing as an idea card, selected by :func:`score_materiality`.

Design rules grounded in the audited study
-------------------------------------------
* **Buy-side only.** The event study found post-filing drift only on *buys*
  (+0.37%/42d, t=2.4); *sells* carry no signal (t≈0). Sells never create a
  material event and the backdrop uses buy accumulation, not net flow.
* **Point-in-time.** Everything keys off ``filing_date`` (the public date), never
  ``transaction_date`` — the 45-day lag is the whole reason fast copy has no edge.
* **ETF-translatable only.** A trade must map to a GICS sector (hence a sector
  ETF) to be material; single names are never surfaced (the project's mandate).
* **Committee (Angle B) is the reserved upgrade.** The highest-value untested edge
  needs point-in-time committee rosters; ``committee_of`` is plumbed but defaults
  off (contributes 0) until those rosters are wired.

Thresholds — calibrated on the real 2018-06-19→2026 kadoa history
-----------------------------------------------------------------
Percentiles below are empirical, from that sample (15,186 single-stock buys):

* Size: 78% of buys are \\$1k–15k; ``>=\\$250k`` is the top ~1% (**major**),
  ``>=\\$100k`` the top ~3% (**notable**), ``>=\\$50k`` the top ~7%.
* Sector cluster (distinct members buying one GICS sector in a trailing 14 days):
  p75=4, p95=6, p99=8.
* Same-name cluster (distinct members buying one ticker in 14d): p99=3.
* Member activity: top-decile lifetime single-stock trades = 322.
* Composite ``>=50`` fires on 4.8% of trading days (~25/yr ≈ 2/mo → "most days
  zero cards"); ``>=65`` on ~0.8% (~6/yr) = a headline event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.backtest.congress_signal import sector_flow_zscore, sector_net_flow
from alpha_lab.data.congress_universe import sector_etf_map

# --- calibrated constants (real 2018–2026 kadoa data; see module docstring) -----------
SIZE_MAJOR = 250_001      # top ~1% of buys
SIZE_NOTABLE = 100_001    # top ~3%
SIZE_MINOR = 50_001       # top ~7%
SECCLUSTER_STRONG = 8     # p99 distinct members / sector / 14d
SECCLUSTER_NOTABLE = 6    # p95
SECCLUSTER_MILD = 4       # p75
TKRCLUSTER_STRONG = 3     # p99 distinct members / ticker / 14d
TKRCLUSTER_PAIR = 2
ACTIVE_MEMBER_TRADES = 322  # top-decile lifetime single-stock trades
STALE_DAYS = 45           # days-to-file beyond the statutory deadline
STALE_DISCOUNT = 0.7      # discount a late (stale) filing's score
CLUSTER_WINDOW_DAYS = 14  # ≈ 10 trading days
SCORE_MATERIAL = 50.0     # card-worthy (≈25/yr)
SCORE_MAJOR = 65.0        # headline (≈6/yr)
Z_NOTABLE = 2.0           # sector buy-flow z (p90)
Z_STRONG = 3.0            # ~p95
FRESH_TRADING_DAYS = 10   # a filing is "new" if public within ~2 weeks

_TIER = {0: "—", 1: "material", 2: "major"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def active_members(trades: pd.DataFrame, *, min_trades: int = ACTIVE_MEMBER_TRADES) -> set[str]:
    """Members in the top activity decile (>= ``min_trades`` lifetime single-stock trades).

    A weak proxy for "habitually-trading / well-connected" — kept as a minor
    tie-breaker (+5), never a primary driver (the study did not validate any
    specific member's edge).
    """
    counts = trades.groupby("member").size()
    return set(counts[counts >= min_trades].index)


def _distinct_member_cluster(buys: pd.DataFrame, key: str, *, window_days: int) -> pd.Series:
    """For each buy row, # of *distinct members* who bought the same ``key``
    (``"sector"`` or ``"ticker"``) within the trailing ``window_days`` calendar
    days, inclusive. Leak-safe: only looks backward from each row's filing day.
    """
    d = buys["filing_date"].dt.normalize()
    out = pd.Series(0, index=buys.index, dtype=int)
    for _, idx in buys.groupby(key).groups.items():
        g = buys.loc[idx]
        days = d.loc[idx].values
        mem = g["member"].values
        win = np.timedelta64(window_days, "D")
        out.loc[idx] = [
            len(set(mem[(days <= days[i]) & (days > days[i] - win)])) for i in range(len(g))
        ]
    return out


# --------------------------------------------------------------------------------------
# Materiality scoring (per buy filing)
# --------------------------------------------------------------------------------------
def score_materiality(
    trades: pd.DataFrame,
    sector_of: pd.Series,
    *,
    actives: set[str] | None = None,
    committee_of=None,
    committee_bonus: float = 25.0,
) -> pd.DataFrame:
    """Score every BUY filing 0–100 on how *material* it is for the daily brief.

    Components (all calibrated on real data — see module docstring):
    ``size`` (0/12/25/40) + ``sector_cluster`` (0/10/20/30) +
    ``ticker_cluster`` (0/10/25) + ``active_member`` (0/5) + ``committee``
    (0 or ``committee_bonus``), times a 0.7 staleness factor if the filing is
    late (days-to-file > 45). Non-buys and trades with no sector map score 0.

    Parameters
    ----------
    trades : normalized PTR long frame (``filing_date, transaction_date, member,
        sign, ticker, amount_low, party`` …).
    sector_of : Series ticker → GICS sector ("Unknown" ⇒ not translatable ⇒ 0).
    actives : set of active-member names (see :func:`active_members`); ``None``
        derives it from ``trades``.
    committee_of : optional callable ``(member, sector) -> bool`` — True when the
        member sits on a committee with jurisdiction over the sector (Angle B).
        Defaults to off (the PIT rosters are not yet wired).

    Returns
    -------
    The buy rows with added columns: ``sector, sec_cluster, tkr_cluster,
    days_to_file, score, tier`` (tier ∈ {"—","material","major"}).
    """
    buys = trades[trades["sign"] == 1].copy()
    if buys.empty:
        return buys.assign(sector=[], sec_cluster=[], tkr_cluster=[],
                           days_to_file=[], score=[], tier=[])
    buys["sector"] = buys["ticker"].map(sector_of)
    actives = active_members(trades) if actives is None else actives

    sec_buys = buys[buys["sector"].notna() & (buys["sector"] != "Unknown")]
    buys["sec_cluster"] = _distinct_member_cluster(
        sec_buys, "sector", window_days=CLUSTER_WINDOW_DAYS
    ).reindex(buys.index).fillna(0).astype(int)
    buys["tkr_cluster"] = _distinct_member_cluster(
        buys, "ticker", window_days=CLUSTER_WINDOW_DAYS
    ).reindex(buys.index).fillna(0).astype(int)
    buys["days_to_file"] = (buys["filing_date"] - buys["transaction_date"]).dt.days

    def _row_score(r) -> float:
        translatable = pd.notna(r["sector"]) and r["sector"] != "Unknown"
        if not translatable:
            return 0.0
        lo = r["amount_low"] if pd.notna(r["amount_low"]) else 1001
        size = 40 if lo >= SIZE_MAJOR else 25 if lo >= SIZE_NOTABLE else 12 if lo >= SIZE_MINOR else 0
        sc = r["sec_cluster"]
        secc = 30 if sc >= SECCLUSTER_STRONG else 20 if sc >= SECCLUSTER_NOTABLE else 10 if sc >= SECCLUSTER_MILD else 0
        tc = r["tkr_cluster"]
        tkrc = 25 if tc >= TKRCLUSTER_STRONG else 10 if tc >= TKRCLUSTER_PAIR else 0
        act = 5 if r["member"] in actives else 0
        comm = committee_bonus if (committee_of is not None and committee_of(r["member"], r["sector"])) else 0
        raw = size + secc + tkrc + act + comm
        if pd.notna(r["days_to_file"]) and r["days_to_file"] > STALE_DAYS:
            raw *= STALE_DISCOUNT
        return float(raw)

    buys["score"] = buys.apply(_row_score, axis=1)
    buys["tier"] = buys["score"].apply(
        lambda s: _TIER[2] if s >= SCORE_MAJOR else _TIER[1] if s >= SCORE_MATERIAL else _TIER[0]
    )
    return buys


def material_events(
    scored: pd.DataFrame,
    *,
    asof: pd.Timestamp | str | None = None,
    fresh_trading_days: int = FRESH_TRADING_DAYS,
    min_score: float = SCORE_MATERIAL,
    aggregate: bool = True,
) -> pd.DataFrame:
    """The material buy filings that became public in the recent window.

    Filters :func:`score_materiality` output to ``score >= min_score`` and
    ``filing_date`` within roughly ``fresh_trading_days`` of ``asof`` (calendar
    proxy = trading days × 1.45). Adds the sector ``etf`` column.

    With ``aggregate=True`` (default) collapses raw transaction lines to one row
    per **(filing_date, member, sector)** — a member's several same-sector buys on
    one filing are one *event*, not many lines. Aggregated columns: ``score`` /
    ``sec_cluster`` = max, ``n_trades`` = count, ``top_amount`` = largest single
    ticket, ``names`` = up to three source tickers (provenance only — the
    expression is always the ETF). Sorted by score, descending.
    """
    if scored.empty:
        return scored
    asof = pd.Timestamp(asof) if asof is not None else scored["filing_date"].max()
    lo = asof - pd.Timedelta(days=int(round(fresh_trading_days * 1.45)))
    etf = sector_etf_map()
    ev = scored[(scored["score"] >= min_score) & (scored["filing_date"] > lo) & (scored["filing_date"] <= asof)].copy()
    ev["etf"] = ev["sector"].map(etf)
    if not aggregate:
        return ev.sort_values("score", ascending=False).reset_index(drop=True)
    if ev.empty:
        return ev.assign(n_trades=[], top_amount=[], names=[])
    g = ev.groupby(["filing_date", "member", "party", "sector", "etf"], as_index=False).agg(
        score=("score", "max"),
        tier=("tier", lambda s: "major" if (s == "major").any() else "material"),
        sec_cluster=("sec_cluster", "max"),
        n_trades=("ticker", "size"),
        top_amount=("amount_low", "max"),
        names=("ticker", lambda s: ", ".join(pd.unique(s.dropna().astype(str))[:3])),
    )
    return g.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Sector backdrop (standing context line)
# --------------------------------------------------------------------------------------
def sector_backdrop(
    trades: pd.DataFrame,
    sector_of: pd.Series,
    trading_index: pd.DatetimeIndex,
    *,
    window: int = 63,
    z_window: int = 252,
    asof: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Latest buy-accumulation z-score per GICS sector, with ETF + flag.

    Uses **buy-side** flow (the side the study found carries signal), the existing
    :func:`sector_net_flow`/:func:`sector_flow_zscore` machinery, and the calibrated
    z thresholds. Returns one row per sector sorted by z (descending): ``sector,
    etf, zscore, flag`` where flag ∈ {"strong+","notable+","—","notable-","strong-"}.
    """
    buys = trades[trades["sign"] == 1]
    nf = sector_net_flow(buys, sector_of, trading_index, window=window)
    z = sector_flow_zscore(nf, z_window=z_window, min_periods=63)
    asof = pd.Timestamp(asof) if asof is not None else z.index[-1]
    row = z.loc[z.index[z.index <= asof][-1]]
    etf = sector_etf_map()

    def _flag(v: float) -> str:
        if pd.isna(v):
            return "n/a"
        if v >= Z_STRONG:
            return "strong+"
        if v >= Z_NOTABLE:
            return "notable+"
        if v <= -Z_STRONG:
            return "strong-"
        if v <= -Z_NOTABLE:
            return "notable-"
        return "—"

    out = pd.DataFrame({
        "sector": row.index,
        "etf": [etf.get(s, "?") for s in row.index],
        "zscore": row.values,
        "flag": [_flag(v) for v in row.values],
    })
    return out.sort_values("zscore", ascending=False, na_position="last").reset_index(drop=True)


def party_tilt(
    trades: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    window: int = 63,
    z_window: int = 252,
    asof: pd.Timestamp | str | None = None,
) -> dict:
    """Aggregate buy-flow risk-appetite proxy (Angle C), as a small dict.

    Returns ``{"zscore": float, "lean": "risk-on"|"risk-off"|"neutral"}`` from the
    z-score of total congressional buy accumulation — a coarse growth-vs-small
    macro lean (long QQQ / short IWM when strongly net-accumulating).
    """
    from alpha_lab.backtest.congress_signal import aggregate_net_flow

    buys = trades[trades["sign"] == 1]
    agg = aggregate_net_flow(buys, trading_index, window=window)
    mean = agg.rolling(z_window, min_periods=63).mean()
    std = agg.rolling(z_window, min_periods=63).std()
    z = (agg - mean) / std.replace(0.0, np.nan)
    asof = pd.Timestamp(asof) if asof is not None else z.index[-1]
    val = float(z.loc[z.index[z.index <= asof][-1]])
    lean = "risk-on" if val >= Z_NOTABLE else "risk-off" if val <= -Z_NOTABLE else "neutral"
    return {"zscore": val, "lean": lean}
