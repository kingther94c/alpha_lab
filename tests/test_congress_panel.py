"""Unit tests for the congressional idea-flag panel scoring (offline, synthetic).

Covers the subtle bits: buy-only, size tiers, ETF-translatable gating, the
distinct-member cluster count, the staleness discount, and tier cutoffs. No
network / no real cache — a tiny hand-built frame with the canonical columns.
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.analytics.congress_panel import (
    SCORE_MAJOR,
    SCORE_MATERIAL,
    material_events,
    score_materiality,
)

SECTOR_OF = pd.Series({"NVDA": "Technology", "MSFT": "Technology", "XOM": "Energy",
                       "WEIRD": "Unknown"})


def _trade(member, ticker, sign, low, high, fdate, tdate=None, party="D"):
    return {
        "member": member, "ticker": ticker, "sign": sign, "party": party,
        "amount_low": low, "amount_high": high,
        "filing_date": pd.Timestamp(fdate),
        "transaction_date": pd.Timestamp(tdate or fdate),
    }


def test_sells_and_unmapped_score_zero():
    df = pd.DataFrame([
        _trade("A", "NVDA", -1, 250001, 500000, "2026-01-10"),   # sell → 0
        _trade("B", "WEIRD", 1, 5000001, 25000000, "2026-01-10"),  # unmapped → 0
    ])
    sc = score_materiality(df, SECTOR_OF)
    # the buy row exists but scores 0 (not translatable); the sell isn't a buy row
    assert (sc["score"] == 0).all()


def test_size_tiers_monotonic():
    df = pd.DataFrame([
        _trade("A", "NVDA", 1, 1001, 15000, "2026-01-10"),     # small
        _trade("B", "MSFT", 1, 100001, 250000, "2026-02-10"),  # notable
        _trade("C", "XOM", 1, 250001, 500000, "2026-03-10"),   # major-size
    ])
    sc = score_materiality(df, SECTOR_OF).set_index("member")["score"]
    assert sc["A"] < sc["B"] < sc["C"]


def test_cluster_raises_score_and_tier():
    # five distinct members all buy Tech within 14 days → sector cluster builds
    rows = [_trade(m, "NVDA", 1, 50001, 100000, f"2026-01-{10+i:02d}")
            for i, m in enumerate(["A", "B", "C", "D", "E", "F"])]
    sc = score_materiality(pd.DataFrame(rows), SECTOR_OF)
    last = sc.sort_values("filing_date").iloc[-1]   # sees the full cluster
    first = sc.sort_values("filing_date").iloc[0]
    assert last["sec_cluster"] > first["sec_cluster"]
    assert last["score"] > first["score"]


def test_staleness_discount():
    fresh = pd.DataFrame([_trade("A", "NVDA", 1, 250001, 500000, "2026-01-10", "2026-01-05")])
    stale = pd.DataFrame([_trade("A", "NVDA", 1, 250001, 500000, "2026-01-10", "2025-09-01")])
    s_fresh = score_materiality(fresh, SECTOR_OF)["score"].iloc[0]
    s_stale = score_materiality(stale, SECTOR_OF)["score"].iloc[0]
    assert s_stale < s_fresh


def test_material_events_filters_and_tiers():
    rows = [_trade(m, "NVDA", 1, 250001, 500000, f"2026-01-{20+i:02d}")
            for i, m in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"])]
    sc = score_materiality(pd.DataFrame(rows), SECTOR_OF)
    ev = material_events(sc, asof="2026-01-28", fresh_trading_days=10)
    assert (ev["score"] >= SCORE_MATERIAL).all()
    assert "etf" in ev.columns and (ev["etf"] == "XLK").all()
    assert ev["tier"].isin(["material", "major"]).all()
    # a high-cluster major-size cluster should reach the major tier
    assert (sc["score"] >= SCORE_MAJOR).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
