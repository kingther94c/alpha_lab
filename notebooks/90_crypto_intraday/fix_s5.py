"""Phase 0 — fix the S5 macro sleeve. SELECTION IS IN-SAMPLE ONLY (BURN..2025-12-31).

S5 (long-only BTC/ETH spot gated by HYG>50dMA) lost -29.5% in the 2026 holdout: it held
undefended long crypto beta while the credit gate (HYG) stayed risk-on through a crypto-led
drop. We test FRED-free variants and pick the winner purely on in-sample robustness, then
report 2026 once for transparency (2026 is NOT used to choose).

Run: D:/conda/envs/py313/python.exe notebooks/90_crypto_intraday/fix_s5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from alpha_lab.backtest import crypto_book as cb
from alpha_lab.backtest.metrics import summary
from alpha_lab.data.loaders.yfinance import load_prices

BARS, BURN = cb.BARS, cb.BURN
EOY25 = pd.Timestamp("2025-12-31", tz="UTC")
JAN26 = pd.Timestamp("2026-01-01", tz="UTC")


def log(*a):
    print(*a, flush=True)


def _align(series: pd.Series, grid) -> pd.Series:
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
    return s.reindex(grid).ffill()


# ---------------------------------------------------------------- load data once
log("=== loading 2022 -> 2026-07 (PM HOLDOUT released, audited) ===")
bd = cb.load_book_data("2022-01-01", "2026-07-01", allow_holdout=True)
grid = bd.grid
log("grid", grid.min().date(), "->", grid.max().date(), "| rf:", bd.rf_source)

# extra FRED-free macro inputs (credit-spread proxy from yfinance credit ETFs)
lqd = _align(load_prices("LQD", "2022-01-01", "2026-07-01")["LQD"], grid)   # investment-grade
hyg = bd.hyg                                                                # high-yield (already on grid)
credit_rs = (hyg / lqd)        # HY/IG relative strength: falls = credit risk-off (spread widening)

# optional FRED BAA10Y (likely times out here; variant skipped if unavailable)
baa = None
try:
    from alpha_lab.data.loaders.fred import load_series
    raw = load_series("BAA10Y", start="2022-01-01", end="2026-07-01", timeout=20)["BAA10Y"].dropna()
    baa = _align(raw, grid)
    log("FRED BAA10Y: OK")
except Exception as e:  # noqa: BLE001
    log(f"FRED BAA10Y: unavailable ({type(e).__name__}); skipping that variant")


# ---------------------------------------------------------------- S5 variants
def _long(ro, scale_btc=0.5, scale_eth=0.5):
    return pd.DataFrame({"BTC.s": scale_btc * ro, "ETH.s": scale_eth * ro}).reindex(grid)


def v0_baseline(bd):
    ro = (hyg.shift(1) > hyg.shift(1).rolling(50).mean()).astype(float)
    return _long(ro), ["BTC.s", "ETH.s"], False


def v1_voltgt(bd, target=0.25, lev_cap=2.0):
    """Same HYG gate, but risk-normalise the position to ~target vol (S5 ran ~46-51% vol)."""
    ro = (hyg.shift(1) > hyg.shift(1).rolling(50).mean()).astype(float)
    proxy = (0.5 * bd.spot_close[["BTC.s", "ETH.s"]].pct_change()).sum(axis=1)
    vol = proxy.rolling(30).std().shift(1) * np.sqrt(BARS)
    scale = (target / vol).clip(upper=lev_cap).fillna(0.0)
    base = _long(ro)
    return base.mul(scale, axis=0), ["BTC.s", "ETH.s"], False


def v3_creditspread(bd):
    """Cleaner credit signal: HY/IG relative strength (HYG/LQD) above its 50d MA = risk-on."""
    ro = (credit_rs.shift(1) > credit_rs.shift(1).rolling(50).mean()).astype(float)
    return _long(ro), ["BTC.s", "ETH.s"], False


def v3b_baa(bd):
    """FRED BAA10Y credit spread: below trailing MA (tight/calm) = risk-on."""
    ro = (baa.shift(1) < baa.shift(1).rolling(50).mean()).astype(float)
    return _long(ro), ["BTC.s", "ETH.s"], False


def v4_trendstop(bd):
    """HYG gate AND BTC above its own 100d MA — a slow own-asset stop on undefended beta."""
    ro = (hyg.shift(1) > hyg.shift(1).rolling(50).mean()).astype(float)
    up = (bd.spot_close["BTC.s"] > bd.spot_close["BTC.s"].rolling(100).mean()).astype(float)
    return _long(ro * up), ["BTC.s", "ETH.s"], False


def v5_voltgt_creditspread(bd, target=0.25, lev_cap=2.0):
    """V1 + V3: cleaner credit signal AND vol-targeted."""
    ro = (credit_rs.shift(1) > credit_rs.shift(1).rolling(50).mean()).astype(float)
    proxy = (0.5 * bd.spot_close[["BTC.s", "ETH.s"]].pct_change()).sum(axis=1)
    vol = proxy.rolling(30).std().shift(1) * np.sqrt(BARS)
    scale = (target / vol).clip(upper=lev_cap).fillna(0.0)
    return _long(ro).mul(scale, axis=0), ["BTC.s", "ETH.s"], False


def v6_neutral_pair(bd):
    """Market-neutral macro rotation: risk-on -> long ETH.p / short BTC.p (high-beta tilt);
    risk-off -> reverse. Removes outright crypto beta, keeps the macro conditioner."""
    ro = hyg.shift(1) > hyg.shift(1).rolling(50).mean()
    s = pd.Series(np.where(ro, 1.0, -1.0), index=grid)
    w = pd.DataFrame({"ETH.p": 0.5 * s, "BTC.p": -0.5 * s}).reindex(grid)
    return w, ["BTC.p", "ETH.p"], True


VARIANTS = {"V0_baseline": v0_baseline, "V1_voltgt": v1_voltgt, "V3_creditspread": v3_creditspread,
            "V4_trendstop": v4_trendstop, "V5_voltgt+credit": v5_voltgt_creditspread,
            "V6_neutral_pair": v6_neutral_pair}
if baa is not None:
    VARIANTS["V3b_baa"] = v3b_baa


# ---------------------------------------------------------------- book eval (swap S5)
def book_with_s5(builder):
    sw = cb.sleeve_weights(bd)
    sw["S5_macro"] = builder(bd)
    R = {name: cb._excess(bd, cb._bt(bd, w, cols, uf)) for name, (w, cols, uf) in sw.items()}
    R = pd.DataFrame(R)
    combos, _ = cb.combine(R)
    return R, combos


def stats(s, lo, hi):
    s = s.loc[lo:hi].dropna()
    if len(s) < 5:
        return dict(sharpe=np.nan, ret=np.nan, mdd=np.nan, calmar=np.nan)
    sm = summary(s, periods=BARS)
    return dict(sharpe=sm["Sharpe"], ret=(1 + s).prod() - 1, mdd=sm["MaxDD"], calmar=sm["Calmar"])


def meancorr(R, lo, hi):
    c = R.loc[lo:hi].corr().where(~np.eye(R.shape[1], dtype=bool)).abs()
    return float(c.mean().mean())


# ---------------------------------------------------------------- run all variants
rows = []
store = {}
for name, builder in VARIANTS.items():
    R, combos = book_with_s5(builder)
    store[name] = (R, combos)
    s5_is = stats(R["S5_macro"], BURN, EOY25)
    cb_is = stats(combos["combo_eqcap"], BURN, EOY25)
    rows.append(dict(variant=name, mc_is=meancorr(R, BURN, EOY25),
                     s5_sharpe=s5_is["sharpe"], s5_mdd=s5_is["mdd"],
                     cb_sharpe=cb_is["sharpe"], cb_ret=cb_is["ret"], cb_mdd=cb_is["mdd"], cb_calmar=cb_is["calmar"]))
tab = pd.DataFrame(rows).set_index("variant")

log("\n================ IN-SAMPLE (BURN..2025-12-31) — SELECTION BASIS ================")
log(tab.round(3).to_string())

# ---- explicit, in-sample-only selection rule ---------------------------------
base = tab.loc["V0_baseline"]
elig = tab[(tab["cb_sharpe"] >= base["cb_sharpe"] - 0.05) &      # don't hurt the book's Sharpe
           (tab["mc_is"] <= base["mc_is"] + 0.02) &              # keep diversification
           (tab["s5_mdd"] >= base["s5_mdd"])].copy()             # improve (or match) S5 drawdown
elig = elig.drop(index=["V0_baseline"], errors="ignore")
log("\neligible (improve S5 MaxDD, keep book Sharpe & low corr):", list(elig.index) or "none")
# among eligible, maximise the book's in-sample Calmar (return per unit drawdown = robustness)
winner = elig["cb_calmar"].idxmax() if len(elig) else "V0_baseline"
log(f"WINNER (max in-sample combo Calmar among eligible) = {winner}")

# ---- robustness of the vol-target family (IN-SAMPLE only) --------------------
log("\n================ V1 vol-target robustness (IN-SAMPLE BURN..2025) ================")
log(f"{'target':>7s}{'cap':>5s}{'volwin':>7s} | {'cb_shrp':>8s}{'cb_calm':>8s}{'cb_mdd':>8s}{'s5_mdd':>8s}{'mc':>6s}")


def v1_param(bd, target, cap, volwin):
    ro = (hyg.shift(1) > hyg.shift(1).rolling(50).mean()).astype(float)
    proxy = (0.5 * bd.spot_close[["BTC.s", "ETH.s"]].pct_change()).sum(axis=1)
    vol = proxy.rolling(volwin).std().shift(1) * np.sqrt(BARS)
    scale = (target / vol).clip(upper=cap).fillna(0.0)
    return _long(ro).mul(scale, axis=0), ["BTC.s", "ETH.s"], False


for target in (0.20, 0.25, 0.30):
    for cap in (2.0, 3.0):
        for volwin in (20, 30, 45):
            R, combos = book_with_s5(lambda b, t=target, c=cap, v=volwin: v1_param(b, t, c, v))
            ci = stats(combos["combo_eqcap"], BURN, EOY25)
            s5 = stats(R["S5_macro"], BURN, EOY25)
            mc = meancorr(R, BURN, EOY25)
            log(f"{target:>7.2f}{cap:>5.1f}{volwin:>7d} | {ci['sharpe']:>8.3f}{ci['calmar']:>8.3f}"
                f"{ci['mdd']:>+8.2%}{s5['mdd']:>+8.2%}{mc:>6.3f}")

# ---------------------------------------------------------------- 2026 holdout (report only)
log("\n================ 2026 HOLDOUT (report only — NOT used for selection) ================")
log(f"{'variant':18s} {'S5 ret':>9s} {'S5 Sharpe':>10s} {'S5 MaxDD':>9s} | "
    f"{'book ret':>9s} {'book Shrp':>10s} {'book MaxDD':>10s}")
for name in VARIANTS:
    R, combos = store[name]
    s5 = stats(R["S5_macro"], JAN26, None)
    bk = stats(combos["combo_eqcap"], JAN26, None)
    star = "  <-- WINNER" if name == winner else ("  (baseline)" if name == "V0_baseline" else "")
    log(f"{name:18s} {s5['ret']:>+8.2%} {s5['sharpe']:>10.2f} {s5['mdd']:>+8.2%} | "
        f"{bk['ret']:>+8.2%} {bk['sharpe']:>10.2f} {bk['mdd']:>+9.2%}{star}")

btc26 = stats((bd.spot_close["BTC.s"].pct_change() - bd.rf_daily), JAN26, None)
log(f"{'BTC buy&hold':18s} {'':>9s} {'':>10s} {'':>9s} | {btc26['ret']:>+8.2%} {btc26['sharpe']:>10.2f} {btc26['mdd']:>+9.2%}")
log("\nDONE")
