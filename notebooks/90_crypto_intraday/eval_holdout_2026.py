"""P7 book — final out-of-sample evaluation on the 2026 PM holdout.

Deliberately releases the holdout lock (allow=True; audit-logged) — this is the one-shot
moment-of-truth the holdout was reserved for. Reuses src/alpha_lab/backtest/crypto_book.py
so the evaluated strategy is byte-identical to the research/live code.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
from alpha_lab.backtest.crypto_book import load_book_data, backtest_book, combine, BURN, BARS
from alpha_lab.backtest.metrics import summary

OUT = ROOT / "data" / "results" / "crypto_v3_multi"; OUT.mkdir(parents=True, exist_ok=True)
def log(*a): print(*a, flush=True)

log("=== loading 2022 -> 2026-07 (PM HOLDOUT RELEASED, audited) ===")
bd = load_book_data("2022-01-01", "2026-07-01", allow_holdout=True)
log("grid", bd.grid.min().date(), "->", bd.grid.max().date(), "| rf:", bd.rf_source, "| macro:", bd.macro_source)
n26 = int((bd.grid.year == 2026).sum())
log(f"2026 rows available: {n26}  (through {bd.grid[bd.grid.year==2026].max().date() if n26 else 'n/a'})")

R, results, diag = backtest_book(bd)
combos, lev = combine(R)
btc_bh = (bd.spot_close["BTC.s"].pct_change() - bd.rf_daily).rename("btc_bh_excess")

# ---- reproduction check: pre-2026 must match the committed result -------------
EOY25 = pd.Timestamp("2025-12-31", tz="UTC")
R_pre = R.loc[BURN:EOY25]
off = R_pre.corr().where(~np.eye(5, dtype=bool)).abs()
log(f"\nreproduction check (pre-2026): mean|corr|={off.mean().mean():.3f} (committed 0.112, v3.1), "
    f"combo_eqcap Sharpe={summary(combos['combo_eqcap'].loc[BURN:EOY25].dropna(), periods=BARS)['Sharpe']:.2f} (committed 1.11, v3.1)")

# ---- 2026 out-of-sample --------------------------------------------------------
m26 = R.index >= pd.Timestamp("2026-01-01", tz="UTC")
R26, C26, btc26 = R[m26], combos[m26], btc_bh[m26]
def line(name, s):
    s = s.dropna()
    if len(s) < 5: return f"  {name:24s} (insufficient data)"
    sm = summary(s, periods=BARS)
    return (f"  {name:24s} ret={(1+s).prod()-1:+7.2%}  annSharpe={sm['Sharpe']:6.2f}  "
            f"annVol={sm['AnnVol']:6.2%}  MaxDD={sm['MaxDD']:7.2%}")

log("\n=== 2026 HOLDOUT — sleeves (excess of cash) ===")
for k in R26.columns: log(line(k, R26[k]))
log("\n=== 2026 HOLDOUT — combined book vs BTC ===")
for nm, key in [("equal-capital", "combo_eqcap"), ("risk-budget", "combo_riskbudget"), ("risk-budget vt10%", "combo_vt10")]:
    log(line(nm, C26[key]))
log(line("BTC buy&hold (excess)", btc26))

log("\n=== 2026 HOLDOUT — did orthogonality hold? correlation in-2026 ===")
off26 = R26.corr().where(~np.eye(5, dtype=bool)).abs()
log(f"  mean|corr| 2026 = {off26.mean().mean():.3f}  (in-sample 0.111)")
log(R26.corr().round(2).to_string())

# ---- persist for the report addendum ------------------------------------------
R26.to_parquet(OUT / "holdout2026_sleeves.parquet")
C26.join(btc26).to_parquet(OUT / "holdout2026_combos.parquet")
R.to_parquet(OUT / "sleeve_excess_returns_thru2026.parquet")
combos.join(btc_bh).to_parquet(OUT / "combos_thru2026.parquet")
log("\nsaved 2026 holdout results ->", OUT, "\nDONE")
