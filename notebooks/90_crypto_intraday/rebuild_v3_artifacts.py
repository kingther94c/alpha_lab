"""Rebuild the in-sample report artifacts straight from the source of truth (crypto_book.py).

Produces exactly the files render_multi_strategy_report.py consumes — sleeve_excess_returns.parquet,
combos.parquet, corr.parquet, meta.json — for the in-sample window (2022-04 burn .. 2025-12-31,
holdout NOT released). After the S5 vol-target fix this keeps the HTML report consistent with the
committed strategy. (The 2026 holdout parquets are written separately by eval_holdout_2026.py.)

Run: D:/conda/envs/py313/python.exe notebooks/90_crypto_intraday/rebuild_v3_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from alpha_lab.backtest.crypto_book import BURN, backtest_book, combine, load_book_data

OUT = ROOT / "data" / "results" / "crypto_v3_multi"
OUT.mkdir(parents=True, exist_ok=True)
EOY25 = "2025-12-31"


def log(*a):
    print(*a, flush=True)


log("=== rebuilding in-sample artifacts (2022-01 -> 2025-12, holdout locked) ===")
bd = load_book_data("2022-01-01", EOY25, allow_holdout=False)
log("grid", bd.grid.min().date(), "->", bd.grid.max().date(), "| rf:", bd.rf_source)

R, results, diag = backtest_book(bd)
combos, lev = combine(R)
btc_bh = (bd.spot_close["BTC.s"].pct_change() - bd.rf_daily).rename("btc_bh_excess")

# slice to the burn-in start (matches the committed report window)
R = R.loc[BURN:]
combos = combos.loc[BURN:].join(btc_bh.loc[BURN:])
corr = R.corr()

eye = np.eye(R.shape[1], dtype=bool)
off = corr.where(~eye)
meta = {
    "diag": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                 for kk, vv in d.items()} for k, d in diag.items()},
    "mean_offdiag_corr": float(off.abs().mean().mean()),
    "max_offdiag_corr": float(off.abs().max().max()),
    "sum_pairwise_corr": float(off.sum().sum() / 2),
    "diversification_ratio": float(R.std().mean() / R.mean(axis=1).std()),
    "mean_leverage": {k: float(lev[k].loc[BURN:].mean()) for k in R.columns},
    "rf_source": bd.rf_source,
}

R.to_parquet(OUT / "sleeve_excess_returns.parquet")
combos.to_parquet(OUT / "combos.parquet")
corr.to_parquet(OUT / "corr.parquet")
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))

log(f"mean|corr|={meta['mean_offdiag_corr']:.3f}  max={meta['max_offdiag_corr']:.3f}  "
    f"sum_pairwise={meta['sum_pairwise_corr']:.3f}  DR={meta['diversification_ratio']:.2f}x")
log("carry mean leverage:", round(meta["mean_leverage"]["S1_carry"], 1), "x")
log("wrote artifacts ->", OUT, "\nDONE")
