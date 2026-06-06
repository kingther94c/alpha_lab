"""Backtrader replication of the P7 book ("用backtrader复刻这个策略").

backtrader is the *execution/portfolio engine*; the signal comes from the shared
`alpha_lab.backtest.crypto_book` (single source of truth), so this is the same strategy.
Each daily bar the Strategy reads the precomputed combined target weights and rebalances via
`order_target_percent` — exactly the logic a backtrader→Binance live broker would run.

We then verify backtrader's portfolio P&L tracks the vectorized book's gross return (same
positions => same price P&L; funding/financing are vectorized-engine add-ons, not in bt's
default broker). High correlation == faithful replication.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
import backtrader as bt
from alpha_lab.backtest.crypto_book import load_book_data, sleeve_weights, BURN, BARS
from alpha_lab.backtest.metrics import summary


def log(*a): print(*a, flush=True)

log("=== loading data + computing combined equal-capital targets (via crypto_book) ===")
bd = load_book_data("2022-01-01", "2026-07-01", allow_holdout=True)
sw = sleeve_weights(bd)
legs = sorted({c for (w, cols, _) in sw.values() for c in cols})

# Combined equal-capital per-leg target weights (0.2 per sleeve), the same book the executor trades.
W = pd.DataFrame(0.0, index=bd.grid, columns=legs)
for name, (w, cols, _) in sw.items():
    for c in cols:
        W[c] += 0.2 * w[c].reindex(bd.grid).fillna(0.0)
W = W.fillna(0.0)
idx_naive = bd.grid.tz_localize(None)
W.index = idx_naive

# Per-leg price frames for backtrader (daily close as O=H=L=C).
def price_df(leg):
    src = bd.spot_close if leg.endswith(".s") else bd.perp_close
    s = src[leg].reindex(bd.grid).ffill(); s.index = idx_naive
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 0.0})


class BookStrategy(bt.Strategy):
    """Rebalance to crypto_book's precomputed combined daily target weights."""
    def __init__(self):
        self.legmap = {d._name: d for d in self.datas}
        self.rets = {}

    def next(self):
        ts = pd.Timestamp(self.datas[0].datetime.date(0))
        if ts not in W.index:
            return
        row = W.loc[ts]
        for leg, d in self.legmap.items():
            self.order_target_percent(data=d, target=float(row.get(leg, 0.0)))


log(f"legs: {legs}  | bars: {len(W)}")
cer = bt.Cerebro(stdstats=False)
cer.broker.setcash(1_000_000.0)
cer.broker.set_checksubmit(False)   # allow shorts / leverage for replication
for leg in legs:
    cer.adddata(bt.feeds.PandasData(dataname=price_df(leg), name=leg))
cer.addstrategy(BookStrategy)
cer.addanalyzer(bt.analyzers.TimeReturn, _name="tr", timeframe=bt.TimeFrame.Days)
log("running backtrader Cerebro...")
res = cer.run(runonce=False)
tr = res[0].analyzers.tr.get_analysis()
bt_ret = pd.Series({pd.Timestamp(k): v for k, v in tr.items()}).sort_index()
bt_ret.index = bt_ret.index.tz_localize("UTC")

# Vectorized GROSS baseline: same combined weights. backtrader fills on the NEXT bar, so its
# position from W[t] earns ret[t+2]; the vectorized engine lags weights 1 bar (W[t]->ret[t+1]).
# Test both lags and align on the best — the offset is execution timing, not a replication gap.
rets = bd.prices[legs].pct_change()
Wg = pd.DataFrame(W.values, index=bd.grid, columns=legs)
best = None
for lag in (1, 2):
    g = (Wg.shift(lag) * rets).sum(axis=1)
    cm = bt_ret.index.intersection(g.index); cm = cm[cm >= BURN]
    c = float(np.corrcoef(bt_ret.reindex(cm).fillna(0.0).values, g.reindex(cm).fillna(0.0).values)[0, 1])
    log(f"  align check: vectorized lag={lag} -> corr {c:.4f}")
    if best is None or c > best[0]:
        best = (c, lag, g)
corr, lag, gross_vec = best
common = bt_ret.index.intersection(gross_vec.index); common = common[common >= BURN]
a, b = bt_ret.reindex(common).fillna(0.0), gross_vec.reindex(common).fillna(0.0)
log(f"  best alignment: vectorized lag={lag} (backtrader next-bar fill adds 1 bar)")

log("\n=== backtrader vs vectorized (gross, post-burn) ===")
log(f"  daily-return correlation : {corr:.4f}")
log(f"  backtrader total return  : {((1+a).prod()-1)*100:+.1f}%   Sharpe {summary(a, periods=BARS)['Sharpe']:.2f}")
log(f"  vectorized total return  : {((1+b).prod()-1)*100:+.1f}%   Sharpe {summary(b, periods=BARS)['Sharpe']:.2f}")
log(f"  tracking diff (ann)      : {(a-b).std()*np.sqrt(BARS)*100:.2f}%")
log("\n[verdict] backtrader faithfully replicates the book." if corr > 0.95
    else "\n[verdict] tracking lower than expected — investigate execution model.")
log("\nNote: this is the backtest replication. backtrader-LIVE on Binance uses the SAME")
log("order_target_percent logic against a live ccxt-backed store (ccxtbt); for a daily")
log("rebalance the standalone ccxt executor (binance_paper_trade.py) is the simpler bridge.")
