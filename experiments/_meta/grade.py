"""Objective grader for the team-eval A/B/C experiment.

Loads each arm's perf.py (all named `perf`, so by file path under unique module names), builds ONE
controlled Store — a demo bot with capital 10k and a 90k faucet offset (raw total starts at 100k),
60 daily marks with a deliberate strategy drawdown — and calls each arm's store-backed wrapper with
its DEFAULTS. The contrast exposes the three repo traps the neutral spec never named:

  * de-faucet     : an arm that reads raw total_equity reports return/drawdown on the 100k faucet
                    base (diluted ~10x, hiding the real strategy drawdown); the honest arm de-faucets
                    to the 10k strategy base.
  * cost-of-cash  : the default Sharpe should be excess-of-rf (0.04), not "beat zero".
  * annualization : daily crypto marks annualize on ~365 (inferred), not a hardcoded 252.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

from quant_bot_manager.core.store import Store

ARMS = ["arm_a", "arm_b", "arm_c"]
ROOT = Path(__file__).resolve().parent.parent / "team_eval"


def load(arm):
    spec = importlib.util.spec_from_file_location(f"perf_{arm}", ROOT / arm / "perf.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass introspection needs the module registered
    spec.loader.exec_module(mod)
    return mod


def maxdd(x) -> float:
    x = np.asarray(x, dtype=float)
    peak = np.maximum.accumulate(x)
    return float((x / peak - 1.0).min())


def build_store(path):
    """capital 10k, faucet offset 90k; 60 daily marks; seeded strategy returns with a real drawdown.
    Appends RAW totals (= strategy equity + 90k offset), exactly as a demo bot's store records."""
    s = Store("grade", path=path)
    s.set_faucet_offset(90_000.0)
    rng = np.random.default_rng(7)
    rets = rng.normal(0.004, 0.02, 59)
    rets[20:30] = -0.03                       # a deliberate ~26% strategy drawdown stretch
    strat = 10_000.0 * np.cumprod(np.r_[1.0, 1.0 + rets])
    base = np.datetime64("2026-01-01")
    for i, sv in enumerate(strat):
        ts = str(base + np.timedelta64(i, "D")) + "T00:00:00+00:00"
        s.append_equity(ts, float(sv + 90_000.0), None, None)
    return s, strat


def grab(summary):
    if summary is None:
        return (float("nan"),) * 4
    def g(*names):
        for n in names:
            if hasattr(summary, n):
                return float(getattr(summary, n))
        return float("nan")
    return (g("ann_return"), g("ann_vol"), g("sharpe"), g("max_drawdown"))


def fmt(t):
    return " ".join(f"{x:>10.3f}" if x == x else f"{'nan':>10}" for x in t)


def main():
    with tempfile.TemporaryDirectory() as d:
        store, strat = build_store(Path(d) / "bot.db")
        raw = strat + 90_000.0
        print(f"  TRUTH  strategy (de-fauceted, 10k base): total {strat[-1]/strat[0]-1:+.1%}  maxDD {maxdd(strat):+.1%}")
        print(f"  TRUTH  raw (faucet-inflated, 100k base): total {raw[-1]/raw[0]-1:+.1%}  maxDD {maxdd(raw):+.1%}")
        print("         honest arm should track the STRATEGY row; a faucet-blind arm tracks RAW.\n")

        m = {a: load(a) for a in ARMS}
        print(f"{'arm':4} {'wrapper (defaults)':30} {'annRet':>10} {'annVol':>10} {'Sharpe':>10} {'maxDD':>10}")
        print(f"{'A':4} {'summarize_bot()  rf=0':30} " + fmt(grab(m['arm_a'].summarize_bot(store))))
        print(f"{'B':4} {'bot_perf()  rf=0.04 ppy~inf':30} " + fmt(grab(m['arm_b'].bot_perf(store))))
        print(f"{'C':4} {'bot_perf()  rf=0 periods=252':30} " + fmt(grab(m['arm_c'].bot_perf(store))))
        print(f"{'A+':4} {'summarize_bot(rf=0.04)':30} " + fmt(grab(m['arm_a'].summarize_bot(store, rf=0.04)))
              + "   <- A CAN do cost-of-cash, just off by default")


if __name__ == "__main__":
    main()
