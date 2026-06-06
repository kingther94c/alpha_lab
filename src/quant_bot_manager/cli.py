"""quant_bot_manager CLI — the execution-leg entrypoint.

  python -m quant_bot_manager.cli bots                                       # list defined bots
  python -m quant_bot_manager.cli plan      --capital 10000                  # dry order plan (no auth)
  python -m quant_bot_manager.cli rebalance --mode demo --capital 10000      # one-shot rebalance
  python -m quant_bot_manager.cli run       --mode demo --interval-min 15    # continuous process

Bots are defined in ``configs/bots/*.yaml`` and assembled by ``core.registry``.
Live (real money) requires --mode live + --i-understand-live AND env CONFIRM_LIVE=YES.
"""
from __future__ import annotations

import argparse
import os

from quant_bot_manager.core import config, runner
from quant_bot_manager.core.registry import get_bot, list_bots
from quant_bot_manager.core.schema import DEFAULTS


def _common(p):
    p.add_argument("--bot", default=config.DEFAULT_BOT, help=f"one of: {list_bots()}")
    p.add_argument("--mode", choices=["demo", "testnet", "live"], default="demo")
    p.add_argument("--capital", type=float, default=DEFAULTS["capital"])
    p.add_argument("--method", choices=["equal_capital", "risk_budget"], default=DEFAULTS["method"])
    p.add_argument("--max-gross", type=float, default=DEFAULTS["max_gross"])
    p.add_argument("--i-understand-live", action="store_true", help="required for --mode live")


def _live_ok(args) -> bool:
    if args.mode != "live":
        return True
    if args.i_understand_live and os.environ.get("CONFIRM_LIVE") == "YES":
        return True
    print("[stop] LIVE refused: pass --i-understand-live AND set CONFIRM_LIVE=YES (real money).")
    return False


def main():
    ap = argparse.ArgumentParser(prog="quant_bot_manager", description="Run / rebalance / plan a trading bot.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bots", help="list available bots")
    pr = sub.add_parser("run", help="continuous process")
    _common(pr)
    pr.add_argument("--interval-min", type=float, default=DEFAULTS["interval_min"])
    pr.add_argument("--max-cycles", type=int, default=0)
    prb = sub.add_parser("rebalance", help="one-shot rebalance")
    _common(prb)
    prb.add_argument("--from-flat", action="store_true")
    ppl = sub.add_parser("plan", help="dry order plan (no auth)")
    ppl.add_argument("--bot", default=config.DEFAULT_BOT, help=f"one of: {list_bots()}")
    ppl.add_argument("--capital", type=float, default=DEFAULTS["capital"])
    ppl.add_argument("--method", choices=["equal_capital", "risk_budget"], default=DEFAULTS["method"])
    args = ap.parse_args()
    config.load_env()

    if args.cmd == "bots":
        for b in list_bots():
            print(b)
        return
    if args.cmd == "plan":
        runner.print_plan(get_bot(args.bot, "demo"), args.capital, args.method)
        return
    if not _live_ok(args):
        return
    bot = get_bot(args.bot, args.mode)
    if args.cmd == "run":
        runner.run(bot, capital=args.capital, method=args.method, max_gross=args.max_gross,
                   interval_min=args.interval_min, max_cycles=args.max_cycles)
    elif args.cmd == "rebalance":
        bot.broker.connect()
        placed, asof, gross, status = runner.rebalance_once(
            bot, args.capital, args.method, args.max_gross, from_flat=args.from_flat)
        print(f"[{args.mode}] rebalance {status}: {len(placed)} orders (asof {str(asof)[:10]}, gross {gross:.2f}x)")
        for v, s, sy, a in placed:
            print(f"  {v} {s} {sy} {a}")


if __name__ == "__main__":
    main()
