"""Replicate the P7 multi-strategy book on Binance — paper (testnet) first, then live (gated).

Computes today's target weights via `alpha_lab.backtest.crypto_book` (the SAME code as the
backtest, so live == research), translates to per-instrument notional, connects to Binance via
ccxt, and rebalances current positions toward target.

Leg convention: spot legs ``.s`` trade the Binance **spot** account (long-only — the book never
shorts spot); perp legs ``.p`` trade **USD-M futures** (long/short).

Modes
  dry      compute targets + print the order plan from flat. Public price fetch only; NO orders.
  testnet  connect to Binance testnet (ccxt sandbox), read balances/positions, rebalance (mock $).
  live     REAL MONEY. Requires keys + ``--i-understand-live`` AND env ``CONFIRM_LIVE=YES``.

Usage
  python binance_paper_trade.py --mode dry --method equal_capital --capital 10000
  python binance_paper_trade.py --mode testnet --method equal_capital --capital 10000   # needs testnet keys in .env
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

# leg -> ccxt unified symbol
SPOT_SYM = {"BTC.s": "BTC/USDT", "ETH.s": "ETH/USDT"}
PERP_SYM = {"BTC.p": "BTC/USDT:USDT", "ETH.p": "ETH/USDT:USDT",
            "SOL.p": "SOL/USDT:USDT", "BNB.p": "BNB/USDT:USDT"}
EPS = 1e-4


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass


def compute_targets(method: str):
    """Today's per-leg target weights (fraction of capital) from the shared book code."""
    from alpha_lab.backtest.crypto_book import load_book_data, latest_target_weights
    end = (pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bd = load_book_data("2022-01-01", end, allow_holdout=True)
    tgt = latest_target_weights(bd, method=method)
    asof = bd.grid.max()
    last_px = {**{lg: float(bd.spot_close[lg].iloc[-1]) for lg in SPOT_SYM if lg in bd.spot_close},
               **{lg: float(bd.perp_close[lg].iloc[-1]) for lg in PERP_SYM if lg in bd.perp_close}}
    return tgt, asof, last_px


def public_prices(symbols_spot, symbols_perp):
    """Live mid prices from Binance public endpoints (no auth). Returns {} on failure."""
    import ccxt
    out = {}
    try:
        sp = ccxt.binance({"enableRateLimit": True})
        for lg, s in symbols_spot.items():
            out[lg] = float(sp.fetch_ticker(s)["last"])
        fu = ccxt.binanceusdm({"enableRateLimit": True})
        for lg, s in symbols_perp.items():
            out[lg] = float(fu.fetch_ticker(s)["last"])
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] public price fetch failed ({type(e).__name__}: {e}); using last close from data.")
    return out


def build_plan(tgt: pd.Series, capital: float, px: dict) -> pd.DataFrame:
    """Order plan to go from FLAT to target (the paper-trade starting point)."""
    rows = []
    for leg, w in tgt.items():
        if abs(w) < EPS or leg not in px:
            continue
        venue = "spot" if leg.endswith(".s") else "perp"
        sym = SPOT_SYM.get(leg) or PERP_SYM.get(leg)
        notional = w * capital
        qty = notional / px[leg]
        rows.append({"leg": leg, "venue": venue, "symbol": sym, "weight": w,
                     "side": "BUY" if w > 0 else "SELL", "price": px[leg],
                     "notional_usdt": notional, "qty": qty})
    return pd.DataFrame(rows)


def demo_clients():
    """Return (spot, fut) ccxt clients on the Binance DEMO environment (one demo key trades both)."""
    import ccxt
    dk, ds = os.environ.get("BINANCE_DEMO_KEY"), os.environ.get("BINANCE_DEMO_SECRET")
    if not (dk and ds):
        raise RuntimeError("no BINANCE_DEMO_KEY/SECRET in .env")
    spot = ccxt.binance({"apiKey": dk, "secret": ds, "enableRateLimit": True}); spot.enableDemoTrading(True)
    fut = ccxt.binanceusdm({"apiKey": dk, "secret": ds, "enableRateLimit": True}); fut.enableDemoTrading(True)
    spot.load_markets(); fut.load_markets()
    return spot, fut


def place_orders(spot, fut, plan, *, from_flat=False):
    """Market-order current positions toward `plan` (reconcile by default; skip <$5 dust).
    Returns the list of (venue, side, symbol, amount) actually submitted."""
    placed = []
    if fut is not None:
        cur = {} if from_flat else {
            p["symbol"]: float(p.get("contracts") or 0) * (1 if p.get("side") == "long" else -1)
            for p in fut.fetch_positions()}
        for _, r in plan[plan.venue == "perp"].iterrows():
            delta = r.qty - cur.get(r.symbol, 0.0)
            if abs(delta * r.price) < 5:
                continue
            amt = float(fut.amount_to_precision(r.symbol, abs(delta)))
            if amt <= 0:
                continue
            side = "buy" if delta > 0 else "sell"
            fut.create_order(r.symbol, "market", side, amt)
            placed.append(("perp", side, r.symbol, amt))
    if spot is not None:
        bal = {} if from_flat else spot.fetch_balance()
        for _, r in plan[plan.venue == "spot"].iterrows():
            base = r.symbol.split("/")[0]
            have = 0.0 if from_flat else float(bal.get(base, {}).get("free", 0) or 0)
            delta = r.qty - have
            if abs(delta * r.price) < 5:
                continue
            amt = float(spot.amount_to_precision(r.symbol, abs(delta)))
            if amt <= 0:
                continue
            side = "buy" if delta > 0 else "sell"
            spot.create_order(r.symbol, "market", side, amt)
            placed.append(("spot", side, r.symbol, amt))
    return placed


def main():
    ap = argparse.ArgumentParser(description="Paper/live trade the P7 book on Binance via ccxt.")
    ap.add_argument("--mode", choices=["dry", "demo", "testnet", "live"], default="dry")
    ap.add_argument("--method", choices=["equal_capital", "risk_budget"], default="equal_capital")
    ap.add_argument("--capital", type=float, default=10_000.0, help="capital in USDT")
    ap.add_argument("--max-gross", type=float, default=2.0, help="reject if gross exposure exceeds this x capital")
    ap.add_argument("--i-understand-live", action="store_true", help="required for --mode live")
    ap.add_argument("--from-flat", action="store_true",
                    help="place target orders ignoring current positions (clean first establishment; default reconciles)")
    args = ap.parse_args()
    _load_env()

    print(f"=== P7 book -> Binance | mode={args.mode} method={args.method} capital={args.capital:,.0f} USDT ===")
    tgt, asof, last_px = compute_targets(args.method)
    print(f"targets as of {asof.date()} (signal uses data <= t; execute next bar):")
    for leg, w in tgt.items():
        if abs(w) >= EPS:
            print(f"    {leg:7s} {w:+.4f}")

    px = public_prices({k: v for k, v in SPOT_SYM.items() if k in tgt.index},
                       {k: v for k, v in PERP_SYM.items() if k in tgt.index}) or {}
    for lg in tgt.index:  # fall back to last close where public fetch missing
        px.setdefault(lg, last_px.get(lg))
    px = {k: v for k, v in px.items() if v}

    plan = build_plan(tgt, args.capital, px)
    gross = float(plan["notional_usdt"].abs().sum()) / args.capital if not plan.empty else 0.0
    print(f"\norder plan (FLAT -> target), gross exposure = {gross:.2f}x capital:")
    if plan.empty:
        print("  (book is flat today — nothing to trade)")
    else:
        for _, r in plan.iterrows():
            print(f"    {r.side:4s} {r.symbol:16s} ({r.venue:4s})  {r.qty:+.5f}  ~{r.notional_usdt:+,.0f} USDT @ {r.price:,.2f}")
    if gross > args.max_gross:
        print(f"  [RISK] gross {gross:.2f}x exceeds --max-gross {args.max_gross:.2f}x. Reduce capital or cap before trading.")

    if args.mode == "dry":
        print("\n[dry] no connection authenticated, no orders placed. "
              "Add testnet keys to .env and rerun with --mode testnet to paper-trade.")
        return

    # ---- authenticated paths (demo / testnet / live) ----
    import ccxt
    live = args.mode == "live"
    if args.mode == "demo":
        # Binance Demo Trading (demo-*.binance.com): one key trades spot + USD-M futures via ccxt's
        # enableDemoTrading. This is the current Binance mock environment (testnet futures is deprecated).
        dk, ds = os.environ.get("BINANCE_DEMO_KEY"), os.environ.get("BINANCE_DEMO_SECRET")
        if not (dk and ds):
            print("\n[stop] no BINANCE_DEMO_KEY/SECRET in .env — cannot demo-trade.")
            return
        spot = ccxt.binance({"apiKey": dk, "secret": ds, "enableRateLimit": True}); spot.enableDemoTrading(True)
        fut = ccxt.binanceusdm({"apiKey": dk, "secret": ds, "enableRateLimit": True}); fut.enableDemoTrading(True)
        have_spot = have_fut = True
    else:
        pfx = "BINANCE_" + ("" if live else "TESTNET_")
        sk, ss = os.environ.get(pfx + "SPOT_KEY"), os.environ.get(pfx + "SPOT_SECRET")
        fk, fs = os.environ.get(pfx + "FUT_KEY"), os.environ.get(pfx + "FUT_SECRET")
        have_spot, have_fut = bool(sk and ss), bool(fk and fs)
        if not (have_spot or have_fut):
            print(f"\n[stop] no keys in .env ({pfx}SPOT_* and/or {pfx}FUT_*) — cannot {args.mode}-trade.")
            return
        if live and not (args.i_understand_live and os.environ.get("CONFIRM_LIVE") == "YES"):
            print("\n[stop] LIVE refused: pass --i-understand-live AND set CONFIRM_LIVE=YES in the environment.")
            return
        spot = ccxt.binance({"apiKey": sk, "secret": ss, "enableRateLimit": True}) if have_spot else None
        fut = ccxt.binanceusdm({"apiKey": fk, "secret": fs, "enableRateLimit": True}) if have_fut else None
        if not live:
            if spot:
                spot.set_sandbox_mode(True)
            if fut:  # ccxt deprecated set_sandbox_mode for binanceusdm; point URLs at the futures testnet.
                for kk, uu in list(fut.urls["api"].items()):
                    if isinstance(uu, str) and "fapi.binance.com" in uu:
                        fut.urls["api"][kk] = uu.replace("fapi.binance.com", "testnet.binancefuture.com")

    if gross > args.max_gross:
        print("\n[stop] gross exposure over the risk cap — refusing to send orders.")
        return
    need_perp, need_spot = bool((plan.venue == "perp").any()), bool((plan.venue == "spot").any())
    if need_perp and not have_fut:
        print("\n[warn] no futures access — SKIPPING perp legs (spot-only is NET-LONG, not the neutral book).")
    if need_spot and not have_spot:
        print("\n[warn] no spot access — SKIPPING spot legs.")
    if fut:
        fut.load_markets()
    if spot:
        spot.load_markets()
    mode_lbl = "FROM-FLAT" if args.from_flat else "reconcile"
    print(f"\n[{args.mode}] connected ({mode_lbl}). Sending orders toward target...")

    placed = place_orders(spot, fut, plan, from_flat=args.from_flat)
    for venue, side, sym, amt in placed:
        print(f"    {venue} {side} {sym} {amt}")
    print(f"[{args.mode}] orders submitted ({len(placed)} orders).")


if __name__ == "__main__":
    main()
