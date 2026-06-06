"""Continuous mock-trading PROCESS for the P7 book on Binance DEMO.

Runs forever:
  - every cycle  : mark-to-market -> append equity to data/results/crypto_v3_multi/mock_equity_log.csv
  - once/UTC day : pull fresh daily klines + funding from the Binance API (ccxt, leak-safe through
                   yesterday's close), recompute targets via crypto_book, and RECONCILE the demo book

Why live data: the Binance Vision archives used by the backtest lag ~1 month, so a live process must
source current daily bars from the API. Signals still use only data <= yesterday (the forming day's
bar is dropped), so it is leak-safe.

State (last rebalance date) persists to mock_state.json so a restart resumes without double-trading.
Stop with Ctrl-C / kill. For persistence across reboots, run under Task Scheduler / nohup.

Usage:
  python mock_trader_loop.py --interval-min 15 --capital 10000
  python mock_trader_loop.py --max-cycles 1            # one cycle, for testing
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np, pandas as pd
import ccxt
from alpha_lab.backtest import crypto_book as cb
from alpha_lab.data.loaders.yfinance import load_prices
from binance_paper_trade import public_prices, build_plan, demo_clients, place_orders, SPOT_SYM, PERP_SYM

OUT = ROOT / "data" / "results" / "crypto_v3_multi"; OUT.mkdir(parents=True, exist_ok=True)
EQLOG, REBLOG, STATE = OUT / "mock_equity_log.csv", OUT / "mock_rebalance_log.csv", OUT / "mock_state.json"
CONFIG, STATUS = OUT / "bot_config.json", OUT / "bot_status.json"   # UI <-> bot control + monitoring
PERP_CCXT = {"BTC.p": "BTC/USDT:USDT", "ETH.p": "ETH/USDT:USDT", "SOL.p": "SOL/USDT:USDT", "BNB.p": "BNB/USDT:USDT"}
SPOT_CCXT = {"BTC.s": "BTC/USDT", "ETH.s": "ETH/USDT"}


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def build_live_bookdata(lookback_days: int = 420) -> "cb.BookData":
    """Construct a crypto_book.BookData from live Binance API data (daily, through yesterday)."""
    spot_ex = ccxt.binance({"enableRateLimit": True})
    fut_ex = ccxt.binanceusdm({"enableRateLimit": True})

    def closes(ex, sym):
        d = ex.fetch_ohlcv(sym, "1d", limit=lookback_days)
        return pd.Series({pd.Timestamp(r[0], unit="ms", tz="UTC").normalize(): float(r[4]) for r in d})

    perp_close = pd.DataFrame({leg: closes(fut_ex, s) for leg, s in PERP_CCXT.items()})
    spot_close = pd.DataFrame({leg: closes(spot_ex, s) for leg, s in SPOT_CCXT.items()})
    grid = perp_close.index.union(spot_close.index)
    today = utcnow().date().isoformat()
    grid = grid[grid < pd.Timestamp(today, tz="UTC")]            # drop today's forming bar -> leak-safe
    perp_close = perp_close.reindex(grid).ffill()
    spot_close = spot_close.reindex(grid).ffill()
    prices = pd.concat([spot_close, perp_close], axis=1)

    def funding_hist(sym):
        d = fut_ex.fetch_funding_rate_history(sym, limit=1000)
        return pd.Series({pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in d})

    funding = pd.DataFrame({leg: funding_hist(s) for leg, s in PERP_CCXT.items()}).sort_index()
    df_fund = cb._daily_funding(funding, grid)

    hyg = load_prices("HYG", str(grid.min().date()), None)["HYG"]
    hyg.index = pd.DatetimeIndex(hyg.index).tz_localize("UTC")
    hyg = hyg.reindex(grid).ffill()

    naive = grid.tz_localize(None).normalize()
    rf_daily = pd.Series([cb.RF_FALLBACK.get(d.year, 0.04) / 365 for d in naive], index=grid).astype(float)
    return cb.BookData(grid=grid, perp_close=perp_close, spot_close=spot_close, funding=funding,
                       df_fund=df_fund, hyg=hyg, rf_daily=rf_daily, prices=prices,
                       rf_source="fallback", macro_source="yfinance HYG (live)")


def mark_to_market(spot, fut):
    fb = fut.fetch_balance()
    fe = float(fb.get("info", {}).get("totalMarginBalance") or fb["total"].get("USDT", 0))
    sb = spot.fetch_balance()["total"]
    tk = spot.fetch_tickers(["BTC/USDT", "ETH/USDT"])
    se = (sb.get("USDT", 0) or 0) + (sb.get("USDC", 0) or 0) \
        + (sb.get("BTC", 0) or 0) * tk["BTC/USDT"]["last"] + (sb.get("ETH", 0) or 0) * tk["ETH/USDT"]["last"]
    return fe + se, fe, se


def do_rebalance(spot, fut, method, capital, max_gross):
    bd = build_live_bookdata()
    tgt = cb.latest_target_weights(bd, method=method)
    asof = bd.grid.max()
    last_px = {**{lg: float(bd.spot_close[lg].iloc[-1]) for lg in SPOT_CCXT if lg in bd.spot_close},
               **{lg: float(bd.perp_close[lg].iloc[-1]) for lg in PERP_CCXT if lg in bd.perp_close}}
    px = public_prices({k: v for k, v in SPOT_SYM.items() if k in tgt.index},
                       {k: v for k, v in PERP_SYM.items() if k in tgt.index}) or {}
    for lg in tgt.index:
        px.setdefault(lg, last_px.get(lg))
    px = {k: v for k, v in px.items() if v}
    plan = build_plan(tgt, capital, px)
    gross = float(plan["notional_usdt"].abs().sum()) / capital if not plan.empty else 0.0
    if gross > max_gross:
        return [], asof, gross, "SKIPPED(gross>cap)"
    placed = place_orders(spot, fut, plan, from_flat=False)
    return placed, asof, gross, "ok"


def append_csv(path, header, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def read_config(defaults: dict) -> dict:
    """Merge launch defaults with bot_config.json (UI-written) so a running bot can be re-tuned."""
    cfg = dict(defaults)
    if CONFIG.exists():
        try:
            cfg.update({k: v for k, v in json.loads(CONFIG.read_text()).items() if k in cfg})
        except Exception:
            pass
    return cfg


def write_status(d: dict) -> None:
    STATUS.write_text(json.dumps(d, default=str))


def positions_snapshot(spot, fut) -> dict:
    out = {"perp": [], "spot": {}}
    try:
        for p in fut.fetch_positions():
            c = float(p.get("contracts") or 0)
            if c:
                out["perp"].append({"symbol": p["symbol"], "side": p.get("side"), "contracts": c,
                                    "notional": float(p.get("notional") or 0),
                                    "uPnl": float(p.get("unrealizedPnl") or 0), "entry": p.get("entryPrice")})
        b = spot.fetch_balance()["total"]
        out["spot"] = {a: round(float(v), 6) for a, v in b.items() if v and a in ("BTC", "ETH", "USDT", "USDC")}
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:100]
    return out


def main():
    ap = argparse.ArgumentParser(description="Continuous mock-trading process (Binance demo).")
    ap.add_argument("--interval-min", type=float, default=15.0)
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--method", choices=["equal_capital", "risk_budget"], default="equal_capital")
    ap.add_argument("--max-gross", type=float, default=2.0)
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()

    spot, fut = demo_clients()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"last_rebal_date": None}
    defaults = {"capital": args.capital, "method": args.method, "max_gross": args.max_gross,
                "interval_min": args.interval_min, "paused": False}
    pid, started = os.getpid(), utcnow().isoformat()
    print(f"[mock-loop] START {started} pid={pid}", flush=True)

    cyc = 0
    while True:
        cyc += 1
        ts = utcnow()
        cfg = read_config(defaults)
        eq = fe = se = None
        try:
            if cfg["paused"]:
                eq, fe, se = mark_to_market(spot, fut)
                print(f"[mock-loop] {ts.isoformat()} PAUSED equity={eq:.2f}", flush=True)
            else:
                today = ts.date().isoformat()
                if state.get("last_rebal_date") != today:
                    placed, asof, gross, status = do_rebalance(spot, fut, cfg["method"], cfg["capital"], cfg["max_gross"])
                    state["last_rebal_date"] = today
                    STATE.write_text(json.dumps(state))
                    append_csv(REBLOG, ["ts", "signal_asof", "gross", "status", "n_orders", "orders"],
                               [ts.isoformat(), str(asof), f"{gross:.3f}", status, len(placed),
                                ";".join(f"{v} {s} {sy} {a}" for v, s, sy, a in placed)])
                    print(f"[mock-loop] {ts.isoformat()} REBALANCED [{status}] {len(placed)} orders "
                          f"(asof {str(asof)[:10]}, gross {gross:.2f}x)", flush=True)
                eq, fe, se = mark_to_market(spot, fut)
                append_csv(EQLOG, ["ts", "total_equity", "fut_equity", "spot_equity"],
                           [ts.isoformat(), f"{eq:.2f}", f"{fe:.2f}", f"{se:.2f}"])
                print(f"[mock-loop] {ts.isoformat()} equity={eq:.2f} (fut {fe:.2f} + spot {se:.2f})", flush=True)
            write_status({"pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                          "last_rebal_date": state.get("last_rebal_date"), "config": cfg, "cycle": cyc,
                          "equity": eq, "fut_equity": fe, "spot_equity": se,
                          "positions": positions_snapshot(spot, fut), "error": None})
        except Exception as e:  # noqa: BLE001
            print(f"[mock-loop] {ts.isoformat()} cycle error: {type(e).__name__}: {str(e)[:160]}", flush=True)
            try:
                write_status({"pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                              "config": cfg, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            except Exception:
                pass
        if args.max_cycles and cyc >= args.max_cycles:
            print(f"[mock-loop] reached max-cycles={args.max_cycles}, exiting.", flush=True)
            break
        time.sleep(cfg["interval_min"] * 60)


if __name__ == "__main__":
    main()
