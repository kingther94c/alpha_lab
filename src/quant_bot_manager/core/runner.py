"""Runner — turn a Bot into a planned/one-shot/continuous execution.

- ``make_plan``      : strategy targets -> priced order plan (no auth).
- ``print_plan``     : dry view.
- ``rebalance_once`` : connect-and-trade toward target once (gross-capped).
- ``run``            : the continuous process — mark-to-market each cycle, daily rebalance,
                       read config.json (live re-tune / pause), write status.json (UI monitoring).

Writes per-bot artifacts under ``config.paths(bot.name)`` with a stable schema the UI reads.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time

from quant_bot_manager.core import config


def utcnow():
    return dt.datetime.now(dt.UTC)


def _append_csv(path, header, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def make_plan(bot, capital: float, method: str):
    tgt, asof, last_px = bot.strategy(method)
    legs = [lg for lg in tgt.index if abs(float(tgt[lg])) >= config.EPS]
    px = bot.broker.public_prices(legs)
    for lg in legs:
        px.setdefault(lg, last_px.get(lg))
    px = {k: v for k, v in px.items() if v}
    plan = bot.broker.build_plan(tgt, capital, px)
    gross = float(plan["notional_usdt"].abs().sum()) / capital if not plan.empty else 0.0
    return plan, asof, gross


def print_plan(bot, capital: float, method: str):
    plan, asof, gross = make_plan(bot, capital, method)
    print(f"[{bot.name}] targets asof {str(asof)[:10]} | gross {gross:.2f}x | capital ${capital:,.0f} | {method}")
    if plan.empty:
        print("  (flat — nothing to trade)")
    else:
        for _, r in plan.iterrows():
            print(f"  {r.side:4s} {r.symbol:16s} ({r.venue:4s}) {r.qty:+.5f}  ~{r.notional_usdt:+,.0f} USDT @ {r.price:,.2f}")
    return plan, asof, gross


def rebalance_once(bot, capital: float, method: str, max_gross: float, *, from_flat: bool = False):
    plan, asof, gross = make_plan(bot, capital, method)
    if gross > max_gross:
        return [], asof, gross, "SKIPPED(gross>cap)"
    placed = bot.broker.rebalance_to_target(plan, from_flat=from_flat)
    return placed, asof, gross, "ok"


def run(bot, *, capital: float, method: str, max_gross: float, interval_min: float, max_cycles: int = 0):
    p = config.paths(bot.name)
    bot.broker.connect()
    state = json.loads(p["state"].read_text()) if p["state"].exists() else {"last_rebal_date": None}
    defaults = {"capital": capital, "method": method, "max_gross": max_gross,
                "interval_min": interval_min, "paused": False}
    pid, started = os.getpid(), utcnow().isoformat()
    print(f"[{bot.name}] START {started} pid={pid}", flush=True)

    cyc = 0
    while True:
        cyc += 1
        ts = utcnow()
        cfg = dict(defaults)
        if p["config"].exists():
            try:
                cfg.update({k: v for k, v in json.loads(p["config"].read_text()).items() if k in cfg})
            except Exception:
                pass
        eq = fe = se = None
        try:
            if cfg["paused"]:
                eq, fe, se = bot.broker.mark_to_market()
                print(f"[{bot.name}] {ts.isoformat()} PAUSED equity={eq:.2f}", flush=True)
            else:
                today = ts.date().isoformat()
                if state.get("last_rebal_date") != today:
                    placed, asof, gross, status = rebalance_once(bot, cfg["capital"], cfg["method"], cfg["max_gross"])
                    state["last_rebal_date"] = today
                    p["state"].write_text(json.dumps(state))
                    _append_csv(p["rebalance"], ["ts", "signal_asof", "gross", "status", "n_orders", "orders"],
                                [ts.isoformat(), str(asof), f"{gross:.3f}", status, len(placed),
                                 ";".join(f"{v} {s} {sy} {a}" for v, s, sy, a in placed)])
                    print(f"[{bot.name}] {ts.isoformat()} REBALANCED [{status}] {len(placed)} orders "
                          f"(asof {str(asof)[:10]}, gross {gross:.2f}x)", flush=True)
                eq, fe, se = bot.broker.mark_to_market()
                _append_csv(p["equity"], ["ts", "total_equity", "fut_equity", "spot_equity"],
                            [ts.isoformat(), f"{eq:.2f}", f"{fe:.2f}", f"{se:.2f}"])
                print(f"[{bot.name}] {ts.isoformat()} equity={eq:.2f} (fut {fe:.2f} + spot {se:.2f})", flush=True)
            p["status"].write_text(json.dumps(
                {"pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                 "last_rebal_date": state.get("last_rebal_date"), "config": cfg, "cycle": cyc,
                 "equity": eq, "fut_equity": fe, "spot_equity": se,
                 "positions": bot.broker.positions_snapshot(), "error": None}, default=str))
        except Exception as e:  # noqa: BLE001
            print(f"[{bot.name}] {ts.isoformat()} cycle error: {type(e).__name__}: {str(e)[:160]}", flush=True)
            try:
                p["status"].write_text(json.dumps(
                    {"pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                     "config": cfg, "error": f"{type(e).__name__}: {str(e)[:160]}"}, default=str))
            except Exception:
                pass
        if max_cycles and cyc >= max_cycles:
            print(f"[{bot.name}] reached max-cycles={max_cycles}, exiting.", flush=True)
            break
        time.sleep(cfg["interval_min"] * 60)
