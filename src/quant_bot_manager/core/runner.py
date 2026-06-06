"""Runner — turn a Bot into a planned / one-shot / continuous execution.

- ``make_plan``      : strategy targets -> priced order plan (no auth).
- ``print_plan``     : dry view.
- ``rebalance_once`` : connect-and-trade toward target once (gross-capped via ``risk.gross_ok``).
- ``run``            : the continuous process. Each cycle: mark-to-market -> store; consult the
                       kill-switch (``risk.evaluate`` over the stored equity path); rebalance only
                       when allowed and not yet done today; write a rich status blob for the UI.

All runtime state goes through ``core.store.Store`` (SQLite); live config and the kill-switch are
read from the store every cycle, so the UI can re-tune / pause / halt a running bot.
"""
from __future__ import annotations

import datetime as dt
import os
import time

from quant_bot_manager.core import config, risk
from quant_bot_manager.core.schema import BotConfig
from quant_bot_manager.core.store import Store


def utcnow():
    return dt.datetime.now(dt.UTC)


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
    ok, why = risk.gross_ok(gross, max_gross)
    if not ok:
        return [], asof, gross, f"SKIPPED({why})"
    placed = bot.broker.rebalance_to_target(plan, from_flat=from_flat)
    return placed, asof, gross, "ok"


def run(bot, *, capital: float, method: str, max_gross: float, interval_min: float, max_cycles: int = 0):
    store = Store(bot.name)
    bot.broker.connect()
    # CLI flags set the baseline; the bot's YAML defaults fill the rest; the store config overrides
    # both live each cycle (so the UI can re-tune a running bot).
    base = (bot.default_config or BotConfig()).to_dict()
    base.update({"capital": capital, "method": method, "max_gross": max_gross, "interval_min": interval_min})
    pid, started = os.getpid(), utcnow().isoformat()
    print(f"[{bot.name}] START {started} pid={pid}", flush=True)

    cyc = 0
    while True:
        cyc += 1
        ts = utcnow()
        cfg = BotConfig.from_dict({**base, **store.read_config()})
        try:
            eq, fe, se = bot.broker.mark_to_market()
            store.append_equity(ts.isoformat(), eq, fe, se)

            decision = risk.evaluate(cfg, store.all_equity_totals(), auto_halted=store.get_auto_halted())
            if decision.triggered_auto_halt:
                store.set_auto_halted(True)
                print(f"[{bot.name}] {ts.isoformat()} *** AUTO-HALT *** drawdown {decision.drawdown:.1%} "
                      f"breached -{cfg.max_drawdown_pct:.0%}; trading latched off until cleared.", flush=True)

            if decision.can_trade:
                today = ts.date().isoformat()
                if store.get_last_rebal_date() != today:
                    placed, asof, gross, status = rebalance_once(bot, cfg.capital, cfg.method, cfg.max_gross)
                    store.set_last_rebal_date(today)
                    store.append_rebalance(ts.isoformat(), str(asof), gross, status, len(placed),
                                           ";".join(f"{v} {s} {sy} {a}" for v, s, sy, a in placed))
                    print(f"[{bot.name}] {ts.isoformat()} REBALANCED [{status}] {len(placed)} orders "
                          f"(asof {str(asof)[:10]}, gross {gross:.2f}x)", flush=True)
            elif decision.reasons:
                print(f"[{bot.name}] {ts.isoformat()} NO-TRADE: {'; '.join(decision.reasons)}", flush=True)

            store.write_status({
                "pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                "last_rebal_date": store.get_last_rebal_date(), "config": cfg.to_dict(), "cycle": cyc,
                "equity": eq, "fut_equity": fe, "spot_equity": se,
                "positions": bot.broker.positions_snapshot(),
                "can_trade": decision.can_trade, "halted": decision.halted,
                "auto_halted": store.get_auto_halted(), "drawdown": decision.drawdown,
                "risk_reasons": decision.reasons, "error": None})
            print(f"[{bot.name}] {ts.isoformat()} equity={eq:.2f} (fut {fe:.2f} + spot {se:.2f}) cyc={cyc}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad cycle must not kill the loop
            print(f"[{bot.name}] {ts.isoformat()} cycle error: {type(e).__name__}: {str(e)[:160]}", flush=True)
            try:
                store.write_status({"pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                                    "config": cfg.to_dict(), "error": f"{type(e).__name__}: {str(e)[:160]}"})
            except Exception:  # noqa: BLE001
                pass
        if max_cycles and cyc >= max_cycles:
            print(f"[{bot.name}] reached max-cycles={max_cycles}, exiting.", flush=True)
            break
        time.sleep(cfg.interval_min * 60)
