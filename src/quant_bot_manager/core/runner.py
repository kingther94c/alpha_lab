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

from quant_bot_manager.core import config, notify, risk
from quant_bot_manager.core.schema import BotConfig
from quant_bot_manager.core.store import Store


def utcnow():
    return dt.datetime.now(dt.UTC)


ALERT_AFTER_ERRORS = 3   # consecutive cycle errors before ONE out-of-band alert (de-duped per streak)
MAX_SIGNAL_AGE_DAYS = 4  # a signal older than this (lagged/failed feed) is NOT traded (leak-safe != leak-fresh)
RF_ANNUAL = 0.04         # cost-of-cash hurdle rate for live monitoring (~ research RF_FALLBACK; reporting only)


def rf_hurdle(capital: float, elapsed_days: float, rf_annual: float = RF_ANNUAL) -> float:
    """Financing cost of the deployed capital over ``elapsed_days`` (AGENTS.md cost-of-cash) — the risk-free
    hurdle the strategy must clear, so 'is it making money' means 'beat cash', not 'beat zero'."""
    return capital * rf_annual * max(elapsed_days, 0.0) / 365.25


def _days_since(first_ts, now) -> float:
    """Days from an ISO timestamp string to ``now`` (tz-aware); 0 if missing/unparseable."""
    if not first_ts:
        return 0.0
    try:
        first = dt.datetime.fromisoformat(first_ts)
        if first.tzinfo is None:
            first = first.replace(tzinfo=dt.UTC)
        return max((now - first).total_seconds() / 86400.0, 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def rebalance_once(bot, cfg: BotConfig, *, store: Store | None = None, from_flat: bool = False):
    """Rebalance toward target once, honoring the kill-switch AND the gross cap on EVERY caller —
    the runner loop, ``cli rebalance``, and the cockpit's 'Rebalance now'. ``cfg`` is the baseline;
    halt / paused / latched auto-halt and any re-tuned knobs are read FRESH from ``store`` and
    override it, so a halted/paused/auto-halted bot is refused with zero orders no matter which path
    calls in. A cross-process lock serializes the critical section so the loop and a manual rebalance
    can't double-trade. Returns (placed, asof, gross, status); status is one of
    'ok' (only this traded) | 'SKIPPED' (gross cap) | 'BLOCKED' (kill-switch) | 'BUSY' (lock held) | 'DONE' (already today).
    """
    store = store or Store(bot.name)
    live = BotConfig.from_dict({**cfg.to_dict(), **store.read_config()})
    decision = risk.evaluate(live, store.all_equity_totals(), auto_halted=store.get_auto_halted())
    if not decision.can_trade:                        # kill-switch covers the manual path too
        return [], None, 0.0, f"BLOCKED({'; '.join(decision.reasons) or 'halted'})"
    # single-flight: serialize the read-gate -> place -> stamp critical section across processes, so the
    # runner loop and a manual rebalance can never double-trade the same account (a crashed holder's
    # stale lock is auto-stolen, so this can't deadlock the bot).
    if not store.try_claim_rebalance_lock(f"pid{os.getpid()}", utcnow().isoformat()):
        return [], None, 0.0, "BUSY(another rebalance in flight)"
    try:
        today = utcnow().date().isoformat()
        if store.get_last_rebal_date() == today:      # already rebalanced today (by us or the other path)
            return [], None, 0.0, "DONE(already rebalanced today)"
        plan, asof, gross = make_plan(bot, live.capital, live.method)
        age = (utcnow().date() - asof.date()).days if asof is not None else None
        if age is None or age > MAX_SIGNAL_AGE_DAYS:   # stale/failed feed: skip + self-heal (don't stamp the day)
            return [], asof, gross, f"STALE(signal {age if age is not None else '?'}d old)"
        ok, why = risk.gross_ok(gross, live.max_gross)
        if not ok:
            return [], asof, gross, f"SKIPPED({why})"
        placed = bot.broker.rebalance_to_target(plan, from_flat=from_flat)
        if placed:
            store.set_last_rebal_date(today)          # stamp the shared daily ledger on a real fill (else self-heal)
        return placed, asof, gross, "ok"
    finally:
        store.release_rebalance_lock()


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
    err_streak = 0          # consecutive cycle errors; alert once when the streak crosses the threshold
    stale_alerted = False   # de-dup the STALE alert: fire once per stale-feed episode
    while True:
        cyc += 1
        ts = utcnow()
        cfg = BotConfig.from_dict({**base, **store.read_config()})
        try:
            eq, fe, se = bot.broker.mark_to_market()
            store.append_equity(ts.isoformat(), eq, fe, se)
            if store.get_faucet_offset() is None:           # set once: free demo-faucet cash above strategy capital
                totals = store.all_equity_totals()
                store.set_faucet_offset((totals[0] if totals else eq) - cfg.capital)

            # drawdown kill-switch reads DE-FAUCETED equity, so a faucet top-up can't hide a real drawdown
            decision = risk.evaluate(cfg, store.all_strategy_equity(), auto_halted=store.get_auto_halted())
            if decision.triggered_auto_halt:
                store.set_auto_halted(True)
                notify.send("AUTO-HALT", f"{bot.name}: drawdown {decision.drawdown:.1%} breached "
                            f"-{cfg.max_drawdown_pct:.0%}; trading latched off until cleared.")
                print(f"[{bot.name}] {ts.isoformat()} *** AUTO-HALT *** drawdown {decision.drawdown:.1%} "
                      f"breached -{cfg.max_drawdown_pct:.0%}; trading latched off until cleared.", flush=True)

            if decision.can_trade:
                today = ts.date().isoformat()
                if store.get_last_rebal_date() != today:
                    # rebalance_once latches the shared daily ledger itself, but ONLY on a real fill;
                    # a SKIPPED / empty / BLOCKED cycle leaves it unset so the next cycle retries (self-heal).
                    # Known residuals (tracked items, not regressions): a PARTIAL fill still latches and leaves a
                    # half-rebalanced book until tomorrow (-> P7-11b per-leg fill capture); a manual rebalance
                    # racing this loop can both pass the date gate before either stamps (-> P7-16 single-flight);
                    # the audit row is appended AFTER the broker call, so a hard kill mid-placement loses the
                    # structured row (run.log still records it) — full intent/result bracket deferred.
                    placed, asof, gross, status = rebalance_once(bot, cfg, store=store)
                    store.append_rebalance(ts.isoformat(), str(asof), gross, status, len(placed),
                                           ";".join(f"{v} {s} {sy} {a}" for v, s, sy, a in placed))
                    print(f"[{bot.name}] {ts.isoformat()} REBALANCE [{status}] {len(placed)} orders "
                          f"(asof {str(asof)[:10]}, gross {gross:.2f}x)", flush=True)
                    if status.startswith("STALE") and not stale_alerted:
                        notify.send("STALE", f"{bot.name}: {status} — not trading until the data feed catches up.")
                        stale_alerted = True
                    elif not status.startswith("STALE"):
                        stale_alerted = False
            elif decision.reasons:
                print(f"[{bot.name}] {ts.isoformat()} NO-TRADE: {'; '.join(decision.reasons)}", flush=True)

            strat_eq = eq - (store.get_faucet_offset() or 0.0)
            rf_cost = rf_hurdle(cfg.capital, _days_since(store.first_equity_ts(), ts))   # cost-of-cash hurdle
            store.write_status({
                "pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                "last_rebal_date": store.get_last_rebal_date(), "config": cfg.to_dict(), "cycle": cyc,
                "equity": eq, "fut_equity": fe, "spot_equity": se,
                "strategy_equity": strat_eq, "faucet_offset": store.get_faucet_offset(),
                "rf_cost": rf_cost, "excess_equity": strat_eq - rf_cost,   # equity NET of the risk-free financing hurdle
                "positions": bot.broker.positions_snapshot(),
                "can_trade": decision.can_trade, "halted": decision.halted,
                "auto_halted": store.get_auto_halted(), "drawdown": decision.drawdown,
                "risk_reasons": decision.reasons, "error": None})
            print(f"[{bot.name}] {ts.isoformat()} equity={eq:.2f} (fut {fe:.2f} + spot {se:.2f}) cyc={cyc}", flush=True)
            err_streak = 0                              # a clean cycle resets the error streak
        except Exception as e:  # noqa: BLE001 — one bad cycle must not kill the loop
            err_streak += 1
            print(f"[{bot.name}] {ts.isoformat()} cycle error #{err_streak}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            if err_streak == ALERT_AFTER_ERRORS:
                notify.send("ERROR", f"{bot.name}: {err_streak} consecutive cycle errors; "
                            f"latest {type(e).__name__}: {str(e)[:120]}")
            try:
                store.write_status({"pid": pid, "started_at": started, "last_heartbeat": ts.isoformat(),
                                    "config": cfg.to_dict(), "error": f"{type(e).__name__}: {str(e)[:160]}"})
            except Exception:  # noqa: BLE001
                pass
        if max_cycles and cyc >= max_cycles:
            print(f"[{bot.name}] reached max-cycles={max_cycles}, exiting.", flush=True)
            break
        time.sleep(cfg.interval_min * 60)
