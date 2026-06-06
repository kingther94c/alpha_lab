"""Offline end-to-end runner cycle (store + risk + status integration), no network/exchange."""
from __future__ import annotations

import pandas as pd

from quant_bot_manager.brokers.base import Broker
from quant_bot_manager.core.bot import Bot
from quant_bot_manager.core.schema import BotConfig


class FakeBroker(Broker):
    name = "fake"

    def __init__(self):
        self.orders: list = []

    def connect(self) -> None:
        pass

    def public_prices(self, legs):
        return {lg: 100.0 for lg in legs}

    def build_plan(self, targets, capital, prices):
        rows = [{"leg": lg, "venue": "perp", "symbol": lg, "weight": w,
                 "side": "BUY" if w > 0 else "SELL", "price": prices.get(lg, 100.0),
                 "notional_usdt": w * capital, "qty": w * capital / prices.get(lg, 100.0)}
                for lg, w in targets.items() if abs(w) > 1e-9]
        return pd.DataFrame(rows)

    def rebalance_to_target(self, plan, *, from_flat=False):
        placed = [("perp", r.side.lower(), r.symbol, abs(r.qty)) for _, r in plan.iterrows()]
        self.orders += placed
        return placed

    def mark_to_market(self):
        return (1000.0, 600.0, 400.0)

    def positions_snapshot(self):
        return {"perp": [], "spot": {}}


def _fake_strategy(method: str = "equal_capital"):
    tgt = pd.Series({"BTC.p": 0.5, "ETH.p": -0.5})
    return tgt, pd.Timestamp("2026-01-01", tz="UTC"), {"BTC.p": 100.0, "ETH.p": 100.0}


def _patch_paths(monkeypatch, tmp_path):
    from quant_bot_manager.core import store as store_mod

    def fake_paths(bot):
        d = tmp_path / bot
        d.mkdir(parents=True, exist_ok=True)
        return {"dir": d, "db": d / "bot.db", "equity": d / "e.csv", "rebalance": d / "r.csv",
                "state": d / "s.json", "config": d / "c.json", "status": d / "st.json"}

    monkeypatch.setattr(store_mod.config, "paths", fake_paths)
    return store_mod


def test_run_one_cycle_trades_and_records(tmp_path, monkeypatch):
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    bot = Bot("t_runner", _fake_strategy, FakeBroker(), BotConfig(interval_min=1))
    runner.run(bot, capital=10000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=1)

    s = store_mod.Store("t_runner")
    status = s.read_status()
    assert status.get("cycle") == 1 and status.get("can_trade") is True and status.get("error") is None
    assert s.all_equity_totals() == [1000.0]
    assert s.get_last_rebal_date() is not None
    reb = s.read_rebalances_df()
    assert len(reb) == 1 and reb.iloc[0]["status"] == "ok"


def test_run_halted_marks_but_does_not_trade(tmp_path, monkeypatch):
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    store_mod.Store("t_halt").write_config({"halt": True})   # engage kill-switch before the bot starts
    bot = Bot("t_halt", _fake_strategy, FakeBroker(), BotConfig(interval_min=1))
    runner.run(bot, capital=10000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=1)

    s = store_mod.Store("t_halt")
    status = s.read_status()
    assert status.get("halted") is True and status.get("can_trade") is False
    assert s.all_equity_totals() == [1000.0]            # still marks-to-market
    assert s.get_last_rebal_date() is None              # but never rebalanced
    assert len(s.read_rebalances_df()) == 0
