"""Offline end-to-end runner cycle (store + risk + status integration), no network/exchange."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_bot_manager.brokers.base import Broker
from quant_bot_manager.core.bot import Bot
from quant_bot_manager.core.schema import BotConfig


class FakeBroker(Broker):
    name = "fake"

    def __init__(self, *, place_empty: bool = False):
        self.orders: list = []
        self.place_empty = place_empty           # simulate all-legs-failed / broker returns placed=[] w/o raising

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
        if self.place_empty:
            return []                            # no leg filled (e.g. binance swallows per-leg errors -> false 'ok')
        placed = [("perp", r.side.lower(), r.symbol, abs(r.qty)) for _, r in plan.iterrows()]
        self.orders += placed
        return placed

    def mark_to_market(self):
        return (1000.0, 600.0, 400.0)

    def positions_snapshot(self):
        return {"perp": [], "spot": {}}


def _fake_strategy(method: str = "equal_capital"):
    tgt = pd.Series({"BTC.p": 0.5, "ETH.p": -0.5})
    asof = pd.Timestamp(dt.datetime.now(dt.UTC).date(), tz="UTC")   # fresh (today) so it isn't flagged STALE
    return tgt, asof, {"BTC.p": 100.0, "ETH.p": 100.0}


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


# -- P7-14a money-path regression net (specs for P7-01b: kill-switch on the manual path + self-heal) --
# The two self-heal specs assert observable store state through the stable public runner.run path;
# the manual-halt spec pins the store-aware runner.rebalance_once signature P7-01b introduces.
# All three are RED until P7-01b lands, then GREEN — a visibly red spec is the point (do not xfail-hide).

def test_skipped_rebalance_does_not_latch_the_day(tmp_path, monkeypatch):
    """A gross-capped SKIPPED cycle must NOT stamp last_rebal_date — else a transient gross spike
    freezes the book for 24h instead of retrying next cycle."""
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    broker = FakeBroker()
    bot = Bot("t_skip", _fake_strategy, broker, BotConfig(interval_min=1))
    runner.run(bot, capital=10000.0, method="equal_capital", max_gross=0.01, interval_min=1, max_cycles=1)

    s = store_mod.Store("t_skip")
    assert broker.orders == []                          # gross-capped: nothing placed (true today)
    assert s.get_last_rebal_date() is None              # self-heal: SKIPPED must retry, not latch (RED until P7-01b)


def test_failed_rebalance_does_not_latch_the_day(tmp_path, monkeypatch):
    """An all-legs-failed fill (broker returns placed=[] without raising — the binance false-'ok')
    must NOT stamp last_rebal_date; the next cycle must retry."""
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    broker = FakeBroker(place_empty=True)
    bot = Bot("t_fail", _fake_strategy, broker, BotConfig(interval_min=1))
    runner.run(bot, capital=10000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=1)

    s = store_mod.Store("t_fail")
    assert s.get_last_rebal_date() is None              # self-heal: empty/failed fill must retry (RED until P7-01b)


def test_manual_rebalance_refused_when_halted(tmp_path, monkeypatch):
    """The cockpit 'Rebalance now' / `cli rebalance` path must consult the kill-switch too. P7-01b makes
    runner.rebalance_once store-aware and read halt FRESH from the store; this is its executable spec."""
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    s = store_mod.Store("t_mhalt")
    s.write_config({"halt": True})                       # kill-switch engaged in the store
    broker = FakeBroker()
    bot = Bot("t_mhalt", _fake_strategy, broker, BotConfig(interval_min=1))

    placed, asof, gross, status = runner.rebalance_once(bot, BotConfig(), store=s)
    assert placed == [] and broker.orders == []          # zero orders while halted
    assert "BLOCK" in status.upper() or "HALT" in status.upper()


def test_manual_rebalance_stamps_shared_daily_ledger(tmp_path, monkeypatch):
    """A successful manual rebalance stamps last_rebal_date, so the auto loop won't re-trade the same
    UTC day — manual + auto share one ledger (no double-trade)."""
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    s = store_mod.Store("t_shared")
    broker = FakeBroker()
    bot = Bot("t_shared", _fake_strategy, broker, BotConfig(interval_min=1))

    placed, asof, gross, status = runner.rebalance_once(bot, BotConfig(max_gross=2.0), store=s)
    assert status == "ok" and placed and len(broker.orders) == 2   # manual rebalance filled
    assert s.get_last_rebal_date() is not None                     # ...and stamped the shared ledger

    runner.run(bot, capital=10000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=1)
    assert len(broker.orders) == 2                                 # same-day auto cycle no-ops; no re-trade


# -- P7-05 de-faucet: the drawdown kill-switch must read honest (de-fauceted) strategy equity --------

class _DecliningBroker(FakeBroker):
    """FakeBroker whose mark-to-market follows a fixed sequence (last value sticks)."""

    def __init__(self, marks):
        super().__init__()
        self._marks, self._i = list(marks), 0

    def mark_to_market(self):
        m = self._marks[min(self._i, len(self._marks) - 1)]
        self._i += 1
        return (m, m, 0.0)


def test_runner_defaucets_so_killswitch_sees_true_drawdown(tmp_path, monkeypatch):
    """A demo faucet top-up makes RAW equity barely move (110k -> 102k, -7%), but the strategy's real
    equity (capital + PnL) fell 10k -> 2k (-80%). The kill-switch must fire on the de-fauceted series."""
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    broker = _DecliningBroker([110_000.0, 102_000.0])             # faucet 100k + 10k capital, then -8k
    bot = Bot("t_df", _fake_strategy, broker, BotConfig(interval_min=1, max_drawdown_pct=0.20))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)   # no real 60s sleep between cycles
    runner.run(bot, capital=10_000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=2)

    s = store_mod.Store("t_df")
    assert s.get_faucet_offset() == 100_000.0                    # set once on cycle 1: 110k - 10k capital
    assert s.get_auto_halted() is True                           # -80% strategy DD breached -20% (raw -7% would not)
    assert s.read_status().get("strategy_equity") == 2_000.0     # honest equity surfaced for the operator


# -- P7-03 out-of-band alerting: tell the operator on state TRANSITIONS, not every cycle -------------

def test_runner_alerts_out_of_band_on_auto_halt(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import notify, runner
    sent: list = []
    monkeypatch.setattr(notify, "send", lambda event, detail: sent.append((event, detail)) is None)
    broker = _DecliningBroker([110_000.0, 102_000.0])
    bot = Bot("t_alert", _fake_strategy, broker, BotConfig(interval_min=1, max_drawdown_pct=0.20))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)
    runner.run(bot, capital=10_000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=2)
    assert [e for e, _ in sent] == ["AUTO-HALT"]                 # told once when the kill-switch latched


def test_runner_alerts_once_after_consecutive_errors(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import notify, runner
    sent: list = []
    monkeypatch.setattr(notify, "send", lambda event, detail: sent.append((event, detail)) is None)

    class _BoomBroker(FakeBroker):
        def mark_to_market(self):
            raise RuntimeError("exchange down")

    bot = Bot("t_boom", _fake_strategy, _BoomBroker(), BotConfig(interval_min=1))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)
    runner.run(bot, capital=10_000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=5)
    assert [e for e, _ in sent] == ["ERROR"]                    # alert-once-per-streak (cycle 3), not every cycle


def test_notify_is_a_safe_noop_without_a_webhook(monkeypatch):
    from quant_bot_manager.core import notify
    monkeypatch.delenv(notify.ALERT_ENV, raising=False)
    assert notify.send("X", "y") is False                       # no webhook configured -> no-op, never raises


# -- P7-16 single-flight: the loop and a manual rebalance can't double-trade the same account ---------

def test_rebalance_once_is_a_noop_when_already_done_today(tmp_path, monkeypatch):
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    s = store_mod.Store("t_done")
    broker = FakeBroker()
    bot = Bot("t_done", _fake_strategy, broker, BotConfig(interval_min=1))
    _, _, _, st1 = runner.rebalance_once(bot, BotConfig(max_gross=2.0), store=s)
    assert st1 == "ok" and len(broker.orders) == 2                      # first manual rebalance trades + stamps today
    placed, _, _, st2 = runner.rebalance_once(bot, BotConfig(max_gross=2.0), store=s)
    assert placed == [] and "DONE" in st2 and len(broker.orders) == 2   # same UTC day -> shared-ledger no-op


def test_rebalance_once_is_busy_when_lock_held(tmp_path, monkeypatch):
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    s = store_mod.Store("t_busy")
    s.try_claim_rebalance_lock("other", "2099-01-01T00:00:00")          # a fresh (future-dated) holder elsewhere
    broker = FakeBroker()
    bot = Bot("t_busy", _fake_strategy, broker, BotConfig(interval_min=1))
    placed, _, _, status = runner.rebalance_once(bot, BotConfig(max_gross=2.0), store=s)
    assert placed == [] and "BUSY" in status and broker.orders == []    # single-flight refuses the second


# -- P7-08 signal-freshness: leak-safe != leak-fresh — never trade a lagged/failed feed silently --------

def _stale_strategy(method: str = "equal_capital"):
    tgt = pd.Series({"BTC.p": 0.5, "ETH.p": -0.5})
    return tgt, pd.Timestamp("2020-01-01", tz="UTC"), {"BTC.p": 100.0, "ETH.p": 100.0}   # years-old signal


def test_rebalance_once_skips_a_stale_signal(tmp_path, monkeypatch):
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    s = store_mod.Store("t_stale")
    broker = FakeBroker()
    bot = Bot("t_stale", _stale_strategy, broker, BotConfig(interval_min=1))
    placed, asof, gross, status = runner.rebalance_once(bot, BotConfig(max_gross=2.0), store=s)
    assert placed == [] and "STALE" in status and broker.orders == []   # don't trade an old signal
    assert s.get_last_rebal_date() is None                              # not stamped -> self-heals when feed recovers


def test_runner_alerts_once_on_a_stale_feed(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import notify, runner
    sent: list = []
    monkeypatch.setattr(notify, "send", lambda event, detail: sent.append((event, detail)) is None)
    bot = Bot("t_stalealert", _stale_strategy, FakeBroker(), BotConfig(interval_min=1))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)
    runner.run(bot, capital=10_000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=3)
    assert [e for e, _ in sent] == ["STALE"]                            # told once, not every stale cycle


# -- cost-of-cash (AGENTS.md #3): live monitoring reports equity EXCESS of the risk-free financing hurdle --

def test_rf_hurdle_is_cost_of_deployed_capital():
    from quant_bot_manager.core import runner
    assert runner.rf_hurdle(10_000.0, 365.25) == 400.0                  # 4% of 10k over one year
    assert runner.rf_hurdle(10_000.0, 0.0) == 0.0                       # no time elapsed -> no hurdle yet


def test_runner_reports_excess_of_cost_of_cash(tmp_path, monkeypatch):
    store_mod = _patch_paths(monkeypatch, tmp_path)
    from quant_bot_manager.core import runner
    s = store_mod.Store("t_coc")
    s.append_equity("2025-06-14T00:00:00+00:00", 1_000.0, 600.0, 400.0)  # a ~year-old first mark
    bot = Bot("t_coc", _fake_strategy, FakeBroker(), BotConfig(interval_min=1))
    runner.run(bot, capital=10_000.0, method="equal_capital", max_gross=2.0, interval_min=1, max_cycles=1)
    st = s.read_status()
    assert st["rf_cost"] > 300                                          # ~1yr of 4% on 10k ≈ 400 (was charged 0 before)
    assert abs(st["excess_equity"] - (st["strategy_equity"] - st["rf_cost"])) < 1e-9
