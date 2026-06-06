"""Risk gates + kill-switch (execution leg, Option A)."""
from __future__ import annotations

from quant_bot_manager.core.risk import drawdown, evaluate, gross_ok
from quant_bot_manager.core.schema import BotConfig


def test_drawdown():
    assert drawdown([]) == 0.0
    assert drawdown([100, 110, 120]) == 0.0                 # monotonic up -> no drawdown
    assert abs(drawdown([100, 80]) - (-0.2)) < 1e-9
    assert abs(drawdown([100, 150, 120]) - (-0.2)) < 1e-9   # peak 150 -> 120 = -20%


def test_gross_ok():
    assert gross_ok(1.5, 2.0)[0] is True
    ok, why = gross_ok(2.5, 2.0)
    assert ok is False and "gross" in why


def test_evaluate_normal_can_trade():
    d = evaluate(BotConfig(), [100, 101, 102], auto_halted=False)
    assert d.can_trade and not d.halted and not d.triggered_auto_halt


def test_evaluate_paused_blocks_trading_not_halted():
    d = evaluate(BotConfig(paused=True), [100], auto_halted=False)
    assert not d.can_trade and not d.halted
    assert any("paused" in r for r in d.reasons)


def test_evaluate_manual_halt():
    d = evaluate(BotConfig(halt=True), [100], auto_halted=False)
    assert d.halted and not d.can_trade


def test_evaluate_auto_halt_triggers_on_breach():
    cfg = BotConfig(max_drawdown_pct=0.10)
    d = evaluate(cfg, [100, 85], auto_halted=False)         # -15% breaches -10%
    assert d.triggered_auto_halt and d.halted and not d.can_trade


def test_evaluate_auto_halt_latches_after_recovery():
    cfg = BotConfig(max_drawdown_pct=0.10)
    d = evaluate(cfg, [100, 101], auto_halted=True)         # recovered, but latch persists
    assert d.halted and not d.can_trade and not d.triggered_auto_halt
