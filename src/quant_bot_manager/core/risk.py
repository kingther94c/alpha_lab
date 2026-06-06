"""Risk gates and the kill-switch (Option A).

Pure decision functions the runner consults every cycle — no I/O, easy to test. Layers:

  * gross cap   — never place a rebalance whose gross exposure exceeds ``max_gross`` x capital.
  * drawdown    — auto-halt (kill-switch) when mark-to-market equity falls ``max_drawdown_pct``
                  below its running peak. The auto-halt *latches*: once tripped it must be
                  cleared manually (UI / CLI), so a bad drawdown can't silently resume trading.

Plus two manual switches carried in the config: ``paused`` (keep marking-to-market, stop trading)
and ``halt`` (hard stop). ``evaluate`` folds everything into one ``RiskDecision`` the runner acts
on; ``triggered_auto_halt`` tells it when to *persist* a freshly latched halt.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskDecision:
    can_trade: bool                       # ok to place orders this cycle?
    halted: bool                          # hard stop in effect (manual halt or latched auto-halt)
    triggered_auto_halt: bool             # auto-halt tripped *this* evaluation -> caller should latch it
    drawdown: float                       # current peak-to-now drawdown (<= 0)
    reasons: list[str] = field(default_factory=list)


def drawdown(equity: list[float]) -> float:
    """Worst peak-to-current drawdown of an equity path, as a fraction <= 0 (0.0 if empty)."""
    peak = None
    worst = 0.0
    for e in equity:
        if e is None:
            continue
        peak = e if peak is None else max(peak, e)
        if peak and peak > 0:
            worst = min(worst, e / peak - 1.0)
    return worst


def gross_ok(gross: float, max_gross: float) -> tuple[bool, str]:
    """Gate a planned rebalance on its gross exposure."""
    if gross <= max_gross:
        return True, ""
    return False, f"gross {gross:.2f}x > cap {max_gross:.2f}x"


def evaluate(cfg, equity_history: list[float], *, auto_halted: bool) -> RiskDecision:
    """Fold manual switches + drawdown kill-switch into a trade/no-trade decision.

    ``cfg`` is a BotConfig; ``auto_halted`` is the persisted latch from the store.
    """
    reasons: list[str] = []
    dd = drawdown(equity_history)
    halted = bool(cfg.halt) or bool(auto_halted)
    triggered = False
    if cfg.halt:
        reasons.append("manual halt (config.halt)")
    if auto_halted:
        reasons.append("latched auto-halt (drawdown breach) — clear it to resume")
    if not halted and dd <= -abs(cfg.max_drawdown_pct):
        halted = True
        triggered = True
        reasons.append(f"drawdown {dd:.1%} breached -{cfg.max_drawdown_pct:.0%} -> auto-halt")
    if cfg.paused and not halted:
        reasons.append("paused")
    can_trade = not halted and not cfg.paused
    return RiskDecision(can_trade=can_trade, halted=halted, triggered_auto_halt=triggered,
                        drawdown=dd, reasons=reasons)
