"""Typed, validated bot configuration (Option A — config schema).

A ``BotConfig`` is the single validated description of *how* a bot trades: capital, weighting
method, gross cap, cadence, and the risk limits that drive the kill-switch. Validation and type
coercion happen at construction, so a bad config (negative capital, absurd leverage, a typo'd
method) fails fast here instead of reaching the exchange.

The same object is the contract everywhere: YAML ``default_config`` -> ``BotConfig`` -> the
runner's per-cycle settings -> the UI control form. ``DEFAULTS`` is the canonical default dict
(re-exported as ``config.DEFAULT_CONFIG`` for back-compat).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

METHODS = ("equal_capital", "risk_budget")


@dataclass
class BotConfig:
    capital: float = 10_000.0
    method: str = "equal_capital"
    max_gross: float = 2.0          # hard gross-exposure cap (x capital); a rebalance above it is skipped
    interval_min: float = 15.0      # mark-to-market / rebalance cadence
    max_drawdown_pct: float = 0.20  # kill-switch: latch-halt if equity falls this far below its peak
    paused: bool = False            # soft pause: keep marking-to-market, place no orders
    halt: bool = False              # hard manual kill-switch: place no orders

    def __post_init__(self) -> None:
        # coerce first (CLI / JSON / YAML may hand us ints or strings), then validate
        self.capital = float(self.capital)
        self.method = str(self.method)
        self.max_gross = float(self.max_gross)
        self.interval_min = float(self.interval_min)
        self.max_drawdown_pct = float(self.max_drawdown_pct)
        self.paused = bool(self.paused)
        self.halt = bool(self.halt)
        if self.capital <= 0:
            raise ValueError(f"capital must be > 0, got {self.capital}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not 0 < self.max_gross <= 5:
            raise ValueError(f"max_gross must be in (0, 5], got {self.max_gross}")
        if self.interval_min < 1:
            raise ValueError(f"interval_min must be >= 1, got {self.interval_min}")
        if not 0 < self.max_drawdown_pct <= 1:
            raise ValueError(f"max_drawdown_pct must be in (0, 1], got {self.max_drawdown_pct}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> BotConfig:
        """Build from a dict, silently dropping unknown keys (forward/backward tolerant)."""
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def merge(self, updates: dict | None) -> BotConfig:
        """A new, re-validated config with ``updates`` applied (unknown keys ignored)."""
        return BotConfig.from_dict({**self.to_dict(), **(updates or {})})


DEFAULTS: dict = BotConfig().to_dict()
