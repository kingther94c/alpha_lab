"""Bot = a strategy (→ target weights) bound to a broker (execution venue).

The registry lets the manager/UI run several bots by name. Today there is one: the P7 crypto book.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant_bot_manager.brokers.base import Broker
from quant_bot_manager.brokers.binance import BinanceBroker
from quant_bot_manager.strategies import p7_crypto_book

BOTS = ["p7_crypto_book"]


@dataclass
class Bot:
    name: str
    strategy: Callable          # method:str -> (targets: Series, asof, last_px: dict)
    broker: Broker


def get_bot(name: str = "p7_crypto_book", mode: str = "demo") -> Bot:
    if name in ("p7", "p7_crypto_book"):
        return Bot("p7_crypto_book", p7_crypto_book.latest_targets, BinanceBroker(mode))
    raise ValueError(f"unknown bot {name!r}; known: {BOTS}")
