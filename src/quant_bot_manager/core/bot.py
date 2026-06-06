"""Bot = a strategy (-> target weights) bound to a broker (execution venue), plus its default config.

This is just the value object. Assembly from a YAML definition (which strategy / feed / broker /
defaults) lives in ``core.registry`` so this module stays import-light (no ccxt / alpha_lab pulled
in until a bot is actually built).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant_bot_manager.brokers.base import Broker
from quant_bot_manager.core.schema import BotConfig


@dataclass
class Bot:
    name: str
    strategy: Callable                  # method:str -> (targets: Series, asof, last_px: dict)
    broker: Broker
    default_config: BotConfig | None = None
