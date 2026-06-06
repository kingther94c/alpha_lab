"""Plugin contracts for the execution leg (Option C — plugin framework).

Three seams let a bot be assembled from interchangeable parts named in its YAML:

  * Strategy — research alpha adapted to a live target-weight call. Shape: ``(method) -> (targets, asof, last_px)``.
  * Feed     — the market-data source a strategy consumes (live exchange API today; a replay feed
               could back the *same* strategy for paper/backtest parity). Shape: ``(lookback_days) -> bundle``.
  * Broker   — an execution venue (see ``brokers.base.Broker``), re-exported here as the third seam.

These are structural ``Protocol``s: a plain function or class satisfies one by *shape*, no
inheritance required (``latest_targets`` is a Strategy, ``build_live_bookdata`` is a Feed). The
registry (``core.registry``) maps the names in a bot's YAML to concrete implementations and wires
them together.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from quant_bot_manager.brokers.base import Broker  # re-export: Broker is the third plugin seam


@runtime_checkable
class Strategy(Protocol):
    def __call__(self, method: str = ...) -> tuple[pd.Series, Any, dict[str, float]]:
        """Return ``(target_weights, asof, last_px)`` for execution now."""
        ...


@runtime_checkable
class Feed(Protocol):
    def __call__(self, lookback_days: int = ...) -> Any:
        """Return a data bundle a strategy consumes (e.g. ``crypto_book.BookData``)."""
        ...


__all__ = ["Strategy", "Feed", "Broker"]
