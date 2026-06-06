"""Broker interface — the execution-venue contract a Bot trades through.

A Broker turns a *plan* (per-leg target orders) into fills on a venue and reports
positions / equity. Keep concrete venues (Binance, ...) behind this so the runner and
UI never depend on a specific exchange SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Broker(ABC):
    name: str

    @abstractmethod
    def connect(self) -> None:
        """Authenticate / select environment (demo / testnet / live). Raises if keys missing."""

    @abstractmethod
    def public_prices(self, legs: list[str]) -> dict[str, float]:
        """Current mid prices for the given legs (no auth needed), for notional sizing."""

    @abstractmethod
    def build_plan(self, targets: pd.Series, capital: float, prices: dict[str, float]) -> pd.DataFrame:
        """Translate target weights -> an order plan (columns: leg, venue, symbol, side, price, notional_usdt, qty)."""

    @abstractmethod
    def rebalance_to_target(self, plan: pd.DataFrame, *, from_flat: bool = False) -> list[tuple]:
        """Market-order current positions toward the plan. Returns (venue, side, symbol, amount) submitted."""

    @abstractmethod
    def mark_to_market(self) -> tuple[float, float, float]:
        """(total_equity, futures_equity, spot_equity) in USDT."""

    @abstractmethod
    def positions_snapshot(self) -> dict:
        """Serializable snapshot of current positions/balances for monitoring."""
