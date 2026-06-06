"""Bot registry + plugin protocols + pure plan building (execution leg, Options A4/C)."""
from __future__ import annotations

import pandas as pd
import pytest

from quant_bot_manager.brokers.base import Broker
from quant_bot_manager.brokers.binance import BinanceBroker
from quant_bot_manager.core import registry
from quant_bot_manager.core.bot import Bot
from quant_bot_manager.core.protocols import Broker as PBroker
from quant_bot_manager.core.protocols import Feed, Strategy
from quant_bot_manager.core.schema import BotConfig


def test_list_and_default_config():
    assert "p7_crypto_book" in registry.list_bots()
    c = registry.default_config("p7_crypto_book")
    assert isinstance(c, BotConfig) and c.capital > 0


def test_get_bot_builds_without_connecting():
    bot = registry.get_bot("p7_crypto_book", "demo")
    assert isinstance(bot, Bot)
    assert isinstance(bot.broker, Broker)
    assert callable(bot.strategy)
    assert isinstance(bot.default_config, BotConfig)


def test_unknown_bot_raises():
    with pytest.raises(ValueError):
        registry.get_bot("does_not_exist")


def test_protocol_conformance():
    from quant_bot_manager.strategies import p7_crypto_book
    assert isinstance(BinanceBroker("demo"), PBroker)        # Broker ABC
    assert isinstance(p7_crypto_book.latest_targets, Strategy)
    assert isinstance(p7_crypto_book.build_live_bookdata, Feed)


def test_build_plan_is_pure():
    b = BinanceBroker("demo")
    tgt = pd.Series({"BTC.p": 0.5, "ETH.p": -0.3, "BTC.s": 0.2})
    px = {"BTC.p": 60000.0, "ETH.p": 1500.0, "BTC.s": 60000.0}
    plan = b.build_plan(tgt, 10000.0, px)
    assert set(plan["leg"]) == {"BTC.p", "ETH.p", "BTC.s"}
    btc = plan[plan.leg == "BTC.p"].iloc[0]
    assert btc.side == "BUY" and abs(btc.notional_usdt - 5000.0) < 1e-6
    assert plan[plan.leg == "ETH.p"].iloc[0].side == "SELL"
