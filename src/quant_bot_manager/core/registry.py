"""Bot registry — assemble bots from YAML definitions and named plugins (Option C).

A bot YAML (``configs/bots/<name>.yaml``) names a strategy, feed, broker, and default config::

    name: p7_crypto_book
    strategy: p7_crypto_book
    feed: live_binance
    broker: binance
    default_config: {capital: 10000, method: equal_capital, max_gross: 2.0, ...}

``STRATEGIES`` / ``FEEDS`` / ``BROKERS`` (built lazily so importing this module stays cheap — no
ccxt / alpha_lab until a bot is actually constructed) map those names to implementations.
``get_bot`` reads the YAML, looks up the parts, binds the named feed into the strategy, validates
the default config, and returns a ready ``Bot``. ``list_bots`` enumerates the YAML defs so the
manager / UI can run several bots.
"""
from __future__ import annotations

import functools

import yaml

from quant_bot_manager.core import config
from quant_bot_manager.core.bot import Bot
from quant_bot_manager.core.schema import BotConfig

BOTS_DIR = config.ROOT / "configs" / "bots"


def _strategies() -> dict:
    from quant_bot_manager.strategies import p7_crypto_book
    return {"p7_crypto_book": p7_crypto_book.latest_targets}


def _feeds() -> dict:
    from quant_bot_manager.strategies import p7_crypto_book
    return {"live_binance": p7_crypto_book.build_live_bookdata}


def _brokers() -> dict:
    from quant_bot_manager.brokers.binance import BinanceBroker
    return {"binance": BinanceBroker}


def list_bots() -> list[str]:
    """Names of all defined bots (one per ``configs/bots/*.yaml``)."""
    return sorted(p.stem for p in BOTS_DIR.glob("*.yaml")) if BOTS_DIR.exists() else []


def load_def(name: str) -> dict:
    f = BOTS_DIR / f"{name}.yaml"
    if not f.exists():
        raise ValueError(f"unknown bot {name!r}; known: {list_bots()}")
    return yaml.safe_load(f.read_text()) or {}


def default_config(name: str) -> BotConfig:
    """The validated default BotConfig for a bot (from its YAML ``default_config``)."""
    return BotConfig.from_dict(load_def(name).get("default_config"))


def get_bot(name: str = config.DEFAULT_BOT, mode: str = "demo") -> Bot:
    """Build a ready Bot from its YAML def: named strategy + feed (bound) + broker(mode)."""
    d = load_def(name)
    strat_fn = _strategies().get(d.get("strategy"))
    feed_fn = _feeds().get(d.get("feed", "live_binance"))
    broker_cls = _brokers().get(d.get("broker"))
    if strat_fn is None:
        raise ValueError(f"bot {name!r}: unknown strategy {d.get('strategy')!r}; known: {list(_strategies())}")
    if feed_fn is None:
        raise ValueError(f"bot {name!r}: unknown feed {d.get('feed')!r}; known: {list(_feeds())}")
    if broker_cls is None:
        raise ValueError(f"bot {name!r}: unknown broker {d.get('broker')!r}; known: {list(_brokers())}")
    strategy = functools.partial(strat_fn, feed=feed_fn)   # wire the Feed plugin into the Strategy
    return Bot(name=d.get("name", name), strategy=strategy, broker=broker_cls(mode),
               default_config=default_config(name))
