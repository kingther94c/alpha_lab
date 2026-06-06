"""BotConfig schema — validation, coercion, merge (execution leg, Option A)."""
from __future__ import annotations

import pytest

from quant_bot_manager.core.schema import DEFAULTS, BotConfig


def test_defaults_valid_and_exported():
    c = BotConfig()
    assert c.capital > 0
    assert c.method in ("equal_capital", "risk_budget")
    assert DEFAULTS["max_gross"] == c.max_gross


def test_coercion_of_str_and_int():
    c = BotConfig.from_dict({"capital": "5000", "max_gross": 1, "interval_min": 10, "paused": 1})
    assert isinstance(c.capital, float) and c.capital == 5000.0
    assert c.max_gross == 1.0
    assert c.paused is True


def test_from_dict_ignores_unknown_keys():
    c = BotConfig.from_dict({"capital": 2000, "bogus": 42})
    assert c.capital == 2000.0


def test_merge_revalidates():
    c = BotConfig().merge({"max_gross": 3.0})
    assert c.max_gross == 3.0
    with pytest.raises(ValueError):
        BotConfig().merge({"max_gross": 99})


def test_roundtrip_to_from_dict():
    c = BotConfig(capital=1234.0, max_gross=1.5, max_drawdown_pct=0.1)
    assert BotConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()


@pytest.mark.parametrize("bad", [
    {"capital": -1}, {"capital": 0}, {"method": "nope"},
    {"max_gross": 0}, {"max_gross": 6}, {"interval_min": 0},
    {"max_drawdown_pct": 0}, {"max_drawdown_pct": 1.5},
])
def test_invalid_raises(bad):
    with pytest.raises(ValueError):
        BotConfig.from_dict({**DEFAULTS, **bad})
