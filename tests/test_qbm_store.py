"""SQLite state store round-trips (execution leg, Option A). Uses isolated tmp paths."""
from __future__ import annotations

from quant_bot_manager.core.store import Store


def test_equity_roundtrip(tmp_path):
    s = Store("t", path=tmp_path / "bot.db")
    s.append_equity("2026-01-01T00:00:00+00:00", 100.0, 60.0, 40.0)
    s.append_equity("2026-01-01T00:15:00+00:00", 110.0, 65.0, 45.0)
    df = s.read_equity_df()
    assert list(df.columns) == ["ts", "total_equity", "fut_equity", "spot_equity"]
    assert len(df) == 2
    assert s.all_equity_totals() == [100.0, 110.0]


def test_kv_config_and_flags(tmp_path):
    s = Store("t", path=tmp_path / "b.db")
    assert s.read_config() == {}
    s.write_config({"capital": 5000})
    s.write_config({"max_gross": 1.5})
    assert s.read_config() == {"capital": 5000, "max_gross": 1.5}     # merge, not overwrite
    s.set_auto_halted(True)
    assert s.get_auto_halted() is True
    s.set_last_rebal_date("2026-01-02")
    assert s.get_last_rebal_date() == "2026-01-02"


def test_status_and_rebalances(tmp_path):
    s = Store("t", path=tmp_path / "b.db")
    s.write_status({"equity": 123, "cycle": 3})
    assert s.read_status()["cycle"] == 3
    s.append_rebalance("2026-01-01T00:00:00+00:00", "2025-12-31", 1.8, "ok", 4, "perp buy X 1")
    r = s.read_rebalances_df()
    assert len(r) == 1
    assert r.iloc[0]["status"] == "ok" and int(r.iloc[0]["n_orders"]) == 4


def test_custom_path_skips_legacy_import(tmp_path):
    # an explicit path must NOT pull in the real bot's legacy CSV/JSON history
    s = Store("p7_crypto_book", path=tmp_path / "iso.db")
    assert s.all_equity_totals() == []
    assert s.read_config() == {}


def test_empty_reads(tmp_path):
    s = Store("t", path=tmp_path / "e.db")
    assert s.all_equity_totals() == []
    assert len(s.read_equity_df()) == 0
    assert len(s.read_rebalances_df()) == 0
    assert s.read_status() == {}
