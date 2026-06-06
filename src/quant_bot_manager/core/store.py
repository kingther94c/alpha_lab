"""SQLite state store for a bot (Option A — durable, queryable state).

Replaces the per-bot flat files (``equity_log.csv`` / ``rebalance_log.csv`` / ``state.json`` /
``status.json``) with a single ``bot.db`` per bot. SQLite gives atomic writes and concurrent
reads (WAL mode), so the runner can write while the Streamlit cockpit reads — separate processes,
no corruption, no half-written JSON. Schema:

  equity      (ts, total, fut, spot)                              -- mark-to-market time series
  rebalances  (ts, signal_asof, gross, status, n_orders, orders)  -- executed rebalances
  kv          (key, value)                                        -- JSON blobs: status, config,
                                                                      last_rebal_date, auto_halted

On first open it imports any legacy CSV/JSON left in the run dir, so existing demo history
survives the migration. Pass an explicit ``path`` (tests) to get an isolated store with no
legacy import.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from quant_bot_manager.core import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL, total REAL, fut REAL, spot REAL
);
CREATE TABLE IF NOT EXISTS rebalances (
    ts TEXT NOT NULL, signal_asof TEXT, gross REAL, status TEXT, n_orders INTEGER, orders TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Store:
    """Per-bot SQLite state. Cheap to construct; connections are opened per call (process-safe)."""

    def __init__(self, bot: str, path: str | Path | None = None):
        self.bot = bot
        self._custom = path is not None
        self.path = Path(path) if path else config.paths(bot)["db"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._cx() as c:
            c.executescript(_SCHEMA)
        if not self._custom:
            self._import_legacy()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=5000")
        return c

    @contextmanager
    def _cx(self):
        c = self._conn()
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # -- key/value (JSON blobs) --------------------------------------------
    def get_kv(self, key: str, default=None):
        with self._cx() as c:
            row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_kv(self, key: str, value) -> None:
        with self._cx() as c:
            c.execute("INSERT INTO kv(key, value) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value, default=str)))

    # -- equity ------------------------------------------------------------
    def append_equity(self, ts: str, total, fut, spot) -> None:
        with self._cx() as c:
            c.execute("INSERT INTO equity(ts, total, fut, spot) VALUES (?,?,?,?)",
                      (ts, _f(total), _f(fut), _f(spot)))

    def read_equity_df(self) -> pd.DataFrame:
        with self._cx() as c:
            df = pd.read_sql_query(
                "SELECT ts, total AS total_equity, fut AS fut_equity, spot AS spot_equity "
                "FROM equity ORDER BY rowid", c)
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    def all_equity_totals(self) -> list[float]:
        with self._cx() as c:
            rows = c.execute("SELECT total FROM equity ORDER BY rowid").fetchall()
        return [float(r[0]) for r in rows if r[0] is not None]

    # -- rebalances --------------------------------------------------------
    def append_rebalance(self, ts: str, signal_asof: str, gross: float, status: str,
                         n_orders: int, orders: str) -> None:
        with self._cx() as c:
            c.execute("INSERT INTO rebalances(ts, signal_asof, gross, status, n_orders, orders) "
                      "VALUES (?,?,?,?,?,?)",
                      (ts, str(signal_asof), _f(gross), status, int(n_orders), orders))

    def read_rebalances_df(self) -> pd.DataFrame:
        with self._cx() as c:
            return pd.read_sql_query(
                "SELECT ts, signal_asof, gross, status, n_orders, orders "
                "FROM rebalances ORDER BY rowid", c)

    # -- named state (thin wrappers over kv) -------------------------------
    def read_status(self) -> dict:
        return self.get_kv("status") or {}

    def write_status(self, status: dict) -> None:
        self.set_kv("status", status)

    def read_config(self) -> dict:
        return self.get_kv("config") or {}

    def write_config(self, updates: dict) -> dict:
        cur = self.get_kv("config") or {}
        cur.update(updates or {})
        self.set_kv("config", cur)
        return cur

    def get_last_rebal_date(self):
        return self.get_kv("last_rebal_date")

    def set_last_rebal_date(self, day: str) -> None:
        self.set_kv("last_rebal_date", day)

    def get_auto_halted(self) -> bool:
        return bool(self.get_kv("auto_halted"))

    def set_auto_halted(self, value: bool) -> None:
        self.set_kv("auto_halted", bool(value))

    # -- one-time migration of pre-SQLite flat files -----------------------
    def _import_legacy(self) -> None:
        if self.get_kv("_legacy_imported"):
            return
        self.set_kv("_legacy_imported", True)   # latch first: a partial import must never loop
        p = config.paths(self.bot)
        try:
            if p["equity"].exists():
                df = pd.read_csv(p["equity"])
                with self._cx() as c:
                    c.executemany(
                        "INSERT INTO equity(ts, total, fut, spot) VALUES (?,?,?,?)",
                        [(str(r.ts), _f(r.total_equity), _f(r.fut_equity), _f(r.spot_equity))
                         for r in df.itertuples(index=False)])
            if p["rebalance"].exists():
                df = pd.read_csv(p["rebalance"])
                with self._cx() as c:
                    c.executemany(
                        "INSERT INTO rebalances(ts, signal_asof, gross, status, n_orders, orders) "
                        "VALUES (?,?,?,?,?,?)",
                        [(str(r.ts), str(getattr(r, "signal_asof", "")), _f(getattr(r, "gross", 0)),
                          str(getattr(r, "status", "")), int(getattr(r, "n_orders", 0) or 0),
                          str(getattr(r, "orders", ""))) for r in df.itertuples(index=False)])
            if p["state"].exists():
                st = json.loads(p["state"].read_text())
                if st.get("last_rebal_date"):
                    self.set_last_rebal_date(st["last_rebal_date"])
            if p["config"].exists():
                self.set_kv("config", json.loads(p["config"].read_text()))
            if p["status"].exists():
                self.set_kv("status", json.loads(p["status"].read_text()))
        except Exception as e:  # noqa: BLE001 — migration is best-effort; never block startup
            print(f"[store] legacy import warning: {type(e).__name__}: {e}")
