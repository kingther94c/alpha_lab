"""State/control bridge between the UI and a running bot.

The UI reads a bot's state from its SQLite store (``core.store.Store``) and controls the process
via the package CLI — so the UI never imports the trading code (ccxt / alpha_lab) directly. Every
function takes a ``bot`` name (default ``config.DEFAULT_BOT``) so the cockpit can manage several
bots side by side. Defaults come from the bot's YAML via ``registry.default_config``.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys

import pandas as pd

from quant_bot_manager.core import config, registry
from quant_bot_manager.core.store import Store

PYEXE = sys.executable
_ENV = {**os.environ, "PYTHONPATH": str(config.ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def _store(bot: str) -> Store:
    return Store(bot)


# -- reads ----------------------------------------------------------------
def read_equity(bot: str = config.DEFAULT_BOT) -> pd.DataFrame:
    return _store(bot).read_equity_df()


def read_rebalances(bot: str = config.DEFAULT_BOT) -> pd.DataFrame:
    return _store(bot).read_rebalances_df()


def read_status(bot: str = config.DEFAULT_BOT) -> dict:
    return _store(bot).read_status()


def read_config(bot: str = config.DEFAULT_BOT) -> dict:
    """Bot's effective config: YAML defaults overlaid with any live store overrides."""
    cfg = registry.default_config(bot).to_dict()
    cfg.update(_store(bot).read_config())
    return cfg


def write_config(updates: dict, bot: str = config.DEFAULT_BOT) -> None:
    _store(bot).write_config(updates)


def is_running(bot: str = config.DEFAULT_BOT) -> bool:
    s = read_status(bot)
    hb = s.get("last_heartbeat")
    if not hb:
        return False
    try:
        age = (dt.datetime.now(dt.UTC) - dt.datetime.fromisoformat(hb)).total_seconds()
    except Exception:  # noqa: BLE001
        return False
    interval = float((s.get("config") or {}).get("interval_min", 15))
    return age < max(interval * 60 * 2.5, 180)


# -- process / control ----------------------------------------------------
def _cli(*cmd):
    return [PYEXE, "-m", "quant_bot_manager.cli", *cmd]


def start_bot(cfg: dict, bot: str = config.DEFAULT_BOT) -> str:
    if is_running(bot):                                 # single-flight: never run two loops on one bot/account
        return f"already running (pid {read_status(bot).get('pid')})"
    write_config({**cfg, "paused": False}, bot)
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP   # DETACHED_PROCESS
    # Route the detached child's stdout+stderr to the bot's run.log so a crash-on-launch (e.g. a
    # missing BINANCE_DEMO_KEY, an import/ccxt failure) leaves a trace instead of vanishing into
    # DEVNULL. Unbuffered append; the child inherits its own handle, so the parent dropping this
    # reference after Popen returns does not close the child's stream.
    log = open(config.paths(bot)["log"], "ab", buffering=0)
    subprocess.Popen(
        _cli("run", "--bot", bot, "--mode", "demo", "--capital", str(cfg["capital"]),
             "--method", cfg["method"], "--max-gross", str(cfg["max_gross"]),
             "--interval-min", str(cfg["interval_min"])),
        cwd=str(config.ROOT), env=_ENV, creationflags=flags,
        stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    return "start requested"


def stop_bot(bot: str = config.DEFAULT_BOT) -> str:
    pid = read_status(bot).get("pid")
    if not pid:
        return "no running pid in status"
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        import signal
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception as e:   # noqa: BLE001
            return f"kill failed: {e}"
    return f"stopped pid {pid}"


def set_paused(paused: bool, bot: str = config.DEFAULT_BOT) -> None:
    write_config({"paused": bool(paused)}, bot)


def set_halt(halt: bool, bot: str = config.DEFAULT_BOT) -> None:
    """Hard manual kill-switch — the running bot stops placing orders next cycle (keeps marking)."""
    write_config({"halt": bool(halt)}, bot)


def clear_auto_halt(bot: str = config.DEFAULT_BOT) -> None:
    """Release a latched drawdown auto-halt so trading can resume."""
    _store(bot).set_auto_halted(False)


def manual_rebalance(capital: float, method: str, bot: str = config.DEFAULT_BOT) -> str:
    r = subprocess.run(
        _cli("rebalance", "--bot", bot, "--mode", "demo", "--capital", str(capital), "--method", method),
        cwd=str(config.ROOT), env=_ENV, capture_output=True, text=True, timeout=240)
    return (r.stdout or "") + (r.stderr or "")
