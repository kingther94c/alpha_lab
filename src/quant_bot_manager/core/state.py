"""State/control bridge between the UI and the running mock-trading bot.

The UI talks to the bot purely through files (``bot_config.json`` it writes, ``bot_status.json`` the
bot writes) plus process control — so the UI never imports the trading code. This keeps the cockpit
decoupled from the runner/broker (which still live in notebooks/ pending the execution refactor).
"""
from __future__ import annotations
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]                      # .../src/quant_bot_manager/core/state.py -> repo root
DATA = ROOT / "data" / "results" / "crypto_v3_multi"
EQLOG, REBLOG = DATA / "mock_equity_log.csv", DATA / "mock_rebalance_log.csv"
CONFIG, STATUS = DATA / "bot_config.json", DATA / "bot_status.json"
LOOP = ROOT / "notebooks" / "90_crypto_intraday" / "mock_trader_loop.py"
ONESHOT = ROOT / "notebooks" / "90_crypto_intraday" / "binance_paper_trade.py"
PYEXE = sys.executable

DEFAULT_CONFIG = {"capital": 10000.0, "method": "equal_capital", "max_gross": 2.0,
                  "interval_min": 15.0, "paused": False}


def read_equity() -> pd.DataFrame:
    if not EQLOG.exists():
        return pd.DataFrame(columns=["ts", "total_equity", "fut_equity", "spot_equity"])
    df = pd.read_csv(EQLOG)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def read_rebalances() -> pd.DataFrame:
    return pd.read_csv(REBLOG) if REBLOG.exists() else pd.DataFrame()


def read_status() -> dict:
    if not STATUS.exists():
        return {}
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def read_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        try:
            cfg.update(json.loads(CONFIG.read_text()))
        except Exception:
            pass
    return cfg


def write_config(updates: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    cfg = read_config()
    cfg.update(updates)
    CONFIG.write_text(json.dumps(cfg, indent=2))


def is_running() -> bool:
    """True if the bot wrote a heartbeat recently (within ~2.5 intervals)."""
    s = read_status()
    hb = s.get("last_heartbeat")
    if not hb:
        return False
    try:
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(hb)).total_seconds()
    except Exception:
        return False
    interval = float((s.get("config") or {}).get("interval_min", 15))
    return age < max(interval * 60 * 2.5, 180)


def start_bot(cfg: dict) -> str:
    write_config({**cfg, "paused": False})
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP   # DETACHED_PROCESS: survive the UI process
    subprocess.Popen(
        [PYEXE, str(LOOP), "--interval-min", str(cfg["interval_min"]), "--capital", str(cfg["capital"]),
         "--method", cfg["method"], "--max-gross", str(cfg["max_gross"])],
        cwd=str(ROOT), creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    return "start requested"


def stop_bot() -> str:
    pid = read_status().get("pid")
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


def set_paused(paused: bool) -> None:
    write_config({"paused": bool(paused)})


def manual_rebalance(capital: float, method: str) -> str:
    r = subprocess.run(
        [PYEXE, str(ONESHOT), "--mode", "demo", "--method", method, "--capital", str(capital)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=240)
    return (r.stdout or "") + (r.stderr or "")
