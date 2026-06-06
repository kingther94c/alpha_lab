"""State/control bridge between the UI and a running bot.

The UI talks to the bot through files (``config.json`` it writes, ``status.json`` the bot writes,
the equity/rebalance CSVs) plus process control via the package CLI — so the UI never imports the
trading code directly. Paths come from ``core.config`` (per-bot run dir).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

import pandas as pd

from quant_bot_manager.core import config

BOT = config.DEFAULT_BOT
P = config.paths(BOT)
PYEXE = sys.executable
_ENV = {**os.environ, "PYTHONPATH": str(config.ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def read_equity() -> pd.DataFrame:
    if not P["equity"].exists():
        return pd.DataFrame(columns=["ts", "total_equity", "fut_equity", "spot_equity"])
    df = pd.read_csv(P["equity"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def read_rebalances() -> pd.DataFrame:
    return pd.read_csv(P["rebalance"]) if P["rebalance"].exists() else pd.DataFrame()


def read_status() -> dict:
    if not P["status"].exists():
        return {}
    try:
        return json.loads(P["status"].read_text())
    except Exception:
        return {}


def read_config() -> dict:
    cfg = dict(config.DEFAULT_CONFIG)
    if P["config"].exists():
        try:
            cfg.update(json.loads(P["config"].read_text()))
        except Exception:
            pass
    return cfg


def write_config(updates: dict) -> None:
    cfg = read_config()
    cfg.update(updates)
    P["config"].write_text(json.dumps(cfg, indent=2))


def is_running() -> bool:
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


def _cli(*cmd):
    return [PYEXE, "-m", "quant_bot_manager.cli", *cmd]


def start_bot(cfg: dict) -> str:
    write_config({**cfg, "paused": False})
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP   # DETACHED_PROCESS
    subprocess.Popen(
        _cli("run", "--bot", BOT, "--mode", "demo", "--capital", str(cfg["capital"]),
             "--method", cfg["method"], "--max-gross", str(cfg["max_gross"]),
             "--interval-min", str(cfg["interval_min"])),
        cwd=str(config.ROOT), env=_ENV, creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
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
        _cli("rebalance", "--bot", BOT, "--mode", "demo", "--capital", str(capital), "--method", method),
        cwd=str(config.ROOT), env=_ENV, capture_output=True, text=True, timeout=240)
    return (r.stdout or "") + (r.stderr or "")
