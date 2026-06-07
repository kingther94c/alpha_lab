"""One-click mock stack — start/stop/inspect the demo bot + the Streamlit cockpit.

    python scripts/demo.py up       [--bot p7_crypto_book] [--port 8501]
    python scripts/demo.py down     [--bot ...] [--port 8501]
    python scripts/demo.py status   [--bot ...] [--port 8501]

`up` starts the bot on Binance **demo** (mock funds) as a detached process and launches the
Streamlit cockpit; both survive this script exiting. `down` stops them. Use the py313 interpreter
(`D:\\conda\\envs\\py313\\python.exe scripts\\demo.py up`) or activate the env first. Equivalent
make targets: `make demo-up` / `make demo-down` / `make demo-status`.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_bot_manager.core import config, state  # noqa: E402  (needs src on path first)

UI_APP = ROOT / "src" / "quant_bot_manager" / "ui" / "app.py"


def _pidfile(port: int) -> Path:
    return config.RESULTS_DIR / "bots" / f".ui_{port}.pid"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _detached_kwargs() -> dict:
    kw: dict = dict(cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                    env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")})
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP   # DETACHED_PROCESS
    else:
        kw["start_new_session"] = True
    return kw


def _start_ui(port: int) -> str:
    if _port_open(port):
        return f"UI: already serving on :{port} (left as-is)"
    p = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(UI_APP),
         "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        **_detached_kwargs())
    pf = _pidfile(port)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(str(p.pid))
    for _ in range(20):
        if _port_open(port):
            break
        time.sleep(1)
    return f"UI: started on http://localhost:{port} (pid {p.pid})"


def _stop_ui(port: int) -> str:
    pf = _pidfile(port)
    if not pf.exists():
        return f"UI: no tracked pid for :{port} (manual instances are not stopped)"
    pid = pf.read_text().strip()
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError as e:
            return f"UI: kill failed ({e})"
    pf.unlink(missing_ok=True)
    return f"UI: stopped (pid {pid})"


def cmd_up(a) -> None:
    config.load_env()
    if state.is_running(a.bot):
        print(f"bot {a.bot}: already running (pid {state.read_status(a.bot).get('pid')})")
    else:
        print(f"bot {a.bot}: {state.start_bot(state.read_config(a.bot), a.bot)} (mode demo)")
    print(_start_ui(a.port))
    print(f"\ncockpit -> http://localhost:{a.port}   ·   stop with: python scripts/demo.py down")


def cmd_down(a) -> None:
    print(f"bot {a.bot}: {state.stop_bot(a.bot)}")
    print(_stop_ui(a.port))


def cmd_status(a) -> None:
    st = state.read_status(a.bot)
    running = state.is_running(a.bot)
    eq = st.get("equity")
    print(f"bot {a.bot}: {'RUNNING' if running else 'stopped'} | pid {st.get('pid')} | "
          f"cycle {st.get('cycle')} | equity {f'${eq:,.0f}' if eq else '-'} | "
          f"halted {st.get('halted')} | last_rebal {st.get('last_rebal_date')}")
    print(f"UI :{a.port}: {'up' if _port_open(a.port) else 'down'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="demo", description="One-click mock stack (bot + cockpit).")
    ap.add_argument("action", choices=["up", "down", "status"])
    ap.add_argument("--bot", default=config.DEFAULT_BOT)
    ap.add_argument("--port", type=int, default=8501)
    a = ap.parse_args()
    {"up": cmd_up, "down": cmd_down, "status": cmd_status}[a.action](a)


if __name__ == "__main__":
    main()
