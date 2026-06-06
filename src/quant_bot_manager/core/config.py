"""Shared paths, per-bot run directories, and defaults for the execution leg.

Each bot writes its runtime artifacts under ``data/results/bots/<bot>/`` so the manager can
run and monitor several bots side by side. The only coupling to the research leg (``alpha_lab``)
is that a *strategy* adapts an alpha_lab function into target weights — see ``strategies/``.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]            # .../src/quant_bot_manager/core/config.py -> repo root
RESULTS_DIR = ROOT / "data" / "results"
ENV_FILE = ROOT / ".env"

DEFAULT_BOT = "p7_crypto_book"
DEFAULT_CONFIG = {"capital": 10000.0, "method": "equal_capital", "max_gross": 2.0,
                  "interval_min": 15.0, "paused": False}
EPS = 1e-4


def run_dir(bot: str = DEFAULT_BOT) -> Path:
    d = RESULTS_DIR / "bots" / bot
    d.mkdir(parents=True, exist_ok=True)
    return d


def paths(bot: str = DEFAULT_BOT) -> dict:
    """Canonical file paths for a bot's logs / state / config / status."""
    d = run_dir(bot)
    return {"dir": d, "equity": d / "equity_log.csv", "rebalance": d / "rebalance_log.csv",
            "state": d / "state.json", "config": d / "config.json", "status": d / "status.json"}


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except Exception:  # noqa: BLE001
        pass
