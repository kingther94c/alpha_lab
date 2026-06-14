"""P7-04: a detached bot's stdout/stderr go to the bot's run.log (not DEVNULL), so a
crash-on-launch leaves a trace. Asserts the Popen wiring without spawning a real process."""
from __future__ import annotations

import subprocess

from quant_bot_manager.core import config, state


def _patch_paths(monkeypatch, tmp_path):
    def fake_paths(bot):
        d = tmp_path / bot
        d.mkdir(parents=True, exist_ok=True)
        return {"dir": d, "db": d / "bot.db", "log": d / "run.log",
                "equity": d / "e.csv", "rebalance": d / "r.csv",
                "state": d / "s.json", "config": d / "c.json", "status": d / "st.json"}

    monkeypatch.setattr(config, "paths", fake_paths)


def test_start_bot_logs_to_run_log_not_devnull(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"], captured["kwargs"] = cmd, kwargs
        return type("_P", (), {"pid": 4242})()      # start_bot ignores the return value

    monkeypatch.setattr(state.subprocess, "Popen", fake_popen)
    msg = state.start_bot(
        {"capital": 10000.0, "method": "equal_capital", "max_gross": 2.0, "interval_min": 15.0},
        "t_log")

    assert msg == "start requested"
    kw = captured["kwargs"]
    assert kw["stdout"] is not subprocess.DEVNULL and hasattr(kw["stdout"], "write")  # a real file, not /dev/null
    assert kw["stderr"] == subprocess.STDOUT                                          # stderr folds into the log
    log_path = tmp_path / "t_log" / "run.log"
    assert log_path.exists()                                                          # opened for append
    kw["stdout"].write(b"crash trace\n")                                              # the child can write through it
    kw["stdout"].flush()
    assert b"crash trace" in log_path.read_bytes()


def test_start_bot_refuses_when_already_running(tmp_path, monkeypatch):
    # P7-16 single-flight: a second Start must not spawn a second loop on the same bot/account
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(state, "is_running", lambda bot=None: True)
    spawned = {"popen": False}
    monkeypatch.setattr(state.subprocess, "Popen", lambda *a, **k: spawned.__setitem__("popen", True))
    msg = state.start_bot({"capital": 10000.0, "method": "equal_capital",
                           "max_gross": 2.0, "interval_min": 15.0}, "t_dup")
    assert "already running" in msg
    assert spawned["popen"] is False
