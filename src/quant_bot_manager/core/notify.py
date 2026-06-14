"""Out-of-band alerting — tell the operator when the bot needs attention while no one watches the UI.

Best-effort: one optional webhook from the env (``BOT_ALERT_WEBHOOK``). Any failure is swallowed so a
dead hook never kills the trading loop. Callers alert on STATE TRANSITIONS (auto-halt latched, an error
streak crossing a threshold), not every cycle, so the channel stays signal. A process cannot alert its
own death — a stale-heartbeat watcher is a separate, external concern (see the runbook).
"""
from __future__ import annotations

import json
import os
import urllib.request

ALERT_ENV = "BOT_ALERT_WEBHOOK"


def send(event: str, detail: str) -> bool:
    """POST ``{event, detail}`` to ``$BOT_ALERT_WEBHOOK`` if set. Returns True if sent, else False. Never raises."""
    url = os.environ.get(ALERT_ENV)
    if not url:
        return False
    try:
        body = json.dumps({"event": event, "detail": detail}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)   # noqa: S310 — operator-configured webhook, not user input
        return True
    except Exception:  # noqa: BLE001 — a dead hook must never crash the loop
        return False
