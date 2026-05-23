"""PM-holdout enforcement.

Single source of truth for the locked PM-evaluation window. Loaders and the
backtest engine call ``enforce()`` before returning data; a forbidden access
raises ``PMHoldoutAccessError``. An append-only JSONL audit log at
``data/results/pm_holdout_audit.jsonl`` records every check.

To temporarily unlock for the explicit final-eval cell, construct a
``PMHoldout(start, end, allow=True)`` and pass it explicitly to ``enforce``.
There is no global override switch by design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.utils.config import load_config
from alpha_lab.utils.paths import RESULTS_DIR, ensure_dir


def audit_log_path() -> Path:
    """Path to the append-only PM-holdout audit log (JSONL)."""
    return RESULTS_DIR / "pm_holdout_audit.jsonl"


class PMHoldoutAccessError(RuntimeError):
    """Raised when forbidden PM-holdout rows are touched."""


@dataclass(frozen=True)
class PMHoldout:
    """Half-open PM holdout window ``[start, end)``. Both bounds are UTC."""

    start: pd.Timestamp
    end: pd.Timestamp
    allow: bool = False

    @classmethod
    def from_config(cls, name: str = "crypto_intraday") -> "PMHoldout":
        cfg = load_config(name)
        h = cfg.get("pm_holdout") or {}
        if "start" not in h or "end" not in h:
            raise KeyError(
                f"configs/{name}.yaml must define pm_holdout.start and pm_holdout.end"
            )
        return cls(
            start=pd.Timestamp(h["start"], tz="UTC"),
            end=pd.Timestamp(h["end"], tz="UTC"),
            allow=bool(h.get("allow", False)),
        )

    def contains_any(self, index: pd.DatetimeIndex) -> bool:
        idx = _to_utc(index)
        return bool(((idx >= self.start) & (idx < self.end)).any())

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """Boolean Series, True where index is inside ``[start, end)``."""
        idx = _to_utc(index)
        return pd.Series(
            (idx >= self.start) & (idx < self.end),
            index=idx,
            name="in_pm_holdout",
        )


def _to_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _append_audit(payload: dict[str, Any]) -> None:
    log_path = audit_log_path()
    ensure_dir(log_path.parent)
    record = {"ts": datetime.now(tz=timezone.utc).isoformat(), **payload}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def enforce(
    obj: pd.DataFrame | pd.Series,
    *,
    holdout: PMHoldout | None = None,
    context: str = "",
) -> pd.DataFrame | pd.Series:
    """Audit and (if forbidden) refuse PM-holdout access.

    - If ``holdout.allow == False`` and ``obj``'s DatetimeIndex intersects the
      holdout window, raise ``PMHoldoutAccessError``.
    - Always append one JSONL record to ``audit_log_path()``.
    - Returns ``obj`` unchanged (no silent filtering). Callers that want to
      slice the holdout out should do so explicitly; ``enforce`` is the
      final guard.

    Non-time-indexed inputs are returned unchanged with no audit entry.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj
    if holdout is None:
        holdout = PMHoldout.from_config()
    idx = _to_utc(obj.index)
    n_in = int(((idx >= holdout.start) & (idx < holdout.end)).sum())
    payload = {
        "context": context,
        "allow": holdout.allow,
        "holdout_start": holdout.start.isoformat(),
        "holdout_end": holdout.end.isoformat(),
        "n_rows_in_holdout": n_in,
    }
    if n_in > 0 and not holdout.allow:
        _append_audit({**payload, "action": "raised"})
        raise PMHoldoutAccessError(
            f"PM holdout access denied (context={context!r}): {n_in} rows in "
            f"[{holdout.start.isoformat()}, {holdout.end.isoformat()}) "
            f"with allow_pm_holdout=False."
        )
    _append_audit({**payload, "action": "ok"})
    return obj


def safe_forward_returns(
    returns: pd.DataFrame | pd.Series,
    horizon: int,
    *,
    holdout: PMHoldout | None = None,
) -> pd.DataFrame | pd.Series:
    """Forward returns over the next ``horizon`` bars, with rows whose target
    window peeks into the PM-holdout masked to NaN.

    At row ``t`` the standard forward return uses rows ``t+1 .. t+h``. If any
    of those bars fall inside ``[holdout.start, holdout.end)``, the label at
    ``t`` is set to NaN — preventing target leakage across the boundary.

    Does NOT call ``enforce``; intended to be used on data already restricted
    to the pre-holdout window.
    """
    from alpha_lab.data.align import forward_returns as _fwd  # local import to avoid cycle

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if holdout is None:
        holdout = PMHoldout.from_config()
    fwd = _fwd(returns, horizon=horizon)
    if not isinstance(fwd.index, pd.DatetimeIndex):
        return fwd
    idx = _to_utc(fwd.index)
    in_hd = pd.Series(
        ((idx >= holdout.start) & (idx < holdout.end)).astype("float64"),
        index=fwd.index,
    )
    # At row t, mask if any of in_hd[t+1..t+h] is True. Trick: rolling(h).max() at
    # row t = max of in_hd[t-h+1..t]; shift(-h) lifts it to row t-h, i.e. at
    # original row t we then see max of in_hd[t+1..t+h].
    look_ahead = in_hd.rolling(horizon, min_periods=1).max().shift(-horizon)
    mask = look_ahead.fillna(0.0).astype(bool)
    if isinstance(fwd, pd.Series):
        fwd = fwd.where(~mask)
    else:
        fwd = fwd.where(~mask.to_numpy()[:, None])
    return fwd


def read_audit_log() -> pd.DataFrame:
    """Read the append-only JSONL audit log into a DataFrame.

    Returns an empty DataFrame with the canonical columns if the log doesn't
    exist yet.
    """
    cols = ["ts", "context", "allow", "holdout_start", "holdout_end",
            "n_rows_in_holdout", "action"]
    path = audit_log_path()
    if not path.exists():
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def access_summary_for_report() -> dict[str, Any]:
    """Summary for the PM report's "PM Holdout was not accessed" banner.

    Returns a dict with keys ``accessed`` (bool — True if any ``raised`` event),
    ``n_events``, ``n_raises``, ``last_event_ts``.
    """
    df = read_audit_log()
    if df.empty:
        return {"accessed": False, "n_events": 0, "n_raises": 0,
                "last_event_ts": None}
    raised = df[df["action"] == "raised"]
    return {
        "accessed": bool(len(raised) > 0),
        "n_events": int(len(df)),
        "n_raises": int(len(raised)),
        "last_event_ts": str(df["ts"].iloc[-1]),
    }
