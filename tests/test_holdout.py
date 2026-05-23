"""Tests for src/alpha_lab/data/holdout.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.data import holdout as hm


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path, monkeypatch):
    """Redirect the audit log path so tests don't pollute the real log."""
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(hm, "audit_log_path", lambda: log)
    yield log


def _hd(allow: bool = False) -> hm.PMHoldout:
    return hm.PMHoldout(
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-05-01", tz="UTC"),
        allow=allow,
    )


def test_pmholdout_contains_any_and_mask():
    h = _hd()
    idx = pd.date_range("2025-12-30", "2026-01-03", freq="D", tz="UTC")
    assert h.contains_any(idx) is True
    m = h.mask(idx)
    assert m.tolist() == [False, False, True, True, True]


def test_pmholdout_half_open():
    h = _hd()
    # end is exclusive
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-04-30 23:59", tz="UTC")]
    )
    assert h.mask(idx).tolist() == [False, True]


def test_enforce_raises_when_forbidden(_isolate_audit_log):
    h = _hd(allow=False)
    df = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.date_range("2026-01-02", periods=3, freq="D", tz="UTC"),
    )
    with pytest.raises(hm.PMHoldoutAccessError):
        hm.enforce(df, holdout=h, context="test")
    # audit log has exactly one "raised" event
    lines = _isolate_audit_log.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "raised"
    assert rec["context"] == "test"
    assert rec["n_rows_in_holdout"] == 3
    assert rec["allow"] is False


def test_enforce_allows_when_allow_true(_isolate_audit_log):
    h = _hd(allow=True)
    df = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.date_range("2026-01-02", periods=3, freq="D", tz="UTC"),
    )
    out = hm.enforce(df, holdout=h, context="unlocked")
    pd.testing.assert_frame_equal(out, df)
    rec = json.loads(_isolate_audit_log.read_text().strip())
    assert rec["action"] == "ok"
    assert rec["allow"] is True
    assert rec["n_rows_in_holdout"] == 3


def test_enforce_passthrough_when_outside_holdout(_isolate_audit_log):
    h = _hd()
    df = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
    )
    out = hm.enforce(df, holdout=h, context="pre")
    pd.testing.assert_frame_equal(out, df)
    rec = json.loads(_isolate_audit_log.read_text().strip())
    assert rec["action"] == "ok"
    assert rec["n_rows_in_holdout"] == 0


def test_enforce_non_time_indexed_pass(_isolate_audit_log):
    h = _hd()
    df = pd.DataFrame({"x": [1, 2]})  # default RangeIndex
    out = hm.enforce(df, holdout=h, context="rangeindex")
    pd.testing.assert_frame_equal(out, df)
    # nothing written
    assert not _isolate_audit_log.exists() or _isolate_audit_log.read_text() == ""


def test_safe_forward_returns_masks_window_crossing_boundary():
    h = _hd()
    idx = pd.date_range("2025-12-30", "2026-01-05", freq="D", tz="UTC")
    r = pd.Series(0.01, index=idx)  # constant +1%/day
    fwd1 = hm.safe_forward_returns(r, horizon=1, holdout=h)
    # at t=2025-12-30 horizon-1 forward is r[2025-12-31] which is OUTSIDE holdout → not masked
    assert not pd.isna(fwd1.loc["2025-12-30"])
    # at t=2025-12-31 horizon-1 forward is r[2026-01-01] INSIDE holdout → masked
    assert pd.isna(fwd1.loc["2025-12-31"])

    fwd3 = hm.safe_forward_returns(r, horizon=3, holdout=h)
    # at t=2025-12-29 horizon-3 forward is r[12-30, 12-31, 01-01] → touches holdout
    # only 2025-12-29 isn't in the index though; use 2025-12-30 as the first usable t
    # at t=2025-12-30 horizon-3 forward = r[12-31, 01-01, 01-02] → touches holdout → masked
    assert pd.isna(fwd3.loc["2025-12-30"])


def test_safe_forward_returns_horizon_validation():
    with pytest.raises(ValueError):
        hm.safe_forward_returns(pd.Series([1.0]), horizon=0, holdout=_hd())


def test_read_audit_log_empty(_isolate_audit_log):
    df = hm.read_audit_log()
    assert df.empty
    assert list(df.columns) == ["ts", "context", "allow", "holdout_start",
                                "holdout_end", "n_rows_in_holdout", "action"]


def test_access_summary_no_access(_isolate_audit_log):
    # write a few "ok" events
    h = _hd()
    df = pd.DataFrame(
        {"x": [1]},
        index=pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
    )
    hm.enforce(df, holdout=h, context="a")
    hm.enforce(df, holdout=h, context="b")
    summary = hm.access_summary_for_report()
    assert summary["accessed"] is False
    assert summary["n_events"] == 2
    assert summary["n_raises"] == 0


def test_access_summary_after_raise(_isolate_audit_log):
    h = _hd()
    df_in = pd.DataFrame(
        {"x": [1]},
        index=pd.date_range("2026-02-01", periods=1, freq="D", tz="UTC"),
    )
    with pytest.raises(hm.PMHoldoutAccessError):
        hm.enforce(df_in, holdout=h, context="leak")
    summary = hm.access_summary_for_report()
    assert summary["accessed"] is True
    assert summary["n_raises"] == 1
