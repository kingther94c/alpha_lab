"""Tests for src/alpha_lab/data/intraday_calendar.py."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.data import intraday_calendar as ic


def test_expected_bars_half_open_1m():
    idx = ic.expected_bars("2024-01-01", "2024-01-01 00:05", "1m")
    expected = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 00:01", tz="UTC"),
            pd.Timestamp("2024-01-01 00:02", tz="UTC"),
            pd.Timestamp("2024-01-01 00:03", tz="UTC"),
            pd.Timestamp("2024-01-01 00:04", tz="UTC"),
        ]
    )
    pd.testing.assert_index_equal(idx, expected)


def test_expected_bars_24_7_no_weekend_skip():
    # Crypto runs through weekends — Saturday/Sunday bars present.
    idx = ic.expected_bars("2024-01-05", "2024-01-09", "1d")
    # 2024-01-05 Fri, 06 Sat, 07 Sun, 08 Mon (end=09 exclusive)
    assert len(idx) == 4
    weekdays = [t.day_name() for t in idx]
    assert "Saturday" in weekdays and "Sunday" in weekdays


def test_to_pandas_freq_known():
    assert ic.to_pandas_freq("1m") == "1min"
    assert ic.to_pandas_freq("4h") == "4h"
    assert ic.to_pandas_freq("1d") == "1D"


def test_to_pandas_freq_unknown_raises():
    with pytest.raises(ValueError):
        ic.to_pandas_freq("2.5m")


def test_gap_report_empty_when_continuous():
    idx = ic.expected_bars("2024-01-01", "2024-01-01 01:00", "1m")
    df = ic.gap_report(idx, "1m")
    assert df.empty
    assert list(df.columns) == ["gap_start", "gap_end", "n_missing_bars", "duration"]


def test_gap_report_detects_known_gap():
    full = ic.expected_bars("2024-01-01 00:00", "2024-01-01 00:10", "1m")
    # drop 00:03..00:05 (3 bars) and 00:08 (1 bar)
    drop_set = {
        pd.Timestamp("2024-01-01 00:03", tz="UTC"),
        pd.Timestamp("2024-01-01 00:04", tz="UTC"),
        pd.Timestamp("2024-01-01 00:05", tz="UTC"),
        pd.Timestamp("2024-01-01 00:08", tz="UTC"),
    }
    observed = pd.DatetimeIndex([t for t in full if t not in drop_set])
    df = ic.gap_report(observed, "1m")
    assert len(df) == 2
    g1 = df.iloc[0]
    assert g1["gap_start"] == pd.Timestamp("2024-01-01 00:03", tz="UTC")
    assert g1["gap_end"] == pd.Timestamp("2024-01-01 00:05", tz="UTC")
    assert g1["n_missing_bars"] == 3
    g2 = df.iloc[1]
    assert g2["n_missing_bars"] == 1


def test_gap_report_short_index():
    df = ic.gap_report(pd.DatetimeIndex([]), "1m")
    assert df.empty


def test_duplicates_report_finds_dupes():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 00:01", tz="UTC"),
            pd.Timestamp("2024-01-01 00:01", tz="UTC"),
            pd.Timestamp("2024-01-01 00:02", tz="UTC"),
            pd.Timestamp("2024-01-01 00:02", tz="UTC"),
        ]
    )
    dupes = ic.duplicates_report(idx)
    assert set(dupes) == {
        pd.Timestamp("2024-01-01 00:01", tz="UTC"),
        pd.Timestamp("2024-01-01 00:02", tz="UTC"),
    }


def test_duplicates_report_empty_when_unique():
    idx = ic.expected_bars("2024-01-01", "2024-01-01 00:03", "1m")
    assert len(ic.duplicates_report(idx)) == 0
