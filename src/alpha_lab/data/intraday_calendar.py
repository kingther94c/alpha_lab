"""24/7 UTC bar-grid helpers for crypto intraday data.

Distinct from :mod:`alpha_lab.data.calendars` which is business-day for US
equities. Crypto trades continuously — no holidays, no weekend gaps.
"""

from __future__ import annotations

import pandas as pd


_INTERVAL_TO_OFFSET: dict[str, str] = {
    "1m":  "1min",
    "3m":  "3min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "1d":  "1D",
}


def to_pandas_freq(interval: str) -> str:
    """Map a Binance-style interval string to a pandas offset alias."""
    if interval not in _INTERVAL_TO_OFFSET:
        raise ValueError(
            f"Unknown interval: {interval!r}. Expected one of {sorted(_INTERVAL_TO_OFFSET)}."
        )
    return _INTERVAL_TO_OFFSET[interval]


def expected_bars(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str,
) -> pd.DatetimeIndex:
    """Expected continuous UTC bar grid for the half-open interval ``[start, end)``.

    Returns a tz-aware (UTC) ``DatetimeIndex``. ``end`` is exclusive, matching
    the project splits convention.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tz is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    if e.tz is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")
    return pd.date_range(start=s, end=e, freq=to_pandas_freq(interval), inclusive="left")


def gap_report(index: pd.DatetimeIndex, interval: str) -> pd.DataFrame:
    """Detect gaps in *index* against the expected 24/7 grid.

    Returns one row per contiguous gap with columns
    ``gap_start, gap_end, n_missing_bars, duration``. Empty DataFrame if no
    gaps. Assumes *index* is sorted and tz-aware (UTC) or tz-naive.
    """
    cols = ["gap_start", "gap_end", "n_missing_bars", "duration"]
    if len(index) < 2:
        return pd.DataFrame(columns=cols)
    idx = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    idx = idx.sort_values()
    step = pd.Timedelta(_INTERVAL_TO_OFFSET[interval])
    expected = pd.date_range(idx.min(), idx.max(), freq=_INTERVAL_TO_OFFSET[interval], tz="UTC")
    missing = expected.difference(idx)
    if len(missing) == 0:
        return pd.DataFrame(columns=cols)
    groups: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    cur_start = missing[0]
    cur_end = missing[0]
    for ts in missing[1:]:
        if ts == cur_end + step:
            cur_end = ts
        else:
            n = int((cur_end - cur_start) / step) + 1
            groups.append((cur_start, cur_end, n))
            cur_start = ts
            cur_end = ts
    groups.append((cur_start, cur_end, int((cur_end - cur_start) / step) + 1))
    df = pd.DataFrame(groups, columns=["gap_start", "gap_end", "n_missing_bars"])
    df["duration"] = df["gap_end"] - df["gap_start"] + step
    return df


def duplicates_report(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the unique timestamps that occur more than once in *index*."""
    s = pd.Series(index)
    dupes = s[s.duplicated(keep=False)].unique()
    return pd.DatetimeIndex(dupes)
