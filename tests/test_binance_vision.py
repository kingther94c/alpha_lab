"""Tests for src/alpha_lab/data/loaders/binance_vision.py.

Network calls are mocked. ZIP fixtures are written into pytest tmp_paths,
and the loader's RAW_DIR / INTERIM_DIR are redirected via monkeypatch.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

from alpha_lab.data import holdout as hm
from alpha_lab.data.loaders import binance_vision as bv


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    monkeypatch.setattr(bv, "_raw_root", lambda: raw / "binance_vision")
    monkeypatch.setattr(bv, "_interim_root", lambda: interim / "binance")
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(hm, "audit_log_path", lambda: audit)
    yield raw, interim, audit


def _hd_allow_all() -> hm.PMHoldout:
    # holdout window far in the future so test data never overlaps
    return hm.PMHoldout(
        start=pd.Timestamp("2099-01-01", tz="UTC"),
        end=pd.Timestamp("2099-02-01", tz="UTC"),
        allow=False,
    )


def _make_kline_csv(rows: pd.DataFrame, with_header: bool = False) -> str:
    """Render an 11-column kline CSV string."""
    out = io.StringIO()
    if with_header:
        out.write(",".join(bv._KLINE_COLUMNS) + "\n")
    for _, r in rows.iterrows():
        out.write(
            f"{int(r['open_time'])},{r['open']},{r['high']},{r['low']},{r['close']},{r['volume']},"
            f"{int(r['close_time'])},{r['quote_volume']},{int(r['n_trades'])},"
            f"{r['taker_buy_base']},{r['taker_buy_quote']}\n"
        )
    return out.getvalue()


def _write_kline_zip(path: Path, rows: pd.DataFrame, with_header: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = _make_kline_csv(rows, with_header=with_header)
    with zipfile.ZipFile(path, "w") as zf:
        # Inner CSV filename = ZIP stem with .csv extension
        zf.writestr(path.stem + ".csv", csv)
    return path


def _synthetic_kline_rows(start_utc: str, n_bars: int, interval_ms: int = 60_000) -> pd.DataFrame:
    """Generate n synthetic 1-minute bars beginning at start_utc."""
    start_ms = int(pd.Timestamp(start_utc, tz="UTC").value // 1_000_000)
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.1, n_bars))
    rows = []
    for i in range(n_bars):
        ot = start_ms + i * interval_ms
        ct = ot + interval_ms - 1
        c = closes[i]
        rows.append({
            "open_time": ot,
            "open": c - 0.05,
            "high": c + 0.05,
            "low": c - 0.10,
            "close": c,
            "volume": 10.0 + rng.uniform(0, 5),
            "close_time": ct,
            "quote_volume": 1000.0,
            "n_trades": int(rng.integers(10, 100)),
            "taker_buy_base": 5.0,
            "taker_buy_quote": 500.0,
        })
    return pd.DataFrame(rows)


# --- Parsing tests ---------------------------------------------------------

def test_has_header_row_detection():
    assert bv._has_header_row(b"open_time,open,high\n123,1.0,2.0\n") is True
    assert bv._has_header_row(b"1700000000000,1.0,2.0\n") is False
    assert bv._has_header_row(b"-1700000000000,1.0,2.0\n") is False


def test_parse_kline_zip_no_header(tmp_path):
    rows = _synthetic_kline_rows("2024-01-01 00:00", 5)
    z = _write_kline_zip(tmp_path / "BTCUSDT-1m-2024-01.zip", rows, with_header=False)
    df = bv.parse_kline_zip(z)
    assert len(df) == 5
    assert df.index.tz is not None and df.index.tz.utcoffset(None).total_seconds() == 0
    assert list(df.columns) == bv._OHLCV_OUT_COLS
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2024-01-01 00:04", tz="UTC")


def test_parse_kline_zip_with_header(tmp_path):
    rows = _synthetic_kline_rows("2024-06-01 00:00", 3)
    z = _write_kline_zip(tmp_path / "BTCUSDT-1m-2024-06.zip", rows, with_header=True)
    df = bv.parse_kline_zip(z)
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-06-01 00:00", tz="UTC")


# --- _months_in_range ------------------------------------------------------

def test_months_in_range_full_months():
    s = pd.Timestamp("2024-01-01", tz="UTC")
    e = pd.Timestamp("2024-04-01", tz="UTC")
    assert bv._months_in_range(s, e) == ["2024-01", "2024-02", "2024-03"]


def test_months_in_range_partial_starts_and_ends():
    s = pd.Timestamp("2024-01-15", tz="UTC")
    e = pd.Timestamp("2024-03-10", tz="UTC")
    assert bv._months_in_range(s, e) == ["2024-01", "2024-02", "2024-03"]


def test_months_in_range_year_boundary():
    s = pd.Timestamp("2023-11-15", tz="UTC")
    e = pd.Timestamp("2024-02-10", tz="UTC")
    assert bv._months_in_range(s, e) == ["2023-11", "2023-12", "2024-01", "2024-02"]


# --- available_history (mocked S3 XML) -------------------------------------

def _s3_xml(keys_with_sizes: list[tuple[str, int]], truncated: bool = False, next_token: str = "") -> bytes:
    items = "".join(
        f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>"
        for k, s in keys_with_sizes
    )
    next_tok_xml = f"<NextContinuationToken>{next_token}</NextContinuationToken>" if truncated else ""
    return (
        f"<?xml version='1.0'?>"
        f"<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>"
        f"{items}"
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        f"{next_tok_xml}"
        f"</ListBucketResult>"
    ).encode()


def test_available_history_klines(monkeypatch):
    keys = [
        ("data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip", 12345),
        ("data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip.CHECKSUM", 64),
        ("data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-02.zip", 23456),
    ]
    session = MagicMock()
    resp = MagicMock()
    resp.content = _s3_xml(keys)
    resp.raise_for_status = MagicMock()
    session.get.return_value = resp

    df = bv.available_history("perp", "BTCUSDT", "1m", session=session)
    assert list(df["period"]) == ["2024-01", "2024-02"]
    assert all(u.startswith(bv.BASE_URL) for u in df["url"])
    assert df.loc[df["period"] == "2024-01", "size_bytes"].item() == 12345


def test_available_history_funding_rejects_spot():
    with pytest.raises(ValueError):
        bv.available_history("spot", "BTCUSDT", "1m", kind="fundingRate")


def test_available_history_pagination(monkeypatch):
    page1 = _s3_xml(
        [("data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip", 100)],
        truncated=True, next_token="TOKEN2",
    )
    page2 = _s3_xml(
        [("data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-02.zip", 200)],
    )
    session = MagicMock()
    resp1 = MagicMock(); resp1.content = page1; resp1.raise_for_status = MagicMock()
    resp2 = MagicMock(); resp2.content = page2; resp2.raise_for_status = MagicMock()
    session.get.side_effect = [resp1, resp2]
    df = bv.available_history("perp", "BTCUSDT", "1m", session=session)
    assert list(df["period"]) == ["2024-01", "2024-02"]
    assert session.get.call_count == 2


# --- load_klines integration (with pre-placed cache ZIPs) ------------------

def test_load_klines_reads_cached_zips(isolated_dirs):
    raw, interim, _audit = isolated_dirs
    # Pre-place two months of synthetic data for BTCUSDT perp 1m
    z1 = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-01.zip"
    z2 = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-02.zip"
    _write_kline_zip(z1, _synthetic_kline_rows("2024-01-01 00:00", 60))
    _write_kline_zip(z2, _synthetic_kline_rows("2024-02-01 00:00", 60))

    panel = bv.load_klines(
        ["BTCUSDT"], "1m",
        start="2024-01-01", end="2024-03-01",
        market="perp",
        holdout=_hd_allow_all(),
    )
    assert "BTCUSDT" in panel.frames
    df = panel.frames["BTCUSDT"]
    assert len(df) == 120  # 60 from each month
    assert list(df.columns) == bv._OHLCV_OUT_COLS
    # close_panel should produce a wide single-column DataFrame
    cp = panel.close_panel()
    assert list(cp.columns) == ["BTCUSDT"]
    assert len(cp) == 120


def test_load_klines_slices_to_requested_window(isolated_dirs):
    raw, _interim, _audit = isolated_dirs
    z = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-01.zip"
    _write_kline_zip(z, _synthetic_kline_rows("2024-01-01 00:00", 60))
    panel = bv.load_klines(
        ["BTCUSDT"], "1m",
        start="2024-01-01 00:10", end="2024-01-01 00:20",
        market="perp",
        holdout=_hd_allow_all(),
    )
    df = panel.frames["BTCUSDT"]
    # half-open [00:10, 00:20)
    assert len(df) == 10
    assert df.index[0] == pd.Timestamp("2024-01-01 00:10", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2024-01-01 00:19", tz="UTC")


def test_load_klines_skips_missing_months_silently(isolated_dirs, monkeypatch):
    """A 404 for a month should be skipped without aborting the load."""
    raw, _interim, _audit = isolated_dirs
    # Only place the second month's ZIP; first month will 404
    z2 = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-02.zip"
    _write_kline_zip(z2, _synthetic_kline_rows("2024-02-01 00:00", 60))

    # Mock requests so the 2024-01 download 404s; 2024-02 already exists, won't be requested
    original_download = bv._download_zip
    def mock_download(url, dest, *, refresh=False, session=None):
        if dest.exists() and not refresh:
            return dest
        # simulate 404
        resp = MagicMock(); resp.status_code = 404
        err = requests.HTTPError(response=resp)
        raise err
    monkeypatch.setattr(bv, "_download_zip", mock_download)

    panel = bv.load_klines(
        ["BTCUSDT"], "1m",
        start="2024-01-01", end="2024-03-01",
        market="perp",
        holdout=_hd_allow_all(),
    )
    df = panel.frames["BTCUSDT"]
    assert len(df) == 60  # only the available month
    assert df.index[0] == pd.Timestamp("2024-02-01 00:00", tz="UTC")


def test_load_klines_enforces_holdout(isolated_dirs):
    """Loading data that overlaps the holdout must raise PMHoldoutAccessError."""
    raw, _interim, _audit = isolated_dirs
    z = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2026-02.zip"
    _write_kline_zip(z, _synthetic_kline_rows("2026-02-01 00:00", 30))
    hd = hm.PMHoldout(
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-05-01", tz="UTC"),
        allow=False,
    )
    with pytest.raises(hm.PMHoldoutAccessError):
        bv.load_klines(
            ["BTCUSDT"], "1m",
            start="2026-02-01", end="2026-02-02",
            market="perp",
            holdout=hd,
        )


def test_data_quality_report_basic(isolated_dirs):
    raw, _interim, _audit = isolated_dirs
    # 60 bars from start of 2024-01 + 60 from start of 2024-02, no gaps in either month
    z1 = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-01.zip"
    z2 = raw / "binance_vision" / "perp" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-02.zip"
    _write_kline_zip(z1, _synthetic_kline_rows("2024-01-01 00:00", 60))
    _write_kline_zip(z2, _synthetic_kline_rows("2024-02-01 00:00", 60))
    panel = bv.load_klines(
        ["BTCUSDT"], "1m",
        start="2024-01-01", end="2024-03-01",
        market="perp",
        holdout=_hd_allow_all(),
    )
    qr = bv.data_quality_report(panel)
    assert set(qr.columns) >= {
        "symbol", "year_month", "n_bars_expected", "n_bars_actual",
        "n_duplicates", "n_gaps", "max_gap_bars", "n_zero_volume_bars",
        "first_ts", "last_ts", "ok",
    }
    assert set(qr["year_month"]) == {"2024-01", "2024-02"}
    # n_actual = 60 each. expected = 31*24*60 and 29*24*60 (Feb 2024 leap-year)
    row1 = qr[qr["year_month"] == "2024-01"].iloc[0]
    assert row1["n_bars_actual"] == 60
    assert row1["n_duplicates"] == 0
    assert bool(row1["ok"]) is False  # only 60 of ~44640 expected, very incomplete


# --- market path validation ------------------------------------------------

def test_market_path_unknown_raises():
    with pytest.raises(ValueError):
        bv._market_path("derivatives")  # type: ignore[arg-type]
