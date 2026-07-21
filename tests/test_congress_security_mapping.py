from __future__ import annotations

import pandas as pd

from alpha_lab.utils.paths import CONFIGS_DIR


def _mapping() -> pd.DataFrame:
    return pd.read_csv(
        CONFIGS_DIR / "congress_ticker_sector.csv",
        comment="#",
    ).dropna(subset=["ticker", "sector"])


def test_congress_security_mapping_is_unique_and_uses_known_sectors() -> None:
    mapping = _mapping()
    sectors = set(pd.read_csv(CONFIGS_DIR / "us_sector_etf.csv")["sector"])

    assert mapping["ticker"].is_unique
    assert set(mapping["sector"]) <= sectors


def test_congress_security_mapping_includes_recent_tickers() -> None:
    mapping = _mapping().set_index("ticker")["sector"]

    expected = {
        "CRWV": "Technology",
        "ETOR": "Financials",
        "MIAX": "Financials",
        "PSKY": "Communication Services",
        "SARO": "Industrials",
        "SMA": "Real Estate",
        "SNDK": "Technology",
        "TEM": "Health Care",
        "UBER": "Industrials",
        "VIK": "Consumer Discretionary",
    }
    assert mapping.reindex(expected).to_dict() == expected
