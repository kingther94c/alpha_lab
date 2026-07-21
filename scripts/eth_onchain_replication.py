"""Exact ETH replication of the fixed BTC on-chain sizing rule; no retuning."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests
from btc_etf_flow_study import period_metrics, simulate
from btc_onchain_exchange_flow_study import API_URL, METRICS, ROOT, load_data
from btc_scaled_onchain_overlay import scaled_target

DATA_PATH = ROOT / "data" / "interim" / "eth_coinmetrics_onchain.csv"
RESULT_DIR = ROOT / "data" / "results" / "crypto_onchain_replication"


def download_eth() -> pd.DataFrame:
    """Download the ETH metrics matching the BTC study and reuse the same cash rate."""
    params = {
        "assets": "eth",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": "2016-01-01",
        "end_time": "2026-07-19",
        "page_size": 10000,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    records = list(payload["data"])
    while payload.get("next_page_url"):
        response = requests.get(payload["next_page_url"], timeout=60)
        response.raise_for_status()
        payload = response.json()
        records.extend(payload["data"])
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame.pop("time"), utc=True).dt.tz_localize(None).dt.normalize()
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.set_index("date").sort_index()[METRICS]
    btc_cash = load_data()[["IRX", "rf"]]
    frame = frame.join(btc_cash, how="left")
    frame = frame.dropna()
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if frame.index.has_duplicates or not frame.index.equals(expected):
        raise ValueError("ETH daily history contains gaps")
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("ETH history contains invalid values")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(DATA_PATH, index=False, float_format="%.12g")
    return frame


def evaluate(frame: pd.DataFrame) -> None:
    """Evaluate the exact BTC rule on ETH with conservative ETH costs."""
    market = pd.DataFrame({"BTC": frame["PriceUSD"], "rf": frame["rf"]})
    scaled = scaled_target(frame)
    trend = (frame["PriceUSD"] > frame["PriceUSD"].rolling(200).mean()).astype(float)
    buy_hold = pd.Series(1.0, index=frame.index)
    scenarios = (
        ("scaled_onchain_ma200", scaled, "base_lag1", 20.0, 0),
        ("scaled_onchain_ma200", scaled, "base_lag2", 20.0, 1),
        ("scaled_onchain_ma200", scaled, "stress_lag1", 40.0, 0),
        ("price_trend_ma200", trend, "control", 20.0, 0),
        ("buy_hold", buy_hold, "control", 20.0, 0),
    )
    periods = (
        ("early", "2017-01-01", "2021-01-01"),
        ("middle", "2021-01-01", "2024-01-01"),
        ("recent", "2024-01-01", "2026-07-20"),
        ("full", "2017-01-01", "2026-07-20"),
    )
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for name, target, scenario, cost_bps, extra_lag in scenarios:
        result = simulate(target, market, cost_bps=cost_bps, extra_lag=extra_lag)
        for period, start, end in periods:
            rows.append(
                {
                    "asset": "ETH",
                    "candidate": name,
                    "scenario": scenario,
                    "period": period,
                    "cost_bps": cost_bps,
                    "execution_lag_days": 1 + extra_lag,
                    **period_metrics(result, start, end),
                }
            )
        returns[f"{name}__{scenario}__total"] = result["total_return"].loc[
            "2017-01-01":
        ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULT_DIR / "eth_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(RESULT_DIR / "eth_returns.parquet")
    meta = {
        "asset": "ETH",
        "status": "cross-asset replication; exact BTC rule with no retuning",
        "data_first": str(frame.index.min().date()),
        "data_last": str(frame.index.max().date()),
        "sol_status": "Coin Metrics Community catalog lacks the matching SOL metric set",
    }
    (RESULT_DIR / "eth_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(metrics.to_string(index=False))


def main() -> None:
    frame = download_eth() if not DATA_PATH.exists() else pd.read_csv(
        DATA_PATH, parse_dates=["date"]
    ).set_index("date")
    evaluate(frame)


if __name__ == "__main__":
    main()
