"""Evaluate one fixed MA200 strategy with a 50%/100% on-chain sizing overlay."""
from __future__ import annotations

import json

import pandas as pd
from btc_etf_flow_study import period_metrics, simulate
from btc_onchain_exchange_flow_study import (
    BASE_COST_BPS,
    RESULT_DIR,
    STRESS_COST_BPS,
    control_targets,
    load_data,
    market_frame,
)


def scaled_target(data: pd.DataFrame) -> pd.Series:
    """Return a fixed 0%/50%/100% target with conservative on-chain availability."""
    price = data["PriceUSD"]
    trend = price > price.rolling(200).mean()
    scarcity = data["SplyExNtv"] < data["SplyExNtv"].rolling(365).mean().shift(1)
    available = scarcity.copy()
    available.index = available.index + pd.Timedelta(days=1)
    available = available.reindex(data.index).fillna(False).astype(bool)
    conviction = pd.Series(0.5, index=data.index).where(~available, 1.0)
    return (trend.astype(float) * conviction).rename("scaled_onchain_ma200")


def main() -> None:
    data = load_data()
    market = market_frame(data)
    target = scaled_target(data)
    controls = control_targets(data)
    scenarios = (
        ("scaled_onchain_ma200", target, "base_lag1", BASE_COST_BPS, 0),
        ("scaled_onchain_ma200", target, "base_lag2", BASE_COST_BPS, 1),
        ("scaled_onchain_ma200", target, "stress_lag1", STRESS_COST_BPS, 0),
        ("scaled_onchain_ma200", target, "stress_lag2", STRESS_COST_BPS, 1),
        ("price_trend_ma200", controls["price_trend_ma200"], "control", 15.0, 0),
        ("buy_hold", controls["buy_hold"], "control", 15.0, 0),
    )
    periods = (
        ("development", "2015-01-01", "2020-01-01"),
        ("validation", "2020-01-01", "2024-01-01"),
        ("recent", "2024-01-01", "2026-07-20"),
        ("full", "2015-01-01", "2026-07-20"),
    )
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for name, scenario_target, scenario, cost_bps, extra_lag in scenarios:
        result = simulate(
            scenario_target, market, cost_bps=cost_bps, extra_lag=extra_lag
        )
        for period, start, end in periods:
            rows.append(
                {
                    "candidate": name,
                    "scenario": scenario,
                    "period": period,
                    "cost_bps": cost_bps,
                    "execution_lag_days": 1 + extra_lag,
                    **period_metrics(result, start, end),
                }
            )
        returns[f"{name}__{scenario}__total"] = result["total_return"].loc[
            "2015-01-01":
        ]
        returns[f"{name}__{scenario}__excess"] = result["excess_return"].loc[
            "2015-01-01":
        ]
        if name == "scaled_onchain_ma200" and scenario == "base_lag1":
            for year in range(2015, 2027):
                start = f"{year}-01-01"
                end = f"{year + 1}-01-01"
                if pd.Timestamp(start) > result.index.max():
                    continue
                rows.append(
                    {
                        "candidate": name,
                        "scenario": scenario,
                        "period": str(year),
                        "cost_bps": cost_bps,
                        "execution_lag_days": 1,
                        **period_metrics(result, start, end),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULT_DIR / "scaled_overlay_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(RESULT_DIR / "scaled_overlay_returns.parquet")
    meta = {
        "status": "post-hoc paper-monitor test; no untouched holdout",
        "rule": "MA200 risk switch; 50% BTC normally and 100% when exchange supply is below trailing 365d mean",
        "parameter_search": "none; 50% midpoint fixed before evaluation",
    }
    (RESULT_DIR / "scaled_overlay_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(
        metrics.loc[
            metrics["period"].isin(["development", "validation", "recent", "full"])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
