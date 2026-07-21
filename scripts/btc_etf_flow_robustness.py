"""Critical robustness review for the provisional BTC ETF-flow leader.

This script uses development/validation data only. It must run before a final
specification is frozen and before the 2026 holdout is released.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from btc_etf_flow_study import (
    DEV_END,
    RESULT_DIR,
    candidate_targets,
    load_flows,
    load_market,
    period_metrics,
    simulate,
)

from alpha_lab.backtest.metrics import monthly_table

SELECTED = "extreme_inflow_q80_hold10_ma100"
SEED = 20260720


def extreme_target(
    total: pd.Series,
    market: pd.DataFrame,
    *,
    quantile: float,
    hold: int,
    trend_window: int,
) -> pd.Series:
    """Build one extreme-inflow target using only prior threshold observations."""
    threshold = total.rolling(60, min_periods=20).quantile(quantile).shift(1)
    event = total > threshold
    held_event = event.rolling(hold, min_periods=1).max().astype(bool)
    published = held_event.copy()
    published.index = published.index + pd.Timedelta(days=1)
    daily_event = published.reindex(market.index).ffill().fillna(False)
    trend = market["BTC"] > market["BTC"].rolling(trend_window).mean()
    return (daily_event & trend).astype(float)


def block_bootstrap(values: pd.Series, *, block: int = 7, reps: int = 2_000) -> dict[str, float]:
    """Moving-block bootstrap intervals for compounded return and daily Sharpe."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(SEED)
    totals = np.empty(reps)
    sharpes = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        sample = np.concatenate([x[(s + np.arange(block)) % n] for s in starts])[:n]
        totals[i] = np.prod(1.0 + sample) - 1.0
        std = sample.std(ddof=1)
        sharpes[i] = sample.mean() / std * np.sqrt(365.0) if std > 0 else np.nan
    return {
        "return_p05": float(np.quantile(totals, 0.05)),
        "return_p50": float(np.quantile(totals, 0.50)),
        "return_p95": float(np.quantile(totals, 0.95)),
        "sharpe_p05": float(np.nanquantile(sharpes, 0.05)),
        "sharpe_p50": float(np.nanquantile(sharpes, 0.50)),
        "sharpe_p95": float(np.nanquantile(sharpes, 0.95)),
    }


def main() -> None:
    flows = load_flows(end=DEV_END)
    market = load_market(phase="develop", refresh=False)
    targets, specs = candidate_targets(flows, market)
    target = targets[SELECTED]

    rows: list[dict[str, object]] = []
    for cost_bps in (0.0, 15.0, 30.0, 50.0, 100.0, 200.0):
        for extra_lag in (0, 1, 2):
            result = simulate(target, market, cost_bps=cost_bps, extra_lag=extra_lag)
            for split, start, end in (
                ("development", "2024-03-15", "2025-01-01"),
                ("validation", "2025-01-01", "2026-01-01"),
                ("validation_h1", "2025-01-01", "2025-07-01"),
                ("validation_h2", "2025-07-01", "2026-01-01"),
            ):
                rows.append(
                    {
                        "candidate": SELECTED,
                        "split": split,
                        "cost_bps": cost_bps,
                        "execution_lag_days": 1 + extra_lag,
                        **period_metrics(result, start, end),
                    }
                )
    robustness = pd.DataFrame(rows)
    robustness.to_csv(RESULT_DIR / "development_robustness.csv", index=False)

    base = simulate(target, market, cost_bps=15.0)
    monthly = pd.concat(
        {
            "total": monthly_table(base["total_return"].loc["2024-03-15":]),
            "excess": monthly_table(base["excess_return"].loc["2024-03-15":]),
        },
        names=["return_type", "year"],
    )
    monthly.to_csv(RESULT_DIR / "development_selected_monthly.csv")

    # Fixed-spec circular-shift test: preserve the autocorrelation and marginal
    # distribution of ETF flows while breaking their calendar alignment with BTC.
    total = flows["Total"].astype(float)
    observed = period_metrics(base, "2025-01-01", "2026-01-01")
    offsets = np.arange(20, len(total) - 20)
    fixed_rows = []
    for offset in offsets:
        shuffled = pd.Series(np.roll(total.to_numpy(), offset), index=total.index)
        shuffled_target = extreme_target(
            shuffled, market, quantile=0.80, hold=10, trend_window=100
        )
        shuffled_result = simulate(shuffled_target, market, cost_bps=15.0)
        metrics = period_metrics(shuffled_result, "2025-01-01", "2026-01-01")
        fixed_rows.append(
            {
                "offset": int(offset),
                "ExcessCAGR": metrics["ExcessCAGR"],
                "ExcessSharpe": metrics["ExcessSharpe"],
                "MaxDD": metrics["MaxDD"],
            }
        )
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(RESULT_DIR / "permutation_fixed_spec.csv", index=False)

    # Family-wise test: for each shifted flow history, let the entire 27-cell
    # q/hold/trend family choose its best min(development, validation) Sharpe.
    rng = np.random.default_rng(SEED)
    family_offsets = rng.choice(offsets, size=min(200, len(offsets)), replace=False)
    family_rows = []
    observed_cells = []
    for q in (0.70, 0.80, 0.90):
        for hold in (3, 5, 10):
            for tw in (20, 50, 100):
                real_target = extreme_target(total, market, quantile=q, hold=hold, trend_window=tw)
                real_result = simulate(real_target, market, cost_bps=15.0)
                dev = period_metrics(real_result, "2024-03-15", "2025-01-01")
                val = period_metrics(real_result, "2025-01-01", "2026-01-01")
                observed_cells.append(min(dev["ExcessSharpe"], val["ExcessSharpe"]))
    observed_family_best = float(max(observed_cells))

    for offset in family_offsets:
        shuffled = pd.Series(np.roll(total.to_numpy(), offset), index=total.index)
        best = -np.inf
        for q in (0.70, 0.80, 0.90):
            for hold in (3, 5, 10):
                for tw in (20, 50, 100):
                    shuffled_target = extreme_target(
                        shuffled, market, quantile=q, hold=hold, trend_window=tw
                    )
                    result = simulate(shuffled_target, market, cost_bps=15.0)
                    dev = period_metrics(result, "2024-03-15", "2025-01-01")
                    val = period_metrics(result, "2025-01-01", "2026-01-01")
                    best = max(best, min(dev["ExcessSharpe"], val["ExcessSharpe"]))
        family_rows.append({"offset": int(offset), "best_min_sharpe": best})
    family = pd.DataFrame(family_rows)
    family.to_csv(RESULT_DIR / "permutation_familywise.csv", index=False)

    validation_excess = base["excess_return"].loc[
        (base.index >= "2025-01-01") & (base.index < "2026-01-01")
    ]
    bootstrap = block_bootstrap(validation_excess)
    meta = {
        "candidate": SELECTED,
        "spec": specs[SELECTED].__dict__,
        "observed_validation_excess_cagr": observed["ExcessCAGR"],
        "observed_validation_excess_sharpe": observed["ExcessSharpe"],
        "fixed_spec_shift_p_value": float(
            (1 + (fixed["ExcessSharpe"] >= observed["ExcessSharpe"]).sum())
            / (1 + len(fixed))
        ),
        "fixed_spec_shift_percentile": float(
            (fixed["ExcessSharpe"] < observed["ExcessSharpe"]).mean()
        ),
        "observed_family_best_min_sharpe": observed_family_best,
        "familywise_shift_p_value": float(
            (1 + (family["best_min_sharpe"] >= observed_family_best).sum())
            / (1 + len(family))
        ),
        "validation_block_bootstrap": bootstrap,
    }
    (RESULT_DIR / "development_robustness_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(
        robustness.loc[
            (robustness["split"].isin(["development", "validation"]))
            & (robustness["cost_bps"].isin([15.0, 30.0]))
            & (robustness["execution_lag_days"].isin([1, 2]))
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
