"""Revision-2 BTC ETF-flow study: surprise, breadth, and absorption signals.

The revision uses development/validation data only. ``screen`` chooses a provisional
leader by a pre-declared robust score; ``robustness`` attacks that leader before any
2026 price data is released.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from btc_etf_flow_robustness import block_bootstrap
from btc_etf_flow_study import (
    BASE_COST_BPS,
    DEV_END,
    ETF_COLUMNS,
    RESULT_DIR,
    STRESS_COST_BPS,
    Candidate,
    load_flows,
    load_market,
    period_metrics,
    simulate,
)

SEED = 20260720


def rolling_flow_surprise(flows: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    """Rolling OLS residual with every fit ending at the prior ETF observation."""
    total = flows["Total"].astype(float)
    btc = market["BTC"]
    x = pd.DataFrame(
        {
            "ret1": btc.pct_change().reindex(flows.index),
            "ret5": btc.pct_change(5).reindex(flows.index),
        },
        index=flows.index,
    )
    surprise = pd.Series(np.nan, index=flows.index, name="flow_surprise")
    for i in range(60, len(flows)):
        train_x = x.iloc[i - 60 : i]
        train_y = total.iloc[i - 60 : i]
        valid = train_x.notna().all(axis=1) & train_y.notna()
        if valid.sum() < 40 or x.iloc[i].isna().any():
            continue
        design = np.column_stack(
            [np.ones(valid.sum()), train_x.loc[valid, ["ret1", "ret5"]].to_numpy()]
        )
        beta, *_ = np.linalg.lstsq(design, train_y.loc[valid].to_numpy(), rcond=None)
        prediction = float(np.array([1.0, x.iloc[i]["ret1"], x.iloc[i]["ret5"]]) @ beta)
        surprise.iloc[i] = total.iloc[i] - prediction
    return surprise


def event_target(
    event: pd.Series,
    market: pd.DataFrame,
    *,
    hold: int,
    trend_window: int,
) -> pd.Series:
    """Hold after an ETF-date event, publish at t+1, and require trailing price trend."""
    held_event = event.rolling(hold, min_periods=1).max().astype(bool)
    published = held_event.copy()
    published.index = published.index + pd.Timedelta(days=1)
    daily = published.reindex(market.index).ffill().fillna(False)
    trend = market["BTC"] > market["BTC"].rolling(trend_window).mean()
    return (daily & trend).astype(float)


def revision_targets(
    flows: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, Candidate]]:
    """Build the 24 pre-declared revision candidates."""
    total = flows["Total"].astype(float)
    surprise = rolling_flow_surprise(flows, market)
    btc_ret5 = market["BTC"].pct_change(5).reindex(flows.index)
    active = flows[ETF_COLUMNS].notna().sum(axis=1).replace(0, np.nan)
    breadth = (
        (flows[ETF_COLUMNS].fillna(0.0) > 0).sum(axis=1)
        - (flows[ETF_COLUMNS].fillna(0.0) < 0).sum(axis=1)
    ) / active

    targets: dict[str, pd.Series] = {}
    specs: dict[str, Candidate] = {}
    for q in (0.70, 0.80):
        raw_threshold = total.rolling(60, min_periods=20).quantile(q).shift(1)
        surprise_threshold = surprise.rolling(60, min_periods=20).quantile(q).shift(1)
        events = {
            "unexpected": surprise > surprise_threshold,
            "breadth": (total > raw_threshold) & (breadth > 0.0),
            "absorption": (total > raw_threshold) & (btc_ret5 <= 0.0),
        }
        for family, event in events.items():
            for hold in (5, 10):
                for tw in (50, 100):
                    name = f"{family}_q{int(q * 100)}_hold{hold}_ma{tw}"
                    targets[name] = event_target(event, market, hold=hold, trend_window=tw)
                    specs[name] = Candidate(
                        name=name,
                        family=family,
                        trend_window=tw,
                        quantile=q,
                        hold=hold,
                        matched_control=f"price_trend_ma{tw}",
                    )
    return targets, specs


def controls(market: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        f"price_trend_ma{tw}": (
            market["BTC"] > market["BTC"].rolling(tw).mean()
        ).astype(float)
        for tw in (50, 100)
    }


def screen() -> None:
    flows = load_flows(end=DEV_END)
    market = load_market(phase="develop", refresh=False)
    targets, specs = revision_targets(flows, market)
    all_targets = {**controls(market), **targets}

    rows = []
    returns = {}
    for name, target in all_targets.items():
        for cost_label, cost_bps in (("base", BASE_COST_BPS), ("stress", STRESS_COST_BPS)):
            result = simulate(target, market, cost_bps=cost_bps)
            if cost_label == "base":
                returns[f"{name}__total"] = result["total_return"]
                returns[f"{name}__excess"] = result["excess_return"]
            for split, start, end in (
                ("development", "2024-03-15", "2025-01-01"),
                ("validation", "2025-01-01", "2026-01-01"),
            ):
                spec = specs.get(name)
                rows.append(
                    {
                        "candidate": name,
                        "split": split,
                        "cost": cost_label,
                        "family": spec.family if spec else "control",
                        "trend_window": spec.trend_window if spec else int(name.rsplit("ma", 1)[1]),
                        "quantile": spec.quantile if spec else np.nan,
                        "hold": spec.hold if spec else np.nan,
                        **period_metrics(result, start, end),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULT_DIR / "revision2_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(RESULT_DIR / "revision2_returns.parquet")

    base = metrics[metrics["cost"] == "base"]
    wide = base.pivot(index="candidate", columns="split")
    shortlist_rows = []
    for name, spec in specs.items():
        dev_sharpe = wide.loc[name, "ExcessSharpe"]["development"]
        val_sharpe = wide.loc[name, "ExcessSharpe"]["validation"]
        dev_cagr = wide.loc[name, "ExcessCAGR"]["development"]
        val_cagr = wide.loc[name, "ExcessCAGR"]["validation"]
        val_total = wide.loc[name, "TotalReturn"]["validation"]
        val_dd = wide.loc[name, "MaxDD"]["validation"]
        val_turn = wide.loc[name, "AnnTurnover"]["validation"]
        control = spec.matched_control
        control_sharpe = wide.loc[control, "ExcessSharpe"]["validation"]
        control_dd = wide.loc[control, "MaxDD"]["validation"]
        control_cagr = wide.loc[control, "TotalCAGR"]["validation"]
        candidate_cagr = wide.loc[name, "TotalCAGR"]["validation"]
        sharpe_add = val_sharpe - control_sharpe
        dd_reduction = abs(control_dd) - abs(val_dd)
        retention = candidate_cagr / control_cagr if control_cagr > 0 else np.nan
        incremental = (sharpe_add >= 0.05) or (
            dd_reduction >= 0.10 * abs(control_dd) and retention >= 0.80
        )
        eligible = bool(
            dev_cagr > 0 and val_cagr > 0 and val_total > 0 and val_dd >= -0.35 and incremental
        )
        shortlist_rows.append(
            {
                **asdict(spec),
                "eligible": eligible,
                "robust_score": min(dev_sharpe, val_sharpe),
                "development_excess_cagr": dev_cagr,
                "validation_excess_cagr": val_cagr,
                "development_excess_sharpe": dev_sharpe,
                "validation_excess_sharpe": val_sharpe,
                "validation_total_cagr": candidate_cagr,
                "validation_max_dd": val_dd,
                "validation_ann_turnover": val_turn,
                "validation_sharpe_add": sharpe_add,
                "validation_dd_reduction": dd_reduction,
                "validation_cagr_retention": retention,
            }
        )
    shortlist = pd.DataFrame(shortlist_rows).sort_values(
        ["eligible", "robust_score"], ascending=[False, False]
    )
    shortlist.to_csv(RESULT_DIR / "revision2_shortlist.csv", index=False)
    leader = shortlist.loc[shortlist["eligible"]].iloc[0]
    leader_spec = specs[str(leader["name"])]
    payload = {
        "candidate": leader_spec.name,
        "spec": asdict(leader_spec),
        "selection_rule": "max min(development, validation) excess Sharpe among eligible revision-2 cells",
        "holdout_frozen": False,
    }
    (RESULT_DIR / "revision2_leader.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(shortlist.head(24).to_string(index=False))


def robustness() -> None:
    flows = load_flows(end=DEV_END)
    market = load_market(phase="develop", refresh=False)
    leader_payload = json.loads((RESULT_DIR / "revision2_leader.json").read_text(encoding="utf-8"))
    selected = leader_payload["candidate"]
    targets, specs = revision_targets(flows, market)
    if leader_payload["spec"] != asdict(specs[selected]):
        raise ValueError("revision leader specification changed after screening")
    target = targets[selected]

    rows = []
    for cost_bps in (15.0, 30.0, 50.0, 100.0):
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
                        "candidate": selected,
                        "split": split,
                        "cost_bps": cost_bps,
                        "execution_lag_days": 1 + extra_lag,
                        **period_metrics(result, start, end),
                    }
                )
    robust = pd.DataFrame(rows)
    robust.to_csv(RESULT_DIR / "revision2_robustness.csv", index=False)

    base = simulate(target, market, cost_bps=15.0)
    observed_dev = period_metrics(base, "2024-03-15", "2025-01-01")
    observed_val = period_metrics(base, "2025-01-01", "2026-01-01")
    observed_score = min(observed_dev["ExcessSharpe"], observed_val["ExcessSharpe"])

    offsets = np.arange(20, len(flows) - 20)
    rng = np.random.default_rng(SEED)
    test_offsets = rng.choice(offsets, size=min(200, len(offsets)), replace=False)
    fixed_rows = []
    family_rows = []
    for offset in test_offsets:
        shifted = flows.copy()
        shifted.loc[:, ETF_COLUMNS + ["Total"]] = np.roll(
            flows[ETF_COLUMNS + ["Total"]].to_numpy(), offset, axis=0
        )
        shifted_targets, _ = revision_targets(shifted, market)
        fixed_result = simulate(shifted_targets[selected], market, cost_bps=15.0)
        fixed_val = period_metrics(fixed_result, "2025-01-01", "2026-01-01")
        fixed_rows.append(
            {
                "offset": int(offset),
                "ExcessCAGR": fixed_val["ExcessCAGR"],
                "ExcessSharpe": fixed_val["ExcessSharpe"],
            }
        )
        best = -np.inf
        for shifted_target in shifted_targets.values():
            result = simulate(shifted_target, market, cost_bps=15.0)
            dev = period_metrics(result, "2024-03-15", "2025-01-01")
            val = period_metrics(result, "2025-01-01", "2026-01-01")
            best = max(best, min(dev["ExcessSharpe"], val["ExcessSharpe"]))
        family_rows.append({"offset": int(offset), "best_min_sharpe": best})
    fixed = pd.DataFrame(fixed_rows)
    family = pd.DataFrame(family_rows)
    fixed.to_csv(RESULT_DIR / "revision2_permutation_fixed.csv", index=False)
    family.to_csv(RESULT_DIR / "revision2_permutation_familywise.csv", index=False)

    val_excess = base["excess_return"].loc[
        (base.index >= "2025-01-01") & (base.index < "2026-01-01")
    ]
    meta = {
        "candidate": selected,
        "spec": asdict(specs[selected]),
        "observed_robust_score": observed_score,
        "observed_validation_excess_cagr": observed_val["ExcessCAGR"],
        "observed_validation_excess_sharpe": observed_val["ExcessSharpe"],
        "fixed_shift_p_value": float(
            (1 + (fixed["ExcessSharpe"] >= observed_val["ExcessSharpe"]).sum())
            / (1 + len(fixed))
        ),
        "familywise_shift_p_value": float(
            (1 + (family["best_min_sharpe"] >= observed_score).sum())
            / (1 + len(family))
        ),
        "validation_block_bootstrap": block_bootstrap(val_excess),
    }
    (RESULT_DIR / "revision2_robustness_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(
        robust.loc[
            (robust["split"].isin(["development", "validation"]))
            & (robust["cost_bps"].isin([15.0, 30.0]))
            & (robust["execution_lag_days"].isin([1, 2]))
        ].to_string(index=False)
    )


def holdout(*, refresh: bool) -> None:
    """Release the frozen revision-2 rule on 2026 data exactly once."""
    frozen_path = RESULT_DIR / "revision2_frozen_spec.json"
    if not frozen_path.exists():
        raise FileNotFoundError("revision2_frozen_spec.json is required before holdout")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selected = frozen["candidate"]
    flows = load_flows(end=pd.Timestamp("2026-07-18"))
    market = load_market(phase="holdout", refresh=refresh)
    targets, specs = revision_targets(flows, market)
    if selected not in targets:
        raise KeyError(f"frozen candidate {selected!r} is not in revision-2")
    if frozen["spec"] != asdict(specs[selected]):
        raise ValueError("frozen specification does not match revision-2 strategy code")

    rows: list[dict[str, object]] = []
    return_columns: dict[str, pd.Series] = {}
    evaluation_end = str((market.index.max() + pd.Timedelta(days=1)).date())
    scenarios = (
        (selected, targets[selected], "base_lag1", BASE_COST_BPS, 0),
        (selected, targets[selected], "base_lag2", BASE_COST_BPS, 1),
        (selected, targets[selected], "stress_lag1", STRESS_COST_BPS, 0),
        (selected, targets[selected], "stress_lag2", STRESS_COST_BPS, 1),
        ("buy_hold", pd.Series(1.0, index=market.index), "control", BASE_COST_BPS, 0),
        ("price_trend_ma50", controls(market)["price_trend_ma50"], "control", BASE_COST_BPS, 0),
    )
    for name, target, scenario, cost_bps, extra_lag in scenarios:
        result = simulate(target, market, cost_bps=cost_bps, extra_lag=extra_lag)
        rows.append(
            {
                "candidate": name,
                "scenario": scenario,
                "cost_bps": cost_bps,
                "execution_lag_days": 1 + extra_lag,
                **period_metrics(result, "2026-01-01", evaluation_end),
            }
        )
        return_columns[f"{name}__{scenario}__total"] = result["total_return"].loc[
            "2026-01-01":
        ]
        return_columns[f"{name}__{scenario}__excess"] = result["excess_return"].loc[
            "2026-01-01":
        ]

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULT_DIR / "revision2_holdout_metrics.csv", index=False)
    pd.DataFrame(return_columns).to_parquet(RESULT_DIR / "revision2_holdout_returns.parquet")
    meta = {
        "frozen_candidate": selected,
        "frozen_at_utc": frozen["frozen_at_utc"],
        "flow_last": str(flows.index.max().date()),
        "market_last": str(market.index.max().date()),
        "evaluation_end_exclusive": evaluation_end,
        "publication_lag": "ETF date t available t+1 close; position earns no earlier than t+2",
    }
    (RESULT_DIR / "revision2_holdout_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("screen", "robustness", "holdout"), default="screen"
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.mode == "screen":
        screen()
    elif args.mode == "robustness":
        robustness()
    else:
        holdout(refresh=args.refresh)


if __name__ == "__main__":
    main()
