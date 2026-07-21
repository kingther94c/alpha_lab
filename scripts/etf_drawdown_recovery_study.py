"""Pre-2022 drawdown-duration audit and recovery-oriented ETF rule sweep."""

from __future__ import annotations

import itertools
import json

import etf_strategy_50plus_study as base
import numpy as np
import pandas as pd

OUT = base.OUT
PROTOCOL = (
    base.PROJECT_ROOT
    / "docs"
    / "research_decisions"
    / "2026-07-18_etf-drawdown-recovery-protocol.md"
)


def _drawdown_episodes(name: str, returns: pd.Series) -> pd.DataFrame:
    """Return ex-post drawdown episodes with trading-session durations."""
    values = returns.dropna()
    equity = (1.0 + values).cumprod()
    dd = equity / equity.cummax() - 1.0
    underwater = dd < -1e-12
    rows: list[dict[str, object]] = []
    start: int | None = None
    array = dd.to_numpy()
    index = dd.index
    for position, is_underwater in enumerate(underwater.to_numpy()):
        if is_underwater and start is None:
            start = position
        if start is not None and (not is_underwater or position == len(dd) - 1):
            stop = position if not is_underwater else position + 1
            window = array[start:stop]
            trough = start + int(np.argmin(window))
            completed = not is_underwater
            endpoint = position if completed else len(dd) - 1
            rows.append(
                {
                    "strategy": name,
                    "prior_peak": index[max(0, start - 1)].date().isoformat(),
                    "trough": index[trough].date().isoformat(),
                    "recovered": index[position].date().isoformat() if completed else "",
                    "completed": completed,
                    "depth": float(array[trough]),
                    "peak_to_trough_days": int(trough - max(0, start - 1)),
                    "trough_to_recovery_days": int(endpoint - trough),
                    "underwater_days": int(stop - start),
                    "within_20d": completed and endpoint - trough <= 20,
                }
            )
            start = None
    return pd.DataFrame(rows)


def _sector_recovery_targets(
    panel: pd.DataFrame,
    *,
    freq: str,
    cyclical_budget: float,
    momentum_lookback: int,
    trend_lookback: int,
    top_n: int,
) -> pd.DataFrame:
    """Build a causal defensive/cyclical barbell intended to re-enter rebounds faster."""
    returns = panel[base.SECTORS].pct_change()
    moving_average = panel[base.CYCLICAL_SECTORS].rolling(
        trend_lookback, min_periods=trend_lookback
    ).mean()
    momentum = (
        panel[base.CYCLICAL_SECTORS]
        / panel[base.CYCLICAL_SECTORS].shift(momentum_lookback)
        - 1.0
    )
    rows: list[pd.Series] = []
    for date in base._decision_dates(panel.index, freq):
        row = base._safe_row(panel.columns, date)
        row[:] = 0.0
        defensive_budget = 1.0 - cyclical_budget
        downside = (
            returns.loc[:date, base.DEFENSIVE_SECTORS]
            .tail(126)
            .clip(upper=0.0)
            .pow(2)
            .mean()
            .pow(0.5)
        )
        inverse_downside = (1.0 / downside.replace(0.0, np.nan)).dropna()
        if not inverse_downside.empty:
            row.loc[inverse_downside.index] = (
                inverse_downside / inverse_downside.sum() * defensive_budget
            )

        score = momentum.loc[date]
        eligible = score.where(
            (score > 0.0)
            & (panel.loc[date, base.CYCLICAL_SECTORS] > moving_average.loc[date])
        ).dropna().nlargest(top_n)
        if not eligible.empty:
            row.loc[eligible.index] = cyclical_budget / top_n
        row["SHY"] = 1.0 - float(row.sum())
        rows.append(row)
    return base._targets_from_rows(rows, panel.columns)


def _recovery_variants(adjusted: pd.DataFrame) -> list[base.StrategyRun]:
    runs: list[base.StrategyRun] = []

    def add(**kwargs: object) -> None:
        runs.append(base._run_weight_strategy(adjusted, **kwargs))

    for risky, lookback in itertools.product(("SPY", "QQQ"), (20, 50, 75, 100, 150)):
        add(
            name=f"recovery_{risky}_trend_{lookback}d",
            family="Recovery-oriented fast trend",
            independence_group="recovery_fast_trend",
            description=f"Weekly {risky} above {lookback}d moving average; otherwise SHY.",
            assets=[risky, "SHY"],
            target_builder=lambda panel, risky=risky, lookback=lookback: base._trend_targets(
                panel, risky, lookback
            ),
        )

    for risky, target, lookback in itertools.product(
        ("SPY", "QQQ"), (0.08, 0.10, 0.12), (20, 50, 100)
    ):
        add(
            name=f"recovery_{risky}_vol{int(target * 100)}_trend{lookback}",
            family="Recovery-oriented fast trend plus volatility target",
            independence_group="recovery_fast_vol_trend",
            description=(
                f"Weekly unlevered {risky} {target:.0%} volatility target, disabled below "
                f"the {lookback}d moving average; residual SHY."
            ),
            assets=[risky, "SHY"],
            target_builder=lambda panel, risky=risky, target=target, lookback=lookback: (
                base._vol_targets(panel, risky, target, trend_lookback=lookback)
            ),
        )

    for freq, budget, momentum, trend, top_n in itertools.product(
        ("W", "M"), (0.35, 0.50, 0.65), (63, 126), (50, 100, 200), (2, 3)
    ):
        label = f"{freq}_cyc{int(budget * 100)}_mom{momentum}_ma{trend}_top{top_n}"
        add(
            name=f"recovery_sector_barbell_{label}",
            family="Recovery-oriented sector barbell",
            independence_group="recovery_sector_barbell",
            description=(
                f"{freq} defensive/cyclical barbell; {budget:.0%} cyclical budget, "
                f"{momentum}d momentum, {trend}d trend, top {top_n}."
            ),
            assets=base.SECTORS + ["SHY"],
            target_builder=lambda panel, freq=freq, budget=budget, momentum=momentum, trend=trend, top_n=top_n: (
                _sector_recovery_targets(
                    panel,
                    freq=freq,
                    cyclical_budget=budget,
                    momentum_lookback=momentum,
                    trend_lookback=trend,
                    top_n=top_n,
                )
            ),
        )
    return runs


def _existing_duration_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_parquet(OUT / "all_strategy_returns.parquet")
    metrics = pd.read_csv(OUT / "all_strategy_metrics.csv")
    duration_fields = set(base._drawdown_duration_metrics(pd.Series(dtype=float)))
    metrics = metrics.drop(columns=list(duration_fields | {"max_recovery_days"}), errors="ignore")
    duration = pd.DataFrame(
        [
            {"strategy": name, **base._drawdown_duration_metrics(returns[name])}
            for name in returns.columns
        ]
    )
    return metrics.merge(duration, on="strategy", validate="one_to_one"), returns


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    adjusted, _ = base._download_prices()
    if adjusted.index.max() > base.LAST_ALLOWED:
        raise AssertionError("post-2021 observation entered the recovery study")

    existing, existing_returns = _existing_duration_table()
    variants = _recovery_variants(adjusted)
    shy_returns = adjusted["SHY"].pct_change().fillna(0.0)
    variant_metrics = pd.DataFrame([base._metric_row(run, shy_returns) for run in variants])
    variant_metrics["duration_hard_pass"] = (
        variant_metrics["n_5pct_drawdowns"].gt(0)
        & variant_metrics["max_5pct_trough_to_recovery_days"].le(20)
        & variant_metrics["n_unrecovered_5pct_drawdowns"].eq(0)
    )
    variant_metrics["objective_gate"] = (
        variant_metrics["years"].ge(14.0)
        & variant_metrics["cagr"].ge(0.09)
        & variant_metrics["stress_cagr"].ge(0.085)
        & variant_metrics["annual_vol"].le(0.15)
        & variant_metrics["max_drawdown"].ge(-0.25)
        & variant_metrics["cagr_2013_2021"].ge(0.08)
    )
    variant_metrics["joint_gate"] = (
        variant_metrics["duration_hard_pass"] & variant_metrics["objective_gate"]
    )
    variant_metrics = variant_metrics.sort_values(
        [
            "joint_gate",
            "objective_gate",
            "max_5pct_trough_to_recovery_days",
            "median_5pct_trough_to_recovery_days",
            "cagr",
        ],
        ascending=[False, False, True, True, False],
    )

    combined = pd.concat(
        [
            existing.assign(suite="original_104"),
            variant_metrics.assign(suite="recovery_sweep"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined["objective_gate"] = combined["objective_gate"].fillna(
        combined["target_gate"].fillna(False)
    )
    frontier = combined[
        combined["years"].ge(13.5)
        & combined["cagr"].ge(0.09)
        & combined["annual_vol"].le(0.15)
    ].sort_values(
        ["max_5pct_trough_to_recovery_days", "median_5pct_trough_to_recovery_days", "cagr"],
        ascending=[True, True, False],
    )

    variant_returns = pd.concat([run.returns.rename(run.name) for run in variants], axis=1)
    episode_names = list(
        dict.fromkeys(
            [
                "credit_canary_equity_allocation",
                "QQQ_vol_target_12",
                "cross_asset_mom_126_top3",
                "sector_low_ulcer_positive_trend",
                "sector_downside_trend",
                "SPY_trend_150d",
                *frontier.head(12)["strategy"].tolist(),
            ]
        )
    )
    episode_rows = []
    for name in episode_names:
        source = variant_returns if name in variant_returns else existing_returns
        episode_rows.append(_drawdown_episodes(name, source[name]))
    episodes = pd.concat(episode_rows, ignore_index=True)

    existing.to_csv(OUT / "drawdown_recovery_existing_metrics.csv", index=False)
    variant_metrics.to_csv(OUT / "drawdown_recovery_variant_metrics.csv", index=False)
    frontier.to_csv(OUT / "drawdown_recovery_frontier.csv", index=False)
    episodes.to_csv(OUT / "drawdown_recovery_episodes.csv", index=False)
    variant_returns.to_parquet(OUT / "drawdown_recovery_variant_returns.parquet")
    meta = {
        "original_strategy_count": len(existing),
        "recovery_variant_count": len(variant_metrics),
        "hard_20d_pass_count": int(variant_metrics["duration_hard_pass"].sum()),
        "objective_gate_count": int(variant_metrics["objective_gate"].sum()),
        "joint_gate_count": int(variant_metrics["joint_gate"].sum()),
        "last_observation": str(adjusted.index.max().date()),
    }
    (OUT / "drawdown_recovery_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(
        frontier[
            [
                "strategy",
                "suite",
                "cagr",
                "annual_vol",
                "max_drawdown",
                "max_underwater_days",
                "max_5pct_trough_to_recovery_days",
                "median_5pct_trough_to_recovery_days",
                "share_5pct_recovered_within_20d",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
