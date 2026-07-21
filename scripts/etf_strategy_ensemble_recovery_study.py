"""Combine frozen ETF strategy sleeves and test a relaxed 5% return floor."""

from __future__ import annotations

import itertools
import json

import etf_strategy_50plus_study as base
import numpy as np
import pandas as pd

from alpha_lab.backtest.vector import _solve_trade_cost, run_drift_backtest

OUT = base.OUT
PROTOCOL = (
    base.PROJECT_ROOT
    / "docs"
    / "research_decisions"
    / "2026-07-19_etf-ensemble-recovery-protocol.md"
)

FULL_GFC_SLEEVES = [
    "QQQ_vol_target_12",
    "cross_asset_mom_126_top3",
    "sector_low_ulcer_positive_trend",
    "SPY_trend_150d",
    "retail_all_weather_fixed",
    "all_weather_sleeve_trend_200",
    "SPY_95_105_collar",
    "recovery_sector_barbell_M_cyc65_mom63_ma100_top2",
]
CREDIT_SLEEVE = "credit_canary_equity_allocation"
SCALES = (0.25, 0.50, 0.75, 1.00)
PRIMARY_META_BPS = 5.0
STRESS_META_BPS = 10.0


def _monthly_decisions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    decisions = base._period_ends(index, "M")
    return decisions[decisions < index.max()]


def _constant_targets(
    index: pd.DatetimeIndex,
    columns: pd.Index,
    sleeves: tuple[str, ...],
    scale: float,
) -> pd.DataFrame:
    decisions = _monthly_decisions(index)
    target = pd.Series(0.0, index=columns)
    target.loc[list(sleeves)] = scale / len(sleeves)
    target["SHY"] = 1.0 - scale
    return pd.DataFrame(
        np.tile(target.to_numpy(), (len(decisions), 1)),
        index=decisions,
        columns=columns,
    )


def _inverse_vol_targets(
    returns: pd.DataFrame,
    sleeves: list[str],
    *,
    lookback: int,
    scale: float,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for date in _monthly_decisions(returns.index):
        window = returns.loc[:date, sleeves].tail(lookback)
        row = pd.Series(0.0, index=returns.columns, name=date)
        if len(window) >= lookback:
            inverse = (1.0 / window.std().replace(0.0, np.nan)).dropna()
            if len(inverse) == len(sleeves):
                row.loc[sleeves] = inverse / inverse.sum() * scale
        row["SHY"] = 1.0 - float(row.sum())
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=returns.columns, fill_value=0.0)


def _fast_monthly_drift(
    asset_returns: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    primary_bps: float,
    stress_bps: float,
) -> tuple[pd.Series, pd.Series, float, pd.Timestamp]:
    """Run exact drift/cost mechanics without constructing a large result object."""
    columns = asset_returns.columns
    weights = np.zeros(len(columns), dtype=float)
    weights[columns.get_loc("SHY")] = 1.0
    scheduled: dict[int, np.ndarray] = {}
    for date, target in targets.reindex(columns=columns).iterrows():
        decision_position = asset_returns.index.get_loc(date)
        if decision_position + 1 < len(asset_returns):
            scheduled[decision_position + 1] = target.to_numpy(dtype=float)
    if not scheduled:
        raise ValueError("ensemble target schedule produced no trades")

    primary = np.zeros(len(asset_returns), dtype=float)
    stress = np.zeros(len(asset_returns), dtype=float)
    traded_notional = 0.0
    primary_rate = primary_bps / 10_000.0
    stress_rate = stress_bps / 10_000.0
    values = asset_returns.to_numpy(dtype=float)
    for position, daily_returns in enumerate(values):
        pre_values = weights * (1.0 + daily_returns)
        gross_factor = float(pre_values.sum())
        pre_weights = pre_values / gross_factor
        target = scheduled.get(position)
        primary_cost = stress_cost = 0.0
        if target is not None:
            primary_cost = _solve_trade_cost(1.0, pre_weights, target, primary_rate)
            stress_cost = _solve_trade_cost(1.0, pre_weights, target, stress_rate)
            traded_notional += float(np.abs(target * (1.0 - primary_cost) - pre_weights).sum())
            weights = target
        else:
            weights = pre_weights
        primary[position] = gross_factor * (1.0 - primary_cost) - 1.0
        stress[position] = gross_factor * (1.0 - stress_cost) - 1.0

    first_trade = asset_returns.index[min(scheduled)]
    sample_years = len(asset_returns.loc[first_trade:]) / base.PERIODS
    annual_turnover = traded_notional / max(sample_years, 1e-12)
    return (
        pd.Series(primary, index=asset_returns.index).loc[first_trade:],
        pd.Series(stress, index=asset_returns.index).loc[first_trade:],
        annual_turnover,
        first_trade,
    )


def _validate_fast_engine(asset_returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Check the optimized ensemble engine against the canonical drift engine."""
    primary, _, _, first_trade = _fast_monthly_drift(
        asset_returns,
        targets,
        primary_bps=PRIMARY_META_BPS,
        stress_bps=STRESS_META_BPS,
    )
    prices = (1.0 + asset_returns).cumprod()
    initial = pd.Series(0.0, index=prices.columns, name=prices.index[0])
    initial["SHY"] = 1.0
    canonical_targets = pd.concat([initial.to_frame().T, targets])
    canonical = run_drift_backtest(
        canonical_targets,
        prices,
        trading_bps=PRIMARY_META_BPS,
        execution_delay_bars=1,
    ).returns.loc[first_trade:]
    if not np.allclose(primary, canonical, atol=1e-12, rtol=1e-10):
        difference = float((primary - canonical).abs().max())
        raise AssertionError(f"fast ensemble engine differs from canonical engine: {difference}")


def _load_sleeve_returns() -> tuple[pd.DataFrame, pd.Series]:
    original = pd.read_parquet(OUT / "all_strategy_returns.parquet")
    recovery = pd.read_parquet(OUT / "drawdown_recovery_variant_returns.parquet")
    combined = pd.concat([original, recovery], axis=1)
    required = [*FULL_GFC_SLEEVES, CREDIT_SLEEVE]
    missing = sorted(set(required) - set(combined.columns))
    if missing:
        raise KeyError(f"required frozen sleeves are missing: {missing}")
    adjusted = pd.read_parquet(OUT / "market_prices_adjusted_pre2022.parquet")
    if combined.index.max() > base.LAST_ALLOWED or adjusted.index.max() > base.LAST_ALLOWED:
        raise AssertionError("post-2021 data entered the ensemble study")
    return combined[required], adjusted["SHY"].pct_change().fillna(0.0)


def _tier_returns(
    sleeve_returns: pd.DataFrame,
    shy_returns: pd.Series,
    sleeves: list[str],
) -> pd.DataFrame:
    aligned = sleeve_returns[sleeves].dropna(how="any")
    aligned = aligned.loc[base.CORE_START : base.CORE_END]
    aligned["SHY"] = shy_returns.reindex(aligned.index).fillna(0.0)
    return aligned


def _evaluate(
    *,
    name: str,
    tier: str,
    description: str,
    asset_returns: pd.DataFrame,
    targets: pd.DataFrame,
    shy_returns: pd.Series,
) -> tuple[dict[str, object], pd.Series]:
    primary, stress, turnover, _ = _fast_monthly_drift(
        asset_returns,
        targets,
        primary_bps=PRIMARY_META_BPS,
        stress_bps=STRESS_META_BPS,
    )
    run = base.StrategyRun(
        name=name,
        family="Monthly ETF strategy-sleeve ensemble",
        independence_group=tier,
        evidence=tier,
        description=description,
        returns=primary.rename(name),
        stress_returns=stress.rename(name),
        annual_turnover=turnover,
        latest_weights={
            key: float(value)
            for key, value in targets.iloc[-1].items()
            if value > 1e-6
        },
        implementation_note=(
            "Sleeve-level screen; underlying ETF orders require weight aggregation and netting"
        ),
    )
    row = base._metric_row(run, shy_returns)
    row["tier"] = tier
    return row, primary.rename(name)


def _classify(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["original_gate"] = (
        result["cagr"].ge(0.09)
        & result["stress_cagr"].ge(0.085)
        & result["annual_vol"].le(0.15)
        & result["max_drawdown"].ge(-0.25)
    )
    result["relaxed_5pct_gate"] = (
        result["cagr"].ge(0.05)
        & result["stress_cagr"].ge(0.045)
        & result["annual_vol"].le(0.15)
        & result["max_drawdown"].ge(-0.25)
    )
    result["avoided_5pct_drawdown"] = result["n_5pct_drawdowns"].eq(0)
    result["hard_20d_recovery"] = (
        result["n_5pct_drawdowns"].gt(0)
        & result["max_5pct_trough_to_recovery_days"].le(20)
        & result["n_unrecovered_5pct_drawdowns"].eq(0)
    )
    result["median_recovery_20d"] = result[
        "median_5pct_trough_to_recovery_days"
    ].le(20)
    result["majority_recovery_20d"] = result[
        "share_5pct_recovered_within_20d"
    ].ge(0.50)
    result["original_joint"] = result["original_gate"] & (
        result["avoided_5pct_drawdown"] | result["hard_20d_recovery"]
    )
    result["relaxed_joint"] = result["relaxed_5pct_gate"] & (
        result["avoided_5pct_drawdown"] | result["hard_20d_recovery"]
    )
    return result


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    sleeve_returns, shy_returns = _load_sleeve_returns()
    full_returns = _tier_returns(sleeve_returns, shy_returns, FULL_GFC_SLEEVES)
    credit_names = [*FULL_GFC_SLEEVES, CREDIT_SLEEVE]
    credit_returns = _tier_returns(sleeve_returns, shy_returns, credit_names)

    validation_targets = _constant_targets(
        full_returns.index,
        full_returns.columns,
        tuple(FULL_GFC_SLEEVES[:2]),
        0.75,
    )
    _validate_fast_engine(full_returns, validation_targets)

    metric_rows: list[dict[str, object]] = []
    returns_map: dict[str, pd.Series] = {}

    for size in range(2, 6):
        for sleeves in itertools.combinations(FULL_GFC_SLEEVES, size):
            short = "__".join(sleeves)
            for scale in SCALES:
                name = f"full__s{int(scale * 100)}__{short}"
                targets = _constant_targets(
                    full_returns.index, full_returns.columns, sleeves, scale
                )
                row, returns = _evaluate(
                    name=name,
                    tier="full_gfc_2007_2021",
                    description=f"Monthly equal-weight {sleeves}; {scale:.0%} risk, residual SHY.",
                    asset_returns=full_returns,
                    targets=targets,
                    shy_returns=shy_returns,
                )
                metric_rows.append(row)
                returns_map[name] = returns

    for size in range(1, 5):
        for partners in itertools.combinations(FULL_GFC_SLEEVES, size):
            sleeves = (CREDIT_SLEEVE, *partners)
            short = "__".join(sleeves)
            for scale in SCALES:
                name = f"credit__s{int(scale * 100)}__{short}"
                targets = _constant_targets(
                    credit_returns.index, credit_returns.columns, sleeves, scale
                )
                row, returns = _evaluate(
                    name=name,
                    tier="credit_2008_2021",
                    description=f"Monthly equal-weight {sleeves}; {scale:.0%} risk, residual SHY.",
                    asset_returns=credit_returns,
                    targets=targets,
                    shy_returns=shy_returns,
                )
                metric_rows.append(row)
                returns_map[name] = returns

    for tier, asset_returns, sleeves in (
        ("full_gfc_2007_2021", full_returns, FULL_GFC_SLEEVES),
        ("credit_2008_2021", credit_returns, credit_names),
    ):
        for lookback, scale in itertools.product((63, 126), SCALES):
            name = f"{tier}__invvol{lookback}__s{int(scale * 100)}"
            targets = _inverse_vol_targets(
                asset_returns,
                sleeves,
                lookback=lookback,
                scale=scale,
            )
            row, returns = _evaluate(
                name=name,
                tier=tier,
                description=(
                    f"Monthly inverse-vol all-pool diagnostic, {lookback}d risk, "
                    f"{scale:.0%} allocation; residual SHY."
                ),
                asset_returns=asset_returns,
                targets=targets,
                shy_returns=shy_returns,
            )
            metric_rows.append(row)
            returns_map[name] = returns

    metrics = _classify(pd.DataFrame(metric_rows))
    metrics = metrics.sort_values(
        [
            "relaxed_joint",
            "original_joint",
            "relaxed_5pct_gate",
            "max_5pct_trough_to_recovery_days",
            "median_5pct_trough_to_recovery_days",
            "cagr",
        ],
        ascending=[False, False, False, True, True, False],
    )
    rank_names = list(
        dict.fromkeys(
            [
                *metrics[metrics["original_gate"]].head(30)["strategy"].tolist(),
                *metrics[metrics["relaxed_5pct_gate"]].head(30)["strategy"].tolist(),
                *metrics[metrics["avoided_5pct_drawdown"]].head(30)["strategy"].tolist(),
            ]
        )
    )
    selected_returns = pd.concat([returns_map[name] for name in rank_names], axis=1)

    meta = {
        "trial_count": len(metrics),
        "full_gfc_trial_count": int(metrics["tier"].eq("full_gfc_2007_2021").sum()),
        "credit_trial_count": int(metrics["tier"].eq("credit_2008_2021").sum()),
        "original_gate_count": int(metrics["original_gate"].sum()),
        "relaxed_5pct_gate_count": int(metrics["relaxed_5pct_gate"].sum()),
        "avoided_5pct_drawdown_count": int(metrics["avoided_5pct_drawdown"].sum()),
        "hard_20d_recovery_count": int(metrics["hard_20d_recovery"].sum()),
        "original_joint_count": int(metrics["original_joint"].sum()),
        "relaxed_joint_count": int(metrics["relaxed_joint"].sum()),
        "last_observation": str(max(full_returns.index.max(), credit_returns.index.max()).date()),
    }
    metrics.to_csv(OUT / "ensemble_recovery_metrics.csv", index=False)
    selected_returns.to_parquet(OUT / "ensemble_recovery_selected_returns.parquet")
    (OUT / "ensemble_recovery_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    columns = [
        "strategy",
        "tier",
        "cagr",
        "stress_cagr",
        "annual_vol",
        "max_drawdown",
        "max_underwater_days",
        "max_5pct_trough_to_recovery_days",
        "median_5pct_trough_to_recovery_days",
        "share_5pct_recovered_within_20d",
        "n_5pct_drawdowns",
        "original_gate",
        "relaxed_5pct_gate",
    ]
    print("\nTop original-gate ensembles:\n")
    print(metrics[metrics["original_gate"]][columns].head(15).to_string(index=False))
    print("\nTop relaxed-gate ensembles:\n")
    print(metrics[metrics["relaxed_5pct_gate"]][columns].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
