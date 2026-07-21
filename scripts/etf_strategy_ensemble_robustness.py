"""Post-selection robustness checks for the pre-2022 ETF ensemble study."""

from __future__ import annotations

import itertools

import etf_strategy_50plus_study as base
import etf_strategy_ensemble_recovery_study as ensemble
import numpy as np
import pandas as pd

FAST_TYPICAL = (
    "cross_asset_mom_126_top3",
    "recovery_sector_barbell_M_cyc65_mom63_ma100_top2",
)
FAST_WORST = (
    "retail_all_weather_fixed",
    "recovery_sector_barbell_M_cyc65_mom63_ma100_top2",
)
RELAXED_CORE = (
    "QQQ_vol_target_12",
    "sector_low_ulcer_positive_trend",
    "retail_all_weather_fixed",
    "all_weather_sleeve_trend_200",
)


def _fixed_targets(
    index: pd.DatetimeIndex,
    columns: pd.Index,
    weights: dict[str, float],
    *,
    frequency: str,
) -> pd.DataFrame:
    decisions = base._period_ends(index, frequency)
    decisions = decisions[decisions < index.max()]
    row = pd.Series(0.0, index=columns)
    row.loc[list(weights)] = pd.Series(weights)
    row["SHY"] = 1.0 - float(row.sum())
    if (row < -1e-12).any() or not np.isclose(row.sum(), 1.0):
        raise ValueError("robustness target must be long-only and sum to one")
    return pd.DataFrame(
        np.tile(row.to_numpy(), (len(decisions), 1)),
        index=decisions,
        columns=columns,
    )


def _evaluate(
    *,
    name: str,
    candidate: str,
    frequency: str,
    asset_returns: pd.DataFrame,
    targets: pd.DataFrame,
    shy_returns: pd.Series,
    parameter: str,
) -> dict[str, object]:
    primary, stress, turnover, _ = ensemble._fast_monthly_drift(
        asset_returns,
        targets,
        primary_bps=5.0,
        stress_bps=20.0,
    )
    run = base.StrategyRun(
        name=name,
        family="ETF ensemble post-selection robustness",
        independence_group=candidate,
        evidence="post_selection_diagnostic",
        description=parameter,
        returns=primary.rename(name),
        stress_returns=stress.rename(name),
        annual_turnover=turnover,
        latest_weights={
            key: float(value)
            for key, value in targets.iloc[-1].items()
            if value > 1e-6
        },
        implementation_note="Post-selection sensitivity; not an independent candidate",
    )
    row = base._metric_row(run, shy_returns)
    row.update(
        {
            "candidate": candidate,
            "frequency": frequency,
            "parameter": parameter,
            "stress_meta_bps": 20.0,
        }
    )
    return row


def main() -> None:
    sleeve_returns, shy_returns = ensemble._load_sleeve_returns()
    full_returns = ensemble._tier_returns(
        sleeve_returns, shy_returns, ensemble.FULL_GFC_SLEEVES
    )
    rows: list[dict[str, object]] = []

    for candidate, sleeves in (
        ("fast_typical", FAST_TYPICAL),
        ("fast_worst", FAST_WORST),
    ):
        for first_weight, frequency in itertools.product(
            (0.30, 0.40, 0.50, 0.60, 0.70), ("M", "Q")
        ):
            weights = {
                sleeves[0]: first_weight,
                sleeves[1]: 1.0 - first_weight,
            }
            targets = _fixed_targets(
                full_returns.index,
                full_returns.columns,
                weights,
                frequency=frequency,
            )
            rows.append(
                _evaluate(
                    name=f"{candidate}_{frequency}_{int(first_weight * 100)}",
                    candidate=candidate,
                    frequency=frequency,
                    asset_returns=full_returns,
                    targets=targets,
                    shy_returns=shy_returns,
                    parameter=f"first sleeve {first_weight:.0%}; second {1-first_weight:.0%}",
                )
            )

    for scale, frequency in itertools.product(
        (0.40, 0.45, 0.50, 0.55, 0.60), ("M", "Q")
    ):
        weights = {sleeve: scale / len(RELAXED_CORE) for sleeve in RELAXED_CORE}
        targets = _fixed_targets(
            full_returns.index,
            full_returns.columns,
            weights,
            frequency=frequency,
        )
        rows.append(
            _evaluate(
                name=f"relaxed_5pct_{frequency}_{int(scale * 100)}",
                candidate="relaxed_5pct",
                frequency=frequency,
                asset_returns=full_returns,
                targets=targets,
                shy_returns=shy_returns,
                parameter=f"risky ensemble {scale:.0%}; SHY {1-scale:.0%}",
            )
        )

    result = pd.DataFrame(rows)
    result["original_gate_20bp"] = (
        result["cagr"].ge(0.09)
        & result["stress_cagr"].ge(0.085)
        & result["annual_vol"].le(0.15)
        & result["max_drawdown"].ge(-0.25)
    )
    result["relaxed_gate_20bp"] = (
        result["cagr"].ge(0.05)
        & result["stress_cagr"].ge(0.045)
        & result["annual_vol"].le(0.15)
        & result["max_drawdown"].ge(-0.05)
    )
    result.to_csv(ensemble.OUT / "ensemble_recovery_robustness.csv", index=False)
    print(
        result[
            [
                "candidate",
                "frequency",
                "parameter",
                "cagr",
                "stress_cagr",
                "annual_vol",
                "max_drawdown",
                "max_underwater_days",
                "max_5pct_trough_to_recovery_days",
                "median_5pct_trough_to_recovery_days",
                "share_5pct_recovered_within_20d",
                "n_5pct_drawdowns",
                "original_gate_20bp",
                "relaxed_gate_20bp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
