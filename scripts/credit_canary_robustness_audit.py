"""Destructive robustness diagnostics for the pre-2022 credit-canary candidate.

This script is intentionally post-selection. Its variants cannot replace the frozen
200-day weekly rule; they only test whether that result is an artifact of one threshold,
one timing convention, or an unusually favorable average exposure.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from etf_strategy_50plus_study import (
    CORE_END,
    PRIMARY_TRADING_BPS,
    STRESS_TRADING_BPS,
    _decision_dates,
    _fixed_targets,
    _safe_row,
    _targets_from_rows,
)

from alpha_lab.analytics.returns import drawdown
from alpha_lab.backtest.vector import run_drift_backtest
from alpha_lab.utils.paths import PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "results" / "etf_strategy_50plus_pre2022"
PRICE_PATH = OUT / "market_prices_adjusted_pre2022.parquet"
RESULT_PATH = OUT / "credit_canary_robustness.csv"
YEAR_PATH = OUT / "credit_canary_yearly.csv"

ASSETS = ["SPY", "QQQ", "HYG", "IEF", "GLD", "SHY"]
RISK_ON = {"SPY": 0.75, "QQQ": 0.25}
RISK_OFF = {"IEF": 0.50, "GLD": 0.25, "SHY": 0.25}


def build_targets(
    panel: pd.DataFrame,
    *,
    lookback: int = 200,
    frequency: str = "W",
    use_credit: bool = True,
    use_spy: bool = True,
) -> pd.DataFrame:
    """Build an ablation/sensitivity target set without changing sleeve weights."""
    credit_ratio = panel["HYG"] / panel["IEF"]
    credit_ma = credit_ratio.rolling(lookback, min_periods=lookback).mean()
    spy_ma = panel["SPY"].rolling(lookback, min_periods=lookback).mean()
    rows = []
    for date in _decision_dates(panel.index, frequency):
        needed = []
        if use_credit:
            needed.append(credit_ma.loc[date])
        if use_spy:
            needed.append(spy_ma.loc[date])
        if any(pd.isna(value) for value in needed):
            continue
        credit_good = not use_credit or credit_ratio.loc[date] > credit_ma.loc[date]
        spy_good = not use_spy or panel.loc[date, "SPY"] > spy_ma.loc[date]
        allocation = RISK_ON if credit_good and spy_good else RISK_OFF
        row = _safe_row(panel.columns, date)
        row[:] = 0.0
        for ticker, weight in allocation.items():
            row[ticker] = weight
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    return float(np.exp(np.log1p(clean).mean() * 252.0) - 1.0)


def metrics(
    name: str,
    returns: pd.Series,
    stress_returns: pd.Series,
    turnover: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    sample = returns.loc[start:end]
    stress = stress_returns.reindex(sample.index).fillna(0.0)
    dd = drawdown(sample)
    calendar = sample.groupby(sample.index.year).apply(lambda values: (1.0 + values).prod() - 1.0)
    return {
        "diagnostic": name,
        "start": sample.index.min().date().isoformat(),
        "end": sample.index.max().date().isoformat(),
        "cagr": annualized_return(sample),
        "stress_cagr": annualized_return(stress),
        "annual_vol": float(sample.std() * np.sqrt(252.0)),
        "max_drawdown": float(dd.min()),
        "ulcer_index": float(np.sqrt(dd.pow(2).mean())),
        "worst_year": float(calendar.min()),
        "cagr_2008_2012": annualized_return(sample.loc["2008":"2012"]),
        "cagr_2013_2016": annualized_return(sample.loc["2013":"2016"]),
        "cagr_2017_2021": annualized_return(sample.loc["2017":"2021"]),
        "late_gfc_return": float((1.0 + sample.loc[start:"2009-03-09"]).prod() - 1.0),
        "annual_turnover": float(turnover.loc[start:end].sum() / (len(sample) / 252.0)),
        "risk_on_share": np.nan,
    }


def run_targets(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    execution_delay_bars: int = 1,
) -> tuple:
    primary = run_drift_backtest(
        targets,
        panel,
        trading_bps=PRIMARY_TRADING_BPS,
        execution_delay_bars=execution_delay_bars,
    )
    stress = run_drift_backtest(
        targets,
        panel,
        trading_bps=STRESS_TRADING_BPS,
        execution_delay_bars=execution_delay_bars,
    )
    return primary, stress


def main() -> None:
    adjusted = pd.read_parquet(PRICE_PATH)
    if adjusted.index.max() > pd.Timestamp("2021-12-31"):
        raise AssertionError("post-2021 price entered the audit")
    panel = adjusted[ASSETS].dropna(how="any")
    definitions = {
        "frozen_both_200_weekly": build_targets(panel),
        "ablation_spy_only_200_weekly": build_targets(panel, use_credit=False),
        "ablation_credit_only_200_weekly": build_targets(panel, use_spy=False),
        "sensitivity_both_150_weekly": build_targets(panel, lookback=150),
        "sensitivity_both_250_weekly": build_targets(panel, lookback=250),
        "sensitivity_both_200_monthly": build_targets(panel, frequency="M"),
    }
    own_results = {}
    for name, targets in definitions.items():
        own_results[name] = run_targets(panel, targets)
    own_results["timing_both_200_extra_week_delay"] = run_targets(
        panel,
        definitions["frozen_both_200_weekly"],
        execution_delay_bars=5,
    )

    first_trade_dates = [result[0].decision_to_trade.min() for result in own_results.values()]
    common_start = max(pd.Timestamp("2008-05-01"), max(first_trade_dates))
    common_start = panel.index[panel.index.searchsorted(common_start)]

    frozen_targets = definitions["frozen_both_200_weekly"]
    dense = frozen_targets.reindex(panel.index).ffill().loc[common_start:CORE_END]
    mean_target = dense.mean().reindex(panel.columns, fill_value=0.0)
    mean_target = mean_target / mean_target.sum()
    static_average_targets = _fixed_targets(panel, mean_target.to_dict(), freq="Q")
    own_results["static_average_exposure"] = run_targets(panel, static_average_targets)
    own_results["always_risk_on"] = run_targets(panel, _fixed_targets(panel, RISK_ON, freq="Q"))
    own_results["always_risk_off"] = run_targets(panel, _fixed_targets(panel, RISK_OFF, freq="Q"))

    rows = []
    yearly = []
    for name, (primary, stress) in own_results.items():
        row = metrics(
            name,
            primary.returns,
            stress.returns,
            primary.traded_notional,
            common_start,
            CORE_END,
        )
        if name not in {"static_average_exposure", "always_risk_on", "always_risk_off"}:
            weights = primary.weights.loc[common_start:CORE_END]
            row["risk_on_share"] = float((weights[["SPY", "QQQ"]].sum(axis=1) > 0.50).mean())
        rows.append(row)
        annual = primary.returns.loc[common_start:CORE_END].groupby(
            primary.returns.loc[common_start:CORE_END].index.year
        ).apply(lambda values: (1.0 + values).prod() - 1.0)
        for year, value in annual.items():
            yearly.append({"diagnostic": name, "year": int(year), "return": float(value)})

    result = pd.DataFrame(rows).set_index("diagnostic").sort_values("cagr", ascending=False)
    result.to_csv(RESULT_PATH)
    pd.DataFrame(yearly).to_csv(YEAR_PATH, index=False)
    metadata = {
        "common_start": str(common_start.date()),
        "end": str(CORE_END.date()),
        "frozen_rule": "both HYG/IEF and SPY above trailing 200d MA, weekly",
        "static_average_weights": {ticker: float(weight) for ticker, weight in mean_target.items()},
        "post_selection_warning": "Sensitivity winners cannot replace the frozen rule.",
    }
    (OUT / "credit_canary_robustness_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(result.to_string())


if __name__ == "__main__":
    main()
