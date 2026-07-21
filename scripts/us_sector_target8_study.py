"""Pre-holdout 8% target-return / low-drawdown US sector study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from us_sector_rotation_study import (
    DATA_END_EXCLUSIVE,
    HOLDOUT_START,
    LEGACY_SECTORS,
    VALIDATION_END,
    VALIDATION_START,
    annualized_return,
    circular_block_bootstrap,
    load_cash_returns,
    load_data,
    monthly_return,
)

from alpha_lab.analytics.returns import drawdown
from alpha_lab.backtest.collar import SyntheticCollarResult, run_synthetic_collar
from alpha_lab.backtest.vector import DriftBacktestResult, run_drift_backtest
from alpha_lab.data.calendars import rebalance_dates
from alpha_lab.data.loaders.fred import load_series
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.stats.tests import deflated_sharpe_ratio
from alpha_lab.utils.paths import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "us_sector_target8_low_drawdown.html"
RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "us_sector_target8_metrics.csv"
TRAIN_END = pd.Timestamp("2012-12-31")
PRIMARY_BPS = 10.0
STRESS_BPS = 20.0
TARGET_RETURN = 0.08
TARGET_FLOOR = 0.075
TARGET_VOL_MAX = 0.10
TARGET_MAX_DRAWDOWN = -0.15
TARGET_CALMAR = 0.60
LIFETIME_RISK_TRIALS = 11


@dataclass(frozen=True)
class SectorCandidate:
    """Frozen targets and primary/stress drift-aware backtests."""

    name: str
    targets: pd.DataFrame
    primary: DriftBacktestResult
    stress: DriftBacktestResult


def main() -> None:
    prices, _ = load_data()
    legacy = prices[LEGACY_SECTORS].dropna()
    spy = prices["SPY"].reindex(legacy.index)
    cash = load_cash_returns(legacy.index)
    raw_spy, vix, annual_rates = load_collar_inputs(legacy.index)

    collar = run_synthetic_collar(
        spy,
        raw_spy,
        vix,
        cash,
        annual_rates,
    )
    targets = build_all_targets(legacy, spy, vix)
    candidates = {
        name: build_candidate(name, target, legacy, cash)
        for name, target in targets.items()
    }
    benchmark_returns = build_benchmarks(spy, cash, collar)
    metrics = build_metrics(candidates, benchmark_returns, cash)
    collar_metrics = metrics.loc["synthetic_spy_collar"]
    metrics["passes_all_gates"] = apply_gates(metrics, collar_metrics)
    regimes = build_regime_table(candidates, benchmark_returns, spy, vix)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_report(metrics, regimes, candidates, benchmark_returns, collar),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(metrics.to_string())


def load_collar_inputs(
    index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Load raw SPY spot plus VIX/DTB3, all hard-stopped before 2022."""
    cache_dir = Path(gettempdir()) / "alpha_lab_yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    raw = load_prices(
        "SPY",
        "1998-12-01",
        DATA_END_EXCLUSIVE,
        field="Close",
        auto_adjust=False,
        threads=False,
    )["SPY"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    macro = load_series(["VIXCLS", "DTB3"], start="1998-12-01", end="2021-12-31")
    if raw.empty or macro.empty or raw.index.max() >= HOLDOUT_START or macro.index.max() >= HOLDOUT_START:
        raise RuntimeError("collar inputs are empty or crossed the sealed 2022 holdout")
    timeline = macro.index.union(index).sort_values()
    aligned = macro.reindex(timeline).ffill().reindex(index)
    vix = aligned["VIXCLS"].rename("VIX")
    annual_rates = (aligned["DTB3"] / 100.0).shift(1).rename("annual_rate")
    return raw.reindex(index).ffill(), vix, annual_rates


def downside_deviation(returns: pd.DataFrame | pd.Series, window: int = 126):
    """Trailing annualized lower-partial-moment deviation around zero."""
    downside = returns.clip(upper=0.0).pow(2)
    return np.sqrt(downside.rolling(window, min_periods=window).mean()) * np.sqrt(252)


def _allocate_inverse_risk(
    risk: pd.Series,
    eligible: pd.Series,
    *,
    budget: float,
    cap: float,
) -> pd.Series:
    selected = risk[eligible].replace([np.inf, -np.inf, 0.0], np.nan).dropna()
    if selected.empty:
        return pd.Series(0.0, index=risk.index)
    inverse = 1.0 / selected
    weights = budget * inverse / inverse.sum()
    return weights.clip(upper=cap).reindex(risk.index, fill_value=0.0)


def _with_cash(weights: pd.Series, date: pd.Timestamp) -> pd.Series:
    result = weights.copy().rename(date)
    result["CASH"] = max(0.0, 1.0 - float(weights.sum()))
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError(f"target at {date} does not sum to one")
    return result


def c1_smelted_targets(prices: pd.DataFrame) -> pd.DataFrame:
    dates = rebalance_dates(prices.index, freq="ME")
    risk = downside_deviation(prices.pct_change())
    trend = prices > prices.rolling(200, min_periods=200).mean()
    rows = []
    for date in dates:
        if risk.loc[date].isna().all():
            continue
        weights = _allocate_inverse_risk(risk.loc[date], trend.loc[date], budget=1.0, cap=0.25)
        rows.append(_with_cash(weights, date))
    return pd.DataFrame(rows).reindex(columns=[*prices.columns, "CASH"], fill_value=0.0)


def c2_ferry_targets(
    prices: pd.DataFrame,
    spy: pd.Series,
    vix: pd.Series,
) -> pd.DataFrame:
    dates = rebalance_dates(prices.index, freq="W-FRI")
    risk = downside_deviation(prices.pct_change())
    momentum = prices.pct_change(126)
    spy_trend = spy > spy.rolling(200, min_periods=200).mean()
    observed_risk_on = spy_trend & (vix < 30.0)
    active = False
    consecutive = 0
    rows = []
    for date in dates:
        if risk.loc[date].isna().all() or pd.isna(observed_risk_on.loc[date]):
            continue
        if not bool(observed_risk_on.loc[date]):
            active = False
            consecutive = 0
        elif not active:
            consecutive += 1
            if consecutive >= 4:
                active = True
        if active:
            winners = momentum.loc[date].nlargest(4).index
            eligible = pd.Series(prices.columns.isin(winners), index=prices.columns)
            weights = _allocate_inverse_risk(
                risk.loc[date],
                eligible,
                budget=1.0,
                cap=0.30,
            )
        else:
            weights = pd.Series(0.0, index=prices.columns)
            weights.loc[["XLP", "XLV", "XLU"]] = 1.0 / 6.0
        rows.append(_with_cash(weights, date))
    return pd.DataFrame(rows).reindex(columns=[*prices.columns, "CASH"], fill_value=0.0)


def c3_woven_targets(prices: pd.DataFrame) -> pd.DataFrame:
    dates = rebalance_dates(prices.index, freq="ME")
    returns = prices.pct_change()
    risk = downside_deviation(returns)
    momentum = prices.pct_change(126)
    trend = prices > prices.rolling(200, min_periods=200).mean()
    defensive = ["XLP", "XLV", "XLU"]
    cyclical = ["XLK", "XLF", "XLI", "XLY", "XLE", "XLB"]
    rows = []
    for date in dates:
        if risk.loc[date].isna().all():
            continue
        defensive_eligible = pd.Series(prices.columns.isin(defensive), index=prices.columns)
        weights = _allocate_inverse_risk(
            risk.loc[date],
            defensive_eligible,
            budget=0.50,
            cap=0.25,
        )
        ranked = momentum.loc[date, cyclical].nlargest(3).index
        for ticker in ranked:
            if bool(trend.loc[date, ticker]):
                weights.loc[ticker] += 1.0 / 6.0
        rows.append(_with_cash(weights, date))
    return pd.DataFrame(rows).reindex(columns=[*prices.columns, "CASH"], fill_value=0.0)


def c4_downside_budget_targets(prices: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    base = c1_smelted_targets(prices)
    returns = prices.pct_change()
    spy_trend = spy > spy.rolling(200, min_periods=200).mean()
    rows = []
    for date, target in base.iterrows():
        sector_weights = target[prices.columns]
        if sector_weights.sum() <= 0.0:
            rows.append(target)
            continue
        normalized = sector_weights / sector_weights.sum()
        sleeve = returns.loc[:date].tail(126).mul(normalized, axis=1).sum(axis=1)
        downside = float(np.sqrt(sleeve.clip(upper=0.0).pow(2).mean()) * np.sqrt(252))
        equity_budget = min(1.0, 0.08 / downside) if downside > 0.0 else 1.0
        if not bool(spy_trend.loc[date]):
            equity_budget = min(equity_budget, 0.50)
        scaled = sector_weights * equity_budget
        rows.append(_with_cash(scaled, date))
    return pd.DataFrame(rows).reindex(columns=[*prices.columns, "CASH"], fill_value=0.0)


def c5_low_ulcer_targets(prices: pd.DataFrame) -> pd.DataFrame:
    dates = rebalance_dates(prices.index, freq="ME")
    trend = prices > prices.rolling(200, min_periods=200).mean()
    positive = prices.pct_change(252) > 0.0
    rows = []
    for date in dates:
        window = prices.loc[:date].tail(252)
        if len(window) < 252:
            continue
        drawdowns = window / window.cummax() - 1.0
        ulcer = np.sqrt(drawdowns.pow(2).mean())
        eligible = trend.loc[date] & positive.loc[date]
        winners = ulcer[eligible].nsmallest(4).index
        weights = pd.Series(0.0, index=prices.columns)
        weights.loc[winners] = 0.25
        rows.append(_with_cash(weights, date))
    return pd.DataFrame(rows).reindex(columns=[*prices.columns, "CASH"], fill_value=0.0)


def build_all_targets(
    prices: pd.DataFrame,
    spy: pd.Series,
    vix: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Build the five preregistered target streams in one frozen batch."""
    return {
        "C1_smelted_trend_downside": c1_smelted_targets(prices),
        "C2_asymmetric_sector_ferry": c2_ferry_targets(prices, spy, vix),
        "C3_woven_defensive_upside": c3_woven_targets(prices),
        "C4_downside_vol_budget": c4_downside_budget_targets(prices, spy),
        "C5_low_ulcer_positive_trend": c5_low_ulcer_targets(prices),
    }


def build_candidate(
    name: str,
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    cash: pd.Series,
    *,
    released_through: pd.Timestamp | None = None,
) -> SectorCandidate:
    """Backtest one target stream at primary and stress trading costs."""
    latest = targets.index.max()
    if released_through is None and latest >= HOLDOUT_START:
        raise RuntimeError(f"{name} crossed the sealed holdout")
    if released_through is not None and latest > released_through:
        raise RuntimeError(f"{name} crossed the authorized release boundary {released_through.date()}")
    primary = run_drift_backtest(
        targets,
        prices,
        trading_bps=PRIMARY_BPS,
        rebalance_threshold=0.02,
        cash_returns=cash,
    )
    stress = run_drift_backtest(
        targets,
        prices,
        trading_bps=STRESS_BPS,
        rebalance_threshold=0.02,
        cash_returns=cash,
    )
    return SectorCandidate(name, targets, primary, stress)


def build_benchmarks(
    spy: pd.Series,
    cash: pd.Series,
    collar: SyntheticCollarResult,
) -> dict[str, pd.Series]:
    """Return SPY, synthetic collar, and an unlevered 60/40 SPY/cash comparator."""
    spy_returns = spy.pct_change().fillna(0.0).rename("SPY")
    mix_target = pd.DataFrame({"SPY": [0.60], "CASH": [0.40]}, index=[spy.index[0]])
    mix = run_drift_backtest(
        mix_target,
        spy.to_frame("SPY"),
        trading_bps=0.0,
        cash_returns=cash,
    )
    return {
        "SPY": spy_returns,
        "synthetic_spy_collar": collar.returns,
        "spy_60_cash_40": mix.returns,
    }


def _rolling_target_share(returns: pd.Series, *, window: int = 1260) -> float:
    log_returns = np.log1p(returns)
    rolling_cagr = np.exp(log_returns.rolling(window, min_periods=window).sum() * 252.0 / window) - 1.0
    valid = rolling_cagr.dropna()
    return float((valid >= TARGET_FLOOR).mean()) if not valid.empty else np.nan


def _performance_row(
    returns: pd.Series,
    cash: pd.Series,
    *,
    stress_returns: pd.Series | None = None,
    annual_traded_notional: float | None = None,
    n_trials: int = LIFETIME_RISK_TRIALS,
) -> dict[str, float]:
    sample = returns.loc[VALIDATION_START:VALIDATION_END].dropna()
    cash_sample = cash.reindex(sample.index).fillna(0.0)
    excess = sample - cash_sample
    drawdown_path = drawdown(sample)
    cagr = annualized_return(sample)
    annual_vol = float(sample.std() * np.sqrt(252))
    max_drawdown = float(drawdown_path.min())
    ulcer = float(np.sqrt(drawdown_path.pow(2).mean()))
    downside = float(np.sqrt(sample.clip(upper=0.0).pow(2).mean()) * np.sqrt(252))
    tail_cut = float(sample.quantile(0.05))
    cvar = float(sample[sample <= tail_cut].mean())
    downside_sample = sample[sample < 0.0]
    sortino = (
        float(excess.mean() / downside_sample.std() * np.sqrt(252))
        if len(downside_sample) > 1 and downside_sample.std() > 0.0
        else np.nan
    )
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0.0 else np.nan
    calendar_returns = sample.groupby(sample.index.year).apply(lambda values: (1.0 + values).prod() - 1.0)
    log_excess = np.log1p(sample) - np.log1p(cash_sample)
    yearly_log_excess = log_excess.groupby(log_excess.index.year).sum()
    concentration = float(yearly_log_excess.abs().max() / yearly_log_excess.abs().sum())
    monthly_excess = monthly_return(sample) - monthly_return(cash_sample)
    _, bootstrap_low, bootstrap_high, bootstrap_p = circular_block_bootstrap(monthly_excess)
    dsr = deflated_sharpe_ratio(
        sharpe,
        n_obs=len(sample),
        n_trials=n_trials,
        periods=252,
        skew=float(excess.skew()),
        kurt=float(excess.kurt() + 3.0),
    )["dsr"]
    stress_cagr = (
        annualized_return(stress_returns.loc[VALIDATION_START:VALIDATION_END].dropna())
        if stress_returns is not None
        else cagr
    )
    return {
        "cagr": cagr,
        "stress_20bp_cagr": stress_cagr,
        "annual_vol": annual_vol,
        "max_drawdown": max_drawdown,
        "ulcer_index": ulcer,
        "downside_deviation": downside,
        "cvar_5pct_daily": cvar,
        "excess_cash_sharpe": sharpe,
        "sortino": sortino,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0.0 else np.nan,
        "cagr_2013_2016": annualized_return(sample.loc["2013":"2016"]),
        "cagr_2017_2021": annualized_return(sample.loc["2017":"2021"]),
        "worst_calendar_year": float(calendar_returns.min()),
        "largest_year_abs_share": concentration,
        "rolling_5y_target_share": _rolling_target_share(sample),
        "bootstrap_90_low_excess_cash": bootstrap_low,
        "bootstrap_90_high_excess_cash": bootstrap_high,
        "bootstrap_one_sided_p": bootstrap_p,
        "deflated_sharpe_probability": float(dsr),
        "annual_traded_notional": annual_traded_notional,
    }


def build_metrics(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
    cash: pd.Series,
) -> pd.DataFrame:
    """Build target-return and left-tail metrics for candidates and benchmarks."""
    rows = []
    for name, candidate in candidates.items():
        validation = candidate.primary.returns.loc[VALIDATION_START:VALIDATION_END]
        years = len(validation) / 252.0
        rows.append(
            {
                "strategy": name,
                "kind": "sector_candidate",
                **_performance_row(
                    candidate.primary.returns,
                    cash,
                    stress_returns=candidate.stress.returns,
                    annual_traded_notional=float(
                        candidate.primary.traded_notional.loc[VALIDATION_START:VALIDATION_END].sum()
                        / years
                    ),
                ),
            }
        )
    for name, returns in benchmarks.items():
        rows.append(
            {
                "strategy": name,
                "kind": "benchmark",
                **_performance_row(returns, cash),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def apply_gates(metrics: pd.DataFrame, collar: pd.Series) -> pd.Series:
    """Apply the preregistered 8%-target and drawdown-first gate."""
    candidate = metrics["kind"] == "sector_candidate"
    passes = (
        candidate
        & (metrics["cagr"] >= TARGET_FLOOR)
        & (metrics["stress_20bp_cagr"] >= 0.07)
        & (metrics["max_drawdown"] >= TARGET_MAX_DRAWDOWN)
        & (metrics["annual_vol"] <= TARGET_VOL_MAX)
        & (metrics["calmar"] >= TARGET_CALMAR)
        & (metrics["max_drawdown"] > float(collar["max_drawdown"]))
        & (metrics["ulcer_index"] < float(collar["ulcer_index"]))
        & (metrics["cagr"] >= float(collar["cagr"]) - 0.01)
        & (metrics["cagr_2013_2016"] >= 0.06)
        & (metrics["cagr_2017_2021"] >= 0.06)
        & (metrics["worst_calendar_year"] >= -0.10)
        & (metrics["largest_year_abs_share"] <= 0.40)
        & (metrics["rolling_5y_target_share"] >= 0.60)
        & (metrics["bootstrap_90_low_excess_cash"] > 0.0)
        & (metrics["deflated_sharpe_probability"] >= 0.95)
    )
    return passes.rename("passes_all_gates")


def build_gate_matrix(metrics: pd.DataFrame, collar: pd.Series) -> pd.DataFrame:
    """Show which preregistered gate each sector candidate passed."""
    frame = metrics.loc[metrics["kind"] == "sector_candidate"]
    gates = pd.DataFrame(
        {
            "CAGR >= 7.5%": frame["cagr"] >= TARGET_FLOOR,
            "20bp CAGR >= 7%": frame["stress_20bp_cagr"] >= 0.07,
            "Max DD <= 15%": frame["max_drawdown"] >= TARGET_MAX_DRAWDOWN,
            "Vol <= 10%": frame["annual_vol"] <= TARGET_VOL_MAX,
            "Calmar >= 0.60": frame["calmar"] >= TARGET_CALMAR,
            "DD beats collar": frame["max_drawdown"] > float(collar["max_drawdown"]),
            "Ulcer beats collar": frame["ulcer_index"] < float(collar["ulcer_index"]),
            "CAGR near collar": frame["cagr"] >= float(collar["cagr"]) - 0.01,
            "Both subperiods >= 6%": (frame["cagr_2013_2016"] >= 0.06)
            & (frame["cagr_2017_2021"] >= 0.06),
            "Worst year >= -10%": frame["worst_calendar_year"] >= -0.10,
            "Year share <= 40%": frame["largest_year_abs_share"] <= 0.40,
            "5y target share >= 60%": frame["rolling_5y_target_share"] >= 0.60,
            "Bootstrap low > cash": frame["bootstrap_90_low_excess_cash"] > 0.0,
            "DSR >= 95%": frame["deflated_sharpe_probability"] >= 0.95,
        },
        index=frame.index,
    )
    return gates.map(lambda passed: "PASS" if passed else "FAIL")


def build_regime_table(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
    spy: pd.Series,
    vix: pd.Series,
) -> pd.DataFrame:
    """Report lagged trend x fixed-VIX regimes without using them to tune candidates."""
    trend = (spy > spy.rolling(200, min_periods=200).mean()).shift(1)
    high_vix = (vix >= 20.0).shift(1)
    valid = trend.notna() & high_vix.notna()
    trend = trend.fillna(False).astype(bool)
    high_vix = high_vix.fillna(False).astype(bool)
    labels = pd.Series(index=spy.index, dtype="object")
    labels.loc[valid & trend & ~high_vix] = "bull_low_vix"
    labels.loc[valid & trend & high_vix] = "bull_high_vix"
    labels.loc[valid & ~trend & ~high_vix] = "bear_low_vix"
    labels.loc[valid & ~trend & high_vix] = "bear_high_vix"
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    rows = []
    for name, returns in series.items():
        sample = returns.loc[VALIDATION_START:VALIDATION_END]
        for regime in ["bull_low_vix", "bull_high_vix", "bear_low_vix", "bear_high_vix"]:
            selected = labels.reindex(sample.index) == regime
            values = sample[selected].dropna()
            if values.empty:
                continue
            drawdown_path = drawdown(values)
            rows.append(
                {
                    "strategy": name,
                    "regime": regime,
                    "n_days": len(values),
                    "annualized_mean": float(values.mean() * 252.0),
                    "annualized_vol": float(values.std() * np.sqrt(252)),
                    "conditional_max_drawdown": float(drawdown_path.min()),
                }
            )
    return pd.DataFrame(rows).set_index(["strategy", "regime"])


def wealth_figure(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    figure = go.Figure()
    for name, candidate in candidates.items():
        returns = candidate.primary.returns.loc[VALIDATION_START:VALIDATION_END]
        figure.add_scatter(
            x=returns.index,
            y=(1.0 + returns).cumprod(),
            name=name,
            mode="lines",
        )
    for name in ["synthetic_spy_collar", "SPY", "spy_60_cash_40"]:
        returns = benchmarks[name].loc[VALIDATION_START:VALIDATION_END]
        figure.add_scatter(
            x=returns.index,
            y=(1.0 + returns).cumprod(),
            name=name,
            mode="lines",
            line={"width": 3 if name == "synthetic_spy_collar" else 1.5, "dash": "dash"},
        )
    figure.update_layout(
        template="plotly_white",
        title="Development-validation wealth (10 bp sector costs)",
        yaxis_title="Wealth",
        legend={"orientation": "h", "y": -0.28},
        margin={"b": 125},
    )
    return figure


def drawdown_figure(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    figure = go.Figure()
    series = {name: candidate.primary.returns for name, candidate in candidates.items()} | {
        "synthetic_spy_collar": benchmarks["synthetic_spy_collar"],
        "SPY": benchmarks["SPY"],
    }
    for name, returns in series.items():
        sample = returns.loc[VALIDATION_START:VALIDATION_END]
        drawdown_path = drawdown(sample)
        figure.add_scatter(
            x=drawdown_path.index,
            y=drawdown_path,
            name=name,
            mode="lines",
            line={"width": 3 if name == "synthetic_spy_collar" else 1.5},
        )
    figure.update_layout(
        template="plotly_white",
        title="Drawdown paths",
        yaxis_title="Drawdown",
        yaxis={"tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.28},
        margin={"b": 125},
    )
    return figure


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "kind",
        "cagr",
        "stress_20bp_cagr",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "downside_deviation",
        "calmar",
        "worst_calendar_year",
        "rolling_5y_target_share",
        "deflated_sharpe_probability",
        "annual_traded_notional",
        "passes_all_gates",
    ]
    frame = metrics[columns].copy()
    for column in [
        "cagr",
        "stress_20bp_cagr",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "downside_deviation",
        "worst_calendar_year",
        "rolling_5y_target_share",
        "deflated_sharpe_probability",
    ]:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    for column in ["calmar", "annual_traded_notional"]:
        frame[column] = frame[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    return frame


def _format_regimes(regimes: pd.DataFrame) -> pd.DataFrame:
    table = regimes["annualized_mean"].unstack("regime")
    return table.map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")


def render_report(
    metrics: pd.DataFrame,
    regimes: pd.DataFrame,
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
    collar: SyntheticCollarResult,
) -> str:
    """Render a self-contained HTML decision report."""
    winners = metrics.index[metrics["passes_all_gates"]].tolist()
    candidate_metrics = metrics[metrics["kind"] == "sector_candidate"]
    target_attainers = candidate_metrics[candidate_metrics["cagr"] >= TARGET_FLOOR]
    if winners:
        verdict = (
            f"{len(winners)} candidate(s) passed every pre-holdout gate: {', '.join(winners)}. "
            "The 2022+ holdout remains sealed pending leakage and fundamental review."
        )
        verdict_class = "warn"
    elif not target_attainers.empty:
        best = target_attainers.sort_values(
            ["max_drawdown", "ulcer_index"],
            ascending=[False, True],
        ).index[0]
        verdict = (
            f"{len(target_attainers)} candidate(s) reached the 7.5% return tolerance, but none passed "
            f"all drawdown, collar-dominance and statistical gates. Drawdown-first leader: {best}."
        )
        verdict_class = "warn"
    else:
        verdict = (
            "No candidate reached the 7.5% return tolerance after costs. The 2022+ holdout remains sealed."
        )
        verdict_class = "reject"

    definitions = pd.DataFrame(
        [
            ("C1", "Positive 200-day trend + inverse downside risk", "Monthly", "25%"),
            ("C2", "Fast stress exit / four-week slow re-entry", "Weekly", "30%"),
            ("C3", "50% defensive core + 50% eligible cyclical sleeve", "Monthly", "25%"),
            ("C4", "C1 sleeve scaled to an 8% downside-risk budget", "Monthly", "25%"),
            ("C5", "Four lowest-Ulcer positive-trend sectors", "Monthly", "25%"),
        ],
        columns=["ID", "Frozen construction", "Decision frequency", "Sector cap"],
    ).set_index("ID")
    wealth_html = wealth_figure(candidates, benchmarks).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displayModeBar": False, "responsive": True},
    )
    drawdown_html = drawdown_figure(candidates, benchmarks).to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )
    collar_validation = collar.diagnostics.loc[VALIDATION_START:VALIDATION_END]
    gate_matrix = build_gate_matrix(metrics, metrics.loc["synthetic_spy_collar"])
    model_2020_q1 = float(
        (1.0 + benchmarks["synthetic_spy_collar"].loc["2020-01-01":"2020-03-31"]).prod()
        - 1.0
    )
    official_2020_q1 = -0.05
    q1_gap = model_2020_q1 - official_2020_q1
    q1_class = "guard" if abs(q1_gap) <= 0.03 else "warn"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>US Sector Target 8% / Low Drawdown Study</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d8e0ea;--navy:#173b66;--panel:#f6f8fb;--red:#a61b1b;--amber:#8a5a00;}}
body{{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:#fff}}
main{{max-width:1180px;margin:0 auto;padding:42px 28px 72px}}h1{{margin:0 0 8px;color:var(--navy);font-size:32px}}
h2{{margin-top:38px;padding-bottom:8px;border-bottom:2px solid var(--line);color:var(--navy)}}p,li{{line-height:1.6}}
.sub{{color:var(--muted)}}.callout{{padding:18px 20px;border-radius:8px;margin:22px 0;font-weight:600}}
.reject{{background:#fff1f1;border-left:5px solid var(--red)}}.warn{{background:#fff7e6;border-left:5px solid var(--amber)}}
.guard{{background:#eef5ff;border-left:5px solid var(--navy);font-weight:500}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}}.card strong{{display:block;font-size:22px;color:var(--navy)}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
thead th{{background:#edf2f7;color:#27364a}}.small{{color:var(--muted);font-size:13px}}code{{background:#f2f4f7;padding:2px 5px;border-radius:4px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}main{{padding:24px 14px 50px}}}}
</style></head><body><main>
<h1>US Sector Portfolio: 8% Target, Drawdown First</h1>
<p class="sub">Frozen-batch development validation · 2013-2021 · generated 2026-07-17</p>
<div class="callout {verdict_class}">{verdict}</div>
<div class="callout guard"><strong>Holdout guard:</strong> no 2022+ price, VIX or option-proxy observation was loaded.</div>
<div class="grid">
<div class="card"><strong>{int((candidate_metrics['cagr'] >= TARGET_FLOOR).sum())}</strong>return-target attainers</div>
<div class="card"><strong>{int(metrics['passes_all_gates'].sum())}</strong>all-gate passes</div>
<div class="card"><strong>{metrics.loc['synthetic_spy_collar','cagr']:.1%}</strong>synthetic collar CAGR</div>
<div class="card"><strong>{metrics.loc['synthetic_spy_collar','max_drawdown']:.1%}</strong>synthetic collar max DD</div>
</div>

<h2>What was tested</h2>
<p>The objective is lexicographic: first attain at least 7.5% net CAGR, then minimize maximum drawdown,
then Ulcer Index/downside deviation/volatility. Candidate formulas were frozen before results.</p>
<div class="table-wrap">{definitions.to_html(border=0)}</div>

<h2>Primary results</h2>
<div class="table-wrap">{_format_metrics(metrics).to_html(border=0)}</div>
<p class="small">Sector candidates include 10 bp one-way costs and a 2% rebalance threshold. The 20 bp column is the stress case.</p>

<h2>Why each candidate failed</h2>
<p>Every column below is a frozen gate. Passing the return target alone is insufficient.</p>
<div class="table-wrap">{gate_matrix.to_html(border=0)}</div>

<h2>Wealth and drawdown</h2>
{wealth_html}
{drawdown_html}

<h2>Regime performance</h2>
<p>Annualized mean returns in lagged SPY trend × fixed VIX-20 regimes. These are diagnostics, not tuning inputs.</p>
<div class="table-wrap">{_format_regimes(regimes).to_html(border=0)}</div>

<h2>Synthetic SPY collar benchmark</h2>
<p>The benchmark is long SPY total return, long a quarterly 5% OTM put, and short a monthly 10% OTM call—matching the
economic structure of Cboe's <a href="https://cdn.cboe.com/api/global/us_indices/governance/Cboe_Collar_Indices_Methodology.pdf">95-110 Collar methodology</a>.
VIX is a 30-day SPX implied-volatility measure, not a strike-specific option quote; see the
<a href="https://www.cboe.com/tradable_products/vix/faqs">official VIX explanation</a>.</p>
<ul>
<li>ATM proxy: max(VIX, 21-day realised vol + 2 points).</li>
<li>Put IV: ATM proxy + 4 points; call IV: ATM proxy - 2 points, floored at 8%.</li>
<li>Long put paid 10% above model mid; short call receives 10% below model mid.</li>
<li>{int(collar_validation['put_roll'].sum())} quarterly put rolls and {int(collar_validation['call_roll'].sum())} monthly call rolls in development-validation.</li>
<li>Mean model put IV {collar_validation['put_iv'].mean():.1%}; mean model call IV {collar_validation['call_iv'].mean():.1%}.</li>
</ul>
<div class="callout {q1_class}"><strong>External plausibility check:</strong> the model collar returned
{model_2020_q1:.1%} in 2020Q1, versus -5.0% reported by Cboe for CLL. The gap is {q1_gap:+.1%}.
This check was specified after the batch only to diagnose model risk; no IV assumption was retuned.
See Cboe's <a href="https://www.cboe.com/us/index_protection/replays/">historical drawdown comparison</a>.</div>
<div class="callout guard">This is a conservative scenario benchmark, not an executable option-chain backtest. It omits American exercise,
discrete dividends, strike grids, SPX/SPY basis and historical NBBO. A strategy does not become deployable merely by beating it.</div>

<h2>Research controls</h2>
<ul>
<li>All inputs stop before 2022; development-validation was already viewed and is not called untouched OOS.</li>
<li>Signals use trailing data only and trade at the next close.</li>
<li>Five frozen candidates; no after-result sixth variant.</li>
<li>Cash financing, 10/20 bp costs, subperiods, rolling five-year target attainment, block bootstrap and Deflated Sharpe are reported.</li>
<li>Static leakage scan: zero blockers, one false-positive warning (the ex-post CVaR quantile), and informational trailing-window/forward-fill hits reviewed manually.</li>
<li>Manual verdict: no signal lookahead found; target weights earn returns only after formation. The collar remains model-risk-heavy because VIX is not a strike/term-specific executable quote.</li>
</ul>

<h2>Decision</h2>
<p>{'Proceed to independent leakage and fundamental review, but keep the holdout sealed.' if winners else 'Stop for user review. Do not tune this batch or open the holdout.'}</p>
</main></body></html>"""


if __name__ == "__main__":
    main()
