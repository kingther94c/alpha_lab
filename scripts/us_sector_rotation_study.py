"""Pre-holdout US sector rotation and XLK hardware/software study.

Downloads only data strictly before 2022-01-01. The 2022+ pseudo-OOS window
remains sealed until a candidate passes the pre-holdout research gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import yfinance as yf

from alpha_lab.backtest.sector_momentum import (
    multi_horizon_momentum_signal,
    volume_confirmation_signal,
)
from alpha_lab.backtest.vector import DriftBacktestResult, run_drift_backtest
from alpha_lab.data.align import forward_returns
from alpha_lab.data.calendars import rebalance_dates
from alpha_lab.data.loaders.fred import cash_total_return_index, load_series
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.portfolio.vol_target import (
    RollingVolMatchResult,
    rolling_match_benchmark_vol_weights,
)
from alpha_lab.utils.paths import PROJECT_ROOT

DATA_START = "1998-12-01"
DATA_END_EXCLUSIVE = "2022-01-01"
HOLDOUT_START = pd.Timestamp("2022-01-01")
TRAIN_END = pd.Timestamp("2012-12-31")
VALIDATION_START = pd.Timestamp("2013-01-01")
VALIDATION_END = pd.Timestamp("2021-12-31")
TRADING_BPS = 5.0
STRESS_TRADING_BPS = 10.0
VOL_BAND = (0.9, 1.1)
SECTOR_MAX_WEIGHT = 0.60

LEGACY_SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB"]
ALL_SECTORS = [*LEGACY_SECTORS, "XLC", "XLRE"]
TECH_TICKERS = ["XLK", "SOXX", "IGV", "SPY"]
ALL_TICKERS = sorted(set([*ALL_SECTORS, *TECH_TICKERS]))
REPORT_PATH = PROJECT_ROOT / "reports" / "us_sector_rotation_preholdout.html"


@dataclass
class Candidate:
    """Targets, vol diagnostics, and cost-sensitivity backtests."""

    name: str
    raw_targets: pd.DataFrame
    matched: RollingVolMatchResult
    base: DriftBacktestResult
    stress: DriftBacktestResult


def main() -> None:
    prices, volumes = load_data()
    cash_returns = load_cash_returns(prices.index)
    coverage = coverage_table(prices, volumes)

    legacy_prices = prices[LEGACY_SECTORS].dropna()
    legacy_volumes = volumes[LEGACY_SECTORS].reindex(legacy_prices.index)
    legacy_spy = prices["SPY"].reindex(legacy_prices.index).dropna()
    legacy_cash = cash_returns.reindex(legacy_prices.index).fillna(0.0)

    sector_candidates = {
        "risk_matched_equal_weight": build_sector_candidate(
            "risk_matched_equal_weight",
            legacy_prices,
            legacy_volumes,
            legacy_spy,
            legacy_cash,
            mode="equal",
        ),
        "multi_horizon_momentum": build_sector_candidate(
            "multi_horizon_momentum",
            legacy_prices,
            legacy_volumes,
            legacy_spy,
            legacy_cash,
            mode="momentum",
        ),
        "momentum_plus_volume": build_sector_candidate(
            "momentum_plus_volume",
            legacy_prices,
            legacy_volumes,
            legacy_spy,
            legacy_cash,
            mode="momentum_volume",
        ),
    }
    sector_benchmark = spy_benchmark(legacy_spy, legacy_cash)

    bridge_prices = prices[ALL_SECTORS].dropna()
    bridge_volumes = volumes[ALL_SECTORS].reindex(bridge_prices.index)
    bridge_spy = prices["SPY"].reindex(bridge_prices.index).dropna()
    bridge_cash = cash_returns.reindex(bridge_prices.index).fillna(0.0)
    bridge_candidates = {
        "risk_matched_equal_weight": build_sector_candidate(
            "risk_matched_equal_weight",
            bridge_prices,
            bridge_volumes,
            bridge_spy,
            bridge_cash,
            mode="equal",
        ),
        "multi_horizon_momentum": build_sector_candidate(
            "multi_horizon_momentum",
            bridge_prices,
            bridge_volumes,
            bridge_spy,
            bridge_cash,
            mode="momentum",
        ),
        "momentum_plus_volume": build_sector_candidate(
            "momentum_plus_volume",
            bridge_prices,
            bridge_volumes,
            bridge_spy,
            bridge_cash,
            mode="momentum_volume",
        ),
    }
    bridge_benchmark = spy_benchmark(bridge_spy, bridge_cash)

    tech_candidates, tech_direct, tech_signal_stats, tech_pair = build_technology_study(
        prices,
        cash_returns,
    )
    tech_prices = prices[TECH_TICKERS].dropna()
    tech_cash = cash_returns.reindex(tech_prices.index).fillna(0.0)
    tech_benchmark = spy_benchmark(tech_prices["SPY"], tech_cash)

    legacy_metrics = candidate_metrics_table(
        sector_candidates,
        sector_benchmark,
        legacy_cash,
        VALIDATION_START,
        VALIDATION_END,
    )
    bridge_start = max(pd.Timestamp("2019-08-01"), bridge_prices.index.min())
    bridge_metrics = candidate_metrics_table(
        bridge_candidates,
        bridge_benchmark,
        bridge_cash,
        bridge_start,
        VALIDATION_END,
    )
    tech_metrics = candidate_metrics_table(
        tech_candidates,
        tech_benchmark,
        tech_cash,
        VALIDATION_START,
        VALIDATION_END,
    )

    legacy_regimes = regime_metrics_table(
        sector_candidates,
        sector_benchmark,
        legacy_spy,
        VALIDATION_START,
        VALIDATION_END,
    )
    tech_regimes = regime_metrics_table(
        tech_candidates,
        tech_benchmark,
        tech_prices["SPY"],
        VALIDATION_START,
        VALIDATION_END,
    )
    robustness = robustness_table(sector_candidates, sector_benchmark)
    tech_robustness = robustness_table(tech_candidates, tech_benchmark)
    tech_direct_metrics = direct_metrics_table(tech_direct, tech_benchmark, tech_cash)

    html = render_report(
        coverage=coverage,
        legacy_candidates=sector_candidates,
        legacy_benchmark=sector_benchmark,
        legacy_metrics=legacy_metrics,
        bridge_candidates=bridge_candidates,
        bridge_benchmark=bridge_benchmark,
        bridge_metrics=bridge_metrics,
        legacy_regimes=legacy_regimes,
        robustness=robustness,
        tech_candidates=tech_candidates,
        tech_benchmark=tech_benchmark,
        tech_metrics=tech_metrics,
        tech_regimes=tech_regimes,
        tech_direct_metrics=tech_direct_metrics,
        tech_signal_stats=tech_signal_stats,
        tech_pair=tech_pair,
        tech_robustness=tech_robustness,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print("\nLegacy 9-sector validation:\n", legacy_metrics.to_string())
    print("\n11-sector bridge:\n", bridge_metrics.to_string())
    print("\nTechnology validation:\n", tech_metrics.to_string())
    print("\nTechnology paired attribution:\n", tech_pair.to_string())


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load adjusted closes and volume without touching the holdout."""
    cache_dir = Path(gettempdir()) / "alpha_lab_yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    prices = load_prices(
        ALL_TICKERS,
        DATA_START,
        DATA_END_EXCLUSIVE,
        field="Close",
        auto_adjust=True,
        threads=False,
    )
    volumes = load_prices(
        ALL_TICKERS,
        DATA_START,
        DATA_END_EXCLUSIVE,
        field="Volume",
        auto_adjust=False,
        threads=False,
    )
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    volumes.index = pd.DatetimeIndex(volumes.index).tz_localize(None)
    if prices.empty or volumes.empty:
        raise RuntimeError("price or volume loader returned no data")
    if prices.index.max() >= HOLDOUT_START or volumes.index.max() >= HOLDOUT_START:
        raise RuntimeError("price or volume loader crossed the sealed 2022 holdout")
    return prices.sort_index(), volumes.sort_index()


def load_cash_returns(index: pd.DatetimeIndex) -> pd.Series:
    """Leak-safe T-bill returns with calendar-day accrual between price bars."""
    dtb3 = load_series("DTB3", start=DATA_START, end="2021-12-31")["DTB3"]
    timeline = dtb3.index.union(index).sort_values()
    rates = dtb3.reindex(timeline).ffill()
    cash_index = cash_total_return_index(rates)
    return cash_index.reindex(index).pct_change().fillna(0.0).rename("cash_return")


def coverage_table(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in ALL_TICKERS:
        price = prices[ticker].dropna()
        volume = volumes[ticker].reindex(price.index)
        rows.append(
            {
                "ticker": ticker,
                "first_price": price.index.min().date().isoformat(),
                "last_price": price.index.max().date().isoformat(),
                "price_rows": len(price),
                "missing_volume": int(volume.isna().sum()),
                "zero_volume": int((volume == 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def build_sector_candidate(
    name: str,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    spy: pd.Series,
    cash_returns: pd.Series,
    *,
    mode: str,
) -> Candidate:
    """Build a monthly sector candidate and its 5/10 bp backtests."""
    decision_dates = rebalance_dates(prices.index, freq="ME")
    momentum = multi_horizon_momentum_signal(prices)
    if mode == "equal":
        score = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    elif mode == "momentum":
        score = momentum
    elif mode == "momentum_volume":
        volume_score = volume_confirmation_signal(prices, volumes).rank(axis=1, pct=True)
        score = 0.75 * momentum + 0.25 * volume_score
    else:
        raise ValueError(f"unknown sector candidate mode: {mode}")

    rows = []
    for date in decision_dates:
        if len(prices.loc[:date]) < 64:
            continue
        row = score.loc[date].dropna()
        if len(row) < len(prices.columns):
            continue
        if mode == "equal":
            weights = pd.Series(1.0 / len(row), index=row.index, name=date)
        else:
            winners = row.nlargest(3).index
            weights = pd.Series(0.0, index=prices.columns, name=date)
            weights.loc[winners] = 1.0 / len(winners)
        rows.append(weights)
    raw_targets = pd.DataFrame(rows).reindex(columns=prices.columns)
    matched = rolling_match_benchmark_vol_weights(
        raw_targets,
        prices.pct_change(),
        spy.pct_change(),
        band=VOL_BAND,
        turnover_penalty=0.05,
        max_weight=SECTOR_MAX_WEIGHT,
    )
    base = run_drift_backtest(
        matched.weights,
        prices,
        trading_bps=TRADING_BPS,
        cash_returns=cash_returns,
    )
    stress = run_drift_backtest(
        matched.weights,
        prices,
        trading_bps=STRESS_TRADING_BPS,
        cash_returns=cash_returns,
    )
    return Candidate(name, raw_targets, matched, base, stress)


def spy_benchmark(spy: pd.Series, cash_returns: pd.Series) -> DriftBacktestResult:
    target = pd.DataFrame({"SPY": [1.0]}, index=[spy.index[0]])
    return run_drift_backtest(
        target,
        spy.to_frame("SPY"),
        trading_bps=TRADING_BPS,
        cash_returns=cash_returns,
    )


def build_technology_study(
    prices: pd.DataFrame,
    cash_returns: pd.Series,
) -> tuple[dict[str, Candidate], dict[str, DriftBacktestResult], pd.DataFrame, pd.DataFrame]:
    tech = prices[TECH_TICKERS].dropna()
    cash = cash_returns.reindex(tech.index).fillna(0.0)
    returns = tech.pct_change()
    r63 = tech.pct_change(63)
    month_ends = rebalance_dates(tech.index, freq="ME")
    signal = pd.DataFrame(index=month_ends)
    signal["relative_63d"] = (r63["SOXX"] - r63["IGV"]).reindex(month_ends)
    signal["breadth_63d"] = (
        (r63["SOXX"] > r63["SPY"]) & (r63["IGV"] > r63["SPY"])
    ).reindex(month_ends)
    signal = signal.dropna()

    masks = {
        "hardware_lead": signal["relative_63d"] > 0,
        "breadth": signal["breadth_63d"].astype(bool),
        "relative_reversal": signal["relative_63d"] < 0,
        "static_risk_matched_xlk": pd.Series(True, index=signal.index),
    }
    raw_targets = {}
    for name, active in masks.items():
        raw = pd.DataFrame(0.0, index=signal.index, columns=["SPY", "XLK"])
        raw.loc[~active, "SPY"] = 1.0
        raw.loc[active, "XLK"] = 1.0
        raw_targets[name] = raw

    train_breadth_share = float(signal.loc[:TRAIN_END, "breadth_63d"].mean())
    raw_targets["breadth_train_exposure_baseline"] = pd.DataFrame(
        {
            "SPY": 1.0 - train_breadth_share,
            "XLK": train_breadth_share,
        },
        index=signal.index,
    )

    candidates = {}
    for name, raw in raw_targets.items():
        matched = rolling_match_benchmark_vol_weights(
            raw,
            returns[["SPY", "XLK"]],
            returns["SPY"],
            band=VOL_BAND,
            max_weight=1.0,
        )
        base = run_drift_backtest(
            matched.weights,
            tech[["SPY", "XLK"]],
            trading_bps=TRADING_BPS,
            cash_returns=cash,
        )
        stress = run_drift_backtest(
            matched.weights,
            tech[["SPY", "XLK"]],
            trading_bps=STRESS_TRADING_BPS,
            cash_returns=cash,
        )
        candidates[name] = Candidate(name, raw, matched, base, stress)

    direct = {}
    direct_targets = {
        "soxx_igv_winner": _direct_tech_targets(signal, winner=True),
        "soxx_igv_contrarian": _direct_tech_targets(signal, winner=False),
        "soxx_igv_equal_weight": pd.DataFrame(
            0.5,
            index=signal.index,
            columns=["SOXX", "IGV"],
        ),
    }
    for name, target in direct_targets.items():
        direct[name] = run_drift_backtest(
            target,
            tech[["SOXX", "IGV"]],
            trading_bps=TRADING_BPS,
            cash_returns=cash,
        )

    signal_stats = technology_signal_stats(signal, tech)
    pair = pd.DataFrame(
        [
            paired_attribution(
                candidates["relative_reversal"],
                candidates["static_risk_matched_xlk"],
                name="reversal_minus_static_xlk",
            ),
            paired_attribution(
                candidates["breadth"],
                candidates["breadth_train_exposure_baseline"],
                name="breadth_minus_train_exposure_baseline",
            ),
        ]
    )
    return candidates, direct, signal_stats, pair


def _direct_tech_targets(signal: pd.DataFrame, *, winner: bool) -> pd.DataFrame:
    hardware = signal["relative_63d"] > 0
    if not winner:
        hardware = ~hardware
    target = pd.DataFrame(0.0, index=signal.index, columns=["SOXX", "IGV"])
    target.loc[hardware, "SOXX"] = 1.0
    target.loc[~hardware, "IGV"] = 1.0
    return target


def technology_signal_stats(signal: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    forward = forward_returns(prices.pct_change(), horizon=21)
    frame = signal.copy()
    frame["xlk_forward_excess"] = (
        forward["XLK"] - forward["SPY"]
    ).reindex(frame.index)
    rows = []
    for label, start, end in [
        ("train", pd.Timestamp("2002-01-01"), TRAIN_END),
        ("validation", VALIDATION_START, pd.Timestamp("2021-11-30")),
    ]:
        sample = frame.loc[start:end].dropna()
        x = sample["relative_63d"]
        y = sample["xlk_forward_excess"]
        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append(
            {
                "sample": label,
                "observations": len(sample),
                "rank_ic": x.corr(y, method="spearman"),
                "beta": fit.params["relative_63d"],
                "hac_t": fit.tvalues["relative_63d"],
            }
        )
    return pd.DataFrame(rows).set_index("sample")


def candidate_metrics_table(
    candidates: dict[str, Candidate],
    benchmark: DriftBacktestResult,
    cash_returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for name, candidate in candidates.items():
        metrics = performance_metrics(candidate.base, benchmark, cash_returns, start, end)
        stress = performance_metrics(candidate.stress, benchmark, cash_returns, start, end)
        diagnostics = candidate.matched.diagnostics.loc[start:end]
        rows.append(
            {
                "strategy": name,
                **metrics,
                "stress_active_cagr": stress["active_cagr"],
                "stress_active_ir": stress["active_ir"],
                "target_ex_ante_vol_compliance": diagnostics["vol_ratio"].between(
                    VOL_BAND[0] - 1e-7,
                    VOL_BAND[1] + 1e-7,
                ).mean(),
                "target_max_weight": diagnostics["max_weight"].max(),
                "target_weight_sum_error": (diagnostics["weight_sum"] - 1.0).abs().max(),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def performance_metrics(
    result: DriftBacktestResult,
    benchmark: DriftBacktestResult,
    cash_returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    returns = result.returns.loc[start:end]
    benchmark_returns = benchmark.returns.reindex(returns.index)
    cash = cash_returns.reindex(returns.index).fillna(0.0)
    active = returns - benchmark_returns
    excess_cash = returns - cash
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    years = len(returns) / 252.0
    benchmark_excess_cash = benchmark_returns - cash
    regression = sm.OLS(excess_cash, sm.add_constant(benchmark_excess_cash)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 21},
    )
    portfolio_realized_vol = rolling_geometric_vol(result.gross_returns).loc[start:end]
    benchmark_realized_vol = rolling_geometric_vol(benchmark.gross_returns).reindex(
        portfolio_realized_vol.index
    )
    realized_ratio = (portfolio_realized_vol / benchmark_realized_vol).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    realized_p10, realized_p90 = np.percentile(realized_ratio.to_numpy(), [10.0, 90.0])
    return {
        "cagr": annualized_return(returns),
        "active_cagr": annualized_return(returns) - annualized_return(benchmark_returns),
        "vol": float(returns.std() * np.sqrt(252)),
        "excess_cash_sharpe": float(excess_cash.mean() / excess_cash.std() * np.sqrt(252)),
        "active_ir": float(active.mean() / active.std() * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "alpha": float(regression.params.iloc[0] * 252),
        "alpha_hac_t": float(regression.tvalues.iloc[0]),
        "beta": float(regression.params.iloc[1]),
        "portfolio_realized_vol_median": float(portfolio_realized_vol.median()),
        "benchmark_realized_vol_median": float(benchmark_realized_vol.median()),
        "realized_vol_ratio_median": float(realized_ratio.median()),
        "realized_vol_ratio_p10": float(realized_p10),
        "realized_vol_ratio_p90": float(realized_p90),
        "realized_vol_band_share": float(realized_ratio.between(*VOL_BAND).mean()),
        "annual_traded_notional": float(result.traded_notional.loc[start:end].sum() / years),
        "annual_cost_drag": float(result.costs.loc[start:end].sum() / years),
        "n_days": len(returns),
    }


def annualized_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)


def rolling_geometric_vol(
    returns: pd.Series,
    *,
    short_window: int = 21,
    long_window: int = 63,
) -> pd.Series:
    """Rolling geometric mean of short- and long-window annualized volatility."""
    short = returns.rolling(short_window, min_periods=short_window).std() * np.sqrt(252)
    long = returns.rolling(long_window, min_periods=long_window).std() * np.sqrt(252)
    return np.sqrt(short * long).rename("geometric_realized_vol")


def regime_labels(spy: pd.Series) -> pd.Series:
    returns = spy.pct_change()
    trend = spy > spy.rolling(200, min_periods=200).mean()
    vol_21 = returns.rolling(21, min_periods=21).std() * np.sqrt(252)
    vol_63 = returns.rolling(63, min_periods=63).std() * np.sqrt(252)
    geometric_vol = np.sqrt(vol_21 * vol_63)
    high_vol = geometric_vol > geometric_vol.rolling(756, min_periods=504).median().shift(1)
    labels = pd.Series(np.where(trend, "bull", "bear"), index=spy.index)
    labels += "_" + pd.Series(np.where(high_vol, "high_vol", "low_vol"), index=spy.index)
    return labels.shift(1).rename("regime")


def regime_metrics_table(
    candidates: dict[str, Candidate],
    benchmark: DriftBacktestResult,
    spy: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    labels = regime_labels(spy).loc[start:end]
    rows = []
    for name, candidate in candidates.items():
        active = candidate.base.returns.loc[start:end] - benchmark.returns.loc[start:end]
        for regime in ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]:
            selected = labels == regime
            episodes = int((selected & ~selected.shift(1, fill_value=False)).sum())
            regime_active = active[selected]
            rows.append(
                {
                    "strategy": name,
                    "regime": regime,
                    "n_days": int(selected.sum()),
                    "episodes": episodes,
                    "annual_active_mean": (
                        float(regime_active.mean() * 252) if not regime_active.empty else np.nan
                    ),
                    "active_ir": (
                        float(regime_active.mean() / regime_active.std() * np.sqrt(252))
                        if len(regime_active) > 1 and regime_active.std() > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["strategy", "regime"])


def robustness_table(
    candidates: dict[str, Candidate],
    benchmark: DriftBacktestResult,
) -> pd.DataFrame:
    rows = []
    for name, candidate in candidates.items():
        for label, start, end in [
            ("2013-2016", pd.Timestamp("2013-01-01"), pd.Timestamp("2016-12-31")),
            ("2017-2021", pd.Timestamp("2017-01-01"), VALIDATION_END),
        ]:
            returns = candidate.base.returns.loc[start:end]
            benchmark_returns = benchmark.returns.reindex(returns.index)
            active = returns - benchmark_returns
            rows.append(
                {
                    "strategy": name,
                    "subperiod": label,
                    "active_cagr": annualized_return(returns) - annualized_return(benchmark_returns),
                    "active_ir": float(active.mean() / active.std() * np.sqrt(252)),
                }
            )
        validation = candidate.base.returns.loc[VALIDATION_START:VALIDATION_END]
        benchmark_validation = benchmark.returns.reindex(validation.index)
        active_monthly = monthly_return(validation) - monthly_return(benchmark_validation)
        estimate, low, high, p_value = circular_block_bootstrap(active_monthly)
        yearly_log_active = (
            np.log1p(validation) - np.log1p(benchmark_validation)
        ).groupby(validation.index.year).sum()
        rows.append(
            {
                "strategy": name,
                "subperiod": "bootstrap_6m",
                "active_cagr": estimate,
                "active_ir": np.nan,
                "ci_90_low": low,
                "ci_90_high": high,
                "one_sided_p": p_value,
                "largest_year_abs_share": float(
                    yearly_log_active.abs().max() / yearly_log_active.abs().sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index(["strategy", "subperiod"])


def monthly_return(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).groupby(returns.index.to_period("M")).prod() - 1.0


def circular_block_bootstrap(
    values: pd.Series,
    *,
    block_length: int = 6,
    draws: int = 5_000,
    seed: int = 20260717,
) -> tuple[float, float, float, float]:
    sample = values.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(sample)
    raw = np.empty(draws)
    null = np.empty(draws)
    centered = sample - sample.mean()
    for draw in range(draws):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(n))
            indices.extend(((start + np.arange(block_length)) % n).tolist())
        selected = np.asarray(indices[:n])
        raw[draw] = sample[selected].mean() * 12.0
        null[draw] = centered[selected].mean() * 12.0
    estimate = float(sample.mean() * 12.0)
    low, high = np.percentile(raw, [5.0, 95.0])
    p_value = float((np.sum(null >= estimate) + 1) / (draws + 1))
    return estimate, float(low), float(high), p_value


def paired_attribution(left_candidate: Candidate, right_candidate: Candidate, *, name: str) -> pd.Series:
    rows = {}
    for label, left, right in [
        ("5bps", left_candidate.base, right_candidate.base),
        ("10bps", left_candidate.stress, right_candidate.stress),
    ]:
        r = left.returns.loc[VALIDATION_START:VALIDATION_END]
        s = right.returns.reindex(r.index)
        difference = r - s
        monthly_difference = monthly_return(r) - monthly_return(s)
        estimate, low, high, p_value = circular_block_bootstrap(monthly_difference)
        rows[f"{label}_cagr_difference"] = annualized_return(r) - annualized_return(s)
        rows[f"{label}_incremental_ir"] = float(
            difference.mean() / difference.std() * np.sqrt(252)
        )
        rows[f"{label}_bootstrap_ann_mean"] = estimate
        rows[f"{label}_bootstrap_90_low"] = low
        rows[f"{label}_bootstrap_90_high"] = high
        rows[f"{label}_bootstrap_p"] = p_value
    return pd.Series(rows, name=name)


def direct_metrics_table(
    direct: dict[str, DriftBacktestResult],
    benchmark: DriftBacktestResult,
    cash_returns: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, result in direct.items():
        r = result.returns.loc[VALIDATION_START:VALIDATION_END]
        cash = cash_returns.reindex(r.index).fillna(0.0)
        wealth = (1.0 + r).cumprod()
        rows.append(
            {
                "strategy": name,
                "cagr": annualized_return(r),
                "vol": float(r.std() * np.sqrt(252)),
                "excess_cash_sharpe": float((r - cash).mean() / (r - cash).std() * np.sqrt(252)),
                "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
                "annual_traded_notional": float(
                    result.traded_notional.loc[VALIDATION_START:VALIDATION_END].sum()
                    / (len(r) / 252)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def equity_figure(
    candidates: dict[str, Candidate],
    benchmark: DriftBacktestResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    benchmark_returns = benchmark.returns.loc[start:end]
    fig.add_scatter(
        x=benchmark_returns.index,
        y=(1.0 + benchmark_returns).cumprod(),
        name="SPY",
        mode="lines",
    )
    for name, candidate in candidates.items():
        returns = candidate.base.returns.loc[start:end]
        fig.add_scatter(
            x=returns.index,
            y=(1.0 + returns).cumprod(),
            name=name,
            mode="lines",
        )
    fig.update_layout(template="plotly_white", title=title, yaxis_title="Wealth")
    return fig


def render_report(**context: object) -> str:
    legacy_candidates = context["legacy_candidates"]
    legacy_benchmark = context["legacy_benchmark"]
    bridge_candidates = context["bridge_candidates"]
    bridge_benchmark = context["bridge_benchmark"]
    tech_candidates = context["tech_candidates"]
    tech_benchmark = context["tech_benchmark"]
    legacy_metrics = context["legacy_metrics"]
    tech_pair = context["tech_pair"]
    best_sector = legacy_metrics["active_cagr"].idxmax()
    best_sector_active = legacy_metrics.loc[best_sector, "active_cagr"]
    best_sector_stress = legacy_metrics.loc[best_sector, "stress_active_cagr"]
    reversal_increment = tech_pair.loc["reversal_minus_static_xlk", "5bps_cagr_difference"]
    reversal_stress_increment = tech_pair.loc[
        "reversal_minus_static_xlk",
        "10bps_cagr_difference",
    ]
    breadth_increment = tech_pair.loc[
        "breadth_minus_train_exposure_baseline",
        "5bps_cagr_difference",
    ]

    legacy_fig = equity_figure(
        legacy_candidates,
        legacy_benchmark,
        VALIDATION_START,
        VALIDATION_END,
        "Legacy 9-sector validation wealth",
    ).to_html(full_html=False, include_plotlyjs=True)
    bridge_fig = equity_figure(
        bridge_candidates,
        bridge_benchmark,
        pd.Timestamp("2019-08-01"),
        VALIDATION_END,
        "Full 11-sector bridge wealth",
    ).to_html(full_html=False, include_plotlyjs=False)
    tech_fig = equity_figure(
        tech_candidates,
        tech_benchmark,
        VALIDATION_START,
        VALIDATION_END,
        "XLK hardware/software validation wealth",
    ).to_html(full_html=False, include_plotlyjs=False)

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>US Sector Rotation Pre-Holdout Research</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 22px;color:#172033;line-height:1.55}}
h1,h2,h3{{color:#102a43}} .banner{{background:#fff3cd;border:1px solid #e0a800;padding:14px;border-radius:8px}}
.verdict{{background:#e8f1fb;border-left:5px solid #2774ae;padding:14px}} table{{border-collapse:collapse;width:100%;max-width:100%;display:block;overflow-x:auto;white-space:nowrap;font-size:13px;margin:12px 0 26px}}
th,td{{border:1px solid #d9e2ec;padding:6px 8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{background:#eef3f8}} code{{background:#eef3f8;padding:2px 4px}} .small{{font-size:12px;color:#52606d}}
</style></head><body>
<h1>US Sector Rotation 与 XLK 硬件/软件研究</h1>
<p class="small">生成日期：2026-07-17 · 数据硬截止：2021-12-31 · 日收益、5bp one-way 主成本、10bp 压力成本</p>
<div class="banner"><strong>Holdout 状态：</strong>本脚本的价格与成交量请求均以 2022-01-01 为 exclusive end，并对两个返回索引同时设硬断言；本次运行未读取2022+。该区间仅作为 pseudo-OOS 保留集，本文只包含 train 与 validation。</div>

<h2>执行摘要</h2>
<div class="verdict"><strong>当前结论：不打开2022+ holdout。</strong> 9-sector 中表现最好的 <code>{best_sector}</code> 验证期主动CAGR为 {best_sector_active:.2%}，10bp下为 {best_sector_stress:.2%}，价格/成交量rotation候选判定 <code>reject</code>。Hardware lead、直接SOXX/IGV轮动与relative reversal也不支持增量择时alpha；relative reversal相对静态risk-matched XLK的5bp增量仅 {reversal_increment:.2%}，10bp为 {reversal_stress_increment:.2%}。Breadth相对训练期科技暴露基线的5bp增量为 {breadth_increment:.2%}，保留为 <code>park / needs revision</code>，不据此消耗holdout。完整11-sector历史受 XLC/XLRE成立时间限制，只作为2019年后bridge。</div>

<h2>数据覆盖与可行性</h2>
{context['coverage'].to_html()}
<ul>
<li>可交易主线：Yahoo adjusted close 与 volume；成交量只解释为 ETF 二级市场交易活跃度，不称为 fund flow。</li>
<li>Value/PE：免费历史 sector ETF P/E 缺少可靠 point-in-time 版本，暂时 <code>park</code>。</li>
<li>Call/put spread：免费源缺历史 NBBO、完整 strike/expiry 与可执行 bid/ask，真实收益回测暂时 <code>park</code>。</li>
<li>现金：DTB3 使用上一已知报价计息；本研究组合完全投资且不使用杠杆，现金主要覆盖暖机期。</li>
</ul>

<h2>共同方法</h2>
<ul>
<li>月末实际最后交易日形成信号；下一实际交易日收盘成交；新权重从再下一段 close-to-close 收益开始生效。</li>
<li>持仓按份额漂移；成本按实际非现金买入加卖出名义金额收取。</li>
<li>每个目标权重 long-only、总和100%。使用截至决策日的21日和63日协方差；决策日目标组合的ex-ante几何波动与SPY比值必须位于[0.9, 1.1]。表格另列持仓漂移后的realized波动中位数、比值分位数与合规占比；realized结果不是硬保证。</li>
<li>Sector portfolio 单只ETF上限60%；SPY/XLK归因组合不设额外单项上限，以便显式测量静态科技倾斜。</li>
<li>Regime 使用上一日可知标签：SPY vs 200DMA × 21/63日几何波动 vs 过去756日中位数。</li>
<li>Leakage audit：完整研究路径自动扫描0 blocker；唯一warning来自未被本研究调用的既有CVaR全样本尾部分位数诊断。人工确认前瞻收益只用于评分、所有协方差与成交量统计均为截至决策日的滚动窗口。</li>
</ul>

<h2>一、Legacy 9-sector 长历史验证（2013–2021）</h2>
<p>主候选为12-1、6-1、3-1横截面rank ensemble的Top-3月频组合；成交量版本加入25%的自身历史 dollar-volume surprise rank。等权经同一波动约束后作为风险匹配基线。</p>
<p><strong>Verdict：</strong>三者均未战胜SPY；成交量确认相对纯动量只带来小幅改善，2×成本后仍为负，且不能通过两个验证子期同号门槛。</p>
{format_metrics(context['legacy_metrics']).to_html()}
{legacy_fig}
<h3>子期与区块 bootstrap</h3>{format_table(context['robustness']).to_html()}
<h3>Trend × Vol regime</h3>{format_table(context['legacy_regimes']).to_html()}

<h2>二、完整11-sector bridge</h2>
<p>包含 XLC 与 XLRE；由于可交易共同历史和信号暖机要求，有效评估从2019-08开始，统计强度明显低于9-sector研究。</p>
{format_metrics(context['bridge_metrics']).to_html()}
{bridge_fig}

<h2>三、XLK：硬件 SOXX vs 软件 IGV</h2>
<p>检验63日SOXX–IGV相对动量、软硬件同时强于SPY的breadth、训练期后提出的relative reversal，以及静态risk-matched XLK基线。另用训练期breadth出现频率冻结一个SPY/XLK静态混合，作为breadth的等事前科技暴露基线。</p>
<p><strong>Verdict：</strong>hardware lead失败；relative reversal相对静态XLK没有增量，直接SOXX/IGV winner与contrarian均落后等权。Breadth在两个子期与bootstrap上值得继续检查，但收益集中于单一年份、且相对训练期暴露基线的归因尚不足以升级，因此状态为 <code>park / needs revision</code>，不是 <code>reject</code>，也不是可交易结论。</p>
<h3>信号回归</h3>{format_table(context['tech_signal_stats']).to_html()}
<h3>SPY/XLK overlay</h3>{format_metrics(context['tech_metrics']).to_html()}
{tech_fig}
<h3>子期、区块bootstrap与年度集中度</h3>{format_table(context['tech_robustness']).to_html()}
<h3>Relative reversal / breadth 的配对暴露归因</h3>{format_table(context['tech_pair']).to_html()}
<h3>直接 SOXX/IGV 轮动（非SPY波动匹配，仅诊断）</h3>{format_metrics(context['tech_direct_metrics']).to_html()}
<h3>Regime</h3>{format_table(context['tech_regimes']).to_html()}

<h2>研究文献与数据边界</h2>
<ul>
<li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=463005">Hou — Industry Information Diffusion</a>：行业内信息扩散提供lead/lag机制，但不直接证明SOXX可预测IGV或XLK。</li>
<li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2758776">Cohen & Frazzini — Economic Links and Predictable Returns</a>：经济关联与有限注意力提供先验。</li>
<li><a href="https://www.ishares.com/us/products/239705/ishares-semiconductor-etf">SOXX官方页面</a>、<a href="https://www.ishares.com/us/products/239771/IGV">IGV官方页面</a>、<a href="https://www.ssga.com/mainfund/XLK">XLK官方页面</a>。</li>
<li><a href="https://www.cboe.com/us/options/market_statistics/historical_data/">Cboe历史期权统计</a>只能支持部分聚合诊断；真实价差回测仍需付费历史报价链。</li>
</ul>

<h2>决策规则</h2>
<p>只有在10bp成本后 active CAGR&gt;0、active IR≥0.25、两个验证子期同号、区块bootstrap下界不明显为负、波动合规率100%，且不是由单一年份或静态风险暴露解释时，才允许冻结模型并一次性打开2022+。否则保留集继续封存。</p>
</body></html>"""


def format_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return format_table(frame)


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    percent_columns = {
        "cagr",
        "active_cagr",
        "vol",
        "max_drawdown",
        "alpha",
        "annual_cost_drag",
        "stress_active_cagr",
        "target_ex_ante_vol_compliance",
        "portfolio_realized_vol_median",
        "benchmark_realized_vol_median",
        "realized_vol_band_share",
        "annual_active_mean",
        "ci_90_low",
        "ci_90_high",
        "largest_year_abs_share",
    }
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            if column in percent_columns:
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
            else:
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return display


def format_series(series: pd.Series) -> pd.Series:
    display = series.copy().astype(object)
    for index, value in series.items():
        if index.endswith("_p"):
            display.loc[index] = f"{value:.3f}"
        elif any(token in index for token in ["cagr", "mean", "low", "high"]):
            display.loc[index] = f"{value:.2%}"
        else:
            display.loc[index] = f"{value:.3f}"
    return display


if __name__ == "__main__":
    main()
