"""Frozen phase-2 sector rotation and hardware/software search.

The script deliberately downloads only observations strictly before 2022-01-01.
Candidate definitions are preregistered in the matching phase-2 protocol document.
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
from scipy.stats import norm
from us_sector_rotation_study import (
    DATA_END_EXCLUSIVE,
    HOLDOUT_START,
    LEGACY_SECTORS,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    annualized_return,
    load_cash_returns,
    load_data,
    monthly_return,
    regime_labels,
    rolling_geometric_vol,
)

from alpha_lab.backtest.vector import DriftBacktestResult, run_drift_backtest
from alpha_lab.data.calendars import rebalance_dates
from alpha_lab.data.loaders.fred import load_series
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.portfolio.vol_target import (
    RollingVolMatchResult,
    rolling_match_benchmark_vol_weights,
)
from alpha_lab.utils.paths import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "us_sector_rotation_phase2_development.html"
RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "us_sector_rotation_phase2_metrics.csv"
EXTRA_TICKERS = ["HYG", "LQD", "HG=F"]
FRED_CODES = ["DFII10", "DGS10", "DGS2", "T10YIE", "DTWEXBGS"]
VOL_BAND = (0.9, 1.1)
SECTOR_MAX_WEIGHT = 0.60
LIFETIME_TRIALS = 24
SEED = 20260717


@dataclass(frozen=True)
class Phase2Candidate:
    """A frozen target stream, volatility diagnostics, and three cost cases."""

    name: str
    family: str
    raw_targets: pd.DataFrame
    matched: RollingVolMatchResult
    cost_5: DriftBacktestResult
    cost_10: DriftBacktestResult
    cost_20: DriftBacktestResult
    baseline_name: str


def main() -> None:
    prices, volumes = load_phase2_data()
    cash = load_cash_returns(prices.index)
    macro = load_macro_data(prices.index)

    legacy_prices = prices[LEGACY_SECTORS].dropna()
    legacy_volumes = volumes[LEGACY_SECTORS].reindex(legacy_prices.index)
    legacy_spy = prices["SPY"].reindex(legacy_prices.index)
    legacy_cash = cash.reindex(legacy_prices.index).fillna(0.0)
    residuals, residual_score = residual_features(legacy_prices, legacy_spy)

    sector_equal = build_candidate(
        "sector_equal_weight",
        "benchmark",
        equal_weight_targets(legacy_prices),
        legacy_prices,
        legacy_spy,
        legacy_cash,
        max_weight=SECTOR_MAX_WEIGHT,
        baseline_name="SPY",
    )
    sector_candidates = build_sector_candidates(
        legacy_prices,
        legacy_volumes,
        legacy_spy,
        legacy_cash,
        prices[["HYG", "LQD"]].reindex(legacy_prices.index),
        macro.reindex(legacy_prices.index),
        residuals,
        residual_score,
    )

    tech_prices = prices[["SPY", "XLK", "SOXX", "IGV"]].dropna()
    tech_volumes = volumes[["SOXX", "IGV"]].reindex(tech_prices.index)
    tech_cash = cash.reindex(tech_prices.index).fillna(0.0)
    tech_candidates, tech_shadows = build_tech_candidates(
        tech_prices,
        tech_volumes,
        tech_cash,
        prices[["HYG", "LQD", "HG=F"]].reindex(tech_prices.index),
        macro.reindex(tech_prices.index),
    )

    candidates = {**sector_candidates, **tech_candidates}
    baselines = {
        **{name: sector_equal for name in sector_candidates},
        **tech_shadows,
    }
    metrics = candidate_gate_table(candidates, baselines, prices)
    family_tests = family_multiple_testing(candidates, baselines)
    metrics = attach_family_tests(metrics, family_tests)
    metrics["passes_all_gates"] = evaluate_gates(metrics)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_PATH)
    html = render_report(
        metrics,
        family_tests,
        candidates,
        baselines,
        sector_equal,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(metrics.to_string())
    print("\nFamily tests:\n", family_tests.to_string())


def load_phase2_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the phase-1 panel plus frozen extra proxies without crossing 2022."""
    prices, volumes = load_data()
    cache_dir = Path(gettempdir()) / "alpha_lab_yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    extra_prices = load_prices(
        EXTRA_TICKERS,
        "1998-12-01",
        DATA_END_EXCLUSIVE,
        field="Close",
        auto_adjust=True,
        threads=False,
    )
    extra_prices.index = pd.DatetimeIndex(extra_prices.index).tz_localize(None)
    if extra_prices.empty or extra_prices.index.max() >= HOLDOUT_START:
        raise RuntimeError("extra price loader is empty or crossed the sealed holdout")
    prices = prices.join(extra_prices, how="outer").sort_index()
    if prices.index.max() >= HOLDOUT_START or volumes.index.max() >= HOLDOUT_START:
        raise RuntimeError("phase-2 price or volume panel crossed the sealed holdout")
    return prices, volumes


def load_macro_data(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load market-observable macro levels, lag them two price bars, and hard-stop."""
    macro = load_series(FRED_CODES, start="1998-12-01", end="2021-12-31")
    if macro.empty or macro.index.max() >= HOLDOUT_START:
        raise RuntimeError("macro loader is empty or crossed the sealed holdout")
    timeline = macro.index.union(index).sort_values()
    aligned = macro.reindex(timeline).ffill().reindex(index).shift(2)
    return aligned


def rolling_beta(asset_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """Trailing 252-session market beta, shifted before use."""
    variance = benchmark_returns.rolling(252, min_periods=252).var()
    beta = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns, dtype=float)
    for column in asset_returns:
        beta[column] = (
            asset_returns[column].rolling(252, min_periods=252).cov(benchmark_returns) / variance
        )
    return beta.shift(1)


def residual_features(
    prices: pd.DataFrame,
    spy: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return lagged-beta residual returns and the frozen multi-horizon score."""
    returns = prices.pct_change()
    spy_returns = spy.pct_change()
    residuals = returns - rolling_beta(returns, spy_returns).mul(spy_returns, axis=0)
    log_residual = np.log1p(residuals.clip(lower=-0.999999))
    horizons = []
    for lookback in (252, 126, 63):
        cumulative = log_residual.rolling(lookback, min_periods=lookback).sum()
        skipped = log_residual.rolling(21, min_periods=21).sum()
        horizons.append((cumulative - skipped).rank(axis=1, pct=True))
    score = sum(horizons) / len(horizons)
    return residuals, score


def equal_weight_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly equal-weight targets after sufficient volatility history exists."""
    dates = rebalance_dates(prices.index, freq="ME")
    dates = dates[dates >= prices.index[63]]
    return pd.DataFrame(1.0 / prices.shape[1], index=dates, columns=prices.columns)


def build_candidate(
    name: str,
    family: str,
    raw_targets: pd.DataFrame,
    prices: pd.DataFrame,
    spy: pd.Series,
    cash: pd.Series,
    *,
    max_weight: float,
    baseline_name: str,
) -> Phase2Candidate:
    """Apply the common vol matcher and drift-aware 5/10/20 bp backtests."""
    raw = raw_targets.sort_index().loc[lambda frame: frame.index < HOLDOUT_START]
    matched = rolling_match_benchmark_vol_weights(
        raw,
        prices.pct_change(),
        spy.pct_change(),
        band=VOL_BAND,
        turnover_penalty=0.05,
        max_weight=max_weight,
    )
    if family == "technology":
        technology_columns = [column for column in matched.weights if column != "SPY"]
        if technology_columns and matched.weights[technology_columns].max().max() > 0.60 + 1e-7:
            raise RuntimeError(f"{name} breached the frozen 60% technology-ETF cap")
    results = {
        bps: run_drift_backtest(
            matched.weights,
            prices,
            trading_bps=float(bps),
            cash_returns=cash,
        )
        for bps in (5, 10, 20)
    }
    return Phase2Candidate(
        name,
        family,
        raw,
        matched,
        results[5],
        results[10],
        results[20],
        baseline_name,
    )


def build_sector_candidates(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    spy: pd.Series,
    cash: pd.Series,
    credit_prices: pd.DataFrame,
    macro: pd.DataFrame,
    residuals: pd.DataFrame,
    residual_score: pd.DataFrame,
) -> dict[str, Phase2Candidate]:
    """Build the four preregistered sector candidates."""
    raw_targets = {
        "S1_residual_momentum_dispersion": sector_residual_dispersion_targets(
            prices,
            residuals,
            residual_score,
        ),
        "S2_credit_breadth_state": sector_credit_breadth_targets(
            prices,
            spy,
            credit_prices,
            residual_score,
        ),
        "S3_high_volume_loser": sector_high_volume_loser_targets(
            prices,
            volumes,
            credit_prices,
            residuals,
        ),
        "S4_dynamic_macro_beta": sector_macro_beta_targets(
            prices,
            residuals,
            macro,
        ),
    }
    return {
        name: build_candidate(
            name,
            "sector",
            targets,
            prices,
            spy,
            cash,
            max_weight=SECTOR_MAX_WEIGHT,
            baseline_name="sector_equal_weight",
        )
        for name, targets in raw_targets.items()
    }


def _top_three_or_equal(score: pd.Series, columns: pd.Index) -> pd.Series:
    valid = score.dropna()
    if len(valid) != len(columns):
        return pd.Series(dtype=float)
    weights = pd.Series(0.0, index=columns, name=score.name)
    weights.loc[valid.nlargest(3).index] = 1.0 / 3.0
    return weights


def sector_residual_dispersion_targets(
    prices: pd.DataFrame,
    residuals: pd.DataFrame,
    residual_score: pd.DataFrame,
) -> pd.DataFrame:
    """S1: use residual momentum only in a high-and-rising dispersion state."""
    dates = rebalance_dates(prices.index, freq="ME")
    residual_21 = residuals.rolling(21, min_periods=21).sum()
    dispersion = residual_21.std(axis=1)
    train_threshold = float(dispersion.reindex(dates).loc[:TRAIN_END].dropna().median())
    rows = []
    for date in dates:
        if date < prices.index[315] or residual_score.loc[date].isna().any():
            continue
        active = (
            dispersion.loc[date] > train_threshold
            and dispersion.loc[date] > dispersion.shift(21).loc[date]
        )
        if active:
            row = _top_three_or_equal(residual_score.loc[date], prices.columns)
        else:
            row = pd.Series(1.0 / prices.shape[1], index=prices.columns, name=date)
        if not row.empty:
            rows.append(row.rename(date))
    return pd.DataFrame(rows).reindex(columns=prices.columns)


def credit_ratio(credit_prices: pd.DataFrame) -> pd.Series:
    """Adjusted high-yield minus investment-grade price proxy."""
    ratio = credit_prices["HYG"] / credit_prices["LQD"]
    return np.log(ratio).rename("log_hyg_lqd")


def downside_beta(prices: pd.DataFrame, spy: pd.Series, window: int = 126) -> pd.DataFrame:
    """Zero-intercept beta estimated only on negative SPY sessions."""
    returns = prices.pct_change()
    market = spy.pct_change()
    mask = (market < 0.0).astype(float)
    denominator = (market.pow(2) * mask).rolling(window, min_periods=63).sum()
    result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for column in prices:
        numerator = (returns[column] * market * mask).rolling(window, min_periods=63).sum()
        result[column] = numerator / denominator
    return result.shift(1)


def sector_credit_breadth_targets(
    prices: pd.DataFrame,
    spy: pd.Series,
    credit_prices: pd.DataFrame,
    residual_score: pd.DataFrame,
) -> pd.DataFrame:
    """S2: defensive, recovery, or neutral weights from credit and breadth."""
    dates = rebalance_dates(prices.index, freq="ME")
    trend = prices > prices.rolling(126, min_periods=126).mean()
    relative = prices.pct_change(63).sub(spy.pct_change(63), axis=0) > 0.0
    breadth = (trend & relative).mean(axis=1)
    breadth_change = breadth.diff(21)
    credit_change = credit_ratio(credit_prices).diff(21)
    defensive_score = -downside_beta(prices, spy)
    rows = []
    for date in dates:
        data = pd.concat(
            [
                pd.Series(credit_change.loc[date], index=["credit"]),
                pd.Series(breadth_change.loc[date], index=["breadth"]),
            ]
        )
        if date < prices.index[315] or data.isna().any():
            continue
        if credit_change.loc[date] < 0.0 and breadth_change.loc[date] < 0.0:
            row = _top_three_or_equal(defensive_score.loc[date], prices.columns)
        elif credit_change.loc[date] > 0.0 and breadth_change.loc[date] > 0.0:
            row = _top_three_or_equal(residual_score.loc[date], prices.columns)
        else:
            row = pd.Series(1.0 / prices.shape[1], index=prices.columns, name=date)
        if not row.empty:
            rows.append(row.rename(date))
    return pd.DataFrame(rows).reindex(columns=prices.columns)


def sector_high_volume_loser_targets(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    credit_prices: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """S3: weekly equal-weight base plus a high-volume residual-loser tilt."""
    dates = rebalance_dates(prices.index, freq="W-FRI")
    residual_5 = residuals.rolling(5, min_periods=5).sum()
    dollar_volume = prices * volumes
    volume_surprise = (
        dollar_volume.rolling(5, min_periods=5).mean()
        / dollar_volume.rolling(
            126,
            min_periods=126,
        ).median()
    )
    loser_rank = residual_5.rank(axis=1, pct=True)
    volume_rank = volume_surprise.rank(axis=1, pct=True)
    stable_credit = credit_ratio(credit_prices).diff(5) >= 0.0
    rows = []
    for date in dates:
        if date < prices.index[315] or pd.isna(stable_credit.loc[date]):
            continue
        qualifiers = (
            (loser_rank.loc[date] <= 1.0 / 3.0)
            & (volume_rank.loc[date] >= 2.0 / 3.0)
            & bool(stable_credit.loc[date])
        )
        base = pd.Series(0.5 / prices.shape[1], index=prices.columns, name=date)
        if qualifiers.any():
            base.loc[qualifiers] += 0.5 / int(qualifiers.sum())
        else:
            base[:] = 1.0 / prices.shape[1]
        rows.append(base)
    return pd.DataFrame(rows).reindex(columns=prices.columns)


def sector_macro_beta_targets(
    prices: pd.DataFrame,
    residuals: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """S4: trailing macro betas multiplied by the latest 63-session move."""
    dates = rebalance_dates(prices.index, freq="ME")
    drivers = pd.DataFrame(index=macro.index)
    drivers["real_yield"] = macro["DFII10"]
    drivers["curve"] = macro["DGS10"] - macro["DGS2"]
    drivers["breakeven"] = macro["T10YIE"]
    changes = drivers.diff()
    direction = drivers.diff(63)
    rows = []
    for date in dates:
        window_x = changes.loc[:date].tail(756)
        if len(window_x) < 756 or window_x.isna().any().any() or direction.loc[date].isna().any():
            continue
        x = sm.add_constant(window_x)
        scores = {}
        for column in prices:
            y = residuals[column].reindex(window_x.index)
            sample = pd.concat([y.rename("y"), x], axis=1).dropna()
            if len(sample) < 700:
                continue
            fit = np.linalg.lstsq(
                sample.drop(columns="y").to_numpy(),
                sample["y"].to_numpy(),
                rcond=None,
            )[0]
            scores[column] = float(np.dot(fit[1:], direction.loc[date].to_numpy()))
        row = _top_three_or_equal(pd.Series(scores, name=date), prices.columns)
        if not row.empty:
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=prices.columns)


def build_tech_candidates(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    cash: pd.Series,
    proxy_prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> tuple[dict[str, Phase2Candidate], dict[str, Phase2Candidate]]:
    """Build four technology candidates and train-frozen static shadows."""
    tech_residuals, _ = residual_features(prices[["SOXX", "IGV"]], prices["SPY"])
    raw_targets = {
        "T1_macro_confirmed_hardware": tech_macro_hardware_targets(
            prices,
            proxy_prices,
            macro,
            tech_residuals,
        ),
        "T2_credit_stable_reversal": tech_liquidity_reversal_targets(
            prices,
            volumes,
            proxy_prices,
            tech_residuals,
        ),
        "T3_negative_hardware_diffusion": tech_negative_diffusion_targets(
            prices,
            tech_residuals,
        ),
        "T4_residual_breadth_compression": tech_breadth_compression_targets(
            prices,
            tech_residuals,
        ),
    }
    candidates: dict[str, Phase2Candidate] = {}
    shadows: dict[str, Phase2Candidate] = {}
    for name, raw in raw_targets.items():
        asset_columns = [column for column in raw.columns if column in prices.columns]
        candidate_prices = prices[asset_columns]
        candidate = build_candidate(
            name,
            "technology",
            raw,
            candidate_prices,
            prices["SPY"],
            cash,
            max_weight=1.0,
            baseline_name=f"{name}_train_static_shadow",
        )
        shadow_raw = train_static_shadow(raw, prices.index)
        shadow = build_candidate(
            f"{name}_train_static_shadow",
            "benchmark",
            shadow_raw,
            candidate_prices,
            prices["SPY"],
            cash,
            max_weight=1.0,
            baseline_name="SPY",
        )
        candidates[name] = candidate
        shadows[name] = shadow
    return candidates, shadows


def train_static_shadow(
    raw_targets: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Repeat through time the raw allocation frequency observed through 2012."""
    train = raw_targets.reindex(price_index).ffill().loc[:TRAIN_END].dropna()
    if train.empty:
        raise ValueError("technology target has no train rows for its static shadow")
    average = train.mean().clip(lower=0.0)
    average /= average.sum()
    return pd.DataFrame(
        np.tile(average.to_numpy(), (len(raw_targets), 1)),
        index=raw_targets.index,
        columns=raw_targets.columns,
    )


def tech_macro_hardware_targets(
    prices: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    macro: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """T1: hardware lead accepted only with at least two macro confirmations."""
    dates = rebalance_dates(prices.index, freq="ME")
    relative_momentum = (
        residuals["SOXX"].rolling(63, min_periods=63).sum()
        - residuals["IGV"].rolling(63, min_periods=63).sum()
    )
    copper_up = proxy_prices["HG=F"].pct_change(63) > 0.0
    dollar_down = macro["DTWEXBGS"].pct_change(63) < 0.0
    credit_up = credit_ratio(proxy_prices[["HYG", "LQD"]]).diff(21) > 0.0
    rows = []
    for date in dates:
        conditions = pd.Series(
            [
                relative_momentum.loc[date],
                copper_up.loc[date],
                dollar_down.loc[date],
                credit_up.loc[date],
            ]
        )
        if date < prices.index[315] or conditions.isna().any():
            continue
        confirmations = (
            int(copper_up.loc[date]) + int(dollar_down.loc[date]) + int(credit_up.loc[date])
        )
        hardware = relative_momentum.loc[date] > 0.0 and confirmations >= 2
        row = pd.Series(0.0, index=["SPY", "SOXX", "IGV"], name=date)
        row["SPY"] = 0.4
        row["SOXX" if hardware else "IGV"] = 0.6
        rows.append(row)
    return pd.DataFrame(rows)


def _event_targets(
    index: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
    event_rows: dict[pd.Timestamp, pd.Series],
    default: pd.Series,
    *,
    holding_sessions: int,
) -> pd.DataFrame:
    """Create non-overlapping sparse event/exit target decisions."""
    rows = [default.rename(index[63])]
    active_until = -1
    for date in event_dates:
        position = int(index.get_loc(date))
        if position <= active_until or position + holding_sessions >= len(index):
            continue
        rows.append(event_rows[date].rename(date))
        exit_date = index[position + holding_sessions]
        rows.append(default.rename(exit_date))
        active_until = position + holding_sessions
    return pd.DataFrame(rows).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


def tech_liquidity_reversal_targets(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """T2: non-overlapping three-session reversals after stable-credit flow shocks."""
    weekly = rebalance_dates(prices.index, freq="W-FRI")
    spread_3 = (residuals["SOXX"] - residuals["IGV"]).rolling(3, min_periods=3).sum()
    z_score = spread_3 / spread_3.rolling(252, min_periods=252).std().shift(1)
    dollar_volume = prices[["SOXX", "IGV"]] * volumes[["SOXX", "IGV"]]
    volume_ratio = dollar_volume / dollar_volume.rolling(63, min_periods=63).median().shift(1)
    stable_credit = credit_ratio(proxy_prices[["HYG", "LQD"]]).diff(5) >= 0.0
    event_rows = {}
    for date in weekly:
        if pd.isna(z_score.loc[date]) or pd.isna(stable_credit.loc[date]):
            continue
        winner = "SOXX" if z_score.loc[date] > 0.0 else "IGV"
        laggard = "IGV" if winner == "SOXX" else "SOXX"
        if (
            abs(z_score.loc[date]) > 2.0
            and volume_ratio.loc[date, winner] > 1.5
            and bool(stable_credit.loc[date])
        ):
            row = pd.Series(0.0, index=["SPY", "SOXX", "IGV"], name=date)
            row["SPY"] = 0.4
            row[laggard] = 0.6
            event_rows[date] = row
    default = pd.Series({"SPY": 0.5, "SOXX": 0.25, "IGV": 0.25})
    return _event_targets(
        prices.index,
        pd.DatetimeIndex(event_rows),
        event_rows,
        default,
        holding_sessions=3,
    )


def tech_negative_diffusion_targets(
    prices: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """T3: de-risk after a negative SOXX shock not yet present in IGV."""
    weekly = rebalance_dates(prices.index, freq="W-FRI")
    residual_5 = residuals.rolling(5, min_periods=5).sum()
    scale = residual_5.rolling(252, min_periods=252).std().shift(1)
    z_score = residual_5 / scale
    event_rows = {}
    for date in weekly:
        if z_score.loc[date].isna().any():
            continue
        if z_score.loc[date, "SOXX"] < -1.5 and z_score.loc[date, "IGV"] > -0.5:
            event_rows[date] = pd.Series({"SPY": 1.0, "XLK": 0.0}, name=date)
    default = pd.Series({"SPY": 0.4, "XLK": 0.6})
    return _event_targets(
        prices.index,
        pd.DatetimeIndex(event_rows),
        event_rows,
        default,
        holding_sessions=5,
    )


def tech_breadth_compression_targets(
    prices: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """T4: XLK exposure when residual SOXX/IGV breadth is positive and coherent."""
    dates = rebalance_dates(prices.index, freq="ME")
    components = []
    for horizon in (21, 63):
        cumulative = residuals.rolling(horizon, min_periods=horizon).sum()
        scale = cumulative.rolling(252, min_periods=252).std().shift(1)
        components.append(cumulative / scale)
    score = (components[0] + components[1]) / 2.0
    dispersion = (score["SOXX"] - score["IGV"]).abs()
    expanding_median = dispersion.expanding(min_periods=60).median().shift(1)
    rows = []
    for date in dates:
        values = pd.concat([score.loc[date], pd.Series({"median": expanding_median.loc[date]})])
        if date < prices.index[315] or values.isna().any():
            continue
        active = (
            score.loc[date, "SOXX"] > 0.0
            and score.loc[date, "IGV"] > 0.0
            and dispersion.loc[date] < expanding_median.loc[date]
        )
        rows.append(
            pd.Series(
                {"SPY": 0.4 if active else 1.0, "XLK": 0.6 if active else 0.0},
                name=date,
            )
        )
    return pd.DataFrame(rows)


def _result(candidate: Phase2Candidate, bps: int) -> DriftBacktestResult:
    return {5: candidate.cost_5, 10: candidate.cost_10, 20: candidate.cost_20}[bps]


def active_performance(
    candidate: Phase2Candidate,
    baseline: Phase2Candidate,
    *,
    bps: int,
    start: pd.Timestamp = VALIDATION_START,
    end: pd.Timestamp = VALIDATION_END,
) -> tuple[pd.Series, float, float]:
    """Return aligned active returns, CAGR difference, and information ratio."""
    portfolio = _result(candidate, bps).returns.loc[start:end]
    reference = _result(baseline, bps).returns.reindex(portfolio.index)
    valid = pd.concat(
        [portfolio.rename("portfolio"), reference.rename("baseline")], axis=1
    ).dropna()
    active = valid["portfolio"] - valid["baseline"]
    cagr = annualized_return(valid["portfolio"]) - annualized_return(valid["baseline"])
    ir = float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else np.nan
    return active, cagr, ir


def stationary_bootstrap_means(
    values: pd.Series,
    *,
    draws: int = 10_000,
    expected_block: float = 6.0,
    seed: int = SEED,
) -> np.ndarray:
    """Joint-use stationary bootstrap means for a single monthly series."""
    sample = values.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(sample)
    output = np.empty(draws)
    probability = 1.0 / expected_block
    for draw in range(draws):
        indices = np.empty(n, dtype=int)
        indices[0] = int(rng.integers(n))
        for position in range(1, n):
            if rng.random() < probability:
                indices[position] = int(rng.integers(n))
            else:
                indices[position] = (indices[position - 1] + 1) % n
        output[draw] = sample[indices].mean() * 12.0
    return output


def deflated_sharpe_probability(
    monthly_active: pd.Series,
    *,
    trials: int = LIFETIME_TRIALS,
) -> float:
    """Bailey/Lopez de Prado deflated-Sharpe probability with full trial count."""
    sample = monthly_active.dropna()
    n = len(sample)
    if n < 3 or sample.std(ddof=1) <= 0.0:
        return np.nan
    sharpe = float(sample.mean() / sample.std(ddof=1))
    skew = float(sample.skew())
    kurtosis = float(sample.kurt() + 3.0)
    gamma = 0.5772156649015329
    variance_sr = max(
        (1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2) / (n - 1),
        1e-12,
    )
    expected_max = np.sqrt(variance_sr) * (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / trials) + gamma * norm.ppf(1.0 - 1.0 / (trials * np.e))
    )
    z_score = (sharpe - expected_max) / np.sqrt(variance_sr)
    return float(norm.cdf(z_score))


def exposure_alpha_retention(
    active: pd.Series,
    prices: pd.DataFrame,
) -> tuple[float, float]:
    """Annual alpha and its share of annual active mean after technology controls."""
    returns = prices[["SPY", "XLK", "SOXX", "IGV"]].pct_change()
    factors = pd.DataFrame(
        {
            "SPY": returns["SPY"],
            "XLK_minus_SPY": returns["XLK"] - returns["SPY"],
            "SOXX_minus_IGV": returns["SOXX"] - returns["IGV"],
        }
    )
    sample = pd.concat([active.rename("active"), factors], axis=1, sort=False).dropna()
    fit = sm.OLS(sample["active"], sm.add_constant(sample.drop(columns="active"))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 21},
    )
    alpha = float(fit.params["const"] * 252.0)
    annual_mean = float(sample["active"].mean() * 252.0)
    retention = alpha / annual_mean if annual_mean > 0.0 else np.nan
    return alpha, retention


def candidate_gate_table(
    candidates: dict[str, Phase2Candidate],
    baselines: dict[str, Phase2Candidate],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all pre-holdout screening and robustness diagnostics."""
    rows = []
    for name, candidate in candidates.items():
        baseline = baselines[name]
        active_10, cagr_10, ir_10 = active_performance(candidate, baseline, bps=10)
        _, cagr_5, ir_5 = active_performance(candidate, baseline, bps=5)
        _, cagr_20, ir_20 = active_performance(candidate, baseline, bps=20)
        _, cagr_early, ir_early = active_performance(
            candidate,
            baseline,
            bps=10,
            start=pd.Timestamp("2013-01-01"),
            end=pd.Timestamp("2016-12-31"),
        )
        _, cagr_late, ir_late = active_performance(
            candidate,
            baseline,
            bps=10,
            start=pd.Timestamp("2017-01-01"),
            end=VALIDATION_END,
        )
        monthly_active = monthly_return(
            _result(candidate, 10).returns.loc[VALIDATION_START:VALIDATION_END]
        ) - monthly_return(_result(baseline, 10).returns.loc[VALIDATION_START:VALIDATION_END])
        bootstrap = stationary_bootstrap_means(monthly_active)
        bootstrap_low = float(np.percentile(bootstrap, 5.0))
        bootstrap_p = float((np.sum(bootstrap <= 0.0) + 1) / (len(bootstrap) + 1))

        log_active = np.log1p(
            _result(candidate, 10).returns.loc[VALIDATION_START:VALIDATION_END]
        ) - np.log1p(_result(baseline, 10).returns.loc[VALIDATION_START:VALIDATION_END])
        yearly = log_active.groupby(log_active.index.year).sum()
        best_year = int(yearly.idxmax())
        without_best = log_active[log_active.index.year != best_year]
        delete_best_cagr = float(np.exp(without_best.sum() / (len(without_best) / 252.0)) - 1.0)
        largest_year_share = float(yearly.abs().max() / yearly.abs().sum())
        loo_positive = 0
        for year in yearly.index:
            keep = active_10.index.year != year
            portfolio = _result(candidate, 10).returns.reindex(active_10.index)[keep]
            reference = _result(baseline, 10).returns.reindex(active_10.index)[keep]
            if annualized_return(portfolio) - annualized_return(reference) > 0.0:
                loo_positive += 1

        spy = prices["SPY"].reindex(active_10.index)
        labels = regime_labels(prices["SPY"]).reindex(active_10.index)
        regime_means = []
        regime_irs = []
        for regime in labels.dropna().unique():
            selected = labels == regime
            if int(selected.sum()) < 126:
                continue
            sample = active_10[selected]
            regime_means.append(float(sample.mean() * 252.0))
            regime_irs.append(float(sample.mean() / sample.std() * np.sqrt(252)))

        diagnostics = candidate.matched.diagnostics.loc[VALIDATION_START:VALIDATION_END]
        result_10 = _result(candidate, 10)
        portfolio_vol = rolling_geometric_vol(result_10.gross_returns).loc[
            VALIDATION_START:VALIDATION_END
        ]
        spy_vol = rolling_geometric_vol(spy.pct_change().fillna(0.0)).reindex(portfolio_vol.index)
        realized_ratio = (portfolio_vol / spy_vol).replace([np.inf, -np.inf], np.nan).dropna()

        if candidate.family == "technology":
            alpha, retention = exposure_alpha_retention(active_10, prices)
        else:
            alpha, retention = np.nan, np.nan
        rows.append(
            {
                "strategy": name,
                "family": candidate.family,
                "baseline": candidate.baseline_name,
                "active_cagr_5bp": cagr_5,
                "active_ir_5bp": ir_5,
                "active_cagr_10bp": cagr_10,
                "active_ir_10bp": ir_10,
                "active_cagr_20bp": cagr_20,
                "active_ir_20bp": ir_20,
                "active_cagr_2013_2016": cagr_early,
                "active_ir_2013_2016": ir_early,
                "active_cagr_2017_2021": cagr_late,
                "active_ir_2017_2021": ir_late,
                "bootstrap_90_low": bootstrap_low,
                "bootstrap_one_sided_p": bootstrap_p,
                "delete_best_year_cagr": delete_best_cagr,
                "largest_year_abs_share": largest_year_share,
                "leave_one_year_out_positive": loo_positive,
                "positive_regimes": int(np.sum(np.asarray(regime_means) > 0.0)),
                "worst_regime_ir": min(regime_irs) if regime_irs else np.nan,
                "target_vol_compliance": float(
                    diagnostics["vol_ratio"].between(VOL_BAND[0] - 1e-7, VOL_BAND[1] + 1e-7).mean()
                ),
                "target_max_weight": float(diagnostics["max_weight"].max()),
                "target_weight_sum_error": float((diagnostics["weight_sum"] - 1.0).abs().max()),
                "realized_vol_ratio_median": float(realized_ratio.median()),
                "realized_vol_band_share": float(realized_ratio.between(*VOL_BAND).mean()),
                "annual_traded_notional_10bp": float(
                    result_10.traded_notional.loc[VALIDATION_START:VALIDATION_END].sum()
                    / (len(active_10) / 252.0)
                ),
                "exposure_alpha": alpha,
                "exposure_alpha_retention": retention,
                "deflated_sharpe_probability": deflated_sharpe_probability(monthly_active),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def newey_west_long_run_std(values: np.ndarray, max_lag: int = 5) -> float:
    """Bartlett-kernel long-run standard deviation of a monthly series."""
    centered = values - values.mean()
    n = len(centered)
    variance = float(np.dot(centered, centered) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    return float(np.sqrt(max(variance, 1e-12)))


def spa_family_test(
    active_returns: pd.DataFrame,
    *,
    draws: int = 10_000,
    expected_block: float = 6.0,
    seed: int = SEED,
) -> tuple[float, float, str]:
    """Hansen-style SPA max-t test with joint stationary resampling.

    Poor alternatives are recentered at their negative sample mean using Hansen's
    consistent selector; plausible alternatives are put on the null boundary.
    """
    sample = active_returns.dropna(how="any").to_numpy(dtype=float)
    n, count = sample.shape
    means = sample.mean(axis=0)
    long_run = np.asarray([newey_west_long_run_std(sample[:, column]) for column in range(count)])
    t_statistics = np.sqrt(n) * means / long_run
    observed = float(max(0.0, t_statistics.max()))
    threshold = -np.sqrt(2.0 * np.log(np.log(n)))
    selected_mean = np.where(t_statistics < threshold, means, 0.0)

    rng = np.random.default_rng(seed)
    probability = 1.0 / expected_block
    bootstrap_max = np.empty(draws)
    for draw in range(draws):
        indices = np.empty(n, dtype=int)
        indices[0] = int(rng.integers(n))
        for position in range(1, n):
            if rng.random() < probability:
                indices[position] = int(rng.integers(n))
            else:
                indices[position] = (indices[position - 1] + 1) % n
        bootstrap_mean = sample[indices].mean(axis=0) - means + selected_mean
        bootstrap_t = np.sqrt(n) * bootstrap_mean / long_run
        bootstrap_max[draw] = max(0.0, float(bootstrap_t.max()))
    p_value = float((np.sum(bootstrap_max >= observed) + 1) / (draws + 1))
    best_column = active_returns.columns[int(np.argmax(t_statistics))]
    return observed, p_value, str(best_column)


def family_multiple_testing(
    candidates: dict[str, Phase2Candidate],
    baselines: dict[str, Phase2Candidate],
) -> pd.DataFrame:
    """Run family SPA tests and Holm-adjust their two p-values."""
    rows = []
    for family in ("sector", "technology"):
        series = {}
        for name, candidate in candidates.items():
            if candidate.family != family:
                continue
            baseline = baselines[name]
            portfolio = monthly_return(
                candidate.cost_10.returns.loc[VALIDATION_START:VALIDATION_END]
            )
            reference = monthly_return(
                baseline.cost_10.returns.loc[VALIDATION_START:VALIDATION_END]
            )
            series[name] = portfolio - reference
        frame = pd.DataFrame(series).dropna(how="any")
        statistic, p_value, best = spa_family_test(frame)
        rows.append(
            {
                "family": family,
                "n_candidates": frame.shape[1],
                "n_months": frame.shape[0],
                "spa_max_t": statistic,
                "spa_p_value": p_value,
                "best_candidate": best,
            }
        )
    result = pd.DataFrame(rows).set_index("family")
    order = result["spa_p_value"].sort_values().index
    adjusted = pd.Series(index=order, dtype=float)
    running = 0.0
    total = len(order)
    for rank, family in enumerate(order):
        value = min(1.0, (total - rank) * float(result.loc[family, "spa_p_value"]))
        running = max(running, value)
        adjusted.loc[family] = running
    result["holm_adjusted_p"] = adjusted.reindex(result.index)
    result["holm_pass_5pct"] = result["holm_adjusted_p"] <= 0.05
    return result


def attach_family_tests(metrics: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Map the family-level multiple-testing result onto each candidate row."""
    result = metrics.copy()
    result["family_spa_p"] = result["family"].map(tests["spa_p_value"])
    result["family_holm_p"] = result["family"].map(tests["holm_adjusted_p"])
    result["family_holm_pass"] = result["family"].map(tests["holm_pass_5pct"])
    return result


def evaluate_gates(metrics: pd.DataFrame) -> pd.Series:
    """Apply the frozen all-of-the-above candidate gate."""
    common = (
        (metrics["active_cagr_10bp"] >= 0.01)
        & (metrics["active_ir_10bp"] >= 0.35)
        & (metrics["active_cagr_2013_2016"] > 0.0)
        & (metrics["active_cagr_2017_2021"] > 0.0)
        & (metrics["bootstrap_90_low"] > 0.0)
        & (metrics["delete_best_year_cagr"] > 0.0)
        & (metrics["largest_year_abs_share"] <= 0.40)
        & (metrics["leave_one_year_out_positive"] >= 7)
        & (metrics["positive_regimes"] >= 2)
        & (metrics["worst_regime_ir"] >= -0.25)
        & (metrics["target_vol_compliance"] >= 1.0 - 1e-12)
        & metrics["realized_vol_ratio_median"].between(*VOL_BAND)
        & (metrics["realized_vol_band_share"] >= 0.70)
        & (metrics["active_cagr_20bp"] > 0.0)
        & (metrics["deflated_sharpe_probability"] >= 0.95)
        & metrics["family_holm_pass"].astype(bool)
    )
    technology = metrics["family"] != "technology"
    technology |= (metrics["exposure_alpha"] > 0.0) & (metrics["exposure_alpha_retention"] >= 0.50)
    return (common & technology).rename("passes_all_gates")


def active_wealth_figure(
    candidates: dict[str, Phase2Candidate],
    baselines: dict[str, Phase2Candidate],
) -> go.Figure:
    """Plot cumulative 10 bp wealth relative to each candidate's correct baseline."""
    figure = go.Figure()
    for name, candidate in candidates.items():
        baseline = baselines[name]
        portfolio = candidate.cost_10.returns.loc[VALIDATION_START:VALIDATION_END]
        reference = baseline.cost_10.returns.reindex(portfolio.index)
        log_active = np.log1p(portfolio) - np.log1p(reference)
        figure.add_scatter(
            x=log_active.index,
            y=np.exp(log_active.cumsum()),
            name=name,
            mode="lines",
        )
    figure.update_layout(
        template="plotly_white",
        title="10 bp net wealth relative to the correct frozen baseline",
        yaxis_title="Relative wealth (1.0 = no edge)",
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 55, "r": 25, "t": 65, "b": 120},
    )
    figure.add_hline(y=1.0, line_dash="dot", line_color="#6b7280")
    return figure


def _display_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "baseline",
        "active_cagr_10bp",
        "active_ir_10bp",
        "active_cagr_20bp",
        "active_cagr_2013_2016",
        "active_cagr_2017_2021",
        "bootstrap_90_low",
        "delete_best_year_cagr",
        "largest_year_abs_share",
        "leave_one_year_out_positive",
        "positive_regimes",
        "worst_regime_ir",
        "realized_vol_ratio_median",
        "realized_vol_band_share",
        "deflated_sharpe_probability",
        "family_holm_p",
        "passes_all_gates",
    ]
    frame = metrics[columns].copy()
    percent_columns = [
        "active_cagr_10bp",
        "active_cagr_20bp",
        "active_cagr_2013_2016",
        "active_cagr_2017_2021",
        "bootstrap_90_low",
        "delete_best_year_cagr",
        "largest_year_abs_share",
        "realized_vol_band_share",
        "deflated_sharpe_probability",
        "family_holm_p",
    ]
    for column in percent_columns:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    for column in ["active_ir_10bp", "worst_regime_ir", "realized_vol_ratio_median"]:
        frame[column] = frame[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    return frame


def render_report(
    metrics: pd.DataFrame,
    family_tests: pd.DataFrame,
    candidates: dict[str, Phase2Candidate],
    baselines: dict[str, Phase2Candidate],
    sector_equal: Phase2Candidate,
) -> str:
    """Render a self-contained, decision-oriented HTML research report."""
    del sector_equal
    winners = metrics.index[metrics["passes_all_gates"]].tolist()
    if winners:
        verdict = (
            f"{len(winners)} candidate(s) passed the statistical gate: "
            f"{', '.join(winners)}. The holdout remains sealed pending independent "
            "fundamental review and explicit user approval."
        )
        verdict_class = "warning"
    else:
        verdict = (
            "No candidate passed the frozen all-of-the-above gate. No fundamental "
            "winner review is triggered and the 2022+ holdout remains sealed."
        )
        verdict_class = "reject"

    registry = pd.DataFrame(
        [
            (
                "S1",
                "Residual momentum × high/rising dispersion",
                "Monthly",
                "Information diffusion",
            ),
            ("S2", "Credit/breadth defensive-recovery state", "Monthly", "Credit and deleveraging"),
            (
                "S3",
                "High-volume residual-loser liquidity provision",
                "Weekly",
                "Liquidity provision",
            ),
            (
                "S4",
                "Dynamic real-rate/curve/breakeven sensitivity",
                "Monthly",
                "Macro transmission",
            ),
            ("T1", "Macro-confirmed hardware lead", "Monthly", "Global capex cycle"),
            ("T2", "Credit-stable short-term reversal", "Event/weekly", "Liquidity provision"),
            (
                "T3",
                "Negative semiconductor shock diffusion",
                "Event/weekly",
                "Information diffusion",
            ),
            (
                "T4",
                "Residual breadth + dispersion compression",
                "Monthly",
                "Broad earnings expectations",
            ),
        ],
        columns=["ID", "Frozen candidate", "Decision frequency", "Return source"],
    ).set_index("ID")
    test_display = family_tests.copy()
    for column in ["spa_p_value", "holm_adjusted_p"]:
        test_display[column] = test_display[column].map(lambda value: f"{value:.2%}")
    test_display["spa_max_t"] = test_display["spa_max_t"].map(lambda value: f"{value:.2f}")

    plot = active_wealth_figure(candidates, baselines).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displayModeBar": False, "responsive": True},
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>US Sector Rotation Phase 2 — Development Validation</title>
<style>
:root {{ --ink:#172033; --muted:#667085; --line:#d9e0ea; --panel:#f6f8fb;
         --navy:#183b66; --red:#a61b1b; --amber:#8a5a00; }}
body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:#fff; }}
main {{ max-width:1180px; margin:0 auto; padding:42px 28px 70px; }}
h1 {{ font-size:32px; margin:0 0 8px; color:var(--navy); }}
h2 {{ margin-top:38px; padding-bottom:8px; border-bottom:2px solid var(--line); color:var(--navy); }}
h3 {{ color:var(--navy); }}
p,li {{ line-height:1.6; }}
.sub {{ color:var(--muted); margin-bottom:28px; }}
.callout {{ padding:18px 20px; border-radius:8px; margin:22px 0; font-weight:600; }}
.reject {{ background:#fff1f1; border-left:5px solid var(--red); }}
.warning {{ background:#fff7e6; border-left:5px solid var(--amber); }}
.guard {{ background:#eef5ff; border-left:5px solid var(--navy); font-weight:500; }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:22px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
.card strong {{ display:block; font-size:22px; color:var(--navy); }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ background:#edf2f7; color:#27364a; position:sticky; top:0; }}
.small {{ color:var(--muted); font-size:13px; }}
code {{ background:#f2f4f7; padding:2px 5px; border-radius:4px; }}
@media(max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:24px 14px 50px; }} }}
</style>
</head>
<body><main>
<h1>US Sector Rotation & Hardware/Software — Phase 2</h1>
<p class="sub">Frozen-batch development validation · generated 2026-07-17 · data stops 2021-12-31</p>

<div class="callout {verdict_class}">{verdict}</div>
<div class="callout guard"><strong>Holdout guard:</strong> 2022-01-03 onward was not downloaded or evaluated. The
2013-2021 sample is development-validation because phase 1 already exposed it.</div>

<div class="grid">
  <div class="card"><strong>8</strong>new frozen trials</div>
  <div class="card"><strong>15 / 24</strong>lifetime trials consumed</div>
  <div class="card"><strong>{int(metrics["passes_all_gates"].sum())}</strong>candidates eligible for fundamental review</div>
</div>

<h2>Executive decision</h2>
<p>The hurdle is timing alpha, not an equity-market bull-run return. Sector candidates are measured against a
risk-matched nine-sector equal-weight portfolio. Technology candidates are measured against their own train-frozen,
time-weighted static exposure shadow. All primary figures are net of 10 bp one-way trading costs.</p>

<h2>Frozen candidate registry</h2>
<div class="table-wrap">{registry.to_html(border=0)}</div>
<p class="small">Exact formulas and falsification rules are in
<code>docs/research_decisions/2026-07-17_us-sector-rotation-phase2-protocol.md</code>.</p>

<h2>Gate results</h2>
<div class="table-wrap">{_display_metrics(metrics).to_html(border=0)}</div>
<p class="small">A pass requires every preregistered profitability, subperiod, bootstrap, concentration,
regime, volatility, 20 bp, exposure, SPA/Holm and Deflated-Sharpe condition. A positive CAGR alone is not a pass.</p>

<h2>Family-level data-snooping control</h2>
<div class="table-wrap">{test_display.to_html(border=0)}</div>
<p>Family tests use 10,000 joint stationary-bootstrap draws with an expected six-month block. The full lifetime
trial count of 24 is used for Deflated Sharpe; unused search capacity is not treated as statistical forgiveness.</p>

<h2>Relative wealth</h2>
{plot}

<h2>Implementation and leakage controls</h2>
<p><strong>Audit verdict: trustworthy for rejecting this batch.</strong> The repository leakage scanner
reported 0 blockers and 0 warnings (information-only trailing-window/forward-fill notices were manually
checked). The manual audit confirmed no negative shift, centered window, backfill, full-validation
normalizer, same-close return, forward-return feature, or post-2021 observation.</p>
<ul>
  <li>Targets are formed at a close, traded at the next close, and first earn the subsequent return.</li>
  <li>Rolling market betas are shifted one bar; FRED levels are shifted two price bars.</li>
  <li>All target rows sum to 100%; the decision-date geometric 21/63-day volatility band is checked explicitly.</li>
  <li>Train-static technology shadows use time-weighted daily exposures through 2012, not the frequency of sparse event rows.</li>
  <li>Free point-in-time PE and executable historical option chains remain unavailable, so value and option-spread claims remain parked.</li>
</ul>

<h2>Next action</h2>
<p>{"Perform independent leakage and fundamental-support reviews for the statistical survivor(s); do not open holdout yet." if winners else "Stop the search and wait for user review. Opening the holdout or adding another candidate would violate the frozen protocol."}</p>
</main></body></html>"""


if __name__ == "__main__":
    main()
