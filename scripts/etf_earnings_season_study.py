"""Pre-2022 ETF study of earnings-season calendar proxies.

The study deliberately separates two questions:

1. Do fixed, knowable quarter-start calendar windows carry unusual ETF returns?
2. Does the earnings-season label improve otherwise simple ETF allocation rules?

It does not claim that a fixed window identifies actual earnings-announcement
clusters. Exact cluster tests require point-in-time company announcement dates and
release-session data, which are not available in the repository.
"""

from __future__ import annotations

import base64
import io
import json
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from alpha_lab.analytics.returns import drawdown_duration_metrics
from alpha_lab.backtest.metrics import summary
from alpha_lab.utils.paths import PROJECT_ROOT

PRICE_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "etf_strategy_50plus_pre2022"
    / "market_prices_adjusted_pre2022.parquet"
)
OUT = PROJECT_ROOT / "data" / "results" / "etf_earnings_season_pre2022"
REPORT = PROJECT_ROOT / "reports" / "etf_earnings_season_pre2022.html"

LAST_ALLOWED = pd.Timestamp("2021-12-31")
DEVELOPMENT_END = pd.Timestamp("2012-12-31")
VALIDATION_START = pd.Timestamp("2013-01-01")
TRADING_COST_BPS = 5.0
PERIODS = 252

EARNINGS_MONTHS = frozenset({1, 4, 7, 10})
PLACEBO_ONE_MONTH_LATER = frozenset({2, 5, 8, 11})
PLACEBO_TWO_MONTHS_LATER = frozenset({3, 6, 9, 12})

MARKET_ASSETS = ["SPY", "QQQ", "RSP", "IWM"]
SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB"]
TEST_ASSETS = MARKET_ASSETS + SECTORS
PORTFOLIO_ASSETS = sorted(set(TEST_ASSETS + ["SHY", "IEF"]))

WINDOWS = [
    (start_day, length)
    for start_day in [1, 4, 7, 10, 13]
    for length in [3, 5, 8]
    if start_day + length - 1 <= 20
]
PRIMARY_START_DAY = 7
PRIMARY_LENGTH = 8

STRESS_WINDOWS = {
    "GFC": ("2007-10-09", "2009-03-09"),
    "2011": ("2011-07-22", "2011-10-03"),
    "2018_Q4": ("2018-10-01", "2018-12-24"),
    "COVID": ("2020-02-19", "2020-03-23"),
}


def _load_prices() -> pd.DataFrame:
    """Load the frozen adjusted-close panel and enforce the pre-2022 boundary."""
    prices = pd.read_parquet(PRICE_PATH).sort_index().sort_index(axis=1)
    if prices.index.max() > LAST_ALLOWED:
        raise AssertionError("post-2021 observation entered the study")
    missing = sorted(set(PORTFOLIO_ASSETS) - set(prices.columns))
    if missing:
        raise RuntimeError(f"required ETF histories are missing: {missing}")
    panel = prices[PORTFOLIO_ASSETS].dropna(how="any").astype(float)
    if panel.index.min() > pd.Timestamp("2004-01-05"):
        raise RuntimeError("unexpectedly short ETF history")
    return panel


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return known-in-advance calendar fields for every trading session."""
    frame = pd.DataFrame(index=index)
    frame["year"] = index.year
    frame["month"] = index.month
    frame["quarter"] = index.quarter
    frame["quarter_id"] = frame["year"].astype(str) + "Q" + frame["quarter"].astype(str)
    frame["trading_day_of_month"] = frame.groupby(["year", "month"]).cumcount() + 1
    return frame


def _window_mask(
    calendar: pd.DataFrame,
    months: frozenset[int],
    start_day: int,
    length: int,
) -> pd.Series:
    day = calendar["trading_day_of_month"]
    return (
        calendar["month"].isin(months)
        & day.ge(start_day)
        & day.lt(start_day + length)
    ).rename("active")


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values for a family of exploratory tests."""
    values = p_values.astype(float).clip(0.0, 1.0)
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return pd.Series(out, index=values.index)


def _event_returns(
    returns: pd.Series,
    calendar: pd.DataFrame,
    months: frozenset[int],
    start_day: int,
    length: int,
) -> pd.Series:
    mask = _window_mask(calendar, months, start_day, length)
    selected = returns.where(mask).dropna()
    ids = calendar.loc[selected.index, "quarter_id"]
    compounded = (1.0 + selected).groupby(ids).prod() - 1.0
    return compounded.rename(returns.name)


def _t_stat(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 3 or clean.std(ddof=1) == 0.0:
        return np.nan
    return float(clean.mean() / (clean.std(ddof=1) / np.sqrt(len(clean))))


def _p_value(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 3 or clean.std(ddof=1) == 0.0:
        return np.nan
    return float(stats.ttest_1samp(clean, popmean=0.0).pvalue)


def _event_sweep(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Compare earnings-month windows with same-position placebo months."""
    daily = prices.pct_change(fill_method=None)
    rows: list[dict[str, float | int | str]] = []
    for asset in TEST_ASSETS:
        for start_day, length in WINDOWS:
            earnings = _event_returns(
                daily[asset], calendar, EARNINGS_MONTHS, start_day, length
            )
            placebo_1 = _event_returns(
                daily[asset],
                calendar,
                PLACEBO_ONE_MONTH_LATER,
                start_day,
                length,
            )
            placebo_2 = _event_returns(
                daily[asset],
                calendar,
                PLACEBO_TWO_MONTHS_LATER,
                start_day,
                length,
            )
            earnings_spy = _event_returns(
                daily["SPY"], calendar, EARNINGS_MONTHS, start_day, length
            )
            placebo_1_spy = _event_returns(
                daily["SPY"],
                calendar,
                PLACEBO_ONE_MONTH_LATER,
                start_day,
                length,
            )
            placebo_2_spy = _event_returns(
                daily["SPY"],
                calendar,
                PLACEBO_TWO_MONTHS_LATER,
                start_day,
                length,
            )
            aligned = pd.concat(
                {
                    "earnings": earnings,
                    "placebo_1": placebo_1,
                    "placebo_2": placebo_2,
                    "earnings_spy": earnings_spy,
                    "placebo_1_spy": placebo_1_spy,
                    "placebo_2_spy": placebo_2_spy,
                },
                axis=1,
            ).dropna()
            aligned["placebo_mean"] = aligned[["placebo_1", "placebo_2"]].mean(axis=1)
            aligned["difference"] = aligned["earnings"] - aligned["placebo_mean"]
            aligned["earnings_excess"] = aligned["earnings"] - aligned["earnings_spy"]
            aligned["placebo_excess"] = (
                aligned[["placebo_1", "placebo_2"]].to_numpy()
                - aligned[["placebo_1_spy", "placebo_2_spy"]].to_numpy()
            ).mean(axis=1)
            aligned["excess_difference"] = (
                aligned["earnings_excess"] - aligned["placebo_excess"]
            )
            dev = aligned.loc[aligned.index.str[:4].astype(int) <= DEVELOPMENT_END.year]
            val = aligned.loc[aligned.index.str[:4].astype(int) >= VALIDATION_START.year]
            rows.append(
                {
                    "asset": asset,
                    "start_day": start_day,
                    "length": length,
                    "n_quarters": len(aligned),
                    "earnings_mean": aligned["earnings"].mean(),
                    "earnings_t": _t_stat(aligned["earnings"]),
                    "earnings_win_rate": (aligned["earnings"] > 0.0).mean(),
                    "placebo_mean": aligned["placebo_mean"].mean(),
                    "difference_mean": aligned["difference"].mean(),
                    "difference_t": _t_stat(aligned["difference"]),
                    "difference_p": _p_value(aligned["difference"]),
                    "dev_difference": dev["difference"].mean(),
                    "dev_difference_t": _t_stat(dev["difference"]),
                    "val_difference": val["difference"].mean(),
                    "val_difference_t": _t_stat(val["difference"]),
                    "earnings_excess_mean": aligned["earnings_excess"].mean(),
                    "placebo_excess_mean": aligned["placebo_excess"].mean(),
                    "excess_difference_mean": aligned["excess_difference"].mean(),
                    "excess_difference_t": _t_stat(aligned["excess_difference"]),
                    "excess_difference_p": _p_value(aligned["excess_difference"]),
                    "dev_excess_difference": dev["excess_difference"].mean(),
                    "dev_excess_difference_t": _t_stat(dev["excess_difference"]),
                    "val_excess_difference": val["excess_difference"].mean(),
                    "val_excess_difference_t": _t_stat(val["excess_difference"]),
                    "both_subperiods_positive": bool(
                        dev["difference"].mean() > 0.0 and val["difference"].mean() > 0.0
                    ),
                    "both_subperiods_excess_positive": bool(
                        dev["excess_difference"].mean() > 0.0
                        and val["excess_difference"].mean() > 0.0
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["difference_q"] = _bh_qvalues(frame["difference_p"].fillna(1.0))
    frame["excess_difference_q"] = _bh_qvalues(
        frame["excess_difference_p"].fillna(1.0)
    )
    return frame.sort_values(
        ["both_subperiods_positive", "difference_q", "difference_mean"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _empty_weights(index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=columns)


def _fixed_window_weights(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    active_asset: str,
    inactive_asset: str,
    start_day: int,
    length: int,
) -> pd.DataFrame:
    mask = _window_mask(calendar, EARNINGS_MONTHS, start_day, length)
    weights = _empty_weights(prices.index, prices.columns)
    weights.loc[mask, active_asset] = 1.0
    weights.loc[~mask, inactive_asset] = 1.0
    return weights


def _relative_momentum_weights(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    start_day: int,
    length: int,
    lookback: int,
) -> pd.DataFrame:
    """Use prior-close QQQ/SPY relative strength only inside the window."""
    mask = _window_mask(calendar, EARNINGS_MONTHS, start_day, length)
    relative = (prices["QQQ"] / prices["SPY"]).shift(1)
    positive = relative.pct_change(lookback, fill_method=None).gt(0.0)
    weights = _empty_weights(prices.index, prices.columns)
    weights["SPY"] = 1.0
    choose_qqq = mask & positive.fillna(False)
    weights.loc[choose_qqq, ["SPY", "QQQ"]] = [0.0, 1.0]
    return weights


def _relative_momentum_all_year_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame:
    """Own QQQ when prior-close relative momentum is positive, else SPY."""
    relative = (prices["QQQ"] / prices["SPY"]).shift(1)
    positive = relative.pct_change(lookback, fill_method=None).gt(0.0)
    weights = _empty_weights(prices.index, prices.columns)
    weights["SPY"] = 1.0
    weights.loc[positive.fillna(False), ["SPY", "QQQ"]] = [0.0, 1.0]
    return weights


def _sector_momentum_weights(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    start_day: int,
    length: int,
    lookback: int,
    top_n: int = 3,
) -> pd.DataFrame:
    """Pick sectors at each window entry using returns ending one close earlier."""
    mask = _window_mask(calendar, EARNINGS_MONTHS, start_day, length)
    entry = mask & ~mask.shift(1, fill_value=False)
    momentum = prices[SECTORS].shift(1).pct_change(lookback, fill_method=None)
    weights = _empty_weights(prices.index, prices.columns)
    weights["SPY"] = 1.0
    entry_dates = list(prices.index[entry])
    for entry_date in entry_dates:
        scores = momentum.loc[entry_date].dropna().nlargest(top_n)
        if len(scores) != top_n:
            continue
        active_dates = prices.index[
            mask
            & calendar["quarter_id"].eq(calendar.loc[entry_date, "quarter_id"])
        ]
        weights.loc[active_dates, :] = 0.0
        weights.loc[active_dates, scores.index] = 1.0 / top_n
    return weights


def _season_trend_gate_weights(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    start_day: int,
    length: int,
    lookback: int,
) -> pd.DataFrame:
    """Own SPY normally, but use SHY in weak-trend earnings windows."""
    mask = _window_mask(calendar, EARNINGS_MONTHS, start_day, length)
    prior = prices["SPY"].shift(1)
    trend_positive = prior.gt(prior.rolling(lookback, min_periods=lookback).mean())
    defensive = mask & ~trend_positive.fillna(False)
    weights = _empty_weights(prices.index, prices.columns)
    weights["SPY"] = 1.0
    weights.loc[defensive, ["SPY", "SHY"]] = [0.0, 1.0]
    return weights


def _trend_gate_all_year_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame:
    """Own SPY when the prior close is above its trailing mean, else SHY."""
    prior = prices["SPY"].shift(1)
    trend_positive = prior.gt(prior.rolling(lookback, min_periods=lookback).mean())
    weights = _empty_weights(prices.index, prices.columns)
    weights["SHY"] = 1.0
    weights.loc[trend_positive.fillna(False), ["SPY", "SHY"]] = [1.0, 0.0]
    return weights


def _strategy_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    cost_bps: float = TRADING_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """Apply prior-close target holdings to close-to-close returns, net of trading cost."""
    weights = weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("weights must be fully invested")
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    gross = (weights * asset_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 10_000.0
    return net.iloc[1:].rename("return"), turnover.iloc[1:].rename("turnover")


def _subperiod_metrics(returns: pd.Series, prefix: str) -> dict[str, float]:
    metrics = summary(returns)
    return {
        f"{prefix}_cagr": metrics.get("CAGR", np.nan),
        f"{prefix}_vol": metrics.get("AnnVol", np.nan),
        f"{prefix}_sharpe": metrics.get("Sharpe", np.nan),
        f"{prefix}_max_drawdown": metrics.get("MaxDD", np.nan),
    }


def _strategy_metrics(
    name: str,
    family: str,
    returns: pd.Series,
    turnover: pd.Series,
    benchmark: pd.Series,
    matched: pd.Series | None = None,
    **metadata: float | int | str,
) -> dict[str, float | int | str]:
    metrics = summary(returns)
    duration = drawdown_duration_metrics(returns, recovery_target_days=20)
    years = len(returns) / PERIODS
    active = returns.align(benchmark, join="inner")
    active_returns = active[0] - active[1]
    tracking_error = active_returns.std() * np.sqrt(PERIODS)
    row: dict[str, float | int | str] = {
        "name": name,
        "family": family,
        **metadata,
        "cagr": metrics["CAGR"],
        "annual_vol": metrics["AnnVol"],
        "sharpe": metrics["Sharpe"],
        "max_drawdown": metrics["MaxDD"],
        "annual_turnover": turnover.sum() / years,
        "active_return_ann": active_returns.mean() * PERIODS,
        "information_ratio": (
            active_returns.mean() / active_returns.std() * np.sqrt(PERIODS)
            if active_returns.std() > 0.0
            else np.nan
        ),
        "tracking_error": tracking_error,
        "max_5pct_recovery_days": duration["max_material_recovery_days"],
        "median_5pct_recovery_days": duration["median_material_recovery_days"],
        "share_5pct_recovered_within_20d": duration[
            "share_material_recovered_within_target"
        ],
        "n_5pct_drawdowns": duration["material_drawdown_count"],
    }
    row.update(
        _subperiod_metrics(returns.loc[:DEVELOPMENT_END], "dev")
    )
    row.update(
        _subperiod_metrics(returns.loc[VALIDATION_START:], "val")
    )
    dev_active = returns.loc[:DEVELOPMENT_END].align(
        benchmark.loc[:DEVELOPMENT_END], join="inner"
    )
    val_active = returns.loc[VALIDATION_START:].align(
        benchmark.loc[VALIDATION_START:], join="inner"
    )
    for prefix, aligned in [("dev", dev_active), ("val", val_active)]:
        excess = aligned[0] - aligned[1]
        row[f"{prefix}_active_return_ann"] = excess.mean() * PERIODS
        row[f"{prefix}_information_ratio"] = (
            excess.mean() / excess.std() * np.sqrt(PERIODS)
            if excess.std() > 0.0
            else np.nan
        )
    matched_returns = benchmark if matched is None else matched
    matched_aligned = returns.align(matched_returns, join="inner")
    matched_excess = matched_aligned[0] - matched_aligned[1]
    row["matched_active_return_ann"] = matched_excess.mean() * PERIODS
    row["matched_information_ratio"] = (
        matched_excess.mean() / matched_excess.std() * np.sqrt(PERIODS)
        if matched_excess.std() > 0.0
        else np.nan
    )
    for prefix, subset in [
        ("dev", matched_excess.loc[:DEVELOPMENT_END]),
        ("val", matched_excess.loc[VALIDATION_START:]),
    ]:
        row[f"{prefix}_matched_active_return_ann"] = subset.mean() * PERIODS
        row[f"{prefix}_matched_information_ratio"] = (
            subset.mean() / subset.std() * np.sqrt(PERIODS)
            if subset.std() > 0.0
            else np.nan
        )
    return row


def _strategy_sweep(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = prices.pct_change(fill_method=None).iloc[1:]
    benchmark = daily["SPY"].rename("SPY")
    rows: list[dict[str, float | int | str]] = []
    returns_map: dict[str, pd.Series] = {"SPY_buy_hold": benchmark}

    zero_turnover = pd.Series(0.0, index=benchmark.index)
    rows.append(
        _strategy_metrics(
            "SPY_buy_hold",
            "benchmark",
            benchmark,
            zero_turnover,
            benchmark,
            active_asset="SPY",
            inactive_asset="SPY",
            start_day=0,
            length=0,
            lookback=0,
        )
    )

    qqq_all_year: dict[int, pd.Series] = {}
    trend_all_year: dict[int, pd.Series] = {}
    for lookback in [21, 63, 126]:
        qqq_name = f"qqq_relative_momentum_all_year_lb{lookback}"
        qqq_weights = _relative_momentum_all_year_weights(prices, lookback=lookback)
        qqq_strategy, qqq_turnover = _strategy_returns(prices, qqq_weights)
        qqq_all_year[lookback] = qqq_strategy
        returns_map[qqq_name] = qqq_strategy
        rows.append(
            _strategy_metrics(
                qqq_name,
                "all_year_comparator",
                qqq_strategy,
                qqq_turnover,
                benchmark,
                active_asset="QQQ_if_relative_strength_positive",
                inactive_asset="SPY",
                start_day=0,
                length=0,
                lookback=lookback,
                matched_baseline="SPY_buy_hold",
            )
        )

        trend_name = f"spy_trend_gate_all_year_lb{lookback}"
        trend_weights = _trend_gate_all_year_weights(prices, lookback=lookback)
        trend_strategy, trend_turnover = _strategy_returns(prices, trend_weights)
        trend_all_year[lookback] = trend_strategy
        returns_map[trend_name] = trend_strategy
        rows.append(
            _strategy_metrics(
                trend_name,
                "all_year_comparator",
                trend_strategy,
                trend_turnover,
                benchmark,
                active_asset="SPY_if_trend_positive_else_SHY",
                inactive_asset="SHY",
                start_day=0,
                length=0,
                lookback=lookback,
                matched_baseline="SPY_buy_hold",
            )
        )

    for start_day, length in WINDOWS:
        for asset in TEST_ASSETS:
            name = f"season_only_{asset}_s{start_day}_l{length}"
            weights = _fixed_window_weights(
                prices,
                calendar,
                active_asset=asset,
                inactive_asset="SHY",
                start_day=start_day,
                length=length,
            )
            strategy, turnover = _strategy_returns(prices, weights)
            returns_map[name] = strategy
            rows.append(
                _strategy_metrics(
                    name,
                    "season_only",
                    strategy,
                    turnover,
                    benchmark,
                    active_asset=asset,
                    inactive_asset="SHY",
                    start_day=start_day,
                    length=length,
                    lookback=0,
                )
            )

            if asset != "SPY":
                name = f"season_tilt_{asset}_s{start_day}_l{length}"
                weights = _fixed_window_weights(
                    prices,
                    calendar,
                    active_asset=asset,
                    inactive_asset="SPY",
                    start_day=start_day,
                    length=length,
                )
                strategy, turnover = _strategy_returns(prices, weights)
                returns_map[name] = strategy
                rows.append(
                    _strategy_metrics(
                        name,
                        "season_tilt",
                        strategy,
                        turnover,
                        benchmark,
                        active_asset=asset,
                        inactive_asset="SPY",
                        start_day=start_day,
                        length=length,
                        lookback=0,
                    )
                )

        for lookback in [21, 63, 126]:
            name = f"qqq_relative_momentum_s{start_day}_l{length}_lb{lookback}"
            weights = _relative_momentum_weights(
                prices,
                calendar,
                start_day=start_day,
                length=length,
                lookback=lookback,
            )
            strategy, turnover = _strategy_returns(prices, weights)
            returns_map[name] = strategy
            rows.append(
                _strategy_metrics(
                    name,
                    "qqq_relative_momentum",
                    strategy,
                    turnover,
                    benchmark,
                    active_asset="QQQ_if_relative_strength_positive",
                    inactive_asset="SPY",
                    start_day=start_day,
                    length=length,
                    lookback=lookback,
                    matched=qqq_all_year[lookback],
                    matched_baseline=f"qqq_relative_momentum_all_year_lb{lookback}",
                )
            )

            name = f"sector_momentum_s{start_day}_l{length}_lb{lookback}"
            weights = _sector_momentum_weights(
                prices,
                calendar,
                start_day=start_day,
                length=length,
                lookback=lookback,
            )
            strategy, turnover = _strategy_returns(prices, weights)
            returns_map[name] = strategy
            rows.append(
                _strategy_metrics(
                    name,
                    "sector_momentum",
                    strategy,
                    turnover,
                    benchmark,
                    active_asset="top_3_sectors",
                    inactive_asset="SPY",
                    start_day=start_day,
                    length=length,
                    lookback=lookback,
                    matched=trend_all_year[lookback],
                    matched_baseline=f"spy_trend_gate_all_year_lb{lookback}",
                )
            )

            name = f"season_trend_gate_s{start_day}_l{length}_lb{lookback}"
            weights = _season_trend_gate_weights(
                prices,
                calendar,
                start_day=start_day,
                length=length,
                lookback=lookback,
            )
            strategy, turnover = _strategy_returns(prices, weights)
            returns_map[name] = strategy
            rows.append(
                _strategy_metrics(
                    name,
                    "season_trend_gate",
                    strategy,
                    turnover,
                    benchmark,
                    active_asset="SPY_if_trend_positive_else_SHY",
                    inactive_asset="SPY",
                    start_day=start_day,
                    length=length,
                    lookback=lookback,
                )
            )

    metrics = pd.DataFrame(rows)
    metrics["both_subperiod_cagr_positive"] = (
        metrics["dev_cagr"].gt(0.0) & metrics["val_cagr"].gt(0.0)
    )
    metrics["both_subperiod_active_positive"] = (
        metrics["dev_active_return_ann"].gt(0.0)
        & metrics["val_active_return_ann"].gt(0.0)
    )
    metrics["both_subperiod_matched_positive"] = (
        metrics["dev_matched_active_return_ann"].gt(0.0)
        & metrics["val_matched_active_return_ann"].gt(0.0)
    )
    metrics["robust_active_score"] = metrics[
        ["dev_information_ratio", "val_information_ratio"]
    ].min(axis=1)
    metrics["robust_matched_score"] = metrics[
        ["dev_matched_information_ratio", "val_matched_information_ratio"]
    ].min(axis=1)
    return metrics, pd.DataFrame(returns_map)


def _stress_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, (start, end) in STRESS_WINDOWS.items():
        window = returns.loc[start:end]
        for strategy in returns.columns:
            series = window[strategy].dropna()
            rows.append(
                {
                    "stress": name,
                    "strategy": strategy,
                    "total_return": (1.0 + series).prod() - 1.0,
                    "max_drawdown": summary(series).get("MaxDD", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _daily_position_stats(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Average return by trading-day position in quarter-start months."""
    returns = prices[MARKET_ASSETS].pct_change(fill_method=None)
    active = calendar["month"].isin(EARNINGS_MONTHS)
    frame = returns.loc[active].copy()
    frame["trading_day_of_month"] = calendar.loc[active, "trading_day_of_month"]
    means = frame.groupby("trading_day_of_month")[MARKET_ASSETS].mean() * 10_000.0
    counts = frame.groupby("trading_day_of_month").size().rename("n")
    return means.join(counts).reset_index()


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _position_chart(position: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for asset in ["SPY", "QQQ", "RSP", "IWM"]:
        ax.plot(
            position["trading_day_of_month"],
            position[asset],
            marker="o",
            markersize=3,
            label=asset,
        )
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.axvspan(
        PRIMARY_START_DAY,
        PRIMARY_START_DAY + PRIMARY_LENGTH - 1,
        color="#f59e0b",
        alpha=0.12,
        label="pre-specified proxy window",
    )
    ax.set_title("Average daily return by trading-day position (Jan/Apr/Jul/Oct)")
    ax.set_xlabel("Trading day of month")
    ax.set_ylabel("Average return (bp)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.2)
    return _figure_to_base64(fig)


def _equity_chart(returns: pd.DataFrame, names: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    wealth = (1.0 + returns[names].fillna(0.0)).cumprod()
    for name in names:
        ax.plot(wealth.index, wealth[name], label=name)
    ax.set_yscale("log")
    ax.set_title("Selected net wealth curves (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return _figure_to_base64(fig)


def _format_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if column in {
            "cagr",
            "annual_vol",
            "max_drawdown",
            "active_return_ann",
            "dev_cagr",
            "val_cagr",
            "dev_active_return_ann",
            "val_active_return_ann",
            "matched_active_return_ann",
            "dev_matched_active_return_ann",
            "val_matched_active_return_ann",
            "earnings_mean",
            "placebo_mean",
            "difference_mean",
            "dev_difference",
            "val_difference",
            "earnings_excess_mean",
            "placebo_excess_mean",
            "excess_difference_mean",
            "dev_excess_difference",
            "val_excess_difference",
        }:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
        elif column in {
            "sharpe",
            "information_ratio",
            "dev_sharpe",
            "val_sharpe",
            "dev_information_ratio",
            "val_information_ratio",
            "matched_information_ratio",
            "dev_matched_information_ratio",
            "val_matched_information_ratio",
            "difference_t",
            "dev_difference_t",
            "val_difference_t",
            "difference_q",
            "excess_difference_t",
            "excess_difference_q",
        }:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
    return display.to_html(index=False, escape=True, border=0, classes="data")


def _render_report(
    event_stats: pd.DataFrame,
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    stress: pd.DataFrame,
    position: pd.DataFrame,
) -> None:
    primary_events = event_stats[
        event_stats["start_day"].eq(PRIMARY_START_DAY)
        & event_stats["length"].eq(PRIMARY_LENGTH)
        & event_stats["asset"].isin(MARKET_ASSETS)
    ].sort_values("asset")

    primary_names = [
        f"season_only_SPY_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}",
        f"season_only_QQQ_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}",
        f"season_tilt_QQQ_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}",
        f"season_trend_gate_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}_lb126",
    ]
    primary_metrics = metrics[metrics["name"].isin(["SPY_buy_hold"] + primary_names)]
    robust_tilts = metrics[
        metrics["family"].isin(
            ["season_tilt", "qqq_relative_momentum", "sector_momentum", "season_trend_gate"]
        )
        & metrics["both_subperiod_active_positive"]
        & metrics["both_subperiod_matched_positive"]
    ].sort_values(["robust_active_score", "active_return_ann"], ascending=False)
    season_only = metrics[
        metrics["family"].eq("season_only")
        & metrics["both_subperiod_cagr_positive"]
    ].sort_values(["val_sharpe", "dev_sharpe"], ascending=False)

    event_cols = [
        "asset",
        "start_day",
        "length",
        "n_quarters",
        "earnings_mean",
        "placebo_mean",
        "difference_mean",
        "difference_t",
        "difference_q",
        "dev_difference",
        "val_difference",
        "earnings_excess_mean",
        "placebo_excess_mean",
        "excess_difference_mean",
        "excess_difference_t",
        "excess_difference_q",
        "dev_excess_difference",
        "val_excess_difference",
    ]
    metric_cols = [
        "name",
        "family",
        "cagr",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "max_5pct_recovery_days",
        "share_5pct_recovered_within_20d",
        "active_return_ann",
        "information_ratio",
        "dev_cagr",
        "val_cagr",
        "dev_active_return_ann",
        "val_active_return_ann",
        "matched_active_return_ann",
        "matched_information_ratio",
        "dev_matched_active_return_ann",
        "val_matched_active_return_ann",
    ]

    selected_for_chart = ["SPY_buy_hold"] + [
        name for name in primary_names if name in returns.columns
    ]
    pos_chart = _position_chart(position)
    eq_chart = _equity_chart(returns, selected_for_chart)

    primary_spy = primary_events.loc[primary_events["asset"].eq("SPY")].iloc[0]
    primary_qqq = primary_events.loc[primary_events["asset"].eq("QQQ")].iloc[0]
    significant = event_stats[event_stats["difference_q"].lt(0.10)]
    robust_significant = significant[significant["both_subperiods_positive"]]
    significant_excess = event_stats[event_stats["excess_difference_q"].lt(0.10)]
    robust_significant_excess = significant_excess[
        significant_excess["both_subperiods_excess_positive"]
    ]
    accepted_tilts = robust_tilts[
        robust_tilts["dev_active_return_ann"].gt(0.0)
        & robust_tilts["val_active_return_ann"].gt(0.0)
        & robust_tilts["dev_matched_active_return_ann"].gt(0.0)
        & robust_tilts["val_matched_active_return_ann"].gt(0.0)
        & robust_tilts["max_drawdown"].ge(metrics.loc[metrics["name"].eq("SPY_buy_hold"), "max_drawdown"].iloc[0])
    ]

    verdict = (
        "固定财报季日历代理不足以形成可接受的独立 ETF alpha；值得继续的是使用真实、事前可知的高关注公司公告集群日期。"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETF 财报季策略研究（仅截至2021）</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#f4f5f7;color:#172033;line-height:1.58}}
.wrap{{max-width:1180px;margin:auto;padding:34px 22px 70px}} h1{{font-size:34px;margin-bottom:8px}} h2{{margin-top:38px;border-bottom:1px solid #d9dee8;padding-bottom:8px}}
.lead{{font-size:18px;color:#475569}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:22px 0}}
.card{{background:white;border:1px solid #e3e7ee;border-radius:12px;padding:16px;box-shadow:0 2px 8px #1720330a}} .card strong{{display:block;font-size:25px;color:#0f766e}}
.warn{{background:#fff7ed;border-left:5px solid #f97316;padding:15px 18px;border-radius:8px}} .ok{{background:#ecfdf5;border-left:5px solid #10b981;padding:15px 18px;border-radius:8px}}
table.data{{border-collapse:collapse;width:100%;font-size:12px;background:white}} table.data th,table.data td{{padding:7px 8px;border:1px solid #e5e7eb;text-align:right;white-space:nowrap}} table.data th:first-child,table.data td:first-child{{text-align:left}} table.data th{{background:#eef2f7;position:sticky;top:0}}
.scroll{{overflow-x:auto;border-radius:9px;border:1px solid #e5e7eb}} img{{width:100%;height:auto;background:white;border-radius:9px;border:1px solid #e5e7eb}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} .small{{font-size:13px;color:#64748b}}
</style></head><body><div class="wrap">
<h1>ETF 财报季策略研究</h1><p class="lead">固定日历代理、行业确认与风险开关；ETF-only；数据截至 2021-12-31；不使用 2022 年以后信息。</p>
<div class="warn"><b>结论：</b>{escape(verdict)} 固定窗口只能说明季节性，不能证明因果来自财报。</div>
<div class="grid">
<div class="card"><strong>{len(event_stats):,}</strong>事件窗口检验</div>
<div class="card"><strong>{len(metrics)-1:,}</strong>ETF 策略变体</div>
<div class="card"><strong>{len(significant):,}</strong>全扫描 FDR q&lt;10%</div>
<div class="card"><strong>{len(robust_significant):,}</strong>且两半样本同向</div>
<div class="card"><strong>{len(significant_excess):,}</strong>相对SPY FDR q&lt;10%</div>
<div class="card"><strong>{len(robust_significant_excess):,}</strong>相对SPY且两半同向</div>
<div class="card"><strong>{primary_spy['difference_mean']:.2%}</strong>SPY 主窗口 vs 安慰剂</div>
<div class="card"><strong>{primary_qqq['difference_mean']:.2%}</strong>QQQ 主窗口 vs 安慰剂</div>
</div>

<h2>1. 研究问题与机制</h2>
<p>文献支持三个不同但常被混为一谈的现象：个股在公告日前后存在注意力相关的公告溢价；若多家高关注公司在收盘后集中公告，市场在公告前 24 小时可能上行；但公司层面的 PEAD 不一定能聚合成指数层面的可预测收益。本研究因此先测试完全事前可知的日历代理，再判断是否值得接入真实公告日历。</p>
<ul><li><b>主窗口（事前规定）：</b>每年 1/4/7/10 月第 {PRIMARY_START_DAY} 至第 {PRIMARY_START_DAY+PRIMARY_LENGTH-1} 个交易日。</li>
<li><b>安慰剂：</b>同一季度后续两个月的相同交易日位置。</li><li><b>开发/验证：</b>2004–2012 与 2013–2021。</li></ul>

<h2>2. 主窗口事件结果</h2><div class="scroll">{_format_table(primary_events[event_cols], max_rows=20)}</div>
<p class="small">difference 为财报月份窗口收益减去同季度另外两个月相同位置的平均收益；excess_difference 先减 SPY，再做同样的月份安慰剂比较。q 值对全部 {len(event_stats)} 个探索检验做 Benjamini–Hochberg 校正。</p>

<h2>3. 季内收益形状</h2><img src="data:image/png;base64,{pos_chart}" alt="trading day profile">
<p class="small">按交易日位置画均值非常容易受少数季度影响；它用于诊断，不用于选择最优日期。</p>

<h2>4. 预设策略表现</h2><div class="scroll">{_format_table(primary_metrics[metric_cols], max_rows=20)}</div>
<img src="data:image/png;base64,{eq_chart}" alt="selected wealth curves">
<p>成本按每单位单边换手 5bp；从一个 ETF 全仓切换到另一个 ETF 计 10bp。日历在前一日已知；价格/趋势信号均至少滞后一日。SHY 作为现金替代，避免把现金收益免费设为零。</p>

<h2>5. 全扫描中两半样本均为正的覆盖策略</h2>
<div class="scroll">{_format_table(robust_tilts[metric_cols], max_rows=25)}</div>
<p class="small">这张表要求开发期与验证期对 SPY、以及对同信号的全年版本都为正；仍然只是候选生成，不是确认性证据。全扫描后的最优参数存在多重检验偏差，必须另留真实公告日历或未来数据作为最终 holdout。</p>

<h2>6. 纯财报季持仓候选</h2><div class="scroll">{_format_table(season_only[metric_cols], max_rows=25)}</div>

<h2>7. 压力期</h2><div class="scroll">{_format_table(stress[stress['strategy'].isin(selected_for_chart)], max_rows=50)}</div>

<h2>8. 判定</h2>
<div class="ok"><b>可以继续：</b>真实的“高关注公司、收盘后公告集群”策略。每天收盘前，依据当时已确认的公告时间表，若当晚高关注权重超过阈值，则持有 SPY（或 QQQ）；次日收盘退出。高关注权重必须用上一年度媒体量或事前市值固定，不能回填。</div>
<p><b>暂不接受：</b>仅凭 1/4/7/10 月固定宽窗口提高或降低 ETF 仓位；以及扫描后挑出的最优行业/日期组合。它们最多是弱季节性或已有动量/趋势因子的条件化。</p>
<p><b>数据升级：</b>需要历史公告日期、BMO/AMC 会话、当时可知的确认状态、历史成分与上一年度关注度。若只有事后修订日期，回测会发生严重前视偏差。</p>

<h2>9. 独立性与失败模式</h2><ul>
<li>固定季度月份与宏观数据、期权到期、税务和月度资金流重合，不能识别财报因果。</li>
<li>72 个季度样本很少；细分到季度、行业或 3 日窗口后统计功效更弱。</li>
<li>行业 ETF 已加快同业信息传递，可能削弱行业层面的公告后漂移。</li>
<li>真正的集群效应集中在少数日期，宽窗口会稀释；反过来扫描日期会制造过拟合。</li>
<li>SPY/QQQ 收盘前入场与隔夜跳空存在执行价差；当前只有日收盘数据，无法分解隔夜与日内。</li>
</ul>

<h2>10. 可复现产物</h2><ul>
<li><code>event_window_stats.csv</code>：{len(event_stats)} 个 ETF×窗口检验。</li>
<li><code>strategy_metrics.csv</code>：{len(metrics)} 条策略/基准指标。</li>
<li><code>strategy_returns.parquet</code>：全部净收益序列。</li>
<li><code>stress_windows.csv</code>、<code>daily_position_stats.csv</code> 与 <code>meta.json</code>。</li>
</ul>
<p class="small">生成脚本：scripts/etf_earnings_season_study.py。该报告不是投资建议。</p>
</div></body></html>"""
    REPORT.write_text(html, encoding="utf-8")

    meta = {
        "last_allowed": str(LAST_ALLOWED.date()),
        "development_end": str(DEVELOPMENT_END.date()),
        "validation_start": str(VALIDATION_START.date()),
        "trading_cost_bps_per_unit": TRADING_COST_BPS,
        "primary_window": {
            "months": sorted(EARNINGS_MONTHS),
            "start_trading_day": PRIMARY_START_DAY,
            "length": PRIMARY_LENGTH,
        },
        "event_tests": len(event_stats),
        "strategy_rows": len(metrics),
        "fdr_q_lt_10pct": len(significant),
        "fdr_q_lt_10pct_both_subperiods_positive": len(robust_significant),
        "relative_to_spy_fdr_q_lt_10pct": len(significant_excess),
        "relative_to_spy_fdr_q_lt_10pct_both_subperiods_positive": len(
            robust_significant_excess
        ),
        "robust_tilts_that_also_beat_spy_drawdown": len(accepted_tilts),
        "verdict": verdict,
    }
    (OUT / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    prices = _load_prices()
    calendar = _calendar(prices.index)
    OUT.mkdir(parents=True, exist_ok=True)

    event_stats = _event_sweep(prices, calendar)
    metrics, returns = _strategy_sweep(prices, calendar)
    position = _daily_position_stats(prices, calendar)

    primary_names = [
        "SPY_buy_hold",
        f"season_only_SPY_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}",
        f"season_only_QQQ_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}",
        f"season_tilt_QQQ_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}",
        f"season_trend_gate_s{PRIMARY_START_DAY}_l{PRIMARY_LENGTH}_lb126",
    ]
    robust_names = (
        metrics[
            metrics["both_subperiod_active_positive"]
            & metrics["family"].ne("benchmark")
        ]
        .sort_values(["robust_active_score", "active_return_ann"], ascending=False)
        .head(10)["name"]
        .tolist()
    )
    stress_names = list(dict.fromkeys(primary_names + robust_names))
    stress = _stress_table(returns[stress_names])

    event_stats.to_csv(OUT / "event_window_stats.csv", index=False)
    metrics.to_csv(OUT / "strategy_metrics.csv", index=False)
    returns.to_parquet(OUT / "strategy_returns.parquet")
    stress.to_csv(OUT / "stress_windows.csv", index=False)
    position.to_csv(OUT / "daily_position_stats.csv", index=False)
    _render_report(event_stats, metrics, returns, stress, position)

    print(
        json.dumps(
            {
                "report": str(REPORT),
                "event_tests": len(event_stats),
                "strategies": len(metrics) - 1,
                "last_observation": str(prices.index.max().date()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
