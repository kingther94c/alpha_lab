"""Pre-2022 ETF allocation and drawdown-control strategy sweep.

The study is deliberately broad but mechanically simple. It compares long-only ETF
allocations, weekly/monthly risk controls, tactical allocation, sector rules, packaged
strategy ETFs, and synthetic SPY option overlays. Every market-data request stops at
2022-01-01 exclusive. See the frozen protocol in ``docs/research_decisions``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from alpha_lab.analytics.returns import drawdown, drawdown_duration_metrics
from alpha_lab.backtest.collar import (
    SyntheticOptionOverlayConfig,
    run_synthetic_option_overlay,
)
from alpha_lab.backtest.vector import run_drift_backtest
from alpha_lab.stats.tests import deflated_sharpe_ratio
from alpha_lab.utils.paths import PROJECT_ROOT

DOWNLOAD_START = "2004-01-01"
DOWNLOAD_END = "2022-01-01"
LAST_ALLOWED = pd.Timestamp("2021-12-31")
CORE_START = pd.Timestamp("2007-01-03")
CORE_END = LAST_ALLOWED
DECISION_START = pd.Timestamp("2006-12-01")
PRIMARY_TRADING_BPS = 5.0
STRESS_TRADING_BPS = 10.0
PERIODS = 252
TARGET_RETURN = 0.10

OUT = PROJECT_ROOT / "data" / "results" / "etf_strategy_50plus_pre2022"
REPORT = PROJECT_ROOT / "reports" / "etf_strategy_50plus_pre2022.html"
PROTOCOL = (
    PROJECT_ROOT
    / "docs"
    / "research_decisions"
    / "2026-07-18_etf-strategy-50plus-protocol.md"
)

CORE_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "VEU",
    "VNQ",
    "IEF",
    "TLT",
    "SHY",
    "TIP",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "AGG",
]
SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB"]
DEFENSIVE_SECTORS = ["XLP", "XLV", "XLU"]
CYCLICAL_SECTORS = ["XLK", "XLF", "XLI", "XLY", "XLE", "XLB"]
PRODUCT_TICKERS = [
    "AOA",
    "AOR",
    "SPLV",
    "USMV",
    "MTUM",
    "QUAL",
    "PHDG",
    "TAIL",
    "SWAN",
    "NTSX",
    "RPAR",
    "DBMF",
    "KMLM",
    "WTMF",
    "UJAN",
    "QVAL",
    "QMOM",
    "VMOT",
    "OMFL",
    "QAI",
    "VIG",
    "VYM",
    "RSP",
]
INDEX_TICKERS = ["^VIX", "^IRX", "^PUT", "^BXM"]
ALL_TICKERS = sorted(set(CORE_TICKERS + SECTORS + PRODUCT_TICKERS + INDEX_TICKERS))

ALL_WEATHER_WEIGHTS = {
    "SPY": 0.30,
    "TLT": 0.40,
    "IEF": 0.15,
    "GLD": 0.075,
    "DBC": 0.075,
}
EQUAL_WEATHER_ASSETS = ["SPY", "TLT", "IEF", "GLD", "DBC"]
TACTICAL_ASSETS = ["SPY", "EFA", "EEM", "VNQ", "IEF", "TLT", "GLD", "DBC"]

STRESS_WINDOWS = {
    "GFC": ("2007-10-09", "2009-03-09"),
    "2011_euro_US_downgrade": ("2011-07-22", "2011-10-03"),
    "2018_Q4": ("2018-10-01", "2018-12-24"),
    "COVID_crash": ("2020-02-19", "2020-03-23"),
}


@dataclass
class StrategyRun:
    """One strategy return stream plus its research metadata."""

    name: str
    family: str
    independence_group: str
    evidence: str
    description: str
    returns: pd.Series
    stress_returns: pd.Series
    annual_turnover: float
    latest_weights: dict[str, float]
    implementation_note: str = "ETF-only"


def _download_prices() -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted closes and raw SPY, hard-stopped before 2022."""
    adjusted_cache = OUT / "market_prices_adjusted_pre2022.parquet"
    raw_spy_cache = OUT / "raw_spy_pre2022.parquet"
    if adjusted_cache.exists() and raw_spy_cache.exists():
        adjusted = pd.read_parquet(adjusted_cache).sort_index().sort_index(axis=1)
        raw_spy = pd.read_parquet(raw_spy_cache).iloc[:, 0].rename("raw_SPY").sort_index()
        if adjusted.index.max() > LAST_ALLOWED or raw_spy.index.max() > LAST_ALLOWED:
            raise AssertionError("cached market data contains post-2021 observations")
        return adjusted, raw_spy

    yf.cache.set_cache_location(tempfile.gettempdir())
    raw = yf.download(
        ALL_TICKERS,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("market-data download returned no observations")
    adjusted = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    adjusted = adjusted.sort_index().sort_index(axis=1).dropna(how="all")

    raw_spy_frame = yf.download(
        "SPY",
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    raw_spy = raw_spy_frame["Close"]
    if isinstance(raw_spy, pd.DataFrame):
        raw_spy = raw_spy.iloc[:, 0]
    raw_spy = raw_spy.rename("raw_SPY").sort_index()

    if adjusted.index.max() > LAST_ALLOWED or raw_spy.index.max() > LAST_ALLOWED:
        raise AssertionError("post-2021 observation entered the study")
    required = set(CORE_TICKERS + SECTORS + ["^VIX", "^IRX"])
    missing = sorted(ticker for ticker in required if ticker not in adjusted or adjusted[ticker].dropna().empty)
    if missing:
        raise RuntimeError(f"required price histories are missing: {missing}")
    return adjusted, raw_spy


def _period_ends(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return actual last bars for weekly, monthly, quarterly, or annual buckets."""
    aliases = {"W": "W-FRI", "M": "M", "Q": "Q", "Y": "Y"}
    period = index.to_period(aliases[freq])
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(period).last().to_numpy())


def _decision_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    dates = _period_ends(index, freq)
    return dates[(dates >= DECISION_START) & (dates < CORE_END)]


def _fixed_targets(
    panel: pd.DataFrame,
    weights: dict[str, float],
    *,
    freq: str = "Q",
) -> pd.DataFrame:
    dates = _decision_dates(panel.index, freq)
    row = pd.Series(weights, dtype=float).reindex(panel.columns, fill_value=0.0)
    if (row < 0).any() or not np.isclose(row.sum(), 1.0):
        raise ValueError("fixed weights must be non-negative and sum to one")
    return pd.DataFrame(np.tile(row.to_numpy(), (len(dates), 1)), index=dates, columns=panel.columns)


def _targets_from_rows(rows: list[pd.Series], columns: pd.Index) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns, dtype=float)
    frame = pd.DataFrame(rows).reindex(columns=columns, fill_value=0.0).fillna(0.0)
    if (frame < -1e-12).any().any() or not np.allclose(frame.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("dynamic targets must be long-only and fully invested")
    frame[frame.abs() < 1e-12] = 0.0
    return frame


def _safe_row(columns: pd.Index, date: pd.Timestamp, safe: str = "SHY") -> pd.Series:
    row = pd.Series(0.0, index=columns, name=date)
    row[safe] = 1.0
    return row


def _allocate_capped(raw: pd.Series, budget: float, cap: float) -> pd.Series:
    """Allocate a sleeve proportionally with a hard name cap; leave residual unallocated."""
    positive = raw.replace([np.inf, -np.inf], np.nan).dropna()
    positive = positive[positive > 0.0]
    out = pd.Series(0.0, index=raw.index)
    if positive.empty or budget <= 0.0:
        return out
    remaining = float(budget)
    active = positive.copy()
    while remaining > 1e-12 and not active.empty:
        proposal = active / active.sum() * remaining
        room = cap - out.reindex(active.index)
        filled = np.minimum(proposal, room.clip(lower=0.0))
        out.loc[active.index] += filled
        remaining = float(budget - out.sum())
        active = active[out.reindex(active.index) < cap - 1e-12]
        if float(filled.sum()) <= 1e-12:
            break
    return out


def _run_weight_strategy(
    adjusted: pd.DataFrame,
    *,
    name: str,
    family: str,
    independence_group: str,
    description: str,
    assets: list[str],
    target_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> StrategyRun:
    panel = adjusted[assets].dropna(how="any").astype(float)
    if panel.index.max() > LAST_ALLOWED:
        raise AssertionError(f"{name} received post-2021 prices")
    targets = target_builder(panel)
    if targets.empty:
        raise ValueError(f"{name} produced no target weights")
    primary = run_drift_backtest(
        targets,
        panel,
        trading_bps=PRIMARY_TRADING_BPS,
        execution_delay_bars=1,
        rebalance_threshold=0.0,
    )
    stress = run_drift_backtest(
        targets,
        panel,
        trading_bps=STRESS_TRADING_BPS,
        execution_delay_bars=1,
        rebalance_threshold=0.0,
    )
    start = max(CORE_START, pd.Timestamp(primary.decision_to_trade.min()))
    end = min(CORE_END, panel.index.max())
    sample = primary.returns.loc[start:end]
    stress_sample = stress.returns.loc[start:end]
    years = max(len(sample) / PERIODS, 1e-12)
    annual_turnover = float(primary.traded_notional.loc[start:end].sum() / years)
    latest = primary.weights.loc[:end].iloc[-1]
    latest_weights = {key: float(value) for key, value in latest.items() if value > 1e-4}
    return StrategyRun(
        name=name,
        family=family,
        independence_group=independence_group,
        evidence="long_rule",
        description=description,
        returns=sample.rename(name),
        stress_returns=stress_sample.rename(name),
        annual_turnover=annual_turnover,
        latest_weights=latest_weights,
    )


def _trend_targets(panel: pd.DataFrame, risky: str, lookback: int) -> pd.DataFrame:
    moving_average = panel[risky].rolling(lookback, min_periods=lookback).mean()
    rows = []
    for date in _decision_dates(panel.index, "W"):
        row = _safe_row(panel.columns, date)
        if pd.notna(moving_average.loc[date]) and panel.loc[date, risky] > moving_average.loc[date]:
            row[:] = 0.0
            row[risky] = 1.0
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _vol_targets(
    panel: pd.DataFrame,
    risky: str,
    target_vol: float,
    *,
    trend_lookback: int | None = None,
) -> pd.DataFrame:
    returns = panel[risky].pct_change()
    vol = returns.rolling(63, min_periods=63).std() * np.sqrt(PERIODS)
    moving_average = (
        panel[risky].rolling(trend_lookback, min_periods=trend_lookback).mean()
        if trend_lookback is not None
        else None
    )
    rows = []
    for date in _decision_dates(panel.index, "W"):
        exposure = 0.0 if pd.isna(vol.loc[date]) else float(np.clip(target_vol / vol.loc[date], 0.0, 1.0))
        if moving_average is not None and (
            pd.isna(moving_average.loc[date]) or panel.loc[date, risky] <= moving_average.loc[date]
        ):
            exposure = 0.0
        row = _safe_row(panel.columns, date)
        row["SHY"] = 1.0 - exposure
        row[risky] = exposure
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _drawdown_targets(panel: pd.DataFrame, risky: str, policy: str) -> pd.DataFrame:
    dd = panel[risky] / panel[risky].cummax() - 1.0
    policies = {
        "soft": [(-0.20, 0.40), (-0.10, 0.60), (-0.05, 0.80)],
        "medium": [(-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75)],
        "hard": [(-0.12, 0.00), (-0.07, 0.50), (-0.03, 0.75)],
    }
    rows = []
    for date in _decision_dates(panel.index, "W"):
        exposure = 1.0
        for threshold, scaled in policies[policy]:
            if dd.loc[date] <= threshold:
                exposure = scaled
                break
        row = _safe_row(panel.columns, date)
        row["SHY"] = 1.0 - exposure
        row[risky] = exposure
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _hysteresis_targets(panel: pd.DataFrame, risky: str) -> pd.DataFrame:
    ma = panel[risky].rolling(200, min_periods=200).mean()
    vix = panel["^VIX"]
    active = False
    consecutive_good = 0
    rows = []
    for date in _decision_dates(panel.index, "W"):
        good = pd.notna(ma.loc[date]) and panel.loc[date, risky] > ma.loc[date] and vix.loc[date] < 30.0
        if active and not good:
            active = False
            consecutive_good = 0
        elif not active:
            consecutive_good = consecutive_good + 1 if good else 0
            if consecutive_good >= 4:
                active = True
        row = _safe_row(panel.columns, date)
        if active:
            row[:] = 0.0
            row[risky] = 1.0
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _inverse_vol_targets(
    panel: pd.DataFrame,
    risky_assets: list[str],
    lookback: int,
    *,
    cap: float | None = None,
) -> pd.DataFrame:
    returns = panel[risky_assets].pct_change()
    rows = []
    for date in _decision_dates(panel.index, "M"):
        window = returns.loc[:date].tail(lookback).dropna(how="any")
        row = _safe_row(panel.columns, date)
        if len(window) >= lookback:
            inv = 1.0 / window.std().replace(0.0, np.nan)
            inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
            if not inv.empty:
                row[:] = 0.0
                if cap is None:
                    row.loc[inv.index] = inv / inv.sum()
                else:
                    alloc = _allocate_capped(inv.reindex(risky_assets), 1.0, cap)
                    row.loc[risky_assets] = alloc
                    row["SHY"] += 1.0 - float(alloc.sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _weather_trend_targets(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    risky = EQUAL_WEATHER_ASSETS
    base = pd.Series(ALL_WEATHER_WEIGHTS)
    moving_average = panel[risky].rolling(lookback, min_periods=lookback).mean()
    rows = []
    for date in _decision_dates(panel.index, "M"):
        eligible = panel.loc[date, risky] > moving_average.loc[date, risky]
        row = _safe_row(panel.columns, date)
        row.loc[risky] = base.where(eligible, 0.0)
        row["SHY"] = 1.0 - float(row.loc[risky].sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _weather_vol_target(panel: pd.DataFrame, target_vol: float) -> pd.DataFrame:
    base = pd.Series(ALL_WEATHER_WEIGHTS)
    daily = panel[base.index].pct_change().mul(base, axis=1).sum(axis=1)
    vol = daily.rolling(63, min_periods=63).std() * np.sqrt(PERIODS)
    rows = []
    for date in _decision_dates(panel.index, "M"):
        scale = 0.0 if pd.isna(vol.loc[date]) else float(np.clip(target_vol / vol.loc[date], 0.0, 1.0))
        row = _safe_row(panel.columns, date)
        row.loc[base.index] = base * scale
        row["SHY"] = 1.0 - float(row.loc[base.index].sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _gtna_targets(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    risky = TACTICAL_ASSETS
    moving_average = panel[risky].rolling(lookback, min_periods=lookback).mean()
    rows = []
    for date in _decision_dates(panel.index, "M"):
        eligible = (panel.loc[date, risky] > moving_average.loc[date, risky]).fillna(False)
        row = _safe_row(panel.columns, date)
        row.loc[risky] = eligible.astype(float) / len(risky)
        row["SHY"] = 1.0 - float(row.loc[risky].sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _momentum_targets(
    panel: pd.DataFrame,
    *,
    top_n: int,
    lookback: int | None = None,
    horizons: tuple[int, ...] | None = None,
    universe: list[str] = TACTICAL_ASSETS,
    inverse_vol: bool = False,
) -> pd.DataFrame:
    returns = panel[universe].pct_change()
    safe_momentum = panel["SHY"] / panel["SHY"].shift(252) - 1.0
    rows = []
    for date in _decision_dates(panel.index, "M"):
        if horizons is None:
            score = panel.loc[date, universe] / panel[universe].shift(lookback).loc[date] - 1.0
        else:
            components = []
            for horizon in horizons:
                raw = panel.loc[date, universe] / panel[universe].shift(horizon).loc[date] - 1.0
                components.append(raw.rank(pct=True))
            score = pd.concat(components, axis=1).mean(axis=1)
        if pd.isna(safe_momentum.loc[date]):
            rows.append(_safe_row(panel.columns, date))
            continue
        absolute = panel.loc[date, universe] / panel[universe].shift(252).loc[date] - 1.0
        eligible = score[(absolute > safe_momentum.loc[date]) & score.notna()].nlargest(top_n)
        row = _safe_row(panel.columns, date)
        if not eligible.empty:
            row[:] = 0.0
            if inverse_vol:
                vol = returns.loc[:date].tail(63).std().reindex(eligible.index)
                inv = 1.0 / vol.replace(0.0, np.nan)
                inv = inv.dropna()
                row.loc[inv.index] = inv / inv.sum() if not inv.empty else 0.0
            else:
                row.loc[eligible.index] = 1.0 / len(eligible)
            row["SHY"] += 1.0 - float(row.sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _gem_targets(panel: pd.DataFrame) -> pd.DataFrame:
    momentum = panel[["SPY", "VEU", "SHY"]] / panel[["SPY", "VEU", "SHY"]].shift(252) - 1.0
    rows = []
    for date in _decision_dates(panel.index, "M"):
        signal = momentum.loc[date]
        if not signal.notna().all():
            continue
        row = _safe_row(panel.columns, date)
        if signal["SPY"] > signal["SHY"]:
            winner = "SPY" if signal["SPY"] >= signal["VEU"] else "VEU"
            row[:] = 0.0
            row[winner] = 1.0
        else:
            row[:] = 0.0
            row["AGG"] = 1.0
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _defensive_momentum_targets(panel: pd.DataFrame, top_n: int) -> pd.DataFrame:
    universe = ["IEF", "TLT", "GLD", "DBC", "SHY"]
    momentum = panel[universe] / panel[universe].shift(126) - 1.0
    rows = []
    for date in _decision_dates(panel.index, "M"):
        winners = momentum.loc[date].dropna().nlargest(top_n).index
        row = _safe_row(panel.columns, date)
        if len(winners):
            row[:] = 0.0
            row.loc[winners] = 1.0 / len(winners)
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _credit_canary_targets(panel: pd.DataFrame) -> pd.DataFrame:
    credit_ratio = panel["HYG"] / panel["IEF"]
    credit_ma = credit_ratio.rolling(200, min_periods=200).mean()
    spy_ma = panel["SPY"].rolling(200, min_periods=200).mean()
    rows = []
    for date in _decision_dates(panel.index, "W"):
        if pd.isna(credit_ma.loc[date]) or pd.isna(spy_ma.loc[date]):
            continue
        risk_on = (
            credit_ratio.loc[date] > credit_ma.loc[date]
            and panel.loc[date, "SPY"] > spy_ma.loc[date]
        )
        row = _safe_row(panel.columns, date)
        if risk_on:
            row[:] = 0.0
            row["SPY"] = 0.75
            row["QQQ"] = 0.25
        else:
            row[:] = 0.0
            row["IEF"] = 0.50
            row["GLD"] = 0.25
            row["SHY"] = 0.25
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _sector_momentum_targets(
    panel: pd.DataFrame,
    lookback: int | None,
    *,
    inverse_vol: bool,
    multi_horizon: bool = False,
) -> pd.DataFrame:
    returns = panel[SECTORS].pct_change()
    rows = []
    for date in _decision_dates(panel.index, "M"):
        if multi_horizon:
            ranks = []
            for horizon in (63, 126, 252):
                score = panel.loc[date, SECTORS] / panel[SECTORS].shift(horizon).loc[date] - 1.0
                ranks.append(score.rank(pct=True))
            signal = pd.concat(ranks, axis=1).mean(axis=1)
        else:
            signal = panel.loc[date, SECTORS] / panel[SECTORS].shift(lookback).loc[date] - 1.0
        winners = signal.dropna().nlargest(3).index
        row = _safe_row(panel.columns, date)
        if len(winners):
            row[:] = 0.0
            if inverse_vol:
                vol = returns.loc[:date].tail(63).std().reindex(winners)
                inv = 1.0 / vol.replace(0.0, np.nan)
                inv = inv.dropna()
                row.loc[inv.index] = inv / inv.sum()
            else:
                row.loc[winners] = 1.0 / len(winners)
            row["SHY"] += 1.0 - float(row.sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _sector_downside_targets(panel: pd.DataFrame) -> pd.DataFrame:
    returns = panel[SECTORS].pct_change()
    ma = panel[SECTORS].rolling(200, min_periods=200).mean()
    rows = []
    for date in _decision_dates(panel.index, "M"):
        eligible = panel.loc[date, SECTORS] > ma.loc[date, SECTORS]
        window = returns.loc[:date, SECTORS].tail(126)
        downside = window.clip(upper=0.0).pow(2).mean().pow(0.5)
        raw = (1.0 / downside.replace(0.0, np.nan)).where(eligible)
        alloc = _allocate_capped(raw, 1.0, 0.25)
        row = _safe_row(panel.columns, date)
        row.loc[SECTORS] = alloc
        row["SHY"] = 1.0 - float(alloc.sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _sector_barbell_targets(panel: pd.DataFrame) -> pd.DataFrame:
    returns = panel[SECTORS].pct_change()
    ma = panel[SECTORS].rolling(200, min_periods=200).mean()
    momentum = panel[CYCLICAL_SECTORS] / panel[CYCLICAL_SECTORS].shift(126) - 1.0
    rows = []
    for date in _decision_dates(panel.index, "M"):
        row = _safe_row(panel.columns, date)
        row[:] = 0.0
        downside = returns.loc[:date, DEFENSIVE_SECTORS].tail(126).clip(upper=0.0).pow(2).mean().pow(0.5)
        inv = 1.0 / downside.replace(0.0, np.nan)
        if inv.notna().any():
            row.loc[inv.dropna().index] = inv.dropna() / inv.dropna().sum() * 0.50
        eligible = momentum.loc[date].where(
            panel.loc[date, CYCLICAL_SECTORS] > ma.loc[date, CYCLICAL_SECTORS]
        ).dropna().nlargest(3)
        if not eligible.empty:
            row.loc[eligible.index] = 0.50 / 3.0
        row["SHY"] = 1.0 - float(row.sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _sector_vol_budget_targets(panel: pd.DataFrame) -> pd.DataFrame:
    base = _sector_downside_targets(panel)
    returns = panel[SECTORS].pct_change()
    spy_ma = panel["SPY"].rolling(200, min_periods=200).mean()
    rows = []
    for date, target in base.iterrows():
        weights = target.reindex(SECTORS, fill_value=0.0)
        trailing = returns.loc[:date].tail(126).mul(weights, axis=1).sum(axis=1)
        downside = float(np.sqrt(trailing.clip(upper=0.0).pow(2).mean()) * np.sqrt(PERIODS))
        exposure = 0.0 if not np.isfinite(downside) or downside <= 0.0 else min(1.0, 0.08 / downside)
        if panel.loc[date, "SPY"] <= spy_ma.loc[date]:
            exposure = min(exposure, 0.50)
        row = _safe_row(panel.columns, date)
        row.loc[SECTORS] = weights * exposure
        row["SHY"] = 1.0 - float(row.loc[SECTORS].sum())
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _sector_low_ulcer_targets(panel: pd.DataFrame) -> pd.DataFrame:
    ma = panel[SECTORS].rolling(200, min_periods=200).mean()
    momentum = panel[SECTORS] / panel[SECTORS].shift(252) - 1.0
    rows = []
    for date in _decision_dates(panel.index, "M"):
        scores = {}
        for ticker in SECTORS:
            window = panel.loc[:date, ticker].tail(252)
            if len(window) < 252:
                continue
            path_dd = window / window.cummax() - 1.0
            scores[ticker] = float(np.sqrt(path_dd.pow(2).mean()))
        eligible = [
            ticker
            for ticker, score in scores.items()
            if momentum.loc[date, ticker] > 0.0 and panel.loc[date, ticker] > ma.loc[date, ticker]
        ]
        winners = sorted(eligible, key=scores.get)[:4]
        row = _safe_row(panel.columns, date)
        if winners:
            row.loc[winners] = 0.25
            row["SHY"] = 1.0 - 0.25 * len(winners)
        rows.append(row)
    return _targets_from_rows(rows, panel.columns)


def _add_rule_strategies(adjusted: pd.DataFrame) -> list[StrategyRun]:
    runs: list[StrategyRun] = []

    def add(**kwargs) -> None:
        runs.append(_run_weight_strategy(adjusted, **kwargs))

    fixed_specs = {
        "SPY_buy_hold": ("static_equity", {"SPY": 1.0}),
        "QQQ_buy_hold": ("static_equity", {"QQQ": 1.0}),
        "SPY_80_IEF_20": ("balanced_static", {"SPY": 0.80, "IEF": 0.20}),
        "SPY_70_IEF_30": ("balanced_static", {"SPY": 0.70, "IEF": 0.30}),
        "SPY_60_IEF_40": ("balanced_static", {"SPY": 0.60, "IEF": 0.40}),
        "SPY_70_TLT_30": ("balanced_static", {"SPY": 0.70, "TLT": 0.30}),
        "SPY_60_TLT_40": ("balanced_static", {"SPY": 0.60, "TLT": 0.40}),
        "QQQ_70_IEF_30": ("balanced_growth", {"QQQ": 0.70, "IEF": 0.30}),
        "QQQ_60_IEF_40": ("balanced_growth", {"QQQ": 0.60, "IEF": 0.40}),
        "SPY_60_IEF_30_GLD_10": ("balanced_static", {"SPY": 0.60, "IEF": 0.30, "GLD": 0.10}),
        "SPY_50_IEF_30_GLD_20": ("balanced_static", {"SPY": 0.50, "IEF": 0.30, "GLD": 0.20}),
        "global_50_20_10_bond_gold": (
            "global_balanced",
            {"SPY": 0.50, "EFA": 0.20, "EEM": 0.10, "IEF": 0.15, "GLD": 0.05},
        ),
        "retail_all_weather_fixed": ("all_weather_static", ALL_WEATHER_WEIGHTS),
        "five_weather_equal": (
            "all_weather_static",
            {ticker: 1.0 / len(EQUAL_WEATHER_ASSETS) for ticker in EQUAL_WEATHER_ASSETS},
        ),
        "legacy_sectors_equal": ("sector_static", {ticker: 1.0 / len(SECTORS) for ticker in SECTORS}),
        "defensive_sectors_equal": (
            "sector_defensive",
            {ticker: 1.0 / len(DEFENSIVE_SECTORS) for ticker in DEFENSIVE_SECTORS},
        ),
        "sector_defensive_cyclical_barbell_fixed": (
            "sector_static",
            {
                **{ticker: 0.50 / len(DEFENSIVE_SECTORS) for ticker in DEFENSIVE_SECTORS},
                **{ticker: 0.50 / len(CYCLICAL_SECTORS) for ticker in CYCLICAL_SECTORS},
            },
        ),
    }
    for name, (group, weights) in fixed_specs.items():
        assets = list(weights)
        add(
            name=name,
            family="Fixed allocation",
            independence_group=group,
            description=f"Quarterly rebalanced fixed ETF weights: {weights}",
            assets=assets,
            target_builder=lambda panel, weights=weights: _fixed_targets(panel, weights, freq="Q"),
        )

    for risky, lookbacks in {"SPY": (100, 150, 200, 250), "QQQ": (150, 200)}.items():
        for lookback in lookbacks:
            add(
                name=f"{risky}_trend_{lookback}d",
                family="Equity P&L control",
                independence_group="pnl_trend",
                description=f"Weekly {risky} above {lookback}d moving average; otherwise SHY.",
                assets=[risky, "SHY"],
                target_builder=lambda panel, risky=risky, lookback=lookback: _trend_targets(
                    panel, risky, lookback
                ),
            )
    for risky, targets in {"SPY": (0.10, 0.12, 0.15), "QQQ": (0.12, 0.15)}.items():
        for target in targets:
            add(
                name=f"{risky}_vol_target_{int(target * 100)}",
                family="Equity P&L control",
                independence_group="pnl_vol",
                description=f"Weekly unlevered {risky} 63d volatility target {target:.0%}; residual SHY.",
                assets=[risky, "SHY"],
                target_builder=lambda panel, risky=risky, target=target: _vol_targets(panel, risky, target),
            )
    for target in (0.10, 0.12, 0.15):
        add(
            name=f"SPY_vol_{int(target * 100)}_trend_200",
            family="Equity P&L control",
            independence_group="pnl_vol_trend",
            description=f"Weekly SPY {target:.0%} vol target, disabled below 200d MA; residual SHY.",
            assets=["SPY", "SHY"],
            target_builder=lambda panel, target=target: _vol_targets(
                panel, "SPY", target, trend_lookback=200
            ),
        )
    for policy in ("soft", "medium", "hard"):
        add(
            name=f"SPY_drawdown_{policy}",
            family="Equity P&L control",
            independence_group="pnl_drawdown",
            description=f"Weekly SPY high-water drawdown exposure ladder ({policy}); residual SHY.",
            assets=["SPY", "SHY"],
            target_builder=lambda panel, policy=policy: _drawdown_targets(panel, "SPY", policy),
        )
    add(
        name="SPY_fast_exit_slow_reentry",
        family="Equity P&L control",
        independence_group="pnl_hysteresis",
        description="Weekly SPY exit on 200d/VIX failure; re-enter after four consecutive good weeks.",
        assets=["SPY", "SHY", "^VIX"],
        target_builder=lambda panel: _hysteresis_targets(panel, "SPY"),
    )

    weather_assets = EQUAL_WEATHER_ASSETS + ["SHY"]
    for lookback in (63, 126, 252):
        add(
            name=f"all_weather_inverse_vol_{lookback}",
            family="All weather / risk balance",
            independence_group="all_weather_dynamic",
            description=f"Monthly inverse-vol across stock, Treasury, gold, commodity sleeves; {lookback}d risk.",
            assets=weather_assets,
            target_builder=lambda panel, lookback=lookback: _inverse_vol_targets(
                panel, EQUAL_WEATHER_ASSETS, lookback
            ),
        )
    add(
        name="all_weather_inverse_vol_126_cap35",
        family="All weather / risk balance",
        independence_group="all_weather_dynamic",
        description="Monthly 126d inverse-vol all-weather with 35% sleeve cap; residual SHY.",
        assets=weather_assets,
        target_builder=lambda panel: _inverse_vol_targets(panel, EQUAL_WEATHER_ASSETS, 126, cap=0.35),
    )
    for lookback in (100, 200):
        add(
            name=f"all_weather_sleeve_trend_{lookback}",
            family="All weather / risk balance",
            independence_group="all_weather_trend",
            description=f"Fixed all-weather sleeve budgets; failed {lookback}d trend moves to SHY.",
            assets=weather_assets,
            target_builder=lambda panel, lookback=lookback: _weather_trend_targets(panel, lookback),
        )
    for target in (0.10, 0.12):
        add(
            name=f"all_weather_vol_target_{int(target * 100)}",
            family="All weather / risk balance",
            independence_group="all_weather_vol",
            description=f"Monthly unlevered fixed all-weather portfolio scaled to {target:.0%} vol; residual SHY.",
            assets=weather_assets,
            target_builder=lambda panel, target=target: _weather_vol_target(panel, target),
        )

    tactical_panel = TACTICAL_ASSETS + ["SHY"]
    for lookback in (100, 200):
        add(
            name=f"GTAA_trend_{lookback}",
            family="Cross-asset tactical",
            independence_group="tactical_trend",
            description=f"Monthly equal sleeve GTAA; each asset below {lookback}d MA moves to SHY.",
            assets=tactical_panel,
            target_builder=lambda panel, lookback=lookback: _gtna_targets(panel, lookback),
        )
    for lookback in (126, 252):
        for top_n in (1, 2, 3, 4):
            add(
                name=f"cross_asset_mom_{lookback}_top{top_n}",
                family="Cross-asset tactical",
                independence_group="tactical_momentum",
                description=f"Monthly top-{top_n} {lookback}d cross-asset momentum with 12m SHY hurdle.",
                assets=tactical_panel,
                target_builder=lambda panel, lookback=lookback, top_n=top_n: _momentum_targets(
                    panel, top_n=top_n, lookback=lookback
                ),
            )
    for top_n in (2, 3):
        add(
            name=f"cross_asset_multi_horizon_top{top_n}",
            family="Cross-asset tactical",
            independence_group="tactical_momentum",
            description=f"Monthly top-{top_n} average 3/6/12m ranks with 12m SHY hurdle.",
            assets=tactical_panel,
            target_builder=lambda panel, top_n=top_n: _momentum_targets(
                panel, top_n=top_n, horizons=(63, 126, 252)
            ),
        )
    add(
        name="cross_asset_multi_horizon_top3_invvol",
        family="Cross-asset tactical",
        independence_group="tactical_momentum",
        description="Monthly top-3 3/6/12m ranks, 12m SHY hurdle, 63d inverse-vol sizing.",
        assets=tactical_panel,
        target_builder=lambda panel: _momentum_targets(
            panel, top_n=3, horizons=(63, 126, 252), inverse_vol=True
        ),
    )
    add(
        name="GEM_SPY_VEU_AGG",
        family="Cross-asset tactical",
        independence_group="dual_momentum",
        description="Monthly GEM: SPY absolute momentum versus SHY, SPY versus VEU, else AGG.",
        assets=["SPY", "VEU", "AGG", "SHY"],
        target_builder=_gem_targets,
    )
    for top_n in (1, 2):
        add(
            name=f"defensive_momentum_top{top_n}",
            family="Cross-asset tactical",
            independence_group="defensive_momentum",
            description=f"Monthly top-{top_n} 6m momentum among IEF/TLT/GLD/DBC/SHY.",
            assets=["IEF", "TLT", "GLD", "DBC", "SHY"],
            target_builder=lambda panel, top_n=top_n: _defensive_momentum_targets(panel, top_n),
        )
    add(
        name="credit_canary_equity_allocation",
        family="Macro-conditioned allocation",
        independence_group="credit_regime",
        description="Weekly SPY/QQQ only when HYG/IEF and SPY trends are positive; otherwise IEF/GLD/SHY.",
        assets=["SPY", "QQQ", "HYG", "IEF", "GLD", "SHY"],
        target_builder=_credit_canary_targets,
    )

    sector_assets = SECTORS + ["SPY", "SHY"]
    for lookback in (126, 189, 252):
        for inverse_vol in (False, True):
            suffix = "invvol" if inverse_vol else "equal"
            add(
                name=f"sector_momentum_{lookback}_{suffix}",
                family="Sector allocation",
                independence_group="sector_rotation",
                description=f"Monthly top-3 legacy-sector {lookback}d momentum, {suffix} sizing.",
                assets=sector_assets,
                target_builder=lambda panel, lookback=lookback, inverse_vol=inverse_vol: _sector_momentum_targets(
                    panel, lookback, inverse_vol=inverse_vol
                ),
            )
    for inverse_vol in (False, True):
        suffix = "invvol" if inverse_vol else "equal"
        add(
            name=f"sector_momentum_multi_horizon_{suffix}",
            family="Sector allocation",
            independence_group="sector_rotation",
            description=f"Monthly top-3 average 3/6/12m sector ranks, {suffix} sizing.",
            assets=sector_assets,
            target_builder=lambda panel, inverse_vol=inverse_vol: _sector_momentum_targets(
                panel, None, inverse_vol=inverse_vol, multi_horizon=True
            ),
        )
    add(
        name="sector_downside_trend",
        family="Sector allocation",
        independence_group="sector_downside",
        description="Monthly sectors above 200d MA, inverse 126d downside deviation, 25% cap; residual SHY.",
        assets=sector_assets,
        target_builder=_sector_downside_targets,
    )
    add(
        name="sector_defensive_upside_barbell",
        family="Sector allocation",
        independence_group="sector_barbell",
        description="50% defensive inverse-downside sleeve plus trend-eligible top cyclical sectors; residual SHY.",
        assets=sector_assets,
        target_builder=_sector_barbell_targets,
    )
    add(
        name="sector_downside_vol_budget",
        family="Sector allocation",
        independence_group="sector_downside",
        description="Downside/trend sectors scaled to 8% downside-vol budget; 50% cap below SPY 200d MA.",
        assets=sector_assets,
        target_builder=_sector_vol_budget_targets,
    )
    add(
        name="sector_low_ulcer_positive_trend",
        family="Sector allocation",
        independence_group="sector_ulcer",
        description="Monthly four lowest 252d-Ulcer sectors with positive return and 200d trend; residual SHY.",
        assets=sector_assets,
        target_builder=_sector_low_ulcer_targets,
    )
    return runs


def _add_option_runs(
    adjusted: pd.DataFrame,
    raw_spy: pd.Series,
) -> list[StrategyRun]:
    frame = pd.concat(
        [
            adjusted["SPY"].rename("SPY"),
            raw_spy,
            adjusted["^VIX"].rename("VIX"),
            adjusted["^IRX"].rename("IRX"),
            adjusted["SHY"].rename("SHY"),
        ],
        axis=1,
    ).dropna()
    frame = frame.loc[:LAST_ALLOWED]
    base_returns = frame["SPY"].pct_change().fillna(0.0)
    cash_returns = frame["SHY"].pct_change().fillna(0.0)
    annual_rate = (frame["IRX"] / 100.0).shift(1).ffill().fillna(0.0)
    calm_ratio = (frame["VIX"].shift(1) < 20.0).astype(float)

    specs = {
        "SPY_put_50": (0.50, 0.0, SyntheticOptionOverlayConfig()),
        "SPY_put_100": (1.00, 0.0, SyntheticOptionOverlayConfig()),
        "SPY_95_85_put_spread": (
            1.00,
            0.0,
            SyntheticOptionOverlayConfig(short_put_otm=0.15),
        ),
        "SPY_95_110_collar": (
            1.00,
            1.00,
            SyntheticOptionOverlayConfig(call_otm=0.10),
        ),
        "SPY_95_105_collar": (
            1.00,
            1.00,
            SyntheticOptionOverlayConfig(call_otm=0.05),
        ),
        "SPY_90_110_collar": (
            1.00,
            1.00,
            SyntheticOptionOverlayConfig(long_put_otm=0.10, call_otm=0.10),
        ),
        "SPY_calm_VIX_95_85_put_spread": (
            calm_ratio,
            0.0,
            SyntheticOptionOverlayConfig(short_put_otm=0.15),
        ),
    }
    runs = []
    for name, (put_ratio, call_ratio, config) in specs.items():
        primary = run_synthetic_option_overlay(
            base_returns,
            frame["SPY"],
            frame["raw_SPY"],
            frame["VIX"],
            cash_returns,
            annual_rate,
            put_ratio,
            call_ratio,
            config=config,
        )
        stress_config = SyntheticOptionOverlayConfig(
            long_put_otm=config.long_put_otm,
            short_put_otm=config.short_put_otm,
            call_otm=config.call_otm,
            long_put_iv_buffer=config.long_put_iv_buffer,
            short_put_iv_buffer=config.short_put_iv_buffer,
            call_iv_haircut=config.call_iv_haircut,
            realized_vol_buffer=config.realized_vol_buffer,
            minimum_call_iv=config.minimum_call_iv,
            long_option_ask_markup=0.20,
            short_option_bid_haircut=0.20,
        )
        stress = run_synthetic_option_overlay(
            base_returns,
            frame["SPY"],
            frame["raw_SPY"],
            frame["VIX"],
            cash_returns,
            annual_rate,
            put_ratio,
            call_ratio,
            config=stress_config,
        )
        start = max(CORE_START, primary.returns.index.min())
        runs.append(
            StrategyRun(
                name=name,
                family="Synthetic SPY option overlay",
                independence_group="option_overlay",
                evidence="synthetic_proxy",
                description=(
                    f"Synthetic SPY overlay: put ratio {name}, frozen VIX/Black-Scholes/skew model; "
                    "quarterly puts and monthly calls where applicable."
                ),
                returns=primary.returns.loc[start:CORE_END].rename(name),
                stress_returns=stress.returns.loc[start:CORE_END].rename(name),
                annual_turnover=float("nan"),
                latest_weights={},
                implementation_note="SPY options; model-risk screen, not historical executable chains",
            )
        )
    return runs


def _product_group(ticker: str) -> tuple[str, str]:
    if ticker in {"SPLV", "USMV", "MTUM", "QUAL", "QVAL", "QMOM", "VMOT", "OMFL", "VIG", "VYM", "RSP"}:
        return "Live smart-beta ETF", "smart_beta_live"
    if ticker in {"DBMF", "KMLM", "WTMF", "QAI"}:
        return "Live alternative/trend ETF", "managed_futures_live"
    if ticker in {"PHDG", "TAIL", "SWAN", "UJAN"}:
        return "Live protection ETF", "protection_live"
    if ticker in {"NTSX"}:
        return "Live capital-efficient ETF", "capital_efficiency_live"
    if ticker in {"RPAR"}:
        return "Live risk-parity ETF", "risk_parity_live"
    return "Live allocation ETF", "allocation_live"


def _add_product_runs(adjusted: pd.DataFrame) -> list[StrategyRun]:
    runs = []
    for ticker in PRODUCT_TICKERS:
        series = adjusted[ticker].dropna().loc[:LAST_ALLOWED]
        if len(series) < 100:
            continue
        returns = series.pct_change().fillna(0.0)
        if len(returns) > 1:
            returns.iloc[1] -= PRIMARY_TRADING_BPS / 10_000.0
        stress = series.pct_change().fillna(0.0)
        if len(stress) > 1:
            stress.iloc[1] -= STRESS_TRADING_BPS / 10_000.0
        family, group = _product_group(ticker)
        years = len(returns) / PERIODS
        evidence = "live_product_medium" if years >= 8.0 else "short_live_product"
        runs.append(
            StrategyRun(
                name=f"ETF_{ticker}",
                family=family,
                independence_group=group,
                evidence=evidence,
                description=f"Buy-and-hold live ETF {ticker}, evaluated only from its actual inception through 2021.",
                returns=returns.rename(f"ETF_{ticker}"),
                stress_returns=stress.rename(f"ETF_{ticker}"),
                annual_turnover=0.0,
                latest_weights={ticker: 1.0},
            )
        )
    for ticker, label in {"^PUT": "Cboe_SPX_PUT_proxy", "^BXM": "Cboe_SPX_BXM_proxy"}.items():
        series = adjusted[ticker].dropna().loc[CORE_START:CORE_END]
        if len(series) < 100:
            continue
        returns = series.pct_change().fillna(0.0)
        runs.append(
            StrategyRun(
                name=label,
                family="External SPX option-index proxy",
                independence_group="external_option_proxy",
                evidence="external_proxy",
                description=f"Cboe {ticker} total-return index proxy; SPX options make it ineligible for implementation.",
                returns=returns.rename(label),
                stress_returns=returns.rename(label),
                annual_turnover=float("nan"),
                latest_weights={},
                implementation_note="External SPX option index; reference only",
            )
        )
    return runs


def _annualized_return(returns: pd.Series) -> float:
    values = returns.dropna()
    if values.empty or (values <= -1.0).any():
        return float("nan")
    return float(np.exp(np.log1p(values).sum() * PERIODS / len(values)) - 1.0)


def _drawdown_duration_metrics(returns: pd.Series) -> dict[str, float | int]:
    """Map the reusable 5%/20-session duration metrics to report columns."""
    duration = drawdown_duration_metrics(
        returns,
        material_threshold=0.05,
        recovery_target_days=20,
    )
    return {
        "max_underwater_days": duration["max_underwater_days"],
        "max_trough_to_recovery_days": duration["max_trough_to_recovery_days"],
        "max_5pct_trough_to_recovery_days": duration["max_material_recovery_days"],
        "median_5pct_trough_to_recovery_days": duration[
            "median_material_recovery_days"
        ],
        "share_5pct_recovered_within_20d": duration[
            "share_material_recovered_within_target"
        ],
        "n_5pct_drawdowns": duration["material_drawdown_count"],
        "n_unrecovered_5pct_drawdowns": duration[
            "unrecovered_material_drawdown_count"
        ],
    }


def _rolling_cagr(returns: pd.Series, window: int) -> pd.Series:
    logs = np.log1p(returns)
    return np.exp(logs.rolling(window, min_periods=window).sum() * PERIODS / window) - 1.0


def _metric_row(run: StrategyRun, shy_returns: pd.Series) -> dict[str, object]:
    r = run.returns.dropna()
    stress = run.stress_returns.reindex(r.index).dropna()
    cash = shy_returns.reindex(r.index).fillna(0.0)
    excess = r - cash
    dd = drawdown(r)
    years = len(r) / PERIODS
    calendar = r.groupby(r.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    yearly_log_excess = (np.log1p(r) - np.log1p(cash)).groupby(r.index.year).sum()
    concentration = (
        float(yearly_log_excess.abs().max() / yearly_log_excess.abs().sum())
        if yearly_log_excess.abs().sum() > 0.0
        else np.nan
    )
    tail_cut = float(r.quantile(0.05))
    cvar = float(r[r <= tail_cut].mean())
    rolling3 = _rolling_cagr(r, 756).dropna()
    rolling5 = _rolling_cagr(r, 1260).dropna()
    sharpe = float(excess.mean() / excess.std() * np.sqrt(PERIODS)) if excess.std() > 0 else np.nan
    duration = _drawdown_duration_metrics(r)
    row: dict[str, object] = {
        "strategy": run.name,
        "family": run.family,
        "independence_group": run.independence_group,
        "evidence": run.evidence,
        "description": run.description,
        "implementation_note": run.implementation_note,
        "start": r.index.min().date().isoformat(),
        "end": r.index.max().date().isoformat(),
        "years": years,
        "cagr": _annualized_return(r),
        "stress_cagr": _annualized_return(stress),
        "annual_vol": float(r.std() * np.sqrt(PERIODS)),
        "excess_shy_sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "ulcer_index": float(np.sqrt(dd.pow(2).mean())),
        "calmar": _annualized_return(r) / abs(float(dd.min())) if dd.min() < 0 else np.nan,
        "cvar_5pct_daily": cvar,
        "worst_calendar_year": float(calendar.min()),
        "best_calendar_year": float(calendar.max()),
        # Retain the legacy column name for artifact compatibility. It measures
        # total time underwater, not the trough-to-prior-high recovery leg.
        "max_recovery_days": duration["max_underwater_days"],
        **duration,
        "largest_year_abs_excess_share": concentration,
        "annual_turnover": run.annual_turnover,
        "cagr_2007_2012": _annualized_return(r.loc["2007":"2012"]),
        "cagr_2013_2021": _annualized_return(r.loc["2013":"2021"]),
        "rolling_3y_min": float(rolling3.min()) if not rolling3.empty else np.nan,
        "rolling_3y_target_share": float((rolling3 >= TARGET_RETURN).mean()) if not rolling3.empty else np.nan,
        "rolling_5y_min": float(rolling5.min()) if not rolling5.empty else np.nan,
        "rolling_5y_target_share": float((rolling5 >= TARGET_RETURN).mean()) if not rolling5.empty else np.nan,
        "latest_weights": json.dumps(run.latest_weights, ensure_ascii=False, sort_keys=True),
    }
    for event, (start, end) in STRESS_WINDOWS.items():
        event_r = r.loc[start:end]
        complete = not event_r.empty and r.index.min() <= pd.Timestamp(start) and r.index.max() >= pd.Timestamp(end)
        row[f"{event}_return"] = float((1.0 + event_r).prod() - 1.0) if complete else np.nan
        row[f"{event}_maxdd"] = float(drawdown(event_r).min()) if complete else np.nan
    return row


def _build_metrics(runs: list[StrategyRun], adjusted: pd.DataFrame) -> pd.DataFrame:
    shy_returns = adjusted["SHY"].pct_change().fillna(0.0)
    metrics = pd.DataFrame([_metric_row(run, shy_returns) for run in runs]).set_index("strategy")
    rule_mask = metrics["evidence"].isin(["long_rule", "synthetic_proxy"])
    rule_sharpes = metrics.loc[rule_mask, "excess_shy_sharpe"].dropna().to_numpy()
    metrics["deflated_sharpe_probability"] = np.nan
    for name in metrics.index[rule_mask]:
        r = next(run.returns for run in runs if run.name == name)
        excess = r - shy_returns.reindex(r.index).fillna(0.0)
        observed = float(metrics.loc[name, "excess_shy_sharpe"])
        metrics.loc[name, "deflated_sharpe_probability"] = deflated_sharpe_ratio(
            observed,
            n_obs=len(excess),
            trial_sharpes=rule_sharpes,
            periods=PERIODS,
            skew=float(excess.skew()),
            kurt=float(excess.kurt() + 3.0),
        )["dsr"]
    metrics["target_gate"] = (
        (metrics["evidence"] == "long_rule")
        & (metrics["years"] >= 14.0)
        & (metrics["cagr"] >= 0.09)
        & (metrics["stress_cagr"] >= 0.085)
        & (metrics["annual_vol"] <= 0.15)
        & (metrics["max_drawdown"] >= -0.25)
        & (metrics["cagr_2013_2021"] >= 0.08)
        & (metrics["largest_year_abs_excess_share"] <= 0.50)
    )
    metrics["selection_score"] = (
        6.0 * metrics["cagr"].clip(upper=0.14)
        - 2.0 * metrics["annual_vol"]
        + 1.5 * metrics["max_drawdown"]
        - 1.0 * metrics["ulcer_index"]
        + 0.5 * metrics["calmar"].clip(upper=2.0)
        + 0.25 * metrics["cagr_2013_2021"].fillna(-1.0)
    )
    return metrics.sort_values(["target_gate", "selection_score"], ascending=[False, False])


def _select_independent(metrics: pd.DataFrame) -> pd.DataFrame:
    group_order = [
        "balanced_static",
        "balanced_growth",
        "pnl_trend",
        "pnl_vol",
        "pnl_drawdown",
        "all_weather_dynamic",
        "all_weather_trend",
        "tactical_trend",
        "tactical_momentum",
        "dual_momentum",
        "credit_regime",
        "sector_downside",
        "sector_barbell",
        "sector_ulcer",
        "option_overlay",
        "smart_beta_live",
        "managed_futures_live",
        "protection_live",
        "capital_efficiency_live",
    ]
    rows = []
    for group in group_order:
        candidates = metrics[metrics["independence_group"] == group].copy()
        if candidates.empty:
            continue
        candidates["evidence_rank"] = candidates["evidence"].map(
            {
                "long_rule": 4,
                "live_product_medium": 3,
                "synthetic_proxy": 2,
                "short_live_product": 1,
                "external_proxy": 0,
            }
        ).fillna(0)
        winner = candidates.sort_values(
            ["target_gate", "evidence_rank", "selection_score"],
            ascending=[False, False, False],
        ).iloc[0].copy()
        winner["selection_status"] = (
            "pass" if bool(winner["target_gate"]) else "near_target" if winner["cagr"] >= 0.08 else "research_only"
        )
        rows.append(winner)
    selected = pd.DataFrame(rows)
    selected.index.name = "strategy"
    return selected


def _regime_table(selected_names: list[str], runs: list[StrategyRun], adjusted: pd.DataFrame) -> pd.DataFrame:
    spy = adjusted["SPY"].dropna().loc[:LAST_ALLOWED]
    spy_return = spy.pct_change()
    prior_price = spy.shift(1)
    prior_ma = spy.rolling(200, min_periods=200).mean().shift(1)
    prior_rv = (spy_return.rolling(21, min_periods=21).std() * np.sqrt(PERIODS)).shift(1)
    prior_vol_threshold = prior_rv.rolling(756, min_periods=252).median().shift(1)
    state = pd.Series(index=spy.index, dtype="object")
    bull = prior_price > prior_ma
    high = prior_rv > prior_vol_threshold
    state.loc[bull & ~high] = "bull_low_vol"
    state.loc[bull & high] = "bull_high_vol"
    state.loc[~bull & ~high] = "bear_low_vol"
    state.loc[~bull & high] = "bear_high_vol"
    rows = []
    run_map = {run.name: run for run in runs}
    for name in selected_names:
        r = run_map[name].returns
        for regime in ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]:
            sample = r[state.reindex(r.index) == regime].dropna()
            rows.append(
                {
                    "strategy": name,
                    "regime": regime,
                    "observations": len(sample),
                    "share": len(sample) / max(len(r), 1),
                    "conditional_ann_return": _annualized_return(sample),
                    "conditional_ann_vol": float(sample.std() * np.sqrt(PERIODS)) if len(sample) > 1 else np.nan,
                    "worst_day": float(sample.min()) if len(sample) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_table(selected_names: list[str], runs: list[StrategyRun], draws: int = 2000) -> pd.DataFrame:
    run_map = {run.name: run for run in runs}
    rows = []
    for name in selected_names:
        monthly = (1.0 + run_map[name].returns).resample("ME").prod() - 1.0
        values = monthly.dropna().to_numpy()
        if len(values) < 36:
            rows.append(
                {
                    "strategy": name,
                    "draws": 0,
                    "cagr_p05": np.nan,
                    "cagr_median": np.nan,
                    "prob_cagr_ge_8": np.nan,
                    "prob_cagr_ge_10": np.nan,
                    "maxdd_p05": np.nan,
                }
            )
            continue
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        block = 6
        n_blocks = int(np.ceil(len(values) / block))
        cagr_draws = np.empty(draws)
        maxdd_draws = np.empty(draws)
        for draw_idx in range(draws):
            starts = rng.integers(0, len(values) - block + 1, size=n_blocks)
            sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
            cagr_draws[draw_idx] = np.exp(np.log1p(sample).mean() * 12.0) - 1.0
            equity = np.cumprod(1.0 + sample)
            maxdd_draws[draw_idx] = np.min(equity / np.maximum.accumulate(equity) - 1.0)
        rows.append(
            {
                "strategy": name,
                "draws": draws,
                "cagr_p05": float(np.percentile(cagr_draws, 5.0)),
                "cagr_median": float(np.median(cagr_draws)),
                "prob_cagr_ge_8": float(np.mean(cagr_draws >= 0.08)),
                "prob_cagr_ge_10": float(np.mean(cagr_draws >= 0.10)),
                "maxdd_p05": float(np.percentile(maxdd_draws, 5.0)),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def _selected_returns(selected_names: list[str], runs: list[StrategyRun]) -> pd.DataFrame:
    run_map = {run.name: run for run in runs}
    return pd.concat([run_map[name].returns.rename(name) for name in selected_names], axis=1)


def _figure_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    center: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    decimals: int = 2,
) -> None:
    """Draw a small annotated heatmap without an optional seaborn dependency."""
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if vmin is None:
        vmin = float(finite.min()) if len(finite) else center - 1.0
    if vmax is None:
        vmax = float(finite.max()) if len(finite) else center + 1.0
    bound = max(abs(vmin - center), abs(vmax - center))
    vmin, vmax = center - bound, center + bound
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(frame.columns)), labels=frame.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(frame.index)), labels=frame.index)
    threshold = center + 0.55 * bound
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if abs(value - center) > abs(threshold - center) else "black"
                ax.text(column, row, f"{value:.{decimals}f}", ha="center", va="center", fontsize=6, color=color)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, shrink=0.8)


def _build_charts(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    returns: pd.DataFrame,
    stress: pd.DataFrame,
    correlation: pd.DataFrame,
) -> dict[str, str]:
    plt.style.use("ggplot")
    long = metrics[metrics["evidence"].isin(["long_rule", "synthetic_proxy"])].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    points = ax.scatter(
        long["annual_vol"] * 100,
        long["cagr"] * 100,
        c=(-long["max_drawdown"] * 100),
        cmap="magma_r",
        s=55,
        alpha=0.75,
    )
    ax.axvline(15, color="#b22222", linestyle="--", linewidth=1)
    ax.axhline(10, color="#1f5f99", linestyle="--", linewidth=1)
    for name in selected.index:
        if name in long.index:
            row = long.loc[name]
            ax.annotate(name, (row["annual_vol"] * 100, row["cagr"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(title="Long-history rules and synthetic option screens", xlabel="Annual volatility (%)", ylabel="CAGR (%)")
    fig.colorbar(points, ax=ax, label="Maximum drawdown magnitude (%)")
    scatter = _figure_uri(fig)

    common = returns.dropna(how="all").fillna(0.0)
    equity = (1.0 + common).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    equity.plot(ax=ax, linewidth=1.3)
    ax.set_yscale("log")
    ax.set(title="Selected alternatives: growth of $1 (individual available histories)", ylabel="Wealth, log scale", xlabel="")
    ax.legend(fontsize=7, ncol=2)
    equity_uri = _figure_uri(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for column in common:
        drawdown(common[column]).plot(ax=ax, label=column, linewidth=1.0)
    ax.set(title="Selected alternatives: drawdown paths", ylabel="Drawdown", xlabel="")
    ax.legend(fontsize=7, ncol=2)
    drawdown_uri = _figure_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    _heatmap(
        ax,
        correlation,
        title="Monthly return correlation among selected alternatives",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    correlation_uri = _figure_uri(fig)

    stress_matrix = stress.pivot(index="strategy", columns="event", values="event_return") * 100.0
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(stress_matrix))))
    _heatmap(
        ax,
        stress_matrix,
        title="Historical stress-window total returns (%)",
        cmap="RdYlGn",
        decimals=1,
    )
    ax.set(xlabel="", ylabel="")
    stress_uri = _figure_uri(fig)
    return {
        "scatter": scatter,
        "equity": equity_uri,
        "drawdown": drawdown_uri,
        "correlation": correlation_uri,
        "stress": stress_uri,
    }


def _format_table(frame: pd.DataFrame, percent: set[str] | None = None, decimals: int = 2) -> str:
    percent = percent or set()
    display = frame.copy()
    for column in display.columns:
        if column in percent:
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{100 * float(x):.{decimals}f}%")
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.{decimals}f}")
    return display.to_html(classes="data", border=0, escape=True)


def _render_report(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    bootstrap: pd.DataFrame,
    regime: pd.DataFrame,
    stress: pd.DataFrame,
    correlation: pd.DataFrame,
    charts: dict[str, str],
) -> str:
    core = metrics[metrics["evidence"].isin(["long_rule", "synthetic_proxy"])].copy()
    products = metrics[metrics["evidence"].isin(["live_product_medium", "short_live_product"])].copy()
    headline_cols = [
        "family",
        "evidence",
        "years",
        "cagr",
        "stress_cagr",
        "annual_vol",
        "max_drawdown",
        "max_underwater_days",
        "max_5pct_trough_to_recovery_days",
        "share_5pct_recovered_within_20d",
        "ulcer_index",
        "worst_calendar_year",
        "cagr_2007_2012",
        "cagr_2013_2021",
        "rolling_5y_target_share",
        "deflated_sharpe_probability",
        "target_gate",
    ]
    selected_view = selected[headline_cols + ["selection_status", "latest_weights"]]
    bootstrap_view = bootstrap.reindex(selected.index)
    regime_pivot = regime.pivot(index="strategy", columns="regime", values="conditional_ann_return")
    product_view = products.sort_values("selection_score", ascending=False)[
        ["family", "evidence", "start", "years", "cagr", "annual_vol", "max_drawdown", "worst_calendar_year"]
    ]
    top_core = core.sort_values(["target_gate", "selection_score"], ascending=[False, False]).head(35)[headline_cols]

    pct = {
        "cagr",
        "stress_cagr",
        "annual_vol",
        "max_drawdown",
        "share_5pct_recovered_within_20d",
        "ulcer_index",
        "worst_calendar_year",
        "cagr_2007_2012",
        "cagr_2013_2021",
        "rolling_5y_target_share",
        "deflated_sharpe_probability",
        "cagr_p05",
        "cagr_median",
        "prob_cagr_ge_8",
        "prob_cagr_ge_10",
        "maxdd_p05",
        "bull_low_vol",
        "bull_high_vol",
        "bear_low_vol",
        "bear_high_vol",
    }
    pass_count = int(metrics["target_gate"].sum())
    total_count = len(metrics)
    long_count = int((metrics["evidence"] == "long_rule").sum())
    source_links = [
        ("Bridgewater — The All Weather Story", "https://www.bridgewater.com/research-and-insights/the-all-weather-story"),
        ("RPAR official strategy page", "https://www.rparetf.com/rpar"),
        ("Cboe collar-index methodology", "https://cdn.cboe.com/api/global/us_indices/governance/Cboe_Collar_Indices_Methodology.pdf"),
        ("Cboe downside benchmark overview", "https://www.cboe.com/insights/posts/benchmark-indices-series-hedging-downside-exposure-with-pput-cll-and-cllz-indices"),
        ("Innovator defined-outcome education", "https://www.innovatoretfs.com/education/"),
        ("AQR — A Century of Evidence on Trend-Following", "https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing"),
        ("NBER — Volatility Managed Portfolios", "https://www.nber.org/papers/w22208"),
        ("Faber tactical asset-allocation paper", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461"),
        ("Antonacci dual-momentum paper", "https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2042750"),
        ("WisdomTree NTSX official page", "https://www.wisdomtree.com/us/products/capital-efficient/ntsx"),
        ("KraneShares KMLM official page", "https://kraneshares.com/etf/kmlm/"),
        ("Amplify SWAN official page", "https://amplifyetfs.com/swan/"),
        ("Cambria TAIL official page", "https://www.cambriafunds.com/tail"),
        ("Invesco PHDG official page", "https://www.invesco.com/us/en/financial-products/etfs/invesco-sp-500-downside-hedged-etf.html"),
        ("Invesco SPLV official page", "https://www.invesco.com/us/en/financial-products/etfs/invesco-sp-500-low-volatility-etf.html"),
        ("iShares MTUM official page", "https://www.ishares.com/us/products/251614/"),
    ]
    sources_html = "".join(
        f'<li><a href="{escape(url)}">{escape(label)}</a></li>' for label, url in source_links
    )
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>ETF strategy 50+ pre-2022 study</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#17212b;line-height:1.48}}
h1,h2,h3{{color:#123b5d}} .lead{{font-size:1.1rem}} .warn{{background:#fff3cd;border-left:5px solid #d39e00;padding:12px 16px}}
.good{{background:#e8f5e9;border-left:5px solid #2e7d32;padding:12px 16px}} .bad{{background:#fdecea;border-left:5px solid #b71c1c;padding:12px 16px}}
.data{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 24px}} .data th,.data td{{border:1px solid #d7dde3;padding:6px 8px;text-align:right}}
.data th:first-child,.data td:first-child{{text-align:left;position:sticky;left:0;background:white}} .data thead th{{background:#eef3f7}}
.chart{{width:100%;max-width:1300px;margin:10px 0 28px}} code{{background:#f1f3f5;padding:2px 5px}} .small{{font-size:12px;color:#53616f}}
</style></head><body>
<h1>ETF-only 10% return / &lt;15% volatility / drawdown-control study</h1>
<p class="lead"><b>Data boundary:</b> 2004 warm-up, core comparison 2007-01-03 to 2021-12-31; every download stopped at 2022-01-01 exclusive.</p>
<div class="warn"><b>Interpretation:</b> historical CAGR is not a forward expected return. Short live ETF records and synthetic option marks are kept in separate evidence tiers. A buffer ETF is not principal-guaranteed, and a modeled collar is not an executable-chain backtest.</div>

<h2>1 · Executive result</h2>
<p>The frozen batch ran <b>{total_count}</b> rows, including <b>{long_count}</b> long-history ETF-rule portfolios. <b>{pass_count}</b> long-history rows passed every frozen 9%+ return / 15% vol / -25% drawdown / cost / subperiod / concentration gate. The table below deliberately takes one representative per mechanism cluster; rows marked near-target or research-only are alternatives, not silent gate relaxations.</p>
<p><b>Duration convention:</b> <code>max_underwater_days</code> counts the full peak-to-recovery episode. <code>max_5pct_trough_to_recovery_days</code> counts only the trough-to-prior-high leg for episodes that reached at least -5%. Open episodes are censored at 2021-12-31, so their duration is only a lower bound. The 20-business-day share is an ex-post diagnostic and never enters a trading signal.</p>
{_format_table(selected_view, pct)}
<img class="chart" src="{charts['scatter']}" alt="return volatility scatter">

<h2>2 · Ten-plus relatively independent alternatives</h2>
<p>Independence is enforced first by economic mechanism and portfolio construction, then checked with monthly return correlation and stress/regime fingerprints. Parameter neighbours remain counted as trials but do not earn extra candidate slots.</p>
<img class="chart" src="{charts['equity']}" alt="selected equity curves">
<img class="chart" src="{charts['drawdown']}" alt="selected drawdown paths">
<img class="chart" src="{charts['correlation']}" alt="selected correlation matrix">

<h3>Six-month block-bootstrap uncertainty</h3>
<p>These are resampling diagnostics, not forecasts. Short-history funds with fewer than 36 monthly observations are intentionally blank.</p>
{_format_table(bootstrap_view, pct)}

<h2>3 · Historical stress and regime behavior</h2>
<img class="chart" src="{charts['stress']}" alt="historical stress heatmap">
{_format_table(stress.pivot(index='strategy', columns='event', values='event_return'), set(stress['event']))}
<h3>Prior-known trend × volatility regimes</h3>
<p>Each return day is labeled using the previous session's SPY 200DMA state and realized-volatility state. Values are conditional annualized returns, so they measure state sensitivity rather than calendar portfolio CAGR.</p>
{_format_table(regime_pivot, pct)}

<h2>4 · Core rule leaderboard</h2>
<p>The display is capped at 35 rows for readability; the CSV contains every trial. Deflated Sharpe uses all long-rule and synthetic-option trials, including losing parameter neighbours.</p>
{_format_table(top_core, pct)}

<h2>5 · Packaged ETF reality check</h2>
<p>These are actual fund-price histories through 2021, net of each fund's embedded expenses. The evidence column and start date matter more than rank: several products existed for only two or three years before the cutoff and never saw the GFC.</p>
{_format_table(product_view, pct)}

<h2>6 · What was tested</h2>
<ul>
<li>Quarterly fixed US/global stock, Treasury, gold, commodity and sector allocations.</li>
<li>Weekly SPY/QQQ moving-average, 63-day volatility target, drawdown ladder and fast-exit/slow-reentry P&amp;L controls.</li>
<li>Monthly fixed/inverse-vol/capped/trend-gated all-weather allocations.</li>
<li>Monthly GTAA, GEM, top-N cross-asset momentum, multi-horizon momentum and defensive momentum.</li>
<li>Monthly sector momentum, downside-deviation, barbell, downside-vol budget and Ulcer rules.</li>
<li>One frozen HYG/IEF credit-canary regime rule.</li>
<li>Seven synthetic SPY put/put-spread/collar overlays with conservative option-entry haircuts and financing.</li>
<li>Twenty-three packaged allocation, smart-beta, managed-futures, tail-risk, buffer, risk-parity and capital-efficient ETFs.</li>
</ul>

<h2>7 · Limits that matter</h2>
<ul>
<li>The common sample contains only one global financial crisis and one pandemic crash. Regime counts are small.</li>
<li>Today's ETF universe has survivorship and product-launch bias. The nine-sector universe is frozen, but delisted strategy ETFs are not represented.</li>
<li>Inverse volatility is a simple risk-balance proxy, not a full Bridgewater portfolio and not a forecast of growth/inflation surprises.</li>
<li>The SPY option module uses VIX and Black-Scholes with skew and spread haircuts. It lacks historical NBBO, exact strikes, American exercise, discrete dividends, tax and crisis liquidity.</li>
<li>Managed-futures ETFs are useful diversifiers, but their live pre-cutoff records are too short. Long index backfills are not treated as ETF performance.</li>
<li>Defined-outcome buffers apply point-to-point over an outcome period, have upside caps, can lose beyond the buffer, and are not guarantees.</li>
<li>No 2022 stock/bond inflation shock was used. A forward paper period is still required before allocation.</li>
</ul>

<h2>8 · Sources used to form testable hypotheses</h2><ul>{sources_html}</ul>
<p class="small">Research only, not investment advice. Reproducible runner: <code>scripts/etf_strategy_50plus_study.py</code>. Frozen protocol: <code>{escape(str(PROTOCOL.relative_to(PROJECT_ROOT)))}</code>.</p>
</body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adjusted, raw_spy = _download_prices()
    runs = _add_rule_strategies(adjusted)
    runs.extend(_add_option_runs(adjusted, raw_spy))
    runs.extend(_add_product_runs(adjusted))
    if len(runs) < 79:
        raise AssertionError(f"frozen batch expected at least 79 rows, got {len(runs)}")
    if any(run.returns.index.max() > LAST_ALLOWED for run in runs):
        raise AssertionError("a strategy return stream contains post-2021 data")

    metrics = _build_metrics(runs, adjusted)
    selected = _select_independent(metrics)
    selected_names = selected.index.tolist()
    returns = _selected_returns(selected_names, runs)
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    correlation = monthly.corr(min_periods=24)
    regime = _regime_table(selected_names, runs, adjusted)
    bootstrap = _bootstrap_table(selected_names, runs)
    stress_rows = []
    for name in selected_names:
        for event in STRESS_WINDOWS:
            stress_rows.append(
                {
                    "strategy": name,
                    "event": event,
                    "event_return": metrics.loc[name, f"{event}_return"],
                    "event_maxdd": metrics.loc[name, f"{event}_maxdd"],
                }
            )
    stress = pd.DataFrame(stress_rows)
    charts = _build_charts(metrics, selected, returns, stress, correlation)
    report = _render_report(metrics, selected, bootstrap, regime, stress, correlation, charts)

    adjusted.to_parquet(OUT / "market_prices_adjusted_pre2022.parquet")
    raw_spy.to_frame().to_parquet(OUT / "raw_spy_pre2022.parquet")
    pd.concat([run.returns.rename(run.name) for run in runs], axis=1, sort=False).to_parquet(
        OUT / "all_strategy_returns.parquet"
    )
    metrics.to_csv(OUT / "all_strategy_metrics.csv")
    selected.to_csv(OUT / "selected_independent_candidates.csv")
    returns.to_parquet(OUT / "selected_returns.parquet")
    correlation.to_csv(OUT / "selected_monthly_correlation.csv")
    regime.to_csv(OUT / "selected_regimes.csv", index=False)
    stress.to_csv(OUT / "selected_stress_windows.csv", index=False)
    bootstrap.to_csv(OUT / "selected_bootstrap.csv")
    metadata = {
        "generated_at": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "download_start": DOWNLOAD_START,
        "download_end_exclusive": DOWNLOAD_END,
        "last_observation": str(adjusted.index.max().date()),
        "core_start": str(CORE_START.date()),
        "core_end": str(CORE_END.date()),
        "strategy_count": len(runs),
        "long_rule_count": int((metrics["evidence"] == "long_rule").sum()),
        "target_gate_pass_count": int(metrics["target_gate"].sum()),
        "selected": selected_names,
        "primary_trading_bps": PRIMARY_TRADING_BPS,
        "stress_trading_bps": STRESS_TRADING_BPS,
    }
    (OUT / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("\nSelected candidates:\n")
    print(
        selected[
            [
                "family",
                "evidence",
                "selection_status",
                "cagr",
                "stress_cagr",
                "annual_vol",
                "max_drawdown",
                "cagr_2013_2021",
                "target_gate",
            ]
        ].to_string()
    )
    print(f"\nReport: {REPORT}")
    print(f"Artifacts: {OUT}")


if __name__ == "__main__":
    main()
