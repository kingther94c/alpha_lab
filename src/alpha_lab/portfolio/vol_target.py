"""Volatility-targeting overlay.

Scales a stream of target weights so the resulting portfolio runs at a
specified ex-ante volatility, estimated from trailing realized vol on the
*same* (unscaled) weighted return stream.

Kept deliberately simple: one-period lag on the vol estimate to avoid
look-ahead, optional leverage cap, no transaction-cost model (the calling
backtest engine handles costs on turnover).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha_lab.analytics.risk import geometric_portfolio_vol


class VolTargetInfeasible(RuntimeError):
    """Raised when no long-only fully-invested portfolio satisfies the vol band."""


@dataclass(frozen=True)
class VolMatchResult:
    """A fully-invested portfolio and its benchmark-relative geometric vol."""

    weights: pd.Series
    vol_ratio: float
    portfolio_vol: float
    benchmark_vol: float


@dataclass(frozen=True)
class RollingVolMatchResult:
    """Rolling matched weights and decision-date volatility diagnostics."""

    weights: pd.DataFrame
    diagnostics: pd.DataFrame


def vol_target_scalar(
    weighted_returns: pd.Series,
    *,
    target_vol: float = 0.10,
    lookback_days: int = 63,
    periods: int = 252,
    max_leverage: float = 1.5,
    min_leverage: float = 0.0,
) -> pd.Series:
    """Return a daily leverage scalar so portfolio vol targets ``target_vol``.

    Parameters
    ----------
    weighted_returns : daily return stream of the *unscaled* target portfolio
        (e.g. ``(weights * asset_returns).sum(axis=1)``).
    target_vol : annualized vol target (e.g. ``0.10`` for 10%).
    lookback_days : trailing window for realized vol.
    periods : periods per year used to annualize (252 for daily).
    max_leverage, min_leverage : clipping bounds applied to the raw scalar.

    Returns
    -------
    Series of the same index. Values at the start of the sample, before
    ``lookback_days`` of history exist, are NaN; callers typically ffill or
    drop these.
    """
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")

    rolling_vol = weighted_returns.rolling(lookback_days, min_periods=lookback_days).std() * np.sqrt(periods)
    scalar = (target_vol / rolling_vol).shift(1)
    return scalar.clip(lower=min_leverage, upper=max_leverage)


def apply_vol_target(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    lookback_days: int = 63,
    periods: int = 252,
    max_leverage: float = 1.5,
    min_leverage: float = 0.0,
) -> pd.DataFrame:
    """Scale a wide weights frame so the resulting portfolio targets ``target_vol``.

    The scalar is built from the trailing realized vol of ``(weights * asset_returns).sum(1)``
    and shifted by one day to avoid look-ahead, then broadcast across columns.
    """
    weighted = (weights * asset_returns).sum(axis=1)
    scalar = vol_target_scalar(
        weighted,
        target_vol=target_vol,
        lookback_days=lookback_days,
        periods=periods,
        max_leverage=max_leverage,
        min_leverage=min_leverage,
    )
    return weights.mul(scalar, axis=0)


def match_benchmark_vol_weights(
    raw_weights: pd.Series,
    cov_short: pd.DataFrame,
    cov_long: pd.DataFrame,
    *,
    benchmark_vol_short: float,
    benchmark_vol_long: float,
    band: tuple[float, float] = (0.9, 1.1),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    max_weight: float = 1.0,
    tolerance: float = 1e-7,
) -> VolMatchResult:
    """Find the closest long-only simplex portfolio inside a benchmark vol band.

    ``cov_short`` and ``cov_long`` must be annualized covariance matrices with
    identical labels. The optimizer never rescales or renormalizes its output
    after solving, so the volatility constraint and sum-to-one constraint are
    checked on the exact returned weights.
    """
    lower, upper = band
    if not 0 < lower <= upper:
        raise ValueError("band must satisfy 0 < lower <= upper")
    if benchmark_vol_short <= 0 or benchmark_vol_long <= 0:
        raise ValueError("benchmark volatilities must be positive")
    if turnover_penalty < 0:
        raise ValueError("turnover_penalty must be >= 0")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if list(cov_short.index) != list(cov_short.columns):
        raise ValueError("cov_short labels must match on rows and columns")
    if list(cov_long.index) != list(cov_long.columns):
        raise ValueError("cov_long labels must match on rows and columns")
    if not cov_short.index.equals(cov_long.index):
        raise ValueError("covariance matrices must have identical labels")

    labels = cov_short.index
    n_assets = len(labels)
    if n_assets == 0 or n_assets * max_weight < 1.0 - tolerance:
        raise VolTargetInfeasible("position cap cannot form a fully-invested portfolio")

    raw = raw_weights.reindex(labels).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float, copy=True)
    if raw.sum() <= 0:
        raise ValueError("raw_weights must contain positive exposure")
    raw /= raw.sum()
    if raw.max() > max_weight + tolerance:
        raw = _project_capped_simplex(raw, max_weight)

    previous = raw.copy()
    if previous_weights is not None:
        previous = previous_weights.reindex(labels).fillna(0.0).clip(lower=0.0).to_numpy(
            dtype=float,
            copy=True,
        )
        if previous.sum() > 0:
            previous /= previous.sum()
            previous = _project_capped_simplex(previous, max_weight)
        else:
            previous = raw.copy()

    benchmark_vol = float(np.sqrt(benchmark_vol_short * benchmark_vol_long))

    def ratio(weights: np.ndarray) -> float:
        return geometric_portfolio_vol(weights, cov_short, cov_long) / benchmark_vol

    raw_ratio = ratio(raw)
    raw_is_feasible = lower - tolerance <= raw_ratio <= upper + tolerance
    if raw_is_feasible and (previous_weights is None or turnover_penalty == 0.0):
        weights = pd.Series(raw, index=labels, name=raw_weights.name)
        return VolMatchResult(weights, raw_ratio, raw_ratio * benchmark_vol, benchmark_vol)

    def objective(weights: np.ndarray) -> float:
        distance = float(np.square(weights - raw).sum())
        turnover = float(np.square(weights - previous).sum())
        return distance + turnover_penalty * turnover

    constraints = (
        {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        {"type": "ineq", "fun": lambda weights: ratio(weights) - lower},
        {"type": "ineq", "fun": lambda weights: upper - ratio(weights)},
    )
    all_starts = _optimizer_starts(raw, previous, n_assets, max_weight)

    def start_priority(weights: np.ndarray) -> tuple[float, float]:
        start_ratio = ratio(weights)
        distance_to_band = max(lower - start_ratio, start_ratio - upper, 0.0)
        return distance_to_band, objective(weights)

    starts = sorted(all_starts, key=start_priority)[:3]
    feasible_results: list[tuple[float, np.ndarray, float]] = []
    observed_ratios = []
    for start in starts:
        observed_ratios.append(ratio(start))
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n_assets,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        candidate = np.asarray(result.x, dtype=float)
        candidate_ratio = ratio(candidate)
        observed_ratios.append(candidate_ratio)
        if (
            result.success
            and abs(candidate.sum() - 1.0) <= tolerance
            and candidate.min() >= -tolerance
            and candidate.max() <= max_weight + tolerance
            and lower - tolerance <= candidate_ratio <= upper + tolerance
        ):
            feasible_results.append((objective(candidate), candidate, candidate_ratio))

    if not feasible_results:
        raise VolTargetInfeasible(
            "no feasible fully-invested portfolio found; "
            f"observed vol-ratio range {min(observed_ratios):.4f} to {max(observed_ratios):.4f}"
        )

    _, solution, solution_ratio = min(feasible_results, key=lambda item: item[0])
    solution[np.abs(solution) < 1e-12] = 0.0
    weights = pd.Series(solution, index=labels, name=raw_weights.name)
    portfolio_vol = geometric_portfolio_vol(weights, cov_short, cov_long)
    return VolMatchResult(weights, solution_ratio, portfolio_vol, benchmark_vol)


def rolling_match_benchmark_vol_weights(
    raw_targets: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    short_window: int = 21,
    long_window: int = 63,
    periods: int = 252,
    band: tuple[float, float] = (0.9, 1.1),
    turnover_penalty: float = 0.0,
    max_weight: float = 1.0,
) -> RollingVolMatchResult:
    """Apply ``match_benchmark_vol_weights`` at sparse decision dates."""
    if short_window < 2 or long_window < short_window:
        raise ValueError("windows must satisfy 2 <= short_window <= long_window")
    if not raw_targets.index.isin(asset_returns.index).all():
        raise ValueError("every target date must be present in asset_returns")

    rows = []
    diagnostics = []
    previous = None
    labels = raw_targets.columns
    for date, raw in raw_targets.iterrows():
        window = asset_returns.loc[:date, labels].tail(long_window).dropna(how="any")
        benchmark_window = benchmark_returns.reindex(window.index).dropna()
        if len(window) < long_window or len(benchmark_window) < long_window:
            raise ValueError(f"insufficient trailing returns at {date}")

        cov_short = window.tail(short_window).cov() * periods
        cov_long = window.cov() * periods
        benchmark_vol_short = float(benchmark_window.tail(short_window).std() * np.sqrt(periods))
        benchmark_vol_long = float(benchmark_window.std() * np.sqrt(periods))
        matched = match_benchmark_vol_weights(
            raw,
            cov_short,
            cov_long,
            benchmark_vol_short=benchmark_vol_short,
            benchmark_vol_long=benchmark_vol_long,
            band=band,
            previous_weights=previous,
            turnover_penalty=turnover_penalty,
            max_weight=max_weight,
        )
        weights = matched.weights.rename(date)
        rows.append(weights)
        diagnostics.append(
            pd.Series(
                {
                    "vol_ratio": matched.vol_ratio,
                    "portfolio_vol": matched.portfolio_vol,
                    "benchmark_vol": matched.benchmark_vol,
                    "weight_sum": matched.weights.sum(),
                    "max_weight": matched.weights.max(),
                },
                name=date,
            )
        )
        previous = matched.weights

    return RollingVolMatchResult(
        weights=pd.DataFrame(rows).reindex(columns=labels),
        diagnostics=pd.DataFrame(diagnostics),
    )


def _optimizer_starts(
    raw: np.ndarray,
    previous: np.ndarray,
    n_assets: int,
    max_weight: float,
) -> list[np.ndarray]:
    starts = [raw, previous, np.full(n_assets, 1.0 / n_assets)]
    for index in range(n_assets):
        concentrated = np.zeros(n_assets)
        order = [index, *[i for i in range(n_assets) if i != index]]
        remaining = 1.0
        for position in order:
            allocation = min(max_weight, remaining)
            concentrated[position] = allocation
            remaining -= allocation
            if remaining <= 1e-12:
                break
        starts.append(concentrated)
    unique = []
    for start in starts:
        projected = _project_capped_simplex(start, max_weight)
        if not any(np.allclose(projected, existing) for existing in unique):
            unique.append(projected)
    return unique


def _project_capped_simplex(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Euclidean projection onto non-negative weights summing to one with a cap."""
    values = np.asarray(weights, dtype=float)
    lower = float(values.min() - max_weight)
    upper = float(values.max())
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        projected = np.clip(values - midpoint, 0.0, max_weight)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - upper, 0.0, max_weight)
    return projected / projected.sum()
