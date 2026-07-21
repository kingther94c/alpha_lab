"""Return math. Works on Series or DataFrame of prices/returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.Series | pd.DataFrame


def simple_returns(prices: Frame) -> Frame:
    """Arithmetic returns: p_t / p_{t-1} - 1."""
    return prices.pct_change()


def log_returns(prices: Frame) -> Frame:
    """Continuously-compounded returns: ln(p_t / p_{t-1})."""
    return np.log(prices / prices.shift(1))


def cumulative_returns(returns: Frame) -> Frame:
    """Compound simple returns into a wealth index starting at 1."""
    return (1 + returns.fillna(0)).cumprod()


def drawdown(returns: Frame) -> Frame:
    """Drawdown series from simple returns (peak-to-trough, as a negative fraction)."""
    wealth = cumulative_returns(returns)
    # Initial capital is a real high-water mark.  Without the floor, a sample
    # that starts with a loss incorrectly reports zero drawdown until it first
    # makes a new within-sample high.
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def drawdown_duration_metrics(
    returns: pd.Series,
    *,
    material_threshold: float = 0.05,
    recovery_target_days: int = 20,
) -> dict[str, float | int]:
    """Summarize drawdown durations in observed sessions.

    ``max_underwater_days`` spans the full peak-to-recovery episode, while
    ``max_material_recovery_days`` measures only the trough-to-prior-high leg
    for episodes reaching ``material_threshold``. An episode still open at the
    end of the sample is censored there, so its recovery duration is a lower
    bound and it does not count as recovered within the target.
    """
    if not 0.0 < material_threshold < 1.0:
        raise ValueError("material_threshold must be between zero and one")
    if recovery_target_days < 0:
        raise ValueError("recovery_target_days must be non-negative")
    values = returns.dropna()
    empty: dict[str, float | int] = {
        "max_underwater_days": 0,
        "max_trough_to_recovery_days": 0,
        "max_material_recovery_days": 0,
        "median_material_recovery_days": np.nan,
        "share_material_recovered_within_target": np.nan,
        "material_drawdown_count": 0,
        "unrecovered_material_drawdown_count": 0,
    }
    if values.empty:
        return empty

    dd = drawdown(values)
    underwater = dd < -1e-12
    episodes: list[dict[str, float | int | bool]] = []
    start: int | None = None
    array = dd.to_numpy()
    for position, is_underwater in enumerate(underwater.to_numpy()):
        if is_underwater and start is None:
            start = position
        if start is not None and (not is_underwater or position == len(dd) - 1):
            stop = position if not is_underwater else position + 1
            window = array[start:stop]
            trough_offset = int(np.argmin(window))
            trough = start + trough_offset
            completed = not is_underwater
            recovery_endpoint = position if completed else len(dd) - 1
            episodes.append(
                {
                    "depth": float(window[trough_offset]),
                    "underwater_days": int(stop - start),
                    "recovery_days": int(recovery_endpoint - trough),
                    "completed": completed,
                }
            )
            start = None

    if not episodes:
        return empty

    material = [
        episode
        for episode in episodes
        if float(episode["depth"]) <= -material_threshold
    ]
    material_recovery = [int(episode["recovery_days"]) for episode in material]
    recovered_within_target = [
        bool(episode["completed"])
        and int(episode["recovery_days"]) <= recovery_target_days
        for episode in material
    ]
    return {
        "max_underwater_days": max(
            int(episode["underwater_days"]) for episode in episodes
        ),
        "max_trough_to_recovery_days": max(
            int(episode["recovery_days"]) for episode in episodes
        ),
        "max_material_recovery_days": max(material_recovery, default=0),
        "median_material_recovery_days": (
            float(np.median(material_recovery)) if material_recovery else np.nan
        ),
        "share_material_recovered_within_target": (
            float(np.mean(recovered_within_target))
            if recovered_within_target
            else np.nan
        ),
        "material_drawdown_count": len(material),
        "unrecovered_material_drawdown_count": sum(
            not bool(episode["completed"]) for episode in material
        ),
    }


def annualized_vol(returns: Frame, periods: int = 252) -> Frame | float:
    """sqrt(periods) * std. Default assumes daily returns."""
    return returns.std() * np.sqrt(periods)


def sharpe(returns: Frame, rf: float = 0.0, periods: int = 252) -> Frame | float:
    """Naive Sharpe: (mean(excess) / std(excess)) * sqrt(periods).

    ``rf`` is a per-period risk-free rate (same freq as ``returns``).
    """
    excess = returns - rf
    mu = excess.mean()
    sd = excess.std()
    return (mu / sd) * np.sqrt(periods)
