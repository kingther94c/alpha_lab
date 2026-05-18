"""Volatility-targeting overlay.

Scales a stream of target weights so the resulting portfolio runs at a
specified ex-ante volatility, estimated from trailing realized vol on the
*same* (unscaled) weighted return stream.

Kept deliberately simple: one-period lag on the vol estimate to avoid
look-ahead, optional leverage cap, no transaction-cost model (the calling
backtest engine handles costs on turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
