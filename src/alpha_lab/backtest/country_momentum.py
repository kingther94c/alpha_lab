"""Country ETF cross-sectional momentum signal weights."""

import pandas as pd

from alpha_lab.data.calendars import rebalance_dates


def country_momentum_signal(
    prices: pd.DataFrame,
    *,
    lookback_months: int = 12,
    skip_months: int = 1,
) -> pd.DataFrame:
    """Compute 12-1 style monthly total-return momentum from ETF prices."""
    monthly = prices.resample("ME").last()
    return monthly.shift(skip_months) / monthly.shift(lookback_months) - 1


def country_momentum_weights(
    prices: pd.DataFrame,
    *,
    lookback_months: int = 12,
    skip_months: int = 1,
    top_n: int = 5,
    bottom_n: int = 5,
    rebalance: str = "W-WED",
    rank_change_threshold: int = 2,
    leg_gross: float = 0.5,
    weighting: str = "equal",
    vol_lookback_days: int = 60,
) -> pd.DataFrame:
    """Build dollar-neutral long/short country ETF momentum target weights.

    ``rank_change_threshold`` keeps the previous target for names whose rank
    moved less than the threshold, then renormalizes each leg to ``leg_gross``.
    This is a simple turnover buffer around the weekly signal refresh.
    """
    if top_n < 1 or bottom_n < 1:
        raise ValueError("top_n and bottom_n must be >= 1")
    if leg_gross <= 0:
        raise ValueError("leg_gross must be > 0")
    if rank_change_threshold < 0:
        raise ValueError("rank_change_threshold must be >= 0")
    if weighting not in {"equal", "inverse_vol"}:
        raise ValueError("weighting must be 'equal' or 'inverse_vol'")

    signal = country_momentum_signal(
        prices,
        lookback_months=lookback_months,
        skip_months=skip_months,
    )
    signal = signal.reindex(prices.index).ffill()
    ranks = signal.rank(axis=1, ascending=False, method="first")
    returns = prices.pct_change()

    rows = []
    dates = []
    previous_ranks: pd.Series | None = None
    previous_weights: pd.Series | None = None
    for date in rebalance_dates(prices.index, freq=rebalance):
        rank_row = ranks.loc[date].dropna()
        if len(rank_row) < top_n + bottom_n:
            continue

        target = _raw_top_bottom_weights(
            rank_row,
            returns.loc[:date].tail(vol_lookback_days),
            top_n=top_n,
            bottom_n=bottom_n,
            leg_gross=leg_gross,
            weighting=weighting,
        )

        if previous_ranks is not None and previous_weights is not None and rank_change_threshold:
            stable_names = rank_row.index.intersection(previous_ranks.index)
            stable_names = stable_names[(rank_row[stable_names] - previous_ranks[stable_names]).abs() < rank_change_threshold]
            target.loc[stable_names] = previous_weights.reindex(stable_names).fillna(0.0)
            target = _normalize_legs(target, leg_gross=leg_gross)

        rows.append(target.reindex(prices.columns, fill_value=0.0))
        dates.append(date)
        previous_ranks = rank_row
        previous_weights = target

    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).reindex(columns=prices.columns).fillna(0.0)


def _raw_top_bottom_weights(
    ranks: pd.Series,
    returns_window: pd.DataFrame,
    *,
    top_n: int,
    bottom_n: int,
    leg_gross: float,
    weighting: str,
) -> pd.Series:
    long_names = ranks.nsmallest(top_n).index
    short_names = ranks.nlargest(bottom_n).index
    weights = pd.Series(0.0, index=ranks.index)

    if weighting == "equal":
        weights.loc[long_names] = leg_gross / len(long_names)
        weights.loc[short_names] = -leg_gross / len(short_names)
        return weights

    vols = returns_window.reindex(columns=ranks.index).std().replace(0.0, pd.NA)
    long_scores = (1.0 / vols.reindex(long_names)).fillna(0.0)
    short_scores = (1.0 / vols.reindex(short_names)).fillna(0.0)
    if long_scores.sum() <= 0:
        long_scores = pd.Series(1.0, index=long_names)
    if short_scores.sum() <= 0:
        short_scores = pd.Series(1.0, index=short_names)
    weights.loc[long_names] = long_scores / long_scores.sum() * leg_gross
    weights.loc[short_names] = -(short_scores / short_scores.sum() * leg_gross)
    return weights.fillna(0.0)


def _normalize_legs(weights: pd.Series, *, leg_gross: float) -> pd.Series:
    out = weights.copy()
    long_sum = out.clip(lower=0.0).sum()
    short_sum = -out.clip(upper=0.0).sum()
    if long_sum > 0:
        out.loc[out > 0] *= leg_gross / long_sum
    if short_sum > 0:
        out.loc[out < 0] *= leg_gross / short_sum
    return out
