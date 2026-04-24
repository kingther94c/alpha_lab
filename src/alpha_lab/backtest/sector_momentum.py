"""Sector momentum signal construction and view expression helpers."""

import pandas as pd

UNIVERSE_COLUMNS = [
    "sector",
    "signal_etf",
    "long_1x_etf",
    "long_2x_etf",
    "long_3x_etf",
    "inverse_1x_etf",
    "inverse_2x_etf",
    "inverse_3x_etf",
]


def sector_momentum_signal(
    prices: pd.DataFrame,
    *,
    lookback_months: int = 12,
    skip_months: int = 1,
) -> pd.DataFrame:
    """Compute cross-sectional sector momentum from original sector ETF prices."""
    monthly = prices.resample("ME").last()
    return monthly.shift(skip_months) / monthly.shift(lookback_months) - 1


def top_bottom_view_weights(
    signal: pd.DataFrame,
    *,
    top_n: int = 3,
    bottom_n: int = 3,
    long_gross: float = 1.0,
    short_gross: float = 1.0,
) -> pd.DataFrame:
    """Convert row-wise scores into long-top and short-bottom ETF view weights."""
    if top_n < 1 or bottom_n < 1:
        raise ValueError("top_n and bottom_n must be >= 1")

    ranks_high = signal.rank(axis=1, ascending=False, method="first")
    ranks_low = signal.rank(axis=1, ascending=True, method="first")

    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    valid = signal.notna()
    weights = weights.mask((ranks_high <= top_n) & valid, long_gross / top_n)
    weights = weights.mask((ranks_low <= bottom_n) & valid, -short_gross / bottom_n)
    return weights.where(valid, 0.0)


def express_sector_views(
    view_weights: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    mode: str = "original_short",
    leverage: int = 1,
    preserve_exposure: bool = True,
) -> pd.DataFrame:
    """Map signed sector views to tradable ETF target weights.

    ``original_short`` uses the original signal ETF directly and allows negative
    target weights. ``leveraged_etf`` uses long leveraged ETFs for positive views
    and inverse leveraged ETFs for negative views, with positive target weights.
    """
    if leverage not in {1, 2, 3}:
        raise ValueError("leverage must be one of {1, 2, 3}")
    if mode not in {"original_short", "leveraged_etf"}:
        raise ValueError("mode must be 'original_short' or 'leveraged_etf'")

    _validate_universe(universe)
    lookup = universe.set_index("signal_etf")
    missing = sorted(set(view_weights.columns) - set(lookup.index))
    if missing:
        raise ValueError(f"view columns are missing from universe: {missing}")

    if mode == "original_short":
        trade_cols = lookup.loc[view_weights.columns, "long_1x_etf"].copy()
        trade_cols = trade_cols.where(trade_cols.astype(str) != "", trade_cols.index.to_series())
        return _rename_or_sum_columns(view_weights, trade_cols.to_dict())

    long_col = f"long_{leverage}x_etf"
    inverse_col = f"inverse_{leverage}x_etf"
    scale = leverage if preserve_exposure else 1.0
    out = pd.DataFrame(index=view_weights.index)

    for signal_etf in view_weights.columns:
        weights = view_weights[signal_etf]
        long_ticker = lookup.at[signal_etf, long_col]
        inverse_ticker = lookup.at[signal_etf, inverse_col]

        if (weights > 0).any():
            if pd.isna(long_ticker) or not str(long_ticker):
                raise ValueError(f"missing {long_col} for {signal_etf}")
            out[str(long_ticker)] = out.get(str(long_ticker), 0.0) + weights.clip(lower=0.0) / scale

        if (weights < 0).any():
            if pd.isna(inverse_ticker) or not str(inverse_ticker):
                raise ValueError(f"missing {inverse_col} for {signal_etf}")
            out[str(inverse_ticker)] = out.get(str(inverse_ticker), 0.0) + (-weights.clip(upper=0.0)) / scale

    return out.fillna(0.0)


def _validate_universe(universe: pd.DataFrame) -> None:
    missing_cols = sorted(set(UNIVERSE_COLUMNS) - set(universe.columns))
    if missing_cols:
        raise ValueError(f"universe is missing required columns: {missing_cols}")


def _rename_or_sum_columns(weights: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=weights.index)
    for source, target in column_map.items():
        out[str(target)] = out.get(str(target), 0.0) + weights[source]
    return out
