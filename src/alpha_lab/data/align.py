"""Alignment helpers: prices/returns on a shared calendar, forward-return math.

These are the small utilities that make signal-vs-forward-return evaluation
leak-safe and consistent across notebooks.
"""

from __future__ import annotations

import pandas as pd

Frame = pd.Series | pd.DataFrame


def align_prices(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    method: str = "ffill",
) -> pd.DataFrame:
    """Reindex *prices* onto *calendar* and forward-fill missing days.

    Useful when combining series with different native trading calendars
    (e.g. equity ETFs + FX).
    """
    aligned = prices.reindex(calendar)
    if method == "ffill":
        aligned = aligned.ffill()
    elif method == "none":
        pass
    else:
        raise ValueError(f"unknown method: {method!r}")
    return aligned


def forward_returns(returns: Frame, horizon: int = 1) -> Frame:
    """Forward-looking cumulative returns over the next *horizon* periods.

    ``forward_returns(r, 1)`` at date t is the return from t to t+1.
    For h > 1, compounds over the next h periods.

    The last h rows are NaN by construction.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon == 1:
        return returns.shift(-1)
    # Compound simple returns over a rolling forward window of length `horizon`.
    wealth = (1 + returns.fillna(0)).cumprod()
    fwd = wealth.shift(-horizon) / wealth - 1
    return fwd
