"""Trading calendars and rebalance schedules.

Kept lightweight: pandas business-day logic covers the common cases for
ETF / FX research. Swap in ``pandas_market_calendars`` later if exchange-exact
holidays matter.
"""

from __future__ import annotations

import pandas as pd


def trading_days(start: str, end: str | None = None) -> pd.DatetimeIndex:
    """Business-day DatetimeIndex between *start* and *end* inclusive.

    US federal holidays are excluded via pandas' ``USFederalHolidayCalendar``.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.date_range(start=start, end=end, freq=cbd)


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Pick one rebalance date per period from *index*.

    Parameters
    ----------
    index : a DatetimeIndex of available trading days.
    freq : pandas offset alias. Common picks: ``"D"`` (daily),
        ``"W-FRI"`` (weekly), ``"ME"`` (month end), ``"QE"`` (quarter end),
        ``"YE"`` (year end). For weekly and above, the **last** trading day
        of each period in *index* is returned (i.e. rebalance at period end
        using info through that day).

    Returns
    -------
    DatetimeIndex — a subset of *index*.
    """
    if len(index) == 0:
        return index
    if freq == "D":
        return index
    s = pd.Series(index, index=index)
    last_in_period = s.resample(freq).last().dropna()
    return pd.DatetimeIndex(last_in_period.values)
