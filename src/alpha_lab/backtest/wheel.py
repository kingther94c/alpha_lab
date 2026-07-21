"""Synthetic cash-secured-put / covered-call wheel backtest.

The engine is intentionally marked-to-model.  It uses an implied-volatility
proxy rather than pretending that free ETF prices contain a historical option
chain.  Entry decisions use the prior close, fills occur at the current close,
idle cash earns the supplied cash return, and short options are marked daily.

This is a research benchmark, not an execution engine.  A live or
execution-grade study should replace model prices and deltas with timestamped
option-chain quotes and explicitly model early assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, exp, sqrt

import pandas as pd
from scipy.stats import norm

from alpha_lab.backtest.collar import black_scholes_call
from alpha_lab.backtest.put_write import black_scholes_put, strike_for_target_put_delta


@dataclass(frozen=True)
class WheelConfig:
    """Parameters for repeated cash-secured puts followed by covered calls."""

    put_target_delta: float = 0.20
    call_target_delta: float = 0.25
    tenor_trading_days: int = 21
    collateral_fraction: float = 1.0
    put_iv_multiplier: float = 1.0
    call_iv_multiplier: float = 1.0
    entry_spread_fraction: float = 0.05
    commission_per_contract: float = 0.65
    strike_increment: float = 1.0
    contract_multiplier: int = 100
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.put_target_delta < 0.5:
            raise ValueError("put_target_delta must be between 0 and 0.5")
        if not 0.0 < self.call_target_delta < 0.5:
            raise ValueError("call_target_delta must be between 0 and 0.5")
        if self.tenor_trading_days < 1:
            raise ValueError("tenor_trading_days must be positive")
        if not 0.0 < self.collateral_fraction <= 1.0:
            raise ValueError("collateral_fraction must be in (0, 1]")
        if self.put_iv_multiplier <= 0.0 or self.call_iv_multiplier <= 0.0:
            raise ValueError("IV multipliers must be positive")
        if not 0.0 <= self.entry_spread_fraction < 1.0:
            raise ValueError("entry_spread_fraction must be in [0, 1)")
        if self.commission_per_contract < 0.0:
            raise ValueError("commission_per_contract cannot be negative")
        if self.strike_increment <= 0.0:
            raise ValueError("strike_increment must be positive")
        if self.contract_multiplier < 1:
            raise ValueError("contract_multiplier must be positive")


@dataclass
class WheelBacktestResult:
    """Daily marked-to-model outputs for a wheel strategy."""

    returns: pd.Series
    equity: pd.Series
    cash: pd.Series
    option_value: pd.Series
    stock_value: pd.Series
    state: pd.Series
    delta_exposure: pd.Series
    events: pd.DataFrame
    config: WheelConfig


def strike_for_target_call_delta(
    spot: float,
    target_delta: float,
    years_to_expiry: float,
    annual_rate: float,
    annual_vol: float,
    *,
    dividend_yield: float = 0.0,
    strike_increment: float = 1.0,
) -> float:
    """Infer and round up a strike for an OTM call delta target."""
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if not 0.0 < target_delta < 0.5:
        raise ValueError("target_delta must be between 0 and 0.5")
    if years_to_expiry <= 0.0 or annual_vol <= 0.0:
        raise ValueError("expiry and volatility must be positive")
    if strike_increment <= 0.0:
        raise ValueError("strike_increment must be positive")

    probability = target_delta * exp(dividend_yield * years_to_expiry)
    if not 0.0 < probability < 1.0:
        raise ValueError("delta and dividend yield imply an invalid probability")
    d1 = norm.ppf(probability)
    exponent = (
        annual_rate - dividend_yield + 0.5 * annual_vol**2
    ) * years_to_expiry - d1 * annual_vol * sqrt(years_to_expiry)
    raw_strike = spot * exp(exponent)
    return ceil(raw_strike / strike_increment) * strike_increment


def run_wheel_backtest(
    prices: pd.Series,
    implied_vol: pd.Series,
    cash_returns: pd.Series,
    annual_rates: pd.Series,
    *,
    config: WheelConfig = WheelConfig(),  # noqa: B008
    initial_equity: float = 1_000_000.0,
) -> WheelBacktestResult:
    """Run a synthetic QQQ-style wheel with prior-close entry decisions.

    A flat account sells a cash-secured put.  An in-the-money expiry is
    physically assigned into stock; the next eligible close sells a covered
    call on all assigned shares.  An in-the-money call expiry sells the stock
    at the strike and returns the strategy to cash.  Expiry transitions never
    open a replacement option on the same bar.

    The price series should be a split-adjusted total-return proxy when the
    researcher wants dividends reflected in the stock leg.  Because the same
    series is also used for option settlement, the result remains synthetic.
    """
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")

    frame = pd.concat(
        {
            "spot": pd.to_numeric(prices, errors="coerce"),
            "iv": pd.to_numeric(implied_vol, errors="coerce"),
            "cash_return": pd.to_numeric(cash_returns, errors="coerce"),
            "annual_rate": pd.to_numeric(annual_rates, errors="coerce"),
        },
        axis=1,
    ).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame[["iv", "annual_rate", "cash_return"]] = frame[
        ["iv", "annual_rate", "cash_return"]
    ].ffill()
    frame = frame.dropna(subset=["spot", "iv", "annual_rate", "cash_return"])
    if len(frame) < 2:
        raise ValueError("at least two aligned observations are required")
    if (frame["spot"] <= 0.0).any() or (frame["iv"] <= 0.0).any():
        raise ValueError("prices and implied volatility must be positive")

    cash = float(initial_equity)
    stock_shares = 0.0
    option: dict[str, float | int | str | pd.Timestamp] | None = None

    equity_values: list[float] = []
    cash_values: list[float] = []
    option_values: list[float] = []
    stock_values: list[float] = []
    state_values: list[str] = []
    delta_values: list[float] = []
    events: list[dict[str, object]] = []

    index = frame.index
    for position, (date, row) in enumerate(frame.iterrows()):
        spot = float(row["spot"])
        annual_rate = float(row["annual_rate"])
        put_vol = float(row["iv"]) * config.put_iv_multiplier
        call_vol = float(row["iv"]) * config.call_iv_multiplier
        transitioned = False

        if position > 0:
            cash *= 1.0 + float(row["cash_return"])

        if option is not None and position >= int(option["expiry_position"]):
            option_type = str(option["type"])
            strike = float(option["strike"])
            shares = float(option["shares"])
            if option_type == "put":
                if spot < strike:
                    cash -= strike * shares
                    stock_shares = shares
                    event = "put_assignment"
                else:
                    event = "put_expiry_otm"
            else:
                if spot > strike:
                    cash += strike * shares
                    stock_shares = max(stock_shares - shares, 0.0)
                    event = "call_assignment"
                else:
                    event = "call_expiry_otm"
            events.append(
                _event_row(
                    date,
                    event,
                    spot,
                    strike,
                    shares,
                    cash + stock_shares * spot,
                )
            )
            option = None
            transitioned = True

        can_enter = (
            option is None
            and not transitioned
            and position >= 1
            and position + config.tenor_trading_days < len(frame)
        )
        if can_enter:
            signal_spot = float(frame["spot"].iloc[position - 1])
            signal_iv_raw = float(frame["iv"].iloc[position - 1])
            signal_rate = float(frame["annual_rate"].iloc[position - 1])
            years = config.tenor_trading_days / 252.0

            if stock_shares == 0.0:
                signal_vol = signal_iv_raw * config.put_iv_multiplier
                strike = strike_for_target_put_delta(
                    signal_spot,
                    config.put_target_delta,
                    years,
                    signal_rate,
                    signal_vol,
                    dividend_yield=config.dividend_yield,
                    strike_increment=config.strike_increment,
                )
                mid, fill_delta = black_scholes_put(
                    spot,
                    strike,
                    years,
                    annual_rate,
                    put_vol,
                    dividend_yield=config.dividend_yield,
                )
                fill = max(
                    mid * (1.0 - config.entry_spread_fraction),
                    max(strike - spot, 0.0),
                )
                shares = config.collateral_fraction * cash / strike
                option_type = "put"
                event = "put_entry"
            else:
                signal_vol = signal_iv_raw * config.call_iv_multiplier
                strike = strike_for_target_call_delta(
                    signal_spot,
                    config.call_target_delta,
                    years,
                    signal_rate,
                    signal_vol,
                    dividend_yield=config.dividend_yield,
                    strike_increment=config.strike_increment,
                )
                mid, fill_delta = black_scholes_call(
                    spot,
                    strike,
                    years,
                    annual_rate,
                    call_vol,
                    dividend_yield=config.dividend_yield,
                )
                fill = max(mid * (1.0 - config.entry_spread_fraction), 0.0)
                shares = stock_shares
                option_type = "call"
                event = "call_entry"

            commission = (
                shares / config.contract_multiplier * config.commission_per_contract
            )
            cash += fill * shares - commission
            option = {
                "type": option_type,
                "strike": strike,
                "shares": shares,
                "expiry_position": position + config.tenor_trading_days,
            }
            events.append(
                _event_row(
                    date,
                    event,
                    spot,
                    strike,
                    shares,
                    cash + stock_shares * spot - mid * shares,
                    model_mid=mid,
                    fill=fill,
                    fill_delta=fill_delta,
                    signal_spot=signal_spot,
                    signal_iv=signal_vol,
                    commission=commission,
                    expiry=index[position + config.tenor_trading_days],
                )
            )

        option_value = 0.0
        option_delta = 0.0
        if option is not None:
            remaining = max(int(option["expiry_position"]) - position, 0)
            if str(option["type"]) == "put":
                option_mid, option_delta = black_scholes_put(
                    spot,
                    float(option["strike"]),
                    remaining / 252.0,
                    annual_rate,
                    put_vol,
                    dividend_yield=config.dividend_yield,
                )
                state = "short_put"
            else:
                option_mid, option_delta = black_scholes_call(
                    spot,
                    float(option["strike"]),
                    remaining / 252.0,
                    annual_rate,
                    call_vol,
                    dividend_yield=config.dividend_yield,
                )
                state = "covered_call"
            option_value = option_mid * float(option["shares"])
            short_option_delta_shares = -option_delta * float(option["shares"])
        else:
            state = "stock" if stock_shares > 0.0 else "cash"
            short_option_delta_shares = 0.0

        stock_value = stock_shares * spot
        equity = cash + stock_value - option_value
        net_delta_notional = (stock_shares + short_option_delta_shares) * spot
        delta_exposure = net_delta_notional / equity if equity != 0.0 else float("nan")

        equity_values.append(equity)
        cash_values.append(cash)
        option_values.append(option_value)
        stock_values.append(stock_value)
        state_values.append(state)
        delta_values.append(delta_exposure)

    equity_series = pd.Series(equity_values, index=index, name="equity")
    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame = event_frame.set_index("date").sort_index()
    return WheelBacktestResult(
        returns=equity_series.pct_change().fillna(0.0).rename("returns"),
        equity=equity_series,
        cash=pd.Series(cash_values, index=index, name="cash"),
        option_value=pd.Series(option_values, index=index, name="option_value"),
        stock_value=pd.Series(stock_values, index=index, name="stock_value"),
        state=pd.Series(state_values, index=index, name="state"),
        delta_exposure=pd.Series(delta_values, index=index, name="delta_exposure"),
        events=event_frame,
        config=config,
    )


def config_dict(config: WheelConfig) -> dict[str, float | int]:
    """Return a serialization-friendly configuration dictionary."""
    return asdict(config)


def _event_row(
    date: pd.Timestamp,
    event: str,
    spot: float,
    strike: float,
    shares: float,
    equity: float,
    **details: object,
) -> dict[str, object]:
    return {
        "date": date,
        "event": event,
        "spot": spot,
        "strike": strike,
        "shares": shares,
        "equity": equity,
        **details,
    }
