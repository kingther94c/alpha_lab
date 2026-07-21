"""Cash-secured put research helpers.

The historical engine intentionally accepts an implied-volatility proxy rather
than pretending that free daily ETF prices contain an option chain.  It marks
the short put daily with Black-Scholes, invests idle cash at a supplied cash
return, and optionally converts an in-the-money expiry into an underlying
position that is held until recovery or a time limit.

This is a research/advisory model, not an order router.  Production use should
replace model prices and deltas with a live option chain and broker margin data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, floor, log, sqrt
from typing import Literal

import pandas as pd
from scipy.stats import norm

RecoveryRule = Literal["breakeven", "strike"]
StockExitMode = Literal[
    "price_or_timeout",
    "timeout_only",
    "iv_or_timeout",
    "price_or_iv_or_timeout",
    "price_and_iv_or_timeout",
]


@dataclass(frozen=True)
class PutWriteConfig:
    """Parameters for a cash-secured put cycle."""

    target_delta: float = 0.15
    tenor_trading_days: int = 21
    collateral_fraction: float = 0.50
    max_assignment_days: int = 63
    recovery_rule: RecoveryRule = "breakeven"
    iv_multiplier: float = 1.00
    entry_spread_fraction: float = 0.05
    commission_per_contract: float = 0.65
    stock_exit_cost_bps: float = 2.0
    strike_increment: float = 1.0
    contract_multiplier: int = 100
    trend_lookback: int | None = None
    dividend_yield: float = 0.0
    entry_min_iv: float | None = None
    entry_min_iv_percentile: float | None = None
    entry_iv_lookback: int = 252
    entry_min_iv_rv_spread: float | None = None
    realized_vol_lookback: int = 21
    stock_exit_mode: StockExitMode = "price_or_timeout"
    stock_target_return: float = 0.0
    stock_exit_iv_max: float | None = None
    min_assignment_days: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_delta < 0.5:
            raise ValueError("target_delta must be between 0 and 0.5")
        if self.tenor_trading_days < 1:
            raise ValueError("tenor_trading_days must be positive")
        if not 0.0 < self.collateral_fraction <= 1.0:
            raise ValueError("collateral_fraction must be in (0, 1]")
        if self.max_assignment_days < 0:
            raise ValueError("max_assignment_days cannot be negative")
        if self.iv_multiplier <= 0.0:
            raise ValueError("iv_multiplier must be positive")
        if not 0.0 <= self.entry_spread_fraction < 1.0:
            raise ValueError("entry_spread_fraction must be in [0, 1)")
        if self.strike_increment <= 0.0:
            raise ValueError("strike_increment must be positive")
        if self.contract_multiplier < 1:
            raise ValueError("contract_multiplier must be positive")
        if self.trend_lookback is not None and self.trend_lookback < 2:
            raise ValueError("trend_lookback must be at least 2")
        if self.entry_min_iv is not None and self.entry_min_iv <= 0.0:
            raise ValueError("entry_min_iv must be positive")
        if (
            self.entry_min_iv_percentile is not None
            and not 0.0 < self.entry_min_iv_percentile < 1.0
        ):
            raise ValueError("entry_min_iv_percentile must be in (0, 1)")
        if self.entry_iv_lookback < 2:
            raise ValueError("entry_iv_lookback must be at least 2")
        if self.realized_vol_lookback < 2:
            raise ValueError("realized_vol_lookback must be at least 2")
        if self.stock_exit_mode not in {
            "price_or_timeout",
            "timeout_only",
            "iv_or_timeout",
            "price_or_iv_or_timeout",
            "price_and_iv_or_timeout",
        }:
            raise ValueError("unsupported stock_exit_mode")
        if self.stock_target_return <= -1.0:
            raise ValueError("stock_target_return must be greater than -1")
        if self.stock_exit_iv_max is not None and self.stock_exit_iv_max <= 0.0:
            raise ValueError("stock_exit_iv_max must be positive")
        if self.min_assignment_days < 0:
            raise ValueError("min_assignment_days cannot be negative")
        if (
            self.max_assignment_days > 0
            and self.min_assignment_days > self.max_assignment_days
        ):
            raise ValueError("min_assignment_days cannot exceed max_assignment_days")
        if (
            self.stock_exit_mode
            in {
                "iv_or_timeout",
                "price_or_iv_or_timeout",
                "price_and_iv_or_timeout",
            }
            and self.stock_exit_iv_max is None
        ):
            raise ValueError("stock_exit_iv_max is required for an IV-based stock exit")


@dataclass
class PutWriteBacktestResult:
    """Daily marked-to-model outputs for a put-write strategy."""

    returns: pd.Series
    equity: pd.Series
    cash: pd.Series
    option_value: pd.Series
    stock_value: pd.Series
    state: pd.Series
    events: pd.DataFrame
    config: PutWriteConfig


@dataclass(frozen=True)
class PutWriteAdvice:
    """A contract-sized advisory snapshot for one proposed put sale."""

    spot: float
    annual_iv: float
    annual_cash_yield: float
    target_delta: float
    tenor_calendar_days: int
    strike: float
    model_mid: float
    assumed_fill: float
    contracts: int
    collateral: float
    premium: float
    breakeven: float
    downside_to_strike: float
    premium_yield_on_collateral: float
    annualized_premium_yield: float
    price_source: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a serialization-friendly dictionary."""
        return asdict(self)


def black_scholes_put(
    spot: float,
    strike: float,
    years_to_expiry: float,
    annual_rate: float,
    annual_vol: float,
    *,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return European put value and spot delta.

    Delta is in ``[-1, 0]``.  At expiry, the function returns intrinsic value
    and a simple left/right derivative convention.
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if annual_vol <= 0.0:
        raise ValueError("annual_vol must be positive")
    if years_to_expiry <= 0.0:
        intrinsic = max(strike - spot, 0.0)
        if spot < strike:
            delta = -1.0
        elif spot > strike:
            delta = 0.0
        else:
            delta = -0.5
        return intrinsic, delta

    sigma_sqrt_t = annual_vol * sqrt(years_to_expiry)
    d1 = (
        log(spot / strike)
        + (annual_rate - dividend_yield + 0.5 * annual_vol**2) * years_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discounted_strike = strike * exp(-annual_rate * years_to_expiry)
    discounted_spot = spot * exp(-dividend_yield * years_to_expiry)
    price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    delta = -exp(-dividend_yield * years_to_expiry) * norm.cdf(-d1)
    return float(price), float(delta)


def strike_for_target_put_delta(
    spot: float,
    target_delta: float,
    years_to_expiry: float,
    annual_rate: float,
    annual_vol: float,
    *,
    dividend_yield: float = 0.0,
    strike_increment: float = 1.0,
) -> float:
    """Infer and round down a strike for an absolute put delta target."""
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if not 0.0 < target_delta < 0.5:
        raise ValueError("target_delta must be between 0 and 0.5")
    if years_to_expiry <= 0.0 or annual_vol <= 0.0:
        raise ValueError("expiry and volatility must be positive")
    if strike_increment <= 0.0:
        raise ValueError("strike_increment must be positive")

    d1 = -norm.ppf(target_delta)
    exponent = (
        annual_rate
        - dividend_yield
        + 0.5 * annual_vol**2
    ) * years_to_expiry - d1 * annual_vol * sqrt(years_to_expiry)
    raw_strike = spot * exp(exponent)
    return floor(raw_strike / strike_increment) * strike_increment


def build_put_write_advice(
    *,
    spot: float,
    annual_iv: float,
    annual_cash_yield: float,
    account_nav: float,
    config: PutWriteConfig,
    quoted_bid: float | None = None,
    tenor_calendar_days: int | None = None,
) -> PutWriteAdvice:
    """Size one cash-secured put proposal from account NAV.

    ``quoted_bid`` should be supplied for a live advisory.  When it is absent,
    the helper uses a spread-haircut Black-Scholes proxy and labels the source
    accordingly.
    """
    if account_nav <= 0.0:
        raise ValueError("account_nav must be positive")
    if annual_iv <= 0.0:
        raise ValueError("annual_iv must be positive")
    calendar_days = tenor_calendar_days or round(config.tenor_trading_days * 365.25 / 252)
    years = calendar_days / 365.25
    model_vol = annual_iv * config.iv_multiplier
    strike = strike_for_target_put_delta(
        spot,
        config.target_delta,
        years,
        annual_cash_yield,
        model_vol,
        dividend_yield=config.dividend_yield,
        strike_increment=config.strike_increment,
    )
    model_mid, _ = black_scholes_put(
        spot,
        strike,
        years,
        annual_cash_yield,
        model_vol,
        dividend_yield=config.dividend_yield,
    )
    if quoted_bid is None:
        fill = model_mid * (1.0 - config.entry_spread_fraction)
        source = "Black-Scholes proxy"
    else:
        if quoted_bid < 0.0:
            raise ValueError("quoted_bid cannot be negative")
        fill = quoted_bid
        source = "quoted bid"

    per_contract_collateral = strike * config.contract_multiplier
    budget = account_nav * config.collateral_fraction
    contracts = floor(budget / per_contract_collateral)
    collateral = contracts * per_contract_collateral
    commission = contracts * config.commission_per_contract
    premium = contracts * config.contract_multiplier * fill - commission
    breakeven = strike - (
        fill - config.commission_per_contract / config.contract_multiplier
    )
    premium_yield = premium / collateral if collateral else 0.0
    annualized_yield = premium_yield * 365.25 / calendar_days

    return PutWriteAdvice(
        spot=float(spot),
        annual_iv=float(annual_iv),
        annual_cash_yield=float(annual_cash_yield),
        target_delta=config.target_delta,
        tenor_calendar_days=calendar_days,
        strike=float(strike),
        model_mid=float(model_mid),
        assumed_fill=float(fill),
        contracts=contracts,
        collateral=float(collateral),
        premium=float(premium),
        breakeven=float(breakeven),
        downside_to_strike=float(strike / spot - 1.0),
        premium_yield_on_collateral=float(premium_yield),
        annualized_premium_yield=float(annualized_yield),
        price_source=source,
    )


def run_cash_secured_put_backtest(
    prices: pd.Series,
    implied_vol: pd.Series,
    cash_returns: pd.Series,
    annual_rates: pd.Series,
    *,
    config: PutWriteConfig = PutWriteConfig(),  # noqa: B008
    initial_equity: float = 1.0,
) -> PutWriteBacktestResult:
    """Backtest repeated cash-secured puts with optional assignment holding.

    Entry decisions use only the previous close.  The put is entered at the
    current close, receives no same-bar return, and is then marked daily.
    A trend filter, when configured, also uses the previous close and a moving
    average ending at that previous close.

    ``max_assignment_days=0`` cash-settles an in-the-money expiry.  Positive
    values model QQQ-style physical assignment followed by a configurable
    price/IV/time exit. Entry filters and price/IV exit signals use the
    previous close; their trades occur at the current close.
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

    trend = None
    if config.trend_lookback is not None:
        trend = frame["spot"].rolling(config.trend_lookback).mean()
    iv_threshold = None
    if config.entry_min_iv_percentile is not None:
        iv_threshold = frame["iv"].rolling(
            config.entry_iv_lookback,
            min_periods=config.entry_iv_lookback,
        ).quantile(config.entry_min_iv_percentile)
    realized_vol = None
    if config.entry_min_iv_rv_spread is not None:
        realized_vol = (
            frame["spot"]
            .pct_change()
            .rolling(
                config.realized_vol_lookback,
                min_periods=config.realized_vol_lookback,
            )
            .std(ddof=1)
            * sqrt(252.0)
        )

    cash = float(initial_equity)
    option: dict[str, float | int | pd.Timestamp] | None = None
    stock_shares = 0.0
    stock_days = 0
    recovery_price = float("nan")

    equity_values: list[float] = []
    cash_values: list[float] = []
    option_values: list[float] = []
    stock_values: list[float] = []
    states: list[str] = []
    events: list[dict[str, object]] = []

    index = frame.index
    for position, (date, row) in enumerate(frame.iterrows()):
        spot = float(row["spot"])
        annual_rate = float(row["annual_rate"])
        model_vol = float(row["iv"]) * config.iv_multiplier
        transitioned = False

        if position > 0:
            cash *= 1.0 + float(row["cash_return"])

        if option is not None:
            expiry_position = int(option["expiry_position"])
            shares = float(option["shares"])
            strike = float(option["strike"])
            if position >= expiry_position:
                if spot < strike:
                    if config.max_assignment_days == 0:
                        payoff = (strike - spot) * shares
                        cash -= payoff
                        events.append(
                            _event_row(
                                date,
                                "cash_settlement",
                                spot,
                                strike,
                                shares,
                                cash,
                                payoff=payoff,
                            )
                        )
                    else:
                        cash -= strike * shares
                        stock_shares = shares
                        stock_days = 0
                        recovery_price = float(option["recovery_price"]) * (
                            1.0 + config.stock_target_return
                        )
                        events.append(
                            _event_row(
                                date,
                                "assignment",
                                spot,
                                strike,
                                shares,
                                cash + shares * spot,
                                recovery_price=recovery_price,
                            )
                        )
                else:
                    events.append(
                        _event_row(date, "expiry_otm", spot, strike, shares, cash)
                    )
                option = None
                transitioned = True

        elif stock_shares > 0.0:
            stock_days += 1
            signal_spot = float(frame["spot"].iloc[position - 1])
            signal_iv = float(frame["iv"].iloc[position - 1])
            target_signals_allowed = stock_days >= config.min_assignment_days
            price_hit = target_signals_allowed and signal_spot >= recovery_price
            iv_hit = (
                target_signals_allowed
                and config.stock_exit_iv_max is not None
                and signal_iv <= config.stock_exit_iv_max
            )
            timed_out = stock_days >= config.max_assignment_days
            if config.stock_exit_mode == "price_or_timeout":
                target_hit = price_hit
            elif config.stock_exit_mode == "timeout_only":
                target_hit = False
            elif config.stock_exit_mode == "iv_or_timeout":
                target_hit = iv_hit
            elif config.stock_exit_mode == "price_or_iv_or_timeout":
                target_hit = price_hit or iv_hit
            else:
                target_hit = price_hit and iv_hit
            if target_hit or timed_out:
                if timed_out:
                    reason = "timeout"
                elif config.stock_exit_mode == "iv_or_timeout":
                    reason = "iv_target"
                elif config.stock_exit_mode == "price_or_timeout":
                    reason = "price_target"
                elif price_hit and iv_hit:
                    reason = "price_and_iv"
                elif price_hit:
                    reason = "price_target"
                else:
                    reason = "iv_target"
                proceeds = stock_shares * spot
                exit_cost = proceeds * config.stock_exit_cost_bps / 10_000.0
                cash += proceeds - exit_cost
                events.append(
                    _event_row(
                        date,
                        "stock_exit",
                        spot,
                        recovery_price,
                        stock_shares,
                        cash,
                        reason=reason,
                        stock_days=stock_days,
                        exit_cost=exit_cost,
                        signal_spot=signal_spot,
                        signal_iv=signal_iv,
                    )
                )
                stock_shares = 0.0
                stock_days = 0
                recovery_price = float("nan")
                transitioned = True

        is_flat = option is None and stock_shares == 0.0
        enough_trend_history = (
            trend is None
            or (
                position >= 1
                and pd.notna(trend.iloc[position - 1])
                and frame["spot"].iloc[position - 1] >= trend.iloc[position - 1]
            )
        )
        signal_iv_raw = (
            float(frame["iv"].iloc[position - 1]) if position >= 1 else float("nan")
        )
        enough_iv_level = (
            config.entry_min_iv is None or signal_iv_raw >= config.entry_min_iv
        )
        enough_iv_percentile = (
            iv_threshold is None
            or (
                position >= 1
                and pd.notna(iv_threshold.iloc[position - 1])
                and signal_iv_raw >= float(iv_threshold.iloc[position - 1])
            )
        )
        enough_iv_rv_spread = (
            realized_vol is None
            or (
                position >= 1
                and pd.notna(realized_vol.iloc[position - 1])
                and signal_iv_raw - float(realized_vol.iloc[position - 1])
                >= config.entry_min_iv_rv_spread
            )
        )
        can_enter = (
            is_flat
            and not transitioned
            and position >= 1
            and enough_trend_history
            and enough_iv_level
            and enough_iv_percentile
            and enough_iv_rv_spread
            and position + config.tenor_trading_days < len(frame)
        )
        if can_enter:
            signal_spot = float(frame["spot"].iloc[position - 1])
            signal_iv = float(frame["iv"].iloc[position - 1]) * config.iv_multiplier
            signal_rate = float(frame["annual_rate"].iloc[position - 1])
            years = config.tenor_trading_days / 252.0
            strike = strike_for_target_put_delta(
                signal_spot,
                config.target_delta,
                years,
                signal_rate,
                signal_iv,
                dividend_yield=config.dividend_yield,
                strike_increment=config.strike_increment,
            )
            mid, fill_delta = black_scholes_put(
                spot,
                strike,
                years,
                annual_rate,
                model_vol,
                dividend_yield=config.dividend_yield,
            )
            fill = max(
                mid * (1.0 - config.entry_spread_fraction),
                max(strike - spot, 0.0),
            )
            shares = config.collateral_fraction * cash / strike
            commission = (
                shares / config.contract_multiplier * config.commission_per_contract
            )
            cash += fill * shares - commission
            option = {
                "strike": strike,
                "shares": shares,
                "expiry_position": position + config.tenor_trading_days,
                "entry_fill": fill,
                "recovery_price": (
                    strike
                    if config.recovery_rule == "strike"
                    else strike - fill + commission / shares
                ),
            }
            events.append(
                _event_row(
                    date,
                    "put_entry",
                    spot,
                    strike,
                    shares,
                    cash - mid * shares,
                    model_mid=mid,
                    fill=fill,
                    fill_delta=fill_delta,
                    signal_spot=signal_spot,
                    signal_iv=signal_iv,
                    signal_iv_raw=signal_iv_raw,
                    signal_iv_threshold=(
                        float(iv_threshold.iloc[position - 1])
                        if iv_threshold is not None
                        and pd.notna(iv_threshold.iloc[position - 1])
                        else float("nan")
                    ),
                    signal_realized_vol=(
                        float(realized_vol.iloc[position - 1])
                        if realized_vol is not None
                        and pd.notna(realized_vol.iloc[position - 1])
                        else float("nan")
                    ),
                    commission=commission,
                    expiry=index[position + config.tenor_trading_days],
                )
            )

        if option is not None:
            remaining = max(int(option["expiry_position"]) - position, 0)
            option_mid, _ = black_scholes_put(
                spot,
                float(option["strike"]),
                remaining / 252.0,
                annual_rate,
                model_vol,
                dividend_yield=config.dividend_yield,
            )
            option_value = option_mid * float(option["shares"])
            stock_value = 0.0
            state = "short_put"
        elif stock_shares > 0.0:
            option_value = 0.0
            stock_value = stock_shares * spot
            state = "assigned_stock"
        else:
            option_value = 0.0
            stock_value = 0.0
            state = "cash"

        equity = cash + stock_value - option_value
        equity_values.append(equity)
        cash_values.append(cash)
        option_values.append(option_value)
        stock_values.append(stock_value)
        states.append(state)

    equity_series = pd.Series(equity_values, index=index, name="equity")
    returns = equity_series.pct_change().fillna(0.0).rename("returns")
    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame = event_frame.set_index("date").sort_index()

    return PutWriteBacktestResult(
        returns=returns,
        equity=equity_series,
        cash=pd.Series(cash_values, index=index, name="cash"),
        option_value=pd.Series(option_values, index=index, name="option_value"),
        stock_value=pd.Series(stock_values, index=index, name="stock_value"),
        state=pd.Series(states, index=index, name="state"),
        events=event_frame,
        config=config,
    )


def _event_row(
    date: pd.Timestamp,
    event: str,
    spot: float,
    strike: float,
    shares: float,
    equity: float,
    **details: object,
) -> dict[str, object]:
    """Build a consistent event record."""
    return {
        "date": date,
        "event": event,
        "spot": spot,
        "strike": strike,
        "shares": shares,
        "equity": equity,
        **details,
    }
