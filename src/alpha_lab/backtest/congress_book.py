"""Congressional-flow strategy book — the single source of truth for weights.

The same code computes target weights for (a) the research backtest, (b) robustness /
holdout evaluation, and (c) future paper/live execution, so they can never drift apart
(same pattern as :mod:`alpha_lab.backtest.crypto_book`).

What it trades
--------------
Only liquid **sector ETFs / index ETFs** — never the individual stocks that members
disclose. The single-name PTR flow is aggregated to GICS sectors and expressed through
the 11 SPDR sector ETFs (Angle A) and a growth-vs-small macro tilt (Angle C).

Discipline
----------
- Point-in-time: signals key off ``filing_date`` (see :mod:`alpha_lab.backtest.congress_signal`).
- Backtest eval starts only once **all** sector ETFs exist (XLC's 2018 inception is the
  binding constraint), so we never "trade" an ETF before it listed.
- Returns are reported **excess of cash** (3M T-bill financing on deployed capital), per
  the repo's research-artifact contract — commissions + slippage alone understate cost.
- Benchmarks the strategy must beat (net, risk-adjusted): SPY buy-hold, the NANC/KRUZ
  congress-copy ETFs, and a sector equal-weight book.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alpha_lab.backtest.congress_signal import (
    aggregate_net_flow,
    risk_on_weights,
    sector_flow_zscore,
    sector_net_flow,
    sector_tilt_weights,
)
from alpha_lab.backtest.metrics import summary
from alpha_lab.backtest.vector import run_backtest
from alpha_lab.data.congress_universe import load_ticker_sector_map, sector_etf_map
from alpha_lab.data.loaders.congress import load_congress_trades

BARS = 252
BENCH = ["SPY", "NANC", "KRUZ"]          # buy-hold + the two congress-copy ETFs
MACRO = ["QQQ", "IWM"]                    # Angle C growth-vs-small legs
# Conservative one-way costs for liquid sector ETFs (bps).
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 3.0
# 3M T-bill fallback (annualized) when FRED is unreachable — cost of cash.
RF_FALLBACK = {y: r for y, r in {
    2014: 0.0003, 2015: 0.0005, 2016: 0.003, 2017: 0.009, 2018: 0.019, 2019: 0.021,
    2020: 0.004, 2021: 0.0004, 2022: 0.020, 2023: 0.0521, 2024: 0.0505, 2025: 0.0425,
    2026: 0.043}.items()}


@dataclass
class CongressBookData:
    """Everything the book needs, aligned to one trading-day grid."""

    trades: pd.DataFrame             # normalized single-stock congress PTRs
    sector_of: pd.Series             # ticker -> GICS sector
    sector_prices: pd.DataFrame      # 11 SPDR sector ETF closes
    macro_prices: pd.DataFrame       # QQQ / IWM closes
    bench_prices: pd.DataFrame       # SPY / NANC / KRUZ closes
    rf_daily: pd.Series              # per-day risk-free (cost of cash)
    eval_start: pd.Timestamp         # first day all sector ETFs exist
    rf_source: str
    coverage: dict = field(default_factory=dict)

    @property
    def trading_index(self) -> pd.DatetimeIndex:
        return self.sector_prices.index


def _rf_daily(index: pd.DatetimeIndex, start: str, end: str | None) -> tuple[pd.Series, str]:
    """Daily risk-free series on ``index`` from FRED DTB3, with a piecewise fallback."""
    try:
        from alpha_lab.data.loaders.fred import discount_rate_to_daily_rate, load_series
        raw = load_series("DTB3", start=start, end=end, timeout=20)["DTB3"].dropna()
        daily = discount_rate_to_daily_rate(raw)
        daily.index = pd.DatetimeIndex(daily.index).normalize()
        return daily.reindex(index.normalize()).ffill().fillna(0.0).set_axis(index), "FRED DTB3"
    except Exception:  # noqa: BLE001 — offline / FRED down → fallback
        vals = [RF_FALLBACK.get(d.year, 0.04) / 365.0 for d in index]
        return pd.Series(vals, index=index, dtype="float64"), "fallback piecewise"


def load_congress_book_data(
    start: str = "2014-01-01",
    end: str | None = None,
    *,
    refresh: bool = False,
    use_yfinance_sectors: bool = False,
) -> CongressBookData:
    """Load PTR trades, the ticker→sector map, sector/macro/benchmark ETF prices, and rf.

    ``use_yfinance_sectors=False`` (default) relies on the curated CSV + cache only, so
    the build is deterministic and offline-safe; set True to also fill the tail live.
    """
    from alpha_lab.data.loaders.yfinance import load_prices

    trades = load_congress_trades(start=start, end=end, asset_types=("ST",),
                                  chambers=("house", "senate"), refresh=refresh)
    sector_of = load_ticker_sector_map(trades["ticker"].dropna().unique(),
                                       use_yfinance=use_yfinance_sectors)

    sector_etfs = list(sector_etf_map().values())
    sector_prices = load_prices(sector_etfs, start=start, end=end)
    macro_prices = load_prices(MACRO, start=start, end=end)
    bench_prices = load_prices(BENCH, start=start, end=end)

    # Eval starts once every sector ETF has a price (XLC 2018 is the binding constraint).
    firsts = [sector_prices[c].first_valid_index() for c in sector_prices.columns]
    eval_start = max(d for d in firsts if d is not None)

    rf_daily, rf_source = _rf_daily(sector_prices.index, start, end)

    from alpha_lab.data.congress_universe import coverage_report
    cov = coverage_report(trades.assign(gics=trades["ticker"].map(sector_of)), sector_of)

    return CongressBookData(
        trades=trades, sector_of=sector_of, sector_prices=sector_prices,
        macro_prices=macro_prices, bench_prices=bench_prices, rf_daily=rf_daily,
        eval_start=eval_start, rf_source=rf_source, coverage=cov,
    )


# --------------------------------------------------------------------------------------
# Weights (the strategy)
# --------------------------------------------------------------------------------------
def sector_tilt(
    bd: CongressBookData,
    *,
    window: int = 63,
    z_window: int = 252,
    top_n: int = 3,
    bottom_n: int = 3,
    market_neutral: bool = True,
) -> pd.DataFrame:
    """Angle-A target weights (sector-ETF columns) on the book's trading grid.

    Long top-N / short bottom-N sectors by rolling net-flow z-score. Dollar-neutral
    (``market_neutral=True``) strips market beta — the plan's key to beating NANC.
    """
    net = sector_net_flow(bd.trades, bd.sector_of, bd.trading_index, window=window)
    z = sector_flow_zscore(net, z_window=z_window)
    short_gross = 1.0 if market_neutral else 0.0
    w = sector_tilt_weights(z, top_n=top_n, bottom_n=bottom_n,
                            long_gross=1.0, short_gross=short_gross)
    return w.reindex(columns=bd.sector_prices.columns).fillna(0.0)


def risk_on_tilt(bd: CongressBookData, *, window: int = 63, z_window: int = 252) -> pd.DataFrame:
    """Angle-C target weights (QQQ/IWM) from aggregate congressional flow."""
    agg = aggregate_net_flow(bd.trades, bd.trading_index, window=window)
    return risk_on_weights(agg, z_window=z_window).reindex(
        columns=bd.macro_prices.columns).fillna(0.0)


# --------------------------------------------------------------------------------------
# Backtest + cost of cash
# --------------------------------------------------------------------------------------
def _excess(returns: pd.Series, weights: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """Subtract financing (rf × long-gross exposure) — the cost-of-cash hurdle."""
    long_gross = weights.clip(lower=0).sum(axis=1)
    return (returns - rf.reindex(returns.index).fillna(0.0) * long_gross).rename("excess")


def backtest_weights(
    bd: CongressBookData,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    rebalance: str = "W-FRI",
    costs_bps: float = COMMISSION_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
):
    """Run a weight panel through the engine on the eval window; return (result, excess)."""
    w = weights.loc[bd.eval_start:].reindex(columns=prices.columns).fillna(0.0)
    px = prices.loc[bd.eval_start:]
    res = run_backtest(signals=w, prices=px, rebalance=rebalance,
                       costs_bps=costs_bps, slippage_bps=slippage_bps, bars_per_year=BARS)
    return res, _excess(res.returns, res.weights, bd.rf_daily)


# --------------------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------------------
def benchmark_returns(bd: CongressBookData) -> pd.DataFrame:
    """The three reference lines, daily simple returns on the eval window.

    - ``SPY`` buy-hold; ``NANC`` / ``KRUZ`` congress-copy ETFs (exist 2023+);
    - ``SectorEW`` equal-weight of the 11 SPDR sector ETFs (long-only, monthly reb).
    """
    out = {}
    for c in BENCH:
        if c in bd.bench_prices.columns:
            out[c] = bd.bench_prices[c].loc[bd.eval_start:].pct_change()
    ew_w = pd.DataFrame(1.0 / len(bd.sector_prices.columns),
                        index=bd.sector_prices.index, columns=bd.sector_prices.columns)
    ew_res, _ = backtest_weights(bd, ew_w, bd.sector_prices, rebalance="ME")
    out["SectorEW"] = ew_res.returns
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Study runner (building blocks for the report) + execution handoff
# --------------------------------------------------------------------------------------
def run_study(bd: CongressBookData, **tilt_kwargs) -> dict:
    """Run the core Angle-A book + benchmarks; return everything the report needs."""
    w = sector_tilt(bd, **tilt_kwargs)
    res, excess = backtest_weights(bd, w, bd.sector_prices)
    bench = benchmark_returns(bd)

    strat_perf = summary(excess, periods=BARS)
    bench_perf = {c: summary(bench[c].dropna(), periods=BARS) for c in bench.columns}
    return {
        "weights": w,
        "result": res,
        "net_returns": res.returns,
        "excess_returns": excess,
        "benchmarks": bench,
        "strategy_summary": strat_perf,
        "benchmark_summary": bench_perf,
        "eval_start": bd.eval_start,
        "rf_source": bd.rf_source,
        "coverage": bd.coverage,
    }


def latest_target_weights(bd: CongressBookData, **tilt_kwargs) -> pd.Series:
    """Sector-ETF target weights for the most recent bar — what execution would hold.

    Research-leg handoff: a future ``quant_bot_manager`` strategy adapts this to live
    data. Positive = long the sector ETF, negative = short. NOT wired to any live bot.
    """
    w = sector_tilt(bd, **tilt_kwargs)
    return w.iloc[-1].rename("target_weight")


# --------------------------------------------------------------------------------------
# Angle B — committee information advantage (SCAFFOLD; see decision record)
# --------------------------------------------------------------------------------------
def committee_weighted_flow(*args, **kwargs):  # noqa: D401
    """Scaffold for Angle B (committee overlap weighting). NOT implemented.

    The mechanism: up-weight a member's trade when their committee has regulatory /
    appropriations authority over the traded stock's sector (e.g. Armed Services →
    Defense). This needs **point-in-time committee rosters** (Senate.gov / House /
    GovTrack), which are not yet wired into the repo. Implementing it is the top
    "next step" in the decision record; until then Angle B is analyzed qualitatively.
    """
    raise NotImplementedError(
        "Angle B needs point-in-time committee rosters (GovTrack/Senate.gov) — not wired. "
        "See docs/research_decisions/2026-06-19_congressional_trading_signal.md."
    )
