"""Crypto multi-strategy book (P7) — the single source of truth for sleeve weights.

The same code computes target weights for (a) the research backtest, (b) the 2026
holdout evaluation, and (c) live/paper execution, so the three can never drift apart.

Five sleeves, each a different return source (carry · trend · cross-sectional momentum ·
flow/forced-trade · macro-regime). All daily, leak-safe, reported **excess of cash**.
See `docs/research_decisions/crypto_intraday/P7-multi-strategy-book.md`.

Legs are named ``<COIN>.s`` (spot) and ``<COIN>.p`` (USD-M perp), e.g. ``BTC.s``, ``ETH.p``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.backtest.metrics import summary
from alpha_lab.backtest.vector import run_backtest

BARS = 365
BURN = pd.Timestamp("2022-04-01", tz="UTC")
# Conservative per-leg one-way slippage (bps); P6 stress spec.
SLIP = {"BTC.p": 8.0, "ETH.p": 10.0, "SOL.p": 12.0, "BNB.p": 12.0,
        "BTC.s": 15.0, "ETH.s": 17.5, "SOL.s": 20.0, "BNB.s": 20.0}
SYM = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB"}
PERP = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
SPOT = ["BTCUSDT", "ETHUSDT"]
RF_FALLBACK = {2022: 0.020, 2023: 0.0521, 2024: 0.0505, 2025: 0.0425, 2026: 0.043}

SLEEVE_SOURCE = {
    "S1_carry": "perp funding carry (market-neutral)",
    "S2_trend": "time-series momentum (long/short)",
    "S3_xsmom": "cross-sectional momentum (market-neutral)",
    "S4_fundcontra": "funding-extreme contrarian (flow / forced-trade)",
    "S5_macro": "macro credit-regime gate (beyond price-volume)",
}


def banded(z: pd.Series, enter: float, exit_: float) -> pd.Series:
    """Hysteresis position in {-1,0,+1} for a reversion signal (cuts turnover)."""
    raw = pd.Series(
        np.where(z >= enter, -1.0, np.where(z <= -enter, 1.0, np.where(z.abs() <= exit_, 0.0, np.nan))),
        index=z.index,
    )
    return raw.ffill().fillna(0.0)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@dataclass
class BookData:
    grid: pd.DatetimeIndex
    perp_close: pd.DataFrame   # cols BTC.p ETH.p SOL.p BNB.p
    spot_close: pd.DataFrame   # cols BTC.s ETH.s
    funding: pd.DataFrame      # 8h, cols <COIN>.p
    df_fund: pd.DataFrame      # daily-mean funding on grid
    hyg: pd.Series             # macro credit proxy on grid
    rf_daily: pd.Series        # per-day risk-free (cost of cash)
    prices: pd.DataFrame       # spot+perp close, all legs
    rf_source: str
    macro_source: str


def _daily_funding(funding: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    f = funding.copy()
    f.index = pd.DatetimeIndex(f.index).tz_convert("UTC")
    d = f.resample("1D").mean()
    d.index = pd.DatetimeIndex(d.index)
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d.reindex(grid).ffill()


def load_book_data(start: str, end: str, *, allow_holdout: bool = False,
                   holdout_config: str = "crypto_intraday") -> BookData:
    """Load and align everything the book needs. Set ``allow_holdout=True`` to
    deliberately release the PM-holdout lock (audited) for the final OOS eval / live."""
    from alpha_lab.data.loaders.binance_vision import load_klines, load_funding
    from alpha_lab.data.loaders.fred import discount_rate_to_daily_rate, load_series
    from alpha_lab.data.loaders.yfinance import load_prices
    from alpha_lab.data.holdout import PMHoldout

    ho = None
    if allow_holdout:
        base = PMHoldout.from_config(holdout_config)
        ho = PMHoldout(base.start, base.end, allow=True)

    perp = load_klines(PERP, "1d", start, end, market="perp", holdout=ho).close_panel().rename(
        columns={k: f"{v}.p" for k, v in SYM.items()})
    spot = load_klines(SPOT, "1d", start, end, market="spot", holdout=ho).close_panel().rename(
        columns={"BTCUSDT": "BTC.s", "ETHUSDT": "ETH.s"})
    funding = load_funding(PERP, start, end, holdout=ho).rename(columns={k: f"{SYM[k]}.p" for k in PERP})

    grid = perp.index.union(spot.index)
    perp, spot = perp.reindex(grid).ffill(), spot.reindex(grid).ffill()
    prices = pd.concat([spot, perp], axis=1)

    # cost of cash
    rf_src, rf_source = None, "fallback piecewise"
    try:
        rf_src = discount_rate_to_daily_rate(load_series("DTB3", start=start, end=end, timeout=25)["DTB3"].dropna())
        rf_source = "FRED DTB3"
    except Exception:  # noqa: BLE001
        pass
    naive = grid.tz_localize(None).normalize()
    if rf_src is not None:
        r = rf_src.copy(); r.index = pd.DatetimeIndex(r.index).normalize()
        rf_daily = r.reindex(naive).ffill().fillna(0.0)
    else:
        rf_daily = pd.Series([RF_FALLBACK.get(d.year, 0.04) / 365 for d in naive], index=naive)
    rf_daily.index = grid
    rf_daily = rf_daily.astype(float)

    # macro (beyond price-volume): HYG credit ETF
    hyg = load_prices("HYG", start, end)["HYG"]
    hyg.index = pd.DatetimeIndex(hyg.index).tz_localize("UTC")
    hyg = hyg.reindex(grid).ffill()

    return BookData(grid=grid, perp_close=perp, spot_close=spot, funding=funding,
                    df_fund=_daily_funding(funding, grid), hyg=hyg, rf_daily=rf_daily,
                    prices=prices, rf_source=rf_source, macro_source="yfinance HYG 50d trend")


# --------------------------------------------------------------------------------------
# Sleeve weight frames (the strategy itself)
# --------------------------------------------------------------------------------------
def sleeve_weights(bd: BookData) -> dict[str, tuple[pd.DataFrame, list[str], bool]]:
    """Return {name: (weight_frame, price_cols, use_funding)} for the five sleeves."""
    grid, perp, spot, df_fund, hyg = bd.grid, bd.perp_close, bd.spot_close, bd.df_fund, bd.hyg
    out: dict[str, tuple[pd.DataFrame, list[str], bool]] = {}

    # S1 carry — long spot / short perp when 7d funding > 0 (market-neutral)
    act = (df_fund.rolling(7).mean() > 0).astype(float)
    w1 = pd.DataFrame(0.0, index=grid, columns=["BTC.s", "ETH.s", "BTC.p", "ETH.p"])
    for c in ["BTC", "ETH"]:
        w1[f"{c}.s"], w1[f"{c}.p"] = 0.25 * act[f"{c}.p"], -0.25 * act[f"{c}.p"]
    out["S1_carry"] = (w1, ["BTC.s", "ETH.s", "BTC.p", "ETH.p"], True)

    # S2 trend — BTC/ETH perp long>50dMA / short<50dMA
    sma = perp[["BTC.p", "ETH.p"]].rolling(50).mean()
    w2 = 0.5 * (2 * (perp[["BTC.p", "ETH.p"]] > sma).astype(float) - 1)
    out["S2_trend"] = (w2, ["BTC.p", "ETH.p"], True)

    # S3 xsmom — long top2 / short bot2 of 4 perps by 30d return (market-neutral)
    ret30 = perp / perp.shift(30) - 1
    rk = ret30.rank(axis=1)
    w3 = (rk.sub(rk.mean(axis=1), axis=0)) / 3.0
    out["S3_xsmom"] = (w3, ["BTC.p", "ETH.p", "SOL.p", "BNB.p"], True)

    # S4 fundcontra — banded fade of funding z-extremes
    w4 = pd.DataFrame(0.0, index=grid, columns=["BTC.p", "ETH.p"])
    for c in ["BTC.p", "ETH.p"]:
        z = (df_fund[c] - df_fund[c].rolling(30).mean()) / df_fund[c].rolling(30).std()
        w4[c] = 0.5 * banded(z, 1.0, 0.3)
    out["S4_fundcontra"] = (w4, ["BTC.p", "ETH.p"], True)

    # S5 macro — hold crypto only when HYG credit regime risk-on
    ro = (hyg.shift(1) > hyg.shift(1).rolling(50).mean()).astype(float)
    w5 = pd.DataFrame({"BTC.s": 0.5 * ro, "ETH.s": 0.5 * ro}).reindex(grid)
    out["S5_macro"] = (w5, ["BTC.s", "ETH.s"], False)
    return out


# --------------------------------------------------------------------------------------
# Backtest + combine
# --------------------------------------------------------------------------------------
def _bt(bd: BookData, w: pd.DataFrame, cols: list[str], use_funding: bool):
    w = w.reindex(columns=cols).reindex(bd.grid).fillna(0.0)
    fund = bd.funding[[c for c in cols if c in bd.funding.columns]] if use_funding else None
    return run_backtest(signals=w, prices=bd.prices[cols], rebalance=None, costs_bps=0.0,
                        slippage_bps={c: SLIP[c] for c in cols}, funding=fund, bars_per_year=BARS)


def _excess(bd: BookData, res):
    L = res.weights.clip(lower=0).sum(axis=1)
    return (res.returns - bd.rf_daily.reindex(L.index).fillna(0) * L).rename("r")


def backtest_book(bd: BookData):
    """Run all five sleeves. Returns (R, results, diag) where R is the excess-return
    DataFrame, results the per-sleeve BacktestResult, diag a metadata dict."""
    sw = sleeve_weights(bd)
    R, results, diag = {}, {}, {}
    for name, (w, cols, uf) in sw.items():
        res = _bt(bd, w, cols, uf)
        results[name] = res
        R[name] = _excess(bd, res)
        diag[name] = {
            "source": SLEEVE_SOURCE[name],
            "gross_sharpe": summary(res.gross_returns.loc[BURN:], periods=BARS).get("Sharpe", np.nan),
            "ann_turnover": float(res.turnover.loc[BURN:].sum()) / max(len(res.turnover.loc[BURN:]) / BARS, 1e-9),
            "time_in_mkt": float((res.weights.abs().sum(axis=1).loc[BURN:] > 1e-9).mean()),
        }
    return pd.DataFrame(R), results, diag


def combine(R: pd.DataFrame, *, target_each: float = 0.08, lcap: float = 10.0, bars: int = BARS):
    """Equal-capital + risk-budget combinations. Returns (combos_df, lev_df)."""
    svol = R.rolling(60).std().shift(1) * np.sqrt(bars)
    lev = (target_each / svol).clip(upper=lcap).fillna(0.0)
    combo_eqcap = R.mean(axis=1).rename("combo_eqcap")
    combo_rb = (lev * R).mean(axis=1).rename("combo_riskbudget")
    rv = combo_rb.rolling(60).std().shift(1) * np.sqrt(bars)
    combo_vt = ((0.10 / rv).clip(upper=3.0).fillna(0.0) * combo_rb).rename("combo_vt10")
    combos = pd.concat([combo_eqcap, combo_rb, combo_vt], axis=1)
    return combos, lev


def latest_target_weights(bd: BookData, *, method: str = "equal_capital",
                          target_each: float = 0.08, lcap: float = 10.0) -> pd.Series:
    """Per-leg target weights for the most recent bar — what live execution should hold.

    ``method`` ∈ {``equal_capital`` (0.2 per sleeve), ``risk_budget`` (trailing-vol scaled,
    leverage-capped)}. Returns a Series indexed by leg (e.g. BTC.s, ETH.p), summing exposures
    across sleeves. Positive = long, negative = short; spot legs (.s) and perp legs (.p)
    map to Binance spot and USD-M futures respectively.
    """
    sw = sleeve_weights(bd)
    R, _, _ = backtest_book(bd)
    legs = sorted({c for (w, cols, _) in sw.values() for c in cols})
    if method == "risk_budget":
        _, lev = combine(R, target_each=target_each, lcap=lcap)
        scale = {name: float(lev[name].iloc[-1]) / len(sw) for name in sw}
    elif method == "equal_capital":
        scale = {name: 1.0 / len(sw) for name in sw}
    else:
        raise ValueError(f"method must be 'equal_capital' or 'risk_budget', got {method!r}")

    target = pd.Series(0.0, index=legs)
    for name, (w, cols, _) in sw.items():
        row = w.reindex(columns=cols).iloc[-1].fillna(0.0)
        for c in cols:
            target[c] += scale[name] * float(row[c])
    return target.rename("target_weight")
