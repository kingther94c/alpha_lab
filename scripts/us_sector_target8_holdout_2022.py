"""Release and evaluate calendar-year 2022 for the frozen target-8 sector study."""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from us_sector_rotation_study import DATA_START, LEGACY_SECTORS
from us_sector_target8_study import (
    TARGET_FLOOR,
    TARGET_MAX_DRAWDOWN,
    TARGET_VOL_MAX,
    SectorCandidate,
    build_all_targets,
    build_benchmarks,
    build_candidate,
)

from alpha_lab.analytics.returns import drawdown
from alpha_lab.backtest.collar import SyntheticCollarResult, run_synthetic_collar
from alpha_lab.data.loaders.fred import cash_total_return_index, load_series
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.utils.paths import PROJECT_ROOT

HOLDOUT_START = pd.Timestamp("2022-01-01")
HOLDOUT_END = pd.Timestamp("2022-12-31")
DATA_END_EXCLUSIVE = "2023-01-01"
REPORT_PATH = PROJECT_ROOT / "reports" / "us_sector_target8_holdout_2022.html"
RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "us_sector_target8_holdout_2022_metrics.csv"
DEVELOPMENT_RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "us_sector_target8_metrics.csv"


def main() -> None:
    prices, raw_spy, vix, cash, annual_rates = load_released_inputs()
    legacy = prices[LEGACY_SECTORS].dropna()
    spy = prices["SPY"].reindex(legacy.index)
    raw_spy = raw_spy.reindex(legacy.index).ffill()
    vix = vix.reindex(legacy.index).ffill()
    cash = cash.reindex(legacy.index).fillna(0.0)
    annual_rates = annual_rates.reindex(legacy.index).ffill()

    collar = run_synthetic_collar(spy, raw_spy, vix, cash, annual_rates)
    targets = build_all_targets(legacy, spy, vix)
    candidates = {
        name: build_candidate(
            name,
            target,
            legacy,
            cash,
            released_through=HOLDOUT_END,
        )
        for name, target in targets.items()
    }
    benchmarks = build_benchmarks(spy, cash, collar)
    metrics = build_holdout_metrics(candidates, benchmarks, cash)
    metrics["passes_2022_gate"] = apply_holdout_gates(
        metrics,
        metrics.loc["synthetic_spy_collar"],
    )
    regimes = build_holdout_regimes(candidates, benchmarks, spy, vix)
    monthly = build_monthly_returns(candidates, benchmarks)
    average_weights = build_average_weights(candidates)
    development = load_development_metrics()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_report(
            metrics,
            development,
            regimes,
            monthly,
            average_weights,
            candidates,
            benchmarks,
            collar,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(metrics.to_string())


def load_released_inputs() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load history through 2022 while hard-stopping all sources before 2023."""
    cache_dir = Path(gettempdir()) / "alpha_lab_yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    tickers = [*LEGACY_SECTORS, "SPY"]
    prices = load_prices(
        tickers,
        DATA_START,
        DATA_END_EXCLUSIVE,
        field="Close",
        auto_adjust=True,
        threads=False,
    )
    raw_spy = load_prices(
        "SPY",
        DATA_START,
        DATA_END_EXCLUSIVE,
        field="Close",
        auto_adjust=False,
        threads=False,
    )["SPY"]
    macro = load_series(["VIXCLS", "DTB3"], start=DATA_START, end="2022-12-31")

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    raw_spy.index = pd.DatetimeIndex(raw_spy.index).tz_localize(None)
    sources = {"prices": prices.index, "raw_spy": raw_spy.index, "macro": macro.index}
    for name, index in sources.items():
        if index.empty or index.max() > HOLDOUT_END:
            raise RuntimeError(f"{name} is empty or crossed the authorized 2022 boundary")
    if prices.loc[HOLDOUT_START:HOLDOUT_END].empty:
        raise RuntimeError("2022 price release is empty")

    timeline = macro.index.union(prices.index).sort_values()
    aligned = macro.reindex(timeline).ffill().reindex(prices.index)
    cash_index = cash_total_return_index(macro["DTB3"].reindex(timeline).ffill())
    cash = cash_index.reindex(prices.index).pct_change().fillna(0.0).rename("cash_return")
    vix = aligned["VIXCLS"].rename("VIX")
    annual_rates = (aligned["DTB3"] / 100.0).shift(1).rename("annual_rate")
    return prices.sort_index(), raw_spy.sort_index(), vix, cash, annual_rates


def _sample(series: pd.Series) -> pd.Series:
    return series.loc[HOLDOUT_START:HOLDOUT_END].dropna()


def _performance_row(
    returns: pd.Series,
    cash: pd.Series,
    *,
    stress_returns: pd.Series | None = None,
    annual_traded_notional: float | None = None,
    equity_exposure: pd.Series | None = None,
) -> dict[str, float]:
    sample = _sample(returns)
    if sample.empty:
        raise ValueError("2022 return sample is empty")
    cash_sample = cash.reindex(sample.index).fillna(0.0)
    excess = sample - cash_sample
    dd = drawdown(sample)
    downside_sample = sample[sample < 0.0]
    tail_cut = float(sample.quantile(0.05))
    total_return = float((1.0 + sample).prod() - 1.0)
    max_drawdown = float(dd.min())
    stress_return = (
        float((1.0 + _sample(stress_returns)).prod() - 1.0)
        if stress_returns is not None
        else total_return
    )
    if equity_exposure is None:
        exposure = pd.Series(np.nan, index=sample.index)
    else:
        exposure = equity_exposure.reindex(sample.index).dropna()
    return {
        "total_return": total_return,
        "stress_20bp_return": stress_return,
        "annual_vol": float(sample.std() * np.sqrt(252)),
        "max_drawdown": max_drawdown,
        "ulcer_index": float(np.sqrt(dd.pow(2).mean())),
        "downside_deviation": float(
            np.sqrt(sample.clip(upper=0.0).pow(2).mean()) * np.sqrt(252)
        ),
        "cvar_5pct_daily": float(sample[sample <= tail_cut].mean()),
        "excess_cash_sharpe": (
            float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0.0 else np.nan
        ),
        "sortino": (
            float(excess.mean() / downside_sample.std() * np.sqrt(252))
            if len(downside_sample) > 1 and downside_sample.std() > 0.0
            else np.nan
        ),
        "calmar": total_return / abs(max_drawdown) if max_drawdown < 0.0 else np.nan,
        "cash_return": float((1.0 + cash_sample).prod() - 1.0),
        "annual_traded_notional": annual_traded_notional,
        "average_equity_exposure": float(exposure.mean()) if not exposure.empty else np.nan,
        "minimum_equity_exposure": float(exposure.min()) if not exposure.empty else np.nan,
        "maximum_equity_exposure": float(exposure.max()) if not exposure.empty else np.nan,
    }


def build_holdout_metrics(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
    cash: pd.Series,
) -> pd.DataFrame:
    """Build the preregistered single-year holdout scorecard."""
    rows: list[dict[str, object]] = []
    for name, candidate in candidates.items():
        weights = candidate.primary.weights.loc[HOLDOUT_START:HOLDOUT_END]
        cash_column = str(candidate.primary.meta["cash_column"])
        equity_exposure = weights.drop(columns=cash_column).sum(axis=1)
        years = len(_sample(candidate.primary.returns)) / 252.0
        rows.append(
            {
                "strategy": name,
                "kind": "sector_candidate",
                **_performance_row(
                    candidate.primary.returns,
                    cash,
                    stress_returns=candidate.stress.returns,
                    annual_traded_notional=float(
                        candidate.primary.traded_notional.loc[HOLDOUT_START:HOLDOUT_END].sum()
                        / years
                    ),
                    equity_exposure=equity_exposure,
                ),
            }
        )
    benchmark_exposure = {
        "SPY": pd.Series(1.0, index=cash.index),
        "synthetic_spy_collar": pd.Series(1.0, index=cash.index),
        "spy_60_cash_40": pd.Series(np.nan, index=cash.index),
    }
    for name, returns in benchmarks.items():
        rows.append(
            {
                "strategy": name,
                "kind": "benchmark",
                **_performance_row(
                    returns,
                    cash,
                    equity_exposure=benchmark_exposure[name],
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def apply_holdout_gates(metrics: pd.DataFrame, collar: pd.Series) -> pd.Series:
    """Apply only the frozen gates that are meaningful for one calendar year."""
    candidate = metrics["kind"] == "sector_candidate"
    passes = (
        candidate
        & (metrics["total_return"] >= TARGET_FLOOR)
        & (metrics["stress_20bp_return"] >= 0.07)
        & (metrics["annual_vol"] <= TARGET_VOL_MAX)
        & (metrics["max_drawdown"] >= TARGET_MAX_DRAWDOWN)
        & (metrics["max_drawdown"] > float(collar["max_drawdown"]))
        & (metrics["ulcer_index"] < float(collar["ulcer_index"]))
        & (metrics["total_return"] >= float(collar["total_return"]) - 0.01)
    )
    return passes.rename("passes_2022_gate")


def build_monthly_returns(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
) -> pd.DataFrame:
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    rows = {}
    for name, returns in series.items():
        sample = _sample(returns)
        rows[name] = sample.groupby(sample.index.month).apply(lambda values: (1.0 + values).prod() - 1.0)
    table = pd.DataFrame(rows).T
    table.columns = [pd.Timestamp(2022, month, 1).strftime("%b") for month in table.columns]
    return table


def build_average_weights(candidates: dict[str, SectorCandidate]) -> pd.DataFrame:
    """Return average realized 2022 sector and cash weights."""
    rows = {}
    for name, candidate in candidates.items():
        weights = candidate.primary.weights.loc[HOLDOUT_START:HOLDOUT_END]
        rows[name] = weights.mean()
    return pd.DataFrame(rows).T.reindex(columns=[*LEGACY_SECTORS, "CASH"])


def build_holdout_regimes(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
    spy: pd.Series,
    vix: pd.Series,
) -> pd.DataFrame:
    """Evaluate 2022 in lagged SPY-trend × fixed-VIX-20 states."""
    trend = (spy > spy.rolling(200, min_periods=200).mean()).shift(1)
    high_vix = (vix >= 20.0).shift(1)
    labels = pd.Series(index=spy.index, dtype="object")
    valid = trend.notna() & high_vix.notna()
    trend = trend.fillna(False).astype(bool)
    high_vix = high_vix.fillna(False).astype(bool)
    labels.loc[valid & trend & ~high_vix] = "bull_low_vix"
    labels.loc[valid & trend & high_vix] = "bull_high_vix"
    labels.loc[valid & ~trend & ~high_vix] = "bear_low_vix"
    labels.loc[valid & ~trend & high_vix] = "bear_high_vix"
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    rows = []
    for name, returns in series.items():
        sample = _sample(returns)
        for regime in ["bull_low_vix", "bull_high_vix", "bear_low_vix", "bear_high_vix"]:
            values = sample[labels.reindex(sample.index) == regime].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "strategy": name,
                    "regime": regime,
                    "n_days": len(values),
                    "total_return": float((1.0 + values).prod() - 1.0),
                    "annualized_mean": float(values.mean() * 252.0),
                    "annualized_vol": float(values.std() * np.sqrt(252)),
                }
            )
    return pd.DataFrame(rows).set_index(["strategy", "regime"])


def load_development_metrics() -> pd.DataFrame:
    if not DEVELOPMENT_RESULTS_PATH.exists():
        raise FileNotFoundError("development metrics must be regenerated before holdout reporting")
    return pd.read_csv(DEVELOPMENT_RESULTS_PATH, index_col="strategy")


def wealth_figure(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    figure = go.Figure()
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    for name, returns in series.items():
        sample = _sample(returns)
        figure.add_scatter(
            x=sample.index,
            y=(1.0 + sample).cumprod(),
            name=name,
            mode="lines",
            line={"width": 3 if name == "synthetic_spy_collar" else 1.5},
        )
    figure.update_layout(
        template="plotly_white",
        title="2022 holdout wealth",
        yaxis_title="Wealth",
        legend={"orientation": "h", "y": -0.28},
        margin={"b": 125},
    )
    return figure


def drawdown_figure(
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    figure = go.Figure()
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    for name, returns in series.items():
        path = drawdown(_sample(returns))
        figure.add_scatter(
            x=path.index,
            y=path,
            name=name,
            mode="lines",
            line={"width": 3 if name == "synthetic_spy_collar" else 1.5},
        )
    figure.update_layout(
        template="plotly_white",
        title="2022 holdout drawdowns from initial capital",
        yaxis_title="Drawdown",
        yaxis={"tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.28},
        margin={"b": 125},
    )
    return figure


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "kind",
        "total_return",
        "stress_20bp_return",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "downside_deviation",
        "cvar_5pct_daily",
        "excess_cash_sharpe",
        "calmar",
        "annual_traded_notional",
        "average_equity_exposure",
        "passes_2022_gate",
    ]
    frame = metrics[columns].copy()
    percent_columns = [
        "total_return",
        "stress_20bp_return",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "downside_deviation",
        "cvar_5pct_daily",
        "average_equity_exposure",
    ]
    for column in percent_columns:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    for column in ["excess_cash_sharpe", "calmar", "annual_traded_notional"]:
        frame[column] = frame[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    return frame


def _development_comparison(metrics: pd.DataFrame, development: pd.DataFrame) -> pd.DataFrame:
    shared = metrics.index.intersection(development.index)
    return pd.DataFrame(
        {
            "development CAGR": development.loc[shared, "cagr"],
            "2022 return": metrics.loc[shared, "total_return"],
            "development vol": development.loc[shared, "annual_vol"],
            "2022 vol": metrics.loc[shared, "annual_vol"],
            "development max DD": development.loc[shared, "max_drawdown"],
            "2022 max DD": metrics.loc[shared, "max_drawdown"],
        },
        index=shared,
    ).map(lambda value: f"{value:.2%}")


def _gate_matrix(metrics: pd.DataFrame, collar: pd.Series) -> pd.DataFrame:
    frame = metrics.loc[metrics["kind"] == "sector_candidate"]
    gates = pd.DataFrame(
        {
            "Return >= 7.5%": frame["total_return"] >= TARGET_FLOOR,
            "20bp return >= 7%": frame["stress_20bp_return"] >= 0.07,
            "Vol <= 10%": frame["annual_vol"] <= TARGET_VOL_MAX,
            "Max DD <= 15%": frame["max_drawdown"] >= TARGET_MAX_DRAWDOWN,
            "DD beats collar": frame["max_drawdown"] > float(collar["max_drawdown"]),
            "Ulcer beats collar": frame["ulcer_index"] < float(collar["ulcer_index"]),
            "Return near collar": frame["total_return"] >= float(collar["total_return"]) - 0.01,
        },
        index=frame.index,
    )
    return gates.map(lambda passed: "PASS" if passed else "FAIL")


def render_report(
    metrics: pd.DataFrame,
    development: pd.DataFrame,
    regimes: pd.DataFrame,
    monthly: pd.DataFrame,
    average_weights: pd.DataFrame,
    candidates: dict[str, SectorCandidate],
    benchmarks: dict[str, pd.Series],
    collar: SyntheticCollarResult,
) -> str:
    """Render the isolated 2022 holdout decision report."""
    winners = metrics.index[metrics["passes_2022_gate"]].tolist()
    sector = metrics.loc[metrics["kind"] == "sector_candidate"]
    if winners:
        verdict = f"Strict 2022 pass: {', '.join(winners)}. No parameter was changed after release."
        verdict_class = "pass"
    else:
        leader = sector.sort_values(["max_drawdown", "ulcer_index"], ascending=[False, True]).index[0]
        verdict = f"No frozen sector candidate passed every 2022 gate. Drawdown-first leader: {leader}."
        verdict_class = "reject"

    wealth_html = wealth_figure(candidates, benchmarks).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displayModeBar": False, "responsive": True},
    )
    drawdown_html = drawdown_figure(candidates, benchmarks).to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )
    collar_q2 = float(
        (1.0 + benchmarks["synthetic_spy_collar"].loc["2022-04-01":"2022-06-30"]).prod()
        - 1.0
    )
    official_q2 = -0.067
    q2_gap = collar_q2 - official_q2
    q2_class = "guard" if abs(q2_gap) <= 0.03 else "warn"
    regime_returns = regimes["total_return"].unstack("regime")
    regime_counts = regimes["n_days"].unstack("regime")
    regime_table = regime_returns.copy().astype("object")
    for strategy in regime_table.index:
        for regime in regime_table.columns:
            value = regime_returns.loc[strategy, regime]
            count = regime_counts.loc[strategy, regime]
            regime_table.loc[strategy, regime] = (
                f"{value:.2%} (n={int(count)})" if pd.notna(value) and pd.notna(count) else "—"
            )
    monthly_table = monthly.map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    average_weight_table = average_weights.map(
        lambda value: f"{value:.2%}" if pd.notna(value) else "—"
    )
    gate_matrix = _gate_matrix(metrics, metrics.loc["synthetic_spy_collar"])
    collar_diag = collar.diagnostics.loc[HOLDOUT_START:HOLDOUT_END]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>US Sector Target 8% — 2022 Holdout</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d8e0ea;--navy:#173b66;--panel:#f6f8fb;--red:#a61b1b;--green:#176b3a;--amber:#8a5a00;}}
body{{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:#fff}}
main{{max-width:1180px;margin:0 auto;padding:42px 28px 72px}}h1{{margin:0 0 8px;color:var(--navy);font-size:32px}}
h2{{margin-top:38px;padding-bottom:8px;border-bottom:2px solid var(--line);color:var(--navy)}}p,li{{line-height:1.6}}
.sub{{color:var(--muted)}}.callout{{padding:18px 20px;border-radius:8px;margin:22px 0;font-weight:600}}
.reject{{background:#fff1f1;border-left:5px solid var(--red)}}.pass{{background:#ecf8f0;border-left:5px solid var(--green)}}
.guard{{background:#eef5ff;border-left:5px solid var(--navy);font-weight:500}}.warn{{background:#fff7e6;border-left:5px solid var(--amber)}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}}
.card strong{{display:block;font-size:22px;color:var(--navy)}}.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}thead th{{background:#edf2f7;color:#27364a}}
.small{{color:var(--muted);font-size:13px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}main{{padding:24px 14px 50px}}}}
</style></head><body><main>
<h1>US Sector Portfolio: 2022 Holdout Release</h1>
<p class="sub">Frozen candidates · calendar-year 2022 only · generated 2026-07-18</p>
<div class="callout {verdict_class}">{verdict}</div>
<div class="callout guard"><strong>Boundary:</strong> only 2022 was released. No 2023+ price, VIX, rate, or option-proxy observation was loaded.</div>
<div class="grid">
<div class="card"><strong>{int(metrics['passes_2022_gate'].sum())}</strong>strict passes</div>
<div class="card"><strong>{metrics.loc['synthetic_spy_collar','total_return']:.1%}</strong>collar 2022 return</div>
<div class="card"><strong>{metrics.loc['synthetic_spy_collar','max_drawdown']:.1%}</strong>collar max DD</div>
<div class="card"><strong>{metrics.loc['SPY','total_return']:.1%}</strong>SPY 2022 return</div>
</div>

<h2>2022 scorecard</h2>
<div class="table-wrap">{_format_metrics(metrics).to_html(border=0)}</div>
<p class="small">Sector results include 10 bp one-way costs, a 2% rebalance threshold, and T-bill cash returns. Stress returns use 20 bp.</p>

<h2>Frozen gate diagnostics</h2>
<div class="table-wrap">{gate_matrix.to_html(border=0)}</div>

<h2>Average realized 2022 allocation</h2>
<div class="table-wrap">{average_weight_table.to_html(border=0)}</div>

<h2>Development versus holdout</h2>
<div class="table-wrap">{_development_comparison(metrics, development).to_html(border=0)}</div>
<p class="small">Development is 2013-2021 CAGR/risk; holdout is the single 2022 calendar-year return/risk.</p>

<h2>Wealth and drawdown</h2>
{wealth_html}
{drawdown_html}

<h2>Monthly returns</h2>
<div class="table-wrap">{monthly_table.to_html(border=0)}</div>

<h2>2022 regime diagnostics</h2>
<p>Compounded return and observation count within lagged SPY trend × fixed VIX-20 states. These states were not used to tune the candidates.</p>
<div class="table-wrap">{regime_table.to_html(border=0)}</div>

<h2>Synthetic collar model check</h2>
<ul>
<li>{int(collar_diag['put_roll'].sum())} quarterly put rolls and {int(collar_diag['call_roll'].sum())} monthly call rolls in 2022.</li>
<li>Mean model put IV {collar_diag['put_iv'].mean():.1%}; mean model call IV {collar_diag['call_iv'].mean():.1%}.</li>
</ul>
<div class="callout {q2_class}"><strong>External plausibility check:</strong> the model collar returned {collar_q2:.1%} in 2022Q2,
versus -6.7% reported by Cboe for CLL. The gap is {q2_gap:+.1%}. No IV or spread assumption was retuned.</div>
<div class="callout guard">The collar remains a scenario benchmark rather than an executable option-chain backtest. It omits historical NBBO,
strike grids, discrete dividends, American exercise, and SPX/SPY basis.</div>

<h2>Research controls and decision</h2>
<ul>
<li>All five candidates and every parameter were frozen before 2022 was loaded.</li>
<li>Signals use trailing observations and target weights trade at the next close.</li>
<li>Initial capital is included as the first drawdown high-water mark.</li>
<li>2023+ remains sealed and no post-release variant was added.</li>
</ul>
<p>{'The passing candidate may proceed to a separately registered robustness/paper phase; this is not deployment approval.' if winners else 'Retain the development rejection. Any sector-plus-options hybrid must begin as a new preregistered study.'}</p>
</main></body></html>"""


if __name__ == "__main__":
    main()
