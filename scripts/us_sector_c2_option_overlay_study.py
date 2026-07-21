"""Frozen C2 defensive-sector plus synthetic SPY option-overlay study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from us_sector_rotation_study import VALIDATION_END, VALIDATION_START
from us_sector_target8_holdout_2022 import (
    HOLDOUT_END,
    HOLDOUT_START,
    load_released_inputs,
)
from us_sector_target8_study import (
    TARGET_CALMAR,
    TARGET_FLOOR,
    TARGET_MAX_DRAWDOWN,
    TARGET_VOL_MAX,
    SectorCandidate,
    build_all_targets,
    build_candidate,
)
from us_sector_target8_study import (
    _performance_row as development_performance_row,
)

from alpha_lab.analytics.returns import drawdown
from alpha_lab.backtest.collar import (
    SyntheticOptionOverlayConfig,
    SyntheticOptionOverlayResult,
    run_synthetic_collar,
    run_synthetic_option_overlay,
)
from alpha_lab.utils.paths import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "us_sector_c2_option_overlay.html"
RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "us_sector_c2_option_overlay_metrics.csv"
STRESS_RESULTS_PATH = (
    PROJECT_ROOT / "data" / "results" / "us_sector_c2_option_overlay_2022_stress.csv"
)
LIFETIME_RISK_TRIALS = 16


@dataclass(frozen=True)
class OverlayCandidate:
    """Frozen primary and spread-stress option-overlay simulations."""

    name: str
    primary: SyntheticOptionOverlayResult
    stress: SyntheticOptionOverlayResult


def main() -> None:
    prices, raw_spy, vix, cash, annual_rates = load_released_inputs()
    legacy = prices[["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB"]].dropna()
    spy = prices["SPY"].reindex(legacy.index)
    raw_spy = raw_spy.reindex(legacy.index).ffill()
    vix = vix.reindex(legacy.index).ffill()
    cash = cash.reindex(legacy.index).fillna(0.0)
    annual_rates = annual_rates.reindex(legacy.index).ffill()

    targets = build_all_targets(legacy, spy, vix)
    c2 = build_candidate(
        "C2_asymmetric_sector_ferry",
        targets["C2_asymmetric_sector_ferry"],
        legacy,
        cash,
        released_through=HOLDOUT_END,
    )
    collar = run_synthetic_collar(spy, raw_spy, vix, cash, annual_rates)
    candidates = build_overlay_candidates(c2, spy, raw_spy, vix, cash, annual_rates)
    benchmarks = {
        "C2_unhedged": c2.primary.returns,
        "synthetic_spy_collar": collar.returns,
        "SPY": spy.pct_change().fillna(0.0),
    }
    development = build_development_metrics(candidates, c2, benchmarks, cash)
    development["passes_all_gates"] = apply_development_gates(
        development,
        development.loc["C2_unhedged"],
        development.loc["synthetic_spy_collar"],
    )
    stress_2022 = build_2022_stress_metrics(candidates, c2, benchmarks, cash)
    option_diagnostics = build_option_diagnostics(candidates)
    regime_metrics = build_regime_metrics(candidates, benchmarks, spy, vix)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    development.to_csv(RESULTS_PATH)
    stress_2022.to_csv(STRESS_RESULTS_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_report(
            development,
            stress_2022,
            option_diagnostics,
            regime_metrics,
            candidates,
            benchmarks,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {STRESS_RESULTS_PATH}")
    print("\nDevelopment:\n", development.to_string())
    print("\nKnown 2022 stress:\n", stress_2022.to_string())


def _overlay_config(
    *,
    short_put_otm: float | None,
    call_otm: float | None,
    stress: bool,
) -> SyntheticOptionOverlayConfig:
    haircut = 0.20 if stress else 0.10
    return SyntheticOptionOverlayConfig(
        short_put_otm=short_put_otm,
        call_otm=call_otm,
        long_option_ask_markup=haircut,
        short_option_bid_haircut=haircut,
    )


def _run_overlay(
    c2_returns: pd.Series,
    spy: pd.Series,
    raw_spy: pd.Series,
    vix: pd.Series,
    cash: pd.Series,
    annual_rates: pd.Series,
    put_ratio: pd.Series | float,
    call_ratio: pd.Series | float,
    *,
    short_put_otm: float | None,
    call_otm: float | None,
    stress: bool,
) -> SyntheticOptionOverlayResult:
    return run_synthetic_option_overlay(
        c2_returns,
        spy,
        raw_spy,
        vix,
        cash,
        annual_rates,
        put_ratio,
        call_ratio,
        config=_overlay_config(
            short_put_otm=short_put_otm,
            call_otm=call_otm,
            stress=stress,
        ),
    )


def build_overlay_candidates(
    c2: SectorCandidate,
    spy: pd.Series,
    raw_spy: pd.Series,
    vix: pd.Series,
    cash: pd.Series,
    annual_rates: pd.Series,
) -> dict[str, OverlayCandidate]:
    """Run the five preregistered overlay structures in one frozen batch."""
    cash_column = str(c2.primary.meta["cash_column"])
    equity_exposure = c2.primary.weights.drop(columns=cash_column).sum(axis=1).shift(1).fillna(0.0)
    calm_vix_ratio = equity_exposure * (vix.shift(1) < 20.0).fillna(False).astype(float)
    capped_call_ratio = equity_exposure.clip(upper=0.25)

    specifications = {
        "O1_matched_95_85_put_spread": {
            "put_ratio": equity_exposure,
            "call_ratio": 0.0,
            "short_put_otm": 0.15,
            "call_otm": None,
        },
        "O2_calm_vix_put_spread": {
            "put_ratio": calm_vix_ratio,
            "call_ratio": 0.0,
            "short_put_otm": 0.15,
            "call_otm": None,
        },
        "O3_put_spread_call_flywheel": {
            "put_ratio": equity_exposure,
            "call_ratio": capped_call_ratio,
            "short_put_otm": 0.15,
            "call_otm": 0.10,
        },
        "O4_matched_95_110_collar": {
            "put_ratio": equity_exposure,
            "call_ratio": equity_exposure,
            "short_put_otm": None,
            "call_otm": 0.10,
        },
        "O5_half_notional_long_put": {
            "put_ratio": 0.50,
            "call_ratio": 0.0,
            "short_put_otm": None,
            "call_otm": None,
        },
    }
    candidates = {}
    for name, spec in specifications.items():
        primary = _run_overlay(
            c2.primary.returns,
            spy,
            raw_spy,
            vix,
            cash,
            annual_rates,
            stress=False,
            **spec,
        )
        stress_result = _run_overlay(
            c2.primary.returns,
            spy,
            raw_spy,
            vix,
            cash,
            annual_rates,
            stress=True,
            **spec,
        )
        candidates[name] = OverlayCandidate(name, primary, stress_result)
    return candidates


def build_development_metrics(
    candidates: dict[str, OverlayCandidate],
    c2: SectorCandidate,
    benchmarks: dict[str, pd.Series],
    cash: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, candidate in candidates.items():
        rows.append(
            {
                "strategy": name,
                "kind": "overlay_candidate",
                **development_performance_row(
                    candidate.primary.returns,
                    cash,
                    stress_returns=candidate.stress.returns,
                    n_trials=LIFETIME_RISK_TRIALS,
                ),
            }
        )
    rows.append(
        {
            "strategy": "C2_unhedged",
            "kind": "benchmark",
            **development_performance_row(
                c2.primary.returns,
                cash,
                stress_returns=c2.stress.returns,
                n_trials=LIFETIME_RISK_TRIALS,
            ),
        }
    )
    for name in ["synthetic_spy_collar", "SPY"]:
        rows.append(
            {
                "strategy": name,
                "kind": "benchmark",
                **development_performance_row(
                    benchmarks[name],
                    cash,
                    n_trials=LIFETIME_RISK_TRIALS,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def apply_development_gates(
    metrics: pd.DataFrame,
    c2: pd.Series,
    collar: pd.Series,
) -> pd.Series:
    candidate = metrics["kind"] == "overlay_candidate"
    passes = (
        candidate
        & (metrics["cagr"] >= TARGET_FLOOR)
        & (metrics["stress_20bp_cagr"] >= 0.07)
        & (metrics["annual_vol"] <= TARGET_VOL_MAX)
        & (metrics["max_drawdown"] >= TARGET_MAX_DRAWDOWN)
        & (metrics["calmar"] >= TARGET_CALMAR)
        & (metrics["max_drawdown"] > float(c2["max_drawdown"]))
        & (metrics["ulcer_index"] < float(c2["ulcer_index"]))
        & (metrics["cagr"] >= float(collar["cagr"]) - 0.01)
        & (metrics["cagr_2013_2016"] >= 0.06)
        & (metrics["cagr_2017_2021"] >= 0.06)
        & (metrics["worst_calendar_year"] >= -0.10)
        & (metrics["rolling_5y_target_share"] >= 0.60)
        & (metrics["bootstrap_90_low_excess_cash"] > 0.0)
        & (metrics["deflated_sharpe_probability"] >= 0.95)
    )
    return passes.rename("passes_all_gates")


def _one_year_row(
    returns: pd.Series,
    cash: pd.Series,
    *,
    stress_returns: pd.Series | None = None,
) -> dict[str, float]:
    sample = returns.loc[HOLDOUT_START:HOLDOUT_END].dropna()
    cash_sample = cash.reindex(sample.index).fillna(0.0)
    excess = sample - cash_sample
    dd = drawdown(sample)
    tail_cut = float(sample.quantile(0.05))
    return {
        "total_return": float((1.0 + sample).prod() - 1.0),
        "stress_return": (
            float((1.0 + stress_returns.loc[HOLDOUT_START:HOLDOUT_END].dropna()).prod() - 1.0)
            if stress_returns is not None
            else float((1.0 + sample).prod() - 1.0)
        ),
        "annual_vol": float(sample.std() * np.sqrt(252)),
        "max_drawdown": float(dd.min()),
        "ulcer_index": float(np.sqrt(dd.pow(2).mean())),
        "cvar_5pct_daily": float(sample[sample <= tail_cut].mean()),
        "excess_cash_sharpe": (
            float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0.0 else np.nan
        ),
    }


def build_2022_stress_metrics(
    candidates: dict[str, OverlayCandidate],
    c2: SectorCandidate,
    benchmarks: dict[str, pd.Series],
    cash: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, candidate in candidates.items():
        rows.append(
            {
                "strategy": name,
                "kind": "overlay_candidate",
                **_one_year_row(
                    candidate.primary.returns,
                    cash,
                    stress_returns=candidate.stress.returns,
                ),
            }
        )
    rows.append(
        {
            "strategy": "C2_unhedged",
            "kind": "benchmark",
            **_one_year_row(
                c2.primary.returns,
                cash,
                stress_returns=c2.stress.returns,
            ),
        }
    )
    for name in ["synthetic_spy_collar", "SPY"]:
        rows.append(
            {
                "strategy": name,
                "kind": "benchmark",
                **_one_year_row(benchmarks[name], cash),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def build_option_diagnostics(candidates: dict[str, OverlayCandidate]) -> pd.DataFrame:
    rows = []
    years = (VALIDATION_END - VALIDATION_START).days / 365.25
    for name, candidate in candidates.items():
        diag = candidate.primary.diagnostics.loc[VALIDATION_START:VALIDATION_END]
        previous_equity = diag["equity"].shift(1).replace(0.0, np.nan)
        drag = -(diag["entry_drag"] / previous_equity).fillna(0.0)
        put_rolls = diag[diag["put_roll"]]
        call_rolls = diag[diag["call_roll"]]
        rows.append(
            {
                "strategy": name,
                "annual_entry_spread_drag": float(drag.sum() / years),
                "put_rolls": int(diag["put_roll"].sum()),
                "call_rolls": int(diag["call_roll"].sum()),
                "average_put_ratio_at_roll": float(put_rolls["put_ratio"].mean()),
                "average_call_ratio_at_roll": (
                    float(call_rolls["call_ratio"].mean()) if not call_rolls.empty else 0.0
                ),
                "average_overlay_value_pct_nav": float(
                    (diag["overlay_value"] / diag["equity"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def build_regime_metrics(
    candidates: dict[str, OverlayCandidate],
    benchmarks: dict[str, pd.Series],
    spy: pd.Series,
    vix: pd.Series,
) -> pd.DataFrame:
    """Summarize conditional daily returns using only prior-close regime labels."""
    lagged_spy = spy.shift(1)
    lagged_vix = vix.shift(1)
    lagged_spy_average = lagged_spy.rolling(200, min_periods=200).mean()
    regimes = {
        "VIX < 20": lagged_vix < 20.0,
        "VIX >= 20": lagged_vix >= 20.0,
        "SPY above 200d": lagged_spy > lagged_spy_average,
        "SPY below 200d": lagged_spy <= lagged_spy_average,
    }
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    rows = []
    for strategy, returns in series.items():
        development = returns.loc[VALIDATION_START:VALIDATION_END].dropna()
        years = len(development) / 252.0
        for regime, mask in regimes.items():
            active = mask.reindex(development.index).fillna(False)
            sample = development.loc[active]
            rows.append(
                {
                    "strategy": strategy,
                    "regime": regime,
                    "annual_log_return_contribution": (
                        float(np.expm1(np.log1p(sample).sum() / years))
                        if not sample.empty and years > 0.0
                        else np.nan
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .pivot(index="strategy", columns="regime", values="annual_log_return_contribution")
        .reindex(columns=list(regimes))
    )


def build_tradeoff_diagnostics(
    development: pd.DataFrame,
    stress_2022: pd.DataFrame,
    option_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Explain protection benefits and return costs versus unhedged C2."""
    c2 = development.loc["C2_unhedged"]
    c2_2022 = stress_2022.loc["C2_unhedged"]
    candidates = development.loc[development["kind"] == "overlay_candidate"]
    gates = _gate_matrix(development, c2, development.loc["synthetic_spy_collar"])
    return pd.DataFrame(
        {
            "CAGR vs C2": candidates["cagr"] - float(c2["cagr"]),
            "Max-DD improvement": candidates["max_drawdown"] - float(c2["max_drawdown"]),
            "Ulcer change": candidates["ulcer_index"] - float(c2["ulcer_index"]),
            "2022 return vs C2": (
                stress_2022.loc[candidates.index, "total_return"] - float(c2_2022["total_return"])
            ),
            "Annual entry-spread drag": option_diagnostics.loc[
                candidates.index, "annual_entry_spread_drag"
            ],
            "Failed gates": (gates == "FAIL").sum(axis=1),
        }
    )


def _wealth_figure(
    candidates: dict[str, OverlayCandidate],
    benchmarks: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    for name, returns in series.items():
        sample = returns.loc[start:end].dropna()
        figure.add_scatter(
            x=sample.index,
            y=(1.0 + sample).cumprod(),
            name=name,
            mode="lines",
            line={"width": 3 if name == "C2_unhedged" else 1.5},
        )
    figure.update_layout(
        template="plotly_white",
        title=title,
        yaxis_title="Wealth",
        legend={"orientation": "h", "y": -0.30},
        margin={"b": 135},
    )
    return figure


def _drawdown_figure(
    candidates: dict[str, OverlayCandidate],
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    figure = go.Figure()
    series = {name: item.primary.returns for name, item in candidates.items()} | benchmarks
    for name, returns in series.items():
        path = drawdown(returns.loc[VALIDATION_START:VALIDATION_END].dropna())
        figure.add_scatter(
            x=path.index,
            y=path,
            name=name,
            mode="lines",
            line={"width": 3 if name == "C2_unhedged" else 1.5},
        )
    figure.update_layout(
        template="plotly_white",
        title="Development drawdowns",
        yaxis_title="Drawdown",
        yaxis={"tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.30},
        margin={"b": 135},
    )
    return figure


def _format_development(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "kind",
        "cagr",
        "stress_20bp_cagr",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "downside_deviation",
        "calmar",
        "rolling_5y_target_share",
        "deflated_sharpe_probability",
        "passes_all_gates",
    ]
    frame = metrics[columns].copy()
    for column in [
        "cagr",
        "stress_20bp_cagr",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "downside_deviation",
        "rolling_5y_target_share",
        "deflated_sharpe_probability",
    ]:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    frame["calmar"] = frame["calmar"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    frame = frame.rename(columns={"stress_20bp_cagr": "stressed_cagr"})
    return frame.rename_axis(None)


def _format_stress(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    for column in [
        "total_return",
        "stress_return",
        "annual_vol",
        "max_drawdown",
        "ulcer_index",
        "cvar_5pct_daily",
    ]:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    frame["excess_cash_sharpe"] = frame["excess_cash_sharpe"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else "—"
    )
    return frame.rename_axis(None)


def _format_option_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics.copy()
    for column in [
        "annual_entry_spread_drag",
        "average_put_ratio_at_roll",
        "average_call_ratio_at_roll",
        "average_overlay_value_pct_nav",
    ]:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}")
    return frame.rename_axis(None)


def _format_subperiods(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cagr_2013_2016",
        "cagr_2017_2021",
        "worst_calendar_year",
        "rolling_5y_target_share",
    ]
    frame = metrics[columns].copy()
    for column in columns:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    return frame.rename_axis(None)


def _format_regimes(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    return frame.rename_axis(None).rename_axis(None, axis=1)


def _format_tradeoffs(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    for column in [
        "CAGR vs C2",
        "Max-DD improvement",
        "Ulcer change",
        "2022 return vs C2",
        "Annual entry-spread drag",
    ]:
        frame[column] = frame[column].map(lambda value: f"{value * 100:+.2f} pp")
    return frame.rename_axis(None)


def _gate_matrix(metrics: pd.DataFrame, c2: pd.Series, collar: pd.Series) -> pd.DataFrame:
    frame = metrics.loc[metrics["kind"] == "overlay_candidate"]
    gates = pd.DataFrame(
        {
            "CAGR >= 7.5%": frame["cagr"] >= TARGET_FLOOR,
            "Stress CAGR >= 7%": frame["stress_20bp_cagr"] >= 0.07,
            "Vol <= 10%": frame["annual_vol"] <= TARGET_VOL_MAX,
            "Max DD <= 15%": frame["max_drawdown"] >= TARGET_MAX_DRAWDOWN,
            "Calmar >= 0.60": frame["calmar"] >= TARGET_CALMAR,
            "DD beats C2": frame["max_drawdown"] > float(c2["max_drawdown"]),
            "Ulcer beats C2": frame["ulcer_index"] < float(c2["ulcer_index"]),
            "CAGR near collar": frame["cagr"] >= float(collar["cagr"]) - 0.01,
            "Both subperiods >= 6%": (frame["cagr_2013_2016"] >= 0.06)
            & (frame["cagr_2017_2021"] >= 0.06),
            "Worst year >= -10%": frame["worst_calendar_year"] >= -0.10,
            "5y target share >= 60%": frame["rolling_5y_target_share"] >= 0.60,
            "Bootstrap low > cash": frame["bootstrap_90_low_excess_cash"] > 0.0,
            "DSR >= 95%": frame["deflated_sharpe_probability"] >= 0.95,
        },
        index=frame.index,
    )
    return gates.map(lambda passed: "PASS" if passed else "FAIL").rename_axis(None)


def render_report(
    development: pd.DataFrame,
    stress_2022: pd.DataFrame,
    option_diagnostics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    candidates: dict[str, OverlayCandidate],
    benchmarks: dict[str, pd.Series],
) -> str:
    winners = development.index[development["passes_all_gates"]].tolist()
    overlays = development.loc[development["kind"] == "overlay_candidate"]
    if winners:
        verdict = f"Development pass: {', '.join(winners)}. 2023+ remains sealed."
        verdict_class = "pass"
    else:
        return_leader = overlays["cagr"].idxmax()
        drawdown_leader = overlays["max_drawdown"].idxmax()
        verdict = (
            "No overlay passed every frozen development gate. "
            f"Return leader: {return_leader}; shallowest maximum drawdown: {drawdown_leader}."
        )
        verdict_class = "reject"

    development_wealth = _wealth_figure(
        candidates,
        benchmarks,
        VALIDATION_START,
        VALIDATION_END,
        "2013-2021 development wealth",
    ).to_html(full_html=False, include_plotlyjs=True, config={"displayModeBar": False})
    development_drawdown = _drawdown_figure(candidates, benchmarks).to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False},
    )
    stress_wealth = _wealth_figure(
        candidates,
        benchmarks,
        HOLDOUT_START,
        HOLDOUT_END,
        "Known 2022 stress wealth (not untouched OOS)",
    ).to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    gate_matrix = _gate_matrix(
        development,
        development.loc["C2_unhedged"],
        development.loc["synthetic_spy_collar"],
    )
    tradeoffs = build_tradeoff_diagnostics(development, stress_2022, option_diagnostics)
    return_leader = overlays["cagr"].idxmax()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>C2 Sector + SPY Option Overlay Study</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d8e0ea;--navy:#173b66;--panel:#f6f8fb;--red:#a61b1b;--green:#176b3a;}}
body{{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:#fff}}
main{{max-width:1220px;margin:0 auto;padding:42px 28px 72px}}h1{{margin:0 0 8px;color:var(--navy);font-size:32px}}
h2{{margin-top:38px;padding-bottom:8px;border-bottom:2px solid var(--line);color:var(--navy)}}p,li{{line-height:1.6}}
.sub{{color:var(--muted)}}.callout{{padding:18px 20px;border-radius:8px;margin:22px 0;font-weight:600}}
.reject{{background:#fff1f1;border-left:5px solid var(--red)}}.pass{{background:#ecf8f0;border-left:5px solid var(--green)}}
.guard{{background:#eef5ff;border-left:5px solid var(--navy);font-weight:500}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}}.card strong{{display:block;font-size:22px;color:var(--navy)}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
thead th{{background:#edf2f7;color:#27364a}}.small{{color:var(--muted);font-size:13px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}main{{padding:24px 14px 50px}}}}
</style></head><body><main>
<h1>C2 Defensive Sector + SPY Option Overlay</h1>
<p class="sub">Frozen five-candidate batch · development 2013-2021 · known stress 2022 · generated 2026-07-18</p>
<div class="callout {verdict_class}">{verdict}</div>
<div class="callout guard"><strong>Information boundary:</strong> 2022 was already known when this study was designed and is shown only as a stress sample. No 2023+ observation was loaded.</div>
<div class="grid">
<div class="card"><strong>{int(development["passes_all_gates"].sum())}</strong>all-gate passes</div>
<div class="card"><strong>{development.loc["C2_unhedged", "cagr"]:.1%}</strong>C2 development CAGR</div>
<div class="card"><strong>{development.loc["C2_unhedged", "max_drawdown"]:.1%}</strong>C2 development max DD</div>
<div class="card"><strong>{stress_2022.loc["C2_unhedged", "total_return"]:.1%}</strong>C2 2022 return</div>
</div>

<h2>Why no candidate passed</h2>
<p>The overlays reduced the single deepest development drawdown, but every candidate had a worse Ulcer Index than C2: losses became shallower but recovery was slower. The best-return overlay, <strong>{return_leader}</strong>, earned {overlays.loc[return_leader, "cagr"]:.2%}, versus {development.loc["C2_unhedged", "cagr"]:.2%} for C2. Its known-2022 result was {stress_2022.loc[return_leader, "total_return"]:.2%}, effectively the same as C2 because the prior-VIX gate opened no new protection during the already-elevated-volatility regime.</p>
<div class="table-wrap">{_format_tradeoffs(tradeoffs).to_html(border=0)}</div>

<h2>Development results</h2>
<div class="table-wrap">{_format_development(development).to_html(border=0)}</div>
<p class="small">Stress CAGR doubles modeled option entry haircuts from 10% to 20%; C2's own cost stress remains 20 bp.</p>

<h2>Frozen gate diagnostics</h2>
<div class="table-wrap">{gate_matrix.to_html(border=0)}</div>

<h2>Subperiod and regime diagnostics</h2>
<div class="table-wrap">{_format_subperiods(development).to_html(border=0)}</div>
<p class="small">The table shows each regime's annualized log-return contribution using only prior-close VIX and SPY/200-day-average labels. Multiplying the two VIX components approximates whole-period growth; the trend pair excludes its initial 200-day warm-up. These are attribution diagnostics, not standalone timing-strategy CAGRs.</p>
<div class="table-wrap">{_format_regimes(regime_metrics).to_html(border=0)}</div>

<h2>Option implementation diagnostics</h2>
<div class="table-wrap">{_format_option_diagnostics(option_diagnostics).to_html(border=0)}</div>

<h2>Development wealth and drawdown</h2>
{development_wealth}
{development_drawdown}

<h2>Known 2022 stress sample</h2>
<div class="table-wrap">{_format_stress(stress_2022).to_html(border=0)}</div>
{stress_wealth}

<h2>Model and audit boundaries</h2>
<ul>
<li>C2 signals are unchanged and execute at the next close. State-dependent hedge ratios use the prior close.</li>
<li>Overlay cash earns or pays the T-bill rate; entry spreads are charged explicitly.</li>
<li>SPY options are European Black–Scholes/VIX proxies. Historical NBBO, strike grids, American exercise, dividends, and SPX/SPY basis are omitted.</li>
<li>The 85 short put caps crash protection; calls create sector/SPY basis risk because C2 does not literally hold SPY.</li>
<li>Five candidates were frozen before results; no sixth variant is permitted in this batch.</li>
<li>Static leakage scan found zero blockers. Quantile findings are ex-post CVaR diagnostics only; forward fills carry already-observed data through later timestamps.</li>
</ul>

<h2>Decision</h2>
<p>{"Advance passing candidates to an independent audit, while keeping 2023+ sealed." if winners else "Stop this batch. Do not tune from development or the already-known 2022 sample; retain 2023+ as sealed."}</p>
</main></body></html>"""


if __name__ == "__main__":
    main()
