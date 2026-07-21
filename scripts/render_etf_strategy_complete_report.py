"""Render the complete self-contained pre-2022 ETF strategy HTML report."""

from __future__ import annotations

import base64
import io
import json
import sys
from html import escape

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alpha_lab.analytics.returns import drawdown, drawdown_duration_metrics  # noqa: E402
from alpha_lab.utils.paths import PROJECT_ROOT  # noqa: E402

SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import etf_strategy_ensemble_recovery_study as ensemble  # noqa: E402
import etf_strategy_ensemble_robustness as robustness  # noqa: E402

DATA = PROJECT_ROOT / "data" / "results" / "etf_strategy_50plus_pre2022"
OUTPUT = PROJECT_ROOT / "reports" / "etf_strategy_complete_pre2022.html"

FAST_TYPICAL_NAME = (
    "full__s100__cross_asset_mom_126_top3__"
    "recovery_sector_barbell_M_cyc65_mom63_ma100_top2"
)
FAST_WORST_NAME = (
    "full__s100__retail_all_weather_fixed__"
    "recovery_sector_barbell_M_cyc65_mom63_ma100_top2"
)
RELAXED_ROBUST_NAME = "relaxed_5pct_M_45"

LABELS = {
    FAST_TYPICAL_NAME: "10% candidate · cross-asset momentum + sector barbell",
    FAST_WORST_NAME: "10% candidate · all-weather + sector barbell",
    RELAXED_ROBUST_NAME: "5% candidate · 45% risk sleeves + 55% SHY",
    "SPY": "SPY buy & hold",
}

COLORS = {
    FAST_TYPICAL_NAME: "#0f766e",
    FAST_WORST_NAME: "#2563eb",
    RELAXED_ROBUST_NAME: "#d97706",
    "SPY": "#94a3b8",
}


def _figure_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _percent(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def _number(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _friendly_strategy(name: object) -> str:
    text = str(name)
    if text in LABELS:
        return LABELS[text]
    replacements = {
        "credit_canary_equity_allocation": "HYG/IEF credit canary",
        "QQQ_vol_target_12": "QQQ 12% volatility target",
        "cross_asset_mom_126_top3": "Cross-asset 6m momentum top 3",
        "sector_low_ulcer_positive_trend": "Sector low-Ulcer positive trend",
        "sector_downside_trend": "Sector downside/trend",
        "SPY_trend_150d": "SPY 150d trend",
        "recovery_sector_barbell_M_cyc50_mom63_ma200_top3": (
            "Recovery barbell · 50% cyclical / 63d mom / 200d MA / top 3"
        ),
        "recovery_sector_barbell_M_cyc65_mom63_ma100_top2": (
            "Recovery barbell · 65% cyclical / 63d mom / 100d MA / top 2"
        ),
        "recovery_sector_barbell_M_cyc65_mom63_ma200_top3": (
            "Recovery barbell · 65% cyclical / 63d mom / 200d MA / top 3"
        ),
    }
    return replacements.get(text, text.replace("__", " + "))


def _table(
    frame: pd.DataFrame,
    *,
    percent_columns: set[str] | None = None,
    integer_columns: set[str] | None = None,
    rename: dict[str, str] | None = None,
    strategy_column: str = "strategy",
    classes: str = "data-table",
) -> str:
    percent_columns = percent_columns or set()
    integer_columns = integer_columns or set()
    display = frame.copy()
    if strategy_column in display:
        display[strategy_column] = display[strategy_column].map(_friendly_strategy)
    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(_percent)
        elif column in integer_columns:
            display[column] = display[column].map(
                lambda value: "—" if pd.isna(value) else f"{int(float(value)):,}"
            )
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(_number)
    if rename:
        display = display.rename(columns=rename)
    return display.to_html(index=False, border=0, classes=classes, escape=True)


def _load() -> dict[str, object]:
    required = [
        "drawdown_recovery_existing_metrics.csv",
        "drawdown_recovery_variant_metrics.csv",
        "ensemble_recovery_metrics.csv",
        "ensemble_recovery_robustness.csv",
        "selected_stress_windows.csv",
        "selected_regimes.csv",
        "meta.json",
        "drawdown_recovery_meta.json",
        "ensemble_recovery_meta.json",
    ]
    missing = [name for name in required if not (DATA / name).exists()]
    if missing:
        raise FileNotFoundError(f"report inputs are missing: {missing}")
    return {
        "original": pd.read_csv(DATA / "drawdown_recovery_existing_metrics.csv"),
        "recovery": pd.read_csv(DATA / "drawdown_recovery_variant_metrics.csv"),
        "ensemble": pd.read_csv(DATA / "ensemble_recovery_metrics.csv"),
        "robustness": pd.read_csv(DATA / "ensemble_recovery_robustness.csv"),
        "stress": pd.read_csv(DATA / "selected_stress_windows.csv"),
        "regimes": pd.read_csv(DATA / "selected_regimes.csv"),
        "meta": json.loads((DATA / "meta.json").read_text(encoding="utf-8")),
        "recovery_meta": json.loads(
            (DATA / "drawdown_recovery_meta.json").read_text(encoding="utf-8")
        ),
        "ensemble_meta": json.loads(
            (DATA / "ensemble_recovery_meta.json").read_text(encoding="utf-8")
        ),
    }


def _candidate_returns() -> pd.DataFrame:
    sleeve_returns, shy_returns = ensemble._load_sleeve_returns()
    full = ensemble._tier_returns(sleeve_returns, shy_returns, ensemble.FULL_GFC_SLEEVES)

    target_typical = robustness._fixed_targets(
        full.index,
        full.columns,
        {
            "cross_asset_mom_126_top3": 0.50,
            "recovery_sector_barbell_M_cyc65_mom63_ma100_top2": 0.50,
        },
        frequency="M",
    )
    target_worst = robustness._fixed_targets(
        full.index,
        full.columns,
        {
            "retail_all_weather_fixed": 0.50,
            "recovery_sector_barbell_M_cyc65_mom63_ma100_top2": 0.50,
        },
        frequency="M",
    )
    relaxed_weights = {
        sleeve: 0.45 / len(robustness.RELAXED_CORE)
        for sleeve in robustness.RELAXED_CORE
    }
    target_relaxed = robustness._fixed_targets(
        full.index,
        full.columns,
        relaxed_weights,
        frequency="M",
    )
    returns = {}
    for name, targets in (
        (FAST_TYPICAL_NAME, target_typical),
        (FAST_WORST_NAME, target_worst),
        (RELAXED_ROBUST_NAME, target_relaxed),
    ):
        result, _, _, _ = ensemble._fast_monthly_drift(
            full,
            targets,
            primary_bps=5.0,
            stress_bps=20.0,
        )
        returns[name] = result
    adjusted = pd.read_parquet(DATA / "market_prices_adjusted_pre2022.parquet")
    spy = adjusted["SPY"].pct_change().reindex(full.index).fillna(0.0)
    returns["SPY"] = spy
    frame = pd.concat(returns, axis=1, sort=False).sort_index().loc[:"2021-12-31"]
    if frame.index.max() > pd.Timestamp("2021-12-31"):
        raise AssertionError("post-2021 return entered report")
    return frame


def _candidate_metrics(data: dict[str, object]) -> pd.DataFrame:
    ensemble_metrics = data["ensemble"].set_index("strategy")  # type: ignore[union-attr]
    robust = data["robustness"].set_index("strategy")  # type: ignore[union-attr]
    rows = []
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME):
        row = ensemble_metrics.loc[name].copy()
        row["strategy"] = name
        rows.append(row)
    relaxed = robust.loc[RELAXED_ROBUST_NAME].copy()
    relaxed["strategy"] = RELAXED_ROBUST_NAME
    rows.append(relaxed)
    return pd.DataFrame(rows).set_index("strategy", drop=False)


def _build_charts(
    data: dict[str, object], candidate_returns: pd.DataFrame, candidate_metrics: pd.DataFrame
) -> dict[str, str]:
    charts: dict[str, str] = {}
    common = candidate_returns.dropna()
    equity = (1.0 + common).cumprod()
    equity = equity / equity.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for name in equity:
        ax.plot(
            equity.index,
            equity[name],
            label=LABELS[name],
            color=COLORS[name],
            linewidth=2.0 if name != "SPY" else 1.3,
            alpha=0.95,
        )
    ax.set_yscale("log")
    ax.set(title="Growth of $1 · common pre-2022 sample", ylabel="Wealth (log scale)", xlabel="")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    charts["equity"] = _figure_uri(fig)

    fig, ax = plt.subplots(figsize=(12, 5.0))
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME, RELAXED_ROBUST_NAME):
        dd = drawdown(common[name]) * 100
        ax.plot(dd.index, dd, label=LABELS[name], color=COLORS[name], linewidth=1.8)
    ax.axhline(-5, color="#ef4444", linestyle="--", linewidth=1.0, label="-5% material threshold")
    ax.set(title="Drawdown paths", ylabel="Drawdown (%)", xlabel="")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    charts["drawdown"] = _figure_uri(fig)

    ensemble_metrics = data["ensemble"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    full = ensemble_metrics[ensemble_metrics["tier"] == "full_gfc_2007_2021"]
    credit = ensemble_metrics[ensemble_metrics["tier"] == "credit_2008_2021"]
    ax.scatter(
        full["annual_vol"] * 100,
        full["cagr"] * 100,
        s=14,
        alpha=0.22,
        color="#2563eb",
        label="Full-GFC ensemble trials",
    )
    ax.scatter(
        credit["annual_vol"] * 100,
        credit["cagr"] * 100,
        s=12,
        alpha=0.14,
        color="#64748b",
        label="Credit tier (2008+)",
    )
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME):
        row = candidate_metrics.loc[name]
        ax.scatter(
            row["annual_vol"] * 100,
            row["cagr"] * 100,
            s=120,
            marker="*",
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            "Typical-20d" if name == FAST_TYPICAL_NAME else "Shortest-worst",
            (row["annual_vol"] * 100, row["cagr"] * 100),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )
    relaxed = candidate_metrics.loc[RELAXED_ROBUST_NAME]
    ax.scatter(
        relaxed["annual_vol"] * 100,
        relaxed["cagr"] * 100,
        s=120,
        marker="*",
        color=COLORS[RELAXED_ROBUST_NAME],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.annotate(
        "5% floor",
        (relaxed["annual_vol"] * 100, relaxed["cagr"] * 100),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9,
    )
    ax.axhline(5, color="#d97706", linestyle=":", linewidth=1.1)
    ax.axhline(9, color="#0f766e", linestyle=":", linewidth=1.1)
    ax.axvline(15, color="#ef4444", linestyle="--", linewidth=1.0)
    ax.set(title="1,504 ensemble trials · return vs volatility", xlabel="Annual volatility (%)", ylabel="CAGR (%)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    charts["scatter"] = _figure_uri(fig)

    eligible = ensemble_metrics[
        ensemble_metrics["relaxed_5pct_gate"]
        & ensemble_metrics["n_5pct_drawdowns"].gt(0)
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    points = ax.scatter(
        eligible["max_5pct_trough_to_recovery_days"],
        eligible["median_5pct_trough_to_recovery_days"],
        c=-eligible["max_drawdown"] * 100,
        cmap="viridis_r",
        s=np.clip(eligible["cagr"] * 500, 18, 80),
        alpha=0.38,
    )
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME):
        row = candidate_metrics.loc[name]
        ax.scatter(
            row["max_5pct_trough_to_recovery_days"],
            row["median_5pct_trough_to_recovery_days"],
            s=125,
            marker="*",
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    ax.axhline(20, color="#ef4444", linestyle="--", linewidth=1.0)
    ax.axvline(20, color="#ef4444", linestyle="--", linewidth=1.0)
    ax.set(
        title="Recovery trade-off among relaxed-gate ensembles",
        xlabel="Worst ≥5% trough-to-recovery (sessions)",
        ylabel="Median ≥5% recovery (sessions)",
    )
    ax.grid(alpha=0.20)
    fig.colorbar(points, ax=ax, label="Maximum drawdown magnitude (%)")
    charts["recovery_scatter"] = _figure_uri(fig)

    thresholds = (0.01, 0.02, 0.03, 0.04, 0.05)
    recovery_rows = []
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME, RELAXED_ROBUST_NAME):
        for threshold in thresholds:
            result = drawdown_duration_metrics(
                common[name],
                material_threshold=threshold,
                recovery_target_days=20,
            )
            recovery_rows.append(
                {
                    "strategy": name,
                    "threshold": threshold * 100,
                    "share": result["share_material_recovered_within_target"],
                    "median": result["median_material_recovery_days"],
                }
            )
    recovery_frame = pd.DataFrame(recovery_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME, RELAXED_ROBUST_NAME):
        subset = recovery_frame[recovery_frame["strategy"] == name]
        axes[0].plot(
            subset["threshold"],
            subset["share"] * 100,
            marker="o",
            label=LABELS[name],
            color=COLORS[name],
        )
        axes[1].plot(
            subset["threshold"],
            subset["median"],
            marker="o",
            label=LABELS[name],
            color=COLORS[name],
        )
    axes[0].axhline(50, color="#64748b", linestyle=":")
    axes[0].set(title="Episodes recovered within 20 sessions", xlabel="Drawdown threshold (%)", ylabel="Share (%)")
    axes[1].axhline(20, color="#ef4444", linestyle="--")
    axes[1].set(title="Median trough-to-recovery", xlabel="Drawdown threshold (%)", ylabel="Sessions")
    for ax in axes:
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    charts["thresholds"] = _figure_uri(fig)

    events = [
        ("GFC", "GFC_return"),
        ("2011", "2011_euro_US_downgrade_return"),
        ("2018 Q4", "2018_Q4_return"),
        ("COVID", "COVID_crash_return"),
    ]
    x = np.arange(len(events))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for position, name in enumerate(
        (FAST_TYPICAL_NAME, FAST_WORST_NAME, RELAXED_ROBUST_NAME)
    ):
        row = candidate_metrics.loc[name]
        values = [float(row[column]) * 100 for _, column in events]
        ax.bar(
            x + (position - 1) * width,
            values,
            width,
            label=LABELS[name],
            color=COLORS[name],
        )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in events])
    ax.set(title="Historical stress-window returns", ylabel="Return (%)")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False, fontsize=8)
    charts["stress"] = _figure_uri(fig)
    return charts


def _appendix_table(frame: pd.DataFrame, columns: list[str]) -> str:
    existing = [column for column in columns if column in frame.columns]
    view = frame[existing].copy()
    percent = {
        "cagr",
        "stress_cagr",
        "annual_vol",
        "max_drawdown",
        "share_5pct_recovered_within_20d",
    }
    integers = {
        "max_underwater_days",
        "max_5pct_trough_to_recovery_days",
        "n_5pct_drawdowns",
    }
    return _table(view, percent_columns=percent, integer_columns=integers)


def _render(
    data: dict[str, object], charts: dict[str, str], candidate_metrics: pd.DataFrame
) -> str:
    original = data["original"]  # type: ignore[assignment]
    recovery = data["recovery"]  # type: ignore[assignment]
    ensembles = data["ensemble"]  # type: ignore[assignment]
    robustness_frame = data["robustness"]  # type: ignore[assignment]
    ensemble_meta = data["ensemble_meta"]  # type: ignore[assignment]

    primary_names = [
        "credit_canary_equity_allocation",
        "QQQ_vol_target_12",
        "cross_asset_mom_126_top3",
        "sector_low_ulcer_positive_trend",
        "sector_downside_trend",
        "SPY_trend_150d",
    ]
    primary = original.set_index("strategy").reindex(primary_names).reset_index()
    primary_columns = [
        "strategy",
        "cagr",
        "annual_vol",
        "max_drawdown",
        "max_underwater_days",
        "max_5pct_trough_to_recovery_days",
        "median_5pct_trough_to_recovery_days",
        "share_5pct_recovered_within_20d",
    ]
    primary_table = _table(
        primary[primary_columns],
        percent_columns={
            "cagr",
            "annual_vol",
            "max_drawdown",
            "share_5pct_recovered_within_20d",
        },
        integer_columns={
            "max_underwater_days",
            "max_5pct_trough_to_recovery_days",
        },
        rename={
            "strategy": "Strategy",
            "cagr": "CAGR",
            "annual_vol": "Vol",
            "max_drawdown": "MaxDD",
            "max_underwater_days": "Max underwater",
            "max_5pct_trough_to_recovery_days": "Worst ≥5% recovery",
            "median_5pct_trough_to_recovery_days": "Median ≥5% recovery",
            "share_5pct_recovered_within_20d": "≤20d share",
        },
    )

    recovery_top = recovery[recovery["objective_gate"]].sort_values(
        ["max_5pct_trough_to_recovery_days", "median_5pct_trough_to_recovery_days"]
    ).head(12)
    recovery_table = _table(
        recovery_top[
            [
                "strategy",
                "cagr",
                "stress_cagr",
                "annual_vol",
                "max_drawdown",
                "max_5pct_trough_to_recovery_days",
                "median_5pct_trough_to_recovery_days",
                "share_5pct_recovered_within_20d",
            ]
        ],
        percent_columns={
            "cagr",
            "stress_cagr",
            "annual_vol",
            "max_drawdown",
            "share_5pct_recovered_within_20d",
        },
        integer_columns={"max_5pct_trough_to_recovery_days"},
        rename={
            "strategy": "Recovery variant",
            "cagr": "CAGR",
            "stress_cagr": "10bp CAGR",
            "annual_vol": "Vol",
            "max_drawdown": "MaxDD",
            "max_5pct_trough_to_recovery_days": "Worst recovery",
            "median_5pct_trough_to_recovery_days": "Median",
            "share_5pct_recovered_within_20d": "≤20d share",
        },
    )

    candidate_rows = []
    for name in (FAST_TYPICAL_NAME, FAST_WORST_NAME, RELAXED_ROBUST_NAME):
        row = candidate_metrics.loc[name]
        candidate_rows.append(
            {
                "strategy": name,
                "cagr": row["cagr"],
                "stress_cagr": row["stress_cagr"],
                "annual_vol": row["annual_vol"],
                "max_drawdown": row["max_drawdown"],
                "max_underwater_days": row["max_underwater_days"],
                "max_5pct_trough_to_recovery_days": row[
                    "max_5pct_trough_to_recovery_days"
                ],
                "median_5pct_trough_to_recovery_days": row[
                    "median_5pct_trough_to_recovery_days"
                ],
                "share_5pct_recovered_within_20d": row[
                    "share_5pct_recovered_within_20d"
                ],
            }
        )
    candidate_table = _table(
        pd.DataFrame(candidate_rows),
        percent_columns={
            "cagr",
            "stress_cagr",
            "annual_vol",
            "max_drawdown",
            "share_5pct_recovered_within_20d",
        },
        integer_columns={
            "max_underwater_days",
            "max_5pct_trough_to_recovery_days",
        },
        rename={
            "strategy": "Portfolio",
            "cagr": "CAGR",
            "stress_cagr": "Stress CAGR",
            "annual_vol": "Vol",
            "max_drawdown": "MaxDD",
            "max_underwater_days": "Max underwater",
            "max_5pct_trough_to_recovery_days": "Worst ≥5% recovery",
            "median_5pct_trough_to_recovery_days": "Median",
            "share_5pct_recovered_within_20d": "≤20d share",
        },
    )

    full_original = ensembles[
        (ensembles["tier"] == "full_gfc_2007_2021") & ensembles["original_gate"]
    ].sort_values(
        ["max_5pct_trough_to_recovery_days", "median_5pct_trough_to_recovery_days"]
    ).head(20)
    full_relaxed = ensembles[
        (ensembles["tier"] == "full_gfc_2007_2021")
        & ensembles["relaxed_5pct_gate"]
        & ensembles["avoided_5pct_drawdown"]
    ].sort_values("cagr", ascending=False).head(20)
    ensemble_columns = [
        "strategy",
        "cagr",
        "stress_cagr",
        "annual_vol",
        "max_drawdown",
        "max_underwater_days",
        "max_5pct_trough_to_recovery_days",
        "median_5pct_trough_to_recovery_days",
        "share_5pct_recovered_within_20d",
        "n_5pct_drawdowns",
    ]
    original_ensemble_table = _appendix_table(full_original, ensemble_columns)
    relaxed_ensemble_table = _appendix_table(full_relaxed, ensemble_columns)

    robust_view = robustness_frame[
        [
            "candidate",
            "frequency",
            "parameter",
            "cagr",
            "stress_cagr",
            "annual_vol",
            "max_drawdown",
            "max_5pct_trough_to_recovery_days",
            "median_5pct_trough_to_recovery_days",
            "share_5pct_recovered_within_20d",
            "n_5pct_drawdowns",
        ]
    ]
    robust_table = _table(
        robust_view,
        percent_columns={
            "cagr",
            "stress_cagr",
            "annual_vol",
            "max_drawdown",
            "share_5pct_recovered_within_20d",
        },
        integer_columns={"max_5pct_trough_to_recovery_days", "n_5pct_drawdowns"},
    )

    appendix_columns = [
        "strategy",
        "evidence",
        "years",
        "cagr",
        "stress_cagr",
        "annual_vol",
        "max_drawdown",
        "max_underwater_days",
        "max_5pct_trough_to_recovery_days",
        "median_5pct_trough_to_recovery_days",
        "share_5pct_recovered_within_20d",
        "n_5pct_drawdowns",
    ]
    original_appendix = _appendix_table(original, appendix_columns)
    recovery_appendix = _appendix_table(recovery, appendix_columns)
    ensemble_appendix = _appendix_table(ensembles, appendix_columns + ["tier"])

    source_links = [
        ("Bridgewater · The All Weather Story", "https://www.bridgewater.com/research-and-insights/the-all-weather-story"),
        ("AQR · A Century of Evidence on Trend Following", "https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing"),
        ("NBER · Volatility Managed Portfolios", "https://www.nber.org/papers/w22208"),
        ("Cboe collar index methodology", "https://cdn.cboe.com/api/global/us_indices/governance/Cboe_Collar_Indices_Methodology.pdf"),
        ("Innovator defined-outcome education", "https://www.innovatoretfs.com/education/"),
    ]
    sources = "".join(
        f'<li><a href="{escape(url)}">{escape(label)}</a></li>'
        for label, url in source_links
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETF Strategy Lab · Complete Pre-2022 Report</title>
<style>
:root{{--ink:#172033;--muted:#5b667a;--line:#d9e0ea;--navy:#16324f;--teal:#0f766e;--blue:#2563eb;--amber:#d97706;--red:#b91c1c;--paper:#fff;--wash:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif;line-height:1.55}}
.page{{max-width:1480px;margin:0 auto;background:var(--paper);min-height:100vh;box-shadow:0 0 45px rgba(30,48,75,.08)}}
header{{padding:54px 64px 42px;background:linear-gradient(135deg,#102a43,#164e63 62%,#0f766e);color:white}}
.eyebrow{{text-transform:uppercase;letter-spacing:.13em;font-size:12px;opacity:.78}} h1{{font-size:40px;line-height:1.14;margin:12px 0 16px;max-width:1050px}}
.subtitle{{font-size:18px;max-width:970px;opacity:.9}} .meta{{display:flex;gap:28px;flex-wrap:wrap;margin-top:28px;font-size:13px;opacity:.86}}
main{{padding:38px 64px 72px}} h2{{font-size:27px;color:var(--navy);border-bottom:1px solid var(--line);padding-bottom:10px;margin:46px 0 20px}}
h3{{font-size:19px;color:#21445f;margin:28px 0 10px}} p{{margin:9px 0}} a{{color:#0b6285}} .lead{{font-size:17px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:22px 0}} .card{{border:1px solid var(--line);border-radius:12px;padding:20px;background:#fff;box-shadow:0 4px 18px rgba(31,51,73,.05)}}
.card h3{{margin:0 0 7px;font-size:17px}} .card .metric{{font-size:29px;font-weight:720;margin:8px 0 2px}} .teal{{border-top:5px solid var(--teal)}} .blue{{border-top:5px solid var(--blue)}} .amber{{border-top:5px solid var(--amber)}}
.note,.warning,.success{{padding:14px 17px;border-radius:8px;margin:16px 0}} .note{{background:#edf5fb;border-left:5px solid #2b6f9f}} .warning{{background:#fff7e6;border-left:5px solid var(--amber)}} .success{{background:#eaf7f2;border-left:5px solid var(--teal)}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}} .pill{{border:1px solid var(--line);border-radius:999px;padding:7px 12px;font-size:13px;background:#fff}}
.chart{{width:100%;border:1px solid var(--line);border-radius:10px;padding:7px;background:white;margin:12px 0 28px}}
.table-wrap{{width:100%;overflow:auto;border:1px solid var(--line);border-radius:9px;margin:12px 0 24px}}
.data-table{{border-collapse:collapse;width:100%;font-size:12px;background:white}} .data-table th,.data-table td{{border-bottom:1px solid #e5eaf1;padding:7px 9px;text-align:right;white-space:nowrap}}
.data-table th{{position:sticky;top:0;background:#eaf0f7;color:#243d55;z-index:1}} .data-table th:first-child,.data-table td:first-child{{text-align:left;position:sticky;left:0;background:inherit;max-width:560px;white-space:normal}}
.data-table tr:nth-child(even){{background:#f8fafc}} .data-table tr:hover{{background:#eef6f8}} details{{border:1px solid var(--line);border-radius:9px;padding:12px 15px;margin:13px 0;background:#fbfcfe}} summary{{cursor:pointer;font-weight:650;color:#264a65}}
code{{background:#edf1f5;padding:2px 5px;border-radius:4px;font-size:.92em}} ul{{padding-left:22px}} .small{{font-size:12px;color:var(--muted)}} footer{{padding:25px 64px 42px;color:var(--muted);border-top:1px solid var(--line);font-size:12px}}
@media(max-width:900px){{header,main,footer{{padding-left:24px;padding-right:24px}}.grid{{grid-template-columns:1fr}}h1{{font-size:31px}}}}
@media print{{body{{background:white}}.page{{box-shadow:none}}details{{break-inside:avoid}}header{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}
</style></head><body><div class="page">
<header><div class="eyebrow">Alpha Lab · ETF-only research</div>
<h1>长期回报、低波动与回撤恢复：完整 Pre-2022 策略报告</h1>
<div class="subtitle">从 104 个基础 ETF/期权方案，到 100 个恢复导向变体，再到 1,504 个策略组合。核心问题：能否在约 10% 回报与 15% 以下波动下控制回撤，并尽量在 20 个交易日内恢复？</div>
<div class="meta"><span>样本边界：2007-01 至 2021-12-31</span><span>下载截止：2022-01-01 exclusive</span><span>总试验行：1,708</span><span>ETF-only；期权仅 SPY/QQQ</span></div></header>
<main>
<section><h2>Executive decision</h2>
<p class="lead"><b>组合显著改善了可行前沿，但没有任何方案能诚实保证最坏回撤在 20 个交易日内恢复。</b>如果 20 日指“典型/中位回撤”，一个简单 50/50 组合已经达到；如果允许把历史 CAGR 门槛降到 5%，可以把 MaxDD 压到约 -4.3%，从而在样本内避免 ≥5% 的实质性回撤。</p>
<div class="grid">
<div class="card teal"><h3>首选 · 约10%回报 / 典型20日恢复</h3><div class="metric">11.31% CAGR</div><p>50% 跨资产动量 + 50% 行业杠铃</p><p><b>11.30%</b> 波动 · <b>-16.22%</b> MaxDD</p><p>≥5% 回撤中位恢复 <b>18日</b>；55% 在20日内恢复；最坏仍为266日。</p></div>
<div class="card blue"><h3>备选 · 缩短最坏恢复</h3><div class="metric">116 days</div><p>50% 固定 All Weather + 50% 行业杠铃</p><p><b>10.24%</b> CAGR · <b>8.90%</b> 波动</p><p>最坏恢复从单策略218日降至116日，但中位恢复56日。</p></div>
<div class="card amber"><h3>资本保全 · 放宽到5%</h3><div class="metric">-4.32% MaxDD</div><p>45% 风险袖套 + 55% SHY</p><p><b>5.21%</b> CAGR · <b>3.48%</b> 波动</p><p>样本内没有≥5%回撤；小于5%的浮亏仍可能长期水下。</p></div></div>
<div class="warning"><b>关键区别：</b>回撤深度与恢复时间不是同一目标。低风险组合可以亏得很少，却因为回报速度较慢而长时间略低于旧高；止损也可能减少深度但延长修复。</div>
{candidate_table}
<img class="chart" src="{charts['equity']}" alt="Candidate equity curves">
<img class="chart" src="{charts['drawdown']}" alt="Candidate drawdowns">
</section>

<section><h2>1 · Research design and audit boundary</h2>
<div class="kpis"><span class="pill">104 原始策略</span><span class="pill">100 恢复变体</span><span class="pill">1,504 组合</span><span class="pill">5 bp 主成本</span><span class="pill">10/20 bp 压力成本</span><span class="pill">周度/月度决策</span></div>
<p>所有动态目标在决策收盘形成，并在下一可交易收盘成交；持仓从随后一个 close-to-close 区间才开始赚取收益。SHY 是显式 ETF 安全袖套，不使用免费现金。组合层采用真实持仓漂移与月度再平衡，内部策略成本之外再加元组合成本。</p>
<ul><li><b>原始目标：</b>CAGR ≥9%（接近10%目标）、压力 CAGR ≥8.5%、波动 ≤15%、MaxDD ≥-25%。</li><li><b>放宽目标：</b>CAGR ≥5%、压力 CAGR ≥4.5%，风险门槛不变。</li><li><b>实质回撤：</b>净值从前高下跌至少5%；恢复天数从谷底计至重新达到旧高。</li><li><b>未恢复事件：</b>在2021-12-31截断并视为未通过20日测试。</li></ul>
<div class="success"><b>Leakage verdict:</b> 组合脚本自动扫描 0 blocker / 0 warning。全部窗口为 trailing，未使用未来谷底、未来恢复日、负向 shift、居中窗口、向后填充或全样本信号归一化。快速组合引擎已与仓库标准漂移引擎逐日对账。</div>
</section>

<section><h2>2 · Original 104-strategy sweep</h2>
<p>基础批次覆盖固定股债/黄金/商品配置、SPY/QQQ 趋势与波动控制、行业配置、跨资产动量、信用金丝雀、All Weather、合成 SPY put/spread/collar，以及实际策略 ETF。72 行拥有长期规则历史；只有7行通过冻结门槛，且其中3行是同一家族的 SPY 均线邻居。</p>
{primary_table}
<p>信用金丝雀拥有最小 MaxDD，但其 HYG/IEF 信号直到2008年才完成200日预热，不能冒充完整 GFC 证据。QQQ 波动目标、行业规则与跨资产动量可以达到回报目标，却没有解决恢复尾部。</p>
</section>

<section><h2>3 · Drawdown-duration audit and 100 recovery variants</h2>
<p>旧字段 <code>max_recovery_days</code> 实际测量完整水下期。本研究将其拆成：前高至修复的总水下天数，以及谷底至旧高的真正恢复腿。100个新增变体包括更快的 SPY/QQQ 趋势、波动目标和防御/周期行业杠铃。</p>
<div class="note"><b>结果：</b>100个新变体中21个保留原收益/风险门槛；0个满足“每个≥5%回撤都在20日内恢复”；0个门槛通过者的中位恢复≤20日；0个门槛通过者有过半实质回撤在20日内修复。</div>
{recovery_table}
<p>最接近的单策略恢复方案，其最坏恢复仍约218–228日，中位约22–25日。长尾不仅来自GFC，也来自2010、2015–2016与2018年的缓慢修复。</p>
</section>

<section><h2>4 · Combining sleeves: 1,504-trial result</h2>
<p>组合池冻结为8个机制袖套，并将信用金丝雀放在单独的2008+证据层。枚举2–5袖套等权组合、25%/50%/75%/100%风险预算，以及少量预先限定的逆波动诊断。</p>
<div class="kpis"><span class="pill">{ensemble_meta['original_gate_count']} 原门槛通过</span><span class="pill">{ensemble_meta['relaxed_5pct_gate_count']} 放宽门槛通过</span><span class="pill">{ensemble_meta['avoided_5pct_drawdown_count']} 未出现-5%回撤</span><span class="pill">0 最坏20日恢复</span><span class="pill">64 放宽门槛+避免实质回撤</span></div>
<img class="chart" src="{charts['scatter']}" alt="Ensemble return volatility scatter">
<img class="chart" src="{charts['recovery_scatter']}" alt="Recovery frontier">
<h3>完整 GFC 样本 · 原回报门槛的较短恢复组合</h3><div class="table-wrap">{original_ensemble_table}</div>
<h3>完整 GFC 样本 · 5%门槛且未触及-5%回撤</h3><div class="table-wrap">{relaxed_ensemble_table}</div>
</section>

<section><h2>5 · Candidate A: typical recovery near 20 sessions</h2>
<p><b>冻结规则：</b>50% <code>cross_asset_mom_126_top3</code> + 50% <code>recovery_sector_barbell_M_cyc65_mom63_ma100_top2</code>，月度再平衡，下一收盘成交。</p>
<p>它是“尽量20日恢复”最直接的答案：≥5%回撤的中位恢复18日，20次实质回撤中11次在20日内修复。CAGR 11.31%、波动11.30%、MaxDD -16.22%。2007–2012 CAGR 11.23%，2013–2021 CAGR 11.36%，时期分裂较小。</p>
<div class="warning">最坏恢复仍为266日，最长完整水下期502日。因此这是“典型恢复目标”，不是20日硬承诺。</div>
</section>

<section><h2>6 · Candidate B: minimize the worst historical recovery</h2>
<p><b>冻结规则：</b>50% 固定零售 All Weather + 50% 恢复行业杠铃。它以10.24% CAGR、8.90%波动和-17.17% MaxDD，把最坏实质恢复从单策略约218日缩短到116日。</p>
<p>代价是典型恢复变慢：中位56日，仅11.1%的实质回撤在20日内修复。这个组合适合关心最坏历史尾部，而不是关心多数回撤体验的投资者。</p>
</section>

<section><h2>7 · Candidate C: lower the return floor to 5%</h2>
<p><b>稳健权重：</b>55% SHY；其余45%等分给 QQQ 12%波动目标、行业低-Ulcer、固定 All Weather 与趋势过滤 All Weather，每个袖套11.25%。这些是策略预算，最终 ETF 权重需要把四个袖套的动态目标线性合并并净额成交。</p>
<p>结果为5.21% CAGR、3.48%波动、-4.32% MaxDD；20 bp 元组合成本下 CAGR 5.18%。GFC窗口 +1.57%，2018Q4 -3.68%，COVID窗口 -2.59%，最差日历年 -0.54%。</p>
<div class="note">它在样本内从未出现≥5%回撤；但最大水下期仍有299日。对≥1%回撤，中位恢复18.5日、63.2%在20日内修复、最坏134日。45%风险预算有余量；55%时一旦MaxDD越过-5%，最坏恢复立即跳至约200日。</div>
<img class="chart" src="{charts['thresholds']}" alt="Recovery by drawdown threshold">
</section>

<section><h2>8 · Stress regimes</h2>
<img class="chart" src="{charts['stress']}" alt="Stress returns">
<ul><li>跨资产动量+行业杠铃在GFC接近持平，但在2018Q4损失约14%，说明慢趋势并非每种冲击都有效。</li><li>All Weather+行业杠铃的压力表现较均衡，但典型修复较慢。</li><li>5%方案主要通过55% SHY和低风险袖套压缩损失；未见2022式股债共同下跌环境，不能外推为保本。</li></ul>
</section>

<section><h2>9 · Robustness and implementation</h2>
<p>首选10%组合在跨资产动量40%–50%、行业杠铃50%–60%的邻域内，CAGR 11.31%–11.46%、波动11.30%–11.49%、MaxDD约-15.6%至-16.2%，中位恢复18–19日。季度元再平衡结果几乎相同。最坏恢复方案在All Weather权重40%–60%时保持约116–120日的最坏恢复。</p>
<details><summary>查看30行权重、频率与20bp成本稳健性表</summary><div class="table-wrap">{robust_table}</div></details>
<p>组合收益按虚拟子账户计算：各袖套先扣除自身交易成本，组合层再扣5bp。真实执行应聚合重叠ETF订单，可能减少成本，但也需要处理同一ETF在不同袖套中的相反调仓。合成 collar 只使用模型期权价格，未使用历史NBBO；最终三套首选组合均不依赖该期权模型。</p>
</section>

<section><h2>10 · What this report does not prove</h2>
<ul><li>历史 CAGR 不是未来 expected return；1,708行试验带来显著选择风险。</li><li>ETF宇宙具有存续与产品发行偏差；短历史策略ETF不能与长期规则等价排名。</li><li>样本只包含一次GFC与一次COVID冲击；事件数量有限。</li><li>严格遵守用户要求，没有查看2022及之后数据，因此没有检验后续股债共同通胀冲击。</li><li>5%方案不是本金保证。它只是pre-2022样本中的低风险结果。</li><li>任何硬性20日恢复保证，都需要不诚实地重置高水位，或使用可能破坏波动/回撤预算的杠杆与期权凸性。</li></ul>
<div class="warning"><b>Promotion gate:</b> 在读取2022+之前冻结一套精确规则与ETF聚合方法。只有在用户明确释放样本外数据后，才可评价是否从研究候选升级为资本配置。</div>
</section>

<section><h2>11 · Full trial appendices</h2>
<p>以下折叠表嵌入报告本身，远端打开无需本地CSV。为控制文件体积，仅保留排名与回撤判断所需的核心字段。</p>
<details><summary>Appendix A · 全部104个原始方案</summary><div class="table-wrap">{original_appendix}</div></details>
<details><summary>Appendix B · 全部100个恢复导向变体</summary><div class="table-wrap">{recovery_appendix}</div></details>
<details><summary>Appendix C · 全部1,504个组合</summary><div class="table-wrap">{ensemble_appendix}</div></details>
</section>

<section><h2>12 · Research sources</h2><ul>{sources}</ul>
<p class="small">Sources informed hypotheses only. Every numerical result in this report comes from the locally frozen pre-2022 study artifacts.</p></section>
</main>
<footer>Research only · not investment advice · generated from alpha_lab frozen artifacts · final observation 2021-12-31 · self-contained HTML</footer>
</div></body></html>"""


def main() -> None:
    data = _load()
    candidate_returns = _candidate_returns()
    candidate_metrics = _candidate_metrics(data)
    charts = _build_charts(data, candidate_returns, candidate_metrics)
    html = _render(data, charts, candidate_metrics)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Report: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size:,}")


if __name__ == "__main__":
    main()
