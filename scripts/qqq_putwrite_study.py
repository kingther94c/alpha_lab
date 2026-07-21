"""Research BOXX-like cash collateral plus cash-secured QQQ puts.

The script downloads public daily proxies, calibrates the option-pricing proxy
against the Cboe S&P 500 PutWrite Index on an early window, selects a QQQ
policy using discovery/validation periods only, evaluates a sealed holdout,
and renders a self-contained bilingual HTML report.

Generated files are intentionally written under ``data/results`` and
``reports`` (both gitignored).  Reusable strategy logic lives in
``alpha_lab.backtest.put_write``.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from dataclasses import asdict, replace
from html import escape
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from alpha_lab.backtest.metrics import summary  # noqa: E402
from alpha_lab.backtest.put_write import (  # noqa: E402
    PutWriteBacktestResult,
    PutWriteConfig,
    build_put_write_advice,
    run_cash_secured_put_backtest,
)
from alpha_lab.data.loaders.fred import discount_rate_to_daily_rate, load_series  # noqa: E402

START = "2007-01-03"
END = "2026-07-01"  # yfinance end is exclusive; freezes the study at 2026-06-30.
PERIODS = 252
CASH_FEE_BPS = 19.49  # BOXX net expense ratio shown by the sponsor on 2026-07-01.
OUT = ROOT / "data" / "results" / "qqq_putwrite_cash_yield"
REPORT = ROOT / "reports" / "qqq_putwrite_cash_yield.html"
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "discovery": ("2007-01-03", "2016-12-30"),
    "validation": ("2017-01-03", "2021-12-31"),
    "holdout": ("2022-01-03", "2026-06-30"),
}

COLORS = {
    "selected": "#0f766e",
    "cash": "#2563eb",
    "qqq": "#9ca3af",
    "put": "#7c3aed",
    "danger": "#dc2626",
}


def download_market_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted market histories plus the FRED T-bill rate."""
    tickers = ["QQQ", "SPY", "^VXN", "^VIX", "^PUT", "BOXX", "SGOV"]
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no market data")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.sort_index().dropna(how="all")
    rates = load_series("DTB3", start="2006-12-01", end=END)["DTB3"]
    return prices, rates


def cash_series_on_index(
    discount_rate: pd.Series,
    index: pd.DatetimeIndex,
    *,
    fee_bps: float = CASH_FEE_BPS,
) -> tuple[pd.Series, pd.Series]:
    """Build lagged annual rates and a net daily BOXX-like cash return."""
    daily_rate = discount_rate_to_daily_rate(discount_rate).reindex(index).ffill().shift(1)
    annual_rate = (pd.to_numeric(discount_rate, errors="coerce") / 100.0)
    annual_rate = annual_rate.reindex(index).ffill().shift(1)
    day_gap = index.to_series().diff().dt.days.fillna(0.0)
    gross = (1.0 + daily_rate) ** day_gap - 1.0
    fee_factor = (1.0 - fee_bps / 10_000.0) ** (day_gap / 365.25)
    net = ((1.0 + gross) * fee_factor - 1.0).rename("cash_proxy")
    return annual_rate.rename("annual_rate"), net


def period_slice(series: pd.Series, period: str) -> pd.Series:
    """Slice one named research split."""
    start, end = SPLITS[period]
    return series.loc[start:end]


def stats_row(returns: pd.Series, cash_returns: pd.Series) -> dict[str, float]:
    """Return total metrics and CAGR above the cash hurdle."""
    aligned = pd.concat([returns.rename("strategy"), cash_returns.rename("cash")], axis=1).dropna()
    strategy_stats = summary(aligned["strategy"], periods=PERIODS)
    cash_stats = summary(aligned["cash"], periods=PERIODS)
    excess = aligned["strategy"] - aligned["cash"]
    excess_vol = float(excess.std(ddof=1) * np.sqrt(PERIODS))
    excess_sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(PERIODS))
        if excess.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "CAGR": strategy_stats["CAGR"],
        "AnnVol": strategy_stats["AnnVol"],
        "Sharpe": strategy_stats["Sharpe"],
        "MaxDD": strategy_stats["MaxDD"],
        "Calmar": strategy_stats["Calmar"],
        "CashCAGR": cash_stats["CAGR"],
        "ExcessCAGR": strategy_stats["CAGR"] - cash_stats["CAGR"],
        "ExcessVol": excess_vol,
        "ExcessSharpe": excess_sharpe,
    }


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Compound daily returns to month-end."""
    return ((1.0 + returns).resample("ME").prod() - 1.0).dropna()


def calibrate_proxy(
    prices: pd.DataFrame,
    annual_rate: pd.Series,
    cash_returns: pd.Series,
) -> tuple[float, float, pd.DataFrame, pd.Series]:
    """Calibrate IV multiplier and entry haircut on 2007-2014 only.

    The target is the monthly return RMSE versus the published Cboe PUT total
    return index.  This is a coarse model sanity check, not QQQ option data.
    """
    spy = prices["SPY"].dropna()
    vix = (prices["^VIX"] / 100.0).reindex(spy.index).ffill()
    put_index = prices["^PUT"].dropna()
    actual_returns = put_index.pct_change().fillna(0.0)
    rows: list[dict[str, float]] = []
    synthetic: dict[tuple[float, float], pd.Series] = {}

    for iv_multiplier, spread in product(
        [0.85, 0.925, 1.00, 1.075, 1.15],
        [0.00, 0.05, 0.10],
    ):
        config = PutWriteConfig(
            target_delta=0.49,
            tenor_trading_days=21,
            collateral_fraction=1.0,
            max_assignment_days=0,
            iv_multiplier=iv_multiplier,
            entry_spread_fraction=spread,
            commission_per_contract=0.0,
            stock_exit_cost_bps=0.0,
            strike_increment=1.0,
        )
        result = run_cash_secured_put_backtest(
            spy,
            vix,
            cash_returns.reindex(spy.index),
            annual_rate.reindex(spy.index),
            config=config,
        )
        key = (iv_multiplier, spread)
        synthetic[key] = result.returns
        aligned = pd.concat(
            [
                monthly_returns(result.returns).rename("synthetic"),
                monthly_returns(actual_returns).rename("actual"),
            ],
            axis=1,
        ).dropna()
        discovery = aligned.loc[:"2014-12-31"]
        validation = aligned.loc["2015-01-01":]
        rows.append(
            {
                "iv_multiplier": iv_multiplier,
                "spread_fraction": spread,
                "discovery_rmse": float(
                    np.sqrt(np.mean((discovery["synthetic"] - discovery["actual"]) ** 2))
                ),
                "discovery_corr": float(discovery.corr().iloc[0, 1]),
                "validation_rmse": float(
                    np.sqrt(np.mean((validation["synthetic"] - validation["actual"]) ** 2))
                ),
                "validation_corr": float(validation.corr().iloc[0, 1]),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["discovery_rmse", "discovery_corr"], ascending=[True, False]
    )
    chosen = table.iloc[0]
    key = (float(chosen["iv_multiplier"]), float(chosen["spread_fraction"]))
    return key[0], key[1], table, synthetic[key]


def run_policy_grid(
    qqq: pd.Series,
    vxn: pd.Series,
    cash_returns: pd.Series,
    annual_rate: pd.Series,
    *,
    iv_multiplier: float,
    spread_fraction: float,
) -> tuple[pd.DataFrame, dict[int, PutWriteConfig]]:
    """Run a prespecified parameter grid and score only pre-holdout splits."""
    rows: list[dict[str, float | int | bool]] = []
    configs: dict[int, PutWriteConfig] = {}
    grid = product(
        [0.10, 0.15, 0.20, 0.25],
        [21, 31, 42],
        [0.25, 0.50, 0.75, 1.00],
        [21, 63, 126],
        [None, 200],
    )
    for config_id, (delta, tenor, collateral, hold_days, trend) in enumerate(grid):
        config = PutWriteConfig(
            target_delta=delta,
            tenor_trading_days=tenor,
            collateral_fraction=collateral,
            max_assignment_days=hold_days,
            iv_multiplier=iv_multiplier,
            entry_spread_fraction=spread_fraction,
            commission_per_contract=0.65,
            stock_exit_cost_bps=2.0,
            strike_increment=1.0,
            trend_lookback=trend,
        )
        result = run_cash_secured_put_backtest(
            qqq,
            vxn,
            cash_returns,
            annual_rate,
            config=config,
        )
        row: dict[str, float | int | bool] = {
            "config_id": config_id,
            "delta": delta,
            "tenor_days": tenor,
            "collateral": collateral,
            "hold_days": hold_days,
            "trend_200d": trend is not None,
            "assignments": int(
                (result.events.get("event", pd.Series(dtype=str)) == "assignment").sum()
            ),
            "stock_time": float((result.state == "assigned_stock").mean()),
        }
        for period in SPLITS:
            metrics = stats_row(
                period_slice(result.returns, period),
                period_slice(cash_returns, period),
            )
            for name, value in metrics.items():
                row[f"{period}_{name}"] = value
        row["worst_pre_holdout_excess"] = min(
            float(row["discovery_ExcessCAGR"]),
            float(row["validation_ExcessCAGR"]),
        )
        row["worst_pre_holdout_drawdown"] = min(
            float(row["discovery_MaxDD"]),
            float(row["validation_MaxDD"]),
        )
        row["eligible"] = bool(
            row["worst_pre_holdout_excess"] > 0.0
            and row["worst_pre_holdout_drawdown"] >= -0.20
        )
        configs[config_id] = config
        rows.append(row)

    table = pd.DataFrame(rows)
    eligible = table[table["eligible"]]
    if eligible.empty:
        table["selection_score"] = (
            table["worst_pre_holdout_excess"]
            + 0.25 * table["validation_ExcessSharpe"]
            + 0.25 * table["worst_pre_holdout_drawdown"]
        )
    else:
        table["selection_score"] = table["worst_pre_holdout_excess"].where(
            table["eligible"], -np.inf
        )
    table = table.sort_values(
        ["selection_score", "validation_ExcessSharpe", "collateral"],
        ascending=[False, False, True],
    )
    return table, configs



def comparison_policy_specs(
    base: PutWriteConfig,
) -> list[tuple[str, str, str, PutWriteConfig]]:
    """Return prespecified entry and post-assignment policy variants."""
    specs: list[tuple[str, str, str, PutWriteConfig]] = [
        (
            "baseline",
            "Baseline",
            "Always sell; exit QQQ at breakeven or 126d",
            base,
        )
    ]
    entry_specs = [
        ("iv_abs_20", "VXN ≥ 20%", dict(entry_min_iv=0.20)),
        ("iv_abs_25", "VXN ≥ 25%", dict(entry_min_iv=0.25)),
        ("iv_abs_30", "VXN ≥ 30%", dict(entry_min_iv=0.30)),
        (
            "iv_pct_50",
            "VXN ≥ trailing 50th pct",
            dict(entry_min_iv_percentile=0.50),
        ),
        (
            "iv_pct_70",
            "VXN ≥ trailing 70th pct",
            dict(entry_min_iv_percentile=0.70),
        ),
        (
            "iv_pct_85",
            "VXN ≥ trailing 85th pct",
            dict(entry_min_iv_percentile=0.85),
        ),
        ("iv_rv_0", "VXN − 21d RV ≥ 0pp", dict(entry_min_iv_rv_spread=0.00)),
        ("iv_rv_5", "VXN − 21d RV ≥ 5pp", dict(entry_min_iv_rv_spread=0.05)),
        ("iv_rv_10", "VXN − 21d RV ≥ 10pp", dict(entry_min_iv_rv_spread=0.10)),
        (
            "iv_pct70_rv0",
            "70th pct + VXN−RV ≥ 0",
            dict(entry_min_iv_percentile=0.70, entry_min_iv_rv_spread=0.00),
        ),
    ]
    specs.extend(
        (policy_id, "Entry filter", label, replace(base, **changes))
        for policy_id, label, changes in entry_specs
    )

    exit_specs = [
        ("price_be_21", "Breakeven OR 21d", dict(max_assignment_days=21)),
        ("price_be_63", "Breakeven OR 63d", dict(max_assignment_days=63)),
        (
            "fixed_21",
            "Fixed hold 21d",
            dict(max_assignment_days=21, stock_exit_mode="timeout_only"),
        ),
        (
            "fixed_63",
            "Fixed hold 63d",
            dict(max_assignment_days=63, stock_exit_mode="timeout_only"),
        ),
        (
            "fixed_126",
            "Fixed hold 126d",
            dict(max_assignment_days=126, stock_exit_mode="timeout_only"),
        ),
        (
            "price_strike_63",
            "Strike target OR 63d",
            dict(max_assignment_days=63, recovery_rule="strike"),
        ),
        (
            "price_strike_126",
            "Strike target OR 126d",
            dict(max_assignment_days=126, recovery_rule="strike"),
        ),
        (
            "price_be_plus5_126",
            "Breakeven +5% OR 126d",
            dict(max_assignment_days=126, stock_target_return=0.05),
        ),
        (
            "iv_exit_20",
            "VXN ≤ 20% OR 126d",
            dict(
                max_assignment_days=126,
                stock_exit_mode="iv_or_timeout",
                stock_exit_iv_max=0.20,
            ),
        ),
        (
            "iv_exit_25",
            "VXN ≤ 25% OR 126d",
            dict(
                max_assignment_days=126,
                stock_exit_mode="iv_or_timeout",
                stock_exit_iv_max=0.25,
            ),
        ),
        (
            "iv_exit_30",
            "VXN ≤ 30% OR 126d",
            dict(
                max_assignment_days=126,
                stock_exit_mode="iv_or_timeout",
                stock_exit_iv_max=0.30,
            ),
        ),
        (
            "price_or_iv25",
            "Breakeven OR VXN ≤25% OR 126d",
            dict(
                max_assignment_days=126,
                stock_exit_mode="price_or_iv_or_timeout",
                stock_exit_iv_max=0.25,
            ),
        ),
        (
            "price_and_iv25",
            "Breakeven AND VXN ≤25% OR 126d",
            dict(
                max_assignment_days=126,
                stock_exit_mode="price_and_iv_or_timeout",
                stock_exit_iv_max=0.25,
            ),
        ),
    ]
    specs.extend(
        (policy_id, "Assignment exit", label, replace(base, **changes))
        for policy_id, label, changes in exit_specs
    )

    combined_specs = [
        (
            "combo_iv20_or25",
            "Enter VXN≥20%; exit BE or VXN≤25%",
            dict(
                entry_min_iv=0.20,
                stock_exit_mode="price_or_iv_or_timeout",
                stock_exit_iv_max=0.25,
            ),
        ),
        (
            "combo_pct70_or25",
            "Enter top-30% VXN; exit BE or VXN≤25%",
            dict(
                entry_min_iv_percentile=0.70,
                stock_exit_mode="price_or_iv_or_timeout",
                stock_exit_iv_max=0.25,
            ),
        ),
        (
            "combo_spread5_or25",
            "Enter VXN−RV≥5pp; exit BE or VXN≤25%",
            dict(
                entry_min_iv_rv_spread=0.05,
                stock_exit_mode="price_or_iv_or_timeout",
                stock_exit_iv_max=0.25,
            ),
        ),
        (
            "combo_pct70_be",
            "Enter top-30% VXN; exit BE or 126d",
            dict(entry_min_iv_percentile=0.70),
        ),
    ]
    specs.extend(
        (policy_id, "Combined", label, replace(base, **changes))
        for policy_id, label, changes in combined_specs
    )
    return specs


def run_policy_comparison(
    qqq: pd.Series,
    vxn: pd.Series,
    cash_returns: pd.Series,
    annual_rate: pd.Series,
    base: PutWriteConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, PutWriteBacktestResult],
    dict[str, PutWriteConfig],
]:
    """Compare entry/assignment policies without selecting on the holdout."""
    rows: list[dict[str, float | int | str | bool]] = []
    results: dict[str, PutWriteBacktestResult] = {}
    configs: dict[str, PutWriteConfig] = {}
    years = (qqq.index[-1] - qqq.index[0]).days / 365.25

    for policy_id, family, label, config in comparison_policy_specs(base):
        result = run_cash_secured_put_backtest(
            qqq,
            vxn,
            cash_returns,
            annual_rate,
            config=config,
        )
        events = result.events
        event_names = events.get("event", pd.Series(dtype=str))
        entries = int((event_names == "put_entry").sum())
        assignments = int((event_names == "assignment").sum())
        exits = events[event_names == "stock_exit"]
        row: dict[str, float | int | str | bool] = {
            "policy_id": policy_id,
            "family": family,
            "policy": label,
            "entries": entries,
            "entries_per_year": entries / years,
            "assignments": assignments,
            "assignment_rate": assignments / entries if entries else float("nan"),
            "stock_time": float((result.state == "assigned_stock").mean()),
            "mean_stock_days": (
                float(exits["stock_days"].mean()) if not exits.empty else 0.0
            ),
            "timeout_share": (
                float((exits["reason"] == "timeout").mean())
                if not exits.empty
                else 0.0
            ),
        }
        for period in SPLITS:
            metrics = stats_row(
                period_slice(result.returns, period),
                period_slice(cash_returns, period),
            )
            for name, value in metrics.items():
                row[f"{period}_{name}"] = value
        row["worst_pre_holdout_excess"] = min(
            float(row["discovery_ExcessCAGR"]),
            float(row["validation_ExcessCAGR"]),
        )
        row["worst_pre_holdout_drawdown"] = min(
            float(row["discovery_MaxDD"]),
            float(row["validation_MaxDD"]),
        )
        row["eligible_pre_holdout"] = bool(
            row["worst_pre_holdout_excess"] > 0.0
            and row["worst_pre_holdout_drawdown"] >= -0.20
        )
        rows.append(row)
        results[policy_id] = result
        configs[policy_id] = config

    table = pd.DataFrame(rows)
    baseline = table.loc[table["policy_id"] == "baseline"].iloc[0]
    for period in SPLITS:
        table[f"{period}_vs_baseline"] = (
            table[f"{period}_ExcessCAGR"] - baseline[f"{period}_ExcessCAGR"]
        )
    table["selection_score"] = table["worst_pre_holdout_excess"].where(
        table["eligible_pre_holdout"], -np.inf
    )
    table["pre_holdout_rank"] = (
        table["worst_pre_holdout_excess"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    table = table.sort_values(
        ["selection_score", "validation_ExcessSharpe", "entries_per_year"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return table, results, configs

def robustness_table(

    qqq: pd.Series,
    vxn: pd.Series,
    cash_returns: pd.Series,
    annual_rate: pd.Series,
    selected: PutWriteConfig,
) -> pd.DataFrame:
    """Stress IV mapping and option execution haircut around the selection."""
    rows: list[dict[str, float | str]] = []
    scenarios = [
        ("base", selected.iv_multiplier, selected.entry_spread_fraction),
        ("IV -10%", selected.iv_multiplier * 0.90, selected.entry_spread_fraction),
        ("IV +10%", selected.iv_multiplier * 1.10, selected.entry_spread_fraction),
        (
            "2x spread haircut",
            selected.iv_multiplier,
            min(selected.entry_spread_fraction * 2.0, 0.25),
        ),
        ("10% entry haircut", selected.iv_multiplier, 0.10),
    ]
    for label, iv_multiplier, spread in scenarios:
        config = replace(
            selected,
            iv_multiplier=iv_multiplier,
            entry_spread_fraction=spread,
        )
        result = run_cash_secured_put_backtest(
            qqq,
            vxn,
            cash_returns,
            annual_rate,
            config=config,
        )
        for period in ["validation", "holdout"]:
            metrics = stats_row(
                period_slice(result.returns, period),
                period_slice(cash_returns, period),
            )
            rows.append(
                {
                    "scenario": label,
                    "period": period,
                    "CAGR": metrics["CAGR"],
                    "ExcessCAGR": metrics["ExcessCAGR"],
                    "Sharpe": metrics["Sharpe"],
                    "MaxDD": metrics["MaxDD"],
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_excess(
    excess_returns: pd.Series,
    *,
    block: int = 21,
    samples: int = 2_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a 95% CI for annualized arithmetic excess return."""
    values = excess_returns.dropna().to_numpy()
    if len(values) < block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(values) - block + 1)
    estimates = np.empty(samples)
    blocks_needed = int(np.ceil(len(values) / block))
    for sample in range(samples):
        selected_starts = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate([values[start : start + block] for start in selected_starts])
        estimates[sample] = draw[: len(values)].mean() * PERIODS
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def fig_b64(fig: plt.Figure) -> str:
    """Encode one Matplotlib figure for a self-contained report."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def wealth(returns: pd.Series) -> pd.Series:
    """Return a wealth index starting at one."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Return drawdown from the running wealth peak."""
    curve = wealth(returns)
    return curve / curve.cummax() - 1.0


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def html_table(
    frame: pd.DataFrame,
    *,
    percent_columns: set[str] | None = None,
    digits: int = 2,
) -> str:
    """Render a compact HTML table with basic numeric formatting."""
    percent_columns = percent_columns or set()
    header = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                text = "—"
            elif column in percent_columns:
                text = fmt_pct(float(value))
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.{digits}f}"
            else:
                text = escape(str(value))
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_charts(
    selected_returns: pd.Series,
    cash_returns: pd.Series,
    qqq_returns: pd.Series,
    grid: pd.DataFrame,
    boxx: pd.Series,
    put_actual_returns: pd.Series,
    put_synthetic_returns: pd.Series,
    policy_comparison: pd.DataFrame,
    policy_results: dict[str, PutWriteBacktestResult],
) -> dict[str, str]:
    """Render all report charts."""
    charts: dict[str, str] = {}
    aligned = pd.concat(
        [
            selected_returns.rename("selected"),
            cash_returns.rename("cash"),
            qqq_returns.rename("qqq"),
        ],
        axis=1,
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 4.3))
    for column, label, color, width in [
        ("selected", "Selected put-write + cash", COLORS["selected"], 2.2),
        ("cash", "BOXX-like cash proxy", COLORS["cash"], 1.6),
        ("qqq", "QQQ buy & hold", COLORS["qqq"], 1.4),
    ]:
        ax.plot(wealth(aligned[column]), label=label, color=color, lw=width)
    ax.set_yscale("log")
    ax.set_ylabel("wealth (log scale)")
    ax.set_title("Full-history wealth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    charts["wealth"] = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(
        aligned.index,
        drawdown(aligned["selected"]) * 100,
        0,
        color=COLORS["selected"],
        alpha=0.75,
        label="Selected",
    )
    ax.plot(
        aligned.index,
        drawdown(aligned["qqq"]) * 100,
        color=COLORS["qqq"],
        lw=1.0,
        label="QQQ",
    )
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Marked-to-model drawdown")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    charts["drawdown"] = fig_b64(fig)

    holdout = aligned.loc[SPLITS["holdout"][0] : SPLITS["holdout"][1]]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for column, label, color in [
        ("selected", "Selected (sealed holdout)", COLORS["selected"]),
        ("cash", "Cash proxy", COLORS["cash"]),
        ("qqq", "QQQ", COLORS["qqq"]),
    ]:
        ax.plot(wealth(holdout[column]), label=label, color=color, lw=2.0 if column == "selected" else 1.3)
    ax.set_title("2022-2026 sealed holdout")
    ax.set_ylabel("wealth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    charts["holdout"] = fig_b64(fig)

    surface = (
        grid.groupby(["delta", "collateral"])["worst_pre_holdout_excess"]
        .max()
        .unstack("collateral")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    image = ax.imshow(surface.values * 100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(surface.columns)), [f"{x:.0%}" for x in surface.columns])
    ax.set_yticks(range(len(surface.index)), [f"{x:.0%}" for x in surface.index])
    ax.set_xlabel("collateral fraction")
    ax.set_ylabel("target put delta")
    ax.set_title("Best worst-split excess CAGR (%) by delta × collateral")
    for i in range(surface.shape[0]):
        for j in range(surface.shape[1]):
            ax.text(j, i, f"{surface.iloc[i, j] * 100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    charts["surface"] = fig_b64(fig)

    yearly = pd.DataFrame(
        {
            "Selected": (1.0 + aligned["selected"]).resample("YE").prod() - 1.0,
            "Cash": (1.0 + aligned["cash"]).resample("YE").prod() - 1.0,
            "QQQ": (1.0 + aligned["qqq"]).resample("YE").prod() - 1.0,
        }
    )
    x = np.arange(len(yearly))
    fig, ax = plt.subplots(figsize=(10, 4.0))
    width = 0.27
    ax.bar(x - width, yearly["Selected"] * 100, width, label="Selected", color=COLORS["selected"])
    ax.bar(x, yearly["Cash"] * 100, width, label="Cash", color=COLORS["cash"])
    ax.bar(x + width, yearly["QQQ"] * 100, width, label="QQQ", color=COLORS["qqq"])
    ax.set_xticks(x, yearly.index.year.astype(str), rotation=45, fontsize=8)
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_ylabel("calendar return (%)")
    ax.set_title("Calendar-year return")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=8.5)
    charts["yearly"] = fig_b64(fig)

    if boxx.notna().sum() > 20:
        cash_live = cash_returns.reindex(boxx.index)
        comparison = pd.concat(
            [boxx.pct_change().rename("BOXX"), cash_live.rename("Proxy")], axis=1
        ).dropna()
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.plot(wealth(comparison["BOXX"]), label="Actual BOXX", color="#111827", lw=2.0)
        ax.plot(wealth(comparison["Proxy"]), label="T-bill proxy net 19.49 bp", color=COLORS["cash"], lw=1.5)
        ax.set_title("Cash proxy validation since BOXX inception")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
        charts["boxx"] = fig_b64(fig)

    put_aligned = pd.concat(
        [
            put_actual_returns.rename("Cboe PUT"),
            put_synthetic_returns.rename("Synthetic"),
        ],
        axis=1,
        sort=False,
    ).dropna()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(wealth(put_aligned["Cboe PUT"]), label="Published Cboe PUT index", color=COLORS["put"], lw=2.0)
    ax.plot(wealth(put_aligned["Synthetic"]), label="BS/VIX proxy", color="#111827", lw=1.3)
    ax.set_yscale("log")
    ax.set_title("Proxy validation—not a QQQ option-chain substitute")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    charts["put_proxy"] = fig_b64(fig)

    family_colors = {
        "Baseline": "#111827",
        "Entry filter": "#2563eb",
        "Assignment exit": "#ea580c",
        "Combined": "#7c3aed",
    }
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for family, group in policy_comparison.groupby("family"):
        ax.scatter(
            group["worst_pre_holdout_excess"] * 100,
            group["holdout_ExcessCAGR"] * 100,
            label=family,
            color=family_colors[family],
            alpha=0.80,
            s=45,
        )
    eligible = policy_comparison[
        (policy_comparison["policy_id"] != "baseline")
        & policy_comparison["eligible_pre_holdout"]
    ]
    challenger = eligible.iloc[0]
    baseline_row = policy_comparison[
        policy_comparison["policy_id"] == "baseline"
    ].iloc[0]
    for row, label in [(baseline_row, "baseline"), (challenger, "pre-holdout winner")]:
        ax.annotate(
            label,
            (
                row["worst_pre_holdout_excess"] * 100,
                row["holdout_ExcessCAGR"] * 100,
            ),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    ax.set_xlabel("Worst discovery/validation excess CAGR (%)")
    ax.set_ylabel("Sealed-holdout excess CAGR (%)")
    ax.set_title("Policy selection stability: pre-holdout vs sealed holdout")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8)
    charts["policy_scatter"] = fig_b64(fig)

    challenger_id = str(challenger["policy_id"])
    holdout_start, holdout_end = SPLITS["holdout"]
    policy_holdout = pd.concat(
        [
            policy_results["baseline"].returns.rename("baseline"),
            policy_results[challenger_id].returns.rename("challenger"),
            cash_returns.rename("cash"),
        ],
        axis=1,
    ).dropna().loc[holdout_start:holdout_end]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.plot(
        wealth(policy_holdout["baseline"]),
        label="Simple baseline",
        color="#111827",
        lw=1.8,
    )
    ax.plot(
        wealth(policy_holdout["challenger"]),
        label=f"Pre-holdout winner: {challenger['policy']}",
        color="#ea580c",
        lw=2.0,
    )
    ax.plot(
        wealth(policy_holdout["cash"]),
        label="Cash proxy",
        color=COLORS["cash"],
        lw=1.3,
    )
    ax.set_title("Sealed holdout: baseline vs pre-selected challenger")
    ax.set_ylabel("wealth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    charts["policy_holdout"] = fig_b64(fig)
    return charts


def render_report(
    *,
    selected: PutWriteConfig,
    selected_result,
    selected_metrics: pd.DataFrame,
    grid: pd.DataFrame,
    policy_comparison: pd.DataFrame,
    robustness: pd.DataFrame,
    calibration: pd.DataFrame,
    cash_validation: pd.DataFrame,
    put_validation: pd.DataFrame,
    advice,
    entry_gate: str,
    bootstrap_ci: tuple[float, float],
    charts: dict[str, str],
    data_end: pd.Timestamp,
) -> str:
    """Build a self-contained bilingual HTML research report."""
    assignments = selected_result.events[
        selected_result.events.get("event", pd.Series(dtype=str)) == "assignment"
    ]
    exits = selected_result.events[
        selected_result.events.get("event", pd.Series(dtype=str)) == "stock_exit"
    ]
    timeout_share = (
        float((exits.get("reason", pd.Series(dtype=str)) == "timeout").mean())
        if not exits.empty
        else 0.0
    )
    holdout_row = selected_metrics.loc["holdout"]
    verdict = (
        "accept_monitoring"
        if holdout_row["ExcessCAGR"] > 0 and holdout_row["MaxDD"] >= -0.25
        else "needs_revision"
    )
    selection = pd.DataFrame(
        [
            {
                "Target delta": f"{selected.target_delta:.0%}",
                "Tenor (trading days)": selected.tenor_trading_days,
                "Collateral": f"{selected.collateral_fraction:.0%}",
                "Max stock hold": selected.max_assignment_days,
                "200d gate": selected.trend_lookback is not None,
                "IV multiplier": selected.iv_multiplier,
                "Entry haircut": f"{selected.entry_spread_fraction:.0%}",
            }
        ]
    )
    metric_display = selected_metrics.reset_index(names="Period")[
        ["Period", "CAGR", "CashCAGR", "ExcessCAGR", "AnnVol", "Sharpe", "MaxDD"]
    ]
    robustness_display = robustness.copy()
    calibration_display = calibration.head(8).copy()
    cash_display = cash_validation.copy()
    put_display = put_validation.copy()

    policy_columns = [
        "pre_holdout_rank",
        "policy",
        "eligible_pre_holdout",
        "worst_pre_holdout_excess",
        "validation_ExcessCAGR",
        "holdout_ExcessCAGR",
        "holdout_vs_baseline",
        "holdout_MaxDD",
        "entries_per_year",
        "assignment_rate",
        "stock_time",
    ]
    policy_column_names = {
        "pre_holdout_rank": "Rank",
        "policy": "Policy",
        "eligible_pre_holdout": "Eligible",
        "worst_pre_holdout_excess": "Worst pre excess",
        "validation_ExcessCAGR": "Validation excess",
        "holdout_ExcessCAGR": "Holdout excess",
        "holdout_vs_baseline": "Holdout Δ vs baseline",
        "holdout_MaxDD": "Holdout MaxDD",
        "entries_per_year": "Entries/year",
        "assignment_rate": "Assign rate",
        "stock_time": "Stock time",
    }

    def policy_display(family: str) -> pd.DataFrame:
        subset = policy_comparison[
            policy_comparison["family"].isin(["Baseline", family])
        ]
        return (
            subset.sort_values("pre_holdout_rank")[policy_columns]
            .rename(columns=policy_column_names)
            .reset_index(drop=True)
        )

    entry_display = policy_display("Entry filter")
    exit_display = policy_display("Assignment exit")
    combined_display = policy_display("Combined")
    baseline_policy = policy_comparison[
        policy_comparison["policy_id"] == "baseline"
    ].iloc[0]
    challenger = policy_comparison[
        (policy_comparison["policy_id"] != "baseline")
        & policy_comparison["eligible_pre_holdout"]
    ].iloc[0]
    best_entry = policy_comparison[
        policy_comparison["family"] == "Entry filter"
    ].sort_values("pre_holdout_rank").iloc[0]
    challenger_delta = float(challenger["holdout_vs_baseline"])
    challenger_outcome = "beat" if challenger_delta > 0 else "lagged"

    top_grid = grid.head(12)[
        [
            "delta",
            "tenor_days",
            "collateral",
            "hold_days",
            "trend_200d",
            "worst_pre_holdout_excess",
            "worst_pre_holdout_drawdown",
            "holdout_ExcessCAGR",
            "holdout_MaxDD",
        ]
    ].copy()
    advice_display = pd.DataFrame(
        [
            {
                "Spot": advice.spot,
                "Target Δ": f"{advice.target_delta:.0%}",
                "Calendar DTE": advice.tenor_calendar_days,
                "Model strike": advice.strike,
                "Model mid": advice.model_mid,
                "Assumed fill": advice.assumed_fill,
                "Contracts": advice.contracts,
                "Collateral": advice.collateral,
                "Premium": advice.premium,
                "Breakeven": advice.breakeven,
                "Strike distance": advice.downside_to_strike,
            }
        ]
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QQQ cash-secured put + BOXX-like yield research</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC",Arial,sans-serif;max-width:1080px;margin:0 auto;padding:30px 22px;color:#1f2937;line-height:1.58}}
h1{{font-size:27px;margin:0 0 3px}} h2{{font-size:19px;margin:34px 0 11px;border-bottom:2px solid #e5e7eb;padding-bottom:6px}}
h3{{font-size:15px;margin:20px 0 6px}} .sub{{color:#6b7280;font-size:13px;margin-bottom:14px}}
.verdict{{background:#ecfdf5;border-left:5px solid #0f766e;padding:12px 15px;border-radius:6px;margin:15px 0}}
.warn{{background:#fff7ed;border-left:5px solid #ea580c;padding:11px 14px;border-radius:6px;margin:13px 0}}
.danger{{background:#fef2f2;border-left:5px solid #dc2626;padding:11px 14px;border-radius:6px;margin:13px 0}}
.note{{background:#eff6ff;border-left:5px solid #2563eb;padding:11px 14px;border-radius:6px;margin:13px 0}}
.kpis{{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}} .kpi{{flex:1;min-width:145px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:9px;padding:12px 14px}}
.kpi .v{{font-size:21px;font-weight:700;color:#0f766e}} .kpi .l{{font-size:11px;text-transform:uppercase;color:#6b7280;letter-spacing:.03em}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:9px 0;display:block;overflow-x:auto}}
th,td{{border:1px solid #e5e7eb;padding:6px 8px;text-align:right;white-space:nowrap}} th{{background:#f1f5f9}} th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;border:1px solid #eef2f7;border-radius:8px;margin:9px 0}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px}} li{{margin:5px 0}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:15px}}
a{{color:#1d4ed8}} .small{{font-size:12px;color:#6b7280}}
</style></head><body>
<h1>QQQ OTM put + BOXX-like cash yield</h1>
<div class="sub">中英混合研究报告 · data through {data_end.date()} · daily marked-to-model · discovery 2007–2016 · validation 2017–2021 · sealed holdout 2022–2026H1</div>

<div class="verdict"><b>Verdict: {verdict}.</b> 这套结构有合理机制，但不是“现金白赚利息、put premium 白捡”。
它把 cash carry、volatility risk premium 和被行权后的 QQQ beta 绑在一起。建议把它作为
<b>cash enhancement sleeve</b>，不是 cash replacement；先用真实 option-chain 做 forward paper advisory，再谈下单自动化。</div>

<div class="kpis">
 <div class="kpi"><div class="v">{fmt_pct(holdout_row["ExcessCAGR"])}</div><div class="l">Holdout CAGR above cash</div></div>
 <div class="kpi"><div class="v">{fmt_pct(holdout_row["MaxDD"])}</div><div class="l">Holdout max drawdown</div></div>
 <div class="kpi"><div class="v">{len(assignments)}</div><div class="l">Historical assignments</div></div>
 <div class="kpi"><div class="v">{fmt_pct(float((selected_result.state == "assigned_stock").mean()))}</div><div class="l">Time holding QQQ</div></div>
</div>

<h2>1 · Model frontier & practical sizing / 模型前沿与实操仓位</h2>
{html_table(selection)}
<p><b>Selection rule was frozen before opening the holdout:</b> require positive excess CAGR in both
discovery and validation, cap worst pre-holdout drawdown at 20%, then maximize the weaker split's excess CAGR.
This is a balanced-risk definition of “最合适”; it is not the full-sample highest-return cell.</p>
<div class="note"><b>Practical paper-trading start:</b> keep the selected {selected.target_delta * 100:.0f}Δ / {selected.tenor_trading_days} trading-day / {selected.max_assignment_days}-day recovery rule, but scale collateral from the model frontier's 75% to <b>50% of NAV</b>. This is a conservative two-thirds risk haircut for option-chain, margin and proxy uncertainty—not a second policy selected on holdout performance. The 75% line remains the audited model headline so the haircut is visible rather than rewritten into history.</div>
<ul>
 <li><b>Cash sleeve:</b> use a 1–3 month T-bill/box-spread total-return proxy net of {CASH_FEE_BPS:.2f} bp/year.
 For operations, SGOV/T-bills may be cleaner collateral than BOXX because broker margin treatment is account-specific.</li>
 <li><b>Entry timing:</b> strike and filters use prior-close information; trade is booked at the next close and earns no same-bar return.</li>
 <li><b>Assignment:</b> stop writing new puts; hold QQQ until price reaches premium-adjusted breakeven or the time limit, then sell and wait one day before the next write.</li>
 <li><b>Current model gate:</b> {escape(entry_gate)}. This is illustrative; live advice must ingest executable bid/ask and broker buying-power rules.</li>
</ul>
<div class="warn"><b>Operational caveat.</b> “Cash invested in BOXX” is not identical to cash collateral.
The broker can haircut BOXX, raise house margin during stress, or require liquidation before assignment settlement.
Keep a liquidity buffer and confirm collateral eligibility in writing.</div>

<h2>2 · Historical result / 历史结果</h2>
{html_table(metric_display, percent_columns={"CAGR","CashCAGR","ExcessCAGR","AnnVol","MaxDD"})}
<img alt="wealth" src="data:image/png;base64,{charts["wealth"]}">
<img alt="drawdown" src="data:image/png;base64,{charts["drawdown"]}">
<img alt="holdout" src="data:image/png;base64,{charts["holdout"]}">
<img alt="yearly" src="data:image/png;base64,{charts["yearly"]}">
<p class="small">QQQ is a risk benchmark, not the strategy's objective. Cash is the economic hurdle because the put
uses collateral capacity. Daily option liabilities are marked to model; premium is not recognized as instant profit.</p>

<h2>3 · Strategy policy comparison / IV 选择与接股退出</h2>
<div class="note"><b>Apples-to-apples setup:</b> every row uses {selected.target_delta * 100:.0f}Δ, {selected.tenor_trading_days} trading-day puts,
50% collateral, the same calibrated premium proxy and cash hurdle. Only entry selection or assigned-stock exit changes.
The simple baseline always writes when flat, then exits assigned QQQ at premium-adjusted breakeven or 126 trading days.</div>
<p><b>Pre-holdout winner:</b> {escape(str(challenger["policy"]))} had a worst discovery/validation excess CAGR of
{fmt_pct(float(challenger["worst_pre_holdout_excess"]))}. In the sealed holdout it {challenger_outcome} the simple
baseline by <b>{fmt_pct(abs(challenger_delta))}</b> ({fmt_pct(float(challenger["holdout_ExcessCAGR"]))} versus
{fmt_pct(float(baseline_policy["holdout_ExcessCAGR"]))} excess CAGR). This is exactly why the holdout is shown rather
than quietly retuning the rule.</p>
<p><b>Best IV-entry rule by pre-holdout rank:</b> {escape(str(best_entry["policy"]))}. It produced
{fmt_pct(float(best_entry["holdout_ExcessCAGR"]))} holdout excess CAGR and {fmt_pct(float(best_entry["holdout_MaxDD"]))}
max drawdown, versus baseline {fmt_pct(float(baseline_policy["holdout_ExcessCAGR"]))} and
{fmt_pct(float(baseline_policy["holdout_MaxDD"]))}. Selective IV generally reduced trade count and tail exposure,
but aggressive thresholds often gave up too much premium time.</p>
<img alt="policy stability scatter" src="data:image/png;base64,{charts["policy_scatter"]}">
<img alt="policy holdout comparison" src="data:image/png;base64,{charts["policy_holdout"]}">
<h3>Entry filters / 何时卖 put</h3>
{html_table(entry_display, percent_columns={"Worst pre excess","Validation excess","Holdout excess","Holdout Δ vs baseline","Holdout MaxDD","Assign rate","Stock time"})}
<p class="small">Absolute VXN gates, a 252-day rolling percentile and VXN minus trailing 21-day realized volatility
are all formed at the prior close. Rolling percentiles use only the trailing window—never a full-sample percentile.</p>
<h3>Assigned-stock exits / 接股后何时卖 QQQ</h3>
{html_table(exit_display, percent_columns={"Worst pre excess","Validation excess","Holdout excess","Holdout Δ vs baseline","Holdout MaxDD","Assign rate","Stock time"})}
<p class="small">Price targets are based on premium-adjusted breakeven unless “strike” is stated. Fixed-hold rules
ignore recovery; IV exits wait for VXN normalization. Price/IV conditions observed at close execute at the next close,
while the maximum holding period remains a deterministic timeout.</p>
<h3>Combined policies / 组合规则</h3>
{html_table(combined_display, percent_columns={"Worst pre excess","Validation excess","Holdout excess","Holdout Δ vs baseline","Holdout MaxDD","Assign rate","Stock time"})}
<div class="warn"><b>Decision:</b> keep the simple baseline as the operational reference. Price-target and IV rules are
useful monitoring alternatives, but no challenger earns promotion from this proxy alone. Require real QQQ option-chain
paper data and a second forward holdout before changing the advisory default.</div>

<h2>4 · Parameter surface, not a magic point / 参数面</h2>
<img alt="surface" src="data:image/png;base64,{charts["surface"]}">
{html_table(top_grid, percent_columns={"delta","collateral","worst_pre_holdout_excess","worst_pre_holdout_drawdown","holdout_ExcessCAGR","holdout_MaxDD"})}
<p>Interpretation: delta controls insurance intensity; collateral controls tail inventory. Tenor and post-assignment
holding rule matter, but the economically dominant dial is usually <b>how much NAV may become QQQ</b>.</p>

<h2>5 · Robustness / 稳健性</h2>
{html_table(robustness_display, percent_columns={"CAGR","ExcessCAGR","MaxDD"})}
<div class="note"><b>Block-bootstrap 95% CI</b> for full-history annualized arithmetic excess return:
{fmt_pct(bootstrap_ci[0])} to {fmt_pct(bootstrap_ci[1])}. A CI crossing zero is a “do not oversell” signal,
not permission to optimize harder.</div>
<p>Assignment exits were timeouts in {fmt_pct(timeout_share)} of completed stock episodes.
That number is a useful advisory monitoring KPI: if timeout frequency or assigned-stock time rises materially,
the sleeve has morphed from volatility selling into persistent QQQ ownership.</p>

<h2>6 · Proxy validation / 代理验证</h2>
<div class="grid2">
 <div><h3>Cash proxy vs actual BOXX</h3><img alt="BOXX validation" src="data:image/png;base64,{charts.get("boxx","")}">{html_table(cash_display, percent_columns={"CAGR","TrackingError","CAGRDifference"})}</div>
 <div><h3>Option proxy vs Cboe PUT</h3><img alt="PUT validation" src="data:image/png;base64,{charts["put_proxy"]}">{html_table(put_display, percent_columns={"CAGR","RMSE"})}</div>
</div>
<h3>Early-window calibration candidates</h3>
{html_table(calibration_display, percent_columns={"discovery_rmse","validation_rmse"})}
<div class="danger"><b>Evidence grade: proxy research, not execution-grade backtest.</b>
VXN is an NDX 30-day implied-volatility index, not a timestamped QQQ chain. Black–Scholes misses American exercise,
discrete dividends, strike skew, intraday fills, volatility surface dynamics, and crisis spreads. Calibration against
Cboe PUT checks broad accounting behavior only; it does not validate QQQ OTM premiums.</div>

<h2>7 · Model frontier snapshot / 模型前沿示例</h2>
{html_table(advice_display, percent_columns={"Strike distance"})}
<p>The snapshot assumes a $200k account and model price source <code>{escape(advice.price_source)}</code>.
Contracts are rounded down. For a $200k paper account, the practical starter is <b>one contract</b> (about ${advice.strike * selected.contract_multiplier:,.0f} strike collateral), even though the 75% frontier table permits {advice.contracts}. <b>Do not trade this row:</b> replace model mid with a live bid, use actual expiry/strike,
check open interest and spread, and recompute buying power after applying the broker's BOXX/SGOV haircut.</p>

<h2>8 · Assumptions & failure modes</h2>
<ul>
 <li><b>Return source:</b> T-bill/box carry + volatility risk premium. The put leg is paid for warehousing crash risk;
 assignment is not a free repair mechanism.</li>
 <li><b>Data:</b> adjusted QQQ/SPY closes and VXN/VIX from Yahoo Finance; DTB3 from FRED; no historical QQQ chain.</li>
 <li><b>Option model:</b> European Black–Scholes; target strike from prior-close delta; $1 strike grid; daily theoretical mark;
 calibrated IV multiplier; bid haircut plus $0.65/contract.</li>
 <li><b>Cash:</b> prior observed DTB3 accrued across calendar-day gaps, less BOXX's current net expense ratio. Tax treatment,
 distributions, spreads and broker haircuts are excluded.</li>
 <li><b>Assignment:</b> close-based expiry proxy; QQQ shares use adjusted prices; early exercise and dividend timing omitted.</li>
 <li><b>Tail risk:</b> a crash followed by a slow recovery leaves the sleeve holding QQQ and forgoing fresh premium.
 The max-hold rule realizes losses; removing it merely hides them.</li>
 <li><b>Capacity:</b> QQQ options are liquid, but OTM bid/ask widens sharply in stress. Whole-contract sizing makes small
 accounts lumpy; collateral fraction must be calculated after rounding.</li>
</ul>

<h2>9 · Productionization path / 生产化路径</h2>
<ol>
 <li>Ingest a live QQQ chain with timestamp, bid/ask, IV, delta, DTE, open interest and volume; reject stale/wide quotes.</li>
 <li>Use <code>build_put_write_advice</code> only as the sizing shell; pass the real bid and broker-specific collateral haircut.</li>
 <li>Persist advisory decisions and next-day executable quotes for 3–6 months. Compare model premium, quoted bid and realized fill.</li>
 <li>Add alerts, not autonomous trading: assignment probability, buying-power buffer, BOXX/SGOV liquidity, and timeout-state monitoring.</li>
 <li>Only after paper validation, build an execution adapter under <code>quant_bot_manager</code>; retain the existing live-money gates.</li>
</ol>

<h2>Sources / 来源</h2>
<ul>
 <li><a href="https://cdn.cboe.com/api/global/us_indices/governance/Cboe_PutWrite_Indices_Methodology.pdf">Cboe PutWrite Indices Methodology</a> — collateral account, maturity settlement, roll pricing and daily option marks.</li>
 <li><a href="https://cdn.cboe.com/resources/indices/factsheet/CboeGlobalIndices_PUT-Index.pdf">Cboe PUT Index factsheet</a> — published history and long-run benchmark statistics.</li>
 <li><a href="https://funds.alphaarchitect.com/boxetf/">Alpha Architect BOXX official fund page</a> — strategy, current fee, yield statistics, holdings, distributions and risks.</li>
 <li><a href="https://fred.stlouisfed.org/series/DTB3">FRED DTB3</a> — 3-month Treasury bill discount rate.</li>
</ul>
<p class="small">Research only / 非投资建议. Past and simulated performance do not guarantee future results.
Generated by <code>scripts/qqq_putwrite_study.py</code>; reusable engine:
<code>src/alpha_lab/backtest/put_write.py</code>.</p>
</body></html>"""


def main() -> None:
    prices, fred = download_market_data()
    qqq = prices["QQQ"].dropna()
    annual_rate, cash_returns = cash_series_on_index(fred, qqq.index)
    vxn = (prices["^VXN"] / 100.0).reindex(qqq.index).ffill()
    common = pd.concat(
        [qqq.rename("QQQ"), vxn.rename("VXN"), annual_rate, cash_returns], axis=1
    ).dropna()
    qqq = common["QQQ"]
    vxn = common["VXN"]
    annual_rate = common["annual_rate"]
    cash_returns = common["cash_proxy"]

    iv_multiplier, spread_fraction, calibration, put_synthetic = calibrate_proxy(
        prices,
        annual_rate.reindex(prices.index).ffill(),
        cash_returns.reindex(prices.index).fillna(0.0),
    )
    grid, configs = run_policy_grid(
        qqq,
        vxn,
        cash_returns,
        annual_rate,
        iv_multiplier=iv_multiplier,
        spread_fraction=spread_fraction,
    )
    selected_id = int(grid.iloc[0]["config_id"])
    selected = configs[selected_id]
    selected_result = run_cash_secured_put_backtest(
        qqq,
        vxn,
        cash_returns,
        annual_rate,
        config=selected,
    )
    comparison_base = replace(selected, collateral_fraction=0.50)
    policy_comparison, policy_results, policy_configs = run_policy_comparison(
        qqq,
        vxn,
        cash_returns,
        annual_rate,
        comparison_base,
    )
    challenger_row = policy_comparison[
        (policy_comparison["policy_id"] != "baseline")
        & policy_comparison["eligible_pre_holdout"]
    ].iloc[0]
    challenger_id = str(challenger_row["policy_id"])

    selected_metrics = pd.DataFrame(
        {
            period: stats_row(
                period_slice(selected_result.returns, period),
                period_slice(cash_returns, period),
            )
            for period in SPLITS
        }
    ).T
    selected_metrics.index.name = "Period"
    full_metrics = stats_row(selected_result.returns, cash_returns)
    selected_metrics = pd.concat(
        [pd.DataFrame([full_metrics], index=pd.Index(["full"], name="Period")), selected_metrics]
    )

    robustness = robustness_table(
        qqq,
        vxn,
        cash_returns,
        annual_rate,
        selected,
    )
    excess = selected_result.returns - cash_returns
    bootstrap_ci = block_bootstrap_excess(excess)

    qqq_returns = qqq.pct_change().fillna(0.0)
    boxx = prices["BOXX"].dropna()
    boxx_comparison = pd.concat(
        [
            boxx.pct_change().rename("BOXX"),
            cash_returns.reindex(boxx.index).rename("Proxy"),
        ],
        axis=1,
    ).dropna()
    boxx_stats = []
    for name in ["BOXX", "Proxy"]:
        sm = summary(boxx_comparison[name], periods=PERIODS)
        boxx_stats.append(
            {
                "Series": name,
                "CAGR": sm["CAGR"],
                "AnnVol": sm["AnnVol"],
                "TrackingError": float(
                    (boxx_comparison[name] - boxx_comparison["Proxy"]).std() * np.sqrt(PERIODS)
                ),
                "CAGRDifference": sm["CAGR"]
                - summary(boxx_comparison["Proxy"], periods=PERIODS)["CAGR"],
            }
        )
    cash_validation = pd.DataFrame(boxx_stats)

    put_actual = prices["^PUT"].dropna()
    put_actual_returns = put_actual.pct_change().fillna(0.0)
    put_monthly = pd.concat(
        [
            monthly_returns(put_actual_returns).rename("actual"),
            monthly_returns(put_synthetic).rename("synthetic"),
        ],
        axis=1,
    ).dropna()
    put_validation_rows = []
    for label, start, end in [
        ("calibration 2007-2014", "2007-01-01", "2014-12-31"),
        ("validation 2015-2026", "2015-01-01", END),
    ]:
        sample = put_monthly.loc[start:end]
        for column in ["actual", "synthetic"]:
            sm = summary(sample[column], periods=12)
            put_validation_rows.append(
                {
                    "Period": label,
                    "Series": column,
                    "CAGR": sm["CAGR"],
                    "MonthlyCorr": float(sample.corr().iloc[0, 1]),
                    "RMSE": float(
                        np.sqrt(np.mean((sample["synthetic"] - sample["actual"]) ** 2))
                    ),
                }
            )
    put_validation = pd.DataFrame(put_validation_rows)

    latest_spot = float(qqq.iloc[-1])
    latest_iv = float(vxn.iloc[-1])
    latest_rate = float(annual_rate.iloc[-1])
    advice = build_put_write_advice(
        spot=latest_spot,
        annual_iv=latest_iv,
        annual_cash_yield=latest_rate,
        account_nav=200_000.0,
        config=selected,
    )
    if selected.trend_lookback is None:
        entry_gate = "No trend gate in the selected policy"
    else:
        moving_average = float(qqq.rolling(selected.trend_lookback).mean().iloc[-1])
        entry_gate = (
            f"PASS: QQQ {latest_spot:.2f} ≥ 200d MA {moving_average:.2f}"
            if latest_spot >= moving_average
            else f"WAIT: QQQ {latest_spot:.2f} < 200d MA {moving_average:.2f}"
        )

    charts = build_charts(
        selected_result.returns,
        cash_returns,
        qqq_returns,
        grid,
        boxx,
        put_actual_returns,
        put_synthetic,
        policy_comparison,
        policy_results,
    )
    html = render_report(
        selected=selected,
        selected_result=selected_result,
        selected_metrics=selected_metrics,
        grid=grid,
        policy_comparison=policy_comparison,
        robustness=robustness,
        calibration=calibration,
        cash_validation=cash_validation,
        put_validation=put_validation,
        advice=advice,
        entry_gate=entry_gate,
        bootstrap_ci=bootstrap_ci,
        charts=charts,
        data_end=qqq.index.max(),
    )

    market = pd.concat(
        [
            qqq.rename("QQQ"),
            vxn.rename("VXN"),
            annual_rate.rename("DTB3_annual"),
            cash_returns.rename("cash_proxy_return"),
        ],
        axis=1,
    )
    market.to_parquet(OUT / "market_proxies.parquet")
    grid.to_csv(OUT / "parameter_grid.csv", index=False)
    calibration.to_csv(OUT / "put_proxy_calibration.csv", index=False)
    robustness.to_csv(OUT / "robustness.csv", index=False)
    policy_comparison.to_csv(OUT / "policy_comparison.csv", index=False)
    pd.DataFrame(
        {policy_id: result.returns for policy_id, result in policy_results.items()}
    ).to_parquet(OUT / "policy_returns.parquet")
    policy_event_frames = []
    for policy_id, result in policy_results.items():
        if result.events.empty:
            continue
        event_frame = result.events.reset_index().copy()
        event_frame.insert(0, "policy_id", policy_id)
        event_frame.insert(
            1,
            "family",
            policy_comparison.set_index("policy_id").loc[policy_id, "family"],
        )
        policy_event_frames.append(event_frame)
    pd.concat(policy_event_frames, ignore_index=True).to_parquet(
        OUT / "policy_events.parquet",
        index=False,
    )
    selected_metrics.to_csv(OUT / "selected_metrics.csv")
    selected_result.returns.to_frame().join(cash_returns).join(qqq_returns.rename("QQQ_return")).to_parquet(
        OUT / "selected_returns.parquet"
    )
    selected_result.events.to_csv(OUT / "selected_events.csv")
    meta = {
        "generated_at": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "data_start": str(qqq.index.min().date()),
        "data_end": str(qqq.index.max().date()),
        "splits": SPLITS,
        "cash_fee_bps": CASH_FEE_BPS,
        "selected_config": asdict(selected),
        "selected_config_id": selected_id,
        "comparison_baseline_config": asdict(comparison_base),
        "pre_holdout_challenger_id": challenger_id,
        "pre_holdout_challenger_config": asdict(policy_configs[challenger_id]),
        "bootstrap_excess_ci": bootstrap_ci,
        "advice_200k_model_only": advice.to_dict(),
        "entry_gate": entry_gate,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(html, encoding="utf-8")

    print(f"Report: {REPORT}")
    print(f"Artifacts: {OUT}")
    print("Selected config:", json.dumps(asdict(selected), indent=2))
    print(selected_metrics.to_string())


if __name__ == "__main__":
    main()
