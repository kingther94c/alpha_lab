"""Leak-safe BTC spot-ETF flow study with a frozen 2026 holdout.

Development deliberately stops at 2026-01-01. Run ``--phase holdout`` only after
``frozen_spec.json`` has been written by the researcher following critical review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from alpha_lab.backtest.metrics import summary

ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / "data" / "interim" / "btc_etf_flows_farside.csv"
RESULT_DIR = ROOT / "data" / "results" / "btc_etf_flow_study"
DEV_START = pd.Timestamp("2024-03-15")
VALID_START = pd.Timestamp("2025-01-01")
HOLDOUT_START = pd.Timestamp("2026-01-01")
DEV_END = HOLDOUT_START
BASE_COST_BPS = 15.0
STRESS_COST_BPS = 30.0
ETF_COLUMNS = [
    "IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC",
    "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    flow_window: int | None = None
    trend_window: int | None = None
    quantile: float | None = None
    hold: int | None = None
    matched_control: str = "buy_hold"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_flows(*, end: pd.Timestamp) -> pd.DataFrame:
    """Load and validate the reconstructed Farside table, clipped before ``end``."""
    flows = pd.read_csv(FLOW_PATH, parse_dates=["date"]).sort_values("date")
    if flows["date"].duplicated().any():
        raise ValueError("duplicate ETF-flow dates")
    if not flows["date"].is_monotonic_increasing:
        raise ValueError("ETF-flow dates are not sorted")
    recomputed = flows[ETF_COLUMNS].fillna(0.0).sum(axis=1)
    if not np.allclose(recomputed, flows["Total"], atol=1e-9):
        raise ValueError("fund rows do not sum to reported Total")
    return flows.loc[flows["date"] < end].set_index("date")


def _download_close(symbol: str, start: str, end: str) -> pd.Series:
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        series = close[symbol] if symbol in close.columns else close.iloc[:, 0]
    else:
        series = raw["Close"]
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    series = pd.Series(series.to_numpy(dtype=float), index=index.normalize(), name=symbol)
    return series[~series.index.duplicated(keep="last")].sort_index()


def load_market(*, phase: str, refresh: bool) -> pd.DataFrame:
    """Download/cache BTC and 13-week T-bill yield without touching ``data/raw``."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cache = RESULT_DIR / f"market_{phase}.parquet"
    if cache.exists() and not refresh:
        market = pd.read_parquet(cache)
        market.index = pd.DatetimeIndex(market.index)
        return market

    if phase == "develop":
        end = "2026-01-01"
    elif phase == "holdout":
        end = "2026-07-20"
    else:
        raise ValueError(f"unknown phase {phase!r}")

    btc = _download_close("BTC-USD", "2023-09-01", end)
    irx = _download_close("^IRX", "2023-09-01", end)
    market = pd.DataFrame({"BTC": btc}).dropna()
    market["IRX"] = irx.reindex(market.index).ffill()
    if market["IRX"].isna().any():
        raise ValueError("IRX has leading missing values; refusing to backfill from the future")
    market["rf"] = (market["IRX"] / 100.0 / 365.0).clip(lower=0.0, upper=0.15 / 365.0)
    if not market.index.is_monotonic_increasing or market.index.has_duplicates:
        raise ValueError("market index must be sorted and unique")
    if not np.isfinite(market.to_numpy()).all() or (market["BTC"] <= 0).any():
        raise ValueError("invalid market values")
    market.to_parquet(cache)
    return market


def flow_features(flows: pd.DataFrame, market_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build ETF-date features, then publish them conservatively on date + 1 UTC day."""
    f = pd.DataFrame(index=flows.index)
    total = flows["Total"].astype(float)
    for window in (1, 3, 5, 10):
        series = total.rolling(window, min_periods=window).sum()
        f[f"sum_{window}"] = series
        prior = series.rolling(60, min_periods=20)
        f[f"mean_{window}"] = prior.mean().shift(1)
        for q in (0.10, 0.20, 0.30, 0.70, 0.80, 0.90):
            f[f"q{int(q * 100):02d}_{window}"] = prior.quantile(q).shift(1)

    active = flows[ETF_COLUMNS].notna().sum(axis=1).replace(0, np.nan)
    f["breadth"] = (
        (flows[ETF_COLUMNS].fillna(0.0) > 0).sum(axis=1)
        - (flows[ETF_COLUMNS].fillna(0.0) < 0).sum(axis=1)
    ) / active

    available = f.copy()
    available.index = available.index + pd.Timedelta(days=1)
    if available.index.has_duplicates:
        raise ValueError("duplicate conservative availability dates")
    return available.reindex(market_index).ffill()


def candidate_targets(
    flows: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, Candidate]]:
    """Return the pre-registered candidate target weights and specifications."""
    daily = flow_features(flows, market.index)
    trend = {
        window: (market["BTC"] > market["BTC"].rolling(window).mean()).fillna(False)
        for window in (20, 50, 100)
    }
    targets: dict[str, pd.Series] = {
        "buy_hold": pd.Series(1.0, index=market.index),
    }
    specs: dict[str, Candidate] = {
        "buy_hold": Candidate("buy_hold", "control"),
    }
    for tw in (20, 50, 100):
        name = f"price_trend_ma{tw}"
        targets[name] = trend[tw].astype(float)
        specs[name] = Candidate(name, "control", trend_window=tw)

    for fw in (1, 3, 5, 10):
        sum_flow = daily[f"sum_{fw}"]
        name = f"flow_sign_w{fw}"
        targets[name] = (sum_flow > 0).astype(float)
        specs[name] = Candidate(name, "flow_sign", flow_window=fw)

        for tw in (20, 50, 100):
            control = f"price_trend_ma{tw}"

            name = f"flow_confirm_w{fw}_ma{tw}"
            targets[name] = ((sum_flow > 0) & trend[tw]).astype(float)
            specs[name] = Candidate(
                name, "flow_confirm", fw, tw, matched_control=control,
            )

            name = f"flow_above_mean_w{fw}_ma{tw}"
            targets[name] = ((sum_flow > daily[f"mean_{fw}"]) & trend[tw]).astype(float)
            specs[name] = Candidate(
                name, "flow_above_mean", fw, tw, matched_control=control,
            )

            for q in (0.10, 0.20, 0.30):
                name = f"flow_riskoff_w{fw}_q{int(q * 100)}_ma{tw}"
                targets[name] = (
                    (sum_flow > daily[f"q{int(q * 100):02d}_{fw}"]) & trend[tw]
                ).astype(float)
                specs[name] = Candidate(
                    name, "flow_riskoff", fw, tw, q, matched_control=control,
                )

    # Event candidates are defined on the ETF observation grid first, so 'hold' means
    # ETF observations rather than calendar days. They are then published at t+1 and ffilled.
    total = flows["Total"].astype(float)
    for q in (0.70, 0.80, 0.90):
        threshold = total.rolling(60, min_periods=20).quantile(q).shift(1)
        event = total > threshold
        for hold in (3, 5, 10):
            held_event = event.rolling(hold, min_periods=1).max().astype(bool)
            published = held_event.copy()
            published.index = published.index + pd.Timedelta(days=1)
            daily_event = published.reindex(market.index).ffill().fillna(False)
            name = f"extreme_inflow_q{int(q * 100)}_hold{hold}"
            targets[name] = daily_event.astype(float)
            specs[name] = Candidate(name, "extreme_inflow_hold", quantile=q, hold=hold)
            for tw in (20, 50, 100):
                control = f"price_trend_ma{tw}"
                name = f"extreme_inflow_q{int(q * 100)}_hold{hold}_ma{tw}"
                targets[name] = (daily_event & trend[tw]).astype(float)
                specs[name] = Candidate(
                    name,
                    "extreme_inflow_hold",
                    trend_window=tw,
                    quantile=q,
                    hold=hold,
                    matched_control=control,
                )

    for q in (0.10, 0.20, 0.30):
        threshold = total.rolling(60, min_periods=20).quantile(q).shift(1)
        event = total < threshold
        for hold in (3, 5, 10):
            held_event = event.rolling(hold, min_periods=1).max().astype(bool)
            published = held_event.copy()
            published.index = published.index + pd.Timedelta(days=1)
            daily_event = published.reindex(market.index).ffill().fillna(False)
            name = f"extreme_outflow_reversal_q{int(q * 100)}_hold{hold}"
            targets[name] = daily_event.astype(float)
            specs[name] = Candidate(
                name, "extreme_outflow_reversal", quantile=q, hold=hold,
            )

    return targets, specs


def simulate(
    target: pd.Series,
    market: pd.DataFrame,
    *,
    cost_bps: float,
    extra_lag: int = 0,
) -> pd.DataFrame:
    """Long BTC/cash simulation; the target is lagged before earning returns."""
    target = target.reindex(market.index).ffill().fillna(0.0).clip(0.0, 1.0)
    held = target.shift(1 + extra_lag).fillna(0.0)
    btc_return = market["BTC"].pct_change().fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * cost_bps / 10_000.0
    total = held * btc_return + (1.0 - held) * market["rf"] - cost
    excess = total - market["rf"]
    return pd.DataFrame(
        {
            "target": target,
            "held": held,
            "btc_return": btc_return,
            "rf": market["rf"],
            "turnover": turnover,
            "cost": cost,
            "total_return": total,
            "excess_return": excess,
        }
    )


def period_metrics(result: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sample = result.loc[(result.index >= start) & (result.index < end)]
    total = summary(sample["total_return"], periods=365)
    excess = summary(sample["excess_return"], periods=365)
    years = max(len(sample) / 365.0, 1e-9)
    return {
        "TotalReturn": float((1.0 + sample["total_return"]).prod() - 1.0),
        "TotalCAGR": total["CAGR"],
        "TotalSharpe": total["Sharpe"],
        "MaxDD": total["MaxDD"],
        "Calmar": total["Calmar"],
        "ExcessReturn": float((1.0 + sample["excess_return"]).prod() - 1.0),
        "ExcessCAGR": excess["CAGR"],
        "ExcessSharpe": excess["Sharpe"],
        "AnnTurnover": float(sample["turnover"].sum() / years),
        "Trades": int((sample["turnover"] > 1e-12).sum()),
        "TimeInMarket": float(sample["held"].mean()),
        "NPeriods": int(len(sample)),
    }


def develop(*, refresh: bool) -> None:
    """Evaluate the frozen candidate grid without loading any 2026 BTC returns."""
    flows = load_flows(end=DEV_END)
    market = load_market(phase="develop", refresh=refresh)
    if market.index.max() >= HOLDOUT_START:
        raise ValueError("development market cache touches the frozen 2026 holdout")
    targets, specs = candidate_targets(flows, market)

    metric_rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for name, target in targets.items():
        for cost_label, cost_bps in (("base", BASE_COST_BPS), ("stress", STRESS_COST_BPS)):
            result = simulate(target, market, cost_bps=cost_bps)
            if cost_label == "base":
                returns[f"{name}__total"] = result["total_return"]
                returns[f"{name}__excess"] = result["excess_return"]
            for split, start, end in (
                ("development", "2024-03-15", "2025-01-01"),
                ("validation", "2025-01-01", "2026-01-01"),
            ):
                metric_rows.append(
                    {
                        "candidate": name,
                        "split": split,
                        "cost": cost_label,
                        **asdict(specs[name]),
                        **period_metrics(result, start, end),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULT_DIR / "development_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(RESULT_DIR / "development_returns.parquet")

    base = metrics.loc[metrics["cost"] == "base"].copy()
    wide = base.pivot(index="candidate", columns="split")
    shortlist_rows: list[dict[str, object]] = []
    for name, spec in specs.items():
        if spec.family == "control":
            continue
        train = wide.loc[name, "ExcessCAGR"]["development"]
        valid = wide.loc[name, "ExcessCAGR"]["validation"]
        valid_total = wide.loc[name, "TotalReturn"]["validation"]
        valid_dd = wide.loc[name, "MaxDD"]["validation"]
        train_sharpe = wide.loc[name, "ExcessSharpe"]["development"]
        valid_sharpe = wide.loc[name, "ExcessSharpe"]["validation"]
        valid_turn = wide.loc[name, "AnnTurnover"]["validation"]

        control = spec.matched_control
        control_sharpe = wide.loc[control, "ExcessSharpe"]["validation"]
        control_dd = wide.loc[control, "MaxDD"]["validation"]
        control_cagr = wide.loc[control, "TotalCAGR"]["validation"]
        candidate_cagr = wide.loc[name, "TotalCAGR"]["validation"]
        sharpe_add = valid_sharpe - control_sharpe
        dd_reduction = abs(control_dd) - abs(valid_dd)
        cagr_retention = candidate_cagr / control_cagr if control_cagr > 0 else np.nan
        incremental_gate = (sharpe_add >= 0.05) or (
            dd_reduction >= 0.10 * abs(control_dd) and cagr_retention >= 0.80
        )
        eligible = bool(
            train > 0
            and valid > 0
            and valid_total > 0
            and valid_dd >= -0.35
            and incremental_gate
        )
        score = min(train_sharpe, valid_sharpe) + 0.25 * wide.loc[name, "Calmar"][
            "validation"
        ] - 0.02 * valid_turn
        shortlist_rows.append(
            {
                **asdict(spec),
                "eligible": eligible,
                "score": score,
                "development_excess_cagr": train,
                "validation_excess_cagr": valid,
                "development_excess_sharpe": train_sharpe,
                "validation_excess_sharpe": valid_sharpe,
                "validation_total_cagr": candidate_cagr,
                "validation_max_dd": valid_dd,
                "validation_ann_turnover": valid_turn,
                "matched_control": control,
                "validation_sharpe_add": sharpe_add,
                "validation_dd_reduction": dd_reduction,
                "validation_cagr_retention": cagr_retention,
            }
        )

    shortlist = pd.DataFrame(shortlist_rows).sort_values(
        ["eligible", "score"], ascending=[False, False]
    )
    shortlist.to_csv(RESULT_DIR / "development_shortlist.csv", index=False)

    meta = {
        "phase": "develop",
        "flow_source": "https://farside.co.uk/bitcoin-etf-flow-all-data/",
        "flow_sha256": _sha256(FLOW_PATH),
        "flow_rows_used": len(flows),
        "flow_first": str(flows.index.min().date()),
        "flow_last": str(flows.index.max().date()),
        "market_first": str(market.index.min().date()),
        "market_last": str(market.index.max().date()),
        "candidate_count": len(specs),
        "eligible_count": int(shortlist["eligible"].sum()),
        "base_cost_bps": BASE_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "publication_lag": "flow trade-date t -> feature available at t+1 close -> earns t+2 return",
    }
    (RESULT_DIR / "development_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(shortlist.head(20).to_string(index=False))


def holdout(*, refresh: bool) -> None:
    """Release the frozen 2026 holdout for one pre-written specification."""
    spec_path = RESULT_DIR / "frozen_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            "frozen_spec.json is required before releasing the holdout; review development first"
        )
    frozen = json.loads(spec_path.read_text(encoding="utf-8"))
    selected = frozen["candidate"]
    flows = load_flows(end=pd.Timestamp("2026-07-18"))
    market = load_market(phase="holdout", refresh=refresh)
    targets, specs = candidate_targets(flows, market)
    if selected not in targets:
        raise KeyError(f"frozen candidate {selected!r} is not in the pre-registered grid")
    if frozen["spec"] != asdict(specs[selected]):
        raise ValueError("frozen specification does not match current strategy code")

    rows = []
    return_cols = {}
    for cost_label, cost_bps in (("base", BASE_COST_BPS), ("stress", STRESS_COST_BPS)):
        for extra_lag in (0, 1):
            result = simulate(
                targets[selected], market, cost_bps=cost_bps, extra_lag=extra_lag
            )
            label = f"{cost_label}_lag{1 + extra_lag}"
            rows.append(
                {
                    "candidate": selected,
                    "scenario": label,
                    **period_metrics(result, "2026-01-01", "2026-07-18"),
                }
            )
            return_cols[f"{label}__total"] = result["total_return"].loc["2026-01-01":]
            return_cols[f"{label}__excess"] = result["excess_return"].loc["2026-01-01":]
    pd.DataFrame(rows).to_csv(RESULT_DIR / "holdout_metrics.csv", index=False)
    pd.DataFrame(return_cols).to_parquet(RESULT_DIR / "holdout_returns.parquet")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("develop", "holdout"), default="develop")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.phase == "develop":
        develop(refresh=args.refresh)
    else:
        holdout(refresh=args.refresh)


if __name__ == "__main__":
    main()
