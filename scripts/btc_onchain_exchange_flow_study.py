"""Leak-aware BTC on-chain exchange-flow strategy study.

Screening and robustness stop before 2024. The external period is released only
after ``frozen_spec.json`` is written following critical review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from btc_etf_flow_robustness import block_bootstrap
from btc_etf_flow_study import period_metrics, simulate

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interim" / "btc_coinmetrics_onchain.csv"
RESULT_DIR = ROOT / "data" / "results" / "btc_onchain_exchange_flow"
API_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = [
    "PriceUSD",
    "FlowInExNtv",
    "FlowOutExNtv",
    "SplyExNtv",
    "CapMVRVCur",
]
SCREEN_END = pd.Timestamp("2024-01-01")
BASE_COST_BPS = 15.0
STRESS_COST_BPS = 30.0
SEED = 20260720


@dataclass(frozen=True)
class OnchainCandidate:
    """Frozen description of an on-chain candidate."""

    name: str
    family: str
    feature_window: int | None
    hold: int
    trend_window: int
    quantile: float | None
    matched_control: str


def file_sha256(path: Path) -> str:
    """Return a stable SHA-256 hash for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_irx(start: str, end: str) -> pd.Series:
    raw = yf.download(
        "^IRX",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo returned no data for ^IRX")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.Series(close.to_numpy(float), index=index.normalize(), name="IRX")


def download_data() -> None:
    """Fetch official Coin Metrics Community data and the cash-rate proxy."""
    params = {
        "assets": "btc",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": "2013-01-01",
        "end_time": "2026-07-19",
        "page_size": 10000,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    records = list(payload["data"])
    while payload.get("next_page_url"):
        response = requests.get(payload["next_page_url"], timeout=60)
        response.raise_for_status()
        payload = response.json()
        records.extend(payload["data"])

    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame.pop("time"), utc=True).dt.tz_localize(None).dt.normalize()
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.set_index("date").sort_index()[METRICS]
    irx = _download_irx("2012-12-01", "2026-07-20")
    frame["IRX"] = irx.reindex(frame.index, method="ffill")
    frame["rf"] = (frame["IRX"] / 100.0 / 365.0).clip(0.0, 0.15 / 365.0)

    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("on-chain data index must be unique and sorted")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if not frame.index.equals(expected):
        raise ValueError("Coin Metrics daily history contains calendar gaps")
    if frame[METRICS + ["rf"]].isna().any().any():
        missing = frame[METRICS + ["rf"]].isna().sum()
        raise ValueError(f"missing on-chain or cash-rate values:\n{missing[missing > 0]}")
    if (frame[["PriceUSD", "SplyExNtv"]] <= 0).any().any():
        raise ValueError("non-positive price or exchange supply")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reset_index()
    output.to_csv(DATA_PATH, index=False, float_format="%.12g")
    print(
        json.dumps(
            {
                "rows": len(output),
                "first": str(frame.index.min().date()),
                "last": str(frame.index.max().date()),
                "sha256": file_sha256(DATA_PATH),
                "source": API_URL,
            },
            indent=2,
        )
    )


def load_data(*, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load and validate the cached on-chain dataset."""
    frame = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    if end is not None:
        frame = frame.loc[frame.index < end]
    if frame.empty or frame.index.has_duplicates:
        raise ValueError("invalid cached on-chain dataset")
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("cached on-chain dataset contains invalid values")
    return frame


def _event_target(
    event: pd.Series,
    data: pd.DataFrame,
    *,
    hold: int,
    trend_window: int,
) -> pd.Series:
    """Publish a day-t event at t+1 and combine it with a trailing trend filter."""
    held_event = event.rolling(hold, min_periods=1).max().fillna(False).astype(bool)
    published = held_event.copy()
    published.index = published.index + pd.Timedelta(days=1)
    available = published.reindex(data.index).fillna(False)
    trend = data["PriceUSD"] > data["PriceUSD"].rolling(trend_window).mean()
    return (available & trend).astype(float)


def candidate_targets(
    data: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, OnchainCandidate]]:
    """Construct the 26 pre-declared on-chain candidates."""
    supply = data["SplyExNtv"]
    net_outflow = data["FlowOutExNtv"] - data["FlowInExNtv"]
    net_features = {
        window: net_outflow.rolling(window, min_periods=window).sum() / supply
        for window in (7, 30)
    }
    reserve_features = {
        window: -supply.pct_change(window)
        for window in (30, 90)
    }
    targets: dict[str, pd.Series] = {}
    specs: dict[str, OnchainCandidate] = {}

    def add(
        *,
        family: str,
        feature_window: int | None,
        event: pd.Series,
        hold: int,
        trend_window: int,
        quantile: float | None,
    ) -> None:
        q_label = "na" if quantile is None else str(int(quantile * 100))
        w_label = "na" if feature_window is None else str(feature_window)
        name = f"{family}_w{w_label}_q{q_label}_hold{hold}_ma{trend_window}"
        control = f"price_trend_ma{trend_window}"
        targets[name] = _event_target(
            event, data, hold=hold, trend_window=trend_window
        )
        specs[name] = OnchainCandidate(
            name=name,
            family=family,
            feature_window=feature_window,
            hold=hold,
            trend_window=trend_window,
            quantile=quantile,
            matched_control=control,
        )

    for window, feature in net_features.items():
        threshold = feature.rolling(730, min_periods=365).quantile(0.70).shift(1)
        for hold in (7, 30):
            for trend_window in (100, 200):
                add(
                    family="net_outflow",
                    feature_window=window,
                    event=feature > threshold,
                    hold=hold,
                    trend_window=trend_window,
                    quantile=0.70,
                )

    for window, feature in reserve_features.items():
        threshold = feature.rolling(730, min_periods=365).quantile(0.70).shift(1)
        for hold in (7, 30):
            for trend_window in (100, 200):
                add(
                    family="reserve_decline",
                    feature_window=window,
                    event=feature > threshold,
                    hold=hold,
                    trend_window=trend_window,
                    quantile=0.70,
                )

    net_30 = net_features[30]
    reserve_30 = reserve_features[30]
    for quantile in (0.50, 0.70):
        net_threshold = net_30.rolling(730, min_periods=365).quantile(quantile).shift(1)
        reserve_threshold = (
            reserve_30.rolling(730, min_periods=365).quantile(quantile).shift(1)
        )
        event = (net_30 > net_threshold) & (reserve_30 > reserve_threshold)
        for hold in (7, 30):
            for trend_window in (100, 200):
                add(
                    family="dual_accumulation",
                    feature_window=30,
                    event=event,
                    hold=hold,
                    trend_window=trend_window,
                    quantile=quantile,
                )

    structural = supply < supply.rolling(365, min_periods=365).mean().shift(1)
    for trend_window in (100, 200):
        add(
            family="structural_scarcity",
            feature_window=365,
            event=structural,
            hold=1,
            trend_window=trend_window,
            quantile=None,
        )
    if len(targets) != 26:
        raise AssertionError(f"expected 26 candidates, got {len(targets)}")
    return targets, specs


def control_targets(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Return buy-and-hold and price-only trend controls."""
    price = data["PriceUSD"]
    return {
        "buy_hold": pd.Series(1.0, index=data.index),
        "price_trend_ma100": (price > price.rolling(100).mean()).astype(float),
        "price_trend_ma200": (price > price.rolling(200).mean()).astype(float),
    }


def market_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Map Coin Metrics fields to the simulator's market contract."""
    return pd.DataFrame({"BTC": data["PriceUSD"], "rf": data["rf"]}, index=data.index)


def _metrics_lookup(
    metrics: pd.DataFrame,
    candidate: str,
    split: str,
    cost_bps: float,
) -> pd.Series:
    rows = metrics.loc[
        (metrics["candidate"] == candidate)
        & (metrics["split"] == split)
        & (metrics["cost_bps"] == cost_bps)
    ]
    if len(rows) != 1:
        raise AssertionError(f"non-unique metric lookup for {candidate}, {split}, {cost_bps}")
    return rows.iloc[0]


def screen() -> None:
    """Screen the frozen grid without loading 2024+ observations."""
    data = load_data(end=SCREEN_END)
    market = market_frame(data)
    targets, specs = candidate_targets(data)
    all_targets = {**control_targets(data), **targets}
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for name, target in all_targets.items():
        for cost_bps in (BASE_COST_BPS, STRESS_COST_BPS):
            result = simulate(target, market, cost_bps=cost_bps)
            if cost_bps == BASE_COST_BPS:
                returns[f"{name}__total"] = result["total_return"]
                returns[f"{name}__excess"] = result["excess_return"]
            for split, start, end in (
                ("development", "2015-01-01", "2020-01-01"),
                ("validation", "2020-01-01", "2024-01-01"),
            ):
                rows.append(
                    {
                        "candidate": name,
                        "family": specs[name].family if name in specs else "control",
                        "split": split,
                        "cost_bps": cost_bps,
                        **period_metrics(result, start, end),
                    }
                )
    metrics = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULT_DIR / "screen_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(RESULT_DIR / "screen_returns.parquet")

    shortlist_rows: list[dict[str, object]] = []
    for name, spec in specs.items():
        dev = _metrics_lookup(metrics, name, "development", BASE_COST_BPS)
        val = _metrics_lookup(metrics, name, "validation", BASE_COST_BPS)
        dev_stress = _metrics_lookup(metrics, name, "development", STRESS_COST_BPS)
        val_stress = _metrics_lookup(metrics, name, "validation", STRESS_COST_BPS)
        control = _metrics_lookup(
            metrics, spec.matched_control, "validation", BASE_COST_BPS
        )
        sharpe_add = val["ExcessSharpe"] - control["ExcessSharpe"]
        dd_reduction = abs(control["MaxDD"]) - abs(val["MaxDD"])
        retention = (
            val["TotalCAGR"] / control["TotalCAGR"]
            if control["TotalCAGR"] > 0
            else np.nan
        )
        incremental = (sharpe_add >= 0.05) or (
            dd_reduction >= 0.10 * abs(control["MaxDD"]) and retention >= 0.80
        )
        eligible = bool(
            dev["TotalCAGR"] > 0
            and val["TotalCAGR"] > 0
            and dev["ExcessCAGR"] > 0
            and val["ExcessCAGR"] > 0
            and dev_stress["ExcessCAGR"] > 0
            and val_stress["ExcessCAGR"] > 0
            and val["MaxDD"] >= -0.45
            and incremental
        )
        shortlist_rows.append(
            {
                **asdict(spec),
                "eligible": eligible,
                "robust_score": min(dev["ExcessSharpe"], val["ExcessSharpe"]),
                "development_total_cagr": dev["TotalCAGR"],
                "validation_total_cagr": val["TotalCAGR"],
                "development_excess_cagr": dev["ExcessCAGR"],
                "validation_excess_cagr": val["ExcessCAGR"],
                "development_excess_sharpe": dev["ExcessSharpe"],
                "validation_excess_sharpe": val["ExcessSharpe"],
                "validation_max_dd": val["MaxDD"],
                "validation_ann_turnover": val["AnnTurnover"],
                "validation_time_in_market": val["TimeInMarket"],
                "validation_sharpe_add": sharpe_add,
                "validation_dd_reduction": dd_reduction,
                "validation_cagr_retention": retention,
            }
        )
    shortlist = pd.DataFrame(shortlist_rows).sort_values(
        ["eligible", "robust_score", "validation_ann_turnover"],
        ascending=[False, False, True],
    )
    shortlist.to_csv(RESULT_DIR / "screen_shortlist.csv", index=False)
    eligible = shortlist.loc[shortlist["eligible"]]
    if eligible.empty:
        raise RuntimeError("no on-chain candidate passed the pre-declared gates")
    leader_name = str(eligible.iloc[0]["name"])
    leader = {
        "candidate": leader_name,
        "spec": asdict(specs[leader_name]),
        "selection_rule": "max min development/validation excess Sharpe; turnover tie-break",
        "external_period_released": False,
    }
    (RESULT_DIR / "provisional_leader.json").write_text(
        json.dumps(leader, indent=2), encoding="utf-8"
    )
    print(json.dumps(leader, indent=2))
    print(shortlist.head(26).to_string(index=False))


def robustness() -> None:
    """Attack the provisional leader using pre-2024 observations only."""
    data = load_data(end=SCREEN_END)
    market = market_frame(data)
    targets, specs = candidate_targets(data)
    leader = json.loads((RESULT_DIR / "provisional_leader.json").read_text("utf-8"))
    selected = leader["candidate"]
    if leader["spec"] != asdict(specs[selected]):
        raise ValueError("provisional specification changed after screening")

    rows: list[dict[str, object]] = []
    for cost_bps in (15.0, 30.0, 50.0):
        for extra_lag in (0, 1, 2):
            result = simulate(
                targets[selected], market, cost_bps=cost_bps, extra_lag=extra_lag
            )
            for split, start, end in (
                ("development", "2015-01-01", "2020-01-01"),
                ("validation", "2020-01-01", "2024-01-01"),
                ("validation_h1", "2020-01-01", "2022-01-01"),
                ("validation_h2", "2022-01-01", "2024-01-01"),
            ):
                rows.append(
                    {
                        "candidate": selected,
                        "split": split,
                        "cost_bps": cost_bps,
                        "execution_lag_days": 1 + extra_lag,
                        **period_metrics(result, start, end),
                    }
                )
    robust = pd.DataFrame(rows)
    robust.to_csv(RESULT_DIR / "robustness.csv", index=False)

    base = simulate(targets[selected], market, cost_bps=BASE_COST_BPS)
    observed_dev = period_metrics(base, "2015-01-01", "2020-01-01")
    observed_val = period_metrics(base, "2020-01-01", "2024-01-01")
    observed_score = min(observed_dev["ExcessSharpe"], observed_val["ExcessSharpe"])
    shift_columns = ["FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "CapMVRVCur"]
    offsets = np.arange(365, len(data) - 365)
    rng = np.random.default_rng(SEED)
    test_offsets = rng.choice(offsets, size=min(120, len(offsets)), replace=False)
    fixed_rows: list[dict[str, float]] = []
    family_rows: list[dict[str, float]] = []
    for offset in test_offsets:
        shifted = data.copy()
        shifted.loc[:, shift_columns] = np.roll(
            data[shift_columns].to_numpy(), int(offset), axis=0
        )
        shifted_targets, _ = candidate_targets(shifted)
        fixed_result = simulate(shifted_targets[selected], market, cost_bps=BASE_COST_BPS)
        fixed_val = period_metrics(fixed_result, "2020-01-01", "2024-01-01")
        fixed_rows.append(
            {
                "offset": float(offset),
                "ExcessCAGR": fixed_val["ExcessCAGR"],
                "ExcessSharpe": fixed_val["ExcessSharpe"],
            }
        )
        best = -np.inf
        for shifted_target in shifted_targets.values():
            result = simulate(shifted_target, market, cost_bps=BASE_COST_BPS)
            dev = period_metrics(result, "2015-01-01", "2020-01-01")
            val = period_metrics(result, "2020-01-01", "2024-01-01")
            best = max(best, min(dev["ExcessSharpe"], val["ExcessSharpe"]))
        family_rows.append({"offset": float(offset), "best_min_sharpe": best})
    fixed = pd.DataFrame(fixed_rows)
    family = pd.DataFrame(family_rows)
    fixed.to_csv(RESULT_DIR / "permutation_fixed.csv", index=False)
    family.to_csv(RESULT_DIR / "permutation_familywise.csv", index=False)

    validation_excess = base["excess_return"].loc["2020-01-01":"2023-12-31"]
    meta = {
        "candidate": selected,
        "spec": asdict(specs[selected]),
        "observed_robust_score": observed_score,
        "observed_validation_excess_cagr": observed_val["ExcessCAGR"],
        "observed_validation_excess_sharpe": observed_val["ExcessSharpe"],
        "fixed_shift_p_value": float(
            (1 + (fixed["ExcessSharpe"] >= observed_val["ExcessSharpe"]).sum())
            / (1 + len(fixed))
        ),
        "familywise_shift_p_value": float(
            (1 + (family["best_min_sharpe"] >= observed_score).sum())
            / (1 + len(family))
        ),
        "validation_block_bootstrap_28d": block_bootstrap(
            validation_excess, block=28
        ),
    }
    (RESULT_DIR / "robustness_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(robust.to_string(index=False))


def holdout() -> None:
    """Release 2024+ performance for one frozen rule."""
    frozen_path = RESULT_DIR / "frozen_spec.json"
    if not frozen_path.exists():
        raise FileNotFoundError("frozen_spec.json is required before external release")
    frozen = json.loads(frozen_path.read_text("utf-8"))
    data = load_data()
    market = market_frame(data)
    targets, specs = candidate_targets(data)
    selected = frozen["candidate"]
    if selected not in targets or frozen["spec"] != asdict(specs[selected]):
        raise ValueError("frozen rule does not match current candidate code")

    scenarios = (
        (selected, targets[selected], "base_lag1", 15.0, 0),
        (selected, targets[selected], "base_lag2", 15.0, 1),
        (selected, targets[selected], "stress_lag1", 30.0, 0),
        (selected, targets[selected], "stress_lag2", 30.0, 1),
        ("buy_hold", control_targets(data)["buy_hold"], "control", 15.0, 0),
        (
            specs[selected].matched_control,
            control_targets(data)[specs[selected].matched_control],
            "control",
            15.0,
            0,
        ),
    )
    end = str((data.index.max() + pd.Timedelta(days=1)).date())
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for name, target, scenario, cost_bps, extra_lag in scenarios:
        result = simulate(target, market, cost_bps=cost_bps, extra_lag=extra_lag)
        rows.append(
            {
                "candidate": name,
                "scenario": scenario,
                "cost_bps": cost_bps,
                "execution_lag_days": 1 + extra_lag,
                **period_metrics(result, "2024-01-01", end),
            }
        )
        returns[f"{name}__{scenario}__total"] = result["total_return"].loc[
            "2024-01-01":
        ]
        returns[f"{name}__{scenario}__excess"] = result["excess_return"].loc[
            "2024-01-01":
        ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULT_DIR / "holdout_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(RESULT_DIR / "holdout_returns.parquet")
    meta = {
        "candidate": selected,
        "frozen_at_utc": frozen["frozen_at_utc"],
        "data_sha256": file_sha256(DATA_PATH),
        "external_first": "2024-01-01",
        "external_last": str(data.index.max().date()),
        "label_vintage_warning": "current exchange-address labels may backfill history",
    }
    (RESULT_DIR / "holdout_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("download", "screen", "robustness", "holdout"), required=True
    )
    args = parser.parse_args()
    if args.phase == "download":
        download_data()
    elif args.phase == "screen":
        screen()
    elif args.phase == "robustness":
        robustness()
    else:
        holdout()


if __name__ == "__main__":
    main()
