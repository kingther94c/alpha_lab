# P0 — Crypto Intraday Foundation + Baselines

**Slug**: `crypto_intraday/P0-baselines`
**Date started**: 2026-05-23
**Researcher**: Kelvin Chen
**Status**: accept
**Notebooks**:
- [00_data_inventory](../../../notebooks/90_crypto_intraday/00_data_inventory.ipynb)
- [01_baselines](../../../notebooks/90_crypto_intraday/01_baselines.ipynb)
**Supersedes**: —

## Research question

With the BTC/ETH intraday framework's foundation built (Binance Vision loader, PM-holdout enforcement, walk-forward / purged-embargo CV, funding-aware vectorized backtester, Backtrader cross-check oracle), do the trivial baselines behave as expected on perp 1m data end-to-end?

## Hypothesis & rationale

This is a diagnostic, not an alpha claim. The hypothesis is that the foundation is plumbed correctly. Falsifiable expectations:
- Always-flat strategy returns exactly 0 in every cost scenario.
- Buy-and-hold gross return matches the underlying asset's close-to-close return (modulo the 1-bar lag).
- Cost ordering holds (zero ≥ base ≥ stress) for any strategy with positive turnover.
- A uniform-random weight signal at 1m frequency is annihilated by transaction costs.
- PM holdout window is never accessed; audit log proves zero raises.

If any of these fail, there is a bug in the data, cost, funding, or holdout layer that must be fixed before P1 begins.

## Universe & data

- Universe file: [configs/crypto_intraday_universe.csv](../../../configs/crypto_intraday_universe.csv) — BTCUSDT, ETHUSDT on both spot and USD-M perp.
- Source: data.binance.vision public archives (monthly ZIPs).
- Available history (full inventory, no download): all four streams have continuous monthly archives from at least 2019-09 (perp) / 2017-08 (spot) through the most recent published month. Earliest common-start = 2019-11 (latest of the four firsts).
- Research window for the baselines notebook: 2024-06-01 → 2024-07-01 (one month, well inside the train slice).
- Funding panel: 90 events over the month at the expected 8h cadence; rates centered around ~0.01%/8h for both symbols (typical).

## Signal & portfolio construction

Five baselines:
1. `always_flat` — weights = 0 for every bar, every symbol.
2. `bh_btc` — weights = {BTC: 1.0, ETH: 0}.
3. `bh_eth` — weights = {BTC: 0, ETH: 1.0}.
4. `equal_weight` — weights = {BTC: 0.5, ETH: 0.5}.
5. `random` — `uniform(-1, 1)` weights per bar per symbol, seed=0 (sanity / null).

Each runs at 1m on perp through three cost scenarios pulled from [configs/crypto_intraday.yaml](../../../configs/crypto_intraday.yaml): `zero`, `perp_base`, `perp_stress`. `perp_base` and `perp_stress` include funding cost; `zero` excludes it. Slippage is currently a scalar in `run_backtest`; the baselines average the two symbols' per-side slip until per-symbol slip lands in P1.

`bars_per_year = 525_600` (365 × 24 × 60) for 1m perp.

## Headline performance

Net total return over 2024-06-01 → 2024-07-01:

| strategy      | zero    | perp_base | perp_stress |
|---------------|---------|-----------|-------------|
| always_flat   | 0.0000  | 0.0000    | 0.0000      |
| bh_btc        | -0.0716 | -0.0797   | -0.0798     |
| bh_eth        | -0.0879 | -0.0962   | -0.0963     |
| equal_weight  | -0.0791 | -0.0873   | -0.0874     |
| random        | -0.0205 | -1.0000   | -1.0000     |

(June 2024 was a down month for crypto; BTC -7.16%, ETH -8.79%.)

## Diagnostics

- **`always_flat == 0`** across all scenarios — confirmed by assertion in the notebook.
- **BH gross matches underlying**: `bh_btc` gross over the month = -0.071628, while BTC close-to-close = -0.071628. Diff = 0.000000. Same alignment for ETH.
- **Cost ordering**: zero ≥ base ≥ stress holds strictly for `equal_weight` and `random`; for BH the differences come from the small one-shot turnover at the start plus funding.
- **Funding bite**: BH BTC pays ~80 bps of funding drag over the month (roughly 30 × 3 × 1 bp BTC funding × 1.0 held weight, consistent with the panel).
- **Random under costs**: net of base/stress costs the random signal loses 100% (clipped). Average per-bar turnover ~0.5 × ~43200 bars × ~6.25 bps round-trip ≈ 13,500% in cost drag. Confirms the cost model is doing real work and that uncontrolled turnover at 1m frequency is uneconomical. Important null result for ML in P3.

## Robustness

- Foundation tests: 115 pytest tests green (`pytest -q`).
- Backtrader cross-check (`tests/test_bt_vs_vector.py`) — `always_long_no_costs`, `long_only_directional_agreement`, `costs_are_a_drag`, `flat_zero_returns` all pass.
- The L/S-with-costs case is intentionally **not** asserted to match the vector engine; cost-model semantics (BT: commission × fill notional + slippage on fill price; vector: flat bps × turnover) diverge by design for short legs. The cross-check's load-bearing role is validating execution / return mechanics on the always-long case.

## Failure modes

- **Per-symbol slip not yet supported**: scalar `slippage_bps` means we average across symbols. For BH this is irrelevant (no recurring trading); for any cross-sectional strategy this becomes material. P1 work.
- **Daily-archive fallback for current month is a TODO**: monthly archives lag ~1 day after month-end. For research on closed windows this is fine; for any "to today" run we'll miss ~0–30 days.
- **Funding floored to bar containing it**: a funding event at HH:00.001 lands in the HH:00 bar (correct), but the search uses `searchsorted(side="right") - 1`, dropping events that pre-date the first loaded bar. Documented in `_bucket_funding_to_bars`.
- **Single-asset Backtrader cross-check**: we only validate the engines match on one symbol. Multi-asset cross-check would need a separate fixture; deferred.

## Decision

**Accept**. The P0 foundation is healthy: data flows correctly, leak-safe lag holds, costs / funding apply as designed, PM holdout is enforced and audited, and a tiny independent oracle agrees with the vector engine on the load-bearing always-long case.

Justification: all five quantitative sanity checks listed in the hypothesis pass, with `bh_btc` gross matching BTC close-to-close to 6 decimals. The null result on `random` (cost-annihilated at 1m) is exactly what we want before any strategy claims an edge.

## Next steps (P1)

1. Horizon-discovery notebook: IC and rank-IC of simple lagged-return signals against {5m, 15m, 30m, 1h, 4h} forward returns at 1m, 5m, 15m sampling. Output a (feature, horizon, interval) → IC ± block-bootstrap CI heatmap. Commit to 1–2 working intervals and 2–3 working horizons for downstream work.
2. Rule-based family at the selected intervals: MA crossover, trend filter, RSI MR, Bollinger MR/breakout, Donchian breakout, MACD, volume shock, VWAP distance.
3. Extend `run_backtest` to accept per-symbol slippage (dict). Lift to `src/` once P1 needs it.
4. Populate `docs/research_decisions/crypto_intraday/idea_log.md` with first ~20 ideas (source, intuition, leakage risk, status).

## Appendix

- Notebook paths: see top.
- Commit (this work): TBD.
- Data caches: `data/raw/binance_vision/{spot,perp}/*` and `data/interim/binance/*.parquet` (gitignored).
- PM holdout audit log: `data/results/pm_holdout_audit.jsonl` — 10 entries, all action=`ok`, no raises. Banner statement: **PM Holdout was not accessed.**
