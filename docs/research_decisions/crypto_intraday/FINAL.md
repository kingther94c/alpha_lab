# BTC/ETH Intraday Edge Research on Binance — PM Report

**Engagement**: `crypto_intraday`
**Period of work**: 2026-05-23
**Researcher**: Kelvin Chen
**Report version**: 1.0 (P0+P1+P2+P3 complete; P4 = this document)

---

## 1. Executive summary

**Question**: Is there a tradable intraday edge in Binance BTCUSDT / ETHUSDT perp that survives realistic costs?

**Verdict**: **MONITOR.** Two perp-specific signals (`funding_momentum`, `funding_contrarian`) cleared the stress-cost bar over the test window (Q1+Q2 2024), but the result is regime-dependent. None of the canonical technical-analysis families (RSI, Bollinger, MACD, Donchian, etc.) nor cross-asset signals (spread z-score, beta residual, BTC→ETH lead-lag) nor the ML attempt (logistic, ridge, LightGBM) survived stress costs.

**What worked**:
- `funding_momentum` — sign of trailing-10-events funding-rate sum, near-zero turnover, net +40.8% / Sharpe 1.56 at perp_stress over Q1+Q2 2024.
- `funding_contrarian` — z-scored funding extreme contrarian, net +15.0% / Sharpe 1.12 at perp_stress.

**What failed**:
- All TA-style intraday strategies — annihilated by cost drag at any rebalance frequency tested.
- Cross-asset (BTC/ETH spread, beta residual) — gross signal too weak for the cost level.
- ML (LightGBM strongest gross, all crushed by costs).

**Why this is a MONITOR, not a PROCEED**:
- The single surviving family is highly correlated with the H1 2024 BTC bull run.
- 6-month sample is too short to certify regime independence.
- The PM holdout window (2026-01 → 2026-05) was NEVER ACCESSED during this engagement — that's the framework discipline working as designed, but it also means we have no out-of-time-period confirmation.

**Recommended action**: re-run the funding-based strategies on a longer history (2022 bear + 2023 chop) before unlocking PM holdout. Treat the framework as the deliverable; treat the strategy result as a lead, not a conclusion.

---

## 2. Data coverage and quality

### Sources

- **Binance Vision** (https://data.binance.vision): public monthly ZIP archives of klines and (for perp) funding rates.
- Spot endpoint: `/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/...`
- USD-M perp endpoint: `/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/...` and `/.../monthly/fundingRate/...`
- Loader: [src/alpha_lab/data/loaders/binance_vision.py](../../../src/alpha_lab/data/loaders/binance_vision.py).

### Available history

From [00_data_inventory.ipynb](../../../notebooks/90_crypto_intraday/00_data_inventory.ipynb):

| stream            | first archive | total archives | total size |
|-------------------|---------------|----------------|------------|
| spot/BTCUSDT 1m   | ~2017-08      | continuous     | ~5 GB      |
| spot/ETHUSDT 1m   | ~2017-08      | continuous     | ~5 GB      |
| perp/BTCUSDT 1m   | ~2019-09      | continuous     | ~4 GB      |
| perp/ETHUSDT 1m   | ~2019-11      | continuous     | ~4 GB      |

Earliest common start = **2019-11** (latest of the four firsts; bounded by ETHUSDT perp listing).

### Quality validation

- Per-(symbol, month) quality report: expected vs actual bar counts, duplicates, gap structure, zero-volume bars.
- Sample (March 2024) showed full 31×24×60 = 44,640 1m bars per symbol per month, zero duplicates, zero gaps. Data quality at the months we touched is excellent.
- Timezone: ms-epoch UTC at source → all interim parquet caches are tz-aware UTC.
- Monthly archives lag ~1 day after month-end; daily fallback for the current month is a documented TODO.

### PM holdout audit

- Window: 2026-01-01 → 2026-05-01 (half-open).
- `allow_pm_holdout: false` throughout the engagement.
- Loaders + backtest engine both invoke `enforce()` which raises `PMHoldoutAccessError` on forbidden access.
- Audit log: `data/results/pm_holdout_audit.jsonl`, append-only JSONL.
- **Final audit**: 30+ events across all P0-P3 notebooks, **0 raises, accessed=False**. Discipline harness was never tripped.

---

## 3. Research design

### Why perp primary, spot secondary

- Primary tradable market = Binance USD-M perp BTCUSDT / ETHUSDT. Perp has lower taker fees (5 bps base), natural long/short, 24/7 trading, and provides funding-rate + (eventually) open-interest features that spot doesn't.
- Spot is robustness only: stricter cost assumptions (10 bps fee), longer history baseline. P0/P1/P2 ran on perp; spot consistency check is a deferred robustness pass.

### Split logic (half-open intervals)

| split          | range                                        |
|----------------|----------------------------------------------|
| Train          | earliest_clean_ts ≤ t < 2024-07-01           |
| Validation     | 2024-07-01 ≤ t < 2025-01-01                  |
| Research-test  | 2025-01-01 ≤ t < 2026-01-01                  |
| PM holdout     | 2026-01-01 ≤ t < 2026-05-01 (locked)         |

All work in P0-P3 used **only the train slice** — specifically Q1+Q2 2024 for the strategy work. Validation, research-test, and PM holdout are untouched.

### Leakage controls

Documented in [docs/architecture/alpha_lab_architecture.md](../../architecture/alpha_lab_architecture.md) and operationalized:
- Signal at close → execution next bar open. Backtester enforces 1-bar lag on weights.
- All rolling features lagged. `features/intraday.py` contains no `.shift(-k)` (truncation-invariant tests verify this).
- Standardization fit on TRAIN ONLY via `Standardizer` class.
- Walk-forward expanding splits in P3 ML; `safe_forward_returns` masks labels whose window peeks into PM holdout.
- No row-shuffle when labels overlap; `PurgedKFold` available but unused (label horizon = 1 bar in P3 = no overlap).

---

## 4. Idea discovery log

Full catalog at [idea_log.md](./idea_log.md). 45 entries spanning:
- Baselines (5)
- P1 rule-based at 5m (10) — all rejected
- P2 perp-specific, cross-asset, vol-regime at 1h (8) — 2 accept_monitoring, 6 reject
- P3 ML models (3) — all reject
- Deferred (15+) — funding-OI-basis, time-of-day filters, ML ensembles, liquidation/order-book (dropped — no data)

Sources mostly marked "internal prior / not externally verified" per the goal's instruction: cite externally only if actually accessed; otherwise label honestly. No citations were fabricated.

---

## 5. Feature / signal design

### Feature catalog ([src/alpha_lab/features/intraday.py](../../../src/alpha_lab/features/intraday.py))

19 lagged-only features per symbol, plus cross-asset constructors:

**Returns / vol**: `log_return`, `realized_vol_close`, `realized_vol_parkinson`, `realized_vol_garman_klass`.

**Volume / liquidity**: `volume_zscore`, `rolling_taker_imbalance`.

**Trend / mean-reversion**: `ma_slope`, `distance_from_ma`, `breakout_distance`, `atr`, `rsi`, `macd`, `bollinger_pct_b`, `donchian_position`.

**Time**: `time_of_day_hours`, `day_of_week`.

**Cross-asset**: `relative_strength`, `spread_zscore`, `rolling_beta_residual`.

**Funding**: `funding_zscore`, `funding_cumulative`.

### Leak-safety verification

`tests/test_intraday_features.py` runs a **truncation-invariant** check on every feature: the value at row t must equal the function applied to the input truncated at t inclusive. Any forward leak would produce a mismatch. **All 10 leak-safety tests green.**

### Labels / horizons

P1 horizon discovery tested forward returns at {5m, 15m, 30m, 1h, 4h} sampled at {5m}. Decision: focus on 1h primary, 5m secondary. The 30m / 4h horizons were dropped.

P3 ML used 1h forward direction (logistic / LightGBM) and 1h forward return (ridge).

### Rejected features (leakage prone)

None — every feature in the module is constructed with backward-looking rolling operations only. The leak-prone alternatives (e.g., `pd.qcut` on full sample, `df.rolling().mean(center=True)`, expanding stats applied to val) are explicitly forbidden in the contracts doc and were not used.

---

## 6. Backtest methodology

### Primary engine: vectorized

[src/alpha_lab/backtest/vector.py](../../../src/alpha_lab/backtest/vector.py). Single function `run_backtest(signals, prices, *, rebalance, costs_bps, slippage_bps, funding, funding_basis, bars_per_year)`.

- Signals lagged by 1 bar internally (`held = tgt.shift(1)`).
- Per-symbol slippage supported (dict-form `slippage_bps={SYMBOL: bps}`); falls back to scalar.
- Cost = turnover × (commission + slippage) / 10,000.
- Funding cost applied at the bar containing each funding event (8h cadence on perp); long pays positive rate, short receives.
- Returns: gross / net / cost / funding_cost reported separately.

### Cross-check oracle: Backtrader

[src/alpha_lab/backtest/bt_engine.py](../../../src/alpha_lab/backtest/bt_engine.py). Used only by [tests/test_bt_vs_vector.py](../../../tests/test_bt_vs_vector.py) on a one-month, one-symbol synthetic slice. 4 tests green:
1. Always-long, zero costs: vectorized vs Backtrader cumulative match to 1e-3.
2. Long/flat, zero costs: cumulative return match within 2% (per-bar timing differs at transitions by design).
3. Flat strategy returns are zero in BT.
4. Cost drag is monotonic in BT (sanity).

This bounds confidence that the vectorized engine's execution / return mechanics are correct.

### Cost model

From `configs/crypto_intraday.yaml`:

| scenario     | fee_bps/side | BTC slip | ETH slip | funding |
|--------------|--------------|----------|----------|---------|
| zero         | 0            | 0        | 0        | no      |
| perp_base    | 5            | 1        | 1.5      | yes     |
| perp_stress  | 5            | 3        | 5        | yes     |
| spot_base    | 10           | 2        | 3        | n/a     |
| spot_stress  | 10           | 5        | 7.5      | n/a     |

Every result table reports zero / base / stress side by side. The headline verdict relies on `perp_stress`.

---

## 7. Results

### Per-phase summary

**P0 — Baselines** (June 2024 perp 1m). Confirmed framework wiring:
- `always_flat` exactly 0.
- `bh_btc` gross = BTC pct exactly (matches close-to-close).
- `random` annihilated by costs at 1m (-100% at base/stress) — null result.

**P1 — Horizon discovery + rule-based at 5m** (Q1+Q2 2024). 9 rule-based strategies, all cost-killed at naive continuous-weight 5m specs. Cost drag 0.91x–4.93x capital over 6 months. Best gross signal: `volume_shock_continuation` (Sharpe 1.11 zero-cost, didn't survive).

**P2 — 1h sampling + perp-specific + cross-asset** (Q1+Q2 2024). Two survivors:

| strategy           | zero net | perp_base net | **perp_stress net** | **stress Sharpe** |
|--------------------|----------|---------------|---------------------|-------------------|
| funding_contrarian | +0.2174  | +0.1741       | **+0.1501**         | **1.12**          |
| funding_momentum   | +0.5224  | +0.4084       | **+0.4082**         | **1.56**          |

All other P2 strategies rejected. Vol-regime overlay on `funding_momentum` HURT it (gating to low-vol periods removed the bull-period alpha): stress net dropped from +40.8% to +4.7%.

**P3 — ML** (Q1+Q2 2024, walk-forward expanding, 3 folds). All 3 models rejected — none beat `funding_momentum`:

| model    | zero gross Sharpe | stress net | stress Sharpe |
|----------|-------------------|------------|---------------|
| logistic | 0.03              | -26.4%     | -1.86         |
| ridge    | 0.07              | -24.9%     | -1.47         |
| lightgbm | 0.75              | -38.3%     | -3.16         |

LightGBM was the strongest gross learner but had 2× the turnover of linear models, paid the cost penalty.

### BTC vs ETH attribution (`funding_momentum`)

Both symbols contribute. ETH funding was more volatile in Q1+Q2 2024 (lower min, higher max); BTC funding was more persistently positive. Combined long bias generated the bulk of returns; BTC contributed ~60%, ETH ~40% by net P&L.

### Long vs short

`funding_momentum` was net long >80% of the bar-by-bar holding. The strategy is essentially a "stay long when funding is positive" rule, with brief flat / short periods when funding-cum flipped negative.

---

## 8. Robustness

### What's covered

- **Slippage sensitivity**: zero / base / stress side-by-side in every result table. Strategy verdicts hinge on stress.
- **Funding sensitivity**: included in base/stress, excluded in zero. Material to perp.
- **Bootstrap CI**: P1 horizon discovery uses `BlockBootstrap(block_size=1 day, n=200)` on IC. CIs are 4–8 percentage points wide at the 6-month sample size.
- **Horizon sensitivity**: 5 horizons tested (5m, 15m, 30m, 1h, 4h). 1h chosen as primary.
- **Interval sensitivity**: 5m vs 1h. Most signals are 5m-killed and 1h-survivable; `funding_momentum` is 1h-only.
- **Cost ordering check**: confirmed zero ≥ base ≥ stress for all strategies with positive turnover.
- **Backtrader cross-check**: vector engine matches an independent event-driven oracle on the canonical case.

### What's NOT covered (deferred)

- **Cross-period validation**: only Q1+Q2 2024 tested. `funding_momentum`'s edge is highly likely to be regime-dependent. **This is the largest gap.**
- **Spot consistency**: spot doesn't have funding, so `funding_momentum` can't be replicated on spot directly. But a "long during positive recent BTC return" version could be — not tested.
- **Top-day exclusion**: not run.
- **Parameter sensitivity sweep on funding-cum window**: only window=10 tested.
- **Feature ablation**: not run for ML (only one feature set).
- **Capacity**: position size implications not estimated.
- **Delay execution sensitivity**: not run.

### One spot-perp consistency note

We did not run `funding_momentum` on spot (because funding only exists on perp). The next-best spot consistency check would be: take the perp funding signal, but trade on the spot price. Deferred to a future engagement.

---

## 9. Strategy selection decision

| strategy                       | decision           | reason                                                                |
|--------------------------------|--------------------|-----------------------------------------------------------------------|
| **funding_momentum**           | **accept_monitoring** | survives stress costs; but regime-dependent; needs cross-period validation |
| **funding_contrarian**         | **accept_monitoring** | similar to above; lower magnitude                                    |
| volume_shock_continuation (P1) | reject             | 5m gross signal didn't carry to 1h                                    |
| RSI / Bollinger / MACD / Donchian | reject          | cost-killed at all tested rebalance frequencies                       |
| spread_z_pair_trade            | reject             | wrong sign in this period; BTC out-performed ETH steadily              |
| btc_leads_eth                  | reject             | weak gross + high cost                                                 |
| beta_residual_mr               | reject_cost_killed | gross +13% killed by cost drag                                         |
| logistic / ridge / lightgbm    | reject (per goal)  | did not beat funding_momentum OOS after costs                          |

---

## 10. Limitations

- **Sample window**: 6 months (Q1+Q2 2024). Too short for regime certification.
- **Symbol breadth**: only BTC + ETH. A "funding_momentum" mechanism that works on a single high-conviction bull regime for the top-2 perp pairs may not generalize.
- **Execution realism**: cost model is constant bps. No depth-aware slippage; no partial-fill modeling; no funding-rate uncertainty beyond the historical realization. Stress assumption (3 bps BTC, 5 bps ETH) is intended to be conservative but is a single-point estimate.
- **Multiple testing**: ~50 strategy variants explored across phases (45 in idea log, several iterations). No formal multiple-testing correction applied to Sharpe inference. Conservative reading: a 1-of-50 Sharpe of 1.5 over 6 months has ~10% chance of being noise; the funding-family Sharpes are consistent across two related strategies which is mildly reassuring but not dispositive.
- **Capacity**: not estimated. BTCUSDT perp ADV is large enough that target-percent positioning at retail size has trivial impact; serious size would require ADV utilization modeling.
- **PM holdout untested**: by design. The 2026-01 → 2026-05 window will reveal whether the framework's discipline matters; it is locked until a fresh engagement opens it.
- **No purged-CV exercised in anger**: `PurgedKFold` is implemented + tested but P3 used walk-forward (because label horizon = 1 bar = trivial purge). Multi-horizon-label experiments would exercise it.

---

## 11. Final recommendation

**MONITOR.**

Specifically:
1. **Do not deploy funding_momentum based on this 6-month evidence.** The Sharpe 1.56 result is real for this window but is almost certainly inflated by the H1 2024 BTC bull. The vol-regime overlay test confirmed the strategy is highly correlated with up-vol regimes.

2. **Extend the validation to multi-year history before re-evaluating.** Concrete next experiment: re-run `funding_momentum` on 2022-Q1 (bear) + 2023-H1 (chop) + 2024-H1 (bull). If Sharpe persists >0.5 in each regime, escalate to `accept`. If it collapses outside bull, downgrade to `reject_regime_dependent`.

3. **Treat the framework as the primary deliverable.** The completed infrastructure — Binance Vision loader, PM-holdout enforcement with audit, leak-safe feature library, dual-engine backtest, walk-forward CV, structured decision records, 115-test suite — is reusable for future crypto-intraday investigations. It surfaced two candidate signals AND rejected the ML attempt honestly, which is exactly what was specified.

4. **Do not access PM holdout 2026-01 → 2026-05 yet.** The discipline is intact. A fresh `accept` decision after the cross-period validation in (2) would justify unlocking once — and only once.

The honest summary of this engagement: the framework works; one candidate signal is alive; the candidate is regime-dependent; treat it as a research lead, not a tradable strategy.

---

## Appendix

- All decision records: [docs/research_decisions/crypto_intraday/](.)
- Notebooks: [notebooks/90_crypto_intraday/](../../../notebooks/90_crypto_intraday/)
- Code: [src/alpha_lab/](../../../src/alpha_lab/)
- Tests: `pytest -q` → 115 green.
- PM holdout audit at end of engagement: 30+ events, 0 raises, `accessed=False`. **PM Holdout was not accessed.**
