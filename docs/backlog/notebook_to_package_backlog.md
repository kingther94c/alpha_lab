# Notebook → package migration backlog

Living list of patterns repeated across notebooks that should later move
into `src/alpha_lab/`. Goals:

- Make notebooks thin: each call should be a one-liner into a helper.
- Test the risky / load-bearing logic once, not re-derive it per notebook.
- Give agents and future-you a punch list that survives between sessions.

**Process:** pick an item, lift the function (with a docstring), update the
calling notebooks to import it, add a test if it's load-bearing, delete the
inline copy. Cross items off — don't silently drop them.

> Last scan: 2026-05-23. Re-scan whenever a new strategy notebook lands.

---

## Bucket 1 — High value (≥2 notebooks, concrete duplication)

| # | Pattern | Notebooks | Target module | Rationale |
|---|---------|-----------|---------------|-----------|
| 1 | Splice live ETF + proxy backfill into a single price column (e.g. DBMF + SG_Trend; cash-index with multi-frequency alignment) | `10_strategy_research/07_dbmf_sizing.ipynb`, `10_strategy_research/12_crypto_equity_vol_target.ipynb` | `data/loaders/` (new `splice.py`, or extend `local.py`) | Splice misalignment is a common silent-bug source — worth testing |
| 2 | Top-N / bottom-N momentum rank → equal-weight view weights (already covered by `backtest.sector_momentum.top_bottom_view_weights`, but notebooks 01, 02, 05 reimplement minor variants) | `10_strategy_research/01_simple_cross_asset_momentum.ipynb`, `02_us_sector_etf_cross_sectional_momentum.ipynb`, `05_cross_sectional_country_etf_momentum.ipynb` | `portfolio/long_only.py` or generalize `backtest.sector_momentum.top_bottom_view_weights` | Three callers means it's time to consolidate; helper exists, port the variants |
| 3 | Stress / crisis regime masks (equity_crash, bond_shock, equity_bond_down, equity_bond_selloff) for conditional performance tables | `10_strategy_research/07_dbmf_sizing.ipynb`, `11_all_weather_risk_parity.ipynb` | `analytics/risk.py` (new `regime_masks` / `conditional_summary`) | Same boilerplate twice — agents will copy it again unless lifted |
| 4 | Polymarket search → dedupe by id → endDate/keyword filter → `top_by_liquidity` per theme | `80_event_study/01_fed_rate_decisions.ipynb`, `80_event_study/02_iran_war.ipynb` | `data/loaders/polymarket.py` (new `filter_to_unique_events`) | Bookkeeping repeated in both event studies |
| 5 | Annual turnover + cost-drag table (`res.turnover.resample("YE").sum()` / `res.costs.resample("YE").sum()`) | `08_crypto_trend_filter.ipynb`, `11_all_weather_risk_parity.ipynb`, `12_crypto_equity_vol_target.ipynb` | `backtest/metrics.py` (new `annual_turnover_and_costs`) | Tiny but appears 3×; cheap to lift, useful in template |
| 6 | Inverse-vol / vol-targeted sleeve allocation with leverage cap and funding cascade | `10_strategy_research/06_dbmf_diversification_overlay.ipynb`, `07_dbmf_sizing.ipynb`, `11_all_weather_risk_parity.ipynb` | `portfolio/long_only.py` (extend) or new `portfolio/sleeves.py` | Logic is non-trivial and currently inline three times |

## Bucket 2 — Worth lifting once seen again

| # | Pattern | Where | Target | Why "wait one more" |
|---|---------|-------|--------|---------------------|
| 7 | Rolling realized vol with quantile-band overlay (q25/q50/q75 shading + current-level annotation) | `50_risk/01_bcom_rolling_vol.ipynb`, `02_rolling_vol_any_ticker.ipynb` | `reporting/charts.py` (new `rolling_vol_with_bands`) | Both notebooks already templates of each other; if a 3rd lands, lift |
| 8 | FOMC rate-outcome regex `_PATTERNS` + `_COLOR_MAP` for stacked-bar event vis | `80_event_study/01_fed_rate_decisions.ipynb` | `data/loaders/polymarket.py` (constants) | If geopolitical / election notebooks need the same vis, generalize |
| 9 | Block-bootstrap / Newey-West Sharpe SE (currently absent; notebooks quote raw Sharpe) | all of `10_strategy_research/` | `stats/tests.py` (new) | Lift the day the first study tries to defend a t-stat |
| 10 | Cost sensitivity sweep (rerun at 0.5× and 2× cost) | none yet — referenced in the template | `backtest/vector.py` or new `backtest/sensitivity.py` | Will be repeated in every robustness section once template lands |

## Bucket 3 — Notebook-local, leave alone

- Hardcoded ticker lists / `UNIVERSE` / `CORE_WEIGHTS` per study — intentionally
  notebook-scoped for reproducibility.
- Crisis-period **date ranges** in `07_dbmf_sizing.ipynb` (2007-10 → 2009-03,
  2022-01-03 → 2022-10-14, etc.) — research-specific framing.
- Single-asset diagnostic plots (price history, rolling vol for one ticker) —
  template-like but not duplicated enough.
- Ad-hoc commentary / interpretation markdown cells.

---

## Research-discipline risks observed during scan

Flagged for awareness only — fix the relevant notebook when next touched, do
not refactor unprompted.

1. **`07_dbmf_sizing.ipynb` — full-sample vol normalization.** The synthetic
   sleeve vol target divides by `sg_realized_excess_vol.std()` over the whole
   sample. Early-period leverage uses information from the 2008/2022 vol
   spikes. Consider a rolling estimator or label the run as diagnostic.
2. **`02_us_sector_etf_cross_sectional_momentum.ipynb` — universe survivorship.**
   `configs/us_sector_etf.csv` was created at one point in time. If any
   constituent was added/removed historically, momentum is conditioned on
   today's survivors. Recommend a dated snapshot per backtest start year.
3. **`01_simple_cross_asset_momentum.ipynb` — verify signal lag.** Monthly
   weights are computed from month-end prices and applied via
   `run_backtest`, which lags by 1 period. Verify by inspection — if a
   future variant builds weights without going through `run_backtest`, this
   risk re-opens.
4. **`08_crypto_trend_filter.ipynb` — 24/7 vs business-day calendar.**
   Crypto trades all week; the yfinance index is calendar-day, so weekend
   rebalances silently fall on the previous Friday. Document, and consider a
   crypto-specific calendar helper.
5. **`12_crypto_equity_vol_target.ipynb` — silent burn-in dependency.**
   `START = "2015-01-01"` with comment "1y burn-in" but BTC-USD data starts
   in 2014. If a future researcher pushes START earlier, the burn-in
   assumption breaks without an error.

---

## Working agreements

- An item moves to "Bucket 1" the second time it's copy-pasted.
- An item leaves the backlog when:
  - the helper is in `src/alpha_lab/`,
  - the calling notebooks import it,
  - and there's at least one test if the helper is leak-sensitive.
- Don't bulk-refactor. Touch the migration when you're already in the
  neighbourhood (e.g. fixing a bug in the relevant notebook).
- New patterns from new notebooks go in immediately, not "next sprint."
