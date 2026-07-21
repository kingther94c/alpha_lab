# BTC spot-ETF flow strategy — pre-holdout protocol

**Date:** 2026-07-20
**Status:** `in_progress`
**Scope:** BTC first; cross-chain ETP extension only after the BTC design survives validation.
**Primary source:** Farside daily U.S. spot-BTC ETF flows, reconstructed from the public all-data table.

## Objective

Test whether *tradably available* U.S. spot-BTC ETF flows add a profitable, net-of-cost edge to a
low-turnover BTC long/cash strategy. A positive in-sample chart is not success: the signal must be
positive in 2025 validation, robust to timing/cost perturbations, and survive one frozen 2026
holdout release.

## Point-in-time contract

- A flow labelled U.S. trade date `t` is treated as known only by the end of UTC calendar day
  `t+1`, because Farside says updates typically arrive in the U.S. evening/night.
- A target formed at the close of `t+1` first earns the close-to-close BTC return ending on `t+2`.
  Same-date and next-UTC-day returns are diagnostics only and cannot support the verdict.
- Rolling means, standard deviations and quantiles use trailing observations and are shifted one
  ETF observation before comparing the current flow.
- Current address/entity labels will not be backfilled into history. Whale/entity strategies remain
  untested until point-in-time labels or prospectively archived snapshots are available.

## Data and frozen splits

- ETF flow table: 2024-01-11 onward; USD millions. Source extraction checksum is recorded in results.
- BTC price: Yahoo `BTC-USD`, daily close; 24/7 calendar.
- Cash hurdle: Yahoo `^IRX` (13-week T-bill yield), forward-filled and converted to a daily rate.
- Burn-in: 2024-01-11 through 2024-03-14.
- Development: 2024-03-15 through 2024-12-31.
- Validation: 2025-01-01 through 2025-12-31.
- Final holdout: 2026-01-01 through the latest complete ETF-flow date. No 2026 BTC returns are loaded
  during development. The holdout is released once, only after a final specification is frozen.

## Candidate families

All candidates are long BTC / cash, evaluated at 15 bps per 100% BTC weight change; 30 bps is the
stress case. Cash earns the T-bill rate. Headline edge returns subtract the cash opportunity cost.

1. `flow_sign`: long when the trailing 1/3/5/10 ETF observations sum positive.
2. `flow_confirm`: `flow_sign` plus BTC above its 20/50/100-day trailing moving average.
3. `flow_riskoff`: price-trend long, but flat below the prior 60-observation 10/20/30% flow quantile.
4. `flow_above_mean`: price-trend long only when flow exceeds its prior 60-observation mean.
5. `extreme_inflow_hold`: after an inflow above the prior 70/80/90% quantile, hold 3/5/10 ETF
   observations, optionally requiring a price trend.
6. `extreme_outflow_reversal`: deliberately adversarial test of whether large outflows mean-revert.

Price-only trend and BTC buy-and-hold are controls, not candidate discoveries.

## Selection and success criteria

- Parameter grids are fixed above before BTC prices are downloaded.
- Select only candidates with positive cash-excess CAGR in both development and validation, positive
  validation total return after costs, and no worse than -35% validation drawdown.
- A flow candidate must add at least 0.05 validation Sharpe versus its matched price-only trend, or
  reduce drawdown by at least 10% without losing more than 20% of CAGR.
- Prefer the simplest rule whose neighboring parameters have the same sign. Reject isolated peaks.
- Freeze one final specification before the 2026 holdout. It must remain profitable at 30 bps,
  remain positive with one extra execution-day lag, and show no leakage blocker.

## Honest verdict vocabulary

- `accept_candidate`: positive validation and holdout under the frozen protocol; suitable for paper
  monitoring, not autonomous live trading.
- `needs_revision`: positive evidence but a timing, data, or robustness gate fails.
- `reject`: no incremental tradable edge after correct publication lag and costs.
