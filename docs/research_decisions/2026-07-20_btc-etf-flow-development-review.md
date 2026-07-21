# BTC spot-ETF flow strategy — development review 1

**Date:** 2026-07-20
**Status:** `needs_revision`
**Holdout:** 2026 BTC returns remain unreleased.

## What survived

The strongest pre-registered family was an event rule: after aggregate ETF inflow exceeds its prior
60-ETF-day 80th percentile, remain eligible for ten ETF observations and hold BTC only above its
100-day moving average. The rule is conservative in timing: trade-date `t` flow is published onto
the `t+1` UTC close and first earns the `t+2` close-to-close return.

At 15 bps per 100% BTC-weight change:

- 2024 development: total CAGR 36.8%, cash-excess CAGR 30.3%, excess Sharpe 0.89.
- 2025 validation: total CAGR 16.7%, cash-excess CAGR 12.0%, excess Sharpe 0.60, MaxDD -9.6%.
- 2025 BTC buy-and-hold lost 6.3%; the matched MA100 control gained 6.2% with MaxDD -17.8%.
- At 30 bps the rule remained positive in both periods.
- With one additional full execution-day lag, it remained positive in both periods, but validation
  excess Sharpe fell from 0.60 to 0.29.
- The q70/q80/q90 and hold3/5/10 neighborhood was broadly positive; this is not one isolated cell.

## Why this is not yet accepted

The apparent edge does not clear a multiple-testing-aware falsification:

- Fixed-spec circular-flow shift p-value: 0.174. Circular shifts preserve flow autocorrelation and
  distribution while breaking calendar alignment with BTC.
- Family-wise circular-shift p-value: 0.239 after allowing every shifted sample to choose among the
  27 q/hold/trend cells.
- Seven-day moving-block bootstrap on 2025 validation cash-excess returns: 5th/50th/95th percentile
  compounded return = -22.1% / +10.6% / +59.5%; the interval includes a material loss.

The economic concern is endogeneity: ETF investors chase BTC returns. A raw extreme-inflow event may
be a dressed-up momentum signal, and random gates inside a noisy trend strategy can occasionally
avoid the wrong months by luck. Farside's historical table also lacks data vintages, so later
corrections cannot be ruled out even though the execution lag handles ordinary nightly publication.

## Revision 2 — frozen before testing

Use only 2024 development and 2025 validation; do not release 2026. Test three focused mechanisms:

1. **Unexpected flow:** subtract a rolling 60-ETF-day OLS expectation based on same-day and trailing
   five-day BTC returns, with coefficients fitted only through `t-1`. Trade positive residual tails.
2. **Breadth-confirmed flow:** require aggregate inflow plus positive participation across multiple
   issuers, reducing dependence on a single fund or GBTC rotation.
3. **Absorption flow:** require large inflow after a flat/negative five-day BTC return, targeting
   price-insensitive demand rather than return-chasing demand.

Only q70/q80, hold5/hold10, and MA50/MA100 are allowed. Select on the minimum of development and
validation excess Sharpe, then repeat the circular-shift and extra-lag tests. The 2026 holdout remains
closed until one rule is frozen or the entire research question is rejected.
