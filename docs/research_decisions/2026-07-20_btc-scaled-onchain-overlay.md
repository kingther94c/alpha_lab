# BTC scaled on-chain overlay — fixed post-failure rule

Date: 2026-07-20
Status: post-hoc salvage test; no untouched holdout remains

The binary on-chain scarcity rule failed its frozen 2024–2026 external period even
though its pre-2024 evidence was strong. This follow-up is a single mechanism-driven
test, not another parameter grid:

- Hold no BTC when BTC is at or below its trailing 200-day moving average.
- Hold 50% BTC when BTC is above MA200 but known-exchange BTC supply is not below
  its trailing 365-day mean.
- Hold 100% BTC when BTC is above MA200 and exchange supply is below its trailing
  365-day mean.
- Treat day-*t* on-chain data as available at *t+1* close and earn no return before
  *t+2*. Charge 15 bps per unit of BTC weight change, stress at 30 bps, and credit
  uninvested cash at the 13-week T-bill proxy.

The 50% base allocation is fixed as the neutral midpoint and is not tuned. Results
after 2024 are exploratory because that price period and the failed binary rule have
already been observed. Acceptance is limited to a paper-monitor candidate. The
strategy must be compared with price-only MA200; if it does not add value, the report
must attribute profitability to trend rather than to on-chain alpha.
