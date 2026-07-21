# Crypto whale / ETF-flow strategy decision

Date: 2026-07-20
Decision: accept one **paper-monitor** candidate; reject binary flow rules

## Accepted paper candidate

`scaled_onchain_ma200`:

- 0% BTC when price is at or below its trailing MA200.
- 50% BTC when price is above MA200 and the conservatively available exchange-
  supply observation is not below its prior 365-day mean.
- 100% BTC when price is above MA200 and exchange supply is below that mean.
- Day-*t* on-chain data is first used at *t+1* close; target weight earns returns
  only from *t+2*. Base cost is 15 bps per unit of BTC weight change, stress 30 bps.
  Cash earns the 13-week T-bill proxy.

BTC results: 2015-07/2026 CAGR 53.8%, excess Sharpe 1.35, maximum drawdown -40.8%;
2024-07/2026 total return 17.7% (13.3% at 30 bps, 7.8% with one more execution day).
The recent extra-lag case has negative excess return versus cash. Price-only MA200
earned 51.7% in the recent period, so recent profit must be attributed primarily to
trend, not to a demonstrated on-chain alpha.

Exact ETH replication without retuning: 2017-07/2026 CAGR 82.7%, Sharpe 1.27,
maximum drawdown -69.1%; 2024-07/2026 total return 29.6%, while buy-and-hold lost
18.0%. Price-only MA200 earned 47.1% in the same recent period.

Latest signal using data through 2026-07-19 is 0% for both BTC and ETH because each
price is below MA200. This is research output, not an order.

## Rejected rules

1. BTC ETF extreme-inflow rule: profitable in 2024/2025, but weak family-wise
   circular-shift evidence and wide bootstrap uncertainty. Not released.
2. BTC ETF absorption rule: 2024/2025 looked strong, then the frozen 2026 result was
   -7.8% at base timing. Rejected.
3. Binary exchange-supply scarcity rule: highly significant pre-2024 evidence, but
   its frozen 2024-07/2026 result was -13.4% versus +51.7% for MA200. Rejected.

## Critical limitations

- The accepted scaled rule was created after the binary rule's external failure;
  there is no untouched holdout. It is eligible only for paper monitoring.
- Coin Metrics Community history exposes current exchange-address labels, not label
  vintages. Later entity discovery may backfill history and inflate old results.
- Exchange deposits, withdrawals, and reserves are not synonymous with whale buying
  or selling. Internal wallet reorganizations, custody migrations, change outputs,
  staking, and bridges can all create false interpretations.
- SOL lacks the matching Coin Metrics Community fields and its U.S. ETF history is
  short. No SOL strategy backtest is claimed.
- No live execution is authorized. Require at least six months of point-in-time paper
  signals, stable label vintages or daily snapshots, and an independent price source
  before reconsidering.
